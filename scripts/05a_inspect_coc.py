"""Free-run the stage-1 student's chain-of-causation on a few windows and print it.

The epoch gate teacher-forces the CoC (`eval/coarse_minade.py`), so nothing so
far shows what the student would actually reason. This loads a merged stage-1
checkpoint, builds a generation prompt that stops at `<|cot_start|>`, and lets
the student decode the CoC itself — printed next to the teacher's cached CoC for
the same window.

    python scripts/05a_inspect_coc.py                    # 8 windows, runs/stage1/best
    python scripts/05a_inspect_coc.py --n 20 --split val
    python scripts/05a_inspect_coc.py --ckpt /data/vla/alpamayo2edge/runs/stage1/best

Not a metric — an eyeball. minADE_6 is blind to the CoC (eval_phase1.md §5), so
"did the text collapse / loop / stay on-topic" is a question only this answers.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402

log = logging.getLogger("inspect_coc")


def _decode(student, ids: list[int]) -> tuple[str, bool, int]:
    """Student ids -> (coc_text, terminated, n_extra). Drops appended
    trajectory/special ids (not in the tokenizer), then cuts at the CoC
    terminator so over-generation past `<|cot_end|>` / `</think>` is not shown."""
    lo = student.new_token_range[0]
    kept = [i for i in ids if i < lo]
    text = student.tokenizer.decode(kept, skip_special_tokens=True)
    terminated = False
    for s in ("<|cot_end|>", "</think>"):
        if s in text:
            text, terminated = text.split(s, 1)[0], True
    return text.strip(), terminated, len(ids) - len(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="merged stage-1 checkpoint dir (default: <runs_root>/stage1/best)")
    ap.add_argument("--split", default="challenging")
    ap.add_argument("--n", type=int, default=8, help="windows to inspect")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)

    student = EdgeStudent(cfg).cuda()
    traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt",
                           weights_only=False)
    student.extend_trajectory_vocab(traj_spec)
    # The merged checkpoint carries the SigLIP2 vision tower (it was materialized
    # before `checkpoint.save` ran in training). `_ensure_vision_tower` is lazy,
    # so attach it now or `load_state_dict` sees 443 unexpected `visual.*` keys.
    student.lm._ensure_vision_tower()

    ckpt = Path(a.ckpt or (Path(cfg.paths.runs_root) / "stage1" / "best"))
    meta = checkpoint.load_into(student, ckpt)
    log.info("checkpoint: %s  (stage=%s epoch=%s coarse_minade=%.3f)",
             ckpt, meta.get("stage"), meta.get("epoch"), meta.get("coarse_minade", float("nan")))
    student.eval()

    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=load_split(cfg, a.split), cot_generation=True)
    n = min(a.n, len(ds))
    pad_id = student.tokenizer.pad_token_id

    gcfg = cfg.teacher
    log.info("decode: temperature=%.2f top_p=%.2f  (teacher release defaults)\n",
             float(gcfg.get("gen_temperature", 1.0)), float(gcfg.get("gen_top_p", 1.0)))

    for i in range(n):
        item = ds[i]
        clip = item["clip_id"]
        teacher_coc = str(item["coc_text"])
        batch = move_batch(collate_stage1([item], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows = student.generate_coc_text(batch, max_new_tokens=a.max_new_tokens)
        student_coc, terminated, n_extra = _decode(student, rows[0])
        flag = "terminated" if terminated else f"NO TERMINATOR (hit {a.max_new_tokens} cap)"
        print("=" * 100)
        print(f"[{i + 1}/{n}] clip {clip}   {len(rows[0])} tokens, {flag}"
              + (f", +{n_extra} non-text ids" if n_extra else ""))
        print("-" * 100)
        print("STUDENT CoC:\n" + student_coc)
        print("-" * 100)
        print("TEACHER CoC:\n" + teacher_coc.strip())
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

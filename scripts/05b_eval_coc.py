"""Comprehensive free-running CoC evaluation: the student's own reasoning, scored.

Fills the gap between the two things that already exist:

  `val CoC NLL`            teacher-forced, token-level CE. Deterministic and a
                           good SELECTOR, but it never lets the student speak and
                           it moves on phrasing as much as on content.
  `05a_inspect_coc.py`     free-running and real, but 8 windows and eyeball-only.

This free-runs the student over hundreds of windows and scores each trace against
the teacher's cached one on maneuver class, ego direction, named objects, and a
safety-asymmetric false-clear rate (`distill/eval/coc_score.py` documents the
grammar those rest on, and `parse_rate` reports how often it held).

    python scripts/05b_eval_coc.py --ckpt <dir> --split val   --n 300
    python scripts/05b_eval_coc.py --ckpt <dir> --split train --n 300

Run BOTH splits: `split_challenging == split_val` (264 identical clips,
eval_phase1.md §5) and selection already saw 60 val windows, so val alone is not
a clean holdout. The train-val gap is the number to read - it measures the CoC
overfitting directly, which matters because val CoC NLL peaks at epoch 1 of 12.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402
from distill.eval import coc_score                                  # noqa: E402

log = logging.getLogger("eval_coc")


def _decode(student, ids: list[int]) -> tuple[str, bool]:
    """Student ids -> (text, terminated). Same contract as 05a's `_decode`:
    drop the appended trajectory/special ids, then cut at the CoC terminator."""
    lo = student.new_token_range[0]
    text = student.tokenizer.decode([i for i in ids if i < lo], skip_special_tokens=True)
    for s in ("<|cot_end|>", "</think>"):
        if s in text:
            return text.split(s, 1)[0].strip(), True
    return text.strip(), False


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cameras", default=None,
                    help="comma-separated camera subset, overriding data.cameras. "
                         "Needed to score a checkpoint under the camera set it was "
                         "TRAINED with: the label cache keys frames per camera at "
                         "100% coverage (08_camera_ablation.py), so any subset is "
                         "free, but scoring run-253 under run 5's 1-camera config "
                         "would measure the train/test mismatch instead of the model.")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=None,
                    help="decode temperature; default = teacher.gen_temperature (0.6, the "
                         "teacher's own). `generate_coc_text` reads cfg.teacher.gen_*, so "
                         "sweeping this shows whether the rare-maneuver mass is still in "
                         "the student's distribution or gone from it.")
    ap.add_argument("--top-p", type=float, default=None, help="default = teacher.gen_top_p")
    ap.add_argument("--adapters", default=None,
                    help="adapter-only checkpoint dir (train_grpo_coc saves one per eval as "
                         "step-NNNN/). Loaded on top of --ckpt, which must then be the run's "
                         "init_ckpt. Lets any RL step be scored on full n after the fact.")
    ap.add_argument("--route-hint", action="store_true",
                    help="give the student the GT-derived route hint, as the RL run did")
    ap.add_argument("--dump", default=None, help="JSONL of every (student, teacher) pair")
    ap.add_argument("--out", default=None, help="JSON summary")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    # Same override point as 08_camera_ablation.py: the context builder reads
    # cfg.data.raw["cameras"] at call time, so this must land before the student.
    if a.cameras:
        cfg.raw["data"]["cameras"] = [c.strip() for c in a.cameras.split(",") if c.strip()]
    log.info("cameras: %s", list(cfg.data.raw["cameras"]))
    if a.temperature is not None:
        cfg.raw["teacher"]["gen_temperature"] = float(a.temperature)
    if a.top_p is not None:
        cfg.raw["teacher"]["gen_top_p"] = float(a.top_p)
    log.info("decode: temperature=%.2f top_p=%.2f", float(cfg.teacher.get("gen_temperature", 1.0)),
             float(cfg.teacher.get("gen_top_p", 1.0)))

    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    ckpt = Path(a.ckpt or (Path(cfg.paths.runs_root) / "stage1" / "best"))
    meta = checkpoint.load_into(student, ckpt)
    log.info("checkpoint: %s (epoch=%s select_on=%s coc_nll=%s)", ckpt, meta.get("epoch"),
             meta.get("select_on"), meta.get("val_coc_nll"))
    if a.adapters:
        student.param_groups_stage1()                  # inject, then load the adapters
        ameta = checkpoint.load_adapters(student, a.adapters)
        log.info("adapters: %s (step=%s)", a.adapters, ameta.get("step"))
    student.eval()

    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=load_split(cfg, a.split), cot_generation=True)
    n = min(a.n, len(ds))
    pad_id = student.tokenizer.pad_token_id
    log.info("split %s: %d windows available, scoring %d\n", a.split, len(ds), n)

    rows, dump = [], []
    term = empty_teacher = 0
    s_man, t_man = Counter(), Counter()
    for i in range(n):
        item = ds[i]
        teacher_coc = str(item["coc_text"]).strip()
        if a.route_hint:
            from distill.eval import gt_reward
            path = ds.shards[i]
            window = ds._window(path.parent.name, int(path.stem))
            hint = gt_reward.route_hint(gt_reward.kinematics(item["gt_future_xyz"]))
            item = {**item, "student": ds.ctx.build(window, coc_text=None, nav_text=hint)}
        batch = move_batch(collate_stage1([item], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.generate_coc_text(batch, max_new_tokens=a.max_new_tokens)
        student_coc, terminated = _decode(student, out[0])
        term += terminated
        r = coc_score.score_pair(student_coc, teacher_coc)
        if r is None:
            empty_teacher += 1
        else:
            rows.append(r)
            s_man[r["student_maneuver"]] += 1
            t_man[r["teacher_maneuver"]] += 1
        dump.append({"clip": item["clip_id"], "student": student_coc,
                     "teacher": teacher_coc, "terminated": terminated, "score": r})
        if (i + 1) % 25 == 0:
            log.info("  %d/%d", i + 1, n)

    haz = [r for r in rows if r["hazard_window"]]
    res = {
        "checkpoint": str(ckpt), "epoch": meta.get("epoch"), "split": a.split,
        "cameras": list(cfg.data.raw["cameras"]),
        "temperature": float(cfg.teacher.get("gen_temperature", 1.0)),
        "top_p": float(cfg.teacher.get("gen_top_p", 1.0)),
        "n_generated": n, "n_scored": len(rows), "n_teacher_empty": empty_teacher,
        "termination_rate": term / max(n, 1),
        "parse_rate": _mean([r["parsed"] for r in rows]),
        "maneuver_acc": _mean([r["maneuver_match"] for r in rows]),
        "direction_acc": _mean([r["direction_match"] for r in rows]),
        "object_recall": _mean([r["obj_recall"] for r in rows]),
        "object_precision": _mean([r["obj_precision"] for r in rows]),
        "token_f1": _mean([r["token_f1"] for r in rows]),
        "n_hazard_windows": len(haz),
        "false_clear_rate": _mean([r["false_clear"] for r in haz]),
        "student_maneuvers": dict(s_man.most_common()),
        "teacher_maneuvers": dict(t_man.most_common()),
    }

    log.info("\n%s  split=%s  epoch=%s  n=%d", ckpt.name, a.split, meta.get("epoch"), len(rows))
    log.info("  termination      %.3f", res["termination_rate"])
    log.info("  parse rate       %.3f   (grammar held)", res["parse_rate"])
    log.info("  maneuver acc     %.3f", res["maneuver_acc"] or float("nan"))
    log.info("  direction acc    %.3f", res["direction_acc"] or float("nan"))
    log.info("  object recall    %.3f   precision %.3f",
             res["object_recall"] or float("nan"), res["object_precision"] or float("nan"))
    log.info("  token F1         %.3f", res["token_f1"] or float("nan"))
    log.info("  FALSE-CLEAR      %.3f   over %d hazard windows  <- safety-asymmetric",
             res["false_clear_rate"] or float("nan"), len(haz))
    log.info("  maneuver mix, student vs teacher:")
    for k in sorted(set(s_man) | set(t_man), key=lambda k: -t_man[k]):
        log.info("     %-12s student %4d   teacher %4d", k, s_man[k], t_man[k])

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        log.info("-> %s", a.out)
    if a.dump:
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump, "w") as f:
            for d in dump:
                if d["score"]:
                    d["score"] = {k: (list(v) if isinstance(v, set) else v)
                                  for k, v in d["score"].items()}
                f.write(json.dumps(d) + "\n")
        log.info("-> %s", a.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

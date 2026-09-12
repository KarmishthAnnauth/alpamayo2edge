"""Camera ablation for the trained stage-1 student: run the gate (and optionally
the CoC) with an arbitrary camera subset, against a merged checkpoint.

Everything is read from the label cache, which stores frames keyed per camera
(`img|<camera>|<frame>`) at 100% coverage, so a subset costs nothing extra -
the unused JPEGs are simply not decoded. Only `data.cameras` differs between
arms; every other config value is shared, which is what makes this an ablation
rather than two runs.

    # baseline: the 4-camera set the checkpoint was trained with
    python scripts/08_camera_ablation.py --tag 4cam --repeats 3

    # the question: front wide only (NVIDIA's own 1-camera config)
    python scripts/08_camera_ablation.py --tag 1cam-frontwide \
        --cameras camera_front_wide_120fov --repeats 3 --coc 8

    # compare whatever has been written so far
    python scripts/08_camera_ablation.py --report

NB the gate samples `k` trajectories at temperature (`eval/coarse_minade.py`)
and does NOT seed itself, so single runs are noisy - epoch-to-epoch swings of
0.15-0.3 m are normal. `--repeats` re-runs with seed+i and reports the spread,
so the 4-cam -> 1-cam delta arrives with an error bar instead of without one.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.eval.coarse_minade import coarse_minade                # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402

log = logging.getLogger("camera_ablation")


class _P90(logging.Handler):
    """`coarse_minade` returns only minADE but logs p90/n. Catch them so the
    JSON record is complete rather than half the result."""

    def __init__(self):
        super().__init__()
        self.p90 = self.n = None

    def emit(self, record):
        m = re.search(r"\(p90 ([\d.]+), n=(\d+)\)", record.getMessage())
        if m:
            self.p90, self.n = float(m.group(1)), int(m.group(2))


def _decode(student, ids: list[int]) -> tuple[str, bool]:
    """Student ids -> (coc_text, terminated). Mirrors 05a_inspect_coc.py: drop
    appended trajectory ids the tokenizer does not know, then cut at the CoC
    terminator so over-generation past it is not counted as content."""
    lo = student.new_token_range[0]
    text = student.tokenizer.decode([i for i in ids if i < lo], skip_special_tokens=True)
    for s in ("<|cot_end|>", "</think>"):
        if s in text:
            return text.split(s, 1)[0].strip(), True
    return text.strip(), False


def build_student(cfg, ckpt_path: Path):
    student = EdgeStudent(cfg).cuda()
    traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt",
                           weights_only=False)  # pickled callables, D-032
    student.extend_trajectory_vocab(traj_spec)
    # The merged checkpoint carries the SigLIP2 tower; `_ensure_vision_tower` is
    # lazy, so attach it BEFORE load_state_dict or it sees unexpected visual.* keys.
    student.lm._ensure_vision_tower()
    meta = checkpoint.load_into(student, ckpt_path)
    student.eval()
    return student, meta


def run_coc(cfg, student, split: str, n: int) -> list[dict]:
    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=load_split(cfg, split), cot_generation=True)
    pad_id = student.tokenizer.pad_token_id
    gcfg = cfg.teacher
    out = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        batch = move_batch(collate_stage1([item], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows = student.generate_coc_text(batch, max_new_tokens=256)
        text, terminated = _decode(student, rows[0])
        out.append({"clip_id": item["clip_id"], "student_coc": text,
                    "teacher_coc": str(item["coc_text"]).strip(),
                    "terminated": terminated, "n_tokens": len(rows[0])})
        print("=" * 96)
        print(f"[{i + 1}] clip {item['clip_id']}  "
              f"{'terminated' if terminated else 'NO TERMINATOR'}  {len(rows[0])} tok")
        print(f"  STUDENT: {text}")
        print(f"  TEACHER: {str(item['coc_text']).strip()}")
    del ds
    _ = gcfg
    return out


def report(out_dir: Path) -> int:
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        r = json.loads(f.read_text())
        rows.append(r)
    if not rows:
        print(f"no results in {out_dir}")
        return 1
    print(f"\n{'tag':24s} {'cams':>5} {'minADE_6':>20} {'p90':>7} {'runs':>5}")
    print("-" * 66)
    for r in rows:
        g = r["gate"]["minade"]
        cell = f"{st.mean(g):.3f}" + (f" +/- {st.stdev(g):.3f}" if len(g) > 1 else "")
        print(f"{r['tag']:24s} {len(r['cameras']):5d} {cell:>20} "
              f"{r['gate']['p90'] or float('nan'):7.3f} {len(g):5d}")
    base = next((r for r in rows if len(r["cameras"]) > 1), None)
    if base:
        b = st.mean(base["gate"]["minade"])
        print(f"\ndeltas vs {base['tag']} ({b:.3f} m):")
        for r in rows:
            if r is base:
                continue
            d = st.mean(r["gate"]["minade"]) - b
            print(f"  {r['tag']:24s} {d:+.3f} m  ({d / b:+.1%})")
    print("\nNB: compare deltas against the run-to-run spread above - the gate is "
          "unseeded temperature sampling, so a delta inside +/-1 stdev is not a result.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="merged stage-1 checkpoint dir (default: <runs_root>/stage1/best)")
    ap.add_argument("--cameras", default=None,
                    help="comma-separated camera subset; default = config's data.cameras")
    ap.add_argument("--split", default="challenging")
    ap.add_argument("--gate", type=int, default=60,
                    help="windows for coarse minADE (60 = the epoch gate's own setting)")
    ap.add_argument("--coc", type=int, default=0, help="windows to free-run the CoC on (0 = skip)")
    ap.add_argument("--repeats", type=int, default=1, help="gate re-runs, seed+i each")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="name for this arm (default: derived from cameras)")
    ap.add_argument("--out", default=None, help="default: <runs_root>/camera_ablation")
    ap.add_argument("--report", action="store_true", help="summarize existing results and exit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(a.config)
    out_dir = Path(a.out or (Path(cfg.paths.runs_root) / "camera_ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report(out_dir)

    # The ONLY thing that differs between arms. Mutated before the student is
    # built so nothing can capture the old list: `context_builder()` reads
    # cfg.data.raw["cameras"] at call time (edge_wrapper.py), and coarse_minade
    # calls it internally.
    if a.cameras:
        cams = [c.strip() for c in a.cameras.split(",") if c.strip()]
        from distill.student.prompt import CAMERA_INDEX
        unknown = [c for c in cams if c not in CAMERA_INDEX]
        if unknown:
            print(f"unknown camera(s) {unknown}; expected one of {sorted(CAMERA_INDEX)}")
            return 2
        cfg.raw["data"]["cameras"] = cams
    cams = cfg.data.raw["cameras"]
    tag = a.tag or f"{len(cams)}cam-" + "-".join(c.split('_')[1] for c in cams)

    ckpt = Path(a.ckpt or (Path(cfg.paths.runs_root) / "stage1" / "best"))
    student, meta = build_student(cfg, ckpt)
    log.info("checkpoint: %s (stage=%s epoch=%s trained coarse_minade=%.3f)",
             ckpt.resolve(), meta.get("stage"), meta.get("epoch"),
             meta.get("coarse_minade", float("nan")))
    log.info("arm '%s': %d camera(s) %s | split=%s gate=%d repeats=%d",
             tag, len(cams), cams, a.split, a.gate, a.repeats)

    h = _P90()
    logging.getLogger("distill.eval.coarse_minade").addHandler(h)
    minades = []
    for i in range(a.repeats):
        # The gate does not seed itself; seed here so repeats are reproducible
        # and the spread measures the sampler, not an unknown RNG state.
        torch.manual_seed(a.seed + i)
        m = coarse_minade(cfg, student, split=a.split, k=cfg.eval.minade_k,
                          max_windows=a.gate)
        minades.append(float(m))
        log.info("  run %d/%d (seed %d): minADE_%d = %.3f m",
                 i + 1, a.repeats, a.seed + i, cfg.eval.minade_k, m)

    rec = {"tag": tag, "cameras": cams, "split": a.split,
           "checkpoint": str(ckpt.resolve()),
           "checkpoint_meta": {k: meta.get(k) for k in ("stage", "epoch", "coarse_minade")},
           "gate": {"minade": minades, "p90": h.p90, "n_windows": h.n,
                    "k": cfg.eval.minade_k, "seeds": [a.seed + i for i in range(a.repeats)]}}
    if a.coc:
        torch.manual_seed(a.seed)
        rec["coc"] = run_coc(cfg, student, a.split, a.coc)

    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(rec, indent=2))
    mean = st.mean(minades)
    spread = f" +/- {st.stdev(minades):.3f}" if len(minades) > 1 else ""
    log.info("\n%s: minADE_%d = %.3f%s m over %d run(s)  ->  %s",
             tag, cfg.eval.minade_k, mean, spread, len(minades), path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

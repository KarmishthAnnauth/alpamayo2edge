"""Score a free-running CoC dump against the DRIVER, not the teacher (D-038 metrics).

`05b_eval_coc.py --dump` writes one (student, teacher) trace pair per window. This
reads that file and grades both traces with `eval.gt_reward.gt_metrics` against
the cached `gt_future_xyz`, so the student and the teacher are judged on the same
windows by the same rule:

    consistent      stated maneuver vs the driver's speed profile / path, over the
                    windows where the claim is CHECKABLE (STOP, SLOW, YIELD, ...,
                    and FOLLOW/KEEP only when the driver braked from speed)
    gt_false_clear  driver braked hard from speed, trace said clear / keep / accelerate
    direction_ok    driver turned; trace names that direction
    hazard_ungrounded  trace names a hazard; driver did not react
    braked -> says slow/stop   the user's criterion (2026-09-18): when the driver
                    slowed or stopped, did the CoC say so?

Rows without a `window` field (dumps written before this script) are mapped by
dataset order, which `05b` walks in `discover_shards` order for the split.

    python scripts/05f_coc_gt_score.py --dump runs/coc-<run>-val.jsonl --split val
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.data.dataset import discover_shards                    # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.eval import gt_reward                                  # noqa: E402
from distill.eval.coc_score import parse                            # noqa: E402

SLOWING = ("STOP", "SLOW", "YIELD")


def rate(vals) -> tuple[float, int]:
    v = [x for x in vals if x is not None]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    cfg = load_config(a.config)
    shards = discover_shards(Path(cfg.paths.cache_root), load_split(cfg, a.split))
    rows = [json.loads(l) for l in Path(a.dump).read_text().splitlines() if l.strip()]

    agg = {"student": [], "teacher": []}
    braked = {"student": Counter(), "teacher": Counter()}
    n_braked = 0
    # Severity split of the braking windows: a full stop, a hard brake (>2.5 m/s
    # lost over the horizon, no stop), or a mild slowdown (1.5-2.5 m/s). "Keep
    # distance" behind a lead car that eases off is a fair description of the
    # mild case; it is not of a stop.
    sev_n = Counter()
    sev_slow = {"student": Counter(), "teacher": Counter()}
    for i, r in enumerate(rows):
        if "window" in r:
            shard = Path(cfg.paths.cache_root) / r["clip"] / f"{int(r['window']):02d}.npz"
        else:
            shard = shards[i]
            if shard.parent.name != r["clip"]:
                raise RuntimeError(f"row {i}: dump clip {r['clip']} != dataset order "
                                   f"{shard.parent.name}; re-run 05b with the window field")
        with np.load(shard) as z:
            k = gt_reward.kinematics(z["gt_future_xyz"])
        for who in ("student", "teacher"):
            agg[who].append(gt_reward.gt_metrics(r[who], k))
        if k["braked_from_speed"]:
            n_braked += 1
            sev = ("stop" if k["vmin"] < 0.5 else "hard" if k["dv"] < -2.5 else "mild")
            sev_n[sev] += 1
            for who in ("student", "teacher"):
                man = parse(r[who]).get("maneuver") or "NONE"
                braked[who][man] += 1
                sev_slow[who][sev] += man in SLOWING

    print(f"{Path(a.dump).name}: {len(rows)} windows, split {a.split}")
    print(f"  {'metric':<28} {'student':>14} {'teacher':>14}")
    for key, label in (("consistent", "GT-consistent (checkable)"),
                       ("gt_false_clear", "GT false-clear (braked)"),
                       ("direction_ok", "direction stated ok (turns)"),
                       ("hazard_ungrounded", "hazard named, no reaction")):
        s, ns = rate(m[key] for m in agg["student"])
        t, nt = rate(m[key] for m in agg["teacher"])
        print(f"  {label:<28} {s:8.3f} n={ns:<4d} {t:8.3f} n={nt:<4d}")
    print(f"  driver braked from speed: {n_braked} windows; the CoC said")
    for man in SLOWING + ("ADAPT_SPEED", "FOLLOW", "KEEP", "NUDGE", "ACCELERATE", "NONE"):
        s, t = braked["student"][man], braked["teacher"][man]
        if s or t:
            print(f"     {man:<12} student {s:4d} ({100 * s / max(n_braked, 1):4.1f}%)   "
                  f"teacher {t:4d} ({100 * t / max(n_braked, 1):4.1f}%)")
    s_slow = sum(braked["student"][m] for m in SLOWING)
    t_slow = sum(braked["teacher"][m] for m in SLOWING)
    print(f"  braked -> says slow/stop:   student {s_slow / max(n_braked, 1):.3f}   "
          f"teacher {t_slow / max(n_braked, 1):.3f}")
    for sev, label in (("stop", "driver STOPPED"), ("hard", "driver braked HARD"),
                       ("mild", "driver slowed mildly")):
        n = sev_n[sev]
        if n:
            print(f"     {label:<22} n={n:<4d} says slow/stop: student "
                  f"{sev_slow['student'][sev] / n:.3f}   teacher {sev_slow['teacher'][sev] / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

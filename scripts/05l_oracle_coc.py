"""Phase 2 diagnostic: does the flow head listen to the CoC?

Builds two alternative CoC caches for a split from the EXPERT's own kinematics
(`gt_reward.kinematics` on `gt_future_xyz`), using the most frequent texts of
the real cache (`coc_rl7s25.jsonl`) so every line is in-distribution for the
head:

    oracle  - the text says what the expert did (stop / slow / keep / accelerate,
              a turn where the expert turned)
    anti    - the opposite longitudinal class

Evaluating a checkpoint with none / cached / oracle / anti CoC bounds how much
better conditioning could buy (oracle - cached) and how much the head reads
the text at all (oracle - anti).

    python scripts/05l_oracle_coc.py --split val --out-dir runs/coc_probe
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.data import bench2drive as b2d                         # noqa: E402
from distill.eval import gt_reward                                  # noqa: E402

log = logging.getLogger("oracle_coc")

# Most frequent text per maneuver class in coc_rl7s25.jsonl (all 8,903 windows).
TEXT = {
    "STOP": "Stop for the red traffic light since the signal is red",
    "STOP_LEAD": "Stop to keep distance to the lead vehicle since it is stopped ahead",
    "SLOW": "Slow down due to the lead vehicle ahead",
    "KEEP": "Keep lane since the lane is clear ahead",
    "FOLLOW": "Keep distance to the lead vehicle since it is directly ahead in our lane",
    "ACCELERATE": "Accelerate to proceed through the intersection since the traffic light turns green",
    "TURN_LEFT": "Turn left at the intersection since the intersection is clear",
    "TURN_RIGHT": "Turn right at the intersection since the intersection is clear",
}


def classes(k: dict, v0: float) -> tuple[str, str]:
    """(oracle, anti) class keys from the expert's kinematics over the horizon."""
    standing = v0 < 0.5
    if standing:
        if k["v1"] < 0.5:                         # stays stopped
            return "STOP", "ACCELERATE"
        if k["lateral"] == "turn_left":           # pulls away into a turn
            return "TURN_LEFT", "STOP"
        if k["lateral"] == "turn_right":
            return "TURN_RIGHT", "STOP"
        return "ACCELERATE", "STOP"
    if k["speed"] == "stops":
        return "STOP_LEAD", "KEEP"
    if k["speed"] == "brakes":
        return "SLOW", "KEEP"
    if k["lateral"] == "turn_left":
        return "TURN_LEFT", "STOP_LEAD"
    if k["lateral"] == "turn_right":
        return "TURN_RIGHT", "STOP_LEAD"
    if k["speed"] == "speeds":
        return "ACCELERATE", "STOP_LEAD"
    return "KEEP", "STOP_LEAD"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default="runs/coc_probe")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    root = Path(cfg.paths.b2d_cache_root)
    cached = b2d.load_coc_cache(cfg.stage2.coc_cache)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = {"oracle": [], "anti": []}
    tally = Counter()
    for clip in b2d.load_b2d_split(root, a.split):
        for p in sorted((root / clip).glob("[0-9][0-9].npz")):
            w = int(p.stem)
            with np.load(p, allow_pickle=True) as z:
                k = gt_reward.kinematics(z["gt_future_xyz"])
                v0 = float(z["ego_speed_t0"])
            o, an = classes(k, v0)
            tally[o] += 1
            for name, cls in (("oracle", o), ("anti", an)):
                rows[name].append({"clip": clip, "window": w, "student": TEXT[cls],
                                   "terminated": True, "cls": cls,
                                   "cached": cached.get((clip, w))})
    for name, rs in rows.items():
        path = out / f"coc_{name}_{a.split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rs))
        log.info("%d rows -> %s", len(rs), path)
    log.info("oracle classes: %s", dict(tally.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

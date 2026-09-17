"""Which cached windows can the CoC and the GT trajectory be COUPLED on? (D-043)

Run 7 pairs the teacher's chain-of-causation with the driver's own trajectory
tokens in one teacher-forced sequence, and forces the trajectory positions to
read the CoC (image dropout, `dataset.collate_student`). That pairing is only
a lesson when the two agree. Measured over all 20,000 cached windows with the
strict scale from `eval.gt_reward` (2026-09-17):

    FOLLOW / KEEP while the driver braked hard from speed      2,265   11.3%
    STOP with no stop inside the 6.4 s horizon                    596    3.0%
    turn / lane-change direction OPPOSITE to the one taken        303    1.5%
    TURN / ACCELERATE / SLOW / YIELD contradicted by kinematics   629    3.1%

Training on those pairs teaches "say one thing, do another" - the decoupling
D-040 measured, made into a target. They are dropped from the train split.
Soft cases stay: ADAPT_SPEED on a held speed (-0.5), a volunteered lane change
with no offset (-0.5), silence on a turn (-0.5, and the route hint now names
it), and the 305 unparsed traces. Only a claim the driver's future flatly
contradicts (a -1.0 on either scale) is a contradiction here.

The gate and val splits are NOT filtered: the student is judged on every window.
"""
from __future__ import annotations
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..eval import gt_reward
from ..eval.coc_score import parse

log = logging.getLogger(__name__)

#: A term at or below this is a flat contradiction (both scales use -1.0 for it).
CONTRADICTION = -1.0


def contradicted(coc_text: str, gt_future_xyz) -> bool:
    """True when the teacher's stated maneuver or direction is flatly
    contradicted by what the driver did over the horizon."""
    k = gt_reward.kinematics(gt_future_xyz)
    s = parse(str(coc_text))
    man, d = s.get("maneuver"), s.get("direction")
    if gt_reward.direction_term(man, d, k) <= CONTRADICTION:
        return True
    return gt_reward.kinematic_term(man, k, strict=True) <= CONTRADICTION


def route_hint(gt_future_xyz) -> str:
    """The direction the driver took, as the `<|route_start|>` text. Direction
    only - it must not leak the speed profile (`gt_reward.route_hint`)."""
    return gt_reward.route_hint(gt_reward.kinematics(gt_future_xyz))


def grounded_shards(shards: list[Path]) -> tuple[list[Path], dict]:
    """Drop the contradicted windows. Returns `(kept, stats)`; `stats` counts
    dropped windows by the teacher's maneuver class so the log shows WHAT went.
    Reads two small members of each npz; ~20k shards is a minute or two."""
    kept: list[Path] = []
    dropped: Counter = Counter()
    for p in shards:
        with np.load(p, allow_pickle=True) as z:
            text = str(z["coc_text"]) if "coc_text" in z.files else ""
            xyz = z["gt_future_xyz"]
        if contradicted(text, xyz):
            dropped[parse(text).get("maneuver") or "UNPARSED"] += 1
        else:
            kept.append(p)
    stats = {"total": len(shards), "kept": len(kept),
             "dropped": sum(dropped.values()), "dropped_by_maneuver": dict(dropped)}
    return kept, stats


def log_stats(stats: dict) -> None:
    log.info("grounding filter (D-043): kept %d / %d windows, dropped %d (%.1f%%)",
             stats["kept"], stats["total"], stats["dropped"],
             100.0 * stats["dropped"] / max(stats["total"], 1))
    for m, n in sorted(stats["dropped_by_maneuver"].items(), key=lambda x: -x[1]):
        log.info("   dropped %-12s %6d", m, n)

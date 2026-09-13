"""Maneuver-stratified sampling for stage 1 (D-036).

Uniform sampling hands the student the teacher's maneuver *marginal*: over the
full cache FOLLOW is 32.9% of windows and KEEP 20.8%, LANE_CHANGE 2.1%. `text_kl`
is a batch-averaged token KL, so that marginal is the gradient, and the measured
result (`scripts/05b_eval_coc.py`) is a student that emits LANE_CHANGE at 11% of
the teacher's rate and stays silent on direction in ~48% of the windows where the
teacher commits to one - on train as well as val, i.e. never learned rather than
overfit.

This draws each window with probability proportional to
`(1 / freq[class])**alpha`, `class` being the maneuver the teacher's cached
`coc_text` parses to (`eval.coc_score.maneuver`, 98.7% assignable). Windows the
grammar cannot classify - empty traces (0.3%) and the unparsed tail - are drawn at
the uniform rate rather than treated as a rare class, which they are not.

`alpha` is the temper. 0 is uniform; 1 equalises the classes and repeats the 415
LANE_CHANGE windows ~16x per epoch, which is memorisation; 0.5 (square-root, the
multilingual-LM convention for the same problem) takes LANE_CHANGE 2.1% -> 5.1%
and FOLLOW 32.9% -> 20.3%. Draws are with replacement and `num_samples` is the
split size, so steps per epoch do not change.
"""
from __future__ import annotations
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..eval.coc_score import maneuver, split_clause

log = logging.getLogger(__name__)

#: Class for windows the grammar does not assign a maneuver to.
UNASSIGNED = "UNASSIGNED"


def teacher_maneuver(shard: Path) -> str:
    """Maneuver class of one cached window, or UNASSIGNED."""
    with np.load(shard, allow_pickle=True) as z:
        text = str(z["coc_text"]).strip() if "coc_text" in z.files else ""
    if not text:
        return UNASSIGNED
    return maneuver(split_clause(text)[0]) or UNASSIGNED


def maneuver_weights(shards: list[Path], alpha: float) -> tuple[torch.Tensor, dict]:
    """Per-window sampling weights (mean 1.0) plus the mix they imply.

    Returns `(weights, mix)` where `mix[class] = (count, uniform_share,
    effective_share)`. Read once at startup; ~20k shards is under a minute since
    only the `coc_text` member of each npz is decompressed.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"maneuver_sampling.alpha must be in [0, 1], got {alpha}")
    labels = [teacher_maneuver(p) for p in shards]
    n = len(labels)
    counts = Counter(labels)
    assigned = {c: k for c, k in counts.items() if c != UNASSIGNED}
    if not assigned:
        raise RuntimeError("no cached window parsed to a maneuver class - is this "
                           "the right cache_root?")
    n_assigned = sum(assigned.values())
    class_w = {c: (n_assigned / k) ** alpha for c, k in assigned.items()}
    # Unclassified windows draw at the uniform rate: the mean assigned weight.
    class_w[UNASSIGNED] = sum(class_w[c] * assigned[c] for c in assigned) / n_assigned
    w = torch.tensor([class_w[l] for l in labels], dtype=torch.float64)
    w = w / w.mean()
    total = float(sum(class_w[c] * counts[c] for c in counts))
    mix = {c: (counts[c], counts[c] / n, class_w[c] * counts[c] / total)
           for c in sorted(counts, key=lambda c: -counts[c])}
    return w.float(), mix


def log_mix(mix: dict, alpha: float) -> None:
    log.info("maneuver-stratified sampling, alpha=%.2f (D-036):", alpha)
    log.info("   %-13s %6s %8s %10s", "class", "n", "uniform", "effective")
    for c, (k, u, e) in mix.items():
        log.info("   %-13s %6d %7.1f%% %9.1f%%", c, k, 100 * u, 100 * e)

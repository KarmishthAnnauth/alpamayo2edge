"""Coarse minADE: decode the AR tower's discrete trajectory tokens straight to
waypoints via the teacher's detokenizer - no diffusion tower. This is the
stage 1 quality gate (plan 3.3): if the coarse plan is bad, stage 2 refinement
cannot save it, and you find out days earlier.

Three things about the protocol, because each of them was wrong until
2026-08-24 and each fails quietly rather than loudly:

**The trajectory is generated, the CoC is teacher-forced.** The context is built
with `for_generation=True`, so it ends at `<|traj_future_start|>` and the 128
positions are the student's own. The previous version generated from the
TEACHER-FORCED context, i.e. with the teacher's trajectory tokens already in the
prompt. The CoC stays forced on purpose: it keeps the gate comparable across
epochs and matches the training context exactly, at the cost of not measuring
the student's own reasoning. The free-running two-phase decode (CoC, then
trajectory) belongs in the final eval, `scripts/05_eval.py`, not in a
once-per-epoch early-stopping signal.

**Batch size is 1.** `collate_student` right-pads, which is correct for a
teacher-forced forward pass but not for a prefill: with B>1 every shorter sample
would decode its first token after a run of padding. One window at a time costs
wall-clock and buys a number that means something.

**Detokenization goes through `student.detokenize_traj`**, which applies D-031's
`swap_action_dims`. Calling the tokenizer's `decode` directly - which is what
`student.traj_detokenize` used to be - reads the two action dims transposed and
costs 20-128 m of ADE that looks exactly like a student that failed to learn.
"""
from __future__ import annotations
import functools
import logging

import torch
from torch.utils.data import DataLoader

from ..data.dataset import Stage1Dataset, collate_stage1
from ..data.splits import load_split
from .open_loop import evaluate

log = logging.getLogger(__name__)


@torch.no_grad()
def coarse_minade(cfg, student, split: str = "challenging", k: int = 6,
                  max_windows: int | None = None) -> float:
    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=_split_clips(cfg, split),
                       for_generation=True)
    if max_windows is not None and len(ds) > max_windows:
        # Deterministic prefix, not a random sample: the gate is compared against
        # itself across epochs, so the window set must not move between them.
        ds = torch.utils.data.Subset(ds, range(max_windows))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=functools.partial(
                        collate_stage1, pad_id=student.tokenizer.pad_token_id))

    def sample_fn(batch, k):
        modes = []
        for _ in range(k):      # temperature sampling over trajectory tokens
            tok = student.generate_traj_tokens(batch)
            modes.append(student.detokenize_traj(
                tok, batch["hist_xyz"], batch["hist_rot"]).to(batch["gt_future_xyz"]))
        return torch.stack(modes, dim=1)

    out = evaluate(sample_fn, dl, k)
    log.info("coarse minADE_%d on %s: %.3f m (p90 %.3f, n=%d)",
             k, split, out["minade"], out["p90"], out["n"])
    return out["minade"]


def _split_clips(cfg, split: str) -> list[str]:
    return load_split(cfg, split)

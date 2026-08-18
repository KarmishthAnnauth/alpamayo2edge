"""Coarse minADE: decode the AR tower's discrete trajectory tokens straight to
waypoints via the teacher's detokenizer - no diffusion tower. This is the
stage 1 quality gate (plan 3.3): if the coarse plan is bad, stage 2 refinement
cannot save it, and you find out days earlier.
"""
from __future__ import annotations
import functools
import torch
from torch.utils.data import DataLoader

from ..data.dataset import DistillShardDataset, collate_stage1
from .open_loop import evaluate


@torch.no_grad()
def coarse_minade(cfg, student, split: str = "challenging", k: int = 6) -> float:
    ds = DistillShardDataset(cfg, clip_ids=_split_clips(cfg, split))
    dl = DataLoader(ds, batch_size=8, collate_fn=functools.partial(
        collate_stage1, pad_id=student.tokenizer.pad_token_id))

    def sample_fn(batch, k):
        modes = []
        for _ in range(k):  # temperature sampling over trajectory tokens
            tok = student.generate_traj_tokens(batch)
            modes.append(student.traj_detokenize(tok))
        return torch.stack(modes, dim=1)

    return evaluate(sample_fn, dl, k)["minade"]


def _split_clips(cfg, split: str) -> list[str]:
    import json
    from pathlib import Path
    with open(Path(cfg.paths.cache_root) / f"split_{split}.json") as f:
        return json.load(f)

"""Open-loop evaluation: minADE_k over the horizon.

Used for both coarse (discrete tokens -> detokenized waypoints, stage 1 gate)
and full-pipeline (diffusion-refined) evaluation. Keep the challenging split
as the headline number - average minADE on easy driving saturates and stops
being informative at small data scale.
"""
from __future__ import annotations
import torch

from ..data.dataset import move_batch


def ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Single-mode ADE. pred: (B, H, 2+); gt: (B, H, 2+). Ego frame, metres.

    Mean L2 displacement over the horizon, x/y only — the standard open-loop
    number, and the one to quote when a model emits one trajectory rather than a
    set. `min_ade` is its k-sample cousin and is NOT comparable across different
    k, so always report k alongside it.
    """
    return torch.linalg.norm(pred[..., :2] - gt[..., :2], dim=-1).mean(-1)


def min_ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred: (B, K, H, 2+) modes; gt: (B, H, 2+). Positions in ego frame.
    Returns per-sample min over modes of mean L2 displacement (x, y only)."""
    d = torch.linalg.norm(pred[..., :2] - gt[:, None, :, :2], dim=-1)  # (B, K, H)
    return d.mean(-1).min(-1).values


@torch.no_grad()
def evaluate(model_sample_fn, dataloader, k: int) -> dict:
    scores = []
    for batch in dataloader:
        batch = move_batch(batch)
        pred = model_sample_fn(batch, k=k)          # (B, K, H, A)
        scores.append(min_ade(pred, batch["gt_traj"]).cpu())
    s = torch.cat(scores)
    return {"minade": float(s.mean()), "n": int(s.numel()),
            "p90": float(s.quantile(0.9))}

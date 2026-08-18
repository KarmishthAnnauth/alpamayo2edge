"""Open-loop evaluation: minADE_k over the horizon.

Used for both coarse (discrete tokens -> detokenized waypoints, stage 1 gate)
and full-pipeline (diffusion-refined) evaluation. Keep the challenging split
as the headline number - average minADE on easy driving saturates and stops
being informative at small data scale.
"""
from __future__ import annotations
import torch


def min_ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred: (B, K, H, 2+) modes; gt: (B, H, 2+). Positions in ego frame.
    Returns per-sample min over modes of mean L2 displacement (x, y only)."""
    d = torch.linalg.norm(pred[..., :2] - gt[:, None, :, :2], dim=-1)  # (B, K, H)
    return d.mean(-1).min(-1).values


@torch.no_grad()
def evaluate(model_sample_fn, dataloader, k: int) -> dict:
    scores = []
    for batch in dataloader:
        batch = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in batch.items()}
        pred = model_sample_fn(batch, k=k)          # (B, K, H, A)
        scores.append(min_ade(pred, batch["gt_traj"]).cpu())
    s = torch.cat(scores)
    return {"minade": float(s.mean()), "n": int(s.numel()),
            "p90": float(s.quantile(0.9))}

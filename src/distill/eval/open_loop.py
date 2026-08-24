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
def evaluate(model_sample_fn, dataloader, k: int, gt_key: str = "gt_future_xyz",
             device: str = "cuda") -> dict:
    """minADE_k over a dataloader. `model_sample_fn(batch, k) -> (B, K, H, 3+)`.

    The reference is `gt_future_xyz`: ego-frame POSITIONS, which is what a
    displacement error is defined over. It is emphatically NOT `gt_traj`, which
    this function used until 2026-08-24 — that field is the GT future in the
    teacher's ACTION space, so `[..., :2]` there means (accel, curvature) and the
    resulting "metres" were a norm over an accel/curvature plane. It ran, it
    produced a plausible-looking float, and it would have early-stopped stage 1
    on it.
    """
    scores = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        if gt_key not in batch:
            raise KeyError(
                f"{gt_key} missing from the batch — the collator must carry the "
                "positional GT for a distance metric; `gt_traj` is action space "
                "and is not a substitute")
        pred = model_sample_fn(batch, k=k)          # (B, K, H, 3)
        gt = batch[gt_key]
        if pred.shape[-2] != gt.shape[-2]:
            raise ValueError(
                f"horizon mismatch: predicted {pred.shape[-2]} waypoints against "
                f"{gt.shape[-2]} GT — check tokens_per_future_traj against "
                "eval.horizon_s")
        scores.append(min_ade(pred, gt).cpu())
    s = torch.cat(scores)
    return {"minade": float(s.mean()), "n": int(s.numel()),
            "p90": float(s.quantile(0.9))}

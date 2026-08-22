"""Optimizer + schedule, shared by both stages.

Split out of train_stage1 when D-024 made the two stages' parameter groups the
same shape (adapters at 1.0x, new full-rank interface at `new_token_lr_mult`),
so stage 2 stops silently dropping the multiplier.
"""
from __future__ import annotations

import math

import torch


def build_optimizer(groups, lr: float, weight_decay: float = 0.0,
                    betas: tuple[float, float] = (0.9, 0.95)):
    """AdamW over `[{"params": [...], "lr_mult": float}, ...]`.

    Stamps `initial_lr` on every group: `cosine_lr` scales against it, and
    without it the old code's `if "initial_lr" in g` guard was never true, so
    the LR silently stayed flat for the whole run.
    """
    pgs = []
    for g in groups:
        params = [p for p in g["params"] if p.requires_grad]
        if not params:
            continue
        scaled = lr * g.get("lr_mult", 1.0)
        pgs.append({"params": params, "lr": scaled, "initial_lr": scaled})
    if not pgs:
        raise RuntimeError("no trainable parameters — check the LoRA injection")
    return torch.optim.AdamW(pgs, weight_decay=weight_decay, betas=betas)


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def set_lr(opt, scale: float) -> None:
    for g in opt.param_groups:
        g["lr"] = g["initial_lr"] * scale


def trainable_report(groups) -> str:
    parts = []
    for i, g in enumerate(groups):
        n = sum(p.numel() for p in g["params"] if p.requires_grad)
        parts.append(f"group{i}(x{g.get('lr_mult', 1.0)}): {n / 1e6:.1f}M")
    total = sum(p.numel() for g in groups for p in g["params"] if p.requires_grad)
    return f"trainable {total / 1e6:.1f}M [" + ", ".join(parts) + "]"

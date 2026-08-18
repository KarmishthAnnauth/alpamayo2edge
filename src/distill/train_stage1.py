"""Stage 1: distill the teacher reasoner into Edge's AR tower.

Single-GPU (96 GB) loop: bf16 autocast, gradient checkpointing, accumulation.
Early stopping on challenging-split coarse minADE (decoded from discrete
tokens alone - no diffusion tower involved, plan gate 3.3).
"""
from __future__ import annotations
import argparse
import functools
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data.dataset import DistillShardDataset, collate_stage1
from .student.edge_wrapper import EdgeStudent
from .student.layer_map import FeatureProjections, uniform_map
from . import losses
from .eval.coarse_minade import coarse_minade

log = logging.getLogger(__name__)


def build_optimizer(groups, lr, weight_decay):
    params = [{"params": g["params"], "lr": lr * g["lr_mult"]} for g in groups]
    return torch.optim.AdamW(params, weight_decay=weight_decay, betas=(0.9, 0.95))


def cosine_lr(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    torch.backends.cuda.matmul.allow_tf32 = True

    student = EdgeStudent(cfg).cuda()
    traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt")
    student.extend_trajectory_vocab(traj_spec)
    if cfg.stage1.grad_checkpoint:
        student.model.gradient_checkpointing_enable()

    expert_layers = torch.load(Path(cfg.paths.cache_root) / "expert_layers.pt")
    lmap = uniform_map(expert_layers["attended_layers"], student.model.config.num_hidden_layers)
    projections = FeatureProjections(
        lmap, student.model.config.hidden_size, expert_layers["kv_dim"]).cuda()

    train_ds = DistillShardDataset(cfg)
    pad_id = student.tokenizer.pad_token_id
    dl = DataLoader(train_ds, batch_size=cfg.stage1.micro_batch, shuffle=True,
                    num_workers=4, pin_memory=True,
                    collate_fn=functools.partial(collate_stage1, pad_id=pad_id))

    groups = student.param_groups_stage1()
    groups.append({"params": list(projections.parameters()),
                   "lr_mult": cfg.student.new_token_lr_mult})
    opt = build_optimizer(groups, cfg.stage1.lr, cfg.stage1.weight_decay)

    steps_per_epoch = math.ceil(len(dl) / cfg.stage1.grad_accum)
    total_steps = steps_per_epoch * cfg.stage1.epochs
    warmup = int(total_steps * cfg.stage1.warmup_frac)

    best, patience, step = float("inf"), 0, 0
    for epoch in range(cfg.stage1.epochs):
        for i, batch in enumerate(dl):
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student.ar_forward(batch)
                proj = projections({int(k): v for k, v in out["hidden"].items()})
                w = losses.stage_weights(cfg.stage1, step, total_steps)
                # Trajectory logits are the slice of the LM head over traj positions;
                # ar_forward returns logits aligned with the collated token layout.
                loss = (w["traj_kl"] * losses.traj_topk_kl(
                            out["logits"], batch["topk_idx"], batch["topk_logp"], batch["traj_mask"])
                        + w["text_kl"] * losses.text_kl_or_ce(
                            out["logits"], batch["coc"], batch["coc_mask"], vocab_ok=True)
                        + w["feat"] * losses.feature_match(proj, batch["feats"])
                        + w["gt_ce"] * losses.gt_traj_ce(
                            out["logits"], batch["traj"], batch["traj_mask"]))
            (loss / cfg.stage1.grad_accum).backward()
            if (i + 1) % cfg.stage1.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * cosine_lr(step, total_steps, warmup) \
                        if "initial_lr" in g else g["lr"]
                torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps, loss.item())

        score = coarse_minade(cfg, student, split="challenging")
        log.info("epoch %d challenging coarse-minADE %.3f m", epoch, score)
        if score < best:
            best, patience = score, 0
            ckpt = Path(cfg.paths.runs_root) / "stage1" / "best"
            ckpt.mkdir(parents=True, exist_ok=True)
            student.model.save_pretrained(ckpt)
            torch.save(projections.state_dict(), ckpt / "projections.pt")
            torch.save(lmap, ckpt / "layer_map.pt")
        else:
            patience += 1
            if patience >= cfg.stage1.early_stop_patience:
                log.info("early stop: no improvement for %d evals", patience)
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(a.config)

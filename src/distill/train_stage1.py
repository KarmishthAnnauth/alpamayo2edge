"""Stage 1: distill the teacher reasoner into Edge's AR tower.

Single-GPU (RTX 6000 Ada, 48 GB) loop: bf16 autocast, gradient checkpointing,
accumulation. Early stopping on challenging-split coarse minADE (decoded from
discrete tokens alone - no diffusion tower involved, plan gate 3.3).

The AR tower is adapted through LoRA (D-024): trainable is ~25M adapter params
plus the appended trajectory rows, not the 2.48B the full-FT version needed.
Checkpoints are written MERGED via distill.checkpoint - see that module.
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
from .data.dataset import Stage1Dataset, collate_stage1
from .student.edge_wrapper import EdgeStudent
from .student.layer_map import FeatureProjections, uniform_map
from . import checkpoint, losses
from .optim import build_optimizer, cosine_lr, set_lr, trainable_report
from .eval.coarse_minade import coarse_minade

log = logging.getLogger(__name__)


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

    # The shards are teacher targets only; Stage1Dataset re-loads each window so
    # the student gets the same cameras + ego motion the teacher saw (D-028).
    train_ds = Stage1Dataset(cfg, student.context_builder())
    pad_id = student.tokenizer.pad_token_id
    dl = DataLoader(train_ds, batch_size=cfg.stage1.micro_batch, shuffle=True,
                    num_workers=4, pin_memory=True,
                    collate_fn=functools.partial(collate_stage1, pad_id=pad_id))

    groups = student.param_groups_stage1()
    groups.append({"params": list(projections.parameters()),
                   "lr_mult": cfg.student.new_token_lr_mult})
    opt = build_optimizer(groups, cfg.stage1.lr, cfg.stage1.weight_decay)
    log.info("LoRA: %s", student.lora_stats)
    log.info("%s", trainable_report(groups))

    steps_per_epoch = math.ceil(len(dl) / cfg.stage1.grad_accum)
    total_steps = steps_per_epoch * cfg.stage1.epochs
    warmup = int(total_steps * cfg.stage1.warmup_frac)

    best, patience, step = float("inf"), 0, 0
    for epoch in range(cfg.stage1.epochs):
        for i, batch in enumerate(dl):
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student.ar_forward(batch, capture_layers=lmap.keys())
                proj = projections({int(k): v for k, v in out["hidden"].items()})
                w = losses.stage_weights(cfg.stage1, step, total_steps)

                # The context is one long sequence; every loss below wants the
                # compact per-target layout, shifted by one for next-token
                # prediction. gather_targets does both (see its docstring).
                n_traj = batch["traj"].shape[1]
                traj_logits, traj_tgt, traj_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["traj_pos"], n_traj)
                coc_logits, coc_tgt, coc_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["coc_pos"],
                    batch["coc"].shape[1])
                traj_mask = batch["traj_mask"] & traj_ok
                # The teacher cached REGION-RELATIVE bins (D-014); the student's
                # logits are over its whole extended vocabulary, so the top-k
                # indices need the appended-row offset before they can index it.
                topk_idx = batch["topk_idx"] + student.future_base

                loss = (w["traj_kl"] * losses.traj_topk_kl(
                            traj_logits, topk_idx, batch["topk_logp"], traj_mask)
                        + w["text_kl"] * losses.text_kl_or_ce(
                            coc_logits, coc_tgt, batch["coc_mask"] & coc_ok, vocab_ok=True)
                        + w["feat"] * losses.feature_match(proj, batch["feats"])
                        + w["gt_ce"] * losses.gt_traj_ce(
                            traj_logits, traj_tgt, traj_mask))
            (loss / cfg.stage1.grad_accum).backward()
            if (i + 1) % cfg.stage1.grad_accum == 0:
                set_lr(opt, cosine_lr(step, total_steps, warmup))
                torch.nn.utils.clip_grad_norm_(
                    student.trainable_parameters() + list(projections.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps, loss.item())

        score = coarse_minade(cfg, student, split="challenging")
        log.info("epoch %d challenging coarse-minADE %.3f m", epoch, score)
        if score < best:
            best, patience = score, 0
            ckpt = Path(cfg.paths.runs_root) / "stage1" / "best"
            # Merged, so stage 2 can load it into a student with no adapters
            # injected; non-destructive, so this epoch's training continues.
            checkpoint.save(student, ckpt, stage="stage1", epoch=epoch,
                            coarse_minade=float(score))
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

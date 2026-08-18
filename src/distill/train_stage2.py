"""Stage 2: distill the action expert's flow field into Edge's diffusion tower.

Pure supervised regression against cached (t, a_t, v_teacher) tuples - no
teacher model in memory (plan v2). AR tower frozen (optional LoRA). The AR
context KV is computed once per window and reused across its K flow samples.
Scheduled sampling: after `scheduled_sampling_start_frac`, the conditioning
trajectory tokens come from the student's own generation, not teacher forcing.
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
from .data.dataset import DistillShardDataset, collate_stage2
from .student.edge_wrapper import EdgeStudent
from . import losses

log = logging.getLogger(__name__)


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    student = EdgeStudent(cfg).cuda()
    stage1_ckpt = Path(cfg.paths.runs_root) / "stage1" / "best"
    student.model = student.model.from_pretrained(stage1_ckpt).cuda()
    if cfg.stage2.grad_checkpoint:
        student.model.gradient_checkpointing_enable()

    ds = DistillShardDataset(cfg)
    pad_id = student.tokenizer.pad_token_id
    dl = DataLoader(ds, batch_size=cfg.stage2.micro_batch, shuffle=True,
                    num_workers=4, pin_memory=True,
                    collate_fn=functools.partial(collate_stage2, pad_id=pad_id))

    opt = torch.optim.AdamW(
        [p for g in student.param_groups_stage2() for p in g["params"]],
        lr=cfg.stage2.lr, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(dl) / cfg.stage2.grad_accum)
    total_steps = steps_per_epoch * cfg.stage2.epochs
    step = 0
    for epoch in range(cfg.stage2.epochs):
        for i, batch in enumerate(dl):
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            frac = step / max(total_steps, 1)
            use_student_tokens = frac >= cfg.stage2.scheduled_sampling_start_frac

            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_student_tokens:
                    with torch.no_grad():
                        batch["traj"] = student.generate_traj_tokens(batch)
                # One frozen AR pass per window -> KV reused across K flow samples.
                with torch.no_grad():
                    ctx_kv = student.ar_forward(batch).get("kv_cache")
                owner = batch["flow_owner"]
                v_pred = student.flow_forward(
                    batch, a_t=batch["flow_a_t"], t=batch["flow_t"],
                    context_kv=(ctx_kv, owner))
                w = losses.stage_weights(cfg.stage2, step, total_steps)
                a1 = batch["gt_traj"][owner]
                # Teacher schedule VERIFIED (D-012): a_t = t a1 + (1-t) a0 with
                # a0 = noise, t = data weight, so a0 = (a_t - t a1) / (1 - t).
                # Labeler caps t at 0.999, keeping this well-conditioned.
                t = batch["flow_t"].view(-1, 1, 1)
                a0 = (batch["flow_a_t"] - t * a1) / (1 - t).clamp_min(1e-3)
                # stage2.supervision (D-007): teacher_flow distills the expert's
                # velocity field; gt_flow is the ablation baseline (GT-only target).
                supervision = cfg.stage2.get("supervision", "teacher_flow")
                if supervision == "gt_flow":
                    loss = losses.flow_matching_gt(v_pred, a0, a1)
                else:
                    loss = (w["flow_distill"] * losses.flow_distill(v_pred, batch["flow_v"])
                            + w["fm_gt"] * losses.flow_matching_gt(v_pred, a0, a1))
            (loss / cfg.stage2.grad_accum).backward()
            if (i + 1) % cfg.stage2.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps, loss.item())

        ckpt = Path(cfg.paths.runs_root) / "stage2" / f"epoch{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        student.model.save_pretrained(ckpt)
        log.info("saved %s - run scripts/05_eval.py for full-pipeline minADE", ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(a.config)

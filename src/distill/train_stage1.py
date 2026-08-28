"""Stage 1: distill the teacher reasoner into Edge's AR tower.

Single-GPU loop on the SLURM-allocated RTX PRO 6000 Blackwell (96 GB): bf16
autocast, gradient checkpointing, accumulation. Launch via
`sbatch scripts/03_train_stage1.sh` — the Ada is outside SLURM, so a bare
`python -m distill.train_stage1` lands on the wrong card. Early stopping on challenging-split coarse minADE (decoded from
discrete tokens alone - no diffusion tower involved, plan gate 3.3).

The AR tower is adapted through LoRA (D-024): trainable is ~25M adapter params
plus the appended trajectory rows, not the 2.48B the full-FT version needed.
Checkpoints are written MERGED via distill.checkpoint - see that module.
"""
from __future__ import annotations
import argparse
import functools
import os
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data.dataset import Stage1Dataset, collate_stage1, move_batch
from .data.splits import load_split
from .student.edge_wrapper import EdgeStudent
from .student.layer_map import (FeatureProjections, cka_map, pool_prompt_segments,
                                uniform_map)
from . import checkpoint, losses
from .optim import build_optimizer, cosine_lr, set_lr, trainable_report
from .eval.coarse_minade import coarse_minade

log = logging.getLogger(__name__)


def _cached_feature_dim(shard_path, layer: int) -> int:
    """D_t, read off a cached shard's own feature array."""
    import numpy as np
    with np.load(shard_path) as z:
        return int(z[f"feat_{layer}"].shape[-1])


@torch.no_grad()
def _build_layer_map(cfg, student, dl, feat_layers, n_student_layers, pool_len):
    """Teacher->student layer correspondence, per `student.layer_map.mode`.

    `uniform` is the baseline; `cka` is D-008's primary and the config's default,
    and it was dead code until 2026-08-24 — the trainer hardcoded `uniform_map`
    regardless of the setting. CKA needs student features, so it runs here on a
    few batches of the untrained student rather than in a separate script: the
    projections do not exist yet, and the probe is a handful of forward passes.
    """
    mode = str(cfg.student.layer_map.get("mode", "uniform")).lower()
    if mode != "cka":
        return uniform_map(feat_layers, n_student_layers)

    n_batches = max(1, int(cfg.student.layer_map.get("cka_probe_batches", 4)))
    s_acc: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_student_layers)}
    t_acc: dict[int, list[torch.Tensor]] = {l: [] for l in feat_layers}
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batch = move_batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.ar_forward(batch, capture_layers=list(range(n_student_layers)))
        for k, v in out["hidden"].items():
            pooled = pool_prompt_segments(v, batch["n_prompt"], pool_len)
            s_acc[int(k)].append(pooled.reshape(-1, pooled.shape[-1]).float().cpu())
        for l in feat_layers:
            tf = batch["feats"][l]
            t_acc[l].append(tf.reshape(-1, tf.shape[-1]).float().cpu())

    if not t_acc[feat_layers[0]]:
        log.warning("CKA probe saw no batches; falling back to the uniform map")
        return uniform_map(feat_layers, n_student_layers)
    s_feats = {k: torch.cat(v) for k, v in s_acc.items() if v}
    t_feats = {k: torch.cat(v) for k, v in t_acc.items()}
    return cka_map(s_feats, t_feats)


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    torch.backends.cuda.matmul.allow_tf32 = True

    student = EdgeStudent(cfg).cuda()
    # weights_only=False: the spec pickles the teacher tokenizer's bound `decode`
    # and a HistoryTokenize instance, not tensors, and torch >= 2.6 defaults
    # weights_only=True and refuses to unpickle either (D-032).
    traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt",
                           weights_only=False)
    student.extend_trajectory_vocab(traj_spec)
    if cfg.stage1.grad_checkpoint:
        student.enable_gradient_checkpointing()

    expert_layers = torch.load(Path(cfg.paths.cache_root) / "expert_layers.pt",
                               weights_only=False)
    # The layers the CACHE holds, not the ones the expert attends to: the expert
    # attends all 36 (D-004), `teacher.feat_layers` is the 8 that were written.
    feat_layers = [int(l) for l in cfg.teacher.raw["feat_layers"]]
    pool_len = int(cfg.teacher.get("feat_pool_len", 8))
    n_student_layers = student.n_layers

    # The shards are teacher targets only; Stage1Dataset re-loads each window so
    # the student gets the same cameras + ego motion the teacher saw (D-028).
    # Restricted to split_train: the epoch gate evaluates split_challenging, and
    # until 2026-08-24 this took every shard on disk, gate windows included.
    train_ds = Stage1Dataset(cfg, student.context_builder(),
                             clip_ids=load_split(cfg, "train"))
    pad_id = student.tokenizer.pad_token_id
    # Slurm allocates the CPUs; hardcoding a worker count is how a shared node
    # ends up oversubscribed (slurm_tutorial/07 "three mistakes", #1). Falls back
    # to 4 outside a job.
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    dl = DataLoader(train_ds, batch_size=cfg.stage1.micro_batch, shuffle=True,
                    num_workers=n_workers, pin_memory=True,
                    collate_fn=functools.partial(collate_stage1, pad_id=pad_id))

    lmap = _build_layer_map(cfg, student, dl, feat_layers, n_student_layers, pool_len)
    # D_t comes from the cache itself, which is the only authority on it: A1.5's
    # config.json does not carry Cosmos-Reason2-8B's hidden size (D-019), and
    # `expert_layers["kv_dim"]` — what this read until 2026-08-24 — is not a key
    # `probe_expert_conditioning` returns at all (D-032). Reading one shard header
    # costs nothing and cannot disagree with what stage 1 will be regressing onto.
    d_teacher = _cached_feature_dim(train_ds.shards[0], feat_layers[0])
    if d_teacher != int(expert_layers.get("teacher_hidden", d_teacher)):
        log.warning("cached feature dim %d != probed teacher_hidden %d — trusting "
                    "the cache", d_teacher, expert_layers["teacher_hidden"])
    projections = FeatureProjections(
        lmap, student.hidden_size, d_teacher).cuda()
    log.info("layer map (student -> teacher): %s", dict(sorted(lmap.items())))

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
            batch = move_batch(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student.ar_forward(batch, capture_layers=lmap.keys())
                # The cache holds `(pool_len, D_t)` per layer, so the student's
                # per-token states have to be pooled the same way before they can
                # be compared to it (see pool_prompt_segments).
                proj = projections({
                    int(k): pool_prompt_segments(v, batch["n_prompt"], pool_len)
                    for k, v in out["hidden"].items()})
                w = losses.stage_weights(cfg.stage1, step, total_steps)

                # The context is one long sequence; every loss below wants the
                # compact per-target layout, shifted by one for next-token
                # prediction. gather_targets does both (see its docstring).
                n_traj = batch["traj"].shape[1]
                traj_logits, traj_tgt, traj_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["traj_pos"], n_traj)
                # n_targets is the STUDENT's CoC length, not the teacher's.
                # `batch["coc"]` is the teacher's re-tokenizable text in the
                # TEACHER's vocabulary, and the two token counts differ by
                # construction (D-011, cross-family tokenizers). Sizing the
                # gather by it truncated every student CoC that tokenized longer
                # — silently dropping the tail of the text KD target — and the
                # mask below then intersected two different length conventions.
                n_coc = int(batch["coc_pos"].sum(1).max())
                coc_logits, coc_tgt, coc_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["coc_pos"], n_coc)
                traj_mask = batch["traj_mask"] & traj_ok
                # The teacher cached REGION-RELATIVE bins (D-014); the student's
                # logits are over its whole extended vocabulary, so the top-k
                # indices need the appended-row offset before they can index it.
                topk_idx = batch["topk_idx"] + student.future_base

                loss = (w["traj_kl"] * losses.traj_topk_kl(
                            traj_logits, topk_idx, batch["topk_logp"], traj_mask)
                        + w["text_kl"] * losses.text_kl_or_ce(
                            coc_logits, coc_tgt, coc_ok, vocab_ok=True)
                        + w["feat"] * losses.feature_match(proj, batch["feats"])
                        # GT bins, NOT traj_tgt: traj_tgt is read back out of
                        # input_ids, which carry the TEACHER's tokens, so passing it
                        # here made gt_ce a hard-label restatement of traj_kl instead
                        # of an independent anchor. Same appended-row offset as topk.
                        + w["gt_ce"] * losses.gt_traj_ce(
                            traj_logits, batch["gt_traj_tok"] + student.future_base,
                            traj_mask))
            (loss / cfg.stage1.grad_accum).backward()
            if (i + 1) % cfg.stage1.grad_accum == 0:
                set_lr(opt, cosine_lr(step, total_steps, warmup))
                torch.nn.utils.clip_grad_norm_(
                    student.trainable_parameters() + list(projections.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, step, total_steps, loss.item())

        score = coarse_minade(cfg, student, split="challenging",
                              k=int(cfg.eval.minade_k),
                              max_windows=cfg.eval.get("gate_max_windows", None))
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

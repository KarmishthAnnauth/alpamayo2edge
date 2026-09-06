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
from datetime import datetime
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

#: per-term log key -> config `loss_weights` key
_WKEY = {"traj": "traj_kl", "traj_accel": "traj_kl_accel", "text": "text_kl",
         "struct": "struct_ce", "feat": "feat", "gt_soft": "gt_ce"}

#: logged but NOT summed into the loss. `gt` is the run-1..3 one-hot `gt_ce`,
#: kept so the curve stays comparable across runs now that `gt_soft` carries the
#: gradient; it shares `gt_ce`'s weight key and would otherwise double-count.
_DIAG = ("gt",)


def _run_best_dir(cfg, stage: str = "stage1") -> Path:
    """This run's own `best` checkpoint dir, so a second run never clobbers the
    first (2026-08-31: job 207 was about to overwrite job 183's 2.352 m).
    `SLURM_JOB_ID` names it when present, otherwise a timestamp."""
    rid = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(cfg.paths.runs_root) / stage / f"run-{rid}" / "best"


def _promote_best(run_best: Path) -> None:
    """Point `<stage>/best` at this run's checkpoint via a relative symlink, so
    stage 2 and the eval scripts keep resolving one stable path. A pre-isolation
    real directory sitting at `best` is moved aside once."""
    stage_root = run_best.parent.parent          # runs_root/<stage>
    link = stage_root / "best"
    if link.exists() and not link.is_symlink() and link.is_dir():
        legacy = stage_root / f"legacy-best-{int(link.stat().st_mtime)}"
        link.rename(legacy)
        log.warning("moved pre-isolation checkpoint %s -> %s", link, legacy)
    tmp = stage_root / f".best.{os.getpid()}.tmp"
    tmp.unlink(missing_ok=True)
    os.symlink(run_best.relative_to(stage_root), tmp)   # e.g. run-207/best
    os.replace(tmp, link)


@torch.no_grad()
def _coc_nll(cfg, student, split: str = "val", max_windows: int = 64) -> tuple[float, float]:
    """Teacher-forced CoC / structural cross-entropy on a held-out split.

    The epoch gate is blind to the CoC (eval_phase1.md §5); this is the number
    that says whether `text_kl` / `struct_ce` are doing anything. No generation —
    one forward pass over `max_windows` val windows. Returns (text_nll, struct_nll),
    each a mean over micro-batches so it is comparable across runs.
    """
    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, split))
    if max_windows and len(ds) > max_windows:
        ds = torch.utils.data.Subset(ds, range(max_windows))
    dl = DataLoader(ds, batch_size=cfg.stage1.micro_batch, shuffle=False, num_workers=2,
                    collate_fn=functools.partial(
                        collate_stage1, pad_id=student.tokenizer.pad_token_id))
    text_tot = struct_tot = 0.0
    n = 0
    for batch in dl:
        batch = move_batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.ar_forward(batch, capture_layers=())
            n_coc = int(batch["coc_pos"].sum(1).max())
            coc_logits, coc_tgt, coc_ok = losses.gather_targets(
                out["logits"], batch["input_ids"], batch["coc_pos"], n_coc)
            n_struct = int(batch["struct_pos"].sum(1).max())
            s_logits, s_tgt, s_ok = losses.gather_targets(
                out["logits"], batch["input_ids"], batch["struct_pos"], n_struct)
            text_tot += float(losses.text_kl_or_ce(coc_logits, coc_tgt, coc_ok, vocab_ok=True))
            struct_tot += float(losses.text_kl_or_ce(s_logits, s_tgt, s_ok, vocab_ok=True))
        n += 1
    return (text_tot / max(n, 1), struct_tot / max(n, 1))


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

    run_best = _run_best_dir(cfg)
    log.info("checkpoints -> %s  (<stage>/best symlinks here)", run_best)

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
                # The `<|cot_end|><|traj_future_start|>` boundary tokens. Run 1
                # never supervised these, so the student could not terminate its
                # CoC and free-ran into the base model's `</think>` (job 202,
                # eval_phase1.md §1). Same `input_ids`-as-target CE as the CoC.
                n_struct = int(batch["struct_pos"].sum(1).max())
                struct_logits, struct_tgt, struct_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["struct_pos"], n_struct)
                traj_mask = batch["traj_mask"] & traj_ok
                # The 128-token future stream is 64 waypoints x 2 action dims,
                # interleaved, in the teacher's EMISSION order: even = curvature,
                # odd = accel (D-031's swap, re-verified against `gt_traj`'s
                # columns at r = 0.999/1.000). `gather_targets` preserves that
                # order — `nonzero` is sorted and position 0 can never be a
                # trajectory token — so the compact layout keeps the same parity.
                #
                # They are split because the teacher is worth distilling on one
                # and not the other. Cache measurement, 400 clips / 102k
                # positions, teacher mass within +/-w bins of the GT bin:
                #
                #        +/-2    +/-10   +/-20   +/-100
                #   curv 0.420   0.708   0.806   0.931
                #   acc  0.028   0.094   0.170   0.577
                #
                # Its path shape is good; its speed profile is close to
                # uninformative about GT even at a +/-100-bin tolerance. That is
                # the `traj_kl` vs `gt_ce` fight, and it lives almost entirely in
                # the accel half of the stream — so weight the halves apart
                # rather than turning the whole teacher down (run 4).
                dim0 = torch.zeros_like(traj_mask)
                dim0[:, 0::2] = True
                curv_mask, acc_mask = traj_mask & dim0, traj_mask & ~dim0
                # The teacher cached REGION-RELATIVE bins (D-014); the student's
                # logits are over its whole extended vocabulary, so the top-k
                # indices need the appended-row offset before they can index it.
                topk_idx = batch["topk_idx"] + student.future_base

                # Per-term, unweighted — so the log can attribute the curve
                # (eval_phase1.md §4). The GT terms use the cached GT bins
                # (`gt_bins`), NOT `traj_tgt`:
                # `traj_tgt` is read back out of `input_ids`, which carry the
                # TEACHER's tokens, so passing it there made `gt` a hard-label
                # restatement of `traj` instead of an independent anchor. Same
                # appended-row offset as `topk_idx`.
                gt_bins = batch["gt_traj_tok"] + student.future_base
                # One forward, two reductions: the per-position KL does not
                # depend on the mask, and it carries a full-vocab log_softmax
                # (~276 MB of retained activation at micro_batch 4).
                traj_kl_pos = losses.traj_topk_kl_per_pos(
                    traj_logits, topk_idx, batch["topk_logp"])
                terms = {
                    "traj": losses.masked_mean(traj_kl_pos, curv_mask),
                    "traj_accel": losses.masked_mean(traj_kl_pos, acc_mask),
                    "text": losses.text_kl_or_ce(
                        coc_logits, coc_tgt, coc_ok, vocab_ok=True),
                    "struct": losses.text_kl_or_ce(
                        struct_logits, struct_tgt, struct_ok, vocab_ok=True),
                    "feat": losses.feature_match(proj, batch["feats"]),
                    # Distance-aware, because a one-hot over 3000 bins is what
                    # kept this term flat at ~6 nats for all of job 243. Floors
                    # near ln(sigma*sqrt(2*pi*e)), NOT 0 — see gt_traj_soft_ce.
                    "gt_soft": losses.gt_traj_soft_ce(
                        traj_logits, gt_bins, traj_mask,
                        sigma_bins=float(cfg.stage1.get("gt_soft_sigma_bins", 6.0)),
                        lo=student.future_base,
                        hi=student.future_base + student.n_future_bins),
                }
                loss = sum(w.get(_WKEY[k], 0.0) * v
                           for k, v in terms.items() if k not in _DIAG)
                with torch.no_grad():   # run-1..3 comparability only (_DIAG)
                    terms["gt"] = losses.gt_traj_ce(traj_logits, gt_bins, traj_mask)
            (loss / cfg.stage1.grad_accum).backward()
            if (i + 1) % cfg.stage1.grad_accum == 0:
                set_lr(opt, cosine_lr(step, total_steps, warmup))
                torch.nn.utils.clip_grad_norm_(
                    student.trainable_parameters() + list(projections.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    breakdown = " ".join(f"{k} {float(v):.3f}" for k, v in terms.items())
                    log.info("epoch %d step %d/%d loss %.4f [%s]",
                             epoch, step, total_steps, loss.item(), breakdown)

        score = coarse_minade(cfg, student, split="challenging",
                              k=int(cfg.eval.minade_k),
                              max_windows=cfg.eval.get("gate_max_windows", None))
        # The gate teacher-forces the CoC and scores only the trajectory (§5), so
        # it cannot see whether `text`/`struct` are learning. This can (§4): same
        # two CE terms, teacher-forced, on the held-out val split.
        text_nll, struct_nll = _coc_nll(cfg, student,
                                        max_windows=cfg.eval.get("gate_max_windows", 64))
        # `gt_ce` is annealed over the run (losses.stage_weights); log the current
        # weight next to the gate so the schedule is visible per epoch (wandb_tail).
        gt_ce_w = losses.stage_weights(cfg.stage1, step, total_steps).get("gt_ce", 0.0)
        log.info("epoch %d challenging coarse-minADE %.3f m | val CoC NLL %.3f "
                 "struct NLL %.3f | gt_ce_w %.3f",
                 epoch, score, text_nll, struct_nll, gt_ce_w)
        if score < best:
            best, patience = score, 0
            # Per-run dir (see _run_best_dir); `<stage>/best` is repointed to it.
            # Merged, so stage 2 can load it into a student with no adapters
            # injected; non-destructive, so this epoch's training continues.
            checkpoint.save(student, run_best, stage="stage1", epoch=epoch,
                            coarse_minade=float(score))
            torch.save(projections.state_dict(), run_best / "projections.pt")
            torch.save(lmap, run_best / "layer_map.pt")
            _promote_best(run_best)
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

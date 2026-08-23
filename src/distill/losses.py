"""Losses for both stages. Shapes follow data/dataset.py collators."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def gather_targets(logits: torch.Tensor, input_ids: torch.Tensor,
                   pos_mask: torch.Tensor, n_targets: int):
    """Line the student's sequence up with the cached teacher targets.

    The student context is one long sequence (cameras, ego bins, instruction,
    CoC, trajectory bins), but every loss here is written against a compact
    per-target layout. This selects the marked positions and returns
    `(logits, target_ids, valid)` shaped (B, n_targets, V) / (B, n_targets) /
    (B, n_targets).

    TWO alignment rules, both easy to get silently wrong:

    * **Autoregressive shift.** The logit that PREDICTS the token at position p
      sits at p-1. We gather at `positions - 1`, never at the positions.
    * **Targets come from the sequence, not from the cache.** The cached CoC ids
      are in the TEACHER's vocabulary; the student's own ids for the same text
      are already in `input_ids`, because the context builder tokenized the
      teacher's `coc_text` with the student's tokenizer. Reading targets back
      out of `input_ids` sidesteps the whole vocab-match question (D-011).

    The batch loop is deliberate: micro-batches are 4-8 here, and a vectorized
    scatter would obscure the shift rule for no measurable gain.
    """
    B, L, V = logits.shape
    out = logits.new_zeros(B, n_targets, V)
    tgt = input_ids.new_zeros(B, n_targets)
    valid = torch.zeros(B, n_targets, dtype=torch.bool, device=logits.device)
    for b in range(B):
        pos = pos_mask[b].nonzero(as_tuple=True)[0]
        pos = pos[pos > 0][:n_targets]          # position 0 has no predictor
        k = pos.shape[0]
        if k == 0:
            continue
        out[b, :k] = logits[b, pos - 1]
        tgt[b, :k] = input_ids[b, pos]
        valid[b, :k] = True
    return out, tgt, valid


def traj_topk_kl(student_logits: torch.Tensor, topk_idx: torch.Tensor,
                 topk_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) on discrete trajectory tokens over K+1 buckets:
    the teacher's cached top-k plus an explicit tail bucket.

    student_logits: (B, L, V); topk_idx: (B, L, K); topk_logp: (B, L, K) are
    the teacher's *full-softmax* log-probs at its top-k (as cached by the
    labeler); mask: (B, L).

    The tail bucket matters: without it, renormalizing the top-k makes a
    student that exactly matches the teacher still pay a constant penalty for
    its own out-of-top-k mass, and the optimum shifts to concentrating all
    mass on the support. With the tail bucket, exact match => KL = 0.
    """
    eps = 1e-9
    logp_s = F.log_softmax(student_logits, dim=-1)
    logp_s_k = torch.gather(logp_s, -1, topk_idx.clamp_min(0))       # (B,L,K)
    p_t_k = topk_logp.exp()                                          # (B,L,K)
    p_t_tail = (1 - p_t_k.sum(-1)).clamp_min(0.0)                    # (B,L)
    p_s_tail = (1 - logp_s_k.exp().sum(-1)).clamp_min(eps)           # (B,L)
    kl = (p_t_k * (topk_logp - logp_s_k)).sum(-1) \
        + p_t_tail * (torch.log(p_t_tail + eps) - torch.log(p_s_tail))
    kl = kl.clamp_min(0.0)  # numerical guard
    return (kl * mask).sum() / mask.sum().clamp_min(1)


def text_kl_or_ce(student_logits: torch.Tensor, target_ids: torch.Tensor,
                  mask: torch.Tensor, vocab_ok: bool) -> torch.Tensor:
    """Short-CoC / meta-action distillation. With matching vocabs we use CE on
    teacher-generated tokens (equivalent to KL against the teacher's sample).
    On vocab mismatch, target_ids must already be re-tokenized into the
    student vocab by the labeler (Phase 0.3 fallback)."""
    del vocab_ok  # both branches reduce to CE once re-tokenization is upstream
    ce = F.cross_entropy(student_logits.transpose(1, 2), target_ids, reduction="none")
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def feature_match(proj_student: dict[int, torch.Tensor],
                  teacher_feats: dict[int, torch.Tensor],
                  kind: str = "smooth_l1") -> torch.Tensor:
    """Regression on pooled features at mapped layers.

    Orion-Lite's ablation (Table 5) found plain regression beats distributional
    metrics for continuous latents, with L1-family most robust to outlier
    activations - hence smooth_l1 default rather than KL/cosine.
    """
    losses = []
    for layer, tf in teacher_feats.items():
        if layer not in proj_student:
            continue
        sf = proj_student[layer]
        tf = tf.to(sf.dtype)
        if kind == "cosine":
            losses.append(1 - F.cosine_similarity(sf, tf, dim=-1).mean())
        else:
            losses.append(F.smooth_l1_loss(sf, tf))
    return torch.stack(losses).mean() if losses else torch.zeros((), device="cuda")


def gt_traj_ce(student_logits: torch.Tensor, gt_token_ids: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Anchor CE on ground-truth trajectory tokens (guards against teacher errors).

    `gt_token_ids` MUST come from the cache's `gt_traj_token_ids` (the GT future run
    through the teacher's own tokenizer), offset into the student's appended rows.
    Passing the targets `gather_targets` returns instead reads the TEACHER's tokens
    back out of `input_ids` and quietly turns this into a hard-label copy of
    `traj_topk_kl` - which is exactly what it was until 2026-08-23.
    """
    ce = F.cross_entropy(student_logits.transpose(1, 2), gt_token_ids, reduction="none")
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def flow_distill(v_student: torch.Tensor, v_teacher: torch.Tensor) -> torch.Tensor:
    """Regress student velocity onto cached teacher velocity at the same (a_t, t).
    Distills the flow field pointwise - no sampling, no mode averaging.
    Computed in fp32, mirroring the teacher's pred.float() loss (D-006)."""
    return F.mse_loss(v_student.float(), v_teacher.float())


def flow_matching_gt(v_student: torch.Tensor, a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    """Flow-matching target, VERIFIED against the teacher (D-012):
    a_t = t a1 + (1-t) a0 (a0 noise, a1 data, t = data weight), v* = a1 - a0.
    Fp32 loss, mirroring the teacher (D-006)."""
    return F.mse_loss(v_student.float(), (a1 - a0).float())


def stage_weights(cfg_stage, step: int, total_steps: int) -> dict[str, float]:
    """Time-varying loss weights: feature warmup (stage 1) and GT annealing
    (stage 2) from the config fractions."""
    w = dict(cfg_stage.loss_weights.raw)
    frac = step / max(total_steps, 1)
    if "feat" in w:
        warm = cfg_stage.get("feat_warmup_frac", 0.0) or 0.0
        if warm > 0:
            w["feat"] *= min(frac / warm, 1.0)
    if "flow_distill" in w:
        anneal = cfg_stage.get("anneal_to_gt_frac", 0.0) or 0.0
        if anneal > 0 and frac > 1 - anneal:
            a = (frac - (1 - anneal)) / anneal
            w["flow_distill"] *= (1 - a)
            w["fm_gt"] *= (1 + a)
    return w

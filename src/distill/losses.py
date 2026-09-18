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


def traj_topk_kl_per_pos(student_logits: torch.Tensor, topk_idx: torch.Tensor,
                         topk_logp: torch.Tensor) -> torch.Tensor:
    """Per-position KL(teacher || student), (B, L), before any masking.

    Split out from `traj_topk_kl` for run 4, which reduces the SAME positions
    under two different masks (curvature and accel). Calling the reducing
    version twice would run `log_softmax` over the full ~135k-row vocabulary
    twice and keep both outputs alive for backward — ~276 MB of activation each
    at micro_batch 4 — for a quantity that does not depend on the mask at all.
    """
    eps = 1e-9
    logp_s = F.log_softmax(student_logits, dim=-1)
    logp_s_k = torch.gather(logp_s, -1, topk_idx.clamp_min(0))       # (B,L,K)
    p_t_k = topk_logp.exp()                                          # (B,L,K)
    p_t_tail = (1 - p_t_k.sum(-1)).clamp_min(0.0)                    # (B,L)
    p_s_tail = (1 - logp_s_k.exp().sum(-1)).clamp_min(eps)           # (B,L)
    kl = (p_t_k * (topk_logp - logp_s_k)).sum(-1) \
        + p_t_tail * (torch.log(p_t_tail + eps) - torch.log(p_s_tail))
    return kl.clamp_min(0.0)  # numerical guard


def masked_mean(per_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a (B, L) per-position loss over a (B, L) boolean mask."""
    return (per_pos * mask).sum() / mask.sum().clamp_min(1)


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

    Reduce two masks over one forward with `traj_topk_kl_per_pos` + `masked_mean`
    instead of calling this twice — see that function.
    """
    return masked_mean(
        traj_topk_kl_per_pos(student_logits, topk_idx, topk_logp), mask)


def text_kl_or_ce(student_logits: torch.Tensor, target_ids: torch.Tensor,
                  mask: torch.Tensor, vocab_ok: bool) -> torch.Tensor:
    """Short-CoC / meta-action distillation. With matching vocabs we use CE on
    teacher-generated tokens (equivalent to KL against the teacher's sample).
    On vocab mismatch, target_ids must already be re-tokenized into the
    student vocab by the labeler (Phase 0.3 fallback)."""
    del vocab_ok  # both branches reduce to CE once re-tokenization is upstream
    if student_logits.shape[1] == 0 or not bool(mask.any()):
        # Every sample in the micro-batch had this span masked out (image
        # dropout, D-043): nothing to score, and cross_entropy over a zero-length
        # target axis is an error rather than a zero.
        return student_logits.new_zeros(())
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

    Run 4 demoted this to a DIAGNOSTIC: `gt_traj_soft_ce` is what carries the
    gradient. Kept because it is the number runs 1-3 logged, so the curve stays
    comparable across runs. See that function for why.
    """
    ce = F.cross_entropy(student_logits.transpose(1, 2), gt_token_ids, reduction="none")
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def gt_traj_soft_ce(student_logits: torch.Tensor, gt_token_ids: torch.Tensor,
                    mask: torch.Tensor, sigma_bins: float,
                    lo: int, hi: int, per_pos: bool = False) -> torch.Tensor:
    """Distance-aware GT trajectory anchor: CE against a Gaussian over the bins
    NEIGHBOURING the GT bin, rather than a one-hot on the GT bin itself.

    Why this exists (run-4 cache measurement, 400 clips / 102k positions). The
    future region is 3000 bins, so uniform NLL is ln(3000) = 8.01 nats. Job 243's
    `gt_ce` sat flat at ~6.0 nats for all 8 epochs while `traj_kl` fell 4.4 ->
    0.38: a one-hot target over bins that fine is nearly unlearnable, because
    nothing in the loss knows that bin 1498 and bin 1502 are the same trajectory
    to within a few centimetres. Upweighting a signal shaped like that buys
    gradient variance, not accuracy - which is exactly what runs 2 and 3 bought.

    A Gaussian target restores the metric the bin index already carries. It is
    ordinary CE, so it composes with everything else here; as `sigma_bins` -> 0
    it degenerates back to `gt_traj_ce`.

    NOTE ON THE LOGGED VALUE: this term has a non-zero floor, unlike the one-hot
    CE. A student matching the target exactly still pays the target's own entropy,
    ~= ln(sigma * sqrt(2*pi*e)) = 3.21 nats at sigma_bins=6. Do not read `gt_soft`
    plateauing near ~3.2 as a failure to learn, and do not compare it to job 243's
    ~6.0 - compare the `gt` diagnostic for that.

    `lo`/`hi` bound the student's future region (`future_base`,
    `future_base + n_future_bins`). Offsets landing outside are dropped and the
    target is renormalised over what remains, so waypoints near a region edge
    stay proper distributions instead of quietly leaking mass onto the history
    bins or the special tokens that sit immediately above them.
    """
    half = max(1, int(round(3.0 * sigma_bins)))                  # +/-3 sigma
    off = torch.arange(-half, half + 1, device=student_logits.device)   # (W,)
    bins = gt_token_ids.unsqueeze(-1) + off                      # (B, T, W)
    in_range = (bins >= lo) & (bins < hi)
    q = torch.exp(-0.5 * (off.float() / sigma_bins) ** 2)        # (W,)
    q = q.expand_as(bins) * in_range
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = F.log_softmax(student_logits, dim=-1)                 # full-vocab denominator
    logp_w = torch.gather(logp, -1, bins.clamp(lo, hi - 1))      # (B, T, W)
    ce = -(q * logp_w).sum(-1)                                   # (B, T)
    if per_pos:
        return ce * mask          # (B, T), unreduced - for the dependence probes
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
    """Time-varying loss weights: feature warmup + GT-CE anneal (stage 1) and
    flow->GT annealing (stage 2), from the config fractions."""
    w = dict(cfg_stage.loss_weights.raw)
    frac = step / max(total_steps, 1)
    if "feat" in w:
        warm = cfg_stage.get("feat_warmup_frac", 0.0) or 0.0
        if warm > 0:
            w["feat"] *= min(frac / warm, 1.0)
    if "gt_ce" in w:
        # Linear ramp from the configured `loss_weights.gt_ce` to `gt_ce_end`
        # across the run; early stop can truncate it. The ramp runs in whichever
        # direction the two endpoints imply.
        #
        # Runs 2-3 ramped it DOWN (the old key name, `gt_ce_min`, still reads):
        # raw `gt_ce` sat flat while `traj_kl` fell, so a fixed weight drifted to
        # ~80% of the loss and smothered the teacher-distribution signal.
        #
        # Run 4 ramps it UP. The cache measurement behind that (see
        # `gt_traj_soft_ce`) says the flatness was the one-hot target's shape,
        # not a genuinely dominant term - and separately, that the teacher's
        # own accel distribution is close to uninformative about GT (only 17% of
        # its mass within +/-20 bins). So the GT anchor should END as the primary
        # trajectory signal, with the teacher leading early while the student is
        # still learning the token geometry at all.
        gt_end = cfg_stage.get("gt_ce_end", cfg_stage.get("gt_ce_min", None))
        if gt_end is not None:
            w["gt_ce"] += (float(gt_end) - w["gt_ce"]) * min(max(frac, 0.0), 1.0)
    if "flow_distill" in w:
        anneal = cfg_stage.get("anneal_to_gt_frac", 0.0) or 0.0
        if anneal > 0 and frac > 1 - anneal:
            a = (frac - (1 - anneal)) / anneal
            w["flow_distill"] *= (1 - a)
            w["fm_gt"] *= (1 + a)
    return w

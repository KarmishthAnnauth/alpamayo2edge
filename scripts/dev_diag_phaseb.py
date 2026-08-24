"""Throwaway diagnostic for the D-022 probe's BROKEN verdict. Delete when done.

The probe reported argmax_match 0.906-0.953 and a catastrophic discrete ADE
(20-128 m) alongside region_mass 0.987 and a 0.01-0.34 m quantization floor.
Those four numbers are mutually inconsistent under the probe's own reading of
them, so this asks three questions in one GPU pass:

  A. WHY is argmax_match < 1? A true off-by-one between `out_b.logits` and
     `out_b.sequences` would give a match near ZERO, not 0.93. A 5-9% scattered
     mismatch is the signature of an extra logits processor perturbing the
     scores the token was drawn from - `gen_cfg` is deep-copied from
     `model.vlm.generation_config`, so whatever that carries (repetition_penalty,
     no_repeat_ngram_size, ...) is silently in force. If that is it, the cache is
     correctly ALIGNED and only the probe's assertion is wrong.

  B. If the emitted token is not the raw argmax, WHERE does it sit? Rank in the
     top-k and its log-prob gap. Cheap, and it distinguishes "a near-tie flipped"
     from "the score was actively penalised".

  C. Is the 128-token stream laid out the way the detokenizer expects? The floor
     (GT encoded then decoded) is near-perfect, which proves the codec and the
     coordinate frame are fine - so a huge ADE on the model's own tokens points
     at the STREAM, not the decoder. 128 = 64 waypoints x 2 dims interleaved; if
     the model emits dim-major where `decode` reads waypoint-major, the floor
     still passes and the discrete decode still explodes. Tested by decoding both
     streams both ways.

    python scripts/dev_diag_phaseb.py            # first clip of the raw index
    python scripts/dev_diag_phaseb.py --clips 3
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from distill.config import load_config                       # noqa: E402
from distill.data.preprocess import iter_windows, list_clip_ids  # noqa: E402
from distill.eval.open_loop import ade                       # noqa: E402
from distill.teacher.wrapper import TeacherWrapper           # noqa: E402

# Every generation_config field that makes transformers build a logits processor.
PROCESSOR_FIELDS = [
    "repetition_penalty", "encoder_repetition_penalty", "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size", "bad_words_ids", "min_length",
    "min_new_tokens", "forced_bos_token_id", "forced_eos_token_id",
    "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids",
    "sequence_bias", "guidance_scale", "diversity_penalty", "num_beam_groups",
    "renormalize_logits", "epsilon_cutoff", "eta_cutoff", "typical_p",
    "exponential_decay_length_penalty", "top_k", "top_p", "temperature",
    "do_sample", "num_beams",
]


def hist_poses(window):
    d = window.data
    return (d["ego_history_xyz"][:, -1].float().cpu(),
            d["ego_history_rot"][:, -1].float().cpu())


def tokens_to_xyz(teacher, tokens, window):
    hx, hr = hist_poses(window)
    fut_xyz, _, _ = teacher.future_traj_tokenizer.decode(
        hx, hr, tokens.reshape(1, -1).long().cpu())
    return fut_xyz[0]


def swap_dims(tokens: torch.Tensor) -> torch.Tensor:
    """Reinterpret an interleaved (waypoint-major) stream as dim-major, i.e. what
    you get if the two action dims were written in the other order."""
    return tokens.reshape(-1, 2).flip(-1).reshape(-1)


def transpose_dims(tokens: torch.Tensor) -> torch.Tensor:
    """Reinterpret [d0 d0 ... d1 d1 ...] (dim-blocked) as interleaved."""
    t = tokens.reshape(2, -1).t().reshape(-1)
    return t


def report_gen_config(teacher) -> None:
    gc = teacher.model.vlm.generation_config
    print("=" * 72)
    print("A. generation_config carried into Phase B (deep-copied from the VLM)")
    print("=" * 72)
    for f in PROCESSOR_FIELDS:
        v = getattr(gc, f, "<absent>")
        default = v in (None, 0, 1.0, False, "<absent>", [])
        flag = "" if default else "   <-- ACTIVE, builds a processor"
        print(f"    {f:38s} {str(v):>12s}{flag}")
    print()


def diag_window(teacher, cfg, window) -> None:
    tc = cfg.teacher
    out = teacher.label_window(
        window, k_flow=tc.flow_targets_per_window, topk=tc.topk_logits,
        max_coc=tc.max_coc_tokens, n_traj_samples=tc.n_traj_samples,
        greedy_traj=True)

    tok = out.traj_token_ids.long().cpu()          # (128,) region-relative
    gt_tok = out.gt_traj_token_ids.long().cpu()    # (128,)
    idx = out.traj_topk_idx.long().cpu()           # (128, 32)
    logp = out.traj_topk_logp.float().cpu()        # (128, 32)
    T = tok.numel()

    print("=" * 72)
    print(f"B. argmax mismatch anatomy  ({window.clip_id})")
    print("=" * 72)
    top1 = idx[:, 0]
    mism = (tok != top1).nonzero().flatten()
    print(f"    mismatched positions: {mism.numel()}/{T} "
          f"(argmax_match {1 - mism.numel() / T:.3f})")
    if mism.numel():
        print(f"    positions: {mism.tolist()[:40]}")
        contiguous = bool(mism.numel() > 1 and (mism.diff() == 1).all())
        print(f"    contiguous run? {contiguous}   "
              f"(a true off-by-one would be ALL positions, not a scatter)")

        ranks, gaps, seen_before = [], [], 0
        for p in mism.tolist():
            hit = (idx[p] == tok[p]).nonzero().flatten()
            r = int(hit[0]) if hit.numel() else -1
            ranks.append(r)
            gaps.append(float(logp[p, 0] - (logp[p, r] if r >= 0 else float("nan"))))
            # Repetition-penalty signature: the RAW argmax we did not emit is a
            # token that already appeared earlier in the stream.
            if (tok[:p] == top1[p]).any():
                seen_before += 1
        ranks_a = np.array(ranks)
        print(f"    emitted token's rank in top-k: "
              f"{[int(r) for r in ranks_a[:20]]}{' ...' if len(ranks) > 20 else ''}")
        print(f"      not in top-32 at all: {(ranks_a < 0).sum()}/{len(ranks)}")
        print(f"      median logp gap (top1 - emitted): {np.nanmedian(gaps):+.4f}")
        print(f"    raw-argmax token already emitted earlier: "
              f"{seen_before}/{mism.numel()}"
              f"   <-- high = repetition_penalty / no_repeat_ngram")
    print()

    print("=" * 72)
    print("C. stream layout: emitted vs GT-encoded, and decode both ways")
    print("=" * 72)
    for name, t in (("emitted", tok), ("gt_enc", gt_tok)):
        pd = t.reshape(-1, 2).numpy()
        print(f"    {name:8s} first 16: {t[:16].tolist()}")
        for d in range(2):
            col = pd[:, d]
            print(f"      dim{d}: unique {len(np.unique(col)):3d}  "
                  f"range [{col.min():4d}, {col.max():4d}]  "
                  f"mean {col.mean():7.1f}  std {col.std():6.1f}")

    gt_xyz = out.gt_future_xyz.float().cpu()
    variants = {
        "emitted   as-is":      tok,
        "emitted   dim-swap":   swap_dims(tok),
        "emitted   transposed": transpose_dims(tok),
        "gt_enc    as-is":      gt_tok,          # the floor - must be ~0
        "gt_enc    dim-swap":   swap_dims(gt_tok),
        "gt_enc    transposed": transpose_dims(gt_tok),
    }
    print()
    for name, t in variants.items():
        try:
            xyz = tokens_to_xyz(teacher, t, window)
            a = float(ade(xyz[None], gt_xyz[None])[0])
            print(f"    ADE  {name:22s} {a:9.2f} m")
        except Exception as e:  # a layout the decoder rejects outright is a result too
            print(f"    ADE  {name:22s}    FAILED: {type(e).__name__}: {e}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=1)
    a = ap.parse_args()

    cfg = load_config()
    clips = list_clip_ids(cfg)[:a.clips]
    teacher = TeacherWrapper(cfg)
    report_gen_config(teacher)
    for clip_id in clips:
        for w_idx, window in iter_windows(cfg, clip_id):
            diag_window(teacher, cfg, window)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

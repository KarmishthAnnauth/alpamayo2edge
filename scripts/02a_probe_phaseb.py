"""D-022 probe: does Alpamayo 1.5 actually emit usable discrete trajectory tokens?

THE OPEN RISK. A1.5's release strips the future-token fusion path
(`TrajectoryFusionMixin.fuse_traj_tokens` fuses history only), so nothing in the
shipped code proves the 3000-bin future region is a *trained* target rather than
3000 vocabulary rows that were allocated and never supervised. `label_window`
Phase B will happily emit 128 bin ids either way, and nothing downstream checks.
Run this on 2-3 clips BEFORE spending GPU-days on `02_label.py`.

It calls the real `label_window` (with `greedy_traj=True`), so it validates the
exact code path the labeling run uses. It writes NOTHING into `cache_root` - no
shards, no manifest - so it cannot pollute a later labeling run.

    python scripts/02a_probe_phaseb.py --clips 3
    python scripts/02a_probe_phaseb.py --clips 3 --sampled   # also score the
                                                             # stochastic decode
                                                             # the real run uses

Four independent lines of evidence, weakest to strongest:

  region_mass  The probability mass the teacher puts on its top-32 future bins,
               out of the WHOLE vocabulary. `traj_topk_logp` are full-softmax
               log-probs, deliberately not renormalized, so this is directly
               readable. A trained head concentrates here (~0.9). An untrained
               one leaves the mass on text tokens (~1e-3) and the restricted
               decode is then just sampling from noise. THE DECISIVE NUMBER.
  ADE          Detokenized tokens vs GT, against two references: the expert's
               own Euler rollout (the trusted released path) and the
               quantization floor (the cached GT bins decoded back - the best
               any token sequence could do, and the target stage-1's gt_ce
               anchor trains on). Absolute ADE means little; the ratio to those
               two means everything.
  token health Distinct bins per action dim, longest repeated run, fraction
               clamped to a region edge. Catches the failure ADE can miss: a
               collapsed curvature dim decodes to a smooth, plausible-looking,
               entirely straight line.
  argmax match In greedy mode the emitted token MUST be a maximiser of the
               captured region log-probs. This tests our own plumbing, not the
               teacher - an off-by-one between `out_b.logits` steps and the
               `out_b.sequences` slice would silently misalign every KD target in
               the cache. Failing here is a bug in us and invalidates the rest of
               the readout. Ties count: the logits are bf16, adjacent bins tie
               exactly, and `topk` and `argmax` break those ties differently
               (D-031) - which says nothing about alignment.

Exit codes, so this can gate a shell chain: 0 PASS, 1 MARGINAL, 3 FAIL,
4 BROKEN (our plumbing). MARGINAL is deliberately non-zero - it means decide
deliberately, not proceed by default.

Needs the same environment as the labeling run: `pip install -e ../alpamayo1.5
-e ../physical_ai_av`, HF auth, ~30GB of GPU memory.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from distill.config import load_config                       # noqa: E402
from distill.data.preprocess import iter_windows, list_clip_ids  # noqa: E402
from distill.eval.open_loop import ade                       # noqa: E402
from distill.teacher.wrapper import TeacherWrapper           # noqa: E402

# Thresholds. Deliberately generous: this is a go/no-go gate on a binary
# question ("is the future region supervised at all?"), not a quality metric.
MASS_PASS, MASS_MARGINAL = 0.5, 0.05
ADE_PASS_RATIO, ADE_MARGINAL_RATIO = 2.0, 4.0
ADE_PASS_SLACK, ADE_MARGINAL_SLACK = 1.0, 3.0    # metres, for small expert ADE


def hist_poses(window) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, T, 3) / (1, T, 3, 3) history, the shapes both action-space entry
    points want. The loader hands back (1, 1, T, ...); `[:, -1]` drops the
    n_traj axis exactly as the released rollout does."""
    d = window.data
    return (d["ego_history_xyz"][:, -1].float().cpu(),
            d["ego_history_rot"][:, -1].float().cpu())


def tokens_to_xyz(teacher, tokens: torch.Tensor, window) -> torch.Tensor:
    """Region-relative bin ids -> (H, 3) waypoints. Delegates to the wrapper so
    the emission-order dim swap (D-031) is applied in exactly one place."""
    return teacher.detokenize_traj(tokens, window)


def actions_to_xyz(teacher, actions: torch.Tensor, window) -> torch.Tensor:
    """Normalized actions (B, H, A) -> (B, H, 3) waypoints, through the MODEL's
    action space (the one that produced them - the tokenizer carries its own
    instance and there is no guarantee the normalization constants agree).

    Runs on the model's device: `action_to_traj` pulls its normalization stats
    to the action's device, but `estimate_t0_states` is not audited for that, so
    keeping everything where the model lives is the safe choice.
    """
    dev = teacher.device
    hx, hr = hist_poses(window)
    b = actions.shape[0]
    xyz, _ = teacher.model.action_space.action_to_traj(
        actions.float().to(dev), hx.expand(b, -1, -1).to(dev),
        hr.expand(b, -1, -1, -1).to(dev))
    return xyz.float().cpu()


def token_health(tokens: torch.Tensor, n_bins: int, dims: int = 2) -> dict:
    """Degeneracy checks. Tokens interleave the action dims (128 = 64 x 2), so
    per-dim statistics are the informative ones: a constant curvature dim is a
    straight line, which ADE alone can score as merely mediocre."""
    t = tokens.long().cpu().numpy()
    per_dim = t.reshape(-1, dims)

    def longest_run(x):
        return int(np.diff(np.flatnonzero(np.concatenate(
            ([True], x[1:] != x[:-1], [True])))).max())

    # Per dim, then max. On the interleaved stream a run length is almost always
    # 1 even when a dim is pinned to a constant, which would make this check
    # silently useless - the exact failure it exists to catch.
    return {
        "n_unique": int(np.unique(t).size),
        "n_unique_per_dim": [int(np.unique(per_dim[:, d]).size) for d in range(dims)],
        "max_run": max(longest_run(per_dim[:, d]) for d in range(dims)),
        "edge_frac": float(((t == 0) | (t == n_bins - 1)).mean()),
        "bin_mean": float(t.mean()),
        "bin_std": float(t.std()),
    }


def tie_aware_match(out) -> float:
    """Fraction of steps where the emitted token is a maximiser of the captured
    region log-probs. Still catches the real failure this check exists for - an
    off-by-one between the logits steps and the sequence slice puts the emitted
    token at an arbitrary rank, usually outside the top-k entirely - while not
    firing on bf16 ties, which carry no information about alignment."""
    tok = out.traj_token_ids.long().cpu()
    idx = out.traj_topk_idx.long().cpu()
    logp = out.traj_topk_logp.float().cpu()
    hit = (idx == tok[:, None])                       # (T, K) where the emitted id sits
    found = hit.any(-1)
    emitted_logp = torch.where(found, (logp * hit).sum(-1), torch.tensor(float("-inf")))
    return float((found & (emitted_logp >= logp[:, 0])).float().mean())


def probe_window(teacher, cfg, window, sampled: bool) -> dict:
    tc = cfg.teacher
    out = teacher.label_window(
        window, k_flow=tc.flow_targets_per_window, topk=tc.topk_logits,
        max_coc=tc.max_coc_tokens, n_traj_samples=tc.n_traj_samples,
        greedy_traj=True)

    gt = out.gt_future_xyz.float().cpu()                      # (H, 3)
    disc = tokens_to_xyz(teacher, out.traj_token_ids, window)  # (H, 3)
    expert = actions_to_xyz(teacher, out.traj_samples, window)  # (n_s, H, 3)
    # Quantization floor: the cache's own GT bins, decoded straight back. The ADE
    # no token sequence can beat, and a frame check besides - a decode living in a
    # different frame than `ego_future_xyz` shows up here as a large number rather
    # than as a Phase-B failure. Reading the CACHED field rather than recomputing
    # it means this also validates what stage-1's gt_ce anchor will train on.
    floor = tokens_to_xyz(teacher, out.gt_traj_token_ids, window)

    logp = out.traj_topk_logp.float()
    rec = {
        "clip_id": window.clip_id,
        "t0_us": int(window.t0_us),
        "ade_discrete": float(ade(disc[None], gt[None])[0]),
        # Mean over samples, not min: with n_traj_samples this small, min would
        # flatter the expert and understate the gap the discrete head has to close.
        "ade_expert": float(ade(expert, gt.expand_as(expert)).mean()),
        "ade_floor": float(ade(floor[None], gt[None])[0]),
        "ade_discrete_vs_expert": float(ade(disc[None].expand_as(expert), expert).mean()),
        "region_mass": float(logp.exp().sum(-1).mean()),
        "top1_prob": float(logp[:, 0].exp().mean()),
        # Ties count as matches. In greedy mode the emitted token must be *a*
        # maximiser of the region logits, not literally `topk_idx[:, 0]`: the
        # logits are bf16, exact ties between adjacent bins are common, and
        # `torch.topk` and generate's `argmax` break them differently. Measured
        # on three clips: every mismatch sat at rank 1 with a top1-minus-emitted
        # log-prob gap of exactly 0.0000. Comparing ids alone reported 0.906-0.953
        # and cried BROKEN over an alignment that was never wrong (D-031).
        "argmax_match": float(tie_aware_match(out)),
        "coc_text": out.coc_text[:200],
        **token_health(out.traj_token_ids, teacher.n_future_bins),
        # The same statistics on the GT stream, as the reference the degeneracy
        # check needs. Both are in emission order (D-031), so they are directly
        # comparable dim for dim.
        **{f"gt_{k}": v for k, v in
           token_health(out.gt_traj_token_ids, teacher.n_future_bins).items()},
    }
    rec["_xyz"] = {"gt": gt.numpy(), "discrete": disc.numpy(),
                   "floor": floor.numpy(), "expert": expert.numpy()}

    if sampled:
        # What the labeling run will actually cache. Reported separately so a
        # temperature that quietly wrecks the trajectory is visible rather than
        # averaged into the greedy number.
        s = teacher.label_window(
            window, k_flow=1, topk=tc.topk_logits, max_coc=tc.max_coc_tokens,
            n_traj_samples=1, greedy_traj=False)
        rec["ade_sampled"] = float(ade(
            tokens_to_xyz(teacher, s.traj_token_ids, window)[None], gt[None])[0])
        # Token health on the SAMPLED stream too. Without this the degeneracy
        # check describes the greedy decode, which is the one thing in this
        # script that never reaches the cache.
        rec.update({f"sampled_{k}": v for k, v in
                    token_health(s.traj_token_ids, teacher.n_future_bins).items()})
    return rec


def waypoint_table(xyz: dict, idx=(9, 19, 31, 63)) -> list[str]:
    """x/y at ~1s, 2s, 3.2s and the 6.4s horizon. Cheap, dependency-free, and
    the fastest way to see a trajectory that stands still or drives backwards."""
    lines = [f"    {'':10s}" + "".join(f"{(i + 1) / 10:>8.1f}s" for i in idx)]
    # expert[0], not the mean: averaging multimodal samples can produce a
    # trajectory no sample resembles.
    for name, arr in (("gt", xyz["gt"]), ("discrete", xyz["discrete"]),
                      ("expert[0]", xyz["expert"][0]), ("floor", xyz["floor"])):
        xs = "".join(f"{arr[i, 0]:>8.1f}" for i in idx)
        ys = "".join(f"{arr[i, 1]:>8.1f}" for i in idx)
        lines += [f"    {name:10s}{xs}   (x)", f"    {'':10s}{ys}   (y)"]
    return lines


def is_degenerate(r: dict, prefix: str = "") -> bool:
    """Is an action dim collapsed *beyond what the road itself is doing*?

    `prefix` selects which decode's stream to judge: "" for the greedy one,
    "sampled_" for the one the labeling run actually caches.

    The absolute form of this check ("<=2 distinct bins, or a run of >=16") is
    unreadable on the very case it was written for. On a straight road the GT
    curvature is genuinely near-constant, so a correct teacher MUST emit a
    near-constant curvature dim - the first probe run flagged 4 distinct bins and
    a run of 42 on a clip whose GT drifts 12.9 m laterally over 227 m, which is a
    straight road being described accurately. With no reference the check cannot
    tell that from a dead head, so it fires on healthy windows and blocks PASS.

    Referenced against the GT stream's own per-dim statistics it means what it was
    meant to mean: the teacher flattened a dim that GT does not have flat.

    `max_run` is printed but is NOT a trigger. It compares an absolute run length
    between a smooth model output and a noisy measured GT, so any model smoother
    than its own target scores worse on it - it penalises the behaviour we want.
    The window that proved it emitted MORE distinct bins per dim than GT (18 vs
    15, 48 vs 41) and scored the best ADE of the run (0.62 m, against the
    expert's 2.99 m), and the clause called it degenerate. Distinct bins per dim
    is the instrument; run length is context for a human reading the table.
    """
    for n, gn in zip(r[f"{prefix}n_unique_per_dim"], r["gt_n_unique_per_dim"]):
        # `n <= 2` alone false-positives on a stationary vehicle, where GT is
        # every bit as flat (one probe window read emitted [1, 10] against GT
        # [1, 3] and was called degenerate). Require GT to actually have
        # variation before calling a flat dim a collapse.
        if (n <= 2 and gn > 2) or n * 4 <= gn:
            return True
    return False


def verdict(rows: list[dict]) -> tuple[str, list[str]]:
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    mass, a_exp = med("region_mass"), med("ade_expert")
    match = min(r["argmax_match"] for r in rows)
    notes = []

    # Judge the decode that reaches the cache. `02_label.py` samples (top_p 0.98,
    # temperature 0.6); the greedy stream exists only so a human can read the
    # waypoint table without stochastic noise, and it is systematically worse -
    # measured 2.77 m greedy vs 1.22 m sampled over ten windows, because greedy
    # decoding of a head whose top-1 sits near 0.5 latches onto one bin and
    # repeats it. Gating on greedy fails a cache that is fine.
    scored_sampled = "ade_sampled" in rows[0]
    prefix = "sampled_" if scored_sampled else ""
    a_disc = med("ade_sampled" if scored_sampled else "ade_discrete")
    which = "sampled" if scored_sampled else "greedy"
    if not scored_sampled:
        notes.append(
            "scored on the GREEDY decode, which is not what 02_label.py caches. "
            "Re-run with --sampled before treating any ADE or degeneracy reading "
            "here as a verdict on the cache.")

    if match < 1.0:
        return "BROKEN", [
            f"argmax_match {match:.3f} < 1.0 in greedy mode. This is OUR bug, not the "
            "teacher's: the emitted token is not even a maximiser of the captured region "
            "log-probs (ties are already allowed for), so every top-k KD target in the "
            "cache would be misaligned with its token. Fix the `out_b.logits` / "
            "`out_b.sequences` indexing in `label_window` before reading anything else "
            "here. Sanity check first that the decode really was greedy."]

    ade_ok = a_disc <= max(ADE_PASS_RATIO * a_exp, a_exp + ADE_PASS_SLACK)
    ade_marginal = a_disc <= max(ADE_MARGINAL_RATIO * a_exp, a_exp + ADE_MARGINAL_SLACK)
    degenerate = [r for r in rows if is_degenerate(r, prefix)]
    if degenerate:
        notes.append(
            f"{len(degenerate)}/{len(rows)} windows have an action dim that is "
            f"collapsed RELATIVE TO GT in the {which} stream - a constant "
            "accel/curvature decodes to a smooth line that ADE can score as merely "
            "mediocre. Treat a good ADE here as coincidence.")
    # A broken floor does not un-train the teacher's head - region_mass is
    # measured on raw logits and stands on its own - but it does mean every ADE
    # here is measured through a suspect detokenizer, and that same detokenizer
    # is what the stage-1 coarse-minADE gate runs on. So: no PASS on it.
    floor_bad = med("ade_floor") > 1.0
    if floor_bad:
        notes.append(
            f"quantization floor is {med('ade_floor'):.2f} m, which is high for a pure "
            "encode/decode round trip - suspect a frame or normalization mismatch in the "
            "detokenization path. Fix that first: every ADE above is measured through it, "
            "and so is the stage-1 coarse-minADE gate.")

    if mass >= MASS_PASS and ade_ok and not degenerate and not floor_bad:
        return "PASS", notes + [
            "The future region is supervised and decodes to sane trajectories. D-022 "
            "closes: proceed to `02_label.py`. Record the numbers as D-031."]
    if mass >= MASS_PASS and floor_bad:
        return "MARGINAL", notes + [
            f"region_mass {mass:.3f} says the future region IS supervised, which is the "
            "half of D-022 that gates the plan. Re-run once the detokenizer is fixed to "
            "confirm the trajectories themselves."]
    if mass < MASS_MARGINAL and not ade_ok:
        return "FAIL", notes + [
            f"region_mass {mass:.2e} - the teacher puts essentially no probability on the "
            "future bins, and the trajectories are bad. The restricted decode is sampling "
            "from an unsupervised region. Take D-022's fallback: stage 1 drops the "
            "trajectory-token target and reduces to sequence-level CoC KD plus a "
            "continuous trajectory target, with the coarse-minADE gate running off the "
            "expert instead of detokenized tokens."]
    return "MARGINAL", notes + [
        f"region_mass {mass:.3f}, {which} ADE {a_disc:.2f} m vs expert {a_exp:.2f} m. "
        "Ambiguous. Widen to ~10 clips, plot the trajectories from the saved npz, and "
        "decide deliberately - this gates GPU-weeks, so it is worth the extra hour."]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--clips-file", type=str, default=None,
                    help="JSON list of clip ids, e.g. cache_root/curated_500.json. "
                         "Defaults to the first --clips of the raw index, which is "
                         "deterministic and needs no curation run.")
    ap.add_argument("--windows-per-clip", type=int, default=1,
                    help="1 is enough for a go/no-go; raise to widen the sample.")
    ap.add_argument("--sampled", action="store_true",
                    help="also score the stochastic decode the real labeling run uses")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default: <runs_root>/phaseb_probe)")
    a = ap.parse_args()

    cfg = load_config()
    out_dir = Path(a.out or Path(cfg.paths.runs_root) / "phaseb_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.clips_file:
        clips = json.load(open(a.clips_file))[:a.clips]
    else:
        clips = list_clip_ids(cfg)[:a.clips]
    print(f"probing {len(clips)} clips x {a.windows_per_clip} window(s)\n")

    teacher = TeacherWrapper(cfg)
    rows = []
    for clip_id in clips:
        for w_idx, window in iter_windows(cfg, clip_id):
            if w_idx >= a.windows_per_clip:
                break
            r = probe_window(teacher, cfg, window, a.sampled)
            xyz = r.pop("_xyz")
            np.savez(out_dir / f"{clip_id}_{w_idx:02d}.npz", **xyz)
            rows.append(r)

            print(f"{clip_id} [w{w_idx}]")
            print(f"    ADE  discrete {r['ade_discrete']:6.2f} m | expert "
                  f"{r['ade_expert']:6.2f} m | floor {r['ade_floor']:5.2f} m | "
                  f"discrete-vs-expert {r['ade_discrete_vs_expert']:6.2f} m"
                  + (f" | sampled {r['ade_sampled']:6.2f} m" if a.sampled else ""))
            print(f"    mass region {r['region_mass']:.4f} | top1 {r['top1_prob']:.4f} "
                  f"| argmax_match {r['argmax_match']:.3f}")
            print(f"    bins unique {r['n_unique']}/128 (per dim "
                  f"{r['n_unique_per_dim']}) | max_run {r['max_run']} | "
                  f"edge {r['edge_frac']:.3f}")
            if a.sampled:
                print(f"      sampled    {r['sampled_n_unique']}/128 (per dim "
                      f"{r['sampled_n_unique_per_dim']}) | max_run "
                      f"{r['sampled_max_run']}        <- THE CACHED STREAM")
            print(f"      vs GT      {r['gt_n_unique']}/128 (per dim "
                  f"{r['gt_n_unique_per_dim']}) | max_run {r['gt_max_run']}"
                  "        <- the reference; a straight road is flat in both")
            print("\n".join(waypoint_table(xyz)))
            print(f"    CoC: {r['coc_text'][:110]!r}\n")

    if not rows:
        print("no windows probed")
        return 2

    v, notes = verdict(rows)
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    print("=" * 72)
    print(f"D-022 PROBE: {v}   ({len(rows)} windows)")
    print(f"  median region_mass  {med('region_mass'):.4f}   "
          f"(pass >= {MASS_PASS}, fail < {MASS_MARGINAL})")
    print(f"  median ADE  discrete {med('ade_discrete'):6.2f} m | "
          f"expert {med('ade_expert'):6.2f} m | floor {med('ade_floor'):5.2f} m")
    if a.sampled:
        print(f"  median ADE  sampled  {med('ade_sampled'):6.2f} m  "
              "(what 02_label.py caches)")
    print("=" * 72)
    for n in notes:
        print(f"  - {n}")

    (out_dir / "probe.json").write_text(json.dumps(
        {"verdict": v, "notes": notes, "windows": rows}, indent=2))
    print(f"\nwrote {out_dir}/probe.json + one npz per window")
    return {"PASS": 0, "MARGINAL": 1, "FAIL": 3, "BROKEN": 4}[v]


if __name__ == "__main__":
    raise SystemExit(main())

"""Teacher reference lines on the stage-1 gate's own 60 windows.

`coarse_minade` reports the student's minADE_6 against GT, but nothing said what
the TEACHER scores on those same windows - so "2.937 m" had no reference line and
could not be read as near-ceiling or far from it (eval_phase1.md §1 quoted D-031's
1.79 m, which is 10 windows, single-sample, and therefore not comparable).

Three lines, all from the label cache - no GPU, no teacher weights, no streaming:

  codec floor   `gt_traj_token_ids` -> waypoints. GT pushed through the teacher's
                own tokenizer and back. Validates the detokenization path end to
                end; D-031 measured ~0.01 m, so a large number here means the
                harness is wrong, not the teacher.
  token path    `traj_token_ids` -> waypoints. What stage 1 distils, and the
                ceiling its gate is measured against. ONE sample per window is
                cached, so this is ADE_1. Since minADE_k <= ADE_1 for any k, it
                is an UPPER BOUND on the teacher's minADE_6, not a match - quote
                it as a bound.
  action expert `traj_samples`, the continuous rollout, k=`teacher.n_traj_samples`.
                The head the published Alpamayo numbers come from, and the one the
                expert/flow path in stage 2 actually inherits from.

Plus two HYBRIDS that split the blame between the two interleaved action dims.
D-035 measured the teacher's accel mass at 2.8% within +/-2 bins of the GT bin
against curvature's 42%, but that is a distribution statistic - these two convert
it into metres. The stream is 64 waypoints x 2 dims, EVEN = curvature, ODD =
accel (D-031 emission order, the same convention `traj_kl` / `traj_kl_accel`
split on). Substituting one dim from `gt_traj_token_ids` isolates the other:

  teacher curvature + GT accel  -> what the path shape alone is worth
  GT curvature + teacher accel  -> what the speed profile alone is worth

Window set and ordering are taken from `discover_shards` over the challenging
split, then the same deterministic `range(gate_max_windows)` prefix the gate
takes - so these numbers line up window-for-window with the epoch gate.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from distill.data import frames
from distill.data.dataset import discover_shards
from distill.eval.coarse_minade import _split_clips
from distill.teacher.wrapper import TeacherWrapper, swap_action_dims

log = logging.getLogger(__name__)


def ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean L2 over the horizon, x/y only - open_loop.ade for a single mode."""
    return float(torch.linalg.norm(pred[..., :2] - gt[..., :2], dim=-1).mean())


def min_ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """pred (K, H, 3+) modes against one gt (H, 3+) - open_loop.min_ade, B=1."""
    d = torch.linalg.norm(pred[..., :2] - gt[None, ..., :2], dim=-1)   # (K, H)
    return float(d.mean(-1).min())


def main(config: str, split: str, max_windows: int | None, out: str | None) -> None:
    cfg = OmegaConf.load(config)
    cache_root = Path(cfg.paths.cache_root)

    # Config-only: rebuilds the future tokenizer from the checkpoint config with
    # no weights, which is all detokenization needs (wrapper.py:141).
    tw = TeacherWrapper(cfg, load_model=False)
    tok = tw.future_traj_tokenizer
    space = tok.action_space

    shards = discover_shards(cache_root, _split_clips(cfg, split))
    if max_windows is not None:
        shards = shards[:max_windows]          # the gate's own deterministic prefix
    if not shards:
        raise RuntimeError(f"no cached shards for split {split!r}")

    rows = {"floor": [], "token": [], "expert": [],
            "curv_teacher": [], "accel_teacher": []}
    skipped = 0
    for path in shards:
        clip_id, w_idx = path.parent.name, int(path.stem)
        cached = frames.input_path(cache_root, clip_id, w_idx)
        if not cached.exists():
            # Streaming the clip back would make a "free" measurement cost hours.
            skipped += 1
            continue
        d = np.load(path, allow_pickle=True)
        wd = frames.load_window_input(cached, clip_id).data
        hx = torch.as_tensor(wd["ego_history_xyz"])[:, -1].float().cpu()   # (1, T, 3)
        hr = torch.as_tensor(wd["ego_history_rot"])[:, -1].float().cpu()   # (1, T, 3, 3)
        gt = torch.as_tensor(d["gt_future_xyz"], dtype=torch.float32)      # (H, 3)

        # Both token fields are stored in the teacher's EMISSION order, so both
        # need D-031's swap before `decode`, whose convention is `encode`'s.
        for key, name in (("gt_traj_token_ids", "floor"), ("traj_token_ids", "token")):
            toks = swap_action_dims(torch.as_tensor(d[key]).reshape(1, -1).long().cpu())
            fut, _, _ = tok.decode(hx, hr, toks)
            rows[name].append(ade(fut[0], gt))

        # Hybrids: interleaved (curvature, accel) per waypoint in emission order,
        # so ::2 is curvature and 1::2 is accel. Build each from one source and
        # the other dim from GT, then decode through the same path as above.
        t_tok = torch.as_tensor(d["traj_token_ids"]).reshape(-1).long()
        g_tok = torch.as_tensor(d["gt_traj_token_ids"]).reshape(-1).long()
        for name, src_even in (("curv_teacher", True), ("accel_teacher", False)):
            mix = g_tok.clone() if src_even else t_tok.clone()
            if src_even:
                mix[::2] = t_tok[::2]        # teacher curvature, GT accel
            else:
                mix[::2] = g_tok[::2]        # GT curvature, teacher accel
            fut, _, _ = tok.decode(hx, hr, swap_action_dims(mix.reshape(1, -1).cpu()))
            rows[name].append(ade(fut[0], gt))

        # Expert rollout: action space -> positions, same as eval/gap.py:88.
        samples = torch.as_tensor(d["traj_samples"], dtype=torch.float32)  # (K, H, A)
        k = samples.shape[0]
        xyz, _ = space.action_to_traj(samples,
                                      hx[0].unsqueeze(0).expand(k, -1, -1),
                                      hr[0].unsqueeze(0).expand(k, -1, -1, -1))
        rows["expert"].append(min_ade(xyz.float(), gt))

    n = len(rows["token"])
    if not n:
        raise RuntimeError("every window was skipped - no cached student inputs")
    k_expert = int(cfg.teacher.n_traj_samples)

    def stat(v: list[float]) -> dict:
        t = torch.tensor(v)
        return {"mean": float(t.mean()), "p90": float(t.quantile(0.9)), "n": len(v)}

    res = {
        "split": split, "n_windows": n, "skipped_uncached": skipped,
        "codec_floor_ade": stat(rows["floor"]),
        "teacher_token_path_ade1": stat(rows["token"]),
        f"teacher_expert_minade{k_expert}": stat(rows["expert"]),
        "hybrid_teacher_curv_gt_accel_ade1": stat(rows["curv_teacher"]),
        "hybrid_gt_curv_teacher_accel_ade1": stat(rows["accel_teacher"]),
    }
    log.info("windows: %d (%d skipped, no cached input)", n, skipped)
    log.info("codec floor            ADE_1     %.3f m (p90 %.3f)",
             res["codec_floor_ade"]["mean"], res["codec_floor_ade"]["p90"])
    log.info("teacher TOKEN PATH     ADE_1     %.3f m (p90 %.3f)   <- stage-1 ceiling "
             "(upper bound on its minADE_6)",
             res["teacher_token_path_ade1"]["mean"], res["teacher_token_path_ade1"]["p90"])
    log.info("teacher ACTION EXPERT  minADE_%d  %.3f m (p90 %.3f)   <- the published head",
             k_expert, res[f"teacher_expert_minade{k_expert}"]["mean"],
             res[f"teacher_expert_minade{k_expert}"]["p90"])
    log.info("  hybrid: teacher CURVATURE + GT accel    %.3f m   <- path shape alone",
             res["hybrid_teacher_curv_gt_accel_ade1"]["mean"])
    log.info("  hybrid: GT curvature + teacher ACCEL    %.3f m   <- speed profile alone",
             res["hybrid_gt_curv_teacher_accel_ade1"]["mean"])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2))
        log.info("-> %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="challenging")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="default: eval.gate_max_windows, i.e. the gate's own prefix")
    ap.add_argument("--out", default="/data/vla/alpamayo2edge/runs/teacher_ceiling.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg0 = OmegaConf.load(a.config)
    mw = a.max_windows if a.max_windows is not None else cfg0.eval.get("gate_max_windows", None)
    main(a.config, a.split, mw, a.out)

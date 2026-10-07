"""Open-loop minADE of the flow head on Bench2Drive windows.

Shared by `scripts/05k_flow_minade.py` (standalone, per-window dump) and the
stage-2 trainer's per-epoch gate. k samples per window from the ODE sampler,
converted through the teacher's unicycle action space, minADE_k / minFDE_k in
the ego xy plane against the expert's future; the constant-velocity baseline
(zero normalised action) is reported next to it because that is what an
untrained head effectively emits (the smoke measured 7.0 m vs CV 6.0 m).
"""
from __future__ import annotations
import functools
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..data import bench2drive as b2d
from ..data.dataset import move_batch

log = logging.getLogger(__name__)

#: truncated horizons reported next to the 6.4 s minADE (steps at 10 Hz)
HORIZONS = {"1s": 10, "2s": 20, "3s": 30, "4s": 40}


def build_val_loader(cfg, student, split: str = "val", n: int | None = None, batch: int = 8,
                     coc=None, num_workers: int = 2, seed: int = 0):
    root = Path(cfg.paths.b2d_cache_root)
    ds = b2d.Bench2DriveDataset(root, student.context_builder(),
                                clip_ids=b2d.load_b2d_split(root, split),
                                coc_lookup=b2d.coc_lookup_from_cache(coc), for_generation=True)
    if n is not None and len(ds) > n:
        # Deterministic SEEDED subset: fixed across epochs like the stage-1 gate's
        # prefix, but shards are ordered by clip id, so a prefix is one or two
        # scenarios (the smoke's 16 windows were all AccidentTwoWays). A seeded
        # permutation covers the scenario mix.
        g = torch.Generator().manual_seed(int(seed))
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                      collate_fn=functools.partial(b2d.collate_b2d, pad_id=student.tokenizer.pad_token_id))


@torch.no_grad()
def evaluate_flow(cfg, student, dl, k: int = 6, steps: int = 10, temperature: float = 1.0,
                  per_window: bool = False, with_t0: bool = True) -> dict:
    """`with_t0`: also draw the deterministic x0 = 0 sample per window (what the
    closed-loop agent drives) and report its ADE as `t0_ade` - the car drives
    ONE plan, and best-of-k rewards covering go AND stay (sampler probe 2026-10-05:
    best-of-16 1.44 m vs T0 3.39 m on run 4)."""
    was_training = student.training
    student.eval()
    space = b2d.load_cache_action_space(Path(cfg.paths.b2d_cache_root), Path(cfg.paths.teacher_repo))
    per, curves, t0 = [], [], time.time()
    for batch in dl:
        batch = move_batch(batch)
        B = batch["gt_traj"].shape[0]
        n_per = k + int(with_t0)
        owner = torch.arange(B, device="cuda").repeat_interleave(n_per)
        x0 = torch.randn(B, n_per, student.n_action_tokens, student.raw_action_dim, device="cuda") * temperature
        if with_t0:
            x0[:, -1] = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = student.build_flow_context(batch)
            out = student.sample_actions(ctx, owner, steps=steps, x0=x0.view(B * n_per, *x0.shape[2:]))
        acts_all = out["actions"].float().cpu().view(B, n_per, 64, 2)
        acts = acts_all[:, :k]
        hx, hr = batch["hist_xyz"].float().cpu(), batch["hist_rot"].float().cpu()
        gt = batch["gt_future_xyz"].float().cpu()
        for b in range(B):
            xyz, _ = space.action_to_traj(acts[b], hx[b].expand(k, -1, -1), hr[b].expand(k, -1, -1, -1))
            disp = (xyz[..., :2] - gt[b, None, :, :2]).norm(dim=-1)                    # (k, 64)
            ade = disp.mean(-1)                                                          # (k,)
            fde = disp[:, -1]
            if with_t0:
                xt0, _ = space.action_to_traj(acts_all[b, k:], hx[b:b + 1], hr[b:b + 1])
                t0_disp = (xt0[0, :, :2] - gt[b, :, :2]).norm(dim=-1)                   # (64,)
            cv, _ = space.action_to_traj(torch.zeros(1, 64, 2), hx[b:b + 1], hr[b:b + 1])
            cv_disp = (cv[0, :, :2] - gt[b, :, :2]).norm(dim=-1)                          # (64,)
            per.append({"clip": batch["clip_ids"][b], "window": batch["window_idx"][b],
                        "scenario": batch["scenario"][b], "braked": bool(batch["should_brake"][b]),
                        "minade": float(ade.min()), "meanade": float(ade.mean()),
                        "minfde": float(fde.min()),
                        "cv_ade": float(cv_disp.mean()),
                        "t0_ade": float(t0_disp.mean()) if with_t0 else float("nan"),
                        "v0": float(batch["ego_speed_t0"][b]),
                        # truncated horizons: minADE over the first H steps (min over the
                        # k samples per horizon) next to constant velocity, because a
                        # 6.4 s mean hides where on the horizon the error lives
                        "minade_h": {h: float(disp[:, :n].mean(-1).min()) for h, n in HORIZONS.items()},
                        "meanade_h": {h: float(disp[:, :n].mean()) for h, n in HORIZONS.items()},
                        "cv_h": {h: float(cv_disp[:n].mean()) for h, n in HORIZONS.items()}})
            curves.append((disp[ade.argmin()].numpy(), cv_disp.numpy()))
    if was_training:
        student.train()

    def mean(key, rows):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")
    br = [r for r in per if r["braked"]]
    nb = [r for r in per if not r["braked"]]
    by = defaultdict(list)
    for r in per:
        by[r["scenario"]].append(r)
    res = {"n": len(per), "k": k, "steps": steps, "seconds": time.time() - t0,
           "minade": mean("minade", per), "meanade": mean("meanade", per),
           "minfde": mean("minfde", per), "cv_ade": mean("cv_ade", per),
           "t0_ade": mean("t0_ade", per),
           "p90_minade": float(np.percentile([r["minade"] for r in per], 90)) if per else float("nan"),
           "braked": {"n": len(br), "minade": mean("minade", br), "t0_ade": mean("t0_ade", br),
                      "cv_ade": mean("cv_ade", br)},
           "not_braked": {"n": len(nb), "minade": mean("minade", nb), "t0_ade": mean("t0_ade", nb),
                          "cv_ade": mean("cv_ade", nb)},
           "by_scenario": {s: {"n": len(v), "minade": mean("minade", v), "cv_ade": mean("cv_ade", v)}
                           for s, v in by.items()},
           "by_horizon": {h: {"minade": float(np.mean([r["minade_h"][h] for r in per])),
                              "meanade": float(np.mean([r["meanade_h"][h] for r in per])),
                              "cv_ade": float(np.mean([r["cv_h"][h] for r in per]))}
                          for h in HORIZONS} if per else {},
           # mean displacement per step (10 Hz) of the best-ADE sample and of constant velocity
           "curve_best": np.mean([c[0] for c in curves], axis=0).tolist() if curves else [],
           "curve_cv": np.mean([c[1] for c in curves], axis=0).tolist() if curves else []}
    if per_window:
        res["per_window"] = per
    return res


def log_result(res: dict, title: str, top: int = 12) -> None:
    log.info("%s  n=%d k=%d: T0 ADE %.3f m  minADE_%d %.3f  meanADE %.3f  minFDE %.3f  p90 %.3f | CV %.3f (%.0f s)",
             title, res["n"], res["k"], res.get("t0_ade", float("nan")), res["k"], res["minade"],
             res["meanade"], res["minfde"], res["p90_minade"], res["cv_ade"], res["seconds"])
    log.info("  braked (%d): minADE %.3f vs CV %.3f | others (%d): %.3f vs %.3f",
             res["braked"]["n"], res["braked"]["minade"], res["braked"]["cv_ade"],
             res["not_braked"]["n"], res["not_braked"]["minade"], res["not_braked"]["cv_ade"])
    if res.get("by_horizon"):
        log.info("  by horizon: " + "  ".join(f"{h} {v['minade']:.2f} (CV {v['cv_ade']:.2f})"
                                              for h, v in res["by_horizon"].items()))
    for s, v in sorted(res["by_scenario"].items(), key=lambda kv: -kv[1]["n"])[:top]:
        log.info("     %-40s n=%3d  minADE %.2f  CV %.2f", s, v["n"], v["minade"], v["cv_ade"])

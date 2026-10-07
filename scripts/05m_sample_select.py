"""Phase 2: which ONE trajectory should the car drive? Selectors over k flow samples.

The car drives a single plan, and the single deterministic sample (T=0, x0 = 0)
is ~3.4 m against best-of-6 ~1.9 m (runs/sampler_probe/). x0 = 0 is the mode of
the prior but not in its typical set, so T=0 may understate the head. This
draws k samples at T=1 per window (plus the T=0 sample, paired, same context)
and scores the selectors a deployed car could actually use, none of which sees
the expert:

    single    sample 0 at T=1                       (one random draw)
    t0        x0 = 0                                (what the closed-loop agent drives)
    mean_act  mean of the k normalised actions, then integrated
    mean_traj mean of the k integrated xy trajectories
    medoid    the sample with the lowest mean xy distance to the others
              (consensus that cannot blur a go/stop split into a half-stop)
    min       oracle best-of-k (expert-selected; the ceiling, not deployable)

each at k in --ks (prefixes of the same k_max draws, so the rows are paired).

    bash scripts/ada_run.sh scripts/05m_sample_select.py \
        --ckpt /data/vla/alpamayo2edge/runs/stage2/sft-run-4-b2d-arlora-traj-cocce/best \
        --out runs/sampler_probe/r4-best-select.json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.data import bench2drive as b2d                         # noqa: E402
from distill.data.dataset import move_batch                         # noqa: E402
from distill.eval.flow_minade import HORIZONS, build_val_loader     # noqa: E402

#: Bench2Drive open-loop planning L2 (Bench2DriveZoo / ORION `compute_planner_metric_stp3`,
#: orion.py:1033-1044 + metric_stp3.py:293): waypoints at 2 Hz (every 5th 10 Hz frame),
#: plan_L2_Ns = mean L2 over the first 2N waypoints. Our step i is t0 + 0.1 (i + 1) s.
B2D_L2 = {"1s": [4, 9], "2s": [4, 9, 14, 19], "3s": [4, 9, 14, 19, 24, 29]}

log = logging.getLogger("sample_select")

_spec = importlib.util.spec_from_file_location("fm05k", Path(__file__).with_name("05k_flow_minade.py"))
_05k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_05k)


def selectors(xy: torch.Tensor, acts: torch.Tensor, space, hx, hr, ks) -> dict[str, torch.Tensor]:
    """xy (k_max, 64, 2) sample trajectories, acts (k_max, 64, 2) -> {name: (64, 2) plan}."""
    out = {"single": xy[0]}
    for k in ks:
        if k == 1:
            continue
        x, a = xy[:k], acts[:k]
        out[f"mean_traj_{k}"] = x.mean(0)
        m, _ = space.action_to_traj(a.mean(0, keepdim=True), hx, hr)
        out[f"mean_act_{k}"] = m[0, :, :2]
        d = (x[:, None] - x[None]).norm(dim=-1).mean(-1)                  # (k, k) mean-ADE between samples
        out[f"medoid_{k}"] = x[d.sum(1).argmin()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True, help="stage-2 checkpoint, or 'none' = the phase-2 init (untrained head)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ks", default="1,4,8,16")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    ks = sorted(int(k) for k in a.ks.split(","))
    kmax = ks[-1]

    student = _05k.load_student(cfg, None if a.ckpt == "none" else a.ckpt)
    student.eval()
    dl = build_val_loader(cfg, student, split=a.split, n=a.n, batch=a.batch, coc=cfg.stage2.get("coc_cache"))
    space = b2d.load_cache_action_space(Path(cfg.paths.b2d_cache_root), Path(cfg.paths.teacher_repo))
    log.info("%s: %d windows, k_max=%d (+ T=0), %d steps, ckpt %s", a.split, len(dl.dataset), kmax, a.steps, a.ckpt)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)

    per, t0 = [], time.time()
    with torch.no_grad():
        for batch in dl:
            batch = move_batch(batch)
            B = batch["gt_traj"].shape[0]
            n_per = kmax + 1                                         # k_max at T=1, then the T=0 draw
            owner = torch.arange(B, device="cuda").repeat_interleave(n_per)
            x0 = torch.randn(B, n_per, student.n_action_tokens, student.raw_action_dim,
                             device="cuda", generator=gen)
            x0[:, -1] = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = student.build_flow_context(batch)
                out = student.sample_actions(ctx, owner, steps=a.steps, x0=x0.view(B * n_per, 64, 2))
            acts = out["actions"].float().cpu().view(B, n_per, 64, 2)
            hx, hr = batch["hist_xyz"].float().cpu(), batch["hist_rot"].float().cpu()
            gt = batch["gt_future_xyz"].float().cpu()[..., :2]
            for b in range(B):
                hxb, hrb = hx[b:b + 1], hr[b:b + 1]
                xyz, _ = space.action_to_traj(acts[b], hx[b].expand(n_per, -1, -1), hr[b].expand(n_per, -1, -1, -1))
                xy = xyz[..., :2]
                plans = selectors(xy[:kmax], acts[b, :kmax], space, hxb, hrb, ks)
                plans["t0"] = xy[-1]
                cv, _ = space.action_to_traj(torch.zeros(1, 64, 2), hxb, hrb)     # constant velocity
                plans["cv"] = cv[0, :, :2]
                disp_all = (xy[:kmax] - gt[b]).norm(dim=-1)                      # (kmax, 64)
                row = {"clip": batch["clip_ids"][b], "window": batch["window_idx"][b],
                       "scenario": batch["scenario"][b], "braked": bool(batch["should_brake"][b]),
                       "ade": {}, "fde": {}, "ade_h": {}, "b2d_l2": {}}
                for name, p in plans.items():
                    d = (p - gt[b]).norm(dim=-1)
                    row["ade"][name], row["fde"][name] = float(d.mean()), float(d[-1])
                    row["ade_h"][name] = {h: float(d[:n].mean()) for h, n in HORIZONS.items()}
                    row["b2d_l2"][name] = {h: float(d[ix].mean()) for h, ix in B2D_L2.items()}
                for k in ks:
                    row["ade"][f"min_{k}"] = float(disp_all[:k].mean(-1).min())
                # spread of the k_max samples (mean pairwise ADE): is the head uncertain here?
                row["spread"] = float((xy[:kmax, None] - xy[None, :kmax]).norm(dim=-1).mean())
                per.append(row)
            log.info("  %d windows, %.0f s", len(per), time.time() - t0)

    names = list(per[0]["ade"])
    br = [r for r in per if r["braked"]]
    nb = [r for r in per if not r["braked"]]

    def m(rows, name, key="ade"):
        return float(np.mean([r[key][name] for r in rows])) if rows else float("nan")
    res = {"ckpt": a.ckpt, "split": a.split, "n": len(per), "ks": ks, "steps": a.steps, "seed": a.seed,
           "seconds": time.time() - t0,
           "ade": {n: m(per, n) for n in names},
           "fde": {n: m(per, n, "fde") for n in per[0]["fde"]},
           "braked": {"n": len(br), **{n: m(br, n) for n in names}},
           "not_braked": {"n": len(nb), **{n: m(nb, n) for n in names}},
           "by_horizon": {n: {h: float(np.mean([r["ade_h"][n][h] for r in per])) for h in HORIZONS}
                          for n in per[0]["ade_h"]},
           "b2d_l2": {n: {h: float(np.mean([r["b2d_l2"][n][h] for r in per])) for h in B2D_L2}
                      for n in per[0]["b2d_l2"]},
           "spread": float(np.mean([r["spread"] for r in per]))}
    by = defaultdict(list)
    for r in per:
        by[r["scenario"]].append(r)
    res["by_scenario"] = {s: {"n": len(v), **{n: m(v, n) for n in names}} for s, v in by.items()}
    res["per_window"] = per

    log.info("%s n=%d (%.0f s)  mean sample spread %.2f m", a.ckpt, res["n"], res["seconds"], res["spread"])
    log.info("  %-14s %7s %7s %8s %8s", "selector", "ADE", "FDE", "braked", "others")
    for n in names:
        log.info("  %-14s %7.3f %7s %8.3f %8.3f", n, res["ade"][n],
                 f"{res['fde'][n]:.3f}" if n in res["fde"] else "-", res["braked"][n], res["not_braked"][n])
    log.info("  Bench2Drive open-loop L2 (2 Hz waypoints): " + " | ".join(
        f"{n} " + " ".join(f"{h} {v:.3f}" for h, v in res["b2d_l2"][n].items()) for n in ("t0", "single", "cv")))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

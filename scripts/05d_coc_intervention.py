"""Does the student's trajectory depend on its own chain-of-causation? (D-040)

GRPO in the recipe's shape trains the CoC THROUGH the trajectory it precedes:
a CoC token is reinforced when rollouts containing it produced lower-ADE
plans. That only works if the plan actually changes when the words change.
Stage 1 trained the trajectory path teacher-forced on the TEACHER's CoC, so
whether the student's plan follows the student's own reasoning is an
assumption, not a measurement. This measures it.

For N val windows, force a fixed set of CoCs (turn left / turn right / stop /
accelerate / keep lane / follow / the cached teacher trace), decode the
trajectory tokens after each, and report per CoC the mean final heading and
mean speed, plus the within-window spread. Strong coupling: "turn left" and
"turn right" pull the heading apart, "stop" and "accelerate" pull the speed
apart. No coupling: every CoC yields the same plan - and no trajectory-based
reward will ever move the reasoning.

    python scripts/05d_coc_intervention.py --ckpt <dir> [--cameras ...] --n 40 --samples 2
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_student, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402
from distill.eval import gt_reward                                  # noqa: E402

log = logging.getLogger("coc_intervention")

FORCED = {
    "turn_left":  "Turn left at the intersection since the route requires it",
    "turn_right": "Turn right at the intersection since the route requires it",
    "stop":       "Stop for the red traffic light since the signal is red",
    "accelerate": "Accelerate to proceed since the road ahead is clear",
    "keep_lane":  "Keep lane since the lane is clear ahead",
    "follow":     "Keep distance to the lead vehicle since it is directly ahead in our lane",
}


def traj_stats(xyz: np.ndarray) -> dict:
    g = xyz[:, :2]
    v = np.linalg.norm(np.diff(g, axis=0), axis=1) * 10.0
    d = np.diff(g, axis=0)
    heads = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    return {"speed": float(v.mean()), "v_end": float(v[-10:].mean()),
            "head_end": float(np.median(heads[-5:])), "ylat": float(g[-1, 1])}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--cameras", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hint", choices=["auto", "none", "match", "true"], default="auto",
                    help="route hint in the context (D-043). none: no slot. true: the "
                         "direction the driver took, for every forced CoC. match: the "
                         "turn CoCs get the hint that AGREES with them (turn_left -> "
                         "'Turn left ahead'), every other CoC gets the true direction - "
                         "so the heading test measures CoC+route together and the speed "
                         "test holds the route fixed. auto: match if stage1.route_hint "
                         "is on in the config, else none.")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    if a.cameras:
        cfg.raw["data"]["cameras"] = [c.strip() for c in a.cameras.split(",") if c.strip()]
    log.info("cameras: %s", list(cfg.data.raw["cameras"]))

    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    meta = checkpoint.load_into(student, Path(a.ckpt))
    if a.adapters:
        student.param_groups_stage1()
        checkpoint.load_adapters(student, a.adapters)
    log.info("checkpoint: %s (epoch=%s) adapters=%s", a.ckpt, meta.get("epoch"), a.adapters)
    student.eval()

    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, a.split),
                       cot_generation=True)
    hint_mode = a.hint
    if hint_mode == "auto":
        hint_mode = "match" if ds.route_hint else "none"
    log.info("route hint mode: %s", hint_mode)
    TURN_HINT = {"turn_left": "Turn left ahead", "turn_right": "Turn right ahead"}

    def hint_for(name: str, k: dict) -> str | None:
        if hint_mode == "none":
            return None
        if hint_mode == "match" and name in TURN_HINT:
            return TURN_HINT[name]
        return gt_reward.route_hint(k)
    pad_id = student.tokenizer.pad_token_id
    n = min(a.n, len(ds))
    rows = []
    for i in range(n):
        item = ds[i]
        path = ds.shards[i]
        window = ds._window(path.parent.name, int(path.stem))
        hx = torch.as_tensor(item["hist_xyz"]).float()[None]
        hr = torch.as_tensor(item["hist_rot"]).float()[None]
        gt = np.asarray(item["gt_future_xyz"], dtype=float)
        k = gt_reward.kinematics(gt)
        cocs = dict(FORCED); cocs["teacher"] = str(item["coc_text"]).strip() or FORCED["follow"]
        rec = {"i": i, "clip": item["clip_id"], "gt": {"head_end": k["h_end"], "speed": k["v0"],
                                                       "lateral": k["lateral"]}, "coc": {}}
        for name, text in cocs.items():
            ctx = ds.ctx.build(window, coc_text=text, for_generation=True,
                               nav_text=hint_for(name, k))
            batch = move_batch(collate_student([{"student": ctx}], pad_id))
            st = []
            for _ in range(a.samples):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tok = student.generate_traj_tokens(batch)
                xyz = student.detokenize_traj(tok, hx, hr)[0].numpy()
                s = traj_stats(xyz); s["ade"] = gt_reward.ade_xy(xyz, gt); st.append(s)
            rec["coc"][name] = {key: float(np.mean([s[key] for s in st])) for key in st[0]}
        rows.append(rec)
        if (i + 1) % 10 == 0:
            log.info("  %d/%d", i + 1, n)

    names = list(FORCED) + ["teacher"]
    log.info("\n%s  cameras=%d  windows=%d  samples/CoC=%d  hint=%s", Path(a.ckpt).parent.name,
             len(cfg.data.raw["cameras"]), n, a.samples, hint_mode)
    log.info("  %-12s %10s %10s %10s %8s", "forced CoC", "head_end", "v_mean", "ylat_end", "ADE")
    for nm in names:
        h = np.mean([r["coc"][nm]["head_end"] for r in rows]); v = np.mean([r["coc"][nm]["speed"] for r in rows])
        y = np.mean([r["coc"][nm]["ylat"] for r in rows]); ade = np.mean([r["coc"][nm]["ade"] for r in rows])
        log.info("  %-12s %+9.1f° %9.2f m/s %+9.2f m %7.2f m", nm, h, v, y, ade)
    # Coupling summaries: how far apart do opposite instructions pull the plan?
    dh = np.mean([r["coc"]["turn_left"]["head_end"] - r["coc"]["turn_right"]["head_end"] for r in rows])
    dv = np.mean([r["coc"]["accelerate"]["v_end"] - r["coc"]["stop"]["v_end"] for r in rows])
    spread_h = np.mean([np.std([r["coc"][nm]["head_end"] for nm in names]) for r in rows])
    spread_ade = np.mean([np.std([r["coc"][nm]["ade"] for nm in names]) for r in rows])
    log.info("\n  coupling:  heading(turn_left) - heading(turn_right) = %+.1f°   "
             "v_end(accelerate) - v_end(stop) = %+.2f m/s", dh, dv)
    log.info("  within-window spread across the 7 CoCs:  heading sd %.1f°   ADE sd %.2f m", spread_h, spread_ade)
    # Does the RIGHT CoC help? ADE with the matching-direction turn CoC on windows where the driver turned.
    turns = [r for r in rows if r["gt"]["lateral"].startswith("turn")]
    if turns:
        match = np.mean([r["coc"]["turn_left" if r["gt"]["lateral"] == "turn_left" else "turn_right"]["ade"] for r in turns])
        wrong = np.mean([r["coc"]["turn_right" if r["gt"]["lateral"] == "turn_left" else "turn_left"]["ade"] for r in turns])
        keep = np.mean([r["coc"]["keep_lane"]["ade"] for r in turns])
        log.info("  on the %d windows where the driver turned: ADE with the CORRECT turn CoC %.2f m, "
                 "the WRONG turn %.2f m, 'keep lane' %.2f m", len(turns), match, wrong, keep)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

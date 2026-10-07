"""Phase 2 baseline: the AR tower's DISCRETE trajectory tokens on Bench2Drive.

What the base student could already do before the flow head existed: the
stage-1 student (run-351/best + RL7 step-25 adapters = the phase-2 init) emits
128 future-bin tokens after the CoC, detokenised through the teacher's unicycle
action space (`student.detokenize_traj`, D-031 swap included). Same protocol as
`coarse_minade` (context ends at <|traj_future_start|>, CoC teacher-forced from
the stage-2 CoC cache, batch 1, sampling at the teacher's decode settings),
on the Bench2Drive val windows instead of PhysicalAI, scored with the
Bench2Drive open-loop L2 of `05m_sample_select.py` - so the numbers pair with
the flow head's.

    bash scripts/ada_run.sh scripts/05p_b2d_token_l2.py --out runs/sampler_probe/init-tokens-b2dL2-val.json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.data.dataset import move_batch                         # noqa: E402
from distill.eval.flow_minade import build_val_loader               # noqa: E402

log = logging.getLogger("b2d_token_l2")
_here = Path(__file__).parent
_load = lambda name, f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(name, _here / f))
_05k = _load("fm05k", "05k_flow_minade.py")
_05m = _load("ss05m", "05m_sample_select.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="none", help="'none' = the phase-2 init")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    student = _05k.load_student(cfg, None if a.ckpt == "none" else a.ckpt)
    student.eval()
    dl = build_val_loader(cfg, student, split=a.split, n=a.n, batch=1, coc=cfg.stage2.get("coc_cache"))
    log.info("%s: %d windows, token path of %s", a.split, len(dl.dataset), a.ckpt)
    per, t0 = [], time.time()
    with torch.no_grad():
        for batch in dl:
            batch = move_batch(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = student.generate_traj_tokens(batch)
            xyz = student.detokenize_traj(tok, batch["hist_xyz"], batch["hist_rot"]).float().cpu()
            xy = xyz.reshape(-1, 64, xyz.shape[-1])[0, :, :2]
            gt = batch["gt_future_xyz"].float().cpu()[0, :, :2]
            d = (xy - gt).norm(dim=-1)
            per.append({"clip": batch["clip_ids"][0], "window": int(batch["window_idx"][0]),
                        "scenario": batch["scenario"][0], "braked": bool(batch["should_brake"][0]),
                        "ade": float(d.mean()),
                        "b2d_l2": {h: float(d[ix].mean()) for h, ix in _05m.B2D_L2.items()}})
            if len(per) % 50 == 0:
                log.info("  %d windows, %.0f s", len(per), time.time() - t0)
    res = {"ckpt": a.ckpt, "path": "ar_tokens", "split": a.split, "n": len(per), "seconds": time.time() - t0,
           "ade": float(np.mean([r["ade"] for r in per])),
           "b2d_l2": {h: float(np.mean([r["b2d_l2"][h] for r in per])) for h in _05m.B2D_L2},
           "per_window": per}
    log.info("token path n=%d: ADE 6.4 s %.3f | Bench2Drive L2 %s", res["n"], res["ade"],
             " ".join(f"{h} {v:.3f}" for h, v in res["b2d_l2"].items()))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

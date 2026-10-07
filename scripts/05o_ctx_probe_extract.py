"""Phase 2: dump what the flow head READS, for a "is the cue in there?" probe.

The flow head attends the reasoner's per-layer K/V (flow_path.build_flow_context).
05m showed no post-hoc sample selection closes the T0 -> best-of-k gap and 05n
showed the stop/departure cause is mostly visible in the frames, so the question
is whether it survives into the context: phase 1's 05h (mean-pooled frozen vision
tokens, PhysicalAI) found only +0.015 AUC over ego history for hard brakes.

This writes, per labelled window, the context VALUES of a few layers (exactly
what the action tokens mix; (L, 8 heads x 128) -> a fixed Gaussian random
projection to --dim, which keeps linear decodability up to a small distortion,
so the probe is a slight LOWER bound) plus the vision-token mask, ego history and
labels; `05o_ctx_probe_fit.py` fits the probes.

Labels (expert future, the first 3 s):
  depart3  ego standing at t0 (v0 < 0.5 m/s): speed > 1 m/s within 3 s
  brake3   ego moving at t0 (v0 > 2 m/s): speed < 0.6 v0 within 3 s

    bash scripts/ada_run.sh scripts/05o_ctx_probe_extract.py --split val
    bash scripts/ada_run.sh scripts/05o_ctx_probe_extract.py --split train
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

log = logging.getLogger("ctx_probe")
_spec = importlib.util.spec_from_file_location("fm05k", Path(__file__).with_name("05k_flow_minade.py"))
_05k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_05k)

DEFAULT_CKPT = "/data/vla/alpamayo2edge/runs/stage2/sft-run-4-b2d-arlora-traj-cocce/best"


def labels(gt_future_xyz: np.ndarray, v0: float) -> tuple[int, int]:
    """(depart3, brake3), -1 where the task does not apply."""
    step = np.diff(np.vstack([np.zeros(3), gt_future_xyz]), axis=0)[:, :2]
    sp = np.linalg.norm(step, axis=1) / 0.1
    dep = int(sp[:30].max() > 1.0) if v0 < 0.5 else -1
    brk = int(sp[:30].min() < 0.6 * v0) if v0 > 2.0 else -1
    return dep, brk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--layers", default="9,18,27")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="/bulk/users/vla/alpamayo2edge/ctx_probe")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    layers = [int(x) for x in a.layers.split(",")]
    out = Path(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)

    student = _05k.load_student(cfg, a.ckpt)
    student.eval()
    stash = {}
    orig = student._reasoner_inputs

    def wrap(b):
        fwd, d = orig(b)
        stash["vis"] = fwd.get("visual_pos_masks")
        return fwd, d
    student._reasoner_inputs = wrap

    dl = build_val_loader(cfg, student, split=a.split, n=a.n, batch=a.batch, coc=cfg.stage2.get("coc_cache"))
    g = torch.Generator(device="cuda").manual_seed(0)              # ONE projection, shared by both splits
    P = torch.randn(1024, a.dim, device="cuda", generator=g) / a.dim ** 0.5
    files = {l: open(out / f"values_L{l}.f16", "wb") for l in layers}
    vis_f = open(out / "vis.u8", "wb")
    meta, offset, t0 = [], 0, time.time()
    with torch.no_grad():
        for batch in dl:
            batch = move_batch(batch)
            gt = batch["gt_future_xyz"].float().cpu().numpy()
            v0 = batch["ego_speed_t0"].float().cpu().numpy()
            lab = [labels(gt[b], float(v0[b])) for b in range(len(v0))]
            if all(d < 0 and k < 0 for d, k in lab):
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = student.build_flow_context(batch)
            vis = stash["vis"]
            Lk = ctx.key_mask.shape[1]
            for b, (dep, brk) in enumerate(lab):
                if dep < 0 and brk < 0:
                    continue
                m = ctx.key_mask[b]
                n_tok = int(m.sum())
                for l in layers:
                    v = ctx.values[l][b][m].reshape(n_tok, -1).float() @ P
                    files[l].write(v.half().cpu().numpy().tobytes())
                vis_f.write(vis[b, :Lk][m].to(torch.uint8).cpu().numpy().tobytes())
                meta.append({"clip": batch["clip_ids"][b], "window": int(batch["window_idx"][b]),
                             "scenario": batch["scenario"][b], "offset": offset, "n_tok": n_tok,
                             "v0": float(v0[b]), "depart3": dep, "brake3": brk,
                             "hist_xyz": batch["hist_xyz"][b].float().cpu().numpy().round(4).tolist()})
                offset += n_tok
            if len(meta) % 100 < len(lab):
                log.info("  %d windows stored, %.0f s", len(meta), time.time() - t0)
    for f in (*files.values(), vis_f):
        f.close()
    (out / "meta.json").write_text(json.dumps({"ckpt": a.ckpt, "layers": layers, "dim": a.dim,
                                               "n_tokens": offset, "windows": meta}))
    log.info("%s: %d windows, %d tokens -> %s (%.0f s)", a.split, len(meta), offset, out, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

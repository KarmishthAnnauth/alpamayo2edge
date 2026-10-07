"""Phase 2, step 3: open-loop minADE of the FLOW HEAD on Bench2Drive.

Replaces the never-run `05_eval.py`. Samples k trajectories per window with the
10-step ODE (flow_path.sample_actions), converts them through the teacher's
unicycle action space and scores minADE_k against the expert's future, next to
the constant-velocity baseline (the zero action = mean accel/curvature, which
is what an untrained head effectively emits) so epoch 0 has a number to beat.

    bash scripts/ada_run.sh scripts/05k_flow_minade.py --split val --n 200        # the init
    bash scripts/ada_run.sh scripts/05k_flow_minade.py --ckpt runs/stage2/epoch0    # a checkpoint

`--ckpt` loads a merged stage-2 checkpoint (AR tower already carries the RL
adapters); without it the phase-2 init is used, i.e. the untrained head.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402

log = logging.getLogger("flow_minade")


def load_student(cfg, ckpt: str | None):
    from distill import train_stage2 as ts2
    if ckpt is None:
        return ts2.load_student(cfg)
    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    meta = checkpoint.load_into(student, Path(ckpt))
    log.info("stage-2 checkpoint %s (epoch=%s)", ckpt, meta.get("epoch"))
    return student


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--steps", type=int, default=None, help="default stage2.sample_steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--coc-cache", default=None, help="default stage2.coc_cache; 'none' = no CoC")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    steps = a.steps or int(cfg.stage2.get("sample_steps", 10))
    coc = cfg.stage2.get("coc_cache") if a.coc_cache is None else (None if a.coc_cache == "none" else a.coc_cache)

    student = load_student(cfg, a.ckpt)
    from distill.eval.flow_minade import build_val_loader, evaluate_flow, log_result
    dl = build_val_loader(cfg, student, split=a.split, n=a.n, batch=a.batch, coc=coc)
    log.info("%s: %d windows, k=%d, %d steps, CoC %s", a.split, len(dl.dataset), a.k, steps, coc or "NONE")
    res = evaluate_flow(cfg, student, dl, k=a.k, steps=steps, temperature=a.temperature, per_window=True)
    res.update(ckpt=a.ckpt, split=a.split, coc=coc)
    log_result(res, a.ckpt or "phase-2 init (untrained head)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

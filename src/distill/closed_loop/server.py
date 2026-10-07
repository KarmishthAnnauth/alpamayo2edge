"""Closed-loop Bench2Drive: the model side.

Holds the stage-2 student and answers plan requests from the leaderboard agent
(`a2e_agent.py`) over a unix socket; see `a2e_common.py` for why there are two
processes. One request = one training-shaped window (4 cameras x 4 frames, 16
ego-history poses, route hint); the reply is one 6.4 s trajectory in the ego
frame plus the CoC it was conditioned on.

Per request, exactly what the open-loop evaluator does on a cached window
(`Bench2DriveDataset.__getitem__` -> `collate_b2d` -> `eval/flow_minade.py`):

    CoC     free-run from the `<|cot_start|>` prompt (`05j_b2d_coc.py generate`)
            unless the agent hands the previous one back (slow-fast split: the
            agent refreshes the CoC every ~2 s, the flow head every plan)
    context prompt + history + hint + CoC up to `<|traj_future_start|>` (P2-05)
    sample  k ODE samples, 10 Euler steps (P2-07), `action_to_traj` in the
            cache's action space (the Bench2Drive-normalised one for runs >= 2)

    bash scripts/ada_run.sh -m distill.closed_loop.server --socket /tmp/a2e.sock \
        --ckpt runs/stage2/sft-run-4-b2d-arlora-traj-cocce/best
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch

from .. import checkpoint
from ..config import load_config
from ..data import bench2drive as b2d
from ..data.dataset import collate_student, move_batch
from ..data.frames import CachedWindow
from ..student.edge_wrapper import EdgeStudent
from . import a2e_common as cm

log = logging.getLogger("a2e_server")


def load_student(cfg, ckpt: str | None):
    """`scripts/05k_flow_minade.py::load_student`: a merged stage-2 checkpoint,
    or the phase-2 init (untrained head) without one."""
    from .. import train_stage2 as ts2
    if ckpt is None:
        return ts2.load_student(cfg)
    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    meta = checkpoint.load_into(student, Path(ckpt))
    log.info("stage-2 checkpoint %s (epoch=%s)", ckpt, meta.get("epoch"))
    return student


class Planner:
    def __init__(self, cfg, student, steps: int, k: int, temperature: float,
                 select: str, coc_max_new_tokens: int):
        self.student = student.eval()
        self.ctxb = student.context_builder()
        self.pad_id = student.tokenizer.pad_token_id
        self.space = b2d.load_cache_action_space(Path(cfg.paths.b2d_cache_root),
                                                 Path(cfg.paths.teacher_repo))
        self.hints = b2d._hint_strings()
        self.steps, self.k, self.temperature, self.select = steps, k, temperature, select
        self.coc_max_new_tokens = coc_max_new_tokens

    def _batch(self, window, coc: str | None, hint: str):
        item = {"student": self.ctxb.build(window, coc_text=coc, traj_bins=None,
                                           nav_text=hint, for_generation=True),
                "img_drop": False}
        return move_batch(collate_student([item], self.pad_id))

    def _decode(self, ids: list[int]) -> tuple[str, bool]:
        lo = self.student.new_token_range[0]
        text = self.student.tokenizer.decode([i for i in ids if i < lo], skip_special_tokens=True)
        for s in ("<|cot_end|>", "</think>"):
            if s in text:
                return text.split(s)[0].strip(), True
        return text.strip(), False

    @torch.no_grad()
    def plan(self, req: dict) -> dict:
        t0 = time.time()
        hist_xyz = torch.from_numpy(req["ego_history_xyz"]).float()       # (16, 3)
        hist_rot = torch.from_numpy(req["ego_history_rot"]).float()       # (16, 3, 3)
        window = CachedWindow(
            clip_id="closed_loop",
            frames_student={slot: req[f"frames|{slot}"] for slot in self.ctxb.cameras},
            data={"ego_history_xyz": hist_xyz.view(1, 1, cm.N_HISTORY, 3),
                  "ego_history_rot": hist_rot.view(1, 1, cm.N_HISTORY, 3, 3)})
        hint = self.hints[req["hint"]]

        coc, terminated, t_coc = req.get("coc"), True, 0.0
        if coc is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = self.student.generate_coc_text(self._batch(window, None, hint),
                                                     max_new_tokens=self.coc_max_new_tokens)
            coc, terminated = self._decode(ids[0])
            t_coc = time.time() - t0

        k = self.k
        owner = torch.zeros(k, dtype=torch.long, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = self.student.build_flow_context(self._batch(window, coc, hint))
            out = self.student.sample_actions(ctx, owner, steps=self.steps,
                                              temperature=self.temperature)
        acts = out["actions"].float().cpu()                               # (k, 64, 2)
        if self.select == "mean":
            acts = acts.mean(0, keepdim=True)
        xyz, _ = self.space.action_to_traj(acts, hist_xyz.expand(acts.shape[0], -1, -1),
                                           hist_rot.expand(acts.shape[0], -1, -1, -1))
        xyz = xyz.numpy().astype(np.float32)
        return {"traj": xyz[0, :, :2].copy(), "samples": xyz[:, :, :2].copy(),
                "coc": coc, "coc_terminated": bool(terminated), "hint_text": hint,
                "t_coc": t_coc, "t_total": time.time() - t0}


def serve(planner: Planner, path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o600)
    srv.listen(1)
    log.info("READY on %s", path)
    try:
        while True:
            conn, _ = srv.accept()
            n, t_sum = 0, 0.0
            try:
                while True:
                    req = cm.recv_msg(conn)
                    op = req.get("op")
                    if op == "ping":
                        cm.send_msg(conn, {"ok": True, "cameras": list(planner.ctxb.cameras),
                                           "k": planner.k, "temperature": planner.temperature,
                                           "steps": planner.steps, "select": planner.select})
                    elif op == "plan":
                        rep = planner.plan(req)
                        n += 1; t_sum += rep["t_total"]
                        cm.send_msg(conn, rep)
                    elif op == "shutdown":
                        cm.send_msg(conn, {"ok": True})
                        return
                    else:
                        cm.send_msg(conn, {"error": f"unknown op {op!r}"})
            except ConnectionError:
                log.info("client gone after %d plans (%.2f s/plan); peak %.1f GiB", n,
                         t_sum / max(n, 1), torch.cuda.max_memory_allocated() / 2 ** 30)
            finally:
                conn.close()
    finally:
        srv.close()
        if os.path.exists(path):
            os.unlink(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="merged stage-2 checkpoint dir; default = the phase-2 init")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--k", type=int, default=1, help="flow samples per plan")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="prior scale; 0 = the single deterministic sample (sampler probe, 2026-10-01)")
    ap.add_argument("--select", default="first", choices=["first", "mean"],
                    help="with k > 1: drive sample 0, or the mean ACTION of the k samples")
    ap.add_argument("--steps", type=int, default=None, help="default stage2.sample_steps")
    ap.add_argument("--coc-max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    student = load_student(cfg, a.ckpt)
    planner = Planner(cfg, student, steps=a.steps or int(cfg.stage2.get("sample_steps", 10)),
                      k=a.k, temperature=a.temperature, select=a.select,
                      coc_max_new_tokens=a.coc_max_new_tokens)
    log.info("student loaded: %.1f GiB allocated; k=%d T=%.2f select=%s steps=%d",
             torch.cuda.memory_allocated() / 2 ** 30, a.k, a.temperature, a.select, planner.steps)
    serve(planner, a.socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

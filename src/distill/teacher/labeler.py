"""Offline teacher labeling pass (plan Phase 1, v2 amendments).

One npz shard per training window under cache_root/<clip_id>/<window_idx>.npz.
Storage per window is O(kilobytes) - flow targets are cached instead of KV
(the design change that makes single-GPU stage 2 pure supervised regression).

Resumable: existing shards are skipped, so the 500 -> 2000 -> 5000 increments
are just re-runs with a larger curated list.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ..teacher.wrapper import TeacherWrapper, TeacherWindowOutput
from ..data import frames
from ..data.preprocess import load_window, window_t0s_us

log = logging.getLogger(__name__)


def shard_path(cache_root: Path, clip_id: str, window_idx: int) -> Path:
    return cache_root / clip_id / f"{window_idx:02d}.npz"


def save_shard(path: Path, out: TeacherWindowOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        traj_token_ids=out.traj_token_ids.cpu().numpy().astype(np.int32),
        traj_topk_idx=out.traj_topk_idx.cpu().numpy().astype(np.int32),
        traj_topk_logp=out.traj_topk_logp.cpu().to(torch.float16).numpy(),
        coc_token_ids=out.coc_token_ids.cpu().numpy().astype(np.int32),
        coc_text=np.str_(out.coc_text),
        flow_t=out.flow_t.cpu().numpy().astype(np.float32),
        flow_a_t=out.flow_a_t.cpu().numpy().astype(np.float32),
        flow_v=out.flow_v.cpu().to(torch.float16).numpy(),
        traj_samples=out.traj_samples.cpu().numpy().astype(np.float32),
        gt_traj=out.gt_traj.cpu().numpy().astype(np.float32),
        gt_traj_token_ids=out.gt_traj_token_ids.cpu().numpy().astype(np.int32),
        gt_future_xyz=out.gt_future_xyz.cpu().numpy().astype(np.float32),
        feat_layers=np.array(sorted(out.feats.keys()), dtype=np.int32),
        **{f"feat_{k}": v.cpu().to(torch.float16).numpy() for k, v in out.feats.items()},
    )


def run_labeling(cfg, clip_ids: list[str]) -> None:
    cache_root = Path(cfg.paths.cache_root)
    teacher = TeacherWrapper(cfg)
    done = skipped = 0
    t0 = time.time()
    cache_frames = bool(cfg.data.get("cache_student_frames", True))
    quality = int(cfg.data.get("frame_jpeg_quality", 92))
    t0s = window_t0s_us(cfg)
    for clip_id in clip_ids:
        for w_idx, t0_us in enumerate(t0s):
            path = shard_path(cache_root, clip_id, w_idx)
            in_path = frames.input_path(cache_root, clip_id, w_idx)
            need_shard = not path.exists()
            # The student's inputs are written even when the teacher targets are
            # already cached: an older cache predates this file, and re-streaming
            # the clip once now beats re-streaming it every epoch later.
            need_input = cache_frames and not in_path.exists()
            # Decide BEFORE loading. `load_window` streams, decodes and resizes the
            # window, and it used to run on every iteration - including the ones
            # about to be skipped. That made the nested increments pay for their
            # predecessors twice: `--n 2000` re-streamed all 500 already-labeled
            # clips to write nothing. Resume is now genuinely free.
            if not need_shard and not need_input:
                skipped += 1
                continue

            window = load_window(cfg, clip_id, t0_us)
            if need_input:
                frames.save_window_input(in_path, window, quality=quality)
            if not need_shard:
                skipped += 1
                continue
            out = teacher.label_window(
                window,
                k_flow=cfg.teacher.flow_targets_per_window,
                topk=cfg.teacher.topk_logits,
                max_coc=cfg.teacher.max_coc_tokens,
                n_traj_samples=cfg.teacher.n_traj_samples,
            )
            save_shard(path, out)
            done += 1
            if done % 50 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                remaining = len(clip_ids) * cfg.data.windows_per_clip - done - skipped
                log.info("labeled %d (skipped %d) | %.2f win/s | eta %.1f h",
                         done, skipped, rate, remaining / max(rate, 1e-6) / 3600)
    manifest = {"clips": clip_ids, "windows_per_clip": cfg.data.windows_per_clip,
                "config_snapshot": cfg.raw}
    with open(cache_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

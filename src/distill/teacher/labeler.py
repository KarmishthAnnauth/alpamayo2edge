"""Offline teacher labeling pass (plan Phase 1, v2 amendments).

One npz shard per training window under cache_root/<clip_id>/<window_idx>.npz.
Storage per window is O(kilobytes) - flow targets are cached instead of KV
(the design change that makes single-GPU stage 2 pure supervised regression).

Resumable: existing shards are skipped, so the 500 -> 2000 -> 5000 increments
are just re-runs with a larger curated list.
"""
from __future__ import annotations
import collections
import os
import dataclasses
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..teacher.wrapper import TeacherWrapper, TeacherWindowOutput
from ..data import frames
from ..data.preprocess import load_window, window_t0s_us

log = logging.getLogger(__name__)


@dataclasses.dataclass
class _Item:
    """One window that actually needs work, decided from the filesystem alone."""
    clip_id: str
    w_idx: int
    t0_us: int
    path: Path
    in_path: Path
    need_shard: bool
    need_input: bool


@dataclasses.dataclass
class _Timing:
    """Where the wall clock goes, per phase. `wait` is the one that matters when
    tuning the prefetch pool: it is the part of `fetch` that did NOT hide behind
    teacher compute, so wait -> 0 means streaming is fully overlapped and the run
    is finally GPU-bound."""
    fetch: float = 0.0     # summed load_window duration (cost, across threads)
    wait: float = 0.0      # main thread blocked on the next window (unhidden)
    teacher: float = 0.0
    save: float = 0.0
    input: float = 0.0

    def line(self, n: int) -> str:
        n = max(n, 1)
        return (f"fetch {self.fetch / n:.1f}s wait {self.wait / n:.1f}s "
                f"teacher {self.teacher / n:.1f}s save {self.save / n:.1f}s "
                f"input {self.input / n:.1f}s")


def plan_work(cfg, cache_root: Path, clip_ids: list[str],
              cache_frames: bool) -> tuple[list[_Item], int]:
    """Filesystem-only scan of what needs doing. Streams nothing.

    Keeping this ahead of the fetch pool is what preserves free resume: only
    windows that need work are ever submitted, so `--n 2000` does not re-stream
    the 500 clips it already has. Pinned by tests/test_labeler_resume_offline.py.
    """
    t0s = window_t0s_us(cfg)
    work: list[_Item] = []
    skipped = 0
    for clip_id in clip_ids:
        for w_idx, t0_us in enumerate(t0s):
            path = shard_path(cache_root, clip_id, w_idx)
            in_path = frames.input_path(cache_root, clip_id, w_idx)
            need_shard = not path.exists()
            # The student's inputs are written even when the teacher targets are
            # already cached: an older cache predates this file, and re-streaming
            # the clip once now beats re-streaming it every epoch later.
            need_input = cache_frames and not in_path.exists()
            if not need_shard and not need_input:
                skipped += 1
                continue
            work.append(_Item(clip_id, w_idx, t0_us, path, in_path,
                              need_shard, need_input))
    return work, skipped


def _timed_load(cfg, item: _Item):
    t = time.perf_counter()
    window = load_window(cfg, item.clip_id, item.t0_us)
    return window, time.perf_counter() - t


def iter_windows_prefetched(cfg, work: list[_Item], workers: int, depth: int):
    """Yield (item, window, fetch_s, wait_s) in list order, with up to `depth`
    windows in flight.

    Streaming is network-bound and the teacher is GPU-bound, so they overlap
    almost perfectly; the loader itself is NVIDIA's `load_physical_aiavdataset`
    and fetches its cameras serially, which is why the concurrency has to live
    at the window level rather than the camera level.

    Order is preserved deliberately. Shards are keyed by (clip_id, w_idx) so
    correctness does not need it, but in-order consumption bounds memory to
    `depth` decoded windows and keeps the log readable against the curated list.
    `workers <= 1` takes the plain serial path - the one to use when debugging,
    since a traceback from a pool thread loses the enclosing context.

    Yields `window=None, err=<exception>` for a window that could not be
    fetched rather than raising: `huggingface_hub` already retries 5x, and a
    fetch that fails past that is a bad clip or a network blip, not a reason to
    lose a 27-hour run. Nothing is written for a failed window, so the next run
    simply picks it up again.
    """
    if workers <= 1:
        for item in work:
            t = time.perf_counter()
            try:
                window, fetch_s = _timed_load(cfg, item)
                err = None
            except Exception as e:            # not BaseException: Ctrl-C still exits
                window, fetch_s, err = None, time.perf_counter() - t, e
            yield item, window, fetch_s, time.perf_counter() - t, err
        return

    pending: collections.deque = collections.deque()
    remaining = iter(work)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fetch") as pool:
        def fill():
            while len(pending) < depth:
                item = next(remaining, None)
                if item is None:
                    return
                pending.append((item, pool.submit(_timed_load, cfg, item)))

        fill()
        while pending:
            item, fut = pending.popleft()
            t = time.perf_counter()
            try:
                window, fetch_s = fut.result()
                err = None
            except Exception as e:
                window, fetch_s, err = None, 0.0, e
            wait_s = time.perf_counter() - t
            fill()  # refill BEFORE yielding, so fetching runs during the teacher pass
            yield item, window, fetch_s, wait_s, err


def shard_path(cache_root: Path, clip_id: str, window_idx: int) -> Path:
    return cache_root / clip_id / f"{window_idx:02d}.npz"


def atomic_savez(path: Path, compressed: bool, **arrays) -> None:
    """Write an npz that is either complete or absent, never half-written.

    Resume keys off `path.exists()`, so a shard truncated by a kill mid-write is
    worse than a missing one: it is skipped forever and only surfaces later as a
    loader error, by which point the run that produced it is long gone. Same
    directory for the temp file so `os.replace` is a same-filesystem rename,
    which POSIX makes atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends '.npz' unless the name already ends in it, so name the
    # temp file accordingly and let it write exactly where we expect.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_shard(path: Path, out: TeacherWindowOutput) -> None:
    atomic_savez(
        path,
        True,
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
    done = 0
    t0 = time.time()
    cache_frames = bool(cfg.data.get("cache_student_frames", True))
    quality = int(cfg.data.get("frame_jpeg_quality", 92))
    workers = int(cfg.teacher.get("prefetch_workers", 4))
    # One more in flight than there are fetchers, so a worker never idles waiting
    # for the main thread to consume. Each in-flight window holds decoded frames,
    # so this is the memory knob as well as the throughput one.
    depth = int(cfg.teacher.get("prefetch_depth", max(workers + 2, 2)))

    work, skipped = plan_work(cfg, cache_root, clip_ids, cache_frames)
    total = len(clip_ids) * cfg.data.windows_per_clip
    log.info("labeling %d windows (%d already cached, %d total) | "
             "prefetch workers=%d depth=%d", len(work), skipped, total, workers, depth)

    # A run this long is unattended by definition, so one bad window must not end
    # it. Nothing is written for a failure, so the next run retries it for free.
    # `max_consecutive_failures` is the circuit breaker: scattered failures are
    # the network, but a solid run of them is systemic (expired token, full disk,
    # dead GPU) and grinding through 9,861 windows writing nothing is worse than
    # stopping.
    max_consecutive = int(cfg.teacher.get("max_consecutive_failures", 25))
    failed: list[tuple[str, int, str]] = []
    consecutive = 0

    tm = _Timing()
    for item, window, fetch_s, wait_s, err in iter_windows_prefetched(
            cfg, work, workers, depth):
        tm.fetch += fetch_s
        tm.wait += wait_s
        if err is None:
            try:
                if item.need_input:
                    t = time.perf_counter()
                    frames.save_window_input(item.in_path, window, quality=quality)
                    tm.input += time.perf_counter() - t
                if not item.need_shard:
                    skipped += 1
                    consecutive = 0
                    continue
                t = time.perf_counter()
                out = teacher.label_window(
                    window,
                    k_flow=cfg.teacher.flow_targets_per_window,
                    topk=cfg.teacher.topk_logits,
                    max_coc=cfg.teacher.max_coc_tokens,
                    n_traj_samples=cfg.teacher.n_traj_samples,
                )
                tm.teacher += time.perf_counter() - t
                t = time.perf_counter()
                save_shard(item.path, out)
                tm.save += time.perf_counter() - t
            except Exception as e:            # not BaseException: Ctrl-C still exits
                err = e

        if err is not None:
            consecutive += 1
            failed.append((item.clip_id, item.w_idx, f"{type(err).__name__}: {err}"))
            log.warning("window %s[%d] FAILED (%d consecutive, %d total): %s: %s",
                        item.clip_id, item.w_idx, consecutive, len(failed),
                        type(err).__name__, str(err)[:200])
            if consecutive >= max_consecutive:
                log.error("%d consecutive failures - stopping. This is systemic, "
                          "not the network. Re-run to resume; nothing was written "
                          "for the failed windows.", consecutive)
                break
            continue

        consecutive = 0
        done += 1
        if done % 50 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            remaining = total - done - skipped
            log.info("labeled %d (skipped %d) | %.2f win/s | eta %.1f h | %s",
                     done, skipped, rate, remaining / max(rate, 1e-6) / 3600,
                     tm.line(done))

    if done:
        wall = time.time() - t0
        log.info("done: %d windows in %.2f h (%.1f s/window) | %s",
                 done, wall / 3600, wall / done, tm.line(done))
        # fetch/wall is the honest speedup readout: at workers=1 it sits near the
        # fraction of the run spent streaming; with the pool working it can exceed
        # 1.0, which just means more than one window-second of fetching happened
        # per second of wall clock.
        log.info("streaming overlap: %.1fx (%.0f s fetched / %.0f s wall), "
                 "unhidden wait %.0f%% of wall",
                 tm.fetch / max(wall, 1e-6), tm.fetch, wall,
                 100.0 * tm.wait / max(wall, 1e-6))

    if failed:
        log.warning("%d windows failed and were skipped - re-run to retry them "
                    "(nothing was written, so they cost nothing to redo)", len(failed))
        for clip_id, w_idx, msg in failed[:20]:
            log.warning("  %s[%d] %s", clip_id, w_idx, msg)
        if len(failed) > 20:
            log.warning("  ... and %d more; full list in failed_windows.json",
                        len(failed) - 20)
        with open(cache_root / "failed_windows.json", "w") as f:
            json.dump([{"clip_id": c, "window": w, "error": m} for c, w, m in failed],
                      f, indent=2)

    manifest = {"clips": clip_ids, "windows_per_clip": cfg.data.windows_per_clip,
                "config_snapshot": cfg.raw}
    with open(cache_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

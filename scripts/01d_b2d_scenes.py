"""Build the replay scenes of the Bench2Drive window cache (DiffGRPO reward input).

    python scripts/01d_b2d_scenes.py --limit 5             # 5 clips, timing
    python scripts/01d_b2d_scenes.py --workers 16          # every clip in the cache
    python scripts/01d_b2d_scenes.py --clips A,B            # named clips

For every window of every clip in `<cache>/manifest.json` (anchors from
`<clip>/windows.json`), reads the raw annotations once per clip and writes
`<clip>/{w:02d}_scene.npz` (distill.data.b2d_scene). CPU only, resumable (an
existing scene file is skipped). The build self-checks the box/yaw convention
on every window and fails the clip loudly if it does not hold.
"""
from __future__ import annotations
import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

log = logging.getLogger("b2d_scenes")


def _build_clip(args: tuple[str, str, str]) -> dict:
    raw_root, cache_root, clip_id = Path(args[0]), Path(args[1]), args[2]
    from distill.data import bench2drive as b2d
    from distill.data import b2d_scene as sc
    t_start = time.perf_counter()
    rec = dict(clip_id=clip_id, scenes=0, skipped=0, seconds=0.0, error=None, actors=0)
    try:
        rows = json.loads((cache_root / clip_id / "windows.json").read_text())
        todo = [(int(r["idx"]), int(r["anchor_frame"])) for r in rows
                if not sc.scene_path(cache_root, clip_id, int(r["idx"])).exists()]
        rec["skipped"] = len(rows) - len(todo)
        if todo:
            anno = b2d.load_clip_anno(raw_root / clip_id, cameras=[])
            n_need = max(t0 for _, t0 in todo) + b2d.N_FUTURE + 1
            boxes = sc.load_clip_boxes(raw_root / clip_id, min(anno.n, n_need))
            for w_idx, t0 in todo:
                scene = sc.build_scene(anno, boxes, t0, clip_id)
                sc.save_scene(sc.scene_path(cache_root, clip_id, w_idx), scene)
                rec["scenes"] += 1
                rec["actors"] += int(scene["actor_id"].shape[0])
    except Exception as e:      # one bad clip must not end an unattended run
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = time.perf_counter() - t_start
    return rec


def main() -> int:
    from distill.config import load_config
    from distill.data import bench2drive as b2d
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--root", default=str(b2d.DEFAULT_ROOT), help="raw extracted clips")
    ap.add_argument("--cache", default=None, help="window cache (default paths.b2d_cache_root)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clips", default=None, help="comma-separated clip ids")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    cache = Path(a.cache or cfg.paths.b2d_cache_root)
    clips = json.loads((cache / "manifest.json").read_text())["clips"]
    if a.clips:
        want = set(a.clips.split(","))
        clips = [c for c in clips if c in want]
    if a.limit:
        clips = clips[:a.limit]
    log.info("%d clips -> scenes under %s (raw %s)", len(clips), cache, a.root)
    jobs = [(a.root, str(cache), c) for c in clips]
    t0 = time.perf_counter()
    recs = []
    with mp.get_context("spawn").Pool(max(1, a.workers)) as pool:
        for rec in pool.imap_unordered(_build_clip, jobs):
            recs.append(rec)
            if rec["error"]:
                log.error("%s FAILED: %s", rec["clip_id"], rec["error"])
            else:
                log.info("%-55s %3d scenes (+%d cached) %4d actor-rows %6.2f s  [%d/%d]",
                         rec["clip_id"], rec["scenes"], rec["skipped"], rec["actors"],
                         rec["seconds"], len(recs), len(jobs))
    ok = [r for r in recs if not r["error"]]
    print(f"\nbuilt {sum(r['scenes'] for r in ok)} scenes on {len(ok)} clips "
          f"({len(recs) - len(ok)} failed) in {time.perf_counter() - t0:.1f} s wall")
    failed = [r for r in recs if r["error"]]
    for r in failed[:20]:
        print("  FAILED", r["clip_id"], r["error"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the Bench2Drive window cache for phase 2 (flow-head training).

    python scripts/01c_b2d_windows.py --limit 5            # 5 clips, timing, summary
    python scripts/01c_b2d_windows.py --workers 8          # everything with a .done marker
    python scripts/01c_b2d_windows.py --summary-only       # re-print the table + splits

Walks `/bulk/datasets/bench2drive/extracted/*/` (ONLY dirs carrying `.done` -
the extraction job deletes tarballs as it verifies them and wipes partial dirs),
turns every clip into windows (`distill.data.bench2drive`), and writes them in
the labeler's on-disk layout:

    <out>/<clip>/{w:02d}.npz          targets: gt_traj (64x2 action space),
                                       gt_future_xyz/rot, should_brake, expert
                                       controls, route hint + raw command, scenario,
                                       town, route, weather, anchor frame, and
                                       gt_traj_token_ids when the tokenizer spec
                                       is available
    <out>/<clip>/{w:02d}_input.npz    student frames (JPEG) + ego history
    <out>/<clip>/windows.json         one small row per window (summaries)
    <out>/splits.json                 train/val by town+route, stratified by scenario
    <out>/manifest.json, split_train.json, split_val.json   labeler-format lists

Resumable: a window whose two files exist is skipped. CPU only. Wall time per
clip is printed (JPEG decode + resize dominates: 4 frames x 3 raw cameras per
window plus the tele crop).
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

log = logging.getLogger("b2d_windows")

_W: dict = {}   # per-worker state (action space, tokenizer, params)


def _init_worker(params_dict: dict, spec_path: str | None, teacher_repo: str | None,
                 action_stats: dict | None = None):
    import torch
    torch.set_num_threads(1)
    from distill.data import bench2drive as b2d
    _W["params"] = b2d.WindowParams(**params_dict)
    _W["space"] = b2d.load_action_space(Path(teacher_repo) if teacher_repo else None,
                                        stats=action_stats)
    _W["tokenizer_fn"] = None
    if spec_path:
        spec = torch.load(spec_path, weights_only=False)   # pickled callables, D-032
        _W["tokenizer_fn"] = spec["tokenizer_fn"]


def _build_clip(args: tuple[str, str]) -> dict:
    clip_dir, out_root = Path(args[0]), Path(args[1])
    from distill.data import bench2drive as b2d
    from distill.data import frames as frames_mod
    t_start = time.perf_counter()
    rec = dict(clip_id=clip_dir.name, windows=0, skipped=0, seconds=0.0, error=None)
    try:
        params, space, tok = _W["params"], _W["space"], _W["tokenizer_fn"]
        anno = b2d.load_clip_anno(clip_dir, params.cameras)
        anchors = b2d.anchor_frames(anno.n, params.stride)
        rows = []
        wj = out_root / clip_dir.name / "windows.json"
        old = {r["idx"]: r for r in json.loads(wj.read_text())} if wj.exists() else {}
        reader = b2d._FrameReader(clip_dir, params)
        for w_idx, t0 in enumerate(anchors):
            tp = b2d.targets_path(out_root, clip_dir.name, w_idx)
            ip = frames_mod.input_path(out_root, clip_dir.name, w_idx)
            if tp.exists() and ip.exists() and w_idx in old:
                rows.append(old[w_idx]); rec["skipped"] += 1
                continue
            w = b2d.build_window(anno, t0, params, space, reader, tok, with_frames=True)
            reader.clear()
            b2d.save_window(out_root, w_idx, w, quality=params.jpeg_quality)
            rows.append(b2d.window_record(w_idx, w))
            rec["windows"] += 1
        tmp = wj.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, indent=0))
        tmp.replace(wj)
        rec["n_frames"] = anno.n
        rec["total_windows"] = len(rows)
    except Exception as e:      # one bad clip must not end an unattended run
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = time.perf_counter() - t_start
    return rec


def _print_summary(out_root: Path, sp: dict | None):
    from distill.data import bench2drive as b2d
    s = b2d.summarize(out_root)
    print(f"\n== Bench2Drive cache {out_root}: {s['windows']} windows, "
          f"braking-window fraction {s['braking_frac']:.3f} (any flagged future frame; "
          f"mean flagged-frame fraction {s['brake_frame_frac']:.3f}), "
          f"route hint (command) == kinematic hint on {s['route_hint_agree_frac']:.1%}; "
          f"action-space floor at 6.4 s p50 {s['action_floor_p50']:.2f} m / p90 {s['action_floor_p90']:.2f} m")
    print(f"   route commands: {s['commands']}")
    print(f"\n   {'scenario':38s} {'windows':>8s} {'braking':>8s} {'frames':>8s} {'floor50':>8s} {'floor90':>8s}")
    for k, v in s["by_scenario"].items():
        print(f"   {k:38s} {v['windows']:8d} {v['braking_frac']:8.3f} {v['brake_frame_frac']:8.3f} "
              f"{v['action_floor_p50']:8.2f} {v['action_floor_p90']:8.2f}")
    print(f"\n   {'town':38s} {'windows':>8s} {'braking':>8s} {'frames':>8s}")
    for k, v in s["by_town"].items():
        print(f"   {k:38s} {v['windows']:8d} {v['braking_frac']:8.3f} {v['brake_frame_frac']:8.3f}")
    if sp:
        print(f"\n   splits: {len(sp['train'])} train clips / {len(sp['val'])} val clips "
              f"({len(sp['train_routes'])} / {len(sp['val_routes'])} routes, seed {sp['seed']}, "
              f"val fraction {sp['val_fraction']})")
        for scn, v in sp["by_scenario"].items():
            print(f"      {scn:38s} routes {v['routes']:3d}  val {v['val_routes']:2d}  "
                  f"clips {v['clips']:3d}  val {v['val_clips']:3d}")


def main():
    from distill.config import load_config
    from distill.data import bench2drive as b2d

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(b2d.DEFAULT_ROOT))
    ap.add_argument("--out", default=str(b2d.DEFAULT_CACHE))
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stride", type=int, default=b2d.DEFAULT_STRIDE, help="frames between anchors")
    ap.add_argument("--route-rule", default="horizon", choices=["horizon", "near_far"])
    ap.add_argument("--limit", type=int, default=None, help="build only the first N done clips")
    ap.add_argument("--clips", default=None, help="comma-separated clip ids to build")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spec", default=None,
                    help="traj_tokenizer_spec.pt for gt_traj_token_ids "
                         "(default: <paths.cache_root>/traj_tokenizer_spec.pt if present)")
    ap.add_argument("--no-tokens", action="store_true", help="skip gt_traj_token_ids")
    ap.add_argument("--action-stats", default=None, help="JSON with accel/curvature mean+std (physical units) to re-normalise the action space; written to <out>/action_space.json")
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(a.config)
    out_root = Path(a.out)
    action_stats = None
    if a.action_stats:
        import json as _json
        from distill.data import bench2drive as _b2d
        raw = _json.loads(Path(a.action_stats).read_text())
        action_stats = {k: float(raw[k]) for k in _b2d.NORM_KEYS}
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / _b2d.ACTION_SPACE_FILE).write_text(_json.dumps(
            {**action_stats, "source": raw.get("source"), "stats_file": str(a.action_stats)}, indent=1))
        log.info("action-space normalisation from %s: %s", a.action_stats, action_stats)
    out_root.mkdir(parents=True, exist_ok=True)

    if not a.summary_only:
        params = b2d.WindowParams.from_cfg(cfg, stride=a.stride, route_rule=a.route_rule)
        spec = None
        if not a.no_tokens:
            spec = a.spec or str(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt")
            if not Path(spec).exists():
                log.warning("no tokenizer spec at %s - shards get no gt_traj_token_ids", spec)
                spec = None
        teacher_repo = str(cfg.paths.teacher_repo) if Path(str(cfg.paths.teacher_repo)).exists() else None

        clips = b2d.list_done_clips(Path(a.root))
        if a.clips:
            want = set(a.clips.split(","))
            clips = [c for c in clips if c.name in want]
        if a.limit:
            clips = clips[:a.limit]
        log.info("%d done clips to build -> %s (stride %d, %s, tokens=%s)",
                 len(clips), out_root, params.stride, params.cameras, spec is not None)

        jobs = [(str(c), str(out_root)) for c in clips]
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        recs = []
        with ctx.Pool(max(1, a.workers), initializer=_init_worker,
                      initargs=(params.__dict__, spec, teacher_repo, action_stats)) as pool:
            for rec in pool.imap_unordered(_build_clip, jobs):
                recs.append(rec)
                if rec["error"]:
                    log.error("%s FAILED: %s", rec["clip_id"], rec["error"])
                else:
                    log.info("%-55s %3d windows (+%d cached)  %6.2f s  [%d/%d]",
                             rec["clip_id"], rec["windows"], rec["skipped"], rec["seconds"],
                             len(recs), len(jobs))
        wall = time.perf_counter() - t0
        ok = [r for r in recs if not r["error"]]
        built = sum(r["windows"] for r in ok)
        print(f"\nbuilt {built} windows on {len(ok)} clips ({len(recs) - len(ok)} failed) in "
              f"{wall:.1f} s wall with {a.workers} workers; per-clip build time "
              f"mean {sum(r['seconds'] for r in ok) / max(len(ok), 1):.2f} s "
              f"({built / max(sum(r['seconds'] for r in ok), 1e-9):.1f} windows/s per worker)")

    # Splits over every clip with shards in the cache (built now or earlier).
    have = sorted(p.name for p in out_root.iterdir()
                  if p.is_dir() and (p / "windows.json").exists())
    sp = b2d.write_splits(out_root, have, a.val_fraction, a.seed) if have else None
    _print_summary(out_root, sp)


if __name__ == "__main__":
    main()

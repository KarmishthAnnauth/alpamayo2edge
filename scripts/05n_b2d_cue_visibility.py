"""Phase 2: is the CAUSE of the expert's stop / departure visible to the model at t0?

P2-15 explained the CoC's missing braking anticipation on CARLA with "scripted
stops, often invisible 6.4 s ahead" - never tested. This tests it on the val
windows from the Bench2Drive annotations (privileged state the model never
gets) and splits run 4's error by the answer. CPU only.

Per window (bucket from `buckets_val.json`, scripts/01e_b2d_buckets.py):

  braking buckets (hard_brake, brake, to_stop): event te = first frame in
    [t0, t0+64) where the expert's brake control > 0.1. Cause at te, first match:
      light   a traffic light with affects_ego that is red/yellow at te
      stop    a stop sign with affects_ego at te
      actor   the nearest vehicle / walker inside the ego's corridor
              (x in [0, 25] m, |y| < 2.5 m, ego frame at that frame) at any
              frame in [te, te+2 s]
      none    nothing found (privileged/scripted, or our heuristic missed it)
  start: event te = first frame with speed > 0.5 m/s; cause = what held the
    ego at te-0.5 s: red light / stop sign / actor in the corridor (x in [0, 15])
    / none.

Visibility at t0 (last input frame), deliberately generous:
  actor   present at t0, projects inside one of the three input cameras
          (rgb_front, rgb_front_left, rgb_front_right), distance <= 50 m
          (walkers 35 m), lidar num_points >= 5 (occlusion proxy)
  light   its state at t0: red/yellow at t0 = the cue is there; green at t0 =
          it turns later (the timing is not in the frames), plus in-camera
  stop    in camera and <= 50 m at t0

    python scripts/05n_b2d_cue_visibility.py \
        --err runs/sampler_probe/r4-epoch7-T0-k1.json \
        --err-best runs/flow-minade-sft-run-4-b2d-arlora-traj-cocce-best-valfull.json \
        --out runs/coc_probe/cue_visibility_val.json
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402

from distill.data import bench2drive as b2d                         # noqa: E402

CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")         # the three input cameras (P2-10)
BRAKING = ("hard_brake", "brake", "to_stop")
RED, YELLOW, GREEN = 0, 1, 2
H = b2d.N_FUTURE


@lru_cache(maxsize=256)
def frame(clip_dir: str, k: int) -> dict:
    with gzip.open(Path(clip_dir) / "anno" / f"{k:05d}.json.gz", "rt") as f:
        return json.load(f)


def ego_of(d: dict) -> dict:
    return next(b for b in d["bounding_boxes"] if b["class"] == "ego_vehicle")


def to_ego(d: dict, p) -> np.ndarray:
    """World point -> CARLA ego frame (x fwd, y right, z up) of frame d."""
    w2e = np.asarray(ego_of(d)["world2ego"], dtype=np.float64)
    return (w2e @ np.r_[np.asarray(p, dtype=np.float64)[:3], 1.0])[:3]


def in_camera(d: dict, p_ego: np.ndarray) -> bool:
    for c in CAMS:
        s = d["sensors"][c]
        q = np.linalg.inv(np.asarray(s["cam2ego"])) @ np.r_[p_ego, 1.0]
        if q[0] <= 0.5:
            continue
        K = np.asarray(s["intrinsic"])
        u, v = K[0, 0] * q[1] / q[0] + K[0, 2], -K[1, 1] * q[2] / q[0] + K[1, 2]
        if 0 <= u < s["image_size_x"] and 0 <= v < s["image_size_y"]:
            return True
    return False


def actors(d: dict):
    return [b for b in d["bounding_boxes"] if b["class"] in ("vehicle", "walker")]


def affecting(d: dict, cls: str, pred=lambda b: True):
    return [b for b in d["bounding_boxes"] if b["class"] == cls and b.get("affects_ego") and pred(b)]


def corridor_actor(clip: str, frames: range, x_max: float) -> dict | None:
    best = None
    for k in frames:
        d = frame(clip, k)
        for b in actors(d):
            p = to_ego(d, b["center"])
            if 0.0 <= p[0] <= x_max and abs(p[1]) < 2.5 and (best is None or p[0] < best[0]):
                best = (p[0], b["id"], b["class"], k)
    return None if best is None else {"id": best[1], "class": best[2], "x": float(best[0]), "frame": best[3]}


def visible_actor(clip: str, t0: int, aid: str) -> tuple[bool, str]:
    d = frame(clip, t0)
    b = next((b for b in actors(d) if b["id"] == aid), None)
    if b is None:
        return False, "absent_at_t0"
    p = to_ego(d, b["center"])
    rng = 35.0 if b["class"] == "walker" else 50.0
    if not in_camera(d, p):
        return False, "out_of_fov"
    if float(np.hypot(p[0], p[1])) > rng:
        return False, "too_far"
    if int(b.get("num_points") or 0) < 5:
        return False, "occluded"
    return True, "visible"


def classify(clip: str, t0: int, bucket: str, n_frames: int) -> dict:
    anno = [frame(clip, k) for k in range(t0, min(t0 + H, n_frames))]
    row = {"bucket": bucket}
    if bucket in BRAKING:
        te = next((t0 + i for i, d in enumerate(anno) if float(d["brake"]) > 0.1), None)
    elif bucket == "start":
        te = next((t0 + i for i, d in enumerate(anno) if float(d["speed"]) > 0.5), None)
    else:
        return row
    if te is None:
        row.update(cause="no_event")
        return row
    row["onset_s"] = round((te - t0) * b2d.DT, 1)
    probe = te if bucket in BRAKING else max(t0, te - 5)
    d = frame(clip, probe)
    lights = affecting(d, "traffic_light", lambda b: b.get("state") in (RED, YELLOW))
    stops = affecting(d, "traffic_sign", lambda b: "stop" in b.get("type_id", ""))
    if lights:
        lid = lights[0]["id"]
        d0 = frame(clip, t0)
        l0 = next((b for b in d0["bounding_boxes"] if b["id"] == lid), None)
        row["cause"] = "light"
        if l0 is None:
            row["vis"], row["why"] = False, "absent_at_t0"
        elif l0.get("state") in (RED, YELLOW):
            row["vis"], row["why"] = True, "red_at_t0"
        else:
            row["vis"], row["why"] = False, "green_at_t0"
        if l0 is not None:
            row["light_in_cam"] = in_camera(d0, to_ego(d0, l0["center"]))
        return row
    if stops:
        sid = stops[0]["id"]
        d0 = frame(clip, t0)
        s0 = next((b for b in d0["bounding_boxes"] if b["id"] == sid), None)
        row["cause"] = "stop"
        if s0 is None:
            row["vis"], row["why"] = False, "absent_at_t0"
        else:
            p = to_ego(d0, s0["center"])
            row["vis"] = bool(in_camera(d0, p) and np.hypot(p[0], p[1]) <= 50.0)
            row["why"] = "visible" if row["vis"] else "out_of_view"
        return row
    span = (range(te, min(te + 20, n_frames)) if bucket in BRAKING else range(probe, probe + 1))
    a = corridor_actor(clip, span, 25.0 if bucket in BRAKING else 15.0)
    if a is None:
        row.update(cause="none", vis=False, why="no_cause_found")
        return row
    row.update(cause="actor", actor_class=a["class"], actor_x=a["x"])
    row["vis"], row["why"] = visible_actor(clip, t0, a["id"])
    if row["vis"]:
        # visible at t0, but already in the path or still outside it (cut-in / crossing intent)
        p0 = to_ego(frame(clip, t0), next(b for b in actors(frame(clip, t0)) if b["id"] == a["id"])["center"])
        row["in_path_at_t0"] = bool(abs(p0[1]) < 2.5 and p0[0] >= 0)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/bulk/users/vla/alpamayo2edge/b2d_cache_b2dnorm")
    ap.add_argument("--root", default=str(b2d.DEFAULT_ROOT))
    ap.add_argument("--split", default="val")
    ap.add_argument("--err", required=True, help="per-window json for the single-sample ADE (minade = T0 k=1)")
    ap.add_argument("--err-best", default=None, help="per-window json for best-of-6")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cache = Path(a.cache)
    buckets = json.loads((cache / f"buckets_{a.split}.json").read_text())["buckets"]
    err = {(w["clip"], int(w["window"])): w["minade"] for w in json.loads(Path(a.err).read_text())["per_window"]}
    best = ({(w["clip"], int(w["window"])): w["minade"] for w in json.loads(Path(a.err_best).read_text())["per_window"]}
            if a.err_best else {})

    rows = []
    for clip in sorted({k.split("/")[0] for k in buckets}):
        recs = json.loads((cache / clip / "windows.json").read_text())
        recs = recs["windows"] if isinstance(recs, dict) else recs
        clip_dir = str(Path(a.root) / clip)
        n = b2d.count_frames(Path(clip_dir))
        for r in recs:
            key = f"{clip}/{int(r['idx'])}"
            if key not in buckets:
                continue
            row = classify(clip_dir, int(r["anchor_frame"]), buckets[key], n)
            row.update(clip=clip, window=int(r["idx"]), scenario=r["scenario"],
                       ade_t0=err.get((clip, int(r["idx"]))), ade_best6=best.get((clip, int(r["idx"]))))
            rows.append(row)
        frame.cache_clear()
    print(f"{len(rows)} windows")

    ev = [r for r in rows if r.get("cause") not in (None,)]
    def label(r):
        if r["cause"] in ("none", "no_event"):
            return r["cause"]
        return f"{r['cause']}:{'VIS' if r['vis'] else 'not'}:{r['why']}"
    def agg(rs):
        e = [r["ade_t0"] for r in rs if r["ade_t0"] is not None]
        b = [r["ade_best6"] for r in rs if r["ade_best6"] is not None]
        return len(rs), (np.mean(e) if e else float("nan")), (np.mean(b) if b else float("nan")), sum(e)
    tot_err = sum(r["ade_t0"] for r in rows if r["ade_t0"] is not None)
    summary = {}
    for grp in (BRAKING, ("start",)):
        name = "braking" if grp == BRAKING else "start"
        rs = [r for r in ev if r["bucket"] in grp]
        print(f"\n== {name} windows: {len(rs)}  (onset median {np.median([r['onset_s'] for r in rs if 'onset_s' in r]):.1f} s)")
        print(f"  {'cause : visible at t0 : why':42s} {'n':>4s} {'T0 ADE':>7s} {'best6':>6s} {'%allT0err':>9s}")
        by = defaultdict(list)
        for r in rs:
            by[label(r)].append(r)
        summary[name] = {}
        for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
            n_, e, b, s = agg(v)
            summary[name][k] = {"n": n_, "ade_t0": e, "ade_best6": b, "share_of_T0_err": s / tot_err}
            print(f"  {k:42s} {n_:4d} {e:7.2f} {b:6.2f} {100 * s / tot_err:8.1f}%")
        for vis in (True, False):
            v = [r for r in rs if r.get("vis") is vis]
            n_, e, b, s = agg(v)
            print(f"  {'-> ALL ' + ('VISIBLE' if vis else 'NOT visible / none'):42s} {n_:4d} {e:7.2f} {b:6.2f} {100 * s / tot_err:8.1f}%")
        v = [r for r in rs if r.get("vis") is None]
        if v:
            n_, e, b, s = agg(v)
            print(f"  {'-> no event found':42s} {n_:4d} {e:7.2f} {b:6.2f} {100 * s / tot_err:8.1f}%")
        # visible actor: already in path vs entering later (intent)
        va = [r for r in rs if r.get("cause") == "actor" and r.get("vis")]
        if va:
            for ip in (True, False):
                v = [r for r in va if r.get("in_path_at_t0") is ip]
                n_, e, b, s = agg(v)
                print(f"     visible actor {'already in path' if ip else 'outside path at t0':24s} {n_:4d} {e:7.2f} {b:6.2f}")
        # onset time
        print("  by onset:", "  ".join(
            f"{lo}-{hi}s n={len(v)} T0 {np.mean([r['ade_t0'] for r in v]):.2f} vis {np.mean([bool(r.get('vis')) for r in v]):.0%}"
            for lo, hi in ((0, 1), (1, 3), (3, 6.5))
            for v in [[r for r in rs if 'onset_s' in r and lo <= r['onset_s'] < hi and r['ade_t0'] is not None]] if v))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

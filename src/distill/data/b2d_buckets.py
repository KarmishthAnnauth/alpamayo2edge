"""Bucketed sampling of Bench2Drive training windows (phase 2, SFT run 5).

The flow head's remaining error is go/stop timing (PHASE2 memory, run 2-4:
moving->stop, standing->depart, hard brakes), while a uniform epoch spends a
quarter of its windows on an ego that never moves. SimLingo's recipe for the
same problem is to bucket the data by what the expert does and resample
(acceleration / deceleration buckets, one for starting from stop, steering
buckets, near-constant-speed samples thinned out). This is that, on the
quantities our cache already stores.

A window gets ONE bucket, the first that matches (rarest and hardest first):

    standing     v0 < 0.5 m/s and never above 0.5 over the 6.4 s
    start        v0 < 0.5 m/s, later above 2 m/s          (departure timing)
    hard_brake   mean accel over the first 2 s < -2.5 m/s^2
    to_stop      v0 > 1 m/s and standing at 6.4 s          (stop timing)
    brake        mean accel over the first 2 s < -1 m/s^2
    turn         lateral offset at 6.4 s > 3 m             (turns, lane changes)
    accel        mean accel over the first 2 s > +1 m/s^2
    cruise       everything else

`sampling_weights` turns target shares per bucket (`stage2.buckets.share`) into
per-window weights for a `WeightedRandomSampler`: an epoch keeps its length and
draws bucket b with probability share[b], uniformly inside the bucket. The
validation gate is NOT resampled - it stays the natural mix, so run 5's gate
compares with runs 1-4.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

BUCKETS = ("standing", "start", "hard_brake", "to_stop", "brake", "turn", "accel", "cruise")
DT = 0.1
BUCKET_FILE = "buckets_{split}.json"


def window_features(gt_future_xyz: np.ndarray, ego_speed_t0: float) -> dict:
    xyz = np.asarray(gt_future_xyz, dtype=np.float64)
    step = np.diff(np.vstack([np.zeros(3), xyz]), axis=0)[:, :2]
    speed = np.linalg.norm(step, axis=1) / DT                  # (64,) m/s at t0+0.1 .. t0+6.4
    v0 = float(ego_speed_t0)
    return dict(v0=v0, vmax=float(speed.max()), vend=float(speed[-1]),
                a2=float((speed[19] - v0) / 2.0), lat=float(abs(xyz[-1, 1])))


def bucket_of(f: dict) -> str:
    if f["v0"] < 0.5:
        if f["vmax"] < 0.5:
            return "standing"
        if f["vmax"] > 2.0:
            return "start"
    if f["a2"] < -2.5:
        return "hard_brake"
    if f["v0"] > 1.0 and f["vend"] < 0.5:
        return "to_stop"
    if f["a2"] < -1.0:
        return "brake"
    if f["lat"] > 3.0:
        return "turn"
    if f["a2"] > 1.0:
        return "accel"
    return "cruise"


def shard_key(path: Path) -> str:
    return f"{path.parent.name}/{int(path.stem)}"


def assign(shards: list[Path]) -> dict[str, str]:
    """`{clip/window: bucket}` for target shards (reads two arrays per shard)."""
    out = {}
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            out[shard_key(p)] = bucket_of(window_features(z["gt_future_xyz"], float(z["ego_speed_t0"])))
    return out


def load(cache_root: Path, split: str = "train") -> dict[str, str]:
    f = Path(cache_root) / BUCKET_FILE.format(split=split)
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing - run scripts/01e_b2d_buckets.py")
    return json.loads(f.read_text())["buckets"]


def sampling_weights(buckets: list[str], share: dict[str, float]) -> np.ndarray:
    """Per-window weight = share[bucket] / count[bucket], shares renormalised over
    the buckets that occur. An unknown or missing bucket name raises: a typo in
    the config would otherwise silently zero a bucket."""
    bad = set(share) - set(BUCKETS)
    if bad:
        raise ValueError(f"unknown bucket(s) in stage2.buckets.share: {sorted(bad)}")
    count = Counter(buckets)
    missing = [b for b in count if b not in share]
    if missing:
        raise ValueError(f"stage2.buckets.share has no entry for {sorted(missing)}")
    total = sum(float(share[b]) for b in count)
    return np.array([float(share[b]) / total / count[b] for b in buckets], dtype=np.float64)


def table(buckets: list[str], share: dict[str, float] | None = None) -> str:
    count, n = Counter(buckets), len(buckets)
    rows = [f"{'bucket':12s} {'windows':>8s} {'natural':>8s}" + (f" {'target':>8s} {'repeat':>7s}" if share else "")]
    total = sum(float(share[b]) for b in count) if share else 1.0
    for b in BUCKETS:
        c = count.get(b, 0)
        row = f"{b:12s} {c:8d} {100 * c / n:7.1f}%"
        if share and c:
            tgt = float(share[b]) / total
            row += f" {100 * tgt:7.1f}% {tgt / (c / n):6.2f}x"
        rows.append(row)
    return "\n".join(rows)

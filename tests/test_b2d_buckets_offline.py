"""Bucketed sampling (`data/b2d_buckets.py`), CPU only.

Pins: the bucket rules on synthetic windows (priority order included), that the
sampler draws each bucket with its configured share, and that a config typo or a
stale bucket table raises instead of silently reverting to uniform sampling.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.data import b2d_buckets as bk                 # noqa: E402

T = (np.arange(64) + 1) * 0.1


def _xyz(speed, lat_end=0.0):
    x = np.cumsum(np.asarray(speed) * 0.1)
    return np.stack([x, np.linspace(0, lat_end, 64), np.zeros(64)], axis=1)


def _bucket(speed, v0, lat_end=0.0):
    return bk.bucket_of(bk.window_features(_xyz(speed, lat_end), v0))


def test_bucket_rules():
    assert _bucket(np.zeros(64), 0.0) == "standing"
    assert _bucket(np.clip(1.5 * (T - 2.0), 0, None), 0.0) == "start"          # waits 2 s, departs
    assert _bucket(np.clip(8.0 - 3.5 * T, 0, None), 8.0) == "hard_brake"       # also stops: hard first
    assert _bucket(np.clip(6.0 - 1.5 * T, 0, None), 6.0) == "to_stop"
    assert _bucket(np.clip(10.0 - 1.2 * T, 5.0, None), 10.0) == "brake"        # slows, keeps moving
    assert _bucket(np.full(64, 5.0), 5.0, lat_end=6.0) == "turn"
    assert _bucket(3.0 + 1.5 * T, 3.0) == "accel"
    assert _bucket(np.full(64, 7.0), 7.0) == "cruise"
    # creeping from standstill (never above 2 m/s) is neither standing nor a start
    assert _bucket(np.full(64, 1.0), 0.0) == "cruise"


def test_sampling_weights_hit_the_shares():
    buckets = ["standing"] * 500 + ["start"] * 300 + ["hard_brake"] * 50 + ["cruise"] * 150
    share = {"standing": 0.05, "start": 0.35, "hard_brake": 0.30, "cruise": 0.30}
    w = bk.sampling_weights(buckets, share)
    assert abs(w.sum() - 1.0) < 1e-12
    rng = np.random.default_rng(0)
    draw = Counter(np.array(buckets)[rng.choice(len(buckets), size=40000, p=w)])
    for b, s in share.items():
        assert abs(draw[b] / 40000 - s) < 0.01
    # shares are renormalised over the buckets that occur
    w2 = bk.sampling_weights(["standing", "start", "start"], {"standing": 0.1, "start": 0.3, "turn": 0.6})
    np.testing.assert_allclose(w2, [0.25, 0.375, 0.375])
    with pytest.raises(ValueError):
        bk.sampling_weights(buckets, {**share, "crusie": 0.1})
    with pytest.raises(ValueError):
        bk.sampling_weights(buckets, {"standing": 1.0})


def test_trainer_sampler(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cosmos_framework")
    import json
    from distill import train_stage2 as ts2
    from distill.config import Cfg
    shards = [tmp_path / "clipA" / f"{i:02d}.npz" for i in range(4)]
    table = {bk.shard_key(p): b for p, b in zip(shards, ["standing", "standing", "start", "turn"])}
    (tmp_path / bk.BUCKET_FILE.format(split="train")).write_text(json.dumps({"buckets": table}))
    share = {"standing": 0.0, "start": 0.5, "turn": 0.5}
    cfg = Cfg({"paths": {"b2d_cache_root": str(tmp_path)},
               "stage2": {"dataset": "bench2drive", "buckets": {"enabled": True, "share": share}}})
    class _DS:
        def __init__(self, shards): self.shards = shards
        def __len__(self): return len(self.shards)
    ds = _DS(shards)
    sampler = ts2.build_bucket_sampler(cfg, ds)
    torch.manual_seed(0)
    assert set(list(sampler)) <= {2, 3} and len(sampler) == 4          # standing never drawn
    cfg.raw["stage2"]["buckets"]["enabled"] = False
    assert ts2.build_bucket_sampler(cfg, ds) is None
    cfg.raw["stage2"]["buckets"]["enabled"] = True
    ds.shards = shards + [tmp_path / "clipB" / "00.npz"]              # a window with no bucket
    with pytest.raises(RuntimeError):
        ts2.build_bucket_sampler(cfg, ds)

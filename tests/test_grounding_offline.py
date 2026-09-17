"""Grounding filter + route hint (D-043): synthetic futures, no GPU, no model."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.data.grounding import contradicted, grounded_shards, route_hint  # noqa: E402


def future(v0: float, v1: float, bearing_deg: float = 0.0, n: int = 64) -> np.ndarray:
    """A 6.4 s path at 10 Hz: speed ramps v0 -> v1, heading bends toward
    `bearing_deg` over the last third. Same construction as test_gt_reward_offline."""
    v = np.linspace(v0, v1, n)
    heads = np.zeros(n)
    k = n // 3
    heads[-k:] = np.linspace(0, np.radians(bearing_deg), k)
    d = np.stack([np.cos(heads), np.sin(heads)], 1) * (v / 10.0)[:, None]
    xy = np.cumsum(d, axis=0)
    return np.concatenate([xy, np.zeros((n, 1))], 1).astype(np.float32)


HOLD = future(10, 10)
STOP = future(10, 0)
BRAKE = future(12, 8)
SPEED = future(5, 10)
LEFT = future(8, 8, bearing_deg=70)
RIGHT = future(8, 8, bearing_deg=-70)


@pytest.mark.parametrize("text, xyz, expect", [
    # kept: claim held, or unverifiable
    ("Keep lane since the lane is clear", HOLD, False),
    ("Stop for the red traffic light since the signal is red", STOP, False),
    ("Turn left at the intersection since the route requires it", LEFT, False),
    ("Nudge to the left since there are cones ahead", HOLD, False),           # uncheckable
    ("Adapt speed to traffic since vehicles ahead are slowing", HOLD, False),  # soft -0.5
    ("", HOLD, False),                                                         # unparsed -0.5
    # dropped: flat contradictions
    ("Keep lane since the lane is clear", STOP, True),          # FOLLOW/KEEP while braking from speed
    ("Stop for the red traffic light since the signal is red", HOLD, True),
    ("Turn left at the intersection since the route requires it", RIGHT, True),   # opposite
    ("Turn left at the intersection since the route requires it", HOLD, True),    # no turn
    ("Accelerate to proceed since the road ahead is clear", STOP, True),
    ("Slow down since a pedestrian is crossing ahead", SPEED, True),
])
def test_contradicted(text, xyz, expect):
    assert contradicted(text, xyz) is expect


def test_route_hint_is_direction_only():
    assert route_hint(LEFT) == "Turn left ahead"
    assert route_hint(RIGHT) == "Turn right ahead"
    assert route_hint(STOP) == "Continue straight"    # a stop leaks nothing about speed


def test_grounded_shards_drops_only_contradicted(tmp_path):
    rows = [("Keep lane since the lane is clear", HOLD),
            ("Keep lane since the lane is clear", STOP),
            ("Stop for the red traffic light since the signal is red", STOP),
            ("Turn right at the intersection since the route requires it", LEFT)]
    shards = []
    for i, (t, xyz) in enumerate(rows):
        p = tmp_path / f"{i:02d}.npz"
        np.savez(p, coc_text=np.str_(t), gt_future_xyz=xyz)
        shards.append(p)
    kept, stats = grounded_shards(shards)
    assert [p.name for p in kept] == ["00.npz", "02.npz"]
    assert stats == {"total": 4, "kept": 2, "dropped": 2,
                     "dropped_by_maneuver": {"KEEP": 1, "TURN": 1}}

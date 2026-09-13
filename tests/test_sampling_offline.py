"""Maneuver-stratified sampling (D-036): weights from a synthetic cache, no GPU."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.data.sampling import UNASSIGNED, maneuver_weights, teacher_maneuver  # noqa: E402

# 8 FOLLOW, 2 LANE_CHANGE, 1 empty, 1 unparseable -> the shape of the real problem.
TRACES = (["Keep distance to the lead vehicle since it is ahead in our lane"] * 8
          + ["Lane change to the left due to the right lane ending"] * 2
          + [""] + ["Zorbulate the flux capacitor"])


@pytest.fixture
def shards(tmp_path):
    out = []
    for i, t in enumerate(TRACES):
        p = tmp_path / f"{i:02d}.npz"
        np.savez(p, coc_text=np.str_(t), traj_token_ids=np.zeros(4, np.int32))
        out.append(p)
    return out


def test_teacher_maneuver(shards):
    assert teacher_maneuver(shards[0]) == "FOLLOW"
    assert teacher_maneuver(shards[8]) == "LANE_CHANGE"
    assert teacher_maneuver(shards[10]) == UNASSIGNED     # empty
    assert teacher_maneuver(shards[11]) == UNASSIGNED     # unparseable


def test_alpha_zero_is_uniform(shards):
    w, mix = maneuver_weights(shards, alpha=0.0)
    assert w.shape == (12,)
    assert np.allclose(w.numpy(), 1.0)
    for c, (k, uniform, effective) in mix.items():
        assert math.isclose(uniform, effective)


def test_alpha_one_equalises_classes(shards):
    _, mix = maneuver_weights(shards, alpha=1.0)
    assert math.isclose(mix["FOLLOW"][2], mix["LANE_CHANGE"][2])


def test_alpha_half_matches_formula(shards):
    w, mix = maneuver_weights(shards, alpha=0.5)
    # Ratio between a LANE_CHANGE draw and a FOLLOW draw is sqrt(8/2) = 2.
    assert math.isclose(float(w[8] / w[0]), 2.0, rel_tol=1e-6)
    assert math.isclose(float(w.mean()), 1.0, rel_tol=1e-6)
    # Effective shares still sum to one and the rare class moved up, not to parity.
    assert math.isclose(sum(e for _, _, e in mix.values()), 1.0, rel_tol=1e-6)
    assert mix["LANE_CHANGE"][1] < mix["LANE_CHANGE"][2] < mix["FOLLOW"][2]


def test_unassigned_draw_at_uniform_rate(shards):
    w, mix = maneuver_weights(shards, alpha=0.5)
    # Mean of the assigned weights, so they are neither promoted as "rare" nor dropped.
    assigned = w[:10]
    assert math.isclose(float(w[10]), float(assigned.mean()), rel_tol=1e-6)
    assert math.isclose(float(w[11]), float(assigned.mean()), rel_tol=1e-6)
    assert mix[UNASSIGNED][0] == 2


def test_alpha_out_of_range(shards):
    with pytest.raises(ValueError):
        maneuver_weights(shards, alpha=1.5)

"""The parts of the headroom measurement that don't need a GPU.

The metric arithmetic and the guard rails are worth pinning: a gap reported on
mismatched windows, or an ADE averaged over the wrong axis, is worse than no
number at all because it looks authoritative.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.eval import gap  # noqa: E402
from distill.eval.open_loop import ade, min_ade  # noqa: E402


def test_ade_is_mean_displacement_over_the_horizon():
    gt = torch.zeros(1, 4, 3)
    pred = torch.zeros(1, 4, 3)
    pred[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])   # x offsets
    assert float(ade(pred, gt)) == pytest.approx(2.5)


def test_ade_ignores_z():
    gt = torch.zeros(1, 3, 3)
    pred = torch.zeros(1, 3, 3)
    pred[0, :, 2] = 100.0                                 # only z differs
    assert float(ade(pred, gt)) == pytest.approx(0.0)


def test_min_ade_takes_the_best_mode():
    gt = torch.zeros(1, 3, 3)
    pred = torch.zeros(1, 2, 3, 3)
    pred[0, 0, :, 0] = 5.0                                # bad mode
    pred[0, 1, :, 0] = 1.0                                # good mode
    assert float(min_ade(pred, gt)) == pytest.approx(1.0)


def test_av_actions_to_xyz_accumulates_deltas():
    """Frame-relative deltas must integrate; treating them as absolute is the
    failure the calibration phase exists to catch."""
    act = torch.zeros(4, 9)
    act[:, 0] = 1.0                                       # 1 m forward per step
    xyz = gap.av_actions_to_xyz(act)
    assert xyz[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_av_actions_to_xyz_ignores_the_rotation_half():
    act = torch.zeros(2, 9)
    act[:, 3:] = 99.0                                     # rot6d columns
    xyz = gap.av_actions_to_xyz(act)
    assert xyz.shape == (2, 3) and float(xyz.abs().sum()) == 0.0


def test_av_actions_to_xyz_rejects_too_few_dims():
    with pytest.raises(ValueError, match="expected >=3 action dims"):
        gap.av_actions_to_xyz(torch.zeros(4, 2))


def _blob(model, keys, offset):
    n, k, h = len(keys), 1, 4
    pred = np.zeros((n, k, h, 3), dtype=np.float32)
    pred[..., 0] = offset
    return {"pred": pred, "gt": np.zeros((n, h, 3), dtype=np.float32),
            "keys": keys, "model": model, "n_samples": k}


def test_report_refuses_mismatched_windows():
    """A gap across different clips is not a gap."""
    t = _blob("teacher", ["a/00", "b/00"], 1.0)
    e = _blob("edge", ["a/00", "c/00"], 3.0)
    with pytest.raises(RuntimeError, match="different windows"):
        gap.report(t, e)


def test_report_states_the_gap():
    t = _blob("teacher", ["a/00", "b/00"], 1.0)
    e = _blob("edge", ["a/00", "b/00"], 3.0)
    text = gap.report(t, e)
    assert "GAP +2.000 m" in text
    assert "66.7%" in text          # 2.0 / 3.0 of the zero-shot error


def test_score_truncates_to_the_shorter_horizon():
    b = _blob("m", ["a/00"], 1.0)
    b["gt"] = np.zeros((1, 2, 3), dtype=np.float32)       # gt shorter than pred
    assert gap.score(b)["ade"] == pytest.approx(1.0)


def test_save_load_round_trip(tmp_path):
    b = _blob("teacher", ["a/00", "b/01"], 2.0)
    gap.save(b, tmp_path / "teacher.npz")
    back = gap.load(tmp_path / "teacher.npz")
    assert back["keys"] == b["keys"]
    assert back["model"] == "teacher"
    np.testing.assert_allclose(back["pred"], b["pred"])

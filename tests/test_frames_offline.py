"""The student-input cache must survive a JPEG round trip with channels intact.

A silent RGB/BGR swap here trains the student on blue cars and is invisible
until someone looks at a decoded frame — long after a labeling run and a
training run. `frames_student` is RGB; OpenCV is BGR; both directions flip.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.data import frames  # noqa: E402


class _Window:
    def __init__(self, frames_student, data):
        self.frames_student = frames_student
        self.data = data


def _window(h=32, w=48, n_frames=2):
    rng = np.random.default_rng(0)
    fs = {
        "camera_front_wide_120fov": rng.integers(0, 255, (n_frames, h, w, 3), dtype=np.uint8),
        "camera_cross_left_120fov": rng.integers(0, 255, (n_frames, h, w, 3), dtype=np.uint8),
    }
    data = {"ego_history_xyz": torch.randn(1, 1, 16, 3),
            "ego_history_rot": torch.randn(1, 1, 16, 3, 3)}
    return _Window(fs, data)


def test_round_trip_preserves_shape_and_camera_keys(tmp_path):
    w = _window()
    frames.save_window_input(tmp_path / "00_input.npz", w)
    back = frames.load_window_input(tmp_path / "00_input.npz", "clip")
    assert set(back.frames_student) == set(w.frames_student)
    for cam, arr in w.frames_student.items():
        assert back.frames_student[cam].shape == arr.shape
        assert back.frames_student[cam].dtype == np.uint8


def test_round_trip_preserves_channel_order(tmp_path):
    """A pure-red frame must come back red, not blue."""
    red = np.zeros((1, 16, 16, 3), dtype=np.uint8)
    red[..., 0] = 255
    w = _Window({"camera_front_wide_120fov": red},
                {"ego_history_xyz": torch.zeros(1, 1, 4, 3),
                 "ego_history_rot": torch.zeros(1, 1, 4, 3, 3)})
    frames.save_window_input(tmp_path / "00_input.npz", w, quality=100)
    got = frames.load_window_input(tmp_path / "00_input.npz", "c").frames_student
    px = got["camera_front_wide_120fov"][0, 8, 8]
    assert px[0] > 200 and px[1] < 60 and px[2] < 60, f"channels swapped: {px}"


def test_frames_stay_in_capture_order(tmp_path):
    """Frame 0 must decode as frame 0 — keys are strings, so '10' must not sort
    before '2' once we go past 9 frames."""
    n = 12
    arr = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    for i in range(n):
        arr[i, :, :, 1] = i * 20          # green ramp encodes the index
    w = _Window({"camera_front_wide_120fov": arr},
                {"ego_history_xyz": torch.zeros(1, 1, 4, 3),
                 "ego_history_rot": torch.zeros(1, 1, 4, 3, 3)})
    frames.save_window_input(tmp_path / "00_input.npz", w, quality=100)
    got = frames.load_window_input(tmp_path / "00_input.npz", "c").frames_student
    greens = [int(got["camera_front_wide_120fov"][i, 4, 4, 1]) for i in range(n)]
    assert greens == sorted(greens), f"frame order scrambled: {greens}"


def test_ego_history_round_trips_as_tensors(tmp_path):
    w = _window()
    frames.save_window_input(tmp_path / "00_input.npz", w)
    back = frames.load_window_input(tmp_path / "00_input.npz", "clip")
    for key in ("ego_history_xyz", "ego_history_rot"):
        assert torch.is_tensor(back.data[key])
        torch.testing.assert_close(back.data[key], w.data[key], rtol=1e-6, atol=1e-6)


def test_cached_window_satisfies_the_context_builder_interface(tmp_path):
    """ContextBuilder only ever touches these two attributes."""
    w = _window()
    frames.save_window_input(tmp_path / "00_input.npz", w)
    back = frames.load_window_input(tmp_path / "00_input.npz", "clip")
    assert hasattr(back, "frames_student")
    assert "ego_history_xyz" in back.data and "ego_history_rot" in back.data

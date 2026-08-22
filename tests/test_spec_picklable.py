"""`traj_tokenizer_spec.pt` must survive torch.save.

`scripts/00_verify.py` — the FIRST thing that runs on the GPU box — writes the
spec with `torch.save`, and everything downstream reads it back. Anything in
that dict that pickle cannot handle kills step one of the pipeline. A closure
did exactly that once (D-029), hence this test.
"""
import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.teacher.wrapper import HistoryTokenize  # noqa: E402


class _StubTokenizer:
    """Stands in for DeltaTrajectoryTokenizer; module-level so it pickles."""
    vocab_size = 1000

    def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
        # Record the inversion the real call relies on: the history is passed
        # as the FUTURE argument, with one reference pose as the history.
        assert hist_xyz.shape[1] == 1, "reference pose must be a single frame"
        return torch.zeros(fut_xyz.shape[0], 48, dtype=torch.long)


def test_history_tokenize_is_picklable():
    fn = HistoryTokenize(_StubTokenizer())
    pickle.loads(pickle.dumps(fn))          # would raise on a closure


def test_history_tokenize_survives_torch_save(tmp_path):
    spec = {"hist_tokenize_fn": HistoryTokenize(_StubTokenizer()),
            "vocab_size": 3000, "hist_vocab_size": 1000, "total_bins": 4000}
    f = tmp_path / "traj_tokenizer_spec.pt"
    torch.save(spec, f)
    back = torch.load(f, weights_only=False)
    assert back["total_bins"] == 4000
    xyz = torch.zeros(1, 1, 16, 3)
    rot = torch.zeros(1, 1, 16, 3, 3)
    assert back["hist_tokenize_fn"](xyz, rot).shape == (1, 48)


def test_history_tokenize_rejects_wrong_rank():
    fn = HistoryTokenize(_StubTokenizer())
    with pytest.raises(AssertionError, match=r"\(B, n_traj, T, 3\)"):
        fn(torch.zeros(1, 16, 3), torch.zeros(1, 16, 3, 3))

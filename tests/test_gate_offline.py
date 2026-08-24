"""The stage-1 gate's arithmetic: which GT it scores against, and which decode
path it detokenizes through.

Neither failure mode here raises. `evaluate` scored ACTION-space GT as if it were
positions and returned a plausible float in fake metres; the student detokenized
without D-031's dim swap and produced trajectories ~20-128 m off, which reads as
"the student did not learn" rather than "the decoder is transposed".

torch-only, no GPU, no checkpoint: `detokenize_traj` is exercised as an unbound
method over a stub so nothing has to load 9 GB of weights.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.eval.open_loop import ade, evaluate, min_ade   # noqa: E402
from distill.teacher.wrapper import swap_action_dims        # noqa: E402


def _batch(**kw):
    b = {"gt_future_xyz": torch.zeros(2, 64, 3)}
    b.update(kw)
    return b


def test_evaluate_scores_against_positional_gt():
    gt = torch.randn(2, 64, 3)
    dl = [{"gt_future_xyz": gt, "gt_traj": torch.randn(2, 64, 2) * 100}]
    out = evaluate(lambda batch, k: batch["gt_future_xyz"][:, None].expand(2, k, 64, 3),
                   dl, k=3, device="cpu")
    assert out["minade"] == pytest.approx(0.0, abs=1e-5)
    assert out["n"] == 2


def test_evaluate_refuses_a_batch_without_positional_gt():
    """`gt_traj` is the teacher's action space, not a fallback reference."""
    dl = [{"gt_traj": torch.randn(2, 64, 2)}]
    with pytest.raises(KeyError):
        evaluate(lambda batch, k: torch.zeros(2, k, 64, 3), dl, k=2, device="cpu")


def test_evaluate_catches_a_horizon_mismatch():
    dl = [_batch()]
    with pytest.raises(ValueError):
        evaluate(lambda batch, k: torch.zeros(2, k, 32, 3), dl, k=2, device="cpu")


def test_min_ade_takes_the_best_mode():
    gt = torch.zeros(1, 4, 3)
    pred = torch.stack([torch.full((1, 4, 3), 5.0), torch.full((1, 4, 3), 0.5)], dim=1)
    assert min_ade(pred, gt).item() == pytest.approx(0.5 * (2 ** 0.5))
    assert ade(pred[:, 0], gt).item() == pytest.approx(5.0 * (2 ** 0.5))


class _StubStudent:
    """Just enough of EdgeStudent for the unbound-method call."""

    def __init__(self):
        self.seen = None

    def _traj_decode(self, hx, hr, tokens):
        self.seen = tokens.clone()
        return torch.zeros(tokens.shape[0], 64, 3), None, None


def test_student_detokenize_applies_the_dim_swap():
    """D-031: the student emits in the teacher's order, `decode` reads encode's."""
    from distill.student.edge_wrapper import EdgeStudent

    stub = _StubStudent()
    toks = torch.arange(128).reshape(1, 128)
    EdgeStudent.detokenize_traj(stub, toks, torch.zeros(1, 16, 3), torch.zeros(1, 16, 3, 3))
    assert torch.equal(stub.seen, swap_action_dims(toks))
    assert not torch.equal(stub.seen, toks)


def test_student_detokenize_returns_positions_shaped_like_gt():
    from distill.student.edge_wrapper import EdgeStudent

    stub = _StubStudent()
    out = EdgeStudent.detokenize_traj(stub, torch.arange(256).reshape(2, 128),
                                      torch.zeros(2, 16, 3), torch.zeros(2, 16, 3, 3))
    assert out.shape == (2, 64, 3)

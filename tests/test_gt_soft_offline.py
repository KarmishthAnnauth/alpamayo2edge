"""Run-4 trajectory supervision: the distance-aware GT anchor and the GT-CE ramp.

Three things here fail silently rather than loudly. The soft target has to stay
a proper distribution at the edges of the 3000-bin future region (otherwise it
leaks mass onto the history bins that sit immediately above it, and the loss
quietly stops being a cross-entropy). Its gradient has to reach the bins AROUND
the GT bin, not just the GT bin — that, and not any change in the loss value, is
what the term is for. And the weight ramp now runs UPWARD, which the old
`gt_ce_min` name implied it did not.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill import losses  # noqa: E402
from distill.config import Cfg  # noqa: E402

LO, HI = 100, 3100          # a stand-in future region: future_base .. +3000


def _logits(B, T, V, peak_bin, sharpness=12.0):
    """Logits peaked on `peak_bin` (broadcast over all positions)."""
    x = torch.full((B, T, V), -1e2)
    bins = torch.arange(V).float()
    x[:] = -sharpness * (bins - peak_bin).abs() / 10.0
    return x


def test_soft_target_prefers_a_near_miss_to_a_far_miss():
    V, gt = 3400, 1500
    mask = torch.ones(1, 1, dtype=torch.bool)
    tgt = torch.full((1, 1), gt)
    near = losses.gt_traj_soft_ce(_logits(1, 1, V, gt + 4), tgt, mask, 6.0, LO, HI)
    far = losses.gt_traj_soft_ce(_logits(1, 1, V, gt + 400), tgt, mask, 6.0, LO, HI)
    assert near < far


def test_soft_target_spreads_the_gradient_over_the_neighbourhood():
    """The mechanism, not the loss value.

    On a Gaussian student the two terms separate a near miss from a far miss by
    the SAME amount — smoothing the target does not amplify the value gap, and
    it was never supposed to. What it changes is where the gradient goes: the
    one-hot term pushes probability onto exactly one bin in 3000 per position,
    so a single GT sample per context says nothing about the local shape and the
    model has to average many samples to recover it. That is the variance job
    243 was paying (`gt_ce` flat at ~6.0 nats for 8 epochs). The soft target
    supplies the local shape on every sample.
    """
    V, gt = 3400, 1500
    mask = torch.ones(1, 1, dtype=torch.bool)
    tgt = torch.full((1, 1), gt)

    def pulled_bins(fn):
        x = _logits(1, 1, V, gt + 30).clone().requires_grad_(True)
        fn(x).backward()
        # Negative grad on a logit = the loss wants that bin's probability up.
        return int((x.grad[0, 0] < -1e-9).sum())

    soft = pulled_bins(lambda x: losses.gt_traj_soft_ce(x, tgt, mask, 6.0, LO, HI))
    hard = pulled_bins(lambda x: losses.gt_traj_ce(x, tgt, mask))
    assert hard == 1, "one-hot CE should pull exactly the GT bin"
    assert soft == 2 * int(round(3 * 6.0)) + 1 == 37


def test_soft_target_renormalises_at_the_region_edge():
    """A GT bin one off the bottom of the region: half the Gaussian's support is
    out of range, and the in-range half must still sum to 1 — i.e. the loss must
    match what the same student scores under an explicitly truncated target."""
    V, gt = 3400, LO + 1
    mask = torch.ones(1, 1, dtype=torch.bool)
    tgt = torch.full((1, 1), gt)
    logits = _logits(1, 1, V, gt)
    got = losses.gt_traj_soft_ce(logits, tgt, mask, 6.0, LO, HI)

    off = torch.arange(-18, 19)
    keep = (gt + off >= LO) & (gt + off < HI)
    q = torch.exp(-0.5 * (off.float() / 6.0) ** 2) * keep
    q = q / q.sum()
    logp = torch.log_softmax(logits[0, 0], -1)[gt + off]
    assert torch.allclose(got, -(q * logp).sum(), atol=1e-5)
    assert not keep.all(), "edge case did not actually truncate — test is vacuous"


def test_soft_target_degenerates_to_one_hot_as_sigma_shrinks():
    V, gt = 3400, 1500
    mask = torch.ones(1, 1, dtype=torch.bool)
    tgt = torch.full((1, 1), gt)
    logits = _logits(1, 1, V, gt + 3)
    soft = losses.gt_traj_soft_ce(logits, tgt, mask, 0.05, LO, HI)
    assert torch.allclose(soft, losses.gt_traj_ce(logits, tgt, mask), atol=1e-4)


def test_masked_positions_do_not_contribute():
    V, gt = 3400, 1500
    tgt = torch.full((1, 2), gt)
    logits = torch.stack([_logits(1, 1, V, gt)[0, 0],
                          _logits(1, 1, V, gt + 900)[0, 0]]).unsqueeze(0)
    both = torch.ones(1, 2, dtype=torch.bool)
    first = torch.tensor([[True, False]])
    assert losses.gt_traj_soft_ce(logits, tgt, first, 6.0, LO, HI) \
        < losses.gt_traj_soft_ce(logits, tgt, both, 6.0, LO, HI)


def test_gt_ce_weight_ramps_upward_and_reaches_its_endpoint():
    cfg = Cfg({"loss_weights": {"gt_ce": 0.5, "traj_kl": 1.0, "traj_kl_accel": 0.1},
                "gt_ce_end": 1.5})
    w0 = losses.stage_weights(cfg, 0, 100)["gt_ce"]
    wm = losses.stage_weights(cfg, 50, 100)["gt_ce"]
    w1 = losses.stage_weights(cfg, 100, 100)["gt_ce"]
    assert w0 == pytest.approx(0.5) and w1 == pytest.approx(1.5)
    assert wm == pytest.approx(1.0)
    # The per-dim teacher weights are static — nothing should be annealing them.
    for step in (0, 50, 100):
        w = losses.stage_weights(cfg, step, 100)
        assert w["traj_kl"] == 1.0 and w["traj_kl_accel"] == 0.1


def test_legacy_gt_ce_min_key_still_ramps_downward():
    """Runs 1-3 configs must keep reproducing. `gt_ce_min` is the old name."""
    cfg = Cfg({"loss_weights": {"gt_ce": 0.25}, "gt_ce_min": 0.06})
    assert losses.stage_weights(cfg, 0, 100)["gt_ce"] == pytest.approx(0.25)
    assert losses.stage_weights(cfg, 100, 100)["gt_ce"] == pytest.approx(0.06)


def test_split_reduction_matches_two_full_calls():
    """The per-position refactor must not change the numbers it replaced.

    Run 4 reduces one forward under two masks instead of calling `traj_topk_kl`
    twice; if those disagree, the split silently retunes both weights.
    """
    B, L, V, K = 2, 8, 60, 5
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    idx = torch.randint(0, V, (B, L, K))
    logp = torch.log_softmax(torch.randn(B, L, K), -1) + torch.log(torch.tensor(0.9))
    mask = torch.ones(B, L, dtype=torch.bool)
    dim0 = torch.zeros_like(mask); dim0[:, 0::2] = True

    per_pos = losses.traj_topk_kl_per_pos(logits, idx, logp)
    for m in (mask & dim0, mask & ~dim0, mask):
        assert torch.allclose(losses.masked_mean(per_pos, m),
                              losses.traj_topk_kl(logits, idx, logp, m), atol=1e-6)


def test_even_odd_masks_partition_the_trajectory_positions():
    """Every valid position lands in exactly one half — a parity slip would put
    the accel weight on curvature and leave half the stream unsupervised."""
    mask = torch.tensor([[True] * 6 + [False] * 2, [True] * 8])
    dim0 = torch.zeros_like(mask); dim0[:, 0::2] = True
    curv, acc = mask & dim0, mask & ~dim0
    assert not (curv & acc).any()                 # disjoint
    assert ((curv | acc) == mask).all()           # and covering
    assert int(curv.sum()) == 7 and int(acc.sum()) == 7
    assert curv[0, 0] and acc[0, 1]               # position 0 is curvature

"""The teacher emits future-trajectory dims in the opposite order to its own
tokenizer, and the cache has to pick one convention and hold it (D-031).

A1.5's 128-token future stream is 64 waypoints x 2 action dims, interleaved.
`DiscreteTrajectoryTokenizer.encode` builds it from
`UnicycleAccelCurvatureActionSpace` as (accel, curvature) per waypoint, and
`decode` reshapes back with the same convention — so the two are mutual
inverses and the quantization floor round-trips to ~0.01 m no matter what the
model does. The model itself emits (curvature, accel).

Nothing in the release catches this: A1.5 strips the future-fusion path, so no
shipped code path ever feeds a model-emitted future token back through `decode`.
On our side it is worse than invisible — it is *plausible*. Decoding the emitted
stream untouched produced ADE 20-128 m against a 0.13 m floor, which reads like
"the discrete head is untrained" (a FAIL verdict on D-022, and a design change)
rather than "the two dims are the wrong way round" (one `flip`). Swapping first
gives 0.62-5.31 m, level with the expert's own 0.57-3.37 m.

The half of this that no probe would ever catch: `gt_traj_token_ids` comes out of
`encode`, so it lands in the OPPOSITE order to `traj_token_ids` and
`traj_topk_*`. Cached that way, stage 1's `gt_ce` anchor would train the student
toward the per-waypoint transpose of what `traj_kl` distils — two supervisions
pulling against each other, on a term deliberately weighted low (0.25) so it
would degrade the run without breaking it. Hence a test.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.teacher.wrapper import swap_action_dims  # noqa: E402


def test_swaps_within_waypoints_not_across():
    """[w0d0 w0d1 w1d0 w1d1 ...] -> [w0d1 w0d0 w1d1 w1d0 ...]."""
    t = torch.arange(8)
    assert swap_action_dims(t).tolist() == [1, 0, 3, 2, 5, 4, 7, 6]


def test_is_its_own_inverse():
    """One helper for both directions, so emission->encode and encode->emission
    cannot drift apart."""
    t = torch.randint(0, 3000, (128,))
    assert torch.equal(swap_action_dims(swap_action_dims(t)), t)


def test_is_not_the_dim_major_transpose():
    """The other candidate layout, and the one the diagnostic ruled out: reading
    the stream as [all accel, then all curvature]. It scored 9.97-77 m where the
    swap scored 0.62-5.31, so pin that these are genuinely different operations
    and a future edit cannot quietly substitute one for the other."""
    t = torch.arange(8)
    transposed = t.reshape(2, -1).t().reshape(-1)
    assert transposed.tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    assert not torch.equal(swap_action_dims(t), transposed)


def test_preserves_shape_and_batches_over_leading_dims():
    t = torch.randint(0, 3000, (4, 128))
    out = swap_action_dims(t)
    assert out.shape == t.shape
    for row in range(4):
        assert torch.equal(out[row], swap_action_dims(t[row]))


def test_dim_identity_is_actually_exchanged():
    """The property the ADE numbers turn on: whichever dim was pinned becomes the
    other one. Mirrors the real signature — on a straight clip the emitted
    stream's FIRST dim sits at bin ~1500 (zero curvature) while `encode`'s first
    dim is the one that varies."""
    waypoints = 64
    pinned = torch.full((waypoints,), 1500)
    varying = torch.arange(waypoints) + 1400
    encode_order = torch.stack([varying, pinned], dim=-1).reshape(-1)

    emitted = swap_action_dims(encode_order)
    assert torch.equal(emitted.reshape(-1, 2)[:, 0], pinned)
    assert torch.equal(emitted.reshape(-1, 2)[:, 1], varying)

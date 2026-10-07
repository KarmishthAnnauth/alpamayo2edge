"""CPU checks for student/flow_path.py: the context crop and the samplers.

The gen pathway itself needs the real MoT layers (GPU smoke, scripts/03d);
what can be pinned offline is the contract around it.
"""
import math
import types

import torch

from distill.student import flow_path as fp


def test_context_key_mask_drops_future_bins_and_after():
    am = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    traj = torch.zeros_like(am)
    traj[0, 3:5] = True          # bins at 3,4; then future_end at 5 (also dropped)
    traj[1, 6:8] = True
    m = fp.context_key_mask(am, traj)
    assert m[0].tolist() == [True, True, True, False, False, False, False, False]
    assert m[1].tolist() == [True] * 6 + [False, False]


def test_context_key_mask_without_span_keeps_prompt():
    am = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    assert fp.context_key_mask(am, None).tolist() == [[True, True, True, False]]
    assert fp.context_key_mask(am, torch.zeros_like(am)).tolist() == [[True, True, True, False]]


class _LinearFieldStudent:
    """flow_forward returning the exact straight-line field u = a1 - a0 for the
    interpolant a_t = t a1 + (1-t) a0: Euler then lands on a1 in one step or ten."""
    n_action_tokens = 64
    raw_action_dim = 2

    def __init__(self, a1, x0):
        self.a1, self.x0 = a1, x0

    def flow_forward(self, a_t, t, ctx, owner):
        t = t.view(-1, 1, 1)
        a0 = (a_t - t * self.a1) / (1 - t).clamp_min(1e-3)
        return self.a1 - a0


def _ctx(n):
    return fp.FlowContext(keys=[], values=[], key_mask=torch.ones(n, 3, dtype=torch.bool),
                          next_pos=torch.zeros(n, dtype=torch.long))


def test_ode_sampler_reaches_data_on_a_linear_field():
    torch.manual_seed(0)
    a1 = torch.randn(3, 64, 2)
    x0 = torch.randn(3, 64, 2)
    st = _LinearFieldStudent(a1, x0)
    out = fp.sample_actions(st, _ctx(3), torch.arange(3), steps=10, x0=x0)
    assert torch.allclose(out["actions"], a1, atol=1e-4)
    assert out["logp"].shape == (3,) and float(out["logp"].abs().sum()) == 0.0


def test_sde_sampler_returns_finite_logp_and_stays_near_data():
    torch.manual_seed(0)
    a1 = torch.zeros(4, 64, 2)
    x0 = torch.randn(4, 64, 2)
    st = _LinearFieldStudent(a1, x0)
    g = torch.Generator().manual_seed(1)
    out = fp.sample_actions(st, _ctx(4), torch.arange(4), steps=10, x0=x0,
                            sde_noise=0.3, generator=g, return_trace=True)
    assert torch.isfinite(out["logp"]).all() and out["logp"].shape == (4,)
    assert len(out["trace"]) == 11
    # the final step is deterministic: the last trace move equals dt * u exactly
    # only for the ODE; here just check the endpoint is much closer to a1 than x0
    assert float((out["actions"] - a1).abs().mean()) < 0.5 * float((x0 - a1).abs().mean())


def test_sde_sampler_seed_reproducible():
    a1 = torch.zeros(2, 64, 2); x0 = torch.randn(2, 64, 2)
    st = _LinearFieldStudent(a1, x0)
    o1 = fp.sample_actions(st, _ctx(2), torch.arange(2), steps=6, x0=x0, sde_noise=0.2,
                           generator=torch.Generator().manual_seed(7))
    o2 = fp.sample_actions(st, _ctx(2), torch.arange(2), steps=6, x0=x0, sde_noise=0.2,
                           generator=torch.Generator().manual_seed(7))
    assert torch.equal(o1["actions"], o2["actions"]) and torch.equal(o1["logp"], o2["logp"])

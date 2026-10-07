"""The SDE sampler behind DiffGRPO, pinned offline (CPU, no model).

Two things have to be true for the policy gradient to mean anything:

  1. the SDE has the ODE's marginals (Flow-GRPO's construction), so sampling
     with noise a > 0 explores the same distribution the head was trained to
     produce - checked on a Gaussian toy where the exact velocity field is
     known in closed form and the marginal variance V(t) can be tracked step
     by step;
  2. the density that `step_logprob` differentiates is the density the sample
     was drawn from - checked by recomputing a recorded chain's per-step
     log-probs with the same weights and demanding equality.

The suspicion recorded in P2-07 (denominator 2 tau vs 2 (1-tau)) was settled
2026-09-29: the implemented drift a^2/(2(1-tau)) IS Flow-GRPO's sigma^2/(2 tau)
with sigma = a sqrt(tau/(1-tau)); the alternative breaks test 1 immediately.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.student import flow_path as fp    # noqa: E402

MU, S = 1.5, 0.5     # data ~ N(MU, S^2), noise ~ N(0, 1), teacher convention a_t = t a1 + (1-t) a0


def _u_exact(x, t):
    """E[a1 - a0 | a_t = x] for the Gaussian toy (closed form)."""
    V = t * t * S * S + (1 - t) ** 2
    return MU + (t * S * S - (1 - t)) / V * (x - t * MU)


def _V(t):
    return t * t * S * S + (1 - t) ** 2


def test_sde_step_preserves_the_marginal_on_a_gaussian_toy():
    torch.manual_seed(0)
    n, steps, a, t_start = 300_000, 100, 0.7, 0.2
    a1 = MU + S * torch.randn(n)
    a0 = torch.randn(n)
    x = t_start * a1 + (1 - t_start) * a0                  # the exact marginal at t_start
    ts = torch.linspace(t_start, 1.0, steps + 1)
    worst = 0.0
    for i in range(steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        u = _u_exact(x, t0)
        if i < steps - 1:
            mean, std = fp.sde_step(x, u, t0, t1 - t0, a)
            x = mean + std * torch.randn(n)
        else:
            x = x + (t1 - t0) * u
        worst = max(worst, abs(float(x.var()) / _V(t1) - 1.0))
    assert worst < 0.03, f"variance departs from the marginal by {worst:.1%}"
    assert abs(float(x.mean()) - MU) < 0.01 and abs(float(x.var()) - S * S) < 0.01


def test_the_rejected_drift_does_not_preserve_the_marginal():
    """Guards the derivation: with 2*tau in the denominator (a^2 confused with
    sigma^2) the variance is 40% off at t = 1. If this ever passes, somebody
    changed the toy, not the sampler."""
    torch.manual_seed(0)
    n, steps, a, t_start = 200_000, 100, 0.7, 0.2
    a1 = MU + S * torch.randn(n); a0 = torch.randn(n)
    x = t_start * a1 + (1 - t_start) * a0
    ts = torch.linspace(t_start, 1.0, steps + 1)
    for i in range(steps):
        t0, t1 = float(ts[i]), float(ts[i + 1]); dt = t1 - t0
        u = _u_exact(x, t0)
        if i < steps - 1:
            tau = 1 - t0; one_m = 1 - tau; v = -u
            drift = v + a * a / (2 * tau) * (x + one_m * v)
            std = a * math.sqrt(tau / one_m) * math.sqrt(dt)
            x = x - drift * dt + std * torch.randn(n)
        else:
            x = x + dt * u
    assert abs(float(x.var()) / (S * S) - 1.0) > 0.2


def test_sde_from_pure_noise_lands_near_the_data_marginal():
    """From t = 0 the first step uses the floored noise scale; the Langevin
    part of the drift pulls the marginal back within a few percent."""
    torch.manual_seed(1)
    n, steps, a = 300_000, 50, 0.7
    x = torch.randn(n)
    ts = torch.linspace(0.0, 1.0, steps + 1)
    for i in range(steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        u = _u_exact(x, t0)
        if i < steps - 1:
            mean, std = fp.sde_step(x, u, t0, t1 - t0, a)
            x = mean + std * torch.randn(n)
        else:
            x = x + (t1 - t0) * u
    assert abs(float(x.mean()) - MU) < 0.02
    assert abs(float(x.var()) / (S * S) - 1.0) < 0.2


class _GaussianFieldStudent:
    """flow_forward = the toy's exact field applied per action entry."""
    n_action_tokens = 64
    raw_action_dim = 2

    def flow_forward(self, a_t, t, ctx, owner, grad_checkpoint=None):
        return _u_exact(a_t, t.view(-1, 1, 1))


def _ctx(n):
    return fp.FlowContext(keys=[], values=[], key_mask=torch.ones(n, 3, dtype=torch.bool),
                          next_pos=torch.zeros(n, dtype=torch.long))


def test_chain_logprobs_reproduce_the_sampling_densities():
    st = _GaussianFieldStudent()
    g = torch.Generator().manual_seed(3)
    out = fp.sample_actions(st, _ctx(5), torch.arange(5), steps=8, sde_noise=0.5,
                            generator=g, return_trace=True)
    assert out["logp_steps"].shape == (5, 8) and len(out["trace"]) == 9
    assert torch.all(out["logp_steps"][:, -1] == 0)          # final step deterministic
    assert torch.allclose(out["logp"], out["logp_steps"].sum(1))
    re = fp.chain_logprobs(st, _ctx(5), torch.arange(5), out["trace"], sde_noise=0.5)
    assert torch.allclose(re, out["logp_steps"], atol=1e-5)
    lp, mean, std = fp.step_logprob(st, _ctx(5), torch.arange(5), out["trace"], 2, 0.5)
    assert torch.allclose(lp, out["logp_steps"][:, 2], atol=1e-5) and std > 0
    assert mean.shape == (5, 64, 2)


def test_step_logprob_rejects_the_deterministic_step():
    st = _GaussianFieldStudent()
    out = fp.sample_actions(st, _ctx(2), torch.arange(2), steps=4, sde_noise=0.5, return_trace=True)
    try:
        fp.step_logprob(st, _ctx(2), torch.arange(2), out["trace"], 3, 0.5)
    except ValueError:
        return
    raise AssertionError("step 3 of a 4-step chain carries no noise")


def test_ode_sampler_has_zero_densities_and_the_data_endpoint():
    st = _GaussianFieldStudent()
    out = fp.sample_actions(st, _ctx(200), torch.arange(200), steps=50, return_trace=False)
    assert float(out["logp_steps"].abs().sum()) == 0.0 and float(out["logp"].abs().sum()) == 0.0
    assert abs(float(out["actions"].mean()) - MU) < 0.1
    assert fp.noisy_steps(10) == list(range(9))

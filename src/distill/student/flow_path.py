"""Phase 2: the student's flow head as an ACTION-ONLY gen pathway, plus samplers.

Cosmos 3 Edge has no separate action head. Actions are one more generation
modality of the MoT gen tower (`*_moe_gen` weights): each waypoint is a token
that enters through `action2llm`, runs the gen pathway of every decoder layer
while attending the reasoner's per-layer K/V, and leaves through `llm2action`
as a velocity. The framework only ships that path packed together with video
latents (`Cosmos3VFMNetworkConfig`: "We do NOT support action only training!")
and samples `[vision | action]` as one state - so this module is our own dense
[N, 64, hidden] implementation of the gen pathway, written against the same
sub-modules the framework's `MoTDecoderLayer.forward(gen_only=True)` uses:

    input_layernorm_moe_gen -> q/k/v_proj_moe_gen -> q/k_norm_moe_gen -> RoPE
    -> attention over [reasoner K/V (normalised for gen) | action K/V]
    -> o_proj_moe_gen -> residual -> post_attention_layernorm_moe_gen
    -> mlp_moe_gen -> residual                      (unified_mot.py:1161-1319)

The reasoner K/V are what the framework's own AR inference feeds the gen
pathway from its `MemoryState`: `k_norm_und_for_gen(k_norm(k_proj(h)))` with
RoPE, and `v_proj(h)`, taken at each layer's post-`input_layernorm` input
(unified_mot.py:660-699, 724-747). They come from one frozen `reasoner_forward`
per window and are reused across that window's K flow samples.

CONTEXT = prompt + ego history + route + CoC + `<|traj_future_start|>`, and
NOT the future trajectory tokens: the teacher's expert masks those out
(alpamayo1_5.py:201-204, the KV crop in sft_alpamayo_r1.py:166-168), and
conditioning the flow head on the token path would make it a refiner of that
path's 3.8 m ADE instead of a planner (D-034). `context_key_mask` does the crop.

Conventions (D-016): the trainers and samplers here speak the TEACHER's
convention - `a_t = t*a1 + (1-t)*a0`, a0 noise, a1 data, t = data weight,
u* = a1 - a0. `EdgeStudent.flow_forward` converts to the student's
(sigma = 1-t, v = -u) at the boundary, so nothing in this file sees the flip.
"""
from __future__ import annotations
import contextlib
import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

log = logging.getLogger(__name__)


@dataclass
class FlowContext:
    """Frozen reasoner K/V the action tokens attend, one window per row."""
    keys: list[torch.Tensor]      # per layer: (B, L, H_kv, D), RoPE'd, gen-normalised
    values: list[torch.Tensor]    # per layer: (B, L, H_kv, D)
    key_mask: torch.Tensor        # (B, L) bool, True = a real context token
    next_pos: torch.Tensor        # (B,) long: RoPE position of the first action token
    final_hidden: torch.Tensor | None = None   # (B, L, hidden) reasoner output

    @property
    def batch_size(self) -> int:
        return int(self.key_mask.shape[0])

    def detach_(self) -> "FlowContext":
        self.keys = [k.detach() for k in self.keys]
        self.values = [v.detach() for v in self.values]
        return self


def context_key_mask(attention_mask: torch.Tensor,
                     traj_pos: torch.Tensor | None) -> torch.Tensor:
    """Which context positions the action tokens may attend.

    Everything real up to and including `<|traj_future_start|>`; nothing from the
    first future trajectory bin onwards (the bins, `<|traj_future_end|>`, eos).
    Rows without a trajectory span (generation contexts) keep the whole prompt.
    """
    mask = attention_mask.bool().clone()
    if traj_pos is None:
        return mask
    traj_pos = traj_pos.bool()
    has = traj_pos.any(dim=1)
    if not bool(has.any()):
        return mask
    first = torch.where(has, traj_pos.float().argmax(dim=1),
                        torch.full_like(has, mask.shape[1], dtype=torch.long))
    pos = torch.arange(mask.shape[1], device=mask.device)[None, :]
    return mask & (pos < first[:, None])


@contextlib.contextmanager
def _capture_attn_inputs(layers):
    """Record `(h_normed, cos, sin)` entering each layer's attention on the
    reasoner path. `MoTDecoderLayer.reasoner_forward` calls
    `self.self_attn.reasoner_forward(...)` directly, so like
    `EdgeStudent._capture_layers` we shadow the bound method on the instance."""
    out: dict[int, tuple] = {}
    patched = []
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        original = attn.reasoner_forward
        had_own = "reasoner_forward" in attn.__dict__

        def _wrap(idx, fn):
            def inner(hidden_states, cos, sin, cache, layer_idx):
                out[idx] = (hidden_states, cos, sin)
                return fn(hidden_states, cos, sin, cache, layer_idx)
            return inner

        attn.reasoner_forward = _wrap(i, original)
        patched.append((attn, original, had_own))
    try:
        yield out
    finally:
        for attn, original, had_own in patched:
            if had_own:
                attn.reasoner_forward = original
            else:
                del attn.__dict__["reasoner_forward"]


def build_flow_context(student, batch, keep_final_hidden: bool = False) -> FlowContext:
    """One reasoner pass per window -> per-layer K/V for the gen pathway.

    Frozen (no graph) unless `student.train_ar_context` is set by
    `param_groups_stage2` (AR-tower LoRA): then the reasoner activations stay
    alive so the flow loss reaches the adapters through the K/V. No activation
    checkpointing exists on that path (stage1.grad_checkpoint), so this is the
    memory the micro-batch has to pay for.

    Memory: 28 layers x (K + V) x L x 8 heads x 128 = 28 x 2 x 2048 x L values per
    window, i.e. the same size as the per-layer hidden states; trimmed to the
    longest kept context in the batch.
    """
    grad = torch.is_grad_enabled() and bool(getattr(student, "train_ar_context", False))
    with torch.set_grad_enabled(grad):
        return _build_flow_context(student, batch, keep_final_hidden)


def _build_flow_context(student, batch, keep_final_hidden: bool) -> FlowContext:
    layers = student._decoder_layers()
    fwd, _ = student._reasoner_inputs(batch)
    with _capture_attn_inputs(layers) as cap:
        h_final = student.lm.model.reasoner_forward(cache=None, **fwd)
    if len(cap) != len(layers):
        raise RuntimeError(f"captured {len(cap)}/{len(layers)} layers - the reasoner "
                           "path no longer routes through self_attn.reasoner_forward")

    key_mask = context_key_mask(batch["attention_mask"], batch.get("traj_pos"))
    n_keep = key_mask.sum(dim=1)                         # (B,)
    L = int(n_keep.max())
    key_mask = key_mask[:, :L]

    # RoPE position of the first action token: right after the last kept
    # context token. mrope collapses to one value on text positions; take the
    # max over the three axes to be safe when the last token is not text.
    B = key_mask.shape[0]
    pos_ids = fwd.get("position_ids")
    last = (n_keep - 1).clamp_min(0)
    if pos_ids is None:
        next_pos = n_keep.clone()
    else:
        if pos_ids.dim() == 2:                           # (B, T)
            pos_ids = pos_ids[None]
        idx = last.view(1, B, 1).expand(pos_ids.shape[0], B, 1)
        next_pos = pos_ids.gather(2, idx).amax(dim=0).squeeze(-1) + 1
    next_pos = next_pos.to(torch.long)

    keys, values = [], []
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        h, cos, sin = cap[i]
        h, cos, sin = h[:, :L], cos[:, :L], sin[:, :L]
        Bh, T, _ = h.shape
        k = attn.k_proj(h).view(Bh, T, attn.num_key_value_heads, attn.head_dim)
        k = attn.k_norm(k)                               # Identity on Nemotron (qk_norm_for_text=False)
        if getattr(attn, "k_norm_und_for_gen", None) is not None:
            k = attn.k_norm_und_for_gen(k)               # RMSNorm BEFORE RoPE (unified_mot.py:686-692)
        _, k = attn._apply_rotary_pos_emb(k, k, cos, sin, unsqueeze_dim=2)
        v = attn.v_proj(h).view(Bh, T, attn.num_key_value_heads, attn.head_dim)
        keys.append(k.contiguous())
        values.append(v.contiguous())
    return FlowContext(keys=keys, values=values, key_mask=key_mask, next_pos=next_pos,
                       final_hidden=h_final if keep_final_hidden else None)


def _gen_layer(layer, x, cos, sin, ctx_k, ctx_v, ctx_mask, k_per_window):
    """One decoder layer, gen pathway only, dense [N, T, hidden] layout.

    The K flow samples of a window share its context K/V WITHOUT copying it
    (an index_select per sample cost 10 GB per forward on the first smoke): the
    queries are regrouped per window, (B, k*T), scored against the window's
    context keys once, and against the action keys with a block-diagonal mask
    so a sample only sees its own 64 tokens. One softmax over both blocks.
    """
    attn = layer.self_attn
    N, T, _ = x.shape
    B = N // k_per_window
    kT = k_per_window * T
    H, Hkv, D = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    L = ctx_k.shape[1]
    g = H // Hkv

    h = layer.input_layernorm_moe_gen(x)
    q = attn.q_norm_moe_gen(attn.q_proj_moe_gen(h).view(N, T, H, D))
    k = attn.k_norm_moe_gen(attn.k_proj_moe_gen(h).view(N, T, Hkv, D))
    v = attn.v_proj_moe_gen(h).view(N, T, Hkv, D)
    q, k = attn._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    dt = q.dtype

    # (B, H, kT, D) queries; GQA by repeating each kv head g times on the keys.
    q = q.reshape(B, kT, H, D).transpose(1, 2)
    Kc = ctx_k.to(dt).repeat_interleave(g, dim=2).transpose(1, 2)        # (B, H, L, D)
    Vc = ctx_v.to(dt).repeat_interleave(g, dim=2).transpose(1, 2)        # (B, H, L, D)
    Ks = k.reshape(B, kT, Hkv, D).repeat_interleave(g, dim=2).transpose(1, 2)   # (B, H, kT, D)
    Vs = v.reshape(B, kT, Hkv, D).repeat_interleave(g, dim=2).transpose(1, 2)

    s_ctx = torch.matmul(q, Kc.transpose(-1, -2)) * attn.scaling          # (B, H, kT, L)
    s_self = torch.matmul(q, Ks.transpose(-1, -2)) * attn.scaling         # (B, H, kT, kT)
    neg = torch.finfo(s_ctx.dtype).min
    s_ctx = s_ctx.masked_fill(~ctx_mask[:, None, None, :], neg)
    same = (torch.arange(kT, device=x.device) // T)
    block = same[:, None] == same[None, :]                                # (kT, kT)
    s_self = s_self.masked_fill(~block[None, None], neg)
    p = torch.softmax(torch.cat([s_ctx, s_self], dim=-1).float(), dim=-1).to(dt)
    out = torch.matmul(p[..., :L], Vc) + torch.matmul(p[..., L:], Vs)     # (B, H, kT, D)
    out = out.transpose(1, 2).reshape(N, T, H * D)
    x = x + attn.o_proj_moe_gen(out)
    h = layer.post_attention_layernorm_moe_gen(x)
    return x + layer.mlp_moe_gen(h)


def _regular_owner(owner: torch.Tensor, B: int) -> int:
    """`owner` must be `arange(B).repeat_interleave(k)`; returns k."""
    N = int(owner.shape[0])
    if N % B:
        raise ValueError(f"{N} samples do not split evenly over {B} windows")
    k = N // B
    expect = torch.arange(B, device=owner.device).repeat_interleave(k)
    if not torch.equal(owner.to(expect.dtype), expect):
        raise ValueError("owner must group each window's samples contiguously "
                         "(arange(B).repeat_interleave(k))")
    return k


def gen_pathway_forward(student, x: torch.Tensor, ctx: FlowContext,
                        owner: torch.Tensor, grad_checkpoint: bool = False) -> torch.Tensor:
    """Run the gen pathway over action-token embeddings `x` (N, T, hidden).

    `owner` (N,) maps each action sequence to its window in `ctx`, grouped per
    window, so the K flow samples of a window share one reasoner pass and one
    copy of its K/V. Returns the pre-`norm_moe_gen` hidden states;
    `EdgeStudent.flow_forward` applies the final norm + `llm2action`.
    """
    N, T, _ = x.shape
    owner = owner.to(ctx.key_mask.device)
    k = _regular_owner(owner, ctx.batch_size)
    pos = ctx.next_pos[owner][:, None] + torch.arange(T, device=x.device)[None, :]  # (N, T)
    cos, sin = student.lm.model.rotary_emb(x, position_ids=pos)                   # (N, T, D)
    use_ckpt = grad_checkpoint and torch.is_grad_enabled()
    for i, layer in enumerate(student._decoder_layers()):
        args = (layer, x, cos, sin, ctx.keys[i], ctx.values[i], ctx.key_mask, k)
        x = _ckpt(_gen_layer, *args, use_reentrant=False) if use_ckpt else _gen_layer(*args)
    return x


# ---------------------------------------------------------------- samplers --

def sde_schedule(steps: int, device=None) -> torch.Tensor:
    """The sampler's time grid, teacher convention: t = linspace(0, 1, steps+1)."""
    return torch.linspace(0.0, 1.0, steps + 1, device=device)


def sde_step(x: torch.Tensor, u: torch.Tensor, t0: float, dt: float, sde_noise: float,
             first_step_floor: float = 1e-2) -> tuple[torch.Tensor, float]:
    """Mean and std of ONE Euler-Maruyama step of Flow-GRPO's SDE, teacher convention.

    Flow-GRPO (Liu et al. 2025, eq. for the rectified-flow SDE with the same
    marginals as the ODE), in their noise-weight time tau with velocity v = dx/dtau:

        dx = [v + sigma^2/(2 tau) (x + (1-tau) v)] dtau + sigma dW,
        sigma = a * sqrt(tau / (1-tau)),

    integrated tau: 1 -> 0. Substituting sigma^2 gives the form implemented here,
    a^2 / (2 (1-tau)) * (x + (1-tau) v), and our tau = 1 - t, v = -u, dtau = -dt.
    VERIFIED 2026-09-29 (tests/test_flow_sde_offline.py): with the exact velocity
    field of a Gaussian toy the per-step variance tracks the true marginal
    (0.638/0.6381, 0.395/0.392, 0.2475/0.246, 0.2436/0.246 over the grid); the
    earlier suspicion that the denominator should be 2 tau was a confusion
    between a and sigma. (1-tau) is floored at `first_step_floor` on the first
    step, where the marginal-preserving noise scale sqrt(tau/(1-tau)) is
    singular; sampling and `step_logprob` share this function, so the policy
    that is sampled is exactly the policy whose density is differentiated.
    """
    tau = 1.0 - t0
    one_m = max(1.0 - tau, first_step_floor)
    v = -u
    drift = v + (sde_noise ** 2) / (2.0 * one_m) * (x + one_m * v)
    std = sde_noise * math.sqrt(tau / one_m) * math.sqrt(dt)
    return x - drift * dt, std                                  # dtau = -dt


def _gauss_logp(mean: torch.Tensor, std: float, x: torch.Tensor) -> torch.Tensor:
    """Per-sample log-density of `x` under N(mean, std^2), MEAN over the 64x2
    action dims (ReCogDrive / Flow-GRPO convention for the policy ratio)."""
    return torch.distributions.Normal(mean, std).log_prob(x).flatten(1).mean(1)


@torch.no_grad()
def sample_actions(student, ctx: FlowContext, owner: torch.Tensor, steps: int = 10,
                   temperature: float = 1.0, sde_noise: float = 0.0,
                   generator: torch.Generator | None = None, x0: torch.Tensor | None = None,
                   return_trace: bool = False, first_step_floor: float = 1e-2) -> dict:
    """Draw one action sample (64 x 2, teacher action space) per `owner` entry.

    ODE (sde_noise = 0): the teacher's own sampler, byte-for-byte in structure
    (flow_matching.py:171-191): `t = linspace(0, 1, steps+1)`, Euler
    `x += dt * u(x, t)`, init `randn * temperature`. 10 steps like A1.5.

    SDE (sde_noise = a > 0): Flow-GRPO's ODE->SDE conversion (`sde_step`), the
    same marginals as the ODE, and each step a Gaussian whose log-density of
    the step actually taken is what DiffGRPO needs. The final step to t = 1 is
    deterministic (no noise into the sample itself).

    Returns `actions` (n, 64, 2), `logp_steps` (n, steps) - per-step per-dim-mean
    log-densities, 0 at deterministic steps - `logp` = their sum, `x0`, and
    `trace` (steps+1 states, the chain) when requested.
    """
    n = int(owner.shape[0])
    dev = ctx.key_mask.device
    if x0 is None:
        x0 = torch.randn(n, student.n_action_tokens, student.raw_action_dim,
                         device=dev, generator=generator) * temperature
    x = x0.float()
    ts = sde_schedule(steps, dev)
    logp_steps = torch.zeros(n, steps, device=dev)
    trace = [x.clone()] if return_trace else None
    for i in range(steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        dt = t1 - t0
        t_vec = torch.full((n,), t0, device=dev)
        u = student.flow_forward(x, t_vec, ctx, owner).float()       # teacher convention
        if sde_noise > 0.0 and i < steps - 1:
            mean, std = sde_step(x, u, t0, dt, sde_noise, first_step_floor)
            eps = torch.randn(x.shape, device=dev, generator=generator)
            x_new = mean + std * eps
            logp_steps[:, i] = _gauss_logp(mean, std, x_new)
            x = x_new
        else:
            x = x + dt * u
        if return_trace:
            trace.append(x.clone())
    return {"actions": x, "logp": logp_steps.sum(1), "logp_steps": logp_steps,
            "x0": x0, "trace": trace}


def noisy_steps(steps: int) -> list[int]:
    """Indices of the chain steps that carry noise (and therefore a density)."""
    return list(range(steps - 1))


def step_logprob(student, ctx: FlowContext, owner: torch.Tensor, trace: list[torch.Tensor],
                 i: int, sde_noise: float, first_step_floor: float = 1e-2,
                 grad_checkpoint: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Log-density of chain step i (trace[i] -> trace[i+1]) under the CURRENT
    weights, with gradient. Returns (logp (n,), mean, std) so a caller can also
    form a KL against another policy's mean at the same state."""
    steps = len(trace) - 1
    if not 0 <= i < steps - 1:
        raise ValueError(f"step {i} carries no noise on a {steps}-step chain")
    ts = sde_schedule(steps, trace[0].device)
    t0, t1 = float(ts[i]), float(ts[i + 1])
    x, x_next = trace[i], trace[i + 1]
    t_vec = torch.full((x.shape[0],), t0, device=x.device)
    if grad_checkpoint is None:
        u = student.flow_forward(x, t_vec, ctx, owner).float()
    else:
        u = student.flow_forward(x, t_vec, ctx, owner, grad_checkpoint=grad_checkpoint).float()
    mean, std = sde_step(x, u, t0, t1 - t0, sde_noise, first_step_floor)
    return _gauss_logp(mean, std, x_next), mean, std


def chain_logprobs(student, ctx: FlowContext, owner: torch.Tensor, trace: list[torch.Tensor],
                   sde_noise: float, first_step_floor: float = 1e-2) -> torch.Tensor:
    """(n, steps) per-step log-densities of a recorded chain; deterministic steps 0.
    Equals `sample_actions(...)["logp_steps"]` when the weights are unchanged."""
    steps = len(trace) - 1
    out = torch.zeros(trace[0].shape[0], steps, device=trace[0].device)
    for i in noisy_steps(steps):
        out[:, i] = step_logprob(student, ctx, owner, trace, i, sde_noise, first_step_floor)[0]
    return out

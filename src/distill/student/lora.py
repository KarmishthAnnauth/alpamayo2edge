"""LoRA plumbing for the Cosmos 3 Edge student (D-024).

A thin adapter over cosmos-framework's own LoRA path
(`cosmos_framework/utils/generator/lora.py`) — the route NVIDIA's Edge recipe
uses for the generation tower, so we inherit its checkpoint-key contract:
`LoraInjectedLinear` SUBCLASSES `nn.Linear` and keeps the pretrained weight at
`<path>.weight`, adding only `<path>.lora_A.weight` / `<path>.lora_B.weight`.

Verified against the local `../cosmos-framework` clone (read 2026-08-22):

- `OmniMoTModel.add_lora(network, lora_rank=, lora_alpha=, lora_target_modules=)`
  DOES exist (omni_mot_model.py:5537) — it is a two-line forwarder to
  `inject_lora_pre_fsdp`. TRAINING_STRATEGY.md §6 says it does not; what is
  actually true is that the OLD `_attach_lora` call signature
  (`add_lora(rank=, alpha=)`, no network arg) does not match it, so that call
  would still have failed on the GPU box. We call the free function directly:
  it takes the subtree we want to scope injection to, which the method's
  `self`-bound form obscures.

- Injection SCOPE matters. `inject_lora_pre_fsdp` matches plain targets by leaf
  NAME over the whole subtree it is handed. The reasoner's SigLIP2 vision tower
  is lazily attached at `language_model.visual` on the first vision prompt
  (unified_mot.py `_ensure_vision_tower`) and its attention leaves are named
  `q_proj` / `k_proj` / `v_proj` (+ `out_proj`) — the SAME leaf names as the AR
  tower's. Passing the causal LM as the root would therefore adapt the vision
  encoder too. We scope every injection to `language_model.model.layers`, which
  holds both towers' decoder weights and nothing else. (The `_moe_gen` suffix
  keeps the two towers apart WITHIN that subtree — D-015.)

- Adapters are constructed on META device by `LoraInjectedLinear.__init__`
  regardless of where the base weight lives, because it is written for the
  pre-FSDP meta-device flow. Our load path (`from_pretrained_dcp(...).to(dev)`)
  is already materialized, so we `to_empty()` the two adapter SUBMODULES — never
  the wrapper, which would blow away the pretrained base weight — before
  `init_lora_weights_post_materialization`.

The masking and merge helpers below duck-type `LoraInjectedLinear` (via the
`lora_A`/`lora_B` attributes) so this module imports without cosmos-framework
installed; only injection/init need the real package.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Targets by stage (D-024 / TRAINING_STRATEGY §2). Leaf names carrying the
# `_moe_gen` suffix are unique to the generation tower; the bare names are the
# reasoner/AR tower. MLP targets are the documented escalation if attention-only
# underfits, and MUST be path-qualified (`mlp_moe_gen.up_proj`) because
# `up_proj`/`down_proj` are shared leaf names across both towers.
AR_ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
GEN_ATTN_TARGETS = ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]


def _is_lora_linear(module: nn.Module) -> bool:
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


def lora_modules(root: nn.Module):
    for path, mod in root.named_modules():
        if _is_lora_linear(mod):
            yield path, mod


def lora_parameters(root: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in root.named_parameters() if "lora_" in n]


def inject(root: nn.Module, targets: list[str], rank: int, alpha: int,
           device: torch.device | str) -> int:
    """Inject + materialize + initialize LoRA adapters under `root`.

    `root` must be the narrowest subtree that contains the targets (see the
    module docstring on scoping). Returns the number of wrapped Linears.

    NOTE: `inject_lora_pre_fsdp` freezes every non-LoRA parameter IN THIS
    SUBTREE as a side effect. Parameters outside `root` keep whatever
    requires_grad they had, so callers must freeze the full model first and
    re-enable the full-rank new parameters afterwards.
    """
    from cosmos_framework.utils.generator.lora import (
        init_lora_weights_post_materialization,
        inject_lora_pre_fsdp,
    )

    # `LoraInjectedLinear` subclasses nn.Linear, so the framework's injector
    # would happily wrap an already-wrapped module and NEST adapters. Refuse.
    already = sum(1 for _ in lora_modules(root))
    if already:
        raise RuntimeError(
            f"{already} LoRA modules already present under this root — injecting "
            "again would nest adapters. Use one EdgeStudent per stage.")
    inject_lora_pre_fsdp(
        root,
        lora_rank=int(rank),
        lora_alpha=int(alpha),
        lora_target_modules=",".join(targets),
    )
    wrapped = sum(1 for _ in lora_modules(root))
    if wrapped == 0:
        raise RuntimeError(
            f"LoRA injection wrapped 0 modules for targets={targets}. The "
            "framework only warns; for us this silently means 'nothing is "
            "trainable', so it is fatal here.")

    # Adapters land on meta (see docstring). Materialize the two adapter
    # submodules only — `root.to_empty(...)` would discard pretrained weights.
    for _, mod in lora_modules(root):
        for adapter in (mod.lora_A, mod.lora_B):
            if adapter.weight.is_meta:
                adapter.to_empty(device=device)
    init_lora_weights_post_materialization(root)
    return wrapped


def describe(root: nn.Module) -> dict:
    """Sanity summary to print once on the GPU box before a long run."""
    paths = [p for p, _ in lora_modules(root)]
    n_params = sum(p.numel() for p in lora_parameters(root))
    leaves: dict[str, int] = {}
    for p in paths:
        leaves[p.split(".")[-1]] = leaves.get(p.split(".")[-1], 0) + 1
    return {"wrapped": len(paths), "by_leaf": leaves, "lora_params": n_params,
            "example": paths[:3]}


# ---------------- merging ----------------
#
# Stage-1 checkpoints MUST NOT carry lora_A/lora_B keys: stage 2 reloads them
# into a plain (un-injected) student, and the framework's own loader has no
# alias for adapter keys. Because `LoraInjectedLinear` preserves `<path>.weight`,
# folding is a pure weight edit — no key renames.

@torch.no_grad()
def merged_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    """LoRA-folded CPU state dict. Does NOT mutate `root`.

    This is the per-epoch save path: merging in place would end training (the
    adapters the optimizer holds would be gone and the base weights moved).
    """
    sd = root.state_dict()
    out = {k: v for k, v in sd.items() if ".lora_A." not in k and ".lora_B." not in k}
    for path, mod in lora_modules(root):
        key = f"{path}.weight" if path else "weight"
        base = sd[key]
        # fp32 for the outer product: rank-r bf16 accumulation is lossy enough
        # to show up as a stage1->stage2 reload discrepancy.
        delta = (mod.lora_B.weight.float() @ mod.lora_A.weight.float()) * mod._lora_scale
        out[key] = (base.float() + delta).to(base.dtype)
    return {k: v.detach().to("cpu") for k, v in out.items()}


@torch.no_grad()
def merge_lora_(root: nn.Module) -> int:
    """Fold adapters into the base weights IN PLACE and drop the wrappers.

    Terminal operation — for export, not mid-training. Returns modules merged.
    """
    merged = 0
    for parent_name, parent in list(root.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not _is_lora_linear(child):
                continue
            delta = (child.lora_B.weight.float() @ child.lora_A.weight.float()) * child._lora_scale
            child.weight.data.add_(delta.to(child.weight.dtype))
            # Rebuild as a plain Linear on meta so no second copy of the weight
            # is ever allocated, then re-point at the existing parameters.
            plain = nn.Linear(child.in_features, child.out_features,
                              bias=child.bias is not None, device="meta")
            plain.weight = child.weight
            if child.bias is not None:
                plain.bias = child.bias
            setattr(parent, child_name, plain)
            merged += 1
    return merged


# ---------------- gradient masking ----------------
#
# `inject` freezes everything and we re-enable a few full-rank tensors by hand.
# Those tensors are ROW-INDEXED and shared with pretrained behaviour, so an
# unmasked gradient is exactly the forgetting channel LoRA was adopted to close
# (TRAINING_STRATEGY §2): embedding/lm_head rows below the appended trajectory
# range, and the 31 embodiment rows of the DomainAwareLinear action head that
# are not ours.

def _row_mask_hook(param: nn.Parameter, keep: torch.Tensor):
    """Zero the gradient of every row where `keep` is False. Returns a handle.

    `keep` is a 1-D bool tensor over dim 0 of `param`. The hook mutates the
    gradient in place: it fires on the temporary pre-accumulation gradient for a
    leaf, so nothing else observes that tensor, and cloning a
    (131072, 2048) embedding gradient every micro-step is not free.
    """
    if not param.requires_grad:
        raise RuntimeError("enable requires_grad before registering a row mask")
    mask = keep.to(device=param.device, dtype=param.dtype).view(-1, *([1] * (param.dim() - 1)))
    return param.register_hook(lambda grad: grad.mul_(mask))


def mask_rows_below_(param: nn.Parameter, first_trainable_row: int):
    """Train only rows >= `first_trainable_row` (appended vocabulary)."""
    keep = torch.zeros(param.shape[0], dtype=torch.bool)
    keep[first_trainable_row:] = True
    return _row_mask_hook(param, keep)


def mask_rows_except_(param: nn.Parameter, row: int):
    """Train only `row` (our embodiment domain in the action head)."""
    keep = torch.zeros(param.shape[0], dtype=torch.bool)
    keep[row] = True
    return _row_mask_hook(param, keep)

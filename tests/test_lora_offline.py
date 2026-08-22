"""Offline checks for the LoRA plumbing (D-024). torch only — no GPU, no
cosmos-framework, no checkpoints. Run before touching the GPU box:

    python -m pytest tests/test_lora_offline.py -q

What this actually buys: `merged_state_dict` and `merge_lora_` reimplement the
adapter's forward as a weight edit, and a wrong scale or a transposed product
there produces a checkpoint that loads cleanly and silently trains from the
wrong weights in stage 2. These tests pin the equivalence numerically.

`_StubLoraLinear` mirrors cosmos-framework's `LoraInjectedLinear`
(utils/generator/lora.py) exactly where our helpers touch it: it subclasses
nn.Linear, keeps the base weight at `.weight`, exposes `lora_A`/`lora_B`
siblings and a `_lora_scale` of alpha/rank, and forwards as
`base(x) + scale * B(A(x))`. Our helpers duck-type on `lora_A`/`lora_B`, so the
stub exercises the real code paths.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.student import lora  # noqa: E402


class _StubLoraLinear(nn.Linear):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self._lora_rank, self._lora_alpha = rank, float(alpha)

    @property
    def _lora_scale(self):
        return self._lora_alpha / self._lora_rank

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias) + \
            self._lora_scale * self.lora_B(self.lora_A(x))


class _Tiny(nn.Module):
    """Two 'layers', one adapted, one not — so grouping/scoping is exercised."""
    def __init__(self, rank=4, alpha=8):
        super().__init__()
        self.q_proj = _StubLoraLinear(nn.Linear(16, 16), rank, alpha)
        self.mlp = nn.Linear(16, 16)
        # lora_B is zero-init in the framework, which would make every merge test
        # trivially pass; train-like weights instead.
        nn.init.normal_(self.q_proj.lora_A.weight, std=0.05)
        nn.init.normal_(self.q_proj.lora_B.weight, std=0.05)

    def forward(self, x):
        return self.mlp(self.q_proj(x))


def test_merged_state_dict_matches_adapter_forward():
    m = _Tiny().eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        y_lora = m(x)

    sd = lora.merged_state_dict(m)
    assert not [k for k in sd if "lora_" in k], "adapter keys survived the merge"

    plain = nn.Module()
    plain.q_proj = nn.Linear(16, 16)
    plain.mlp = nn.Linear(16, 16)
    plain.load_state_dict(sd)
    with torch.no_grad():
        y_merged = plain.mlp(plain.q_proj(x))
    torch.testing.assert_close(y_lora, y_merged, rtol=1e-5, atol=1e-5)


def test_merged_state_dict_does_not_mutate():
    m = _Tiny().eval()
    before = m.q_proj.weight.detach().clone()
    lora.merged_state_dict(m)
    torch.testing.assert_close(m.q_proj.weight, before)
    assert lora._is_lora_linear(m.q_proj), "merge must not strip the live adapters"


def test_merge_in_place_preserves_output_and_drops_wrappers():
    m = _Tiny().eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        y_before = m(x)
    assert lora.merge_lora_(m) == 1
    assert not lora._is_lora_linear(m.q_proj)
    assert type(m.q_proj) is nn.Linear
    with torch.no_grad():
        torch.testing.assert_close(y_before, m(x), rtol=1e-5, atol=1e-5)
    assert not [k for k in m.state_dict() if "lora_" in k]


def test_scale_is_alpha_over_rank():
    """A merge that ignored alpha/rank would still pass a scale-1 test."""
    m = _Tiny(rank=4, alpha=32).eval()
    base = m.q_proj.weight.detach().clone()
    delta = lora.merged_state_dict(m)["q_proj.weight"] - base
    expect = (m.q_proj.lora_B.weight @ m.q_proj.lora_A.weight) * (32 / 4)
    torch.testing.assert_close(delta, expect, rtol=1e-5, atol=1e-5)


def test_mask_rows_below_blocks_pretrained_rows():
    emb = nn.Embedding(10, 4)
    emb.weight.requires_grad_(True)
    lora.mask_rows_below_(emb.weight, 7)          # only rows 7..9 are ours
    emb(torch.arange(10)).sum().backward()
    assert emb.weight.grad[:7].abs().sum() == 0
    assert emb.weight.grad[7:].abs().sum() > 0


def test_mask_rows_except_keeps_one_embodiment():
    fc = nn.Embedding(32, 8)                      # DomainAwareLinear.fc shape
    fc.weight.requires_grad_(True)
    lora.mask_rows_except_(fc.weight, 31)
    fc(torch.arange(32)).sum().backward()
    g = fc.weight.grad
    assert g[:31].abs().sum() == 0
    assert g[31].abs().sum() > 0


def test_mask_survives_gradient_accumulation():
    """The hook mutates the grad in place; accumulation must still be correct."""
    emb = nn.Embedding(6, 3)
    emb.weight.requires_grad_(True)
    lora.mask_rows_below_(emb.weight, 4)
    for _ in range(3):
        emb(torch.arange(6)).sum().backward()
    assert emb.weight.grad[:4].abs().sum() == 0
    torch.testing.assert_close(emb.weight.grad[4:], torch.full((2, 3), 3.0))


def test_mask_requires_grad_enabled_first():
    p = nn.Parameter(torch.zeros(4, 2), requires_grad=False)
    with pytest.raises(RuntimeError):
        lora.mask_rows_below_(p, 2)

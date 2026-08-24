"""Layer mapping and feature pooling — the two halves of stage 1's `feat` loss.

Both were broken in ways that produce no error at all:

* the map was one entry per STUDENT layer (28) onto 8 teacher layers, and
  `FeatureProjections.forward` keys its output by teacher layer, so 20 of the 28
  projections were silently discarded and trained on nothing;
* the student's per-token hidden states were compared against the cache's
  8 pooled segments, which is a shape error at step 1 — the only loud one here.

torch-only, no GPU, no checkpoint.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.student import layer_map as LM  # noqa: E402

FEAT_LAYERS = [3, 8, 12, 17, 21, 26, 30, 35]     # configs/default.yaml
N_STUDENT = 28                                   # Edge's reasoner depth


def test_uniform_map_is_injective_and_covers_every_cached_layer():
    m = LM.uniform_map(FEAT_LAYERS, N_STUDENT)
    assert sorted(m.values()) == FEAT_LAYERS
    assert len(set(m.values())) == len(m) == len(FEAT_LAYERS)
    assert all(0 <= s < N_STUDENT for s in m)


def test_uniform_map_preserves_depth_order():
    m = LM.uniform_map(FEAT_LAYERS, N_STUDENT)
    students = [s for s, _ in sorted(m.items(), key=lambda kv: kv[1])]
    assert students == sorted(students)


def test_uniform_map_refuses_more_teacher_layers_than_student_layers():
    with pytest.raises(ValueError):
        LM.uniform_map(list(range(36)), N_STUDENT)


def test_cka_map_recovers_a_planted_correspondence():
    """Student layer 3*i is a linear image of teacher layer i, plus noise."""
    torch.manual_seed(0)
    n, d_t, d_s = 64, 16, 24
    teacher = {l: torch.randn(n, d_t) for l in FEAT_LAYERS}
    student = {i: torch.randn(n, d_s) for i in range(N_STUDENT)}
    for k, l in enumerate(FEAT_LAYERS):
        w = torch.randn(d_t, d_s)
        student[3 * k] = teacher[l] @ w + 0.01 * torch.randn(n, d_s)
    m = LM.cka_map(student, teacher)
    assert m == {3 * k: l for k, l in enumerate(FEAT_LAYERS)}


def test_cka_map_is_injective_and_monotone_on_noise():
    torch.manual_seed(1)
    teacher = {l: torch.randn(32, 16) for l in FEAT_LAYERS}
    student = {i: torch.randn(32, 24) for i in range(N_STUDENT)}
    m = LM.cka_map(student, teacher)
    assert sorted(m.values()) == FEAT_LAYERS
    assert len(set(m)) == len(FEAT_LAYERS)
    pairs = sorted(m.items(), key=lambda kv: kv[1])
    assert [s for s, _ in pairs] == sorted(s for s, _ in pairs)


def test_projections_reject_a_non_injective_map():
    with pytest.raises(ValueError):
        LM.FeatureProjections({0: 3, 1: 3, 2: 8}, d_student=8, d_teacher=16)


def test_pooling_matches_the_teacher_chunk_semantics():
    """The cache pooled `torch.chunk(h, 8, dim=0)` over the teacher's prefill."""
    torch.manual_seed(0)
    L, D, pool = 101, 4, 8
    h = torch.randn(1, L, D)
    n_prompt = torch.tensor([L])
    got = LM.pool_prompt_segments(h, n_prompt, pool)
    want = torch.stack([c.mean(0) for c in torch.chunk(h[0], pool, dim=0)])
    assert got.shape == (1, pool, D)
    assert torch.allclose(got[0], want)


def test_pooling_ignores_padding_and_the_teacher_forced_answer():
    """Only `[0, n_prompt)` is pooled — the teacher never saw the rest."""
    torch.manual_seed(0)
    h = torch.randn(2, 200, 4)
    n_prompt = torch.tensor([120, 96])
    got = LM.pool_prompt_segments(h, n_prompt, 8)
    h2 = h.clone()
    h2[0, 120:] = 999.0        # answer + padding
    h2[1, 96:] = -999.0
    assert torch.allclose(got, LM.pool_prompt_segments(h2, n_prompt, 8))


def test_pooled_student_and_cached_teacher_are_comparable_shapes():
    """The shape contract stage 1 broke: (B, pool, D_t) on both sides."""
    B, pool, d_s, d_t = 3, 8, 24, 16
    hidden = {0: torch.randn(B, 150, d_s)}
    n_prompt = torch.tensor([150, 130, 90])
    proj = LM.FeatureProjections({0: 3}, d_student=d_s, d_teacher=d_t)
    out = proj({k: LM.pool_prompt_segments(v, n_prompt, pool)
                for k, v in hidden.items()})
    teacher_cached = torch.randn(B, pool, d_t)      # exactly what save_shard writes
    assert out[3].shape == teacher_cached.shape


def test_pooling_refuses_a_prompt_shorter_than_the_segment_count():
    with pytest.raises(ValueError):
        LM.pool_prompt_segments(torch.randn(1, 10, 4), torch.tensor([3]), 8)

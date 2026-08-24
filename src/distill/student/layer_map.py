"""Teacher->student layer correspondence (plan Step 2.2).

Restricted to the teacher layers the action expert actually attends to
(TeacherWrapper.probe_expert_conditioning), since those are the features
stage 1 must shape for stage 2 to consume.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def uniform_map(teacher_layers: list[int], n_student_layers: int) -> dict[int, int]:
    """student_layer -> teacher_layer, ONE student layer per teacher layer.

    The map must be INJECTIVE, and it must be built over the teacher layers the
    cache actually holds (`teacher.feat_layers`), not over every layer the expert
    attends to. Both were wrong until 2026-08-24, and both failed silently:

    * The old version returned one entry per STUDENT layer — 28 entries onto 8
      teacher layers. `FeatureProjections.forward` keys its output by teacher
      layer, so 20 of the 28 projections were overwritten in the dict, trained
      nothing, and which student layer fed each teacher layer was decided by
      iteration order.
    * It was called with `expert_layers["attended_layers"]`, which is all 36
      layers (the expert attends every one, D-004). The cache holds 8. The
      intersection `feature_match` then took was whatever the two happened to
      share.

    Student layers are spread evenly across the stack in the same depth order as
    the teacher layers, so relative depth is preserved.
    """
    ts = sorted(teacher_layers)
    if len(ts) > n_student_layers:
        raise ValueError(
            f"{len(ts)} cached teacher layers but only {n_student_layers} student "
            "layers — the map cannot be injective; drop entries from "
            "teacher.feat_layers")
    out = {}
    for i, t in enumerate(ts):
        s_idx = round(i * (n_student_layers - 1) / max(len(ts) - 1, 1))
        out[int(s_idx)] = int(t)
    return out


def pool_prompt_segments(hidden: torch.Tensor, n_prompt: torch.Tensor,
                         pool_len: int = 8) -> torch.Tensor:
    """(B, L, D) reasoner states -> (B, pool_len, D), pooled like the teacher.

    The cache stores `(feat_pool_len, D_t)` per layer, NOT per-token features:
    `label_window` mean-pools its PREFILL hidden states into `pool_len` equal
    chunks (`teacher/wrapper.py`, `torch.chunk(h, pool_len, dim=0)`). The student
    side had no matching reduction, so `feature_match` compared (B, L, D_t)
    against (B, 8, D_t) and died on the first step of stage 1.

    Pooling is over `[0, n_prompt)` per sample — the student's prompt region, the
    counterpart of the teacher's prefill — which also drops the right-padding
    (`collate_student`) and the teacher-forced answer, neither of which the
    teacher's own features saw.

    `torch.chunk` semantics are mirrored deliberately rather than reimplemented
    with a reshape: chunk sizes are `ceil(L / pool_len)`, so an even-split
    version would silently pool different token ranges than the cache did.
    """
    B, L, D = hidden.shape
    out = hidden.new_zeros(B, pool_len, D)
    for b in range(B):
        n = int(n_prompt[b]) if n_prompt is not None else L
        n = max(1, min(n, L))
        segs = torch.chunk(hidden[b, :n], pool_len, dim=0)
        if len(segs) != pool_len:
            raise ValueError(
                f"prompt of {n} tokens cannot be split into {pool_len} segments "
                "the way the teacher's prefill was — the cache and the student "
                "would be pooling different things")
        out[b] = torch.stack([s.mean(0) for s in segs])
    return out


@torch.no_grad()
def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between (N, Dx) and (N, Dy) feature matrices."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = (x.T @ y).norm() ** 2
    return float(xty / (((x.T @ x).norm()) * ((y.T @ y).norm()) + 1e-8))


@torch.no_grad()
def cka_map(student_feats: dict[int, torch.Tensor],
            teacher_feats: dict[int, torch.Tensor]) -> dict[int, int]:
    """Greedy monotone assignment maximizing CKA (D-008's primary mapping).

    Iterates over the TEACHER layers — the cache holds 8 of them and the student
    has 28, so the teacher side is what must be covered exactly once. Returns
    `student_layer -> teacher_layer`, injective, in the same shape `uniform_map`
    returns and `FeatureProjections` consumes.

    Monotone by construction: each teacher layer's partner must lie deeper than
    the previous one's, which forbids depth inversions and leaves enough student
    layers for the teacher layers still to come. Feats are `layer -> (N, D)`
    pooled features over the SAME probe samples on both sides.
    """
    s_layers, t_layers = sorted(student_feats), sorted(teacher_feats)
    if len(t_layers) > len(s_layers):
        raise ValueError(f"{len(t_layers)} teacher layers vs {len(s_layers)} "
                         "student layers — no injective monotone map exists")
    out: dict[int, int] = {}
    lo = 0
    for n, tj in enumerate(t_layers):
        # Leave one student layer for each teacher layer after this one.
        hi = len(s_layers) - (len(t_layers) - n - 1)
        best, best_s = -1.0, s_layers[lo]
        for k in range(lo, hi):
            score = cka(student_feats[s_layers[k]], teacher_feats[tj])
            if score > best:
                best, best_s = score, s_layers[k]
        out[int(best_s)] = int(tj)
        lo = s_layers.index(best_s) + 1
    return out


class FeatureProjections(nn.Module):
    """One linear projection per mapped pair: student dim -> teacher dim.
    Trained jointly in stage 1; frozen afterwards."""

    def __init__(self, layer_map: dict[int, int], d_student: int, d_teacher: int):
        super().__init__()
        if len(set(layer_map.values())) != len(layer_map):
            raise ValueError(
                f"layer map is not injective ({layer_map}) — forward() keys its "
                "output by teacher layer, so duplicate targets silently discard "
                "projections and train them on nothing")
        self.layer_map = layer_map
        self.proj = nn.ModuleDict({
            str(s): nn.Linear(d_student, d_teacher, bias=False) for s in layer_map
        })

    def forward(self, student_hidden: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Returns teacher_layer -> projected student features."""
        return {self.layer_map[s]: self.proj[str(s)](h)
                for s, h in student_hidden.items() if s in self.layer_map}

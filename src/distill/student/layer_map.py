"""Teacher->student layer correspondence (plan Step 2.2).

Restricted to the teacher layers the action expert actually attends to
(TeacherWrapper.probe_expert_conditioning), since those are the features
stage 1 must shape for stage 2 to consume.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def uniform_map(teacher_layers: list[int], n_student_layers: int) -> dict[int, int]:
    """student_layer -> teacher_layer, evenly spread over the attended set."""
    ts = sorted(teacher_layers)
    out = {}
    for i in range(n_student_layers):
        j = round(i * (len(ts) - 1) / max(n_student_layers - 1, 1))
        out[i] = ts[j]
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
    """Greedy monotone assignment maximizing CKA, preserving depth order.
    Feats: layer -> (N, D) pooled features over the same probe samples."""
    s_layers, t_layers = sorted(student_feats), sorted(teacher_feats)
    out, t_min = {}, 0
    for si in s_layers:
        best, best_t = -1.0, t_layers[t_min]
        for tj in t_layers:
            if tj < best_t and tj < t_layers[t_min]:
                continue
            score = cka(student_feats[si], teacher_feats[tj])
            if score > best and tj >= t_layers[t_min]:
                best, best_t = score, tj
        out[si] = best_t
        t_min = t_layers.index(best_t)  # monotonicity: no depth inversions
    return out


class FeatureProjections(nn.Module):
    """One linear projection per mapped pair: student dim -> teacher dim.
    Trained jointly in stage 1; frozen afterwards."""

    def __init__(self, layer_map: dict[int, int], d_student: int, d_teacher: int):
        super().__init__()
        self.layer_map = layer_map
        self.proj = nn.ModuleDict({
            str(s): nn.Linear(d_student, d_teacher, bias=False) for s in layer_map
        })

    def forward(self, student_hidden: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Returns teacher_layer -> projected student features."""
        return {self.layer_map[s]: self.proj[str(s)](h)
                for s, h in student_hidden.items() if s in self.layer_map}

"""Stage-1 input assembly, without a checkpoint.

The two things worth pinning here are the ones that fail silently on a GPU:
the autoregressive shift in `gather_targets` (off by one = a model that trains
to predict the token it was just given), and the right-padding contract in the
student collator (`reasoner_forward` has no attention mask, so left padding
would poison every position).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill import losses  # noqa: E402
from distill.student import prompt  # noqa: E402


def test_gather_targets_applies_the_autoregressive_shift():
    B, L, V, n = 1, 8, 5, 3
    logits = torch.zeros(B, L, V)
    for p in range(L):
        logits[0, p, 0] = p                      # tag each position by its index
    input_ids = torch.arange(L).unsqueeze(0)
    pos = torch.zeros(B, L, dtype=torch.bool)
    pos[0, 4:7] = True                            # targets at positions 4,5,6

    sel, tgt, valid = losses.gather_targets(logits, input_ids, pos, n)
    # The predictor of position p is at p-1.
    assert sel[0, :, 0].tolist() == [3.0, 4.0, 5.0]
    # Targets are the ids AT those positions, read out of the sequence.
    assert tgt[0].tolist() == [4, 5, 6]
    assert valid[0].tolist() == [True] * 3


def test_gather_targets_pads_and_marks_short_rows():
    B, L, V, n = 2, 6, 4, 5
    logits = torch.randn(B, L, V)
    input_ids = torch.arange(L).repeat(B, 1)
    pos = torch.zeros(B, L, dtype=torch.bool)
    pos[0, 1:4] = True                            # 3 targets
    pos[1, 2:3] = True                            # 1 target
    _, tgt, valid = losses.gather_targets(logits, input_ids, pos, n)
    assert valid[0].tolist() == [True, True, True, False, False]
    assert valid[1].tolist() == [True, False, False, False, False]
    assert tgt[1, 0].item() == 2


def test_gather_targets_drops_position_zero():
    """Position 0 has no predictor; including it would read logits[-1]."""
    logits = torch.randn(1, 4, 3)
    input_ids = torch.arange(4).unsqueeze(0)
    pos = torch.zeros(1, 4, dtype=torch.bool)
    pos[0, 0] = True
    pos[0, 2] = True
    _, tgt, valid = losses.gather_targets(logits, input_ids, pos, 2)
    assert valid[0].tolist() == [True, False]
    assert tgt[0, 0].item() == 2                  # position 0 was skipped


def _fake_ctx(n_ids, coc, traj):
    return {"input_ids": torch.arange(n_ids), "pixel_values": None,
            "image_grid_thw": None, "coc_span": coc, "traj_span": traj,
            "history_span": (0, 48), "n_prompt": 5}


def test_collate_student_right_pads_and_maps_spans():
    from distill.data.dataset import collate_student
    items = [{"student": _fake_ctx(10, (5, 8), (8, 10))},
             {"student": _fake_ctx(6, (2, 4), (4, 6))}]
    out = collate_student(items, pad_id=99)
    assert out["input_ids"].shape == (2, 10)
    # Right padding: the SHORT row's tail is pad, its head is intact.
    assert out["input_ids"][1, :6].tolist() == list(range(6))
    assert out["input_ids"][1, 6:].tolist() == [99] * 4
    assert out["attention_mask"][1].tolist() == [True] * 6 + [False] * 4
    assert out["coc_pos"][0, 5:8].all() and not out["coc_pos"][0, 8]
    assert out["traj_pos"][1, 4:6].all() and not out["traj_pos"][1, 6]


def test_history_slots_carry_bins_when_given():
    """Ego motion is discrete bins in the reserved slots, not a side input."""
    segs = prompt.user_segments(["camera_front_wide_120fov"], 1,
                                hist_bins=list(range(48)))
    a = prompt.assemble(
        segs, encode=lambda s: [1] * len(s),
        special_id=lambda t: 900, bin_id=lambda b: 10_000 + b,
        hist_bin_id=lambda b: 20_000 + b,
        image_tokens=lambda i: 2, image_token_id=777)
    lo, hi = a.history_span
    assert hi - lo == 48
    assert a.input_ids[lo] == 20_000 and a.input_ids[hi - 1] == 20_047


def test_history_bins_must_fill_every_slot():
    segs = prompt.user_segments(["camera_front_wide_120fov"], 1, hist_bins=[1, 2])
    with pytest.raises(ValueError, match="expected 48 history bins"):
        prompt.assemble(segs, encode=lambda s: [1], special_id=lambda t: 900,
                        bin_id=lambda b: b, hist_bin_id=lambda b: b,
                        image_tokens=lambda i: 1, image_token_id=777)

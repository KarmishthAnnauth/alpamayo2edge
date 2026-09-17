"""Image dropout in the student collator (D-043), and the empty-span guard in
the text CE. Fake contexts, no tokenizer, no checkpoint."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill import losses  # noqa: E402
from distill.data.dataset import collate_student  # noqa: E402

PAD = 0


def ctx(n_patches: int, seed: int) -> dict:
    """A 20-token context: 8 prompt ids, CoC at [9, 13), struct at [13, 15),
    trajectory bins at [15, 19)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.arange(1, 21),
        "pixel_values": torch.randn(n_patches, 12, generator=g),
        "image_grid_thw": torch.tensor([[1, 2, n_patches // 2]]),
        "coc_span": (9, 13), "struct_span": (13, 15), "traj_span": (15, 19),
        "history_span": (2, 4), "n_prompt": 8,
    }


def test_dropped_row_loses_frames_and_text_targets_keeps_trajectory():
    items = [{"student": ctx(6, 0), "img_drop": True},
             {"student": ctx(4, 1), "img_drop": False}]
    b = collate_student(items, PAD)
    assert b["img_drop"].tolist() == [True, False]
    # row 0's 6 patches are zero, row 1's 4 patches are untouched
    assert torch.equal(b["pixel_values"][:6], torch.zeros(6, 12))
    assert torch.equal(b["pixel_values"][6:], items[1]["student"]["pixel_values"])
    assert b["image_grid_thw"].shape == (2, 3)
    # text targets masked on the dropped row only
    assert not b["coc_pos"][0].any() and not b["struct_pos"][0].any()
    assert b["coc_pos"][1, 9:13].all() and b["struct_pos"][1, 13:15].all()
    # trajectory targets on both
    assert b["traj_pos"][0, 15:19].all() and b["traj_pos"][1, 15:19].all()
    # the sequence itself is unchanged: the CoC is still forced INPUT
    assert torch.equal(b["input_ids"][0], b["input_ids"][1])


def test_default_is_no_dropout():
    b = collate_student([{"student": ctx(4, 2)}], PAD)
    assert b["img_drop"].tolist() == [False]
    assert b["coc_pos"][0, 9:13].all()
    assert float(b["pixel_values"].abs().sum()) > 0


def test_text_ce_on_an_all_masked_batch_is_zero_not_an_error():
    logits = torch.randn(2, 0, 7)
    tgt = torch.zeros(2, 0, dtype=torch.long)
    ok = torch.zeros(2, 0, dtype=torch.bool)
    v = losses.text_kl_or_ce(logits, tgt, ok, vocab_ok=True)
    assert v.shape == () and float(v) == 0.0
    # non-empty logits but a fully false mask: same answer
    v = losses.text_kl_or_ce(torch.randn(2, 3, 7), torch.zeros(2, 3, dtype=torch.long),
                             torch.zeros(2, 3, dtype=torch.bool), vocab_ok=True)
    assert float(v) == 0.0

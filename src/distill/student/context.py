"""Window + cached teacher targets -> student model inputs.

This is the Stage-1 input assembly that `HANDOFF.md` listed as open. It is
deliberately a standalone object rather than a method on `EdgeStudent`: a
DataLoader worker must be able to build contexts, and `EdgeStudent` owns 9GB of
CUDA weights. `ContextBuilder` holds only a tokenizer, an image processor, and a
handful of integers.

What it produces, per window (D-028's format, `prompt.py` holds the template):

    input_ids       the full teacher-forced sequence
    pixel_values    processed camera frames, Edge's own patch layout
    image_grid_thw  [n_images, 3]
    coc_span        where the teacher's CoC text sits  -> text KD
    traj_span       where the 128 future bins sit      -> trajectory KD
    n_prompt        user/assistant boundary

Ego motion needs no side-channel: it is 48 discrete history bins written into
the reserved slots (D-029), so it arrives inside `input_ids` like everything
else.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

import numpy as np
import torch

from . import prompt as prompt_mod

log = logging.getLogger(__name__)


def load_image_processor(ckpt: str):
    """Edge's own image processor.

    VALIDATE-ON-GPU: which processor class the Cosmos3-Edge snapshot ships is
    not knowable from the framework source alone — it lives in the checkpoint's
    `preprocessor_config.json`. `prepare_multimodal_reasoner_inputs` wants the
    patchified pair (`pixel_values [N_patches, C*p*p]`, `image_grid_thw
    [n_images, 3]`), so whatever comes back must produce that.
    """
    from transformers import AutoImageProcessor, AutoProcessor
    try:
        proc = AutoProcessor.from_pretrained(ckpt)
        return getattr(proc, "image_processor", proc)
    except Exception as e:  # noqa: BLE001 — the fallback is the point
        log.info("AutoProcessor failed (%s); trying AutoImageProcessor", e)
        return AutoImageProcessor.from_pretrained(ckpt)


@dataclasses.dataclass
class ContextBuilder:
    """Assembles one window's context. Cheap to copy into a worker."""

    tokenizer: Any
    image_processor: Any
    cameras: list[str]
    n_frames: int
    future_base: int
    hist_base: int
    special_ids: dict[str, int]
    image_token_id: int
    hist_tokenize: Callable | None = None
    merge_size: int = 1          # spatial_merge_size; grid tokens // merge_size**2

    # ---- pieces -------------------------------------------------------------

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _history_bins(self, window) -> list[int] | None:
        """Ego motion -> 48 region-relative delta bins, the teacher's way."""
        if self.hist_tokenize is None:
            return None
        bins = self.hist_tokenize(window.data["ego_history_xyz"],
                                  window.data["ego_history_rot"])
        return [int(b) for b in bins[0]]

    def _encode_images(self, window):
        """Camera frames -> (pixel_values, image_grid_thw, tokens per image).

        Frames are taken at STUDENT resolution (`preprocess.load_window` already
        resized them) and ordered exactly as the prompt orders the cameras:
        ascending camera index, then frame order.
        """
        if self.image_processor is None:
            return None, None, []
        frames = []
        for name in sorted(self.cameras, key=lambda c: prompt_mod.CAMERA_INDEX[c]):
            arr = window.frames_student[name]          # (F, h, w, 3) uint8
            frames.extend(arr[i] for i in range(min(self.n_frames, len(arr))))
        enc = self.image_processor(images=frames, return_tensors="pt")
        grid = enc["image_grid_thw"]
        # One image's placeholder count is its grid volume after the spatial
        # merge — the same arithmetic HF processors use to expand <image>.
        per_image = [int(t * h * w) // (self.merge_size ** 2)
                     for t, h, w in grid.tolist()]
        return enc["pixel_values"], grid, per_image

    # ---- the whole thing ----------------------------------------------------

    def build(self, window, coc_text: str | None = None,
              traj_bins: list[int] | None = None,
              nav_text: str | None = None) -> dict:
        """Teacher-forced sequence when targets are given; a generation prompt
        when they are not (the assistant turn then opens `<|cot_start|>` and
        stops, exactly as the released `create_message` does)."""
        pixel_values, grid, per_image = self._encode_images(window)

        user = prompt_mod.user_segments(
            self.cameras, self.n_frames, nav_text=nav_text,
            hist_bins=self._history_bins(window))
        assistant = prompt_mod.assistant_segments(coc_text=coc_text, traj_bins=traj_bins)

        common = dict(
            encode=self._encode_text,
            special_id=self.special_ids.__getitem__,
            bin_id=lambda b: self.future_base + int(b),
            hist_bin_id=lambda b: self.hist_base + int(b),
            image_tokens=lambda i: per_image[i] if per_image else 0,
            image_token_id=self.image_token_id,
        )
        n_prompt = len(prompt_mod.assemble(user, **common).input_ids)
        a = prompt_mod.assemble(user + assistant, n_prompt=n_prompt, **common)

        if per_image and len(a.image_spans) != len(per_image):
            raise RuntimeError(
                f"{len(a.image_spans)} image slots in the context but "
                f"{len(per_image)} images processed — camera/frame counts disagree")

        return {
            "input_ids": torch.tensor(a.input_ids, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
            "coc_span": a.coc_span,
            "traj_span": a.traj_span,
            "history_span": a.history_span,
            "n_prompt": a.n_prompt,
        }


def spans_to_mask(span, length: int) -> torch.Tensor:
    m = torch.zeros(length, dtype=torch.bool)
    if span is not None:
        m[span[0]:span[1]] = True
    return m

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

    RESOLVED on the GPU box 2026-08-28 (was D-029's last VALIDATE-ON-GPU item).
    The snapshot's `preprocessor_config.json` names `Cosmos3EdgeImageProcessor`
    and `processor_class: Cosmos3EdgeProcessor` — classes that exist only in
    transformers@main, not in the pinned 4.57.1. On 4.x, `AutoProcessor`
    therefore SILENTLY degrades to a bare `PreTrainedTokenizerFast`: no
    exception, no `image_processor` attribute, so the old
    `getattr(proc, "image_processor", proc)` handed back the tokenizer and the
    first batch died in `_encode_images` with "You need to specify either `text`
    or `text_target`". `AutoImageProcessor` is no fallback either — it rejects
    the same config for an unrecognized `image_processor_type`.

    cosmos-framework ships a native port for exactly this case
    (`build_cosmos3_edge_processor`, golden-pinned against the old remote-code
    object by its own `cosmos3_edge_processing_test.py`), so use it whenever the
    directory is a renewed Edge snapshot and keep AutoProcessor for everything
    else.
    """
    from transformers import AutoImageProcessor, AutoProcessor
    try:
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
            build_cosmos3_edge_processor, is_cosmos3_edge_native_snapshot)
        if is_cosmos3_edge_native_snapshot(ckpt):
            # The inner image processor, not the wrapper: the wrapper's
            # `__call__` expands placeholders inside `text` and raises on
            # text=None, and ContextBuilder assembles its own token stream.
            # The wrapper's spatial_shapes->image_grid_thw conversion is
            # reproduced in `_encode_images` instead.
            proc = build_cosmos3_edge_processor(ckpt)
            ip = getattr(proc, "image_processor", None)
            if ip is None:
                raise RuntimeError(
                    "build_cosmos3_edge_processor returned no image_processor")
            log.info("image processor: %s (native Cosmos3-Edge port)",
                     type(ip).__name__)
            return ip
    except ImportError as e:
        log.info("cosmos-framework processor port unavailable (%s); "
                 "falling back to AutoProcessor", e)

    proc = AutoProcessor.from_pretrained(ckpt)
    ip = getattr(proc, "image_processor", None)
    if ip is not None:
        return ip
    # Do NOT fall through to `proc` itself: a degraded AutoProcessor is a
    # tokenizer, and calling it with images= fails much later and far away.
    log.info("AutoProcessor exposed no image_processor (%s); trying "
             "AutoImageProcessor", type(proc).__name__)
    return AutoImageProcessor.from_pretrained(ckpt)


def _grid_from_encoding(enc):
    """`(pixel_values [N_patches, C*p*p], image_grid_thw [n_images, 3])`.

    Two producer conventions reach here. Qwen-style processors return
    `image_grid_thw` directly. Cosmos3-Edge's `Siglip2ImageProcessorCustom`
    returns the SigLIP2 pair `(pixel_values, spatial_shapes [n_images, 2])`
    instead, and the reasoner's `prepare_multimodal_reasoner_inputs` only speaks
    thw. The conversion below is the one Edge's own processor `__call__` applies
    (cosmos_framework/data/generator/processors/cosmos3_edge_processing.py):
    flatten pixel_values to two dims and prepend a t=1 column — still images, so
    one temporal step each.
    """
    pixel_values = enc["pixel_values"]
    if "image_grid_thw" in enc:
        return pixel_values, enc["image_grid_thw"]
    ss = enc["spatial_shapes"]
    pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
    t_dim = torch.ones((ss.shape[0], 1), dtype=ss.dtype, device=ss.device)
    return pixel_values, torch.cat([t_dim, ss], dim=1)


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
        pixel_values, grid = _grid_from_encoding(enc)
        # One image's placeholder count is its grid volume after the spatial
        # merge — the same arithmetic HF processors use to expand <image>.
        per_image = [int(t * h * w) // (self.merge_size ** 2)
                     for t, h, w in grid.tolist()]
        return pixel_values, grid, per_image

    # ---- the whole thing ----------------------------------------------------

    def build(self, window, coc_text: str | None = None,
              traj_bins: list[int] | None = None,
              nav_text: str | None = None,
              for_generation: bool = False) -> dict:
        """Teacher-forced sequence when targets are given; a generation prompt
        when they are not (the assistant turn then opens `<|cot_start|>` and
        stops, exactly as the released `create_message` does).

        `for_generation=True` alongside a `coc_text` builds the stage-1 gate's
        prefill: everything up to and including `<|traj_future_start|>`, with the
        128 trajectory positions left for the student to emit. `traj_span` is
        then None — there is nothing to score teacher-forced — and `input_ids`
        ends exactly where decoding starts.
        """
        pixel_values, grid, per_image = self._encode_images(window)

        user = prompt_mod.user_segments(
            self.cameras, self.n_frames, nav_text=nav_text,
            hist_bins=self._history_bins(window))
        assistant = prompt_mod.assistant_segments(
            coc_text=coc_text, traj_bins=traj_bins, for_generation=for_generation)

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

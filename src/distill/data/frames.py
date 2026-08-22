"""Cache the STUDENT's view of a window next to the teacher's targets.

The teacher labeling pass already loads every window (`preprocess.load_window`,
which also produces student-resolution copies), so writing the student's inputs
out during that pass is nearly free. Without it, stage-1 training re-streams
every clip from the PhysicalAI-AV interface once per epoch, six epochs deep.

Layout, alongside `{window_idx:02d}.npz` (the teacher targets):

    {window_idx:02d}_input.npz
        img|<camera>|<frame>   JPEG bytes, uint8, student resolution
        ego_history_xyz        (1, 1, T, 3)    float32
        ego_history_rot        (1, 1, T, 3, 3) float32

Ego history is here because it is an INPUT the student needs (48 discrete delta
bins, D-029) and the targets file has no reason to carry it.

COLOR ORDER is the trap. `Window.frames_student` is RGB (the loader's
`image_frames`, permuted); `cv2.imencode`/`imdecode` speak BGR. Both directions
flip, and `test_frames_offline.py` pins the round trip — a silent channel swap
would train the student on blue cars and cost a full run to notice.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FRAME_KEY = "img|{camera}|{index}"


@dataclasses.dataclass
class CachedWindow:
    """Duck-compatible with `preprocess.Window` for everything ContextBuilder
    touches: `.frames_student` and `.data[ego_history_*]`. Deliberately does NOT
    carry the teacher's `data` payload — that is what the cache exists to avoid.
    """
    clip_id: str
    frames_student: dict[str, np.ndarray]
    data: dict[str, Any]


def input_path(cache_root: Path, clip_id: str, window_idx: int) -> Path:
    return Path(cache_root) / clip_id / f"{window_idx:02d}_input.npz"


def save_window_input(path: Path, window, quality: int = 92) -> Path:
    """Write JPEG frames + ego history. Returns the path written."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for camera, frames in window.frames_student.items():
        for i, frame in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1],   # RGB -> BGR
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {camera} frame {i}")
            arrays[FRAME_KEY.format(camera=camera, index=i)] = buf.reshape(-1)

    for key in ("ego_history_xyz", "ego_history_rot"):
        v = window.data[key]
        arrays[key] = (v.detach().cpu().numpy() if hasattr(v, "detach")
                       else np.asarray(v)).astype(np.float32)
    np.savez(path, **arrays)
    return path


def load_window_input(path: Path, clip_id: str) -> CachedWindow:
    """Read a cached student input back into a Window-shaped object."""
    import cv2
    import torch

    frames: dict[str, list[tuple[int, np.ndarray]]] = {}
    data: dict[str, Any] = {}
    with np.load(path) as z:
        for key in z.files:
            if not key.startswith("img|"):
                # Tensors: the history tokenizer is torch code.
                data[key] = torch.from_numpy(z[key])
                continue
            _, camera, index = key.split("|")
            img = cv2.imdecode(z[key], cv2.IMREAD_COLOR)        # BGR
            if img is None:
                raise RuntimeError(f"JPEG decode failed for {key} in {path}")
            frames.setdefault(camera, []).append(
                (int(index), np.ascontiguousarray(img[:, :, ::-1])))

    return CachedWindow(
        clip_id=clip_id,
        # Sort by index explicitly: these are (int, ndarray) tuples, and a plain
        # sorted() would fall through to comparing arrays on any index collision.
        frames_student={c: np.stack([f for _, f in sorted(v, key=lambda t: t[0])])
                        for c, v in frames.items()},
        data=data,
    )

"""Window extraction from PhysicalAI-AV clips.

A "window" is one training sample: synchronized context frames for each
configured camera (4 frames/camera at the teacher's convention), egomotion
history, and the ground-truth future trajectory over the eval horizon.

Data access goes through the `physical_ai_av` package (the same interface the
Alpamayo 2 repo uses via `load_physical_aiavdataset`) - NOT raw NCore zarr.itar
parsing. Clips stream from HF or read from a local snapshot; `paths.dataset_root`
is passed as the interface's local_dir/cache. See DECISIONS.md D-013.
"""
from __future__ import annotations
import dataclasses
from typing import Any, Iterator

import numpy as np

_AVDI = None  # process-wide singleton; the interface caches metadata on init


def get_interface(cfg):
    """PhysicalAIAVDatasetInterface singleton rooted at cfg.paths.dataset_root."""
    global _AVDI
    if _AVDI is None:
        import physical_ai_av
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(
            local_dir=cfg.paths.dataset_root,
            confirm_download_threshold_gb=float("inf"),
        )
    return _AVDI


def list_clip_ids(cfg) -> list[str]:
    """All clip ids in the dataset index (curation filters holdout countries)."""
    return sorted(get_interface(cfg).clip_index.index.tolist())


def clip_metadata(cfg, clip_id: str) -> dict:
    """Best-effort per-clip metadata dict from data_collection.parquet
    (lower-cased keys) for curation strata / country holdout."""
    avdi = get_interface(cfg)
    try:
        row = avdi.data_collection.loc[clip_id]
    except KeyError:
        return {}
    if hasattr(row, "to_dict"):
        d = row.to_dict()
    else:  # duplicated index -> DataFrame; take first
        d = row.iloc[0].to_dict()
    return {str(k).lower(): v for k, v in d.items()}


@dataclasses.dataclass
class Window:
    clip_id: str
    t0_us: int
    data: dict[str, Any]                   # alpamayo2_super model-input format:
                                           # image_frames (N_cam, F, 3, H, W), camera_indices,
                                           # ego_history_/future_ xyz+rot (1,1,T,...), timestamps
    frames_student: dict[str, np.ndarray]  # camera_name -> (F, h, w, 3) uint8, student res
    gt_future_xyz: np.ndarray              # (64, 3) ego-frame future, fp32 (eval minADE)
    gt_future_rot: np.ndarray              # (64, 3, 3)


def _resize_batch(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import cv2
    h, w = hw
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames])


def load_window(cfg, clip_id: str, t0_us: int) -> Window:
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    avdi = get_interface(cfg)
    cameras = [c.lower() for c in cfg.data.raw["cameras"]]
    camera_features = [getattr(avdi.features.CAMERA, c.upper()) for c in cameras]
    data = load_physical_aiavdataset(
        clip_id=clip_id,
        t0_us=t0_us,
        avdi=avdi,
        maybe_stream=True,
        num_frames=cfg.data.context_frames,
        camera_features=camera_features,
        include_calibration=False,
    )
    # Student-resolution copies, keyed by camera name (frames are (F, 3, H, W)).
    hw = tuple(cfg.data.raw["student_resolution"])
    frames_student = {}
    for i, name in enumerate(data["camera_names"]):
        f = data["image_frames"][i].permute(0, 2, 3, 1).numpy()  # (F, H, W, 3) uint8
        frames_student[name] = _resize_batch(f, hw)
    return Window(
        clip_id=clip_id,
        t0_us=t0_us,
        data=data,
        frames_student=frames_student,
        gt_future_xyz=data["ego_future_xyz"][0, 0].numpy().astype(np.float32),
        gt_future_rot=data["ego_future_rot"][0, 0].numpy().astype(np.float32),
    )


def iter_windows(cfg, clip_id: str) -> Iterator[tuple[int, Window]]:
    n = cfg.data.windows_per_clip
    horizon = cfg.eval.horizon_s
    # Anchor windows so 1.6 s of history and the full future horizon fit inside
    # the 20 s clip, spread evenly with margin at both ends.
    t0s = np.linspace(2.0, 20.0 - horizon - 0.5, n)
    for w_idx, t0 in enumerate(t0s):
        yield w_idx, load_window(cfg, clip_id, int(round(t0 * 1e6)))

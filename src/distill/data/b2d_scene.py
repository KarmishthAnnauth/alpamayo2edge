"""Replay scenes for the DiffGRPO reward: what the log says was around the ego.

One scene per window = the logged 2-D boxes of every obstacle actor over the
window's 65 frames (t0 .. t0+64) in the STUDENT ego frame at t0 (rear axle,
+y left, D-038 / P2-11), plus the ego's own logged boxes, the traffic light
affecting the ego per frame, and the route target. `rl/replay_reward.py` rolls
a sampled trajectory out against this - NAVSIM's non-reactive setting, which is
what ReCogDrive's DiffGRPO stage scores against (its metric cache is exactly
this: logged agents, no simulator in the loop).

Obstacles: `vehicle`, `walker`, and `traffic_sign` entries whose `type_id`
starts with `static.prop.` (cones, warning triangles, accident debris - the
things on the road). Pole-mounted signs and traffic lights are not obstacles.

Layout: `<cache>/<clip>/{w:02d}_scene.npz` next to the window's targets, built
by `scripts/01d_b2d_scenes.py` from the raw annotations. Every scene is
self-checked at build time: the ego's own box at t0, transformed into the t0
frame, must sit at (+1.25, 0) axis-aligned - which pins the yaw convention of
CARLA's `rotation` and the box construction on every window.
"""
from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from .bench2drive import (CARLA_TO_EGO, N_FUTURE, REAR_AXLE_OFFSET_M, ClipAnno, _read_anno)

SCENE_SUFFIX = "_scene.npz"
N_SCENE = N_FUTURE + 1          # t0 .. t0 + 64
TL_NONE = -1                    # no light affecting the ego on that frame
TL_RED, TL_YELLOW, TL_GREEN = 0, 1, 2     # carla.TrafficLightState
OBSTACLE_CLASSES = ("vehicle", "walker", "prop")
SELF_CHECK_YAW_DEG = 3.0
SELF_CHECK_POS_M = 0.35


def scene_path(cache_root: Path, clip_id: str, w_idx: int) -> Path:
    return Path(cache_root) / clip_id / f"{w_idx:02d}{SCENE_SUFFIX}"


def obstacle_class(box: dict) -> str | None:
    c = box.get("class")
    if c == "vehicle":
        return "vehicle"
    if c == "walker":
        return "walker"
    if c == "traffic_sign" and str(box.get("type_id", "")).startswith("static.prop."):
        return "prop"
    return None


def _f(v, n=None) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a if n is None else a[:n]


@dataclasses.dataclass
class FrameBoxes:
    """The obstacle boxes of one annotation frame, CARLA world frame."""
    ids: np.ndarray            # (A,) int64
    cls: list[str]
    type_id: list[str]
    center: np.ndarray         # (A, 3) world
    yaw_deg: np.ndarray        # (A,)
    extent: np.ndarray         # (A, 2) half sizes (x, y)
    ego_center: np.ndarray     # (3,)
    ego_yaw_deg: float
    ego_extent: np.ndarray     # (2,)
    ego_bbx_dx: float          # box centre offset along the body x axis from the actor origin
    tl_state: int
    tl_trigger: np.ndarray     # (3,) world, NaN when no light affects the ego
    target_xy: np.ndarray      # (2,) world


def parse_frame(d: dict) -> FrameBoxes:
    ids, cls, tid, cen, yaw, ext = [], [], [], [], [], []
    ego = None
    tl_state, tl_trig = TL_NONE, np.full(3, np.nan)
    for b in d.get("bounding_boxes", []):
        c = b.get("class")
        if c == "ego_vehicle":
            ego = b
            continue
        if c == "traffic_light":
            if str(b.get("affects_ego", "False")) == "True":
                tl_state = int(float(b.get("state", TL_NONE)))
                tl_trig = _f(b.get("trigger_volume_location", b["location"]), 3)
            continue
        oc = obstacle_class(b)
        if oc is None:
            continue
        ids.append(int(b.get("id", -1)))
        cls.append(oc)
        tid.append(str(b.get("type_id", "")))
        cen.append(_f(b.get("center", b["location"]), 3))
        yaw.append(float(_f(b["rotation"])[2]))
        ext.append(_f(b["extent"], 2))
    if ego is None:
        raise KeyError("no ego_vehicle in bounding_boxes")
    A = len(ids)
    return FrameBoxes(
        ids=np.asarray(ids, dtype=np.int64),
        cls=cls, type_id=tid,
        center=np.asarray(cen, dtype=np.float64).reshape(A, 3),
        yaw_deg=np.asarray(yaw, dtype=np.float64),
        extent=np.asarray(ext, dtype=np.float64).reshape(A, 2),
        ego_center=_f(ego.get("center", ego["location"]), 3),
        ego_yaw_deg=float(_f(ego["rotation"])[2]),
        ego_extent=_f(ego["extent"], 2),
        ego_bbx_dx=float(_f(ego.get("bbx_loc", [0, 0, 0]))[0]),
        tl_state=tl_state, tl_trigger=tl_trig,
        target_xy=np.array([float(d.get("x_target", np.nan)), float(d.get("y_target", np.nan))]),
    )


def load_clip_boxes(clip_dir: Path, n_frames: int) -> list[FrameBoxes]:
    clip_dir = Path(clip_dir)
    return [parse_frame(_read_anno(clip_dir / "anno" / f"{k:05d}.json.gz")) for k in range(n_frames)]


# ---- geometry --------------------------------------------------------------

_LOCAL = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)   # corner signs


def box_corners_world(center_xy: np.ndarray, yaw_deg: np.ndarray, extent_xy: np.ndarray) -> np.ndarray:
    """(..., 2), (...), (..., 2) -> (..., 4, 2) corners of yaw-rotated boxes."""
    yaw = np.deg2rad(np.asarray(yaw_deg, dtype=np.float64))
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    local = _LOCAL * np.asarray(extent_xy, dtype=np.float64)[..., None, :]    # (..., 4, 2)
    lx, ly = local[..., 0], local[..., 1]
    x = np.asarray(center_xy)[..., None, 0] + c * lx - s * ly
    y = np.asarray(center_xy)[..., None, 1] + s * lx + c * ly
    return np.stack([x, y], axis=-1)


def world_to_ego_xy(world2ego_t0: np.ndarray, pts: np.ndarray, z: float | np.ndarray = 0.0) -> np.ndarray:
    """CARLA world (..., 2) points at height z -> student ego frame at t0, (..., 2)."""
    pts = np.asarray(pts, dtype=np.float64)
    zz = np.broadcast_to(np.asarray(z, dtype=np.float64)[..., None] if np.ndim(z) else np.float64(z),
                         pts.shape[:-1] + (1,))
    h = np.concatenate([pts, zz, np.ones_like(zz)], axis=-1)      # (..., 4)
    e = h @ world2ego_t0.T                                         # (..., 4) CARLA ego frame
    return e[..., :2] @ CARLA_TO_EGO[:2, :2]                       # y flip (S symmetric)


def corners_yaw(corners: np.ndarray) -> np.ndarray:
    """Heading of boxes from their corner order (edge 3 -> 0 runs along +x body)."""
    e = corners[..., 0, :] - corners[..., 3, :]
    return np.arctan2(e[..., 1], e[..., 0])


# ---- scene -----------------------------------------------------------------

def build_scene(anno: ClipAnno, boxes: list[FrameBoxes], t0: int, clip_id: str | None = None) -> dict[str, Any]:
    if not (0 <= t0 and t0 + N_FUTURE < len(boxes)):
        raise ValueError(f"anchor {t0} + {N_FUTURE} exceeds {len(boxes)} frames")
    w2e = anno.world2ego[t0]
    frames = np.arange(t0, t0 + N_SCENE)
    # actor registry over the window
    reg: dict[int, tuple[str, str]] = {}
    for k in frames:
        fb = boxes[k]
        for i, aid in enumerate(fb.ids):
            reg.setdefault(int(aid), (fb.cls[i], fb.type_id[i]))
    ids = np.array(sorted(reg), dtype=np.int64)
    row = {int(a): i for i, a in enumerate(ids)}
    A = len(ids)
    corners = np.full((A, N_SCENE, 4, 2), np.nan, dtype=np.float32)
    centers = np.full((A, N_SCENE, 2), np.nan, dtype=np.float32)
    ego_c = np.zeros((N_SCENE, 4, 2), dtype=np.float32)
    tl_state = np.full(N_SCENE, TL_NONE, dtype=np.int8)
    tl_stop = np.full((N_SCENE, 2), np.nan, dtype=np.float32)
    for j, k in enumerate(frames):
        fb = boxes[k]
        if A and len(fb.ids):
            cw = box_corners_world(fb.center[:, :2], fb.yaw_deg, fb.extent)        # (a, 4, 2)
            ce = world_to_ego_xy(w2e, cw, fb.center[:, 2:3].repeat(4, 1))
            cc = world_to_ego_xy(w2e, fb.center[:, :2], fb.center[:, 2])
            for i, aid in enumerate(fb.ids):
                r = row[int(aid)]
                corners[r, j] = ce[i]
                centers[r, j] = cc[i]
        ego_c[j] = world_to_ego_xy(w2e, box_corners_world(fb.ego_center[:2], fb.ego_yaw_deg, fb.ego_extent),
                                   fb.ego_center[2])
        tl_state[j] = fb.tl_state
        if fb.tl_state != TL_NONE and np.isfinite(fb.tl_trigger).all():
            tl_stop[j] = world_to_ego_xy(w2e, fb.tl_trigger[:2], fb.tl_trigger[2])
    # self-check: the ego's own box at t0 in the t0 frame is axis-aligned at (+offset, 0)
    fb0 = boxes[t0]
    c0 = ego_c[0].astype(np.float64)
    yaw0 = float(np.degrees(corners_yaw(c0)))
    mid0 = c0.mean(0)
    want_x = REAR_AXLE_OFFSET_M + fb0.ego_bbx_dx
    if abs(yaw0) > SELF_CHECK_YAW_DEG or abs(mid0[0] - want_x) > SELF_CHECK_POS_M or abs(mid0[1]) > SELF_CHECK_POS_M:
        raise RuntimeError(f"scene self-check failed at frame {t0}: ego box yaw {yaw0:.2f} deg, "
                           f"centre {mid0.round(3).tolist()} (want ({want_x:.2f}, 0))")
    target = fb0.target_xy
    target_xy = (world_to_ego_xy(w2e, target, fb0.ego_center[2]) if np.isfinite(target).all()
                 else np.full(2, np.nan))
    return {
        "frames": frames.astype(np.int32),
        "actor_id": ids,
        "actor_class": np.array([reg[int(a)][0] for a in ids], dtype="<U8"),
        "actor_type": np.array([reg[int(a)][1] for a in ids], dtype="<U48"),
        "actor_corners": corners,
        "actor_center": centers,
        "ego_corners_expert": ego_c,
        "ego_extent": fb0.ego_extent.astype(np.float32),
        "ego_center_offset_m": np.float32(want_x),
        "tl_state": tl_state,
        "tl_stop_xy": tl_stop,
        "target_xy": target_xy.astype(np.float32),
        "anchor_frame": np.int32(t0),
        "clip_id": np.str_(clip_id or anno.meta.clip_id),
    }


def save_scene(path: Path, scene: dict[str, Any]) -> Path:
    from ..teacher.labeler import atomic_savez
    atomic_savez(Path(path), True, **scene)
    return Path(path)


def load_scene(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def empty_scene(ego_extent=(2.446, 0.918), offset: float = REAR_AXLE_OFFSET_M) -> dict[str, Any]:
    """A scene with no actors and no lights (tests, and the no-map ablation)."""
    return {
        "frames": np.arange(N_SCENE, dtype=np.int32),
        "actor_id": np.zeros(0, dtype=np.int64),
        "actor_class": np.zeros(0, dtype="<U8"),
        "actor_type": np.zeros(0, dtype="<U48"),
        "actor_corners": np.zeros((0, N_SCENE, 4, 2), dtype=np.float32),
        "actor_center": np.zeros((0, N_SCENE, 2), dtype=np.float32),
        "ego_corners_expert": np.zeros((N_SCENE, 4, 2), dtype=np.float32),
        "ego_extent": np.asarray(ego_extent, dtype=np.float32),
        "ego_center_offset_m": np.float32(offset),
        "tl_state": np.full(N_SCENE, TL_NONE, dtype=np.int8),
        "tl_stop_xy": np.full((N_SCENE, 2), np.nan, dtype=np.float32),
        "target_xy": np.full(2, np.nan, dtype=np.float32),
        "anchor_frame": np.int32(0),
        "clip_id": np.str_("synthetic"),
    }

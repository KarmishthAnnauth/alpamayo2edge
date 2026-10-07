"""Bench2Drive clips -> training windows in the student pipeline's own schema.

Phase 2 trains the flow head on closed-loop CARLA data (Bench2Drive is the eval
target; PhysicalAI-AV minADE is only a proxy). This module turns ONE extracted
Bench2Drive clip into windows shaped exactly like the PhysicalAI-AV windows the
student already consumes, so `student/context.py`, `data/frames.py` and the
stage-2 collators need no change:

    window.frames_student   {config camera slot: (F, h, w, 3) uint8 RGB}
    window.data             ego_history_xyz (1,1,16,3), ego_history_rot (1,1,16,3,3),
                            ego_future_xyz  (1,1,64,3), ego_future_rot  (1,1,64,3,3)
    window.gt_future_xyz    (64, 3) fp32 ego frame at t0     (eval minADE reference)
    window.gt_traj          (64, 2) fp32 UnicycleAccelCurvature, z-normalised (D-002)
    + Bench2Drive labels    should_brake (any future frame; the flag is set on ~70-95%
                            of frames in most scenarios, so also should_brake_frac),
                            expert_controls (64x3 throttle/steer/brake), ego_speed_t0,
                            route command + hint, scenario/town/route/weather,
                            action_floor_m (the action space's own round-trip error
                            at 6.4 s on this window - a floor under any minADE),
                            gt_traj_token_ids (128, emission order) when the teacher's
                            tokenizer spec is available

On disk the layout is the labeler's: `<cache>/<clip>/{w:02d}.npz` targets next to
`{w:02d}_input.npz` (JPEG frames + ego history, `frames.save_window_input`),
`manifest.json` + `split_*.json` at the root, so `dataset.discover_shards` and
`frames.load_window_input` read it unchanged.

CLIP LAYOUT (verified on the extracted set, 2026-09-28): `anno/NNNNN.json.gz` per
frame at 10 Hz and `camera/rgb_{front,front_left,front_right,...}/NNNNN.jpg` at
1600x900. The ego pose comes from the `ego_vehicle` entry of `bounding_boxes`
(`world2ego`, a rigid 4x4), NOT the top-level x/y, which are off by up to 2.6 m.

FRAMES. CARLA is LEFT-handed: x forward, y RIGHT, z up. The teacher/student ego
frame is x forward, +y LEFT (D-038: "turn left" labels have median end bearing
+36 deg). Poses are expressed relative to the anchor pose in CARLA's frame and
then reflected with S = diag(1, -1, 1): p' = S p, R' = S R S. A positive CARLA
`steer` is a RIGHT turn, so after the flip it must show as NEGATIVE y and a
negative yaw; `tests/test_bench2drive_offline.py` pins that on real data.

WINDOW. Anchor t0 with 16 history steps (t0-15..t0) and 64 future steps
(t0+1..t0+64) at 10 Hz, mirroring `alpamayo1_5.load_physical_aiavdataset`
(D-002: 64 waypoints, dt 0.1 s). Student frames: `context_frames` per camera at
[t0-0.3 s, ..., t0], the same spacing as the teacher loader. Anchors are every
`stride` frames (default 20 = 2 s).

CAMERAS. The four config slots map onto CARLA cameras as
    camera_cross_left_120fov   <- rgb_front_left        (CARLA: 70 deg FOV, yawed -55 deg)
    camera_front_wide_120fov   <- rgb_front             (70 deg)
    camera_cross_right_120fov  <- rgb_front_right       (70 deg, +55 deg)
    camera_front_tele_30fov    <- centre crop of rgb_front with a 30 deg horizontal
                                  FOV: half-width f*tan(15 deg) = 306 px at
                                  f = 1142.5, so x in [494, 1106], 16:9 -> 344 rows
                                  centred on cy = 450, then resized to student res.
The slots keep the config names so `context.py` orders and labels them as before;
the raw camera name is kept in the window metadata (`raw_cameras`).

ROUTE HINT. `command_near` / `command_far` are CARLA leaderboard RoadOptions
(1 LEFT, 2 RIGHT, 3 STRAIGHT, 4 LANEFOLLOW, 5 CHANGELANELEFT, 6 CHANGELANERIGHT).
They map onto the existing route vocabulary (`eval/gt_reward.route_hint`: "Turn
left ahead" / "Turn right ahead" / "Continue straight"); there is no lane-change
string in that vocabulary (gt_reward: "There is NO lane-change class in the
route"), so 5/6 read "Continue straight" and the raw command is stored alongside.

Which command? Measured on the extracted clips: the near target sits 1-7 m ahead
on EVERY frame (median 3.7 m) - it is the next dense route waypoint - so the
"near unless the near target is < 5 m, else far" rule degenerates to "always
far", and `command_far` already reads LANEFOLLOW while the ego is mid-turn
(HighwayExit Route291 frame 50: near=LEFT, far=LANEFOLLOW, steer -0.2).
`command_near` is the road option of the segment the ego is ON, so the default
rule (`route_rule="horizon"`) is: the first `command_near` other than
LANEFOLLOW/VOID inside the window's own horizon [t0, t0+64], else the near
command at t0. The spec's rule is kept as `route_rule="near_far"` with its
threshold, and every shard stores `command_near_t0`, `command_far_t0`, the
chosen `command`, and `route_hint_kin` (the phase-1 hint recomputed from the
future path with `data/grounding.route_hint`) so the two can be compared.

REFERENCE POINT. CARLA's actor origin sits mid-body, and a mid-body point slips
relative to the heading in a turn (yaw leads the travel direction by 6-15 deg on
the turning clips). The unicycle action space assumes the velocity is ALONG the
heading, which holds at the rear axle - NVIDIA's rig origin, where the cached
PhysicalAI-AV windows round-trip through `traj_to_action`/`action_to_traj` at
0.002-0.24 m over 6.4 s, against ~1 m for the raw CARLA origin. The ego frame
is therefore moved back along the body axis by `REAR_AXLE_OFFSET_M` = 1.25 m,
the value that minimises |yaw - travel direction| over 8 turning clips (0.41 deg
residual; 3.0 deg at 0 m; the ego is `vehicle.lincoln.mkz_2020` on every clip,
wheelbase 2.85 m). Cameras are unaffected: their extrinsics are never used.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

log = logging.getLogger(__name__)

# ---- constants -------------------------------------------------------------

N_HISTORY = 16              # history steps incl. t0 (A1.5 loader default)
N_FUTURE = 64               # future steps (D-002)
DT = 0.1                    # s; Bench2Drive is 10 Hz, same as the trajectory grid
FPS = 10
DEFAULT_STRIDE = 20         # frames between anchors (2 s)

IMG_W, IMG_H = 1600, 900
FOCAL_PX = 1142.5184053936916   # rgb_front intrinsic (70 deg FOV at 1600 px)
CX, CY = 800.0, 450.0
TELE_FOV_DEG = 30.0

DEFAULT_ROOT = Path("/bulk/datasets/bench2drive/extracted")
DEFAULT_CACHE = Path("/bulk/users/vla/alpamayo2edge/b2d_cache")
DEFAULT_TEACHER_REPO = Path("/bulk/users/vla/alpamayo2edge/teacher-model")

TELE_SLOT = "camera_front_tele_30fov"
#: config camera slot -> CARLA camera directory. The tele slot is a crop of front.
RAW_CAMERA = {
    "camera_cross_left_120fov": "rgb_front_left",
    "camera_front_wide_120fov": "rgb_front",
    "camera_cross_right_120fov": "rgb_front_right",
    TELE_SLOT: "rgb_front",
}

ROAD_OPTION = {-1: "VOID", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT", 4: "LANEFOLLOW",
               5: "CHANGELANELEFT", 6: "CHANGELANERIGHT"}
LANEFOLLOW = 4

#: Release normalisation of the teacher's action space (config.json of
#: nvidia/Alpamayo-1.5-10B, `action_space_cfg`). D-002: use verbatim.
RELEASE_ACTION_SPACE = dict(
    n_waypoints=64, dt=0.1,
    accel_mean=0.02902694707164455, accel_std=0.6810426736454882,
    curvature_mean=0.0002692167976330542, curvature_std=0.026148280660833106,
    accel_bounds=(-9.8, 9.8), curvature_bounds=(-0.33, 0.33),
    theta_lambda=1e-6, theta_ridge=1e-8, v_lambda=1e-6, v_ridge=1e-4,
    a_lambda=1e-4, a_ridge=1e-4, kappa_lambda=1e-4, kappa_ridge=1e-4,
)

#: CARLA (x fwd, y right, z up; left-handed) -> ego (x fwd, y LEFT, z up).
CARLA_TO_EGO = np.diag([1.0, -1.0, 1.0])

#: Ego origin = actor origin moved back along the body x axis by this much
#: (rear axle, see the module docstring). Fitted on data, not read from CARLA.
REAR_AXLE_OFFSET_M = 1.25

_CLIP_RE = re.compile(r"^(?P<scenario>[A-Za-z0-9]+)_(?P<town>Town\d+[A-Za-z]*)"
                      r"_Route(?P<route>\d+)_Weather(?P<weather>\d+)$")


# ---- clip metadata ---------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    scenario: str
    town: str
    route: int
    weather: int

    @property
    def route_key(self) -> str:
        """Split unit: a route in a town. Weather variants of one route stay together."""
        return f"{self.town}/Route{self.route}"


def parse_clip_id(clip_id: str) -> ClipMeta:
    m = _CLIP_RE.match(clip_id)
    if m is None:
        raise ValueError(f"not a Bench2Drive clip name: {clip_id!r}")
    return ClipMeta(clip_id=clip_id, scenario=m["scenario"], town=m["town"],
                    route=int(m["route"]), weather=int(m["weather"]))


def is_done(clip_dir: Path) -> bool:
    """Only clips whose extraction was verified (`.done`) are safe to read: the
    extraction job deletes tarballs as it goes and partial dirs get wiped."""
    return (Path(clip_dir) / ".done").is_file()


def list_done_clips(root: Path = DEFAULT_ROOT) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and is_done(p))


# ---- annotations -----------------------------------------------------------

@dataclasses.dataclass
class ClipAnno:
    meta: ClipMeta
    clip_dir: Path
    n: int                          # usable frames (anno AND every raw camera)
    world2ego: np.ndarray           # (n, 4, 4) f64, CARLA world -> CARLA ego at the
                                    # REAR AXLE (actor origin shifted back, see doc)
    location: np.ndarray            # (n, 3) f64 CARLA world, raw actor origin
    speed: np.ndarray               # (n,) m/s
    throttle: np.ndarray            # (n,)
    steer: np.ndarray               # (n,)  CARLA: + = right
    brake: np.ndarray               # (n,)
    command_near: np.ndarray        # (n,) int RoadOption
    command_far: np.ndarray         # (n,) int
    next_command: np.ndarray        # (n,) int
    xy_command_near: np.ndarray     # (n, 2) world
    xy_command_far: np.ndarray      # (n, 2) world
    should_brake: np.ndarray        # (n,) bool
    weather: dict                   # frame-0 weather dict

    def dist_to_near_target(self, k: int) -> float:
        return float(np.hypot(*(self.xy_command_near[k] - self.location[k, :2])))


def _read_anno(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _ego_box(d: dict) -> dict:
    for b in d.get("bounding_boxes", []):
        if b.get("class") == "ego_vehicle":
            return b
    raise KeyError("no ego_vehicle in bounding_boxes")


def count_frames(clip_dir: Path, cameras: list[str] | None = None) -> int:
    """Frames usable by every consumer: min over anno and each raw camera dir."""
    clip_dir = Path(clip_dir)
    raws = sorted(set(RAW_CAMERA[c] for c in (cameras or RAW_CAMERA)))
    counts = [len(list((clip_dir / "anno").glob("*.json.gz")))]
    counts += [len(list((clip_dir / "camera" / r).glob("*.jpg"))) for r in raws]
    return min(counts)


def load_clip_anno(clip_dir: Path, cameras: list[str] | None = None,
                   rear_axle_offset_m: float = REAR_AXLE_OFFSET_M) -> ClipAnno:
    clip_dir = Path(clip_dir)
    meta = parse_clip_id(clip_dir.name)
    n = count_frames(clip_dir, cameras)
    if n < N_HISTORY + N_FUTURE:
        raise ValueError(f"{meta.clip_id}: {n} frames, need >= {N_HISTORY + N_FUTURE}")
    cols: dict[str, list] = {k: [] for k in (
        "world2ego", "location", "speed", "throttle", "steer", "brake", "command_near",
        "command_far", "next_command", "xy_command_near", "xy_command_far", "should_brake")}
    weather: dict = {}
    for k in range(n):
        d = _read_anno(clip_dir / "anno" / f"{k:05d}.json.gz")
        ego = _ego_box(d)
        cols["world2ego"].append(np.asarray(ego["world2ego"], dtype=np.float64))
        cols["location"].append(np.asarray(ego["location"], dtype=np.float64))
        cols["speed"].append(float(d["speed"]))
        cols["throttle"].append(float(d["throttle"]))
        cols["steer"].append(float(d["steer"]))
        cols["brake"].append(float(d["brake"]))
        cols["command_near"].append(int(d["command_near"]))
        cols["command_far"].append(int(d["command_far"]))
        cols["next_command"].append(int(d.get("next_command", -1)))
        cols["xy_command_near"].append((float(d["x_command_near"]), float(d["y_command_near"])))
        cols["xy_command_far"].append((float(d["x_command_far"]), float(d["y_command_far"])))
        cols["should_brake"].append(bool(d.get("should_brake", False)))
        if k == 0:
            weather = dict(d.get("weather", {}))
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    # Rear-axle ego frame: a point at body coords (-l, 0, 0) becomes the origin,
    # so p_ego' = p_ego + (l, 0, 0)  =>  world2ego' = Trans(l, 0, 0) @ world2ego.
    shift = np.eye(4); shift[0, 3] = float(rear_axle_offset_m)
    arrays["world2ego"] = shift[None] @ arrays["world2ego"]
    return ClipAnno(meta=meta, clip_dir=clip_dir, n=n, weather=weather, **arrays)


# ---- geometry --------------------------------------------------------------

def anchor_frames(n_frames: int, stride: int = DEFAULT_STRIDE,
                  n_history: int = N_HISTORY, n_future: int = N_FUTURE) -> list[int]:
    """Anchors t0 with the full history (t0-15) and future (t0+64) inside the clip."""
    return list(range(n_history - 1, n_frames - n_future, int(stride)))


def relative_poses(world2ego: np.ndarray, t0: int, frames: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Poses of `frames` in the STUDENT ego frame at t0.

    Returns `(xyz (k,3), rot (k,3,3))`, float64. The frame at t0 maps to the
    origin with the identity rotation, which is what `traj_to_action` assumes
    of `traj_history_xyz[..., -1, :]`.
    """
    w0 = world2ego[t0]                                   # world -> ego_t0
    ego2world = np.linalg.inv(world2ego[frames])         # ego_k -> world
    T = w0[None] @ ego2world                             # ego_k -> ego_t0, CARLA frame
    S = CARLA_TO_EGO
    xyz = T[:, :3, 3] @ S                                # S symmetric: p' = S p
    rot = S[None] @ T[:, :3, :3] @ S[None]
    return xyz, rot


def yaw_of(rot: np.ndarray) -> np.ndarray:
    """Heading about +z from a (…,3,3) rotation, radians; + = left in the ego frame."""
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


def tele_crop_box(f_px: float = FOCAL_PX, fov_deg: float = TELE_FOV_DEG,
                  w: int = IMG_W, h: int = IMG_H, cx: float = CX, cy: float = CY
                  ) -> tuple[int, int, int, int]:
    """`(x0, y0, x1, y1)` of the 30-deg-FOV centre crop of the front camera,
    keeping the source aspect (16:9 at 1600x900)."""
    half_w = f_px * math.tan(math.radians(fov_deg / 2.0))
    crop_w = int(round(2 * half_w))
    crop_h = int(round(crop_w * h / w))
    x0 = int(round(cx - crop_w / 2)); y0 = int(round(cy - crop_h / 2))
    return x0, y0, x0 + crop_w, y0 + crop_h


# ---- action space ----------------------------------------------------------

NORM_KEYS = ("accel_mean", "accel_std", "curvature_mean", "curvature_std")
ACTION_SPACE_FILE = "action_space.json"


def load_action_space(teacher_repo: Path | None = DEFAULT_TEACHER_REPO,
                      stats: dict | None = None):
    """The teacher's `UnicycleAccelCurvatureActionSpace` with release stats.

    Reads `action_space_cfg` from the local teacher config when it exists and
    checks it against `RELEASE_ACTION_SPACE`; a mismatch raises, because a
    silently re-estimated normalisation would put `gt_traj` in a different
    space from every cached teacher target.

    `stats` (phase 2, TODO T1 / SFT run 2): override of the four normalisation
    constants with Bench2Drive's own statistics for the domain-31 head. Only
    the affine (mean, std) changes; bounds, dt and the fit lambdas stay the
    teacher's, so `action_to_traj` gives the same metres for the same physical
    controls. A cache built this way carries the constants in
    `<cache>/action_space.json` - read it back with `load_cache_action_space`.
    """
    from alpamayo1_5.action_space.unicycle_accel_curvature import (
        UnicycleAccelCurvatureActionSpace)
    kw = dict(RELEASE_ACTION_SPACE)
    if stats:
        bad = set(stats) - set(NORM_KEYS)
        if bad:
            raise ValueError(f"action-space stats may override only {NORM_KEYS}, got {sorted(bad)}")
        kw.update({k: float(v) for k, v in stats.items()})
        return UnicycleAccelCurvatureActionSpace(**kw)
    cfg_path = Path(teacher_repo) / "config.json" if teacher_repo else None
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path) as f:
            asc = dict(json.load(f)["action_space_cfg"])
        asc.pop("_target_", None)
        for k, v in asc.items():
            ref = kw.get(k)
            if ref is None:
                continue
            if not np.allclose(np.asarray(v, dtype=float), np.asarray(ref, dtype=float)):
                raise ValueError(f"teacher config {cfg_path}: action_space_cfg.{k}={v} "
                                 f"differs from the release value {ref}")
        kw.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in asc.items()})
    return UnicycleAccelCurvatureActionSpace(**kw)


def load_cache_action_space(cache_root: Path | None, teacher_repo: Path | None = DEFAULT_TEACHER_REPO):
    """The action space a cache was built with: `<cache>/action_space.json` when
    the builder wrote one (re-normalised caches), else the teacher's release space."""
    if cache_root is not None:
        f = Path(cache_root) / ACTION_SPACE_FILE
        if f.exists():
            stats = {k: v for k, v in json.loads(f.read_text()).items() if k in NORM_KEYS}
            return load_action_space(teacher_repo, stats=stats)
    return load_action_space(teacher_repo)


def traj_to_action(action_space, data: dict) -> np.ndarray:
    """GT future -> (64, 2) normalised (accel, curvature), the labeler's call
    (`teacher/wrapper.py::label_window`) on the same 4-D tensors."""
    a = action_space.traj_to_action(
        traj_history_xyz=data["ego_history_xyz"],
        traj_history_rot=data["ego_history_rot"],
        traj_future_xyz=data["ego_future_xyz"],
        traj_future_rot=data["ego_future_rot"],
    ).reshape(*action_space.get_action_space_dims())
    return a.float().cpu().numpy()


def action_to_traj(action_space, gt_traj: np.ndarray, data: dict) -> np.ndarray:
    """(64, 2) normalised action -> (64, 3) ego-frame waypoints."""
    import torch
    xyz, _ = action_space.action_to_traj(
        torch.as_tensor(gt_traj, dtype=torch.float32).view(1, 1, N_FUTURE, 2),
        traj_history_xyz=data["ego_history_xyz"],
        traj_history_rot=data["ego_history_rot"])
    return xyz[0, 0].cpu().numpy()


# ---- route hint ------------------------------------------------------------

def _hint_strings() -> dict[str, str]:
    from ..eval import gt_reward
    return {lat: gt_reward.route_hint({"lateral": lat})
            for lat in ("turn_left", "turn_right", "straight")}


def route_hint_text(command: int) -> str:
    """RoadOption -> the phase-1 route vocabulary (never a new string)."""
    s = _hint_strings()
    if command == 1:
        return s["turn_left"]
    if command == 2:
        return s["turn_right"]
    return s["straight"]        # STRAIGHT, LANEFOLLOW, lane changes, VOID


def route_command(anno: ClipAnno, t0: int, rule: str = "horizon",
                  near_min_m: float = 5.0, horizon: int = N_FUTURE) -> int:
    """The RoadOption that applies to the window. See the module docstring."""
    if rule == "horizon":
        hi = min(anno.n, t0 + horizon + 1)
        for c in anno.command_near[t0:hi]:
            if int(c) not in (LANEFOLLOW, -1):
                return int(c)
        return int(anno.command_near[t0])
    if rule == "near_far":
        if anno.dist_to_near_target(t0) < near_min_m:
            return int(anno.command_far[t0])
        return int(anno.command_near[t0])
    raise ValueError(f"unknown route_rule {rule!r}")


# ---- windows ---------------------------------------------------------------

@dataclasses.dataclass
class WindowParams:
    cameras: list[str]
    n_frames: int = 4
    student_hw: tuple[int, int] = (360, 640)
    jpeg_quality: int = 92
    stride: int = DEFAULT_STRIDE
    route_rule: str = "horizon"
    near_min_m: float = 5.0

    @classmethod
    def from_cfg(cls, cfg, **over) -> "WindowParams":
        d = cfg.data
        kw = dict(cameras=[str(c) for c in d.raw["cameras"]],
                  n_frames=int(d.context_frames),
                  student_hw=tuple(int(x) for x in d.raw["student_resolution"]),
                  jpeg_quality=int(d.get("frame_jpeg_quality", 92)))
        kw.update(over)
        return cls(**kw)

    def __post_init__(self):
        unknown = [c for c in self.cameras if c not in RAW_CAMERA]
        if unknown:
            raise ValueError(f"no Bench2Drive source for camera slot(s) {unknown}; "
                             f"known: {sorted(RAW_CAMERA)}")


@dataclasses.dataclass
class B2DWindow:
    """Duck-compatible with `preprocess.Window` for the context builder and the
    frame cache (`.frames_student`, `.data[ego_history_*]`, `.clip_id`)."""
    clip_id: str
    anchor_frame: int
    t0_us: int
    frames_student: dict[str, np.ndarray]     # slot -> (F, h, w, 3) uint8 RGB
    data: dict[str, Any]                      # 4-D torch tensors, A1.5 layout
    gt_future_xyz: np.ndarray                 # (64, 3) f32
    gt_future_rot: np.ndarray                 # (64, 3, 3) f32
    gt_traj: np.ndarray                       # (64, 2) f32 normalised action
    should_brake: bool                        # any over the future (set on ~85% of
                                              # frames in these scenarios - near-constant)
    should_brake_frac: float                  # fraction of future frames flagged
    should_brake_t0: bool
    expert_controls: np.ndarray               # (64, 3) throttle/steer/brake at t0..t0+63
    ego_speed_t0: float
    command: int
    command_near_t0: int
    command_far_t0: int
    route_hint: str
    route_hint_kin: str
    meta: ClipMeta
    weather: dict
    raw_cameras: dict[str, str]
    action_floor_m: float                     # |action_to_traj(gt_traj) - future| at 6.4 s:
                                              # the action space's own floor on this window
    gt_traj_token_ids: np.ndarray | None = None   # (128,) emission order, optional


class _FrameReader:
    """Decodes raw JPEGs once per (camera, frame) and serves student-res copies."""

    def __init__(self, clip_dir: Path, params: WindowParams):
        self.clip_dir = Path(clip_dir)
        self.p = params
        self.box = tele_crop_box()
        self._raw: dict[tuple[str, int], np.ndarray] = {}

    def raw(self, cam: str, k: int) -> np.ndarray:
        import cv2
        key = (cam, k)
        if key not in self._raw:
            path = self.clip_dir / "camera" / cam / f"{k:05d}.jpg"
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"JPEG decode failed: {path}")
            self._raw[key] = np.ascontiguousarray(img[:, :, ::-1])   # BGR -> RGB
        return self._raw[key]

    def student(self, slot: str, k: int) -> np.ndarray:
        import cv2
        img = self.raw(RAW_CAMERA[slot], k)
        if slot == TELE_SLOT:
            x0, y0, x1, y1 = self.box
            img = img[y0:y1, x0:x1]
        h, w = self.p.student_hw
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def clear(self):
        self._raw.clear()


def build_window(anno: ClipAnno, t0: int, params: WindowParams, action_space,
                 reader: _FrameReader | None = None,
                 tokenizer_fn: Callable | None = None,
                 with_frames: bool = True) -> B2DWindow:
    import torch
    from . import grounding

    if not (N_HISTORY - 1 <= t0 < anno.n - N_FUTURE):
        raise ValueError(f"anchor {t0} out of range for {anno.n} frames")
    hist_idx = np.arange(t0 - N_HISTORY + 1, t0 + 1)
    fut_idx = np.arange(t0 + 1, t0 + N_FUTURE + 1)
    h_xyz, h_rot = relative_poses(anno.world2ego, t0, hist_idx)
    f_xyz, f_rot = relative_poses(anno.world2ego, t0, fut_idx)
    # t0 is the origin by construction; pin it exactly (traj_to_action assumes it).
    h_xyz[-1] = 0.0; h_rot[-1] = np.eye(3)
    data = {
        "ego_history_xyz": torch.from_numpy(h_xyz).float().view(1, 1, N_HISTORY, 3),
        "ego_history_rot": torch.from_numpy(h_rot).float().view(1, 1, N_HISTORY, 3, 3),
        "ego_future_xyz": torch.from_numpy(f_xyz).float().view(1, 1, N_FUTURE, 3),
        "ego_future_rot": torch.from_numpy(f_rot).float().view(1, 1, N_FUTURE, 3, 3),
    }
    gt_traj = traj_to_action(action_space, data)
    gt_future_xyz = data["ego_future_xyz"][0, 0].numpy().astype(np.float32)
    rec = action_to_traj(action_space, gt_traj, data)
    action_floor_m = float(np.linalg.norm(rec[-1, :2] - gt_future_xyz[-1, :2]))

    tokens = None
    if tokenizer_fn is not None:
        from ..teacher.wrapper import swap_action_dims
        # Exactly the labeler's call, then to EMISSION order (D-031).
        tok = tokenizer_fn(hist_xyz=data["ego_history_xyz"][:, -1],
                           hist_rot=data["ego_history_rot"][:, -1],
                           fut_xyz=data["ego_future_xyz"][:, -1],
                           fut_rot=data["ego_future_rot"][:, -1])[0]
        tokens = swap_action_dims(torch.as_tensor(tok)).cpu().numpy().astype(np.int32)

    frames: dict[str, np.ndarray] = {}
    if with_frames:
        reader = reader or _FrameReader(anno.clip_dir, params)
        # Same spacing as the teacher loader: [t0-(F-1), ..., t0] at 0.1 s.
        ks = [t0 - (params.n_frames - 1 - i) for i in range(params.n_frames)]
        for slot in params.cameras:
            frames[slot] = np.stack([reader.student(slot, k) for k in ks])

    cmd = route_command(anno, t0, params.route_rule, params.near_min_m)
    ctrl_idx = np.arange(t0, t0 + N_FUTURE)     # control at step i -> waypoint i+1
    return B2DWindow(
        clip_id=anno.meta.clip_id, anchor_frame=int(t0), t0_us=int(t0 * 100_000),
        frames_student=frames, data=data,
        gt_future_xyz=gt_future_xyz,
        gt_future_rot=data["ego_future_rot"][0, 0].numpy().astype(np.float32),
        gt_traj=gt_traj.astype(np.float32),
        should_brake=bool(anno.should_brake[fut_idx].any()),
        should_brake_frac=float(anno.should_brake[fut_idx].mean()),
        should_brake_t0=bool(anno.should_brake[t0]),
        expert_controls=np.stack([anno.throttle[ctrl_idx], anno.steer[ctrl_idx],
                                  anno.brake[ctrl_idx]], axis=1).astype(np.float32),
        ego_speed_t0=float(anno.speed[t0]),
        command=cmd, command_near_t0=int(anno.command_near[t0]),
        command_far_t0=int(anno.command_far[t0]),
        route_hint=route_hint_text(cmd),
        route_hint_kin=grounding.route_hint(gt_future_xyz),
        meta=anno.meta, weather=anno.weather,
        raw_cameras={s: RAW_CAMERA[s] + (":tele_crop" if s == TELE_SLOT else "")
                     for s in params.cameras},
        action_floor_m=action_floor_m,
        gt_traj_token_ids=tokens,
    )


def iter_clip_windows(anno: ClipAnno, params: WindowParams, action_space,
                      tokenizer_fn: Callable | None = None,
                      anchors: list[int] | None = None,
                      with_frames: bool = True) -> Iterator[tuple[int, B2DWindow]]:
    reader = _FrameReader(anno.clip_dir, params) if with_frames else None
    anchors = anchor_frames(anno.n, params.stride) if anchors is None else anchors
    for w_idx, t0 in enumerate(anchors):
        yield w_idx, build_window(anno, t0, params, action_space, reader, tokenizer_fn,
                                  with_frames)
        if reader is not None:
            reader.clear()


# ---- cache -----------------------------------------------------------------

def targets_path(cache_root: Path, clip_id: str, w_idx: int) -> Path:
    return Path(cache_root) / clip_id / f"{w_idx:02d}.npz"


def save_window(cache_root: Path, w_idx: int, window: B2DWindow,
                quality: int = 92) -> tuple[Path, Path]:
    """Targets shard + student input, the labeler's two files (atomic writes)."""
    from ..teacher.labeler import atomic_savez
    from . import frames as frames_mod

    arrays: dict[str, Any] = dict(
        gt_traj=window.gt_traj.astype(np.float32),
        gt_future_xyz=window.gt_future_xyz.astype(np.float32),
        gt_future_rot=window.gt_future_rot.astype(np.float32),
        should_brake=np.bool_(window.should_brake),
        should_brake_frac=np.float32(window.should_brake_frac),
        should_brake_t0=np.bool_(window.should_brake_t0),
        expert_controls=window.expert_controls.astype(np.float32),
        ego_speed_t0=np.float32(window.ego_speed_t0),
        action_floor_m=np.float32(window.action_floor_m),
        command=np.int32(window.command),
        command_near_t0=np.int32(window.command_near_t0),
        command_far_t0=np.int32(window.command_far_t0),
        route_hint=np.str_(window.route_hint),
        route_hint_kin=np.str_(window.route_hint_kin),
        scenario=np.str_(window.meta.scenario),
        town=np.str_(window.meta.town),
        route=np.int32(window.meta.route),
        weather=np.str_(json.dumps(window.weather, sort_keys=True)),
        clip_id=np.str_(window.clip_id),
        anchor_frame=np.int32(window.anchor_frame),
        t0_us=np.int64(window.t0_us),
        raw_cameras=np.str_(json.dumps(window.raw_cameras, sort_keys=True)),
        source=np.str_("bench2drive"),
    )
    if window.gt_traj_token_ids is not None:
        arrays["gt_traj_token_ids"] = window.gt_traj_token_ids.astype(np.int32)
    tp = targets_path(cache_root, window.clip_id, w_idx)
    atomic_savez(tp, True, **arrays)
    ip = frames_mod.input_path(cache_root, window.clip_id, w_idx)
    frames_mod.save_window_input(ip, window, quality=quality)
    return tp, ip


def window_record(w_idx: int, window: B2DWindow) -> dict:
    """Small per-window row for `<clip>/windows.json` (summaries without npz reads)."""
    return dict(idx=w_idx, anchor_frame=window.anchor_frame, command=window.command,
                command_near_t0=window.command_near_t0, command_far_t0=window.command_far_t0,
                route_hint=window.route_hint, route_hint_kin=window.route_hint_kin,
                should_brake=window.should_brake, should_brake_frac=window.should_brake_frac,
                ego_speed_t0=window.ego_speed_t0, action_floor_m=window.action_floor_m,
                scenario=window.meta.scenario, town=window.meta.town,
                route=window.meta.route, weather=window.meta.weather)


def load_targets(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---- splits ----------------------------------------------------------------

def make_splits(clip_ids: list[str], val_fraction: float = 0.10, seed: int = 0) -> dict:
    """Train/val by ROUTE (and town), stratified by scenario.

    Every clip of a route (all weathers) lands on one side, so no val route is
    ever seen in training. Per scenario the routes are ranked by a seeded hash
    and the top ~`val_fraction` (at least one when the scenario has >= 2 routes)
    are held out. Deterministic in `(seed, route)`; adding clips re-ranks only
    within their scenario.
    """
    metas = [parse_clip_id(c) for c in clip_ids]
    by_scn: dict[str, dict[str, list[str]]] = {}
    for m in metas:
        by_scn.setdefault(m.scenario, {}).setdefault(m.route_key, []).append(m.clip_id)

    def rank(route_key: str) -> str:
        return hashlib.sha1(f"b2d-split:{seed}:{route_key}".encode()).hexdigest()

    train, val, val_routes, train_routes = [], [], [], []
    per_scn = {}
    for scn in sorted(by_scn):
        routes = sorted(by_scn[scn], key=rank)
        n = len(routes)
        n_val = max(1, int(n * val_fraction + 0.5)) if n >= 2 else 0
        v, t = routes[:n_val], routes[n_val:]
        val_routes += v; train_routes += t
        for r in v:
            val += by_scn[scn][r]
        for r in t:
            train += by_scn[scn][r]
        per_scn[scn] = {"routes": n, "val_routes": n_val,
                        "clips": sum(len(x) for x in by_scn[scn].values()),
                        "val_clips": sum(len(by_scn[scn][r]) for r in v)}
    assert not (set(train) & set(val))
    return {"seed": seed, "val_fraction": val_fraction, "unit": "town/route",
            "train": sorted(train), "val": sorted(val),
            "train_routes": sorted(train_routes), "val_routes": sorted(val_routes),
            "by_scenario": per_scn}


def write_splits(cache_root: Path, clip_ids: list[str], val_fraction: float = 0.10,
                 seed: int = 0) -> dict:
    """`splits.json` (the full record), plus `manifest.json` and
    `split_train.json` / `split_val.json` in the labeler's formats so
    `dataset.discover_shards` and `splits.load_split` work on this cache."""
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    sp = make_splits(clip_ids, val_fraction, seed)
    (root / "splits.json").write_text(json.dumps(sp, indent=1))
    (root / "manifest.json").write_text(json.dumps({"clips": sorted(clip_ids),
                                                     "source": "bench2drive"}, indent=1))
    for name in ("train", "val"):
        (root / f"split_{name}.json").write_text(json.dumps(sp[name], indent=1))
    return sp


# ---- dataset ---------------------------------------------------------------

class Bench2DriveDataset:
    """Bench2Drive windows through the student's context builder.

    Mirrors `dataset.Stage1Dataset.__getitem__` for everything a flow-head
    trainer touches (`student` context, `gt_traj`, `gt_future_xyz`, `hist_xyz`,
    `hist_rot`, `route_hint`, `img_drop`) and adds the Bench2Drive fields. It is
    NOT a teacher cache: there is no CoC text, no teacher token stream and no
    cached expert velocities. Consequences, each a constructor knob:

      coc_lookup        `(clip_id, w_idx) -> str | None`. None (default) builds
                        the free-running prompt (assistant opens `<|cot_start|>`,
                        like `cot_generation`); a string is teacher-forced and
                        the sequence stops at `<|traj_future_start|>` when
                        `for_generation` (the expert's conditioning point) or
                        carries the GT bins otherwise.
      traj_prefix       "gt" puts `gt_traj_token_ids` (present when the cache
                        was built with the tokenizer spec) in the sequence when
                        not `for_generation`; None leaves the bins out.
      n_flow_samples    K > 0 synthesises `(flow_t, flow_a_t, flow_v)` from the
                        GT action in the teacher's convention (D-012: a_t = t a1 +
                        (1-t) a0, v = a1 - a0, t ~ U[0, t_max) stratified) so
                        `collate_b2d` yields what `train_stage2` reads under
                        `stage2.supervision: gt_flow`. There is no `flow_v` from
                        a teacher here, so `teacher_flow` has nothing to distil.
      route_source      "command" (Bench2Drive navigation, deployment-faithful) or
                        "kinematic" (phase-1's hint from the future path).
      with_scene        load `<clip>/{w:02d}_scene.npz` (data/b2d_scene.py, built
                        by scripts/01d_b2d_scenes.py) into item["scene"] for the
                        DiffGRPO replay reward; missing scenes raise.

    Use `collate_b2d`, not `collate_stage2`: the stage-1 collator pads teacher
    keys this cache does not have.
    """

    def __init__(self, cache_root: Path, context_builder, clip_ids: list[str] | None = None,
                 route_source: str = "command", route_hint: bool = True,
                 coc_lookup: Callable[[str, int], str | None] | None = None,
                 for_generation: bool = True, traj_prefix: str | None = "gt",
                 n_flow_samples: int = 0, flow_t_max: float = 0.999,
                 with_scene: bool = False):
        from .dataset import discover_shards
        if route_source not in ("command", "kinematic"):
            raise ValueError(f"route_source must be 'command' or 'kinematic', got {route_source!r}")
        if traj_prefix not in (None, "gt"):
            raise ValueError(f"traj_prefix must be None or 'gt', got {traj_prefix!r}")
        self.cache_root = Path(cache_root)
        self.ctx = context_builder
        self.route_source = route_source
        self.route_hint = bool(route_hint)
        self.coc_lookup = coc_lookup
        self.for_generation = bool(for_generation)
        self.traj_prefix = traj_prefix
        self.n_flow_samples = int(n_flow_samples)
        self.flow_t_max = float(flow_t_max)
        self.with_scene = bool(with_scene)
        self.shards = discover_shards(self.cache_root, clip_ids)
        if not self.shards:
            raise RuntimeError(f"no Bench2Drive shards under {self.cache_root} - "
                               "run scripts/01c_b2d_windows.py first")

    def __len__(self) -> int:
        return len(self.shards)

    def _flow_samples(self, a1):
        import torch
        k = self.n_flow_samples
        # Stratified uniform over [0, t_max), the labeler's `uniform` setting.
        edges = torch.arange(k, dtype=torch.float32) / k
        t = (edges + torch.rand(k) / k) * self.flow_t_max
        a1 = torch.as_tensor(a1, dtype=torch.float32).unsqueeze(0).expand(k, -1, -1)
        a0 = torch.randn_like(a1)
        a_t = t.view(k, 1, 1) * a1 + (1 - t.view(k, 1, 1)) * a0
        return t, a_t, a1 - a0

    def __getitem__(self, i: int) -> dict:
        from . import frames as frames_mod
        from . import grounding
        path = self.shards[i]
        clip_id, w_idx = path.parent.name, int(path.stem)
        d = load_targets(path)
        d["clip_id"] = clip_id
        d["window_idx"] = w_idx
        window = frames_mod.load_window_input(
            frames_mod.input_path(self.cache_root, clip_id, w_idx), clip_id)

        nav = None
        if self.route_hint:
            nav = (str(d["route_hint"]) if self.route_source == "command"
                   else grounding.route_hint(d["gt_future_xyz"]))
        coc = self.coc_lookup(clip_id, w_idx) if self.coc_lookup is not None else None
        bins = None
        if (coc is not None and not self.for_generation and self.traj_prefix == "gt"
                and "gt_traj_token_ids" in d):
            bins = [int(b) for b in d["gt_traj_token_ids"]]
        d["student"] = self.ctx.build(window, coc_text=coc, traj_bins=bins,
                                      nav_text=nav, for_generation=self.for_generation)
        d["route_hint"] = nav or ""
        d["coc_text"] = coc or ""
        d["img_drop"] = False
        d["hist_xyz"] = window.data["ego_history_xyz"][:, -1][0]
        d["hist_rot"] = window.data["ego_history_rot"][:, -1][0]
        if self.n_flow_samples > 0:
            d["flow_t"], d["flow_a_t"], d["flow_v"] = self._flow_samples(d["gt_traj"])
        if self.with_scene:
            from . import b2d_scene as sc
            sp = sc.scene_path(self.cache_root, clip_id, w_idx)
            if not sp.exists():
                raise FileNotFoundError(f"no replay scene {sp} - run scripts/01d_b2d_scenes.py")
            d["scene"] = sc.load_scene(sp)
        return d


def collate_b2d(batch: list[dict], pad_id: int) -> dict:
    """Stage-2-shaped batch from Bench2Drive items: the student context (padded
    like `collate_student`), GT targets, ego history, metadata, and - when the
    dataset synthesised them - the flattened flow tuples in `collate_stage2`'s
    layout (`flow_owner` maps each tuple to its window)."""
    import torch
    from .dataset import collate_student

    out = collate_student(batch, pad_id)
    stack = lambda key: torch.stack([torch.as_tensor(b[key], dtype=torch.float32) for b in batch])
    out.update(
        gt_traj=stack("gt_traj"), gt_future_xyz=stack("gt_future_xyz"),
        hist_xyz=stack("hist_xyz"), hist_rot=stack("hist_rot"),
        expert_controls=stack("expert_controls"),
        ego_speed_t0=torch.tensor([float(b["ego_speed_t0"]) for b in batch]),
        should_brake=torch.tensor([bool(b["should_brake"]) for b in batch]),
        command=torch.tensor([int(b["command"]) for b in batch]),
        clip_ids=[str(b["clip_id"]) for b in batch],
        window_idx=[int(b["window_idx"]) for b in batch],
        route_hint=[str(b["route_hint"]) for b in batch],
        scenario=[str(b["scenario"]) for b in batch],
        town=[str(b["town"]) for b in batch],
    )
    if all("scene" in b for b in batch):
        out["scene"] = [b["scene"] for b in batch]        # list of numpy dicts, one per window
    if all("gt_traj_token_ids" in b for b in batch):
        tok = torch.stack([torch.as_tensor(b["gt_traj_token_ids"], dtype=torch.long)
                           for b in batch])
        # The only trajectory token stream this cache has; `train_stage2` reads
        # `traj` for its scheduled-sampling swap and `gt_traj_tok` for gt_ce.
        out.update(gt_traj_tok=tok, traj=tok.clone(),
                   traj_mask=torch.ones_like(tok, dtype=torch.bool))
    if all("flow_t" in b for b in batch):
        t, a_t, v, owner = [], [], [], []
        for j, b in enumerate(batch):
            k = int(b["flow_t"].shape[0])
            t.append(torch.as_tensor(b["flow_t"], dtype=torch.float32))
            a_t.append(torch.as_tensor(b["flow_a_t"], dtype=torch.float32))
            v.append(torch.as_tensor(b["flow_v"], dtype=torch.float32))
            owner += [j] * k
        out.update(flow_t=torch.cat(t), flow_a_t=torch.cat(a_t), flow_v=torch.cat(v),
                   flow_owner=torch.as_tensor(owner))
    return out


# ---- summary ---------------------------------------------------------------

def summarize(cache_root: Path) -> dict:
    """Windows per scenario / town, braking fraction, route-hint agreement,
    from the per-clip `windows.json` rows (falls back to the npz shards)."""
    root = Path(cache_root)
    rows: list[dict] = []
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        wj = clip_dir / "windows.json"
        if wj.exists():
            rows += json.loads(wj.read_text())
            continue
        for tp in sorted(clip_dir.glob("*.npz")):
            if tp.name.endswith("_input.npz"):
                continue
            z = load_targets(tp)
            rows.append(dict(idx=int(tp.stem), scenario=str(z["scenario"]), town=str(z["town"]),
                             should_brake=bool(z["should_brake"]),
                             should_brake_frac=float(z["should_brake_frac"]),
                             action_floor_m=float(z["action_floor_m"]),
                             command=int(z["command"]),
                             route_hint=str(z["route_hint"]),
                             route_hint_kin=str(z["route_hint_kin"]), clip_id=str(z["clip_id"])))
    def table(key):
        t: dict[str, dict] = {}
        for r in rows:
            e = t.setdefault(r[key], {"windows": 0, "braking": 0, "frac": 0.0, "floor": []})
            e["windows"] += 1; e["braking"] += int(r["should_brake"])
            e["frac"] += float(r.get("should_brake_frac", r["should_brake"]))
            e["floor"].append(float(r.get("action_floor_m", float("nan"))))
        return {k: {"windows": v["windows"],
                    "braking_frac": v["braking"] / max(v["windows"], 1),
                    "brake_frame_frac": v["frac"] / max(v["windows"], 1),
                    "action_floor_p50": float(np.nanmedian(v["floor"])),
                    "action_floor_p90": float(np.nanpercentile(v["floor"], 90))}
                for k, v in sorted(t.items())}
    n = len(rows)
    agree = sum(r["route_hint"] == r["route_hint_kin"] for r in rows)
    cmds: dict[str, int] = {}
    for r in rows:
        cmds[ROAD_OPTION.get(int(r["command"]), str(r["command"]))] = \
            cmds.get(ROAD_OPTION.get(int(r["command"]), str(r["command"])), 0) + 1
    return {"windows": n, "braking_frac": sum(r["should_brake"] for r in rows) / max(n, 1),
            "brake_frame_frac": sum(float(r.get("should_brake_frac", r["should_brake"]))
                                    for r in rows) / max(n, 1),
            "action_floor_p50": float(np.nanmedian([r.get("action_floor_m", np.nan) for r in rows])) if n else float("nan"),
            "action_floor_p90": float(np.nanpercentile([r.get("action_floor_m", np.nan) for r in rows], 90)) if n else float("nan"),
            "by_scenario": table("scenario"), "by_town": table("town"),
            "route_hint_agree_frac": agree / max(n, 1), "commands": cmds}


# ---------------------------------------------------------------- phase-2 glue --

def load_coc_cache(path) -> dict[tuple[str, int], str]:
    """`scripts/05j_b2d_coc.py generate` output -> {(clip_id, w_idx): coc_text}.
    Rows whose generation did not terminate are kept (the text is what the
    model said); empty rows are dropped so the lookup falls back to None."""
    import json
    out: dict[tuple[str, int], str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        txt = str(r.get("student") or "").strip()
        if txt:
            out[(str(r["clip"]), int(r["window"]))] = txt
    return out


def coc_lookup_from_cache(path):
    """Constructor knob for `Bench2DriveDataset(coc_lookup=...)` (None -> free-running)."""
    if path is None:
        return None
    table = load_coc_cache(path)
    return lambda clip_id, w_idx: table.get((str(clip_id), int(w_idx)))


def load_b2d_split(cache_root, name: str) -> list[str]:
    """Clip ids of `split_<name>.json` under the Bench2Drive cache (written by
    `write_splits`; same list format as the labeler's split files)."""
    import json
    p = Path(cache_root) / f"split_{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p}: run scripts/01c_b2d_windows.py first")
    d = json.loads(p.read_text())
    return list(d["clip_ids"]) if isinstance(d, dict) and "clip_ids" in d else list(d)

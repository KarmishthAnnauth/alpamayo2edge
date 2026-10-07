"""Replay (non-reactive) reward for a sampled trajectory against the logged scene.

PDM-shaped, after NAVSIM's score that ReCogDrive's DiffGRPO stage maximises:

    R = collision_free * corridor_ok * red_light_ok
        * (w_p * progress + w_ttc * ttc_ok + w_c * comfort) / (w_p + w_ttc + w_c)

all terms in [0, 1], so R in [0, 1]. Gates are hard (a collision is a zero,
whatever else the sample did), the weighted part is graded. Defaults follow
ReCogDrive's PDMScorerConfig(progress 10, ttc 5, comfort 2).

  collision_free  no overlap between the ego box along the sample and any
                  logged actor box at the same frame, counted only while the
                  ego is moving (NAVSIM's at-fault rule in its simplest form:
                  a logged car driving into a stopped ego is not the plan's
                  fault). Boxes are oriented rectangles, tested by the
                  separating-axis theorem.
  corridor_ok     the sample never leaves a corridor of `corridor_gate_m`
                  around the expert's path (extended straight beyond its end
                  so out-driving the expert on a straight road is allowed).
                  This is the drivable-area proxy until the Bench2Drive map is
                  wired in - the log is the only evidence of drivable space.
  red_light_ok    while a light affecting the ego is red, the ego does not pass
                  the light's trigger volume (+ margin) along the path.
  progress        arc length reached along the expert's path, relative to the
                  expert's own; 1 when the expert did not move (nothing to
                  progress towards - the gates and comfort decide).
  ttc_ok          the ego projected at its current speed and heading for
                  0.5 s / 1 s never overlaps an actor's box at that frame.
  comfort         NVIDIA's comfort bounds (alpamayo1_x_rl comfort_reward.py):
                  fraction of the six metrics (lon/lat accel, jerk, lon jerk,
                  yaw rate, yaw accel) that stay within bounds over the whole
                  plan. The plan is prefixed with the t0 pose, so the first
                  step's acceleration from the current speed counts.

Hedge audit (phase-1.5 lesson, D-042): "stop now" gets collision_free = 1 by
construction but progress 0 on any window where the expert moved and a
comfort hit for the brake; "copy the expert" scores ~1; "drive off" fails the
corridor gate. Measured on real windows by scripts/05l_reward_audit.py.
"""
from __future__ import annotations
import dataclasses
import math
from typing import Any

import numpy as np

from ..data.b2d_scene import TL_RED, TL_NONE

# NVIDIA comfort bounds (alpamayo1_x_rl/rewards/comfort_reward.py)
MAX_ABS_MAG_JERK = 8.37
MAX_ABS_LAT_ACCEL = 4.89
MAX_LON_ACCEL = 2.40
MIN_LON_ACCEL = -4.05
MAX_ABS_YAW_ACCEL = 1.93
MAX_ABS_LON_JERK = 4.13
MAX_ABS_YAW_RATE = 0.95
COMFORT_BOUNDS = {
    "lon_accel": (MIN_LON_ACCEL, MAX_LON_ACCEL),
    "lat_accel": (-MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL),
    "jerk": (-MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK),
    "lon_jerk": (-MAX_ABS_LON_JERK, MAX_ABS_LON_JERK),
    "yaw_accel": (-MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL),
    "yaw_rate": (-MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE),
}


@dataclasses.dataclass
class RewardConfig:
    progress_weight: float = 10.0
    ttc_weight: float = 5.0
    comfort_weight: float = 2.0
    corridor_gate_m: float = 4.0
    collision_requires_motion: bool = True
    motion_eps_mps: float = 0.1
    ttc_horizons_s: tuple[float, ...] = (0.5, 1.0)
    ttc_min_speed_mps: float = 0.5
    progress_min_ref_m: float = 1.0
    path_extension_m: float = 40.0
    red_light_gate: bool = True
    red_light_margin_m: float = 1.0
    obstacle_classes: tuple[str, ...] = ("vehicle", "walker", "prop")
    dt: float = 0.1

    @classmethod
    def from_cfg(cls, section) -> "RewardConfig":
        if section is None:
            return cls()
        raw = section.raw if hasattr(section, "raw") else dict(section)
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name in raw:
                v = raw[f.name]
                kw[f.name] = tuple(v) if isinstance(v, (list, tuple)) else v
        return cls(**kw)


# ---- geometry --------------------------------------------------------------

def yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> heading about +z."""
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


def ego_corners(xy: np.ndarray, yaw: np.ndarray, extent: np.ndarray, center_offset: float) -> np.ndarray:
    """Rear-axle poses (..., 2), (...) -> body boxes (..., 4, 2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    cx = xy[..., 0] + center_offset * c
    cy = xy[..., 1] + center_offset * s
    ex, ey = float(extent[0]), float(extent[1])
    lx = np.array([ex, ex, -ex, -ex]); ly = np.array([ey, -ey, -ey, ey])
    x = cx[..., None] + c[..., None] * lx - s[..., None] * ly
    y = cy[..., None] + s[..., None] * lx + c[..., None] * ly
    return np.stack([x, y], axis=-1)


def _axes(c: np.ndarray) -> np.ndarray:
    """Two unit edge normals of rectangles (..., 4, 2) -> (..., 2, 2)."""
    e = np.stack([c[..., 1, :] - c[..., 0, :], c[..., 2, :] - c[..., 1, :]], axis=-2)
    n = np.stack([-e[..., 1], e[..., 0]], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def boxes_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Separating-axis test of rectangles a (..., 4, 2) vs b (..., 4, 2),
    broadcast over leading dims. A box with any NaN corner never overlaps."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = np.broadcast_to(a, shape + (4, 2)); b = np.broadcast_to(b, shape + (4, 2))
    valid = np.isfinite(a).all(axis=(-1, -2)) & np.isfinite(b).all(axis=(-1, -2))
    a0 = np.nan_to_num(a); b0 = np.nan_to_num(b)
    axes = np.concatenate([_axes(a0), _axes(b0)], axis=-2)         # (..., 4, 2)
    pa = np.einsum("...kd,...cd->...kc", axes, a0)                 # (..., 4 axes, 4 corners)
    pb = np.einsum("...kd,...cd->...kc", axes, b0)
    sep = (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))    # (..., 4)
    return valid & ~sep.any(-1)


def collision_mask(ego_c: np.ndarray, actor_c: np.ndarray) -> np.ndarray:
    """ego (N, T, 4, 2) vs actors (A, T, 4, 2) -> (N, T) any-actor overlap."""
    if actor_c.shape[0] == 0:
        return np.zeros(ego_c.shape[:2], dtype=bool)
    return boxes_overlap(ego_c[:, None], actor_c[None]).any(axis=1)


# ---- dynamics + comfort (numpy port of NVIDIA's comfort_reward.py) ---------

def _diff(x: np.ndarray, dt: float) -> np.ndarray:
    d = x[..., 1:] - x[..., :-1]
    return np.concatenate([d, d[..., -1:]], axis=-1) / dt


def _diff_yaw(h: np.ndarray, dt: float) -> np.ndarray:
    d = np.diff(h, axis=-1)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    r = np.concatenate([d, d[..., -1:]], axis=-1) / dt
    return r


def dynamics(xy: np.ndarray, yaw: np.ndarray, dt: float = 0.1) -> dict[str, np.ndarray]:
    """(N, T, 2), (N, T) -> per-step speed, accelerations, jerks, yaw rates."""
    dx, dy = _diff(xy[..., 0], dt), _diff(xy[..., 1], dt)
    speed = np.hypot(dx, dy)
    dv = _diff(speed, dt)
    v_lon = dx * np.cos(yaw) + dy * np.sin(yaw)
    v_lat = -dx * np.sin(yaw) + dy * np.cos(yaw)
    yaw_rate = _diff_yaw(yaw, dt)
    return {
        "speed": speed,
        "lon_accel": _diff(v_lon, dt),
        "lat_accel": _diff(v_lat, dt),
        "jerk": _diff(dv, dt),
        "lon_jerk": _diff(_diff(v_lon, dt), dt),
        "yaw_rate": yaw_rate,
        "yaw_accel": _diff(yaw_rate, dt),
    }


def comfort_score(dyn: dict[str, np.ndarray]) -> np.ndarray:
    """Fraction of the six comfort metrics within bounds over the whole plan, (N,)."""
    ok = [((dyn[k] > lo) & (dyn[k] < hi)).all(axis=-1) for k, (lo, hi) in COMFORT_BOUNDS.items()]
    return np.mean(np.stack(ok, axis=-1).astype(np.float64), axis=-1)


# ---- path frame ------------------------------------------------------------

def path_polyline(ref_xy: np.ndarray, extension_m: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Expert path (T, 2) from t0+1 -> polyline (M, 2) from the origin, its
    cumulative arc length (M,), and the expert's own total length (before the
    straight extension along the final heading, or +x when it did not move)."""
    pts = np.concatenate([np.zeros((1, 2)), np.asarray(ref_xy, dtype=np.float64)], axis=0)
    seg = np.diff(pts, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    keep = np.concatenate([[True], seglen > 1e-6])
    pts = pts[keep]
    seg = np.diff(pts, axis=0); seglen = np.linalg.norm(seg, axis=1)
    total = float(seglen.sum())
    if len(pts) >= 2 and total > 1e-3:
        head = seg[-1] / seglen[-1]
    else:
        head = np.array([1.0, 0.0])
    if extension_m > 0:
        pts = np.concatenate([pts, (pts[-1] + head * extension_m)[None]], axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s, total


def project_onto(poly: np.ndarray, s_cum: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points (..., 2) -> (arc length along poly, lateral distance to it)."""
    p0 = poly[:-1]; d = poly[1:] - poly[:-1]
    L2 = np.maximum((d * d).sum(-1), 1e-12)                                     # (M-1,)
    rel = p[..., None, :] - p0                                                   # (..., M-1, 2)
    t = np.clip((rel * d).sum(-1) / L2, 0.0, 1.0)                                # (..., M-1)
    close = p0 + t[..., None] * d
    dist = np.linalg.norm(p[..., None, :] - close, axis=-1)                      # (..., M-1)
    j = dist.argmin(-1)
    s = np.take(s_cum[:-1], j) + np.take_along_axis(t, j[..., None], -1)[..., 0] * np.sqrt(np.take(L2, j))
    return s, np.take_along_axis(dist, j[..., None], -1)[..., 0]


# ---- terms -----------------------------------------------------------------

def _actor_corners(scene: dict, classes: tuple[str, ...]) -> np.ndarray:
    ac = np.asarray(scene["actor_corners"], dtype=np.float64)          # (A, 65, 4, 2)
    if ac.shape[0] == 0:
        return ac
    cls = np.asarray(scene["actor_class"]).astype(str)
    return ac[np.isin(cls, list(classes))]


def score(xy: np.ndarray, yaw: np.ndarray, scene: dict[str, Any], gt_future_xy: np.ndarray,
          cfg: RewardConfig | None = None) -> dict[str, np.ndarray]:
    """Score N sampled plans of one window.

    xy   (N, 64, 2) rear-axle waypoints at t0+0.1 .. t0+6.4 in the t0 frame
    yaw  (N, 64) headings (from the action space's rotation output)
    scene  `b2d_scene` dict of that window; gt_future_xy (64, 2) the expert's.
    Returns per-sample arrays: reward and every term, all (N,).
    """
    cfg = cfg or RewardConfig()
    xy = np.asarray(xy, dtype=np.float64)[..., :2]
    yaw = np.asarray(yaw, dtype=np.float64)
    N, T = xy.shape[:2]
    # plan prefixed with the t0 pose: (N, T+1); index k <-> scene frame k
    xy1 = np.concatenate([np.zeros((N, 1, 2)), xy], axis=1)
    yaw1 = np.concatenate([np.zeros((N, 1)), yaw], axis=1)
    extent = np.asarray(scene["ego_extent"], dtype=np.float64)
    offset = float(scene["ego_center_offset_m"])
    dyn = dynamics(xy1, yaw1, cfg.dt)
    speed = dyn["speed"]                                               # (N, T+1), speed INTO the next step
    moving_into = speed > cfg.motion_eps_mps                           # motion on the step leading to k+1
    moving_at = np.concatenate([moving_into[:, :1], moving_into[:, :-1]], axis=1)   # (N, T+1) motion arriving at k

    actors = _actor_corners(scene, cfg.obstacle_classes)               # (A, 65, 4, 2)
    ego_c = ego_corners(xy1, yaw1, extent, offset)                      # (N, T+1, 4, 2)
    hit = collision_mask(ego_c[:, 1:], actors[:, 1:T + 1] if actors.shape[0] else actors)   # (N, T)
    if cfg.collision_requires_motion:
        hit = hit & moving_at[:, 1:]
    collision = hit.any(axis=1)

    # corridor + progress along the (extended) expert path
    poly, s_cum, expert_len = path_polyline(gt_future_xy, cfg.path_extension_m)
    s, lat = project_onto(poly, s_cum, xy1)                            # (N, T+1)
    inside = lat <= cfg.corridor_gate_m
    corridor_ok = inside.all(axis=1)
    first_exit = np.where(corridor_ok, T + 1, np.argmin(inside, axis=1))
    steps = np.arange(T + 1)[None, :]
    s_in = np.where(steps < first_exit[:, None], s, -np.inf)
    prog_m = np.maximum(s_in.max(axis=1), 0.0)
    if expert_len < cfg.progress_min_ref_m:
        progress = np.ones(N)
    else:
        progress = np.clip(prog_m / expert_len, 0.0, 1.0)

    # time-to-collision proxy: constant-velocity projection vs the same frame's actors
    ttc_viol = np.zeros(N, dtype=bool)
    if actors.shape[0]:
        for h in cfg.ttc_horizons_s:
            adv = speed[:, 1:, None] * h * np.stack([np.cos(yaw), np.sin(yaw)], -1)   # (N, T, 2)
            pc = ego_corners(xy + adv, yaw, extent, offset)
            v = collision_mask(pc, actors[:, 1:T + 1]) & (speed[:, 1:] > cfg.ttc_min_speed_mps)
            ttc_viol |= v.any(axis=1)

    # red light: while red, never past the trigger volume (+margin) along the path
    red_viol = np.zeros(N, dtype=bool)
    if cfg.red_light_gate:
        tl = np.asarray(scene["tl_state"]).astype(int)
        stop = np.asarray(scene["tl_stop_xy"], dtype=np.float64)
        for k in np.nonzero((tl == TL_RED))[0]:
            if k == 0 or not np.isfinite(stop[k]).all():
                continue
            s_stop, _ = project_onto(poly, s_cum, stop[k])
            if s_stop < 0.5:                   # trigger already behind the ego at t0
                continue
            red_viol |= s[:, k] > s_stop + cfg.red_light_margin_m

    comfort = comfort_score(dyn)
    ttc_ok = (~ttc_viol).astype(np.float64)
    wsum = cfg.progress_weight + cfg.ttc_weight + cfg.comfort_weight
    graded = (cfg.progress_weight * progress + cfg.ttc_weight * ttc_ok + cfg.comfort_weight * comfort) / wsum
    gate = (~collision) & corridor_ok & (~red_viol)
    return {
        "reward": np.where(gate, graded, 0.0),
        "collision": collision.astype(np.float64),
        "corridor_ok": corridor_ok.astype(np.float64),
        "red_light_violation": red_viol.astype(np.float64),
        "progress": progress,
        "progress_m": prog_m,
        "expert_len_m": np.full(N, expert_len),
        "ttc_ok": ttc_ok,
        "comfort": comfort,
        "max_lat_dev_m": lat.max(axis=1),
    }


def group_advantages(rewards: np.ndarray, eps: float = 1e-4, min_std: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """(B, G) rewards -> (B, G) group-normalised advantages, (B,) skipped mask
    (a group with no reward spread carries no signal, phase-1.5 convention)."""
    r = np.asarray(rewards, dtype=np.float64)
    mu = r.mean(axis=1, keepdims=True)
    sd = r.std(axis=1, keepdims=True)
    skipped = (sd[:, 0] < min_std)
    adv = np.where(skipped[:, None], 0.0, (r - mu) / (sd + eps))
    return adv, skipped

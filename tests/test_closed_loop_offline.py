"""Closed-loop agent geometry (`closed_loop/a2e_common.py`), CPU only, no CARLA.

What is pinned here and why:

  - the constants the agent duplicates (it cannot import `distill`, which needs
    torch) equal `data/bench2drive.py`'s;
  - the dead-reckoned ego history equals the training windows' history
    (`bench2drive.relative_poses` at the rear axle) on a real clip, fed only
    what the leaderboard gives an agent: speedometer and compass. A sign or
    frame error here hands the flow head a mirrored or drifting history and
    nothing but a bad driving score would show it;
  - a plan re-expressed after the ego moved lands where it was;
  - the controller frame: a plan curving LEFT must steer NEGATIVE (CARLA steer
    > 0 is a right turn), a stopped plan brakes;
  - the geo-reference solve inverts the leaderboard's own GPS formula;
  - the route-hint lookahead reproduces `route_command(rule="horizon")`;
  - the wire format round-trips arrays and text.
"""
from __future__ import annotations

import math
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.closed_loop import a2e_common as cm           # noqa: E402


def test_wire_roundtrip():
    a, b = socket.socketpair()
    msg = {"op": "plan", "coc": None, "hint": "turn_left", "n": 3,
           "frames|x": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
           "h": np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3)}
    cm.send_msg(a, msg)
    out = cm.recv_msg(b)
    assert out["op"] == "plan" and out["coc"] is None and out["n"] == 3
    np.testing.assert_array_equal(out["frames|x"], msg["frames|x"])
    np.testing.assert_array_equal(out["h"], msg["h"])
    assert out["h"].dtype == np.float32


@pytest.mark.parametrize("lat_ref,lon_ref", [(0.0, 0.0), (42.0, 2.0), (49.0, 8.0)])
def test_geo_reference(lat_ref, lon_ref):
    # the leaderboard's forward map (route_manipulation._location_to_gps)
    def to_gps(x, y):
        s = math.cos(math.radians(lat_ref))
        mx = s * lon_ref * math.pi * cm.EARTH_RADIUS_EQUA / 180.0 + x
        my = s * cm.EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - y
        lon = mx * 180.0 / (math.pi * cm.EARTH_RADIUS_EQUA * s)
        lat = 360.0 * math.atan(math.exp(my / (cm.EARTH_RADIUS_EQUA * s))) / math.pi - 90.0
        return lat, lon
    lat, lon = to_gps(1234.5, -876.5)
    la, lo = cm.solve_latlon_ref(lat, lon, 1234.5, -876.5)
    assert abs(la - lat_ref) < 1e-6 and abs(lo - lon_ref) < 1e-6
    for x, y in [(0.0, 0.0), (-3000.0, 4500.0), (1250.0, -870.0)]:
        np.testing.assert_allclose(cm.gps_to_location(*to_gps(x, y), la, lo), [x, y], atol=0.02)


def test_controller_frame_and_speed():
    t = (np.arange(64) + 1) * 0.1
    straight = np.stack([5.0 * t, np.zeros(64)], axis=1)            # 5 m/s
    wps = cm.controller_waypoints(straight)
    assert wps.shape == (6, 2)
    np.testing.assert_allclose(wps[:, 1], 2.5 * (np.arange(6) + 1), atol=1e-6)   # forward
    steer, throttle, brake, meta = cm.PIDController().control_pid(wps, 5.0)
    assert abs(meta["desired_speed"] - 5.0) < 1e-6 and steer == 0.0 and brake == 0.0

    # left arc (y > 0 in the ego frame): right component negative, steer negative
    yaw = 0.15 * t
    left = np.stack([np.cumsum(5.0 * 0.1 * np.cos(yaw)), np.cumsum(5.0 * 0.1 * np.sin(yaw))], axis=1)
    steer, *_ = cm.PIDController().control_pid(cm.controller_waypoints(left), 5.0)
    assert steer < -0.01
    steer, *_ = cm.PIDController().control_pid(cm.controller_waypoints(left * [1, -1]), 5.0)
    assert steer > 0.01

    # a stopped plan under a moving car brakes and gives no throttle
    steer, throttle, brake, _ = cm.PIDController().control_pid(
        cm.controller_waypoints(np.zeros((64, 2))), 4.0)
    assert brake == 1.0 and throttle == 0.0 and steer == 0.0
    # a plan that pulls away from standstill throttles
    _, throttle, brake, _ = cm.PIDController().control_pid(cm.controller_waypoints(straight), 0.0)
    assert throttle > 0.3 and brake == 0.0

    # half a second into the plan the first waypoint is the plan's 1.0 s point
    wps = cm.controller_waypoints(straight, elapsed_s=0.5)
    assert abs(wps[0, 1] - 5.0) < 1e-6


def test_plan_reexpressed_after_motion():
    tr = cm.PoseTracker()
    yaw_rate, v = 0.3, 6.0                       # rad/s to the RIGHT in CARLA (yaw grows), m/s
    for k in range(41):
        tr.update(v, math.pi / 2 + yaw_rate * k / cm.SIM_HZ)
    # a world-fixed point 10 m ahead and 2 m LEFT of the pose 20 ticks ago
    p_old = np.array([[10.0, 2.0]])
    p_now = tr.to_current(p_old, 20)
    # the car advanced ~6 m and turned right by 0.3 rad: the point is nearer and further left
    assert p_now[0, 0] < 10.0 - 4.0 and p_now[0, 1] > 2.0
    # and the inverse composition is consistent: zero motion is the identity
    np.testing.assert_allclose(tr.to_current(p_old, 0), p_old)
    still = cm.PoseTracker()
    for _ in range(30):
        still.update(0.0, 1.0)
    np.testing.assert_allclose(still.to_current(p_old, 20), p_old, atol=1e-9)


def test_route_hint_lookahead():
    pl = cm.RoutePlanner(4.0, 50.0)
    pts = [((float(i), 0.0), cm.LANEFOLLOW) for i in range(30)]
    pts += [((30.0 + i, 0.0), 1) for i in range(10)]            # LEFT from 30 m
    pl.set_route(pts)
    pl.run_step(np.array([0.0, 0.0]))
    assert cm.hint_command(pl.command_ahead(5.0)) == "straight"
    assert cm.hint_command(pl.command_ahead(40.0)) == "turn_left"
    for x in np.arange(0.5, 29.01, 0.5):                         # drive up to the turn
        pl.run_step(np.array([x, 0.0]))
    assert pl.route[1][0][0] >= 30.0                             # near target is now on the turn
    assert cm.hint_command(pl.command_ahead(0.0)) == "turn_left"
    assert cm.hint_command(5) == "straight" and cm.hint_command(2) == "turn_right"


# ---- against the training pipeline ------------------------------------------

def test_constants_match_training():
    pytest.importorskip("torch")
    from distill.data import bench2drive as b2d
    assert (cm.N_HISTORY, cm.N_FUTURE, cm.DT) == (b2d.N_HISTORY, b2d.N_FUTURE, b2d.DT)
    assert (cm.IMG_W, cm.IMG_H, cm.FOCAL_PX) == (b2d.IMG_W, b2d.IMG_H, b2d.FOCAL_PX)
    assert cm.REAR_AXLE_OFFSET_M == b2d.REAR_AXLE_OFFSET_M and cm.TELE_SLOT == b2d.TELE_SLOT
    assert cm.tele_crop_box() == b2d.tele_crop_box()
    assert set(cm.SLOT_SENSOR) == set(b2d.RAW_CAMERA)
    raw = {"CAM_FRONT": "rgb_front", "CAM_FRONT_LEFT": "rgb_front_left",
           "CAM_FRONT_RIGHT": "rgb_front_right"}
    assert {s: raw[v] for s, v in cm.SLOT_SENSOR.items()} == b2d.RAW_CAMERA
    assert cm.LANEFOLLOW == b2d.LANEFOLLOW
    for cmd in (-1, 1, 2, 3, 4, 5, 6):
        assert b2d._hint_strings()[cm.hint_command(cmd)] == b2d.route_hint_text(cmd)


def test_history_matches_training_window():
    """Speedometer + compass, as the agent sees them, against the cached
    windows' ego history on real clips."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from distill.data import bench2drive as b2d
    clips = b2d.list_done_clips()
    if not clips:
        pytest.skip("no extracted Bench2Drive clips")
    pref = [c for c in clips if c.name in ("Accident_Town05_Route219_Weather11",)]
    errs, yaws = [], []
    for clip in (pref + clips[:3])[:3]:
        anno = b2d.load_clip_anno(clip)
        w2e = anno.world2ego
        e2w = np.linalg.inv(w2e)
        yaw = np.arctan2(e2w[:, 1, 0], e2w[:, 0, 0])            # CARLA yaw of the ego
        tr = cm.PoseTracker(sim_hz=b2d.FPS)                      # annotations are 10 Hz
        for t0 in range(anno.n):
            tr.update(float(anno.speed[t0]), float(yaw[t0]) + math.pi / 2)
            if t0 < b2d.N_HISTORY - 1 or t0 % 10:
                continue
            idx = np.arange(t0 - b2d.N_HISTORY + 1, t0 + 1)
            ref_xyz, ref_rot = b2d.relative_poses(w2e, t0, idx)
            xyz, rot = tr.history()
            errs.append(np.abs(xyz[:, :2] - ref_xyz[:, :2]).max())
            yaws.append(np.abs(b2d.yaw_of(rot.astype(np.float64)) - b2d.yaw_of(ref_rot)).max())
    errs, yaws = np.array(errs), np.array(yaws)
    # 1.5 s of history at up to ~15 m/s: dead reckoning at 10 Hz stays within
    # the action space's own round-trip floor (p90 0.50 m at 6.4 s, P2-11)
    assert np.median(errs) < 0.10, (np.median(errs), errs.max())
    assert np.percentile(errs, 95) < 0.40, (np.percentile(errs, 95), errs.max())
    assert yaws.max() < 1e-3

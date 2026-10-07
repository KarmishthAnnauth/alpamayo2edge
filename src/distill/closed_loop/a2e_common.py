"""Closed-loop Bench2Drive: everything the agent and the model server share.

numpy + stdlib ONLY. The leaderboard agent (`a2e_agent.py`) runs inside the
evaluator's interpreter (python 3.10, `carla` 0.9.15, no torch); the student
needs python 3.13 / torch 2.10 and there is no `carla` wheel for it. So the
model lives in its own process (`server.py`) and this module is the contract
between the two: the wire format, and the geometry that turns leaderboard
sensors into the window the flow head was trained on (`data/bench2drive.py`).

Imported as `distill.closed_loop.a2e_common` by the server and as a bare
`a2e_common` by the agent (the evaluator puts the agent's directory on sys.path).

WINDOW (must match `bench2drive.build_window`):
  ego frame        x forward, +y LEFT, origin at the REAR AXLE (actor origin moved
                   back 1.25 m), identity at t0.
  ego history      16 poses at 10 Hz, t0-1.5 s .. t0.
  frames           4 per camera at t0-0.3 s .. t0, 0.1 s apart, 360x640 RGB; the
                   tele slot is the 30-deg centre crop of the front camera.
  route hint       first route command other than LANEFOLLOW inside the 6.4 s
                   horizon, else the command at t0 (`route_rule="horizon"`).

POSE. The leaderboard's GNSS carries 5e-6 deg of noise per axis (agent_wrapper.py
:209-211, ~0.5 m), which is five times the 0.1 m a 1 m/s ego moves between two
history steps, so the history is NOT differenced GPS. It is dead-reckoned from
the speedometer and the IMU compass, both noise-free: the unicycle the action
space assumes (velocity along the heading) holds at the rear axle, which is the
training reference point. GPS is used only for route progress, where 0.5 m is
irrelevant (the planner's pop radius is 4 m).
"""
from __future__ import annotations

import json
import math
import socket
import struct
from collections import deque

import numpy as np

# ---- constants (mirrors of data/bench2drive.py; tests pin them together) ----

N_HISTORY = 16
N_FUTURE = 64
DT = 0.1                         # s, trajectory / history grid
SIM_HZ = 20                      # leaderboard fixed_delta_seconds = 0.05
TICKS_PER_STEP = int(round(SIM_HZ * DT))      # 2 sim ticks per 0.1 s
N_FRAMES = 4
STUDENT_HW = (360, 640)
IMG_W, IMG_H = 1600, 900
FOCAL_PX = 1142.5184053936916
TELE_FOV_DEG = 30.0
REAR_AXLE_OFFSET_M = 1.25

TELE_SLOT = "camera_front_tele_30fov"
#: student camera slot -> leaderboard sensor id
SLOT_SENSOR = {
    "camera_cross_left_120fov": "CAM_FRONT_LEFT",
    "camera_front_wide_120fov": "CAM_FRONT",
    "camera_cross_right_120fov": "CAM_FRONT_RIGHT",
    TELE_SLOT: "CAM_FRONT",
}
#: The Bench2Drive data-collection rig (the dataset's rgb_front / front_left /
#: front_right), same values every Bench2DriveZoo agent uses.
CAMERA_RIG = {
    "CAM_FRONT":       dict(x=0.80, y=0.0, z=1.60, yaw=0.0),
    "CAM_FRONT_LEFT":  dict(x=0.27, y=-0.55, z=1.60, yaw=-55.0),
    "CAM_FRONT_RIGHT": dict(x=0.27, y=0.55, z=1.60, yaw=55.0),
}
CAMERA_FOV = 70
#: GNSS/IMU mount, Bench2DriveZoo's: 1.4 m behind the actor origin.
GPS_X = -1.4

LANEFOLLOW = 4
HINT_LEFT, HINT_RIGHT = 1, 2     # RoadOption values that change the hint
EARTH_RADIUS_EQUA = 6378137.0


# ---- wire -------------------------------------------------------------------
# One message = 4-byte big-endian header length, a JSON header, then the raw
# bytes of every array the header lists. Not pickle: the two ends run different
# python and numpy majors.

def send_msg(sock: socket.socket, fields: dict) -> None:
    meta, arrays, blobs = {}, [], []
    for k, v in fields.items():
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            arrays.append([k, a.dtype.str, list(a.shape)])
            blobs.append(a.tobytes())
        else:
            meta[k] = v
    head = json.dumps({"meta": meta, "arrays": arrays}).encode()
    sock.sendall(struct.pack(">I", len(head)) + head + b"".join(blobs))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict:
    (n,) = struct.unpack(">I", _recv_exact(sock, 4))
    head = json.loads(_recv_exact(sock, n))
    out = dict(head["meta"])
    for name, dtype, shape in head["arrays"]:
        dt = np.dtype(dtype)
        size = int(np.prod(shape)) * dt.itemsize
        out[name] = np.frombuffer(_recv_exact(sock, size), dtype=dt).reshape(shape).copy()
    return out


def connect(path: str, timeout: float | None = None) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    return s


# ---- GPS -> metres ----------------------------------------------------------

def gps_to_location(lat: float, lon: float, lat_ref: float, lon_ref: float) -> np.ndarray:
    """Inverse of the leaderboard's `_location_to_gps` (route_manipulation.py):
    CARLA world (x, y) of a GNSS reading."""
    scale = math.cos(lat_ref * math.pi / 180.0)
    my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
    mx = lon * (math.pi * EARTH_RADIUS_EQUA * scale) / 180.0
    y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0)) - my
    x = mx - scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    return np.array([x, y])


def solve_latlon_ref(lat: float, lon: float, x: float, y: float) -> tuple[float, float]:
    """The map's geo-reference from ONE route point known in both frames.

    The reference differs per town and a wrong one rescales every distance by
    cos(lat_ref) (26% at the old hardcoded 42 deg). Bench2DriveZoo agents solve
    the same two equations with scipy's fsolve; the evaluator's env has no
    scipy, and the system reduces to one unknown: for a given lat_ref the y
    equation is monotone, and lon_ref then follows in closed form.
    """
    def g(lat_ref):
        s = math.cos(math.radians(lat_ref))
        return (s * EARTH_RADIUS_EQUA * (math.log(math.tan(math.radians(90.0 + lat_ref) / 2.0))
                                         - math.log(math.tan(math.radians(90.0 + lat) / 2.0))) - y)
    lo, hi = lat - 1.0, lat + 1.0               # 1 deg = 111 km; no CARLA map is that large
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:
        raise ValueError(f"no geo-reference within 1 deg of lat {lat} for y={y}")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    lat_ref = 0.5 * (lo + hi)
    s = math.cos(math.radians(lat_ref))
    lon_ref = lon - x * 180.0 / (math.pi * EARTH_RADIUS_EQUA * s)
    return lat_ref, lon_ref


# ---- route ------------------------------------------------------------------

class RoutePlanner:
    """Bench2DriveZoo's `team_code/planner.py::RoutePlanner.run_step`, same pop
    rule (so `command_near` means what it meant when the dataset was recorded),
    without the debug plotter. `route` is a deque of (xy, command int)."""

    def __init__(self, min_distance: float = 4.0, max_distance: float = 50.0):
        self.route: deque = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

    def set_route(self, points) -> None:
        self.route.clear()
        for xy, cmd in points:
            self.route.append((np.asarray(xy, dtype=np.float64), int(cmd)))

    def run_step(self, pos: np.ndarray):
        if len(self.route) == 1:
            return self.route[0], self.route[0]
        to_pop, farthest, cum = 0, -np.inf, 0.0
        for i in range(1, len(self.route)):
            if cum > self.max_distance:
                break
            cum += np.linalg.norm(self.route[i][0] - self.route[i - 1][0])
            d = np.linalg.norm(self.route[i][0] - pos)
            if d <= self.min_distance and d > farthest:
                farthest, to_pop = d, i
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
        return self.route[0], self.route[1]

    def command_ahead(self, lookahead_m: float) -> int:
        """`bench2drive.route_command(rule="horizon")` without the future: the
        first command other than LANEFOLLOW on the route within `lookahead_m`
        of the near target, else the near command."""
        near = self.route[1] if len(self.route) > 1 else self.route[0]
        if near[1] not in (LANEFOLLOW, -1):
            return near[1]
        cum = 0.0
        for i in range(2, len(self.route)):
            cum += float(np.linalg.norm(self.route[i][0] - self.route[i - 1][0]))
            if cum > lookahead_m:
                break
            if self.route[i][1] not in (LANEFOLLOW, -1):
                return self.route[i][1]
        return near[1]


def hint_command(cmd: int) -> str:
    """RoadOption -> the key the server maps onto the phase-1 hint strings."""
    return {HINT_LEFT: "turn_left", HINT_RIGHT: "turn_right"}.get(int(cmd), "straight")


# ---- pose -------------------------------------------------------------------

def compass_to_yaw(compass: float) -> float:
    """IMU compass (0 = north = CARLA -y, clockwise) -> CARLA yaw (rad, from +x
    towards +y)."""
    return float(compass) - math.pi / 2.0


class PoseTracker:
    """Dead-reckoned rear-axle pose in a CARLA-handed local frame, one entry per
    sim tick. `history()` returns the 16-step ego history of the training
    windows, in the ego frame at the latest tick."""

    def __init__(self, sim_hz: int = SIM_HZ, keep_s: float = 12.0):
        self.dt = 1.0 / sim_hz
        self.stride = int(round(sim_hz * DT))
        self.poses: deque = deque(maxlen=int(keep_s * sim_hz))    # (x, y, yaw)
        self._xy = np.zeros(2)
        self._last = None                                          # (speed, yaw)

    def update(self, speed: float, compass: float) -> None:
        yaw = compass_to_yaw(compass)
        if self._last is not None:
            v0, y0 = self._last
            # trapezoid over the tick: mean speed along the mean heading
            dyaw = math.atan2(math.sin(yaw - y0), math.cos(yaw - y0))
            ym = y0 + 0.5 * dyaw
            self._xy = self._xy + 0.5 * (v0 + speed) * self.dt * np.array([math.cos(ym), math.sin(ym)])
        self._last = (float(speed), yaw)
        self.poses.append((self._xy[0], self._xy[1], yaw))

    def pose(self, ticks_ago: int = 0):
        return self.poses[max(len(self.poses) - 1 - ticks_ago, 0)]

    def history(self) -> tuple[np.ndarray, np.ndarray]:
        """(xyz (16,3), rot (16,3,3)) f32: poses at t0-1.5 s..t0 in the ego frame
        at t0 (x fwd, +y left). Before 1.5 s of driving the oldest pose is
        repeated, i.e. the ego is taken to have stood there."""
        x0, y0, yaw0 = self.pose(0)
        c, s = math.cos(yaw0), math.sin(yaw0)
        xyz = np.zeros((N_HISTORY, 3), dtype=np.float32)
        rot = np.zeros((N_HISTORY, 3, 3), dtype=np.float32)
        for i in range(N_HISTORY):
            x, y, yaw = self.pose((N_HISTORY - 1 - i) * self.stride)
            dx, dy = x - x0, y - y0
            fwd = c * dx + s * dy
            right = -s * dx + c * dy                 # CARLA: +y is right
            xyz[i] = (fwd, -right, 0.0)
            dyaw = -(math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0)))   # + = left
            cy, sy = math.cos(dyaw), math.sin(dyaw)
            rot[i] = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
        xyz[-1] = 0.0
        rot[-1] = np.eye(3)
        return xyz, rot

    def to_current(self, xy_ego: np.ndarray, ticks_ago: int) -> np.ndarray:
        """Points given in the ego frame of `ticks_ago` (x fwd, y left) ->
        the ego frame of the latest tick."""
        if ticks_ago <= 0:
            return np.asarray(xy_ego, dtype=np.float64)
        xp, yp, yawp = self.pose(ticks_ago)
        x0, y0, yaw0 = self.pose(0)
        p = np.asarray(xy_ego, dtype=np.float64)
        cp, sp = math.cos(yawp), math.sin(yawp)
        # ego(past) -> local CARLA frame (y right = -left)
        wx = xp + cp * p[:, 0] - sp * (-p[:, 1])
        wy = yp + sp * p[:, 0] + cp * (-p[:, 1])
        c, s = math.cos(yaw0), math.sin(yaw0)
        dx, dy = wx - x0, wy - y0
        return np.stack([c * dx + s * dy, -(-s * dx + c * dy)], axis=1)


# ---- frames -----------------------------------------------------------------

def tele_crop_box(f_px: float = FOCAL_PX, fov_deg: float = TELE_FOV_DEG,
                  w: int = IMG_W, h: int = IMG_H) -> tuple[int, int, int, int]:
    """`bench2drive.tele_crop_box`: the 30-deg centre crop of the front camera."""
    half_w = f_px * math.tan(math.radians(fov_deg / 2.0))
    crop_w = int(round(2 * half_w))
    crop_h = int(round(crop_w * h / w))
    x0 = int(round(w / 2.0 - crop_w / 2)); y0 = int(round(h / 2.0 - crop_h / 2))
    return x0, y0, x0 + crop_w, y0 + crop_h


def student_frame(bgra: np.ndarray, slot: str, jpeg_quality: int | None = 92) -> np.ndarray:
    """A leaderboard camera image (H, W, 4 BGRA) -> the (360, 640, 3) RGB frame of
    `slot`, through the training pipeline's own steps: the dataset stored JPEGs,
    `_FrameReader.student` cropped/resized them with INTER_AREA, and the cache
    re-encoded at quality 92 (`frames.save_window_input`)."""
    import cv2
    img = bgra[:, :, :3]
    if slot == TELE_SLOT:
        x0, y0, x1, y1 = tele_crop_box(w=img.shape[1], h=img.shape[0])
        img = img[y0:y1, x0:x1]
    h, w = STUDENT_HW
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[:, :, ::-1])


class FrameBuffer:
    """Student-resolution frames of the last ticks; `window()` returns the 4
    frames per slot at t0-0.3 s..t0. Early ticks repeat the oldest frame."""

    def __init__(self, slots, sim_hz: int = SIM_HZ, n_frames: int = N_FRAMES):
        self.slots = list(slots)
        self.stride = int(round(sim_hz * DT))
        self.n_frames = n_frames
        self.buf: deque = deque(maxlen=(n_frames - 1) * self.stride + 1)

    def push(self, frames: dict) -> None:
        self.buf.append(frames)

    def window(self) -> dict:
        out = {}
        for slot in self.slots:
            picks = []
            for i in range(self.n_frames):
                back = (self.n_frames - 1 - i) * self.stride
                picks.append(self.buf[max(len(self.buf) - 1 - back, 0)][slot])
            out[slot] = np.stack(picks)
        return out


# ---- control ----------------------------------------------------------------

def controller_waypoints(traj_xy: np.ndarray, elapsed_s: float = 0.0,
                         n: int = 6, step_s: float = 0.5) -> np.ndarray:
    """The 10 Hz plan (64 x (x fwd, y left), point i at (i+1)*0.1 s after the
    plan was made, ALREADY expressed in the current ego frame) -> the 2 Hz
    waypoints the Bench2DriveZoo PID expects: (n, 2) as (right, forward) at
    0.5 s, 1.0 s, ... from NOW, i.e. `elapsed_s` further along the plan."""
    t_plan = (np.arange(traj_xy.shape[0]) + 1) * DT
    t_plan = np.concatenate([[0.0], t_plan])
    pts = np.concatenate([np.zeros((1, 2)), np.asarray(traj_xy, dtype=np.float64)])
    t_q = np.clip(elapsed_s + step_s * (np.arange(n) + 1), 0.0, t_plan[-1])
    x = np.interp(t_q, t_plan, pts[:, 0])
    y = np.interp(t_q, t_plan, pts[:, 1])
    return np.stack([-y, x], axis=1)


class _PID:
    def __init__(self, K_P, K_I, K_D, n):
        self.K_P, self.K_I, self.K_D = K_P, K_I, K_D
        self.window = deque([0.0] * n, maxlen=n)

    def step(self, error: float) -> float:
        self.window.append(float(error))
        integral = float(np.mean(self.window))
        derivative = self.window[-1] - self.window[-2]
        return self.K_P * error + self.K_I * integral + self.K_D * derivative


class PIDController:
    """Bench2DriveZoo's `team_code/pid_controller.py` (the controller behind the
    published UniAD / VAD / ORION numbers), same gains and same rules:
    desired speed from the first two 2 Hz waypoints, steering towards the
    waypoint nearest `aim_dist`, brake when the plan is slower than the car."""

    def __init__(self, turn=(1.1, 0.5, 0.4, 40), speed=(5.0, 0.5, 1.0, 40),
                 max_throttle: float = 0.75, brake_speed: float = 0.05,
                 brake_ratio: float = 1.1, clip_delta: float = 0.25, aim_dist: float = 3.5):
        self.turn_controller = _PID(*turn)
        self.speed_controller = _PID(*speed)
        self.max_throttle, self.brake_speed = max_throttle, brake_speed
        self.brake_ratio, self.clip_delta, self.aim_dist = brake_ratio, clip_delta, aim_dist

    def control_pid(self, waypoints: np.ndarray, speed: float):
        """`waypoints` (n, 2) as (right, forward) metres. Returns
        (steer, throttle, brake, metadata)."""
        wp = np.asarray(waypoints, dtype=np.float64)
        desired_speed = (0.75 * np.linalg.norm(wp[0]) * 2.0
                         + 0.25 * np.linalg.norm(wp[1] - wp[0]) * 2.0)
        aim, best = wp[0], 1e5
        for i in range(len(wp) - 1):
            for cand in (wp[i], (wp[i + 1] + wp[i]) / 2.0):
                norm = np.linalg.norm(cand)
                if abs(self.aim_dist - best) > abs(self.aim_dist - norm):
                    aim, best = cand, norm
        # aim[1] is the FORWARD component; no steering off a plan that stands still
        angle = 0.0 if aim[1] <= 0.02 else math.degrees(math.pi / 2 - math.atan2(aim[1], aim[0])) / 90.0
        steer = float(np.clip(self.turn_controller.step(angle), -1.0, 1.0))

        brake = bool(desired_speed < self.brake_speed
                     or (speed / max(desired_speed, 1e-6)) > self.brake_ratio)
        delta = float(np.clip(desired_speed - speed, 0.0, self.clip_delta))
        throttle = float(np.clip(self.speed_controller.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0
        meta = dict(speed=float(speed), desired_speed=float(desired_speed), angle=float(angle),
                    aim=[float(aim[0]), float(aim[1])], delta=delta)
        return steer, throttle, float(brake), meta

"""Closed-loop Bench2Drive agent for the stage-2 student (flow head + CoC).

A leaderboard `AutonomousAgent` in the mould of the Bench2DriveZoo agents
(`team_code/orion_b2d_agent.py`, the thesis' `qwen3vl_b2d_agent.py`): same
camera rig, same GNSS/IMU/speedometer, same PID. It holds no model. Each plan it
assembles the window the flow head was trained on (`a2e_common.py`) and asks
`server.py` for a trajectory; see that module for why.

    --agent        src/distill/closed_loop/a2e_agent.py
    --agent-config <unix socket of the model server>      (the evaluator appends +<save_name>)

Slow-fast split (PHASE2_DECISIONS T3), both in sim time so a slow model only
lowers the sim/wallclock ratio (CARLA is synchronous):

    A2E_PLAN_EVERY   ticks between flow-head plans (default 2 = 10 Hz, the data
                     rate). Between plans the last trajectory is followed,
                     re-expressed in the current ego frame.
    A2E_COC_EVERY_S  seconds between CoC refreshes (default 2.0); in between the
                     previous CoC text is handed back as conditioning.
    A2E_HINT_MIN_M   floor of the route-hint lookahead (default 5 m); the
                     lookahead itself is speed x 6.4 s, the window's horizon.
    SAVE_PATH        debug output root: <SAVE_PATH>/<save_name>/{meta/NNNN.json,
                     metric_info.json[, rgb_front/NNNN.jpg with A2E_SAVE_RGB=1]}
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time

import carla
import cv2
import numpy as np

from leaderboard.autoagents import autonomous_agent

import a2e_common as cm

SAVE_PATH = os.environ.get("SAVE_PATH") or None
PLAN_EVERY = max(1, int(os.environ.get("A2E_PLAN_EVERY", "2")))
COC_EVERY_S = float(os.environ.get("A2E_COC_EVERY_S", "2.0"))
HINT_MIN_M = float(os.environ.get("A2E_HINT_MIN_M", "5.0"))
SAVE_EVERY = max(1, int(os.environ.get("A2E_SAVE_EVERY", "10")))
SAVE_RGB = os.environ.get("A2E_SAVE_RGB", "0") == "1"
HORIZON_S = cm.N_FUTURE * cm.DT


def get_entry_point():
    return "A2EAgent"


class A2EAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        parts = str(path_to_conf_file).split("+")
        self.socket_path = parts[0]
        self.save_name = parts[-1] if len(parts) > 1 else "a2e"
        self.sock = cm.connect(self.socket_path)
        cm.send_msg(self.sock, {"op": "ping"})
        info = cm.recv_msg(self.sock)
        self.slots = list(info["cameras"])
        unknown = [s for s in self.slots if s not in cm.SLOT_SENSOR]
        if unknown:
            raise RuntimeError(f"model wants camera slot(s) {unknown} with no Bench2Drive source")
        print(f"[a2e] model server {self.socket_path}: {info}; plan every {PLAN_EVERY} ticks, "
              f"CoC every {COC_EVERY_S} s", flush=True)

        self.step = -1
        self.initialized = False
        self.pid = cm.PIDController()
        self.tracker = cm.PoseTracker()
        self.frames = cm.FrameBuffer(self.slots)
        self.plan = None                 # (64, 2) ego frame of plan_step
        self.plan_step = -1
        self.coc, self.coc_step = None, -10 ** 9
        self.last_compass = 0.0
        self.meta = {}
        self.metric_info = {}
        self.t_model = 0.0
        self.n_plans = 0

        self.save_path = None
        if SAVE_PATH is not None:
            self.save_path = pathlib.Path(SAVE_PATH) / self.save_name
            (self.save_path / "meta").mkdir(parents=True, exist_ok=True)
            if SAVE_RGB:
                (self.save_path / "rgb_front").mkdir(exist_ok=True)

    def _init(self):
        # The map's geo-reference from the first route point, known in both
        # frames; then the DENSE plan (1 m, `_plan_gps_HACK`) in world metres.
        # Dense, not the 50 m keypoints of `_global_plan`: the dataset's
        # `command_near` is the road option of the segment the ego is on (near
        # target 1-7 m ahead, bench2drive.py), which only a dense route gives.
        loc = self._global_plan_world_coord[0][0].location
        g0 = self._global_plan[0][0]
        self.lat_ref, self.lon_ref = cm.solve_latlon_ref(g0["lat"], g0["lon"], loc.x, loc.y)
        self.planner = cm.RoutePlanner(4.0, 50.0)
        self.planner.set_route(
            (cm.gps_to_location(g["lat"], g["lon"], self.lat_ref, self.lon_ref), cmd.value)
            for g, cmd in self._plan_gps_HACK)
        print(f"[a2e] route: {len(self.planner.route)} points, geo-reference "
              f"({self.lat_ref:.6f}, {self.lon_ref:.6f})", flush=True)
        self.initialized = True

    def sensors(self):
        sensors = [dict(type="sensor.camera.rgb", id=sid, roll=0.0, pitch=0.0,
                        width=cm.IMG_W, height=cm.IMG_H, fov=cm.CAMERA_FOV, **cm.CAMERA_RIG[sid])
                   for sid in sorted({cm.SLOT_SENSOR[s] for s in self.slots})]
        sensors += [
            dict(type="sensor.other.imu", id="IMU", x=cm.GPS_X, y=0.0, z=0.0,
                 roll=0.0, pitch=0.0, yaw=0.0, sensor_tick=0.05),
            dict(type="sensor.other.gnss", id="GPS", x=cm.GPS_X, y=0.0, z=0.0,
                 roll=0.0, pitch=0.0, yaw=0.0, sensor_tick=0.01),
            dict(type="sensor.speedometer", id="SPEED", reading_frequency=cm.SIM_HZ),
        ]
        return sensors

    # -- per tick --------------------------------------------------------------

    def tick(self, input_data):
        self.step += 1
        self.frames.push({slot: cm.student_frame(input_data[cm.SLOT_SENSOR[slot]][1], slot)
                          for slot in self.slots})
        speed = float(input_data["SPEED"][1]["speed"])
        compass = float(input_data["IMU"][1][-1])
        if math.isnan(compass):          # happens for a few frames
            compass = self.last_compass
        self.last_compass = compass
        self.tracker.update(speed, compass)
        lat, lon = input_data["GPS"][1][:2]
        pos = cm.gps_to_location(lat, lon, self.lat_ref, self.lon_ref)
        self.planner.run_step(pos)
        cmd = self.planner.command_ahead(max(speed * HORIZON_S, HINT_MIN_M))
        return dict(speed=speed, compass=compass, pos=pos, command=cmd)

    def _request_plan(self, tick):
        hist_xyz, hist_rot = self.tracker.history()
        refresh = (self.step - self.coc_step) / cm.SIM_HZ >= COC_EVERY_S
        req = {"op": "plan", "hint": cm.hint_command(tick["command"]),
               "coc": None if refresh else self.coc,
               "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot}
        for slot, arr in self.frames.window().items():
            req[f"frames|{slot}"] = arr
        t0 = time.time()
        cm.send_msg(self.sock, req)
        rep = cm.recv_msg(self.sock)
        if "error" in rep:
            raise RuntimeError(f"model server: {rep['error']}")
        self.t_model += time.time() - t0
        self.n_plans += 1
        if refresh:
            self.coc, self.coc_step = rep["coc"], self.step
            print(f"[a2e] step {self.step} v={tick['speed']:.1f} hint={rep['hint_text']!r} "
                  f"CoC: {self.coc}", flush=True)
        self.plan, self.plan_step = rep["traj"].astype(np.float64), self.step
        self.meta.update(hint=rep["hint_text"], coc=self.coc, coc_step=self.coc_step,
                         coc_terminated=bool(rep["coc_terminated"]),
                         t_plan=float(rep["t_total"]), t_coc=float(rep["t_coc"]))

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick = self.tick(input_data)
        if self.plan is None or self.step - self.plan_step >= PLAN_EVERY:
            self._request_plan(tick)

        age = self.step - self.plan_step
        traj_now = self.tracker.to_current(self.plan, age)
        wps = cm.controller_waypoints(traj_now, elapsed_s=age / cm.SIM_HZ)
        steer, throttle, brake, pid_meta = self.pid.control_pid(wps, tick["speed"])

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(np.clip(throttle, 0.0, 0.75))
        control.brake = float(np.clip(brake, 0.0, 1.0))

        self.metric_info[self.step] = self.get_metric_info()
        if self.save_path is not None and self.step % SAVE_EVERY == 0:
            self.meta.update(pid_meta, step=self.step, command=int(tick["command"]),
                             steer=control.steer, throttle=control.throttle, brake=control.brake,
                             plan_age_ticks=age, waypoints=wps.tolist(),
                             plan=self.plan[::5].tolist())
            self.save()
        return control

    def save(self):
        frame = self.step // SAVE_EVERY
        with open(self.save_path / "meta" / f"{frame:04d}.json", "w") as f:
            json.dump(self.meta, f, indent=1)
        if SAVE_RGB:
            rgb = self.frames.buf[-1]["camera_front_wide_120fov" if "camera_front_wide_120fov"
                                      in self.slots else self.slots[0]]
            cv2.imwrite(str(self.save_path / "rgb_front" / f"{frame:04d}.jpg"), rgb[:, :, ::-1])
        if self.step % (SAVE_EVERY * 20) == 0:
            self._save_metric_info()

    def _save_metric_info(self):
        # read by Bench2Drive's tools/efficiency_smoothness_benchmark.py
        with open(self.save_path / "metric_info.json", "w") as f:
            json.dump(self.metric_info, f)

    def destroy(self):
        if getattr(self, "save_path", None) is not None and getattr(self, "metric_info", None):
            self._save_metric_info()
        if getattr(self, "n_plans", 0):
            print(f"[a2e] {self.n_plans} plans over {self.step + 1} ticks, "
                  f"{self.t_model / self.n_plans:.2f} s/plan", flush=True)
        sock = getattr(self, "sock", None)
        if sock is not None:
            sock.close()
            self.sock = None

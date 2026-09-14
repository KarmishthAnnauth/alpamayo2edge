"""Ground-truth-grounded scoring for the chain-of-causation (D-038).

The teacher's trace is an opinion, not a label. Measured on the 19,930 cached
windows against the driver's actual 6.4 s future (`gt_future_xyz`):

  - stated TURN direction matches the driver 56% of the time, is OPPOSITE 17%
    (the teacher was never given a route; at an intersection it is guessing)
  - when it says SLOW / YIELD / ADAPT_SPEED, the driver held or gained speed
    about half the time (a conservative narrator)
  - the driver brakes / steers within 6.4 s in 70% of windows where the teacher
    names a hazard - and in 69% of windows where it names none. The hazard
    mentions carry no information about what the driver did (off-path cones,
    obstacles only the 30-degree tele camera can see)

So a reward that pays for matching the teacher pays for narration. This module
grades a trace against what the driver did instead: the stated maneuver must be
consistent with the speed profile and the path, a stated direction must be the
one taken (and the student is told the route, so it is a fair question), and
naming an object is rewarded only when the driver reacted to something.

All inputs come from the cache: `gt_future_xyz` (64 x 3, ego frame, 10 Hz,
x forward, +y LEFT - confirmed from the teacher's own "turn left" labels, median
end bearing +36 deg) and the trace text. No model, no GPU.
"""
from __future__ import annotations
import numpy as np

from .coc_score import CLEAR, OBJECTS, objects, parse

REASONS = ("VEHICLE", "VRU", "CONSTRUCTION", "SIGNAL", "SIGN", "JUNCTION")   # things a driver reacts to
TURN_DEG, LANE_M = 20.0, 2.0


def kinematics(gt_future_xyz) -> dict:
    """What the driver did over the horizon, in a few discrete facts."""
    g = np.asarray(gt_future_xyz, dtype=float)[:, :2]
    v = np.linalg.norm(np.diff(g, axis=0), axis=1) * 10.0            # m/s at 10 Hz
    v0, v1 = float(v[:5].mean()), float(v[-10:].mean())
    dv, vmin = v1 - v0, float(v.min())
    bearing = float(np.degrees(np.arctan2(g[-1, 1], g[-1, 0])))    # + = left
    ylat = float(g[-1, 1])
    speed = ("stops" if vmin < 0.5 else "brakes" if dv < -1.5 else
             "speeds" if dv > 1.5 else "holds")
    if abs(bearing) > TURN_DEG:
        lateral = "turn_left" if bearing > 0 else "turn_right"
    elif abs(ylat) > LANE_M:
        lateral = "lane_left" if ylat > 0 else "lane_right"
    else:
        lateral = "straight"
    # A reaction strong enough to need an explanation: a stop, a hard brake,
    # or a real lateral move. Ordinary driving (the 70% base rate at looser
    # thresholds) is deliberately NOT a reaction here.
    reacted = vmin < 0.5 or dv < -2.5 or abs(ylat) > 1.5
    return dict(v0=v0, v1=v1, dv=dv, vmin=vmin, bearing=bearing, ylat=ylat,
                speed=speed, lateral=lateral, reacted=reacted,
                braked_from_speed=(speed in ("stops", "brakes") and v0 > 3.0))


def route_hint(k: dict) -> str:
    """The direction the driver took, phrased as a route instruction for the
    student's `<|route_start|>...<|route_end|>` slot. Direction only - it must
    not leak the speed profile, which is what the student is being graded on."""
    return {"turn_left": "Turn left ahead", "turn_right": "Turn right ahead",
            "lane_left": "Change to the left lane ahead",
            "lane_right": "Change to the right lane ahead",
            "straight": "Continue straight"}[k["lateral"]]


def kinematic_term(man: str | None, k: dict) -> float:
    """+1 consistent, -1 contradicted, 0 uncheckable. FOLLOW/KEEP are checked
    against the one thing they promise: that nothing required braking."""
    s = k["speed"]
    if man == "STOP":        return 1.0 if s == "stops" else -1.0
    if man == "SLOW":        return 1.0 if s in ("brakes", "stops") else (-1.0 if s == "speeds" else 0.0)
    if man == "YIELD":       return 1.0 if s in ("brakes", "stops") else (-1.0 if s == "speeds" else 0.0)
    if man == "ADAPT_SPEED": return 1.0 if s != "holds" else -0.5
    if man == "ACCELERATE":  return 1.0 if s == "speeds" else (-1.0 if s in ("brakes", "stops") else 0.0)
    if man == "TURN":        return 1.0 if k["lateral"].startswith("turn") else -1.0
    if man == "LANE_CHANGE": return 1.0 if k["lateral"].startswith("lane") else -1.0
    if man in ("FOLLOW", "KEEP"):
        return -1.0 if k["braked_from_speed"] else 0.5
    if man == "NUDGE":       return 0.0            # sub-lane; a 6.4 s offset cannot check it
    return -0.5                                    # unparsed


def direction_term(man: str | None, d: str | None, k: dict) -> float:
    """Against the direction the driver took. The student was told the route,
    so silence where a direction was taken costs, and a stated direction on a
    straight run costs."""
    lat = k["lateral"]
    if lat == "straight":
        return -0.5 if (man in ("TURN", "LANE_CHANGE") and d in ("left", "right")) else 0.0
    taken = "left" if lat.endswith("left") else "right"
    if d is None:
        return -0.5
    return 1.0 if d == taken else -1.0


def hazard_term(objs: set, k: dict) -> float:
    """Naming a reason should predict a reaction. named & reacted +1;
    named & nothing happened -1 (the off-path cone); silent & reacted -0.5;
    silent & nothing happened +0.5."""
    named = bool(objs & set(REASONS))
    if named:
        return 1.0 if k["reacted"] else -1.0
    return -0.5 if k["reacted"] else 0.5


def gt_reward(student_text: str, terminated: bool, n_tokens: int, k: dict,
              teacher_text: str, w) -> tuple[float, dict]:
    """Scalar reward + its components. `w` = cfg.stage1_rl.reward."""
    if not terminated or n_tokens > int(w.max_tokens):
        return float(w.fail), {"fail": True}
    s = parse(student_text)
    kin = kinematic_term(s["maneuver"], k)
    dr = direction_term(s["maneuver"], s["direction"], k)
    hz = hazard_term(s["objects"], k)
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    r = (float(w.kin) * kin + float(w.dir) * dr + float(w.hazard) * hz
         + float(w.teacher) * tm)
    return r, {"kin": kin, "dir": dr, "hazard": hz, "teacher": tm,
               "student_maneuver": s["maneuver"], "fail": False}


def gt_metrics(text: str, k: dict) -> dict:
    """Teacher-free evaluation fields for one trace."""
    s = parse(text)
    kt = kinematic_term(s["maneuver"], k)
    return {
        "consistent": None if kt == 0.0 else kt > 0,              # checkable claims only
        "gt_false_clear": ((s["maneuver"] in ("KEEP", "ACCELERATE") or bool(CLEAR.search(text)))
                           if k["braked_from_speed"] else None),
        "hazard_ungrounded": (not k["reacted"]) if (s["objects"] & set(REASONS)) else None,
        "direction_ok": (None if k["lateral"] == "straight" else
                         s["direction"] == ("left" if k["lateral"].endswith("left") else "right")),
        "student_maneuver": s["maneuver"],
    }


# ---------------------------------------------------------- trajectory ----

def ade_xy(pred_xyz, gt_xyz) -> float:
    """Mean L2 over the horizon, x/y only (open_loop.ade for one trajectory)."""
    p = np.asarray(pred_xyz, dtype=float)[:, :2]
    g = np.asarray(gt_xyz, dtype=float)[:, :2]
    n = min(len(p), len(g))
    return float(np.linalg.norm(p[:n] - g[:n], axis=1).mean())


def traj_reward(ade: float | None, student_text: str, terminated: bool, n_coc_tokens: int,
                k: dict, teacher_text: str, w) -> tuple[float, dict]:
    """The recipe's shape, adapted (D-039).

    NVIDIA: reward = -w_l2 * ADE / 3 (+ comfort, + judged reasoning), and -1
    outright when ADE >= 3 m, the CoT is missing, or the judge fails. Ours keeps
    the continuous ADE term and the hard failures, but NOT the 3 m gate: the
    student's token path - and the teacher's own, 3.77 m ADE_1 on the gate
    windows - sits around it, and a gate there would leave every group flat.
    ADE is capped at `ade_cap` instead, so the term lives in [-1, 0]. The
    kinematic and direction terms are small teacher-free shaping on the CoC;
    they are not the objective.
    """
    if not terminated or n_coc_tokens > int(w.max_tokens) or ade is None:
        return float(w.fail), {"fail": True}
    cap = float(w.ade_cap)
    r_ade = -min(float(ade), cap) / cap
    s = parse(student_text)
    kin = kinematic_term(s["maneuver"], k)
    dr = direction_term(s["maneuver"], s["direction"], k)
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    r = (float(w.ade) * r_ade + float(w.kin) * kin + float(w.dir) * dr + float(w.teacher) * tm)
    return r, {"ade": float(ade), "r_ade": r_ade, "kin": kin, "dir": dr, "teacher": tm,
               "student_maneuver": s["maneuver"], "fail": False}

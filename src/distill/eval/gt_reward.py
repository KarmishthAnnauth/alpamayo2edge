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
# A turn is a change of HEADING, read off the last 0.5 s of the path, not the
# bearing of the end point: a gentle curve carries 20 deg of bearing at 100 m
# without being a turn. 40 deg of final heading fires on 18% of val windows
# and catches 14 of the teacher's 21 TURN labels; the teacher itself under-
# labels turns (4%) because it narrates the approach ("stop", "keep distance").
TURN_DEG = 40.0
# There is NO lane-change class in the route. Run 2's hint fired "change lane"
# on 33% of windows (any 2 m offset at 60-100 m is road curvature) against the
# teacher's 3%, and the student parroted it at 3.3x. No 6.4 s path detector
# recovers the teacher's LANE_CHANGE windows (0/16 for arc-residual variants),
# and Alpamayo's own nav mode carries turns, not lane changes. Lateral offset
# is kept only to check a lane-change claim the student volunteers.
LANE_M = 2.0


def kinematics(gt_future_xyz) -> dict:
    """What the driver did over the horizon, in a few discrete facts."""
    g = np.asarray(gt_future_xyz, dtype=float)[:, :2]
    v = np.linalg.norm(np.diff(g, axis=0), axis=1) * 10.0            # m/s at 10 Hz
    v0, v1 = float(v[:5].mean()), float(v[-10:].mean())
    dv, vmin = v1 - v0, float(v.min())
    bearing = float(np.degrees(np.arctan2(g[-1, 1], g[-1, 0])))    # + = left
    ylat = float(g[-1, 1])
    d = np.diff(g, axis=0)
    heads = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    h_end = float(np.median(heads[-5:])) if len(heads) >= 5 else float(heads[-1])
    speed = ("stops" if vmin < 0.5 else "brakes" if dv < -1.5 else
             "speeds" if dv > 1.5 else "holds")
    if abs(h_end) > TURN_DEG:
        lateral = "turn_left" if h_end > 0 else "turn_right"
    else:
        lateral = "straight"
    # A reaction strong enough to need an explanation: a stop, a hard brake,
    # or a real lateral move. Ordinary driving (the 70% base rate at looser
    # thresholds) is deliberately NOT a reaction here.
    reacted = vmin < 0.5 or dv < -2.5 or abs(ylat) > 1.5
    return dict(v0=v0, v1=v1, dv=dv, vmin=vmin, bearing=bearing, ylat=ylat, h_end=h_end,
                speed=speed, lateral=lateral, reacted=reacted,
                braked_from_speed=(speed in ("stops", "brakes") and v0 > 3.0))


def route_hint(k: dict) -> str:
    """The direction the driver took, phrased as a route instruction for the
    student's `<|route_start|>...<|route_end|>` slot. Direction only - it must
    not leak the speed profile, which is what the student is being graded on."""
    return {"turn_left": "Turn left ahead", "turn_right": "Turn right ahead",
            "straight": "Continue straight"}[k["lateral"]]


def kinematic_term(man: str | None, k: dict, strict: bool = False) -> float:
    """+1 consistent, -1 contradicted, 0 uncheckable. FOLLOW/KEEP are checked
    against the one thing they promise: that nothing required braking.

    `strict` is the REWARD scale (run 5, D-042); the default is the MEASUREMENT
    scale and must not change, or `gt_metrics` stops being comparable across
    runs. Two rules differ, and both are about refusing to pay the hedge:

      FOLLOW / KEEP   +0.5 -> 0.0 when the driver did not brake from speed.
                      Measured on the 1046 val windows that free credit lands in
                      74% of them, and FOLLOW/KEEP is over half of what the
                      student says. It is a reward for narrating the status quo.
      ADAPT_SPEED     +1.0 -> +0.5. It scores +1 whenever the speed changed at
                      all, which ties or beats the best specific claim in 63% of
                      windows while never risking worse than -0.5. A hedge that
                      ties the truth teaches the student to hedge.

    What is left is one scale: +1 a checkable claim that held, 0 unverifiable or
    unremarkable, -1 contradicted. A correct specific claim now strictly beats
    every hedge, and a dull window still offers nothing for over-claiming - in a
    holds/straight window the best any maneuver scores is 0.
    """
    s = k["speed"]
    # D-049: under `strict`, a speed claim on a window where the driver HELD
    # speed is not "unverifiable" - the driver verifiably did not act on it.
    # RL run 6 found the hole: SLOW scored +1 on braking windows, -1 only on
    # speeding ones and 0 on the 55% of windows where nothing happens, so
    # "slow down" had positive expected reward everywhere and the policy said
    # it on 187 of 500 val windows by step 75 (teacher: 15). -0.5, not -1: a
    # cautionary claim that was not needed is worse than silence but not as
    # bad as "clear" on a braking window.
    hold = -0.5 if strict else 0.0
    if man == "STOP":        return 1.0 if s == "stops" else -1.0
    if man == "SLOW":        return 1.0 if s in ("brakes", "stops") else (-1.0 if s == "speeds" else hold)
    if man == "YIELD":       return 1.0 if s in ("brakes", "stops") else (-1.0 if s == "speeds" else hold)
    if man == "ADAPT_SPEED": return (0.5 if strict else 1.0) if s != "holds" else -0.5
    if man == "ACCELERATE":  return 1.0 if s == "speeds" else (-1.0 if s in ("brakes", "stops") else hold)
    if man == "TURN":        return 1.0 if k["lateral"].startswith("turn") else -1.0
    if man == "LANE_CHANGE": return 1.0 if abs(k["ylat"]) > LANE_M else -0.5   # a volunteered claim
    if man in ("FOLLOW", "KEEP"):
        return -1.0 if k["braked_from_speed"] else (0.0 if strict else 0.5)
    if man == "NUDGE":       return 0.0            # sub-lane; a 6.4 s offset cannot check it
    return -0.5                                    # unparsed


def direction_term(man: str | None, d: str | None, k: dict) -> float:
    """Against the direction the driver took. The student was told the route
    (turns only), so silence where a turn was taken costs, and a stated turn on
    a straight run costs. A volunteered lane-change direction is checked against
    the sign of the lateral offset when there is one, and ignored otherwise."""
    lat = k["lateral"]
    if man == "LANE_CHANGE" and d in ("left", "right"):
        if abs(k["ylat"]) < 1.5:
            return 0.0
        return 1.0 if d == ("left" if k["ylat"] > 0 else "right") else -1.0
    if lat == "straight":
        return -0.5 if (man == "TURN" and d in ("left", "right")) else 0.0
    taken = "left" if lat.endswith("left") else "right"
    if d is None:
        return -0.5
    return 1.0 if d == taken else -1.0


def hazard_term(objs: set, k: dict) -> float:
    """Naming a reason should predict a reaction. named & reacted +1;
    named & nothing happened -1 (the off-path cone); silent & reacted -0.5;
    silent & nothing happened +0.5.

    WEIGHT 0 from run 5 on (D-042), and kept only as the `hazard_ungrounded`
    metric. `reacted` is true in 762 of 1046 val windows, and within a GRPO group
    the window is fixed, so this pays +1.5 for naming any object in 73% of groups
    and -1.5 in 27%: a net +0.69 bounty on mentioning something. That is D-038's
    narration failure re-entering through the ground-truth side. Tightening the
    threshold does not fix it - `reacted` is already the strict definition."""
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
    # `reward.strict` (D-042's hedge-free scale; run 5 wired it for the perspan
    # path only, this mode kept paying FOLLOW/KEEP +0.5 on quiet windows).
    kin = kinematic_term(s["maneuver"], k, strict=bool(getattr(w, "strict", False)))
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
    # `reward.strict` (D-042's hedge-free scale; run 5 wired it for the perspan
    # path only, this mode kept paying FOLLOW/KEEP +0.5 on quiet windows).
    kin = kinematic_term(s["maneuver"], k, strict=bool(getattr(w, "strict", False)))
    dr = direction_term(s["maneuver"], s["direction"], k)
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    r = (float(w.ade) * r_ade + float(w.kin) * kin + float(w.dir) * dr + float(w.teacher) * tm)
    return r, {"ade": float(ade), "r_ade": r_ade, "kin": kin, "dir": dr, "teacher": tm,
               "student_maneuver": s["maneuver"], "fail": False}


# ------------------------------------------------------------ per-span ----

def self_consistency_term(man: str | None, direction: str | None, student_xyz) -> float:
    """Does the CoC describe the plan it precedes? The same kinematic and
    direction rules, pointed at the kinematics of the student's OWN decoded
    trajectory instead of the driver's. Teacher-free and GT-free: the only term
    that can build the CoC -> trajectory chain D-040 found missing (the stage-1
    student's plan is f(images); forcing "turn left" vs "turn right" moves the
    decoded heading by 2-6 deg). Range [-2, +2].

    Scored on the `strict` scale (D-042): calling one's own plan FOLLOW because
    it does not brake is not a description of it, and the hedge-tie argument
    holds with the student's plan in place of the driver's."""
    ks = kinematics(student_xyz)
    return kinematic_term(man, ks, strict=True) + direction_term(man, direction, ks)


def perspan_reward(ade: float | None, student_xyz, student_text: str, terminated: bool,
                   n_coc_tokens: int, k: dict, teacher_text: str, w) -> tuple[float, float, dict]:
    """Two rewards for one joint rollout (run 4, D-040): `(r_traj, r_coc, fields)`.

    r_traj  = ade * (-min(ADE, cap) / cap) + self_consistency * sc
              -> the TRAJECTORY span's advantage
    r_coc   = kin/dir/hazard/teacher text rules against the driver's future
              (the `gt` mode grader) + self_consistency * sc
              -> the COC span's advantage

    Run 5 (D-042) keeps the shape and changes what is paid for. Measured over the
    75 logged steps of run 4, the CoC reward broke down as kin +0.173 (34% of the
    positive total), teacher-match +0.146 (29%), hazard +0.127 (25%), self
    +0.063 (12%), dir -0.017. Two of those are the narration D-038 rejected:
    `hazard` pays for naming an object, and its anchor `reacted` is true in 73%
    of the 1046 val windows, so within a group naming beats silence in 3 groups
    of 4 (expected pull +0.69 per unit weight); `teacher` pays for agreeing with
    a trace whose turn direction is opposite 17% of the time. Both go to weight
    0, `kin` moves to the hedge-free `strict` scale, and what remains -
    kin + dir + self-consistency - is grounded in the driver's own future and in
    the student's own plan, nothing else.

    Run 3 applied one ADE-based advantage to both spans and moved the CoC not at
    all - the plan does not depend on the text, so the ADE credit reaching the
    CoC tokens was noise. Here each span is paid for what it can control, and
    the self-consistency term `sc` is on both: the CoC is rewarded for
    describing the plan, the plan for matching the words. A failure
    (unterminated CoC, no decodable plan) is `fail` on both.
    """
    if not terminated or n_coc_tokens > int(w.max_tokens) or ade is None or student_xyz is None:
        return float(w.fail), float(w.fail), {"fail": True}
    cap = float(w.ade_cap)
    r_ade = -min(float(ade), cap) / cap
    s = parse(student_text)
    kin = kinematic_term(s["maneuver"], k, strict=True)      # reward scale (D-042)
    dr = direction_term(s["maneuver"], s["direction"], k)
    hz = hazard_term(s["objects"], k)                        # weight 0 from run 5 on
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    sc = self_consistency_term(s["maneuver"], s["direction"], student_xyz)
    w_sc = float(getattr(w, "self_consistency", 0.0) or 0.0)
    r_traj = float(w.ade) * r_ade + w_sc * sc
    r_coc = (float(w.kin) * kin + float(w.dir) * dr + float(w.hazard) * hz
             + float(w.teacher) * tm + w_sc * sc)
    return r_traj, r_coc, {"ade": float(ade), "r_ade": r_ade, "kin": kin, "dir": dr,
                           "hazard": hz, "teacher": tm, "self": sc,
                           "student_maneuver": s["maneuver"], "fail": False}

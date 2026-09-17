"""GT-grounded CoC reward (D-038): pure functions on synthetic futures. No GPU."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.eval.gt_reward import (direction_term, gt_metrics, gt_reward, hazard_term,  # noqa: E402
                                    kinematic_term, kinematics, route_hint,
                                    self_consistency_term)

W = SimpleNamespace(kin=1.0, dir=0.5, hazard=0.5, teacher=0.25, fail=-1.0, max_tokens=48)


def future(v0, v1, bearing_deg=0.0, ylat_end=0.0, n=64):
    """Ego-frame path: speed ramps v0 -> v1 over 6.4 s, heading toward bearing."""
    t = np.arange(1, n + 1) / 10.0
    v = np.linspace(v0, v1, n)
    dist = np.cumsum(v) / 10.0
    th = np.radians(bearing_deg) * (t / t[-1])
    x = np.cumsum(np.diff(dist, prepend=0) * np.cos(th))
    y = np.cumsum(np.diff(dist, prepend=0) * np.sin(th)) + ylat_end * (t / t[-1]) ** 2
    return np.stack([x, y, np.zeros(n)], 1)


def test_kinematics_classes():
    assert kinematics(future(10, 10))["speed"] == "holds"
    assert kinematics(future(10, 0.0))["speed"] == "stops"
    assert kinematics(future(10, 6))["speed"] == "brakes"
    assert kinematics(future(5, 10))["speed"] == "speeds"
    assert kinematics(future(8, 8, bearing_deg=70))["lateral"] == "turn_left"      # +y = left
    assert kinematics(future(8, 8, bearing_deg=-70))["lateral"] == "turn_right"
    assert kinematics(future(12, 12, ylat_end=3.5))["lateral"] == "straight"       # no lane class
    assert kinematics(future(12, 12, bearing_deg=15))["lateral"] == "straight"     # a gentle curve
    assert kinematics(future(12, 12))["lateral"] == "straight"


def test_route_hint_is_direction_only():
    assert route_hint(kinematics(future(8, 0.0, bearing_deg=70))) == "Turn left ahead"
    assert route_hint(kinematics(future(12, 12, ylat_end=-3.5))) == "Continue straight"   # lane changes are not route
    assert route_hint(kinematics(future(10, 0.0))) == "Continue straight"      # a stop leaks nothing


def test_kinematic_term_signs():
    k_stop, k_hold, k_speed = kinematics(future(10, 0.0)), kinematics(future(10, 10)), kinematics(future(4, 10))
    assert kinematic_term("STOP", k_stop) == 1.0 and kinematic_term("STOP", k_hold) == -1.0
    assert kinematic_term("SLOW", k_speed) == -1.0 and kinematic_term("SLOW", k_hold) == 0.0
    assert kinematic_term("ACCELERATE", k_speed) == 1.0
    assert kinematic_term("KEEP", k_hold) == 0.5
    assert kinematic_term("KEEP", kinematics(future(10, 2))) == -1.0     # braked from speed: the GT false-clear
    assert kinematic_term("NUDGE", k_hold) == 0.0
    assert kinematic_term(None, k_hold) == -0.5


def test_direction_term_uses_direction_taken():
    k = kinematics(future(8, 8, bearing_deg=70))
    assert direction_term("TURN", "left", k) == 1.0
    assert direction_term("TURN", "right", k) == -1.0
    assert direction_term("TURN", None, k) == -0.5                     # silent, though told the route
    ks = kinematics(future(12, 12))
    assert direction_term("TURN", "left", ks) == -0.5                  # claimed a turn on a straight run
    assert direction_term("LANE_CHANGE", "left", ks) == 0.0            # no offset: a volunteered claim is ignored
    assert direction_term("LANE_CHANGE", "left", kinematics(future(12, 12, ylat_end=3.5))) == 1.0
    assert direction_term("LANE_CHANGE", "right", kinematics(future(12, 12, ylat_end=3.5))) == -1.0
    assert direction_term("FOLLOW", None, ks) == 0.0


def test_hazard_term_rewards_grounded_mentions_only():
    reacted, calm = kinematics(future(12, 0.0)), kinematics(future(12, 12))
    assert hazard_term({"CONSTRUCTION"}, reacted) == 1.0
    assert hazard_term({"CONSTRUCTION"}, calm) == -1.0                  # the cone beside the road
    assert hazard_term(set(), reacted) == -0.5
    assert hazard_term(set(), calm) == 0.5


def test_gt_reward_end_to_end():
    k = kinematics(future(12, 0.0))                                    # driver stopped from 12 m/s
    r_good, s = gt_reward("Stop for the red traffic light since the signal is red", True, 12, k, "", W)
    r_bad, _ = gt_reward("Keep lane since the lane is clear ahead", True, 10, k, "", W)
    assert r_good > 0 > r_bad and s["kin"] == 1.0 and s["hazard"] == 1.0
    assert gt_reward("Stop", False, 64, k, "", W)[0] == -1.0            # unterminated


def test_gt_metrics_fields():
    m = gt_metrics("Keep lane since the lane is clear ahead", kinematics(future(12, 1.0)))
    assert m["consistent"] is False and m["gt_false_clear"] is True
    m2 = gt_metrics("Nudge left due to the cones", kinematics(future(10, 10)))
    assert m2["consistent"] is None and m2["hazard_ungrounded"] is True


# ---- D-039: trajectory-through reward ----
from distill.eval.gt_reward import ade_xy, traj_reward  # noqa: E402

WT = SimpleNamespace(ade=1.0, ade_cap=3.0, kin=0.25, dir=0.25, teacher=0.0, fail=-2.0, max_tokens=48)


def test_ade_xy_ignores_z_and_truncates():
    g = np.zeros((64, 3)); p = np.zeros((64, 3)); p[:, 0] = 1.5; p[:, 2] = 99.0
    assert abs(ade_xy(p, g) - 1.5) < 1e-9
    assert abs(ade_xy(p[:32], g) - 1.5) < 1e-9


def test_traj_reward_shape_and_failures():
    k = kinematics(future(12, 0.0))
    r0, s0 = traj_reward(0.0, "Stop for the red light", True, 8, k, "", WT)
    r3, _ = traj_reward(3.0, "Stop for the red light", True, 8, k, "", WT)
    r9, _ = traj_reward(9.0, "Stop for the red light", True, 8, k, "", WT)
    assert s0["r_ade"] == 0.0 and r0 > r3 and abs(r3 - r9) < 1e-9        # capped, not gated
    assert r0 == 0.25 * 1.0 + 0.25 * 0.0                                  # kin consistent, straight
    assert traj_reward(None, "Stop", True, 8, k, "", WT)[0] == -2.0          # no trajectory decoded
    assert traj_reward(1.0, "Stop", False, 64, k, "", WT)[0] == -2.0        # unterminated CoC
    assert traj_reward(1.0, "Stop", True, 49, k, "", WT)[0] == -2.0         # too long
    # any success beats any failure, even the worst success
    worst, _ = traj_reward(99.0, "Accelerate", True, 8, k, "", WT)         # ade -1, kin -1
    assert worst > -2.0


# ---- run 4: self-consistency against the student's own plan (D-040) ----

def test_self_consistency_stop_vs_own_plan():
    assert self_consistency_term("STOP", None, future(10, 0.0)) == 1.0     # says stop, plan stops
    assert self_consistency_term("STOP", None, future(5, 10)) == -1.0      # says stop, plan speeds up
    assert self_consistency_term("ACCELERATE", None, future(5, 10)) == 1.0
    assert self_consistency_term("ACCELERATE", None, future(10, 0.0)) == -1.0


def test_self_consistency_direction_against_own_plan():
    assert self_consistency_term("TURN", "left", future(8, 8, bearing_deg=70)) == 2.0    # kin + dir
    assert self_consistency_term("TURN", "left", future(8, 8, bearing_deg=-70)) == 0.0   # kin +1, dir -1
    assert self_consistency_term("TURN", "right", future(8, 8)) == -1.5                  # straight plan
    # Silent about a turn while calling the plan KEEP. Was 0.0, when KEEP's free
    # +0.5 cancelled the -0.5 for the missing direction; on the strict reward
    # scale (D-042) KEEP earns nothing and the silence costs.
    assert self_consistency_term("KEEP", None, future(8, 8, bearing_deg=70)) == -0.5


# ---- run 5: the hedge-free reward scale, and the metric scale it must not move (D-042) ----

HEDGES = ("FOLLOW", "KEEP", "ADAPT_SPEED", "NUDGE")
SPECIFIC = ("STOP", "SLOW", "YIELD", "ACCELERATE", "TURN")


def test_measurement_scale_is_unchanged_by_the_strict_flag():
    """`gt_metrics` reads the DEFAULT scale. If these move, every run's
    gt_consistent stops being comparable with every other run's."""
    k_quiet = kinematics(future(12, 12))                  # holds, straight, no braking
    assert kinematic_term("FOLLOW", k_quiet) == 0.5
    assert kinematic_term("KEEP", k_quiet) == 0.5
    assert kinematic_term("ADAPT_SPEED", kinematics(future(10, 0.0))) == 1.0


def test_strict_scale_stops_paying_the_hedge():
    k_quiet = kinematics(future(12, 12))                  # nothing happened
    assert kinematic_term("FOLLOW", k_quiet, strict=True) == 0.0
    assert kinematic_term("KEEP", k_quiet, strict=True) == 0.0
    k_brake = kinematics(future(12, 4))                   # braked from speed
    assert kinematic_term("FOLLOW", k_brake, strict=True) == -1.0     # still contradicted
    k_stop = kinematics(future(10, 0.0))
    assert kinematic_term("ADAPT_SPEED", k_stop, strict=True) == 0.5  # was 1.0
    assert kinematic_term("ADAPT_SPEED", k_quiet, strict=True) == -0.5


def test_strict_scale_makes_the_correct_specific_claim_win():
    """The defect being fixed: ADAPT_SPEED tied or beat the best specific answer
    in 63% of val windows. Under `strict` it must lose to the truth."""
    for fut, right in ((future(10, 0.0), "STOP"), (future(12, 4), "SLOW"), (future(5, 12), "ACCELERATE")):
        k = kinematics(fut)
        best = kinematic_term(right, k, strict=True)
        assert best == 1.0
        for h in HEDGES:
            assert kinematic_term(h, k, strict=True) < best, (right, h)


def test_strict_scale_offers_nothing_for_over_claiming_a_dull_window():
    """A holds/straight window: no maneuver may score above 0, or the reward
    teaches the student to invent events."""
    k = kinematics(future(12, 12))
    for m in HEDGES + SPECIFIC + ("LANE_CHANGE", None):
        assert kinematic_term(m, k, strict=True) <= 0.0, m


def test_self_consistency_uses_the_strict_scale():
    """Describing one's own non-braking plan as FOLLOW earns nothing."""
    assert self_consistency_term("FOLLOW", None, future(12, 12)) == 0.0
    assert self_consistency_term("STOP", None, future(10, 0.0)) == 1.0

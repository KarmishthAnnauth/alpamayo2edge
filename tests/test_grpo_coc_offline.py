"""GRPO-on-CoC (D-037): the pure parts. No GPU, no model."""
import math
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.train_grpo_coc import coc_reward, group_advantages  # noqa: E402

W = SimpleNamespace(maneuver=1.0, direction=0.5, objects=0.5, false_clear=1.0,
                    fail=-1.0, max_tokens=48)
T = "Nudge left due to construction cones blocking the right side of our lane"


def test_reward_exact_match_is_full_marks():
    r, s = coc_reward("Nudge left due to the cones on the right side of our lane", T, True, 20, W)
    assert math.isclose(r, 1.0 + 0.5 + 0.5)          # maneuver + direction + full object recall


def test_reward_passive_answer_with_hazard_is_penalised():
    r, s = coc_reward("Keep lane since the lane is clear ahead", T, True, 14, W)
    assert s["false_clear"] is True
    assert r < 0                                       # 0 + 0 + 0 - 1


def test_reward_no_terminator_is_hard_fail():
    r, s = coc_reward("Nudge left due to construction cones", T, False, 30, W)
    assert r == -1.0 and s["fail"]


def test_reward_too_long_is_hard_fail():
    r, _ = coc_reward("Nudge left due to construction cones", T, True, 49, W)
    assert r == -1.0


def test_reward_empty_teacher_is_unscored():
    r, s = coc_reward("Stop for the red light", "", True, 8, W)
    assert r == 0.0 and s["unscored"]


def test_group_advantages_normalised_and_skips_flat_groups():
    assert group_advantages([1.0, 1.0, 1.0, 1.0]) is None
    a = group_advantages([2.0, 0.0, 0.0, 0.0])
    assert math.isclose(sum(a), 0.0, abs_tol=1e-9)
    assert a[0] > 0 and all(x < 0 for x in a[1:])


# ---- run 4: per-span rewards (D-040) ----
import numpy as np                                                       # noqa: E402
from distill.eval.gt_reward import kinematics, perspan_reward             # noqa: E402

WP = SimpleNamespace(ade=1.0, ade_cap=8.0, kin=1.0, dir=0.5, hazard=0.5, teacher=0.25,
                     self_consistency=0.5, fail=-2.0, max_tokens=48)


def _path(v0, v1, n=64):
    v = np.linspace(v0, v1, n)
    x = np.cumsum(v) / 10.0
    return np.stack([x, np.zeros(n), np.zeros(n)], 1)


def test_perspan_good_plan_bad_coc_gets_opposite_sign_advantages():
    """A rollout whose plan is close to GT but whose CoC contradicts the driver
    must be favoured on the trajectory span and punished on the CoC span."""
    gt = _path(10, 0.0)                              # the driver stops
    k = kinematics(gt)
    good_plan = _path(10, 0.5)                       # ~GT
    bad_plan = _path(10, 12)                         # keeps going: ADE large
    # rollout A: good plan, wrong words; rollout B: bad plan, right words
    ta, ca, _ = perspan_reward(0.4, good_plan, "Accelerate since the road is clear", True, 10, k, "", WP)
    tb, cb, _ = perspan_reward(6.0, bad_plan, "Stop for the red light", True, 10, k, "", WP)
    adv_t = group_advantages([ta, tb])
    adv_c = group_advantages([ca, cb])
    assert adv_t[0] > 0 > adv_t[1]                   # trajectory span: A wins
    assert adv_c[0] < 0 < adv_c[1]                   # CoC span: B wins


def test_perspan_failure_is_fail_on_both_spans():
    k = kinematics(_path(10, 10))
    t, c, s = perspan_reward(None, None, "Keep lane", False, 64, k, "", WP)
    assert t == c == -2.0 and s["fail"]
    t, c, s = perspan_reward(1.0, _path(10, 10), "Keep lane", True, 49, k, "", WP)
    assert t == c == -2.0


def test_perspan_self_consistency_reaches_both_spans():
    """Same CoC, same GT, same ADE: only the student's own plan differs, so the
    two rewards differ by exactly the weighted self-consistency term."""
    k = kinematics(_path(10, 10))                    # driver holds speed
    text = "Stop for the pedestrian"
    t1, c1, s1 = perspan_reward(3.0, _path(10, 0.0), text, True, 10, k, "", WP)   # plan stops
    t2, c2, s2 = perspan_reward(3.0, _path(5, 10), text, True, 10, k, "", WP)     # plan speeds up
    assert s1["self"] == 1.0 and s2["self"] == -1.0
    assert t1 - t2 == pytest.approx(0.5 * 2.0)
    assert c1 - c2 == pytest.approx(0.5 * 2.0)

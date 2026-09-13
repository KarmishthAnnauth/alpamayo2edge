"""GRPO-on-CoC (D-037): the pure parts. No GPU, no model."""
import math
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

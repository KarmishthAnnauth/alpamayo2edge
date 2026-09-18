"""The phase-1 recipe as configured (PHASE1_RUNBOOK.md). Pins the settings a
fresh session relies on when it runs `scripts/run_phase1.sh`, so a stray edit
shows up here rather than 10 GPU-hours later."""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_phase1_recipe_is_the_run8_recipe():
    c = yaml.safe_load(CFG.read_text())
    s = c["stage1"]
    assert s["traj_prefix"] == "gt"                      # D-043
    assert s["route_hint"] is True                       # D-043
    assert s["filter_contradictions"] is True            # D-043
    assert s["prefix_mask_prob"] > 0 and s["prefix_noise_bins"] >= 64   # D-045
    assert s["image_dropout"] == 0.0                     # D-047: the CoC gets every target
    assert s["select_on"] == "coc_gt" and s["coc_gt_windows"] >= 100   # D-047/D-050
    assert s["save_every_epoch"] is True                 # D-046
    assert s["loss_weights"]["gt_ce"] > 0 and s["loss_weights"]["traj_kl"] == 0
    assert s["loss_weights"]["text_kl"] > 0 and s["loss_weights"]["struct_ce"] > 0
    assert s["maneuver_sampling"]["enabled"] is False    # D-036 outcome
    assert len(c["data"]["cameras"]) == 4                # D-036: perception limit at 1 camera


def test_phase15_recipe_has_both_hedges_charged():
    r = yaml.safe_load(CFG.read_text())["stage1_rl"]["reward"]
    assert r["mode"] == "gt" and r["strict"] is True     # D-047: CoC-only, driver-grounded
    assert r["hazard"] == 0 and r["teacher"] == 0        # D-042
    assert r["unverifiable"] < 0                         # D-050: NUDGE charged

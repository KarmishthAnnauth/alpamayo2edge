"""Bench2Drive window builder, on ONE extracted clip, CPU only.

What is pinned here and why:

  - coordinate conventions: CARLA is left-handed (+y right); the student frame
    is +y LEFT (D-038). A positive CARLA `steer` is a RIGHT turn, so the
    ego-frame future must go to NEGATIVE y with a negative yaw. Getting this
    wrong trains the flow head on mirrored turns and no loss curve would show it.
  - speeds from consecutive poses match the annotation's `speed`.
  - `action_to_traj(traj_to_action(future))` reproduces the future - the
    action space is the teacher's, and the round trip is only tight when the
    reference point is the rear axle (`bench2drive.REAR_AXLE_OFFSET_M`).
  - shapes and keys match what `student/context.py` and `data/frames.py` read.
  - splits never put a route on both sides.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("alpamayo1_5")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.config import load_config                     # noqa: E402
from distill.data import bench2drive as b2d                # noqa: E402
from distill.data import frames as frames_mod              # noqa: E402
from distill.eval import gt_reward                         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
#: A clip with a sustained positive-steer stretch (17 frames > 0.15 from frame
#: 131, ~3.5 m/s). Any other `.done` clip is used when it is absent. It is an
#: aggressive clip (launches from standstill through a turn, a 0.7-steer swerve
#: at 10 m/s, an emergency stop at 16 m/s^2), which is what the round-trip
#: test's regimes below are about.
PREFERRED_CLIP = "Accident_Town05_Route219_Weather11"
#: Ordinary driving, for the strict round-trip bound. First `.done` match wins.
GENTLE_PREFIXES = ("HighwayExit_", "NonSignalizedJunctionLeftTurn_", "LaneChange_")
SPEC = Path("/bulk/users/vla/alpamayo2edge/distill_cache/traj_tokenizer_spec.pt")


def _clip_dir() -> Path:
    root = b2d.DEFAULT_ROOT
    if not root.exists():
        pytest.skip(f"{root} not mounted")
    if b2d.is_done(root / PREFERRED_CLIP):
        return root / PREFERRED_CLIP
    done = b2d.list_done_clips(root)
    if not done:
        pytest.skip("no extracted Bench2Drive clip with a .done marker")
    return done[0]


@pytest.fixture(scope="module")
def params():
    return b2d.WindowParams.from_cfg(load_config(REPO / "configs" / "default.yaml"))


@pytest.fixture(scope="module")
def space():
    return b2d.load_action_space()


@pytest.fixture(scope="module")
def anno(params):
    return b2d.load_clip_anno(_clip_dir(), params.cameras)


@pytest.fixture(scope="module")
def windows(anno, params, space):
    """Every window of the clip, with frames only on the first two (speed)."""
    out = []
    reader = b2d._FrameReader(anno.clip_dir, params)
    for i, t0 in enumerate(b2d.anchor_frames(anno.n, params.stride)):
        out.append(b2d.build_window(anno, t0, params, space, reader, with_frames=i < 2))
        reader.clear()
    assert out, "clip too short for one window"
    return out


# ---- pure functions --------------------------------------------------------

def test_parse_clip_id():
    m = b2d.parse_clip_id("NonSignalizedJunctionLeftTurn_Town12_Route1017_Weather7")
    assert (m.scenario, m.town, m.route, m.weather) == (
        "NonSignalizedJunctionLeftTurn", "Town12", 1017, 7)
    assert m.route_key == "Town12/Route1017"
    m = b2d.parse_clip_id("ParkedObstacle_Town10HD_Route372_Weather8")
    assert m.town == "Town10HD" and m.route == 372
    with pytest.raises(ValueError):
        b2d.parse_clip_id("not_a_clip")


def test_tele_crop_box_is_30deg_16_9():
    x0, y0, x1, y1 = b2d.tele_crop_box()
    assert (x0, x1) == (494, 1106)                      # 2 * 1142.5 * tan(15 deg) = 612 px
    assert (x1 - x0, y1 - y0) == (612, 344)
    assert abs((x1 - x0) / (y1 - y0) - 16 / 9) < 0.01
    assert (x0 + x1) / 2 == 800 and (y0 + y1) / 2 == 450


def test_anchor_frames_keep_history_and_future_inside():
    a = b2d.anchor_frames(196, 20)
    assert a[0] == 15 and a[-1] + 64 <= 195 and a == [15, 35, 55, 75, 95, 115, 131][:-1]
    assert b2d.anchor_frames(80, 20) == [15] and b2d.anchor_frames(79, 20) == []


def test_route_hint_uses_the_existing_vocabulary():
    vocab = {gt_reward.route_hint({"lateral": k}) for k in ("turn_left", "turn_right", "straight")}
    for c in (-1, 1, 2, 3, 4, 5, 6):
        assert b2d.route_hint_text(c) in vocab
    assert b2d.route_hint_text(1) == "Turn left ahead"
    assert b2d.route_hint_text(2) == "Turn right ahead"
    assert b2d.route_hint_text(4) == b2d.route_hint_text(5) == "Continue straight"


def test_action_space_is_the_release_one(space):
    assert space.n_waypoints == 64 and abs(space.dt - 0.1) < 1e-9
    assert abs(float(space.accel_std) - 0.6810426736454882) < 1e-6      # fp32 buffers
    assert abs(float(space.curvature_std) - 0.026148280660833106) < 1e-7


# ---- one real clip ---------------------------------------------------------

def test_window_shapes_and_frame(anno, windows, params):
    w = windows[0]
    assert set(w.frames_student) == set(params.cameras)
    for cam in params.cameras:
        assert w.frames_student[cam].shape == (params.n_frames, *params.student_hw, 3)
        assert w.frames_student[cam].dtype == np.uint8
    assert w.raw_cameras["camera_front_tele_30fov"] == "rgb_front:tele_crop"
    assert tuple(w.data["ego_history_xyz"].shape) == (1, 1, 16, 3)
    assert tuple(w.data["ego_history_rot"].shape) == (1, 1, 16, 3, 3)
    assert tuple(w.data["ego_future_xyz"].shape) == (1, 1, 64, 3)
    assert tuple(w.data["ego_future_rot"].shape) == (1, 1, 64, 3, 3)
    assert w.gt_traj.shape == (64, 2) and w.gt_traj.dtype == np.float32
    assert w.gt_future_xyz.shape == (64, 3) and w.gt_future_rot.shape == (64, 3, 3)
    assert w.expert_controls.shape == (64, 3)
    assert w.t0_us == w.anchor_frame * 100_000
    # t0 is the origin with identity heading; history runs backwards along -x.
    h = w.data["ego_history_xyz"][0, 0].numpy()
    assert np.allclose(h[-1], 0.0) and np.allclose(w.data["ego_history_rot"][0, 0, -1].numpy(), np.eye(3), atol=1e-6)
    assert np.all(h[:-1, 0] <= 1e-6)
    # Rotations stay proper (det +1) after the reflection S R S.
    dets = np.linalg.det(w.gt_future_rot)
    assert np.allclose(dets, 1.0, atol=1e-4)
    # Forward motion: the future moves to +x when the car is moving.
    if w.ego_speed_t0 > 1.0:
        assert w.gt_future_xyz[9, 0] > 0.0


def test_speed_from_poses_matches_anno(anno, windows):
    worst = 0.0
    for w in windows:
        f = w.gt_future_xyz[:, :2]
        v = np.linalg.norm(np.diff(f, axis=0), axis=1) * 10.0      # between wp i+1 and i+2
        ref = anno.speed[w.anchor_frame + 2: w.anchor_frame + 65]
        worst = max(worst, float(np.median(np.abs(v - ref))))
    assert worst < 0.3, f"median |pose speed - anno speed| {worst:.3f} m/s"


def test_positive_steer_is_a_right_turn_negative_y(anno, params, space):
    """CARLA +steer = right. In the +y-LEFT ego frame a right turn is -y, -yaw;
    the raw CARLA frame (before the flip) shows the opposite sign."""
    st = anno.steer
    run, best = 0, (0, 0)
    for i, s in enumerate(st):
        run = run + 1 if s > 0.15 else 0
        if run > best[0]:
            best = (run, i - run + 1)
    length, start = best
    if length < 8:
        pytest.skip("clip has no sustained positive-steer stretch")
    t0 = min(max(15, start - 2), anno.n - 65)
    w = b2d.build_window(anno, t0, params, space, with_frames=False)
    k = min(start - t0 - 1 + length, 63)                    # end of the stretch
    f, rot = w.gt_future_xyz, w.gt_future_rot
    assert f[k, 1] < -0.5, f"expected negative y for a right turn, got {f[k, 1]:.2f}"
    assert b2d.yaw_of(rot[k]) < -0.05
    # And in the CARLA frame, before the flip, the same point is at +y.
    fut = np.arange(t0 + 1, t0 + 65)
    T = anno.world2ego[t0][None] @ np.linalg.inv(anno.world2ego[fut])
    assert T[k, 1, 3] > 0.5 and np.isclose(T[k, 1, 3], -f[k, 1], atol=1e-3)


def _round_trip(w, space):
    rec = b2d.action_to_traj(space, w.gt_traj, w.data)
    return np.linalg.norm(rec[:, :2] - w.gt_future_xyz[:, :2], axis=1)


def test_action_round_trip_regimes(anno, windows, space):
    """action_to_traj(traj_to_action(future)) ~ future, by regime.

    The release action space is exact on the teacher's own data (0.002-0.24 m
    at 6.4 s on cached PhysicalAI-AV windows) and on ordinary Bench2Drive
    driving once the ego frame sits at the rear axle (`test_gentle_clip_round_trip`).
    Two Bench2Drive regimes fall outside what a unicycle with the release
    regularisation can represent, and they are pinned here at their measured
    level so a regression in the CONVERSION (which would move the steady
    windows too) is still caught:
      launch    the car starts below 1 m/s and steers in the first second; the
                space clamps curvature to zero under 0.6 m/s, so a few degrees
                of early heading error become 1-1.7 m over the horizon.
      steady    the car never drops below 1 m/s: 0.16-0.4 m at 6.4 s, the top
                of that range being a 0.7-steer swerve at 10 m/s (tyre slip).
    Before the rear-axle shift the steady windows sat at 0.5-1.1 m.
    """
    steady, launch = [], []
    for w in windows:
        vmin = float(anno.speed[w.anchor_frame: w.anchor_frame + 65].min())
        (steady if vmin > 1.0 else launch).append(_round_trip(w, space))
    assert steady, "no window with the car moving throughout"
    last = [float(e[-1]) for e in steady]
    mean = [float(e.mean()) for e in steady]
    assert max(last) < 0.5, f"steady round trip {max(last):.3f} m at 6.4 s"
    assert max(mean) < 0.3
    for e in launch:
        assert float(e[-1]) < 2.0 and float(e[:20].max()) < 0.3   # the loss is late and lateral


def test_round_trip_floor_over_clips(params, space):
    """The action-space floor over the first 20 extracted clips, and the pin
    that the rear-axle ego origin is what makes it tight.

    Measured 2026-09-28 on 1372 windows of 150 clips (err at 6.4 s): all
    windows p50 0.03 m / p90 0.50 m; windows where the car never drops below
    1 m/s p50 0.07 / p90 0.46 / max 0.92 m. With the CARLA actor origin instead
    of the rear axle the heading leads the travel direction by 3 deg on
    average and the moving windows sit at 0.5-1.1 m.
    """
    root = b2d.DEFAULT_ROOT
    if not root.exists():
        pytest.skip(f"{root} not mounted")
    clips = b2d.list_done_clips(root)[:20]
    if len(clips) < 5:
        pytest.skip("fewer than 5 extracted clips")
    err = {0.0: [], b2d.REAR_AXLE_OFFSET_M: []}     # (turning?, steady?, err) per window
    for c in clips:
        for l_r, acc in err.items():
            a = b2d.load_clip_anno(c, params.cameras, rear_axle_offset_m=l_r)
            for _, w in b2d.iter_clip_windows(a, params, space, with_frames=False):
                turning = abs(float(np.degrees(b2d.yaw_of(w.gt_future_rot)[-1]))) > 20.0
                steady = float(a.speed[w.anchor_frame: w.anchor_frame + 65].min()) > 1.0
                acc.append((turning, steady, float(_round_trip(w, space)[-1])))
    ours = err[b2d.REAR_AXLE_OFFSET_M]
    all_err = np.array([e for _, _, e in ours])
    steady = np.array([e for _, st, e in ours if st])
    assert len(steady) >= 5
    assert np.median(all_err) < 0.2
    assert np.median(steady) < 0.3 and np.percentile(steady, 90) < 0.7, \
        f"steady p50 {np.median(steady):.2f} p90 {np.percentile(steady, 90):.2f}"
    # The reference point only matters where the car turns (slip ~ l_r * curvature):
    # measured 0.535 -> 0.196 m median on |yaw| > 20 deg windows of these clips.
    turn0 = [e for t, _, e in err[0.0] if t]
    turn1 = [e for t, _, e in ours if t]
    if len(turn1) >= 5:
        assert np.median(turn0) > 1.5 * np.median(turn1), "the rear-axle shift no longer matters"


def test_gentle_clip_round_trip(params, space):
    """Ordinary driving: every window of a HighwayExit/junction/lane-change
    clip stays under the measured cross-clip steady max (0.92 m)."""
    root = b2d.DEFAULT_ROOT
    if not root.exists():
        pytest.skip(f"{root} not mounted")
    cands = [p for p in b2d.list_done_clips(root) if p.name.startswith(GENTLE_PREFIXES)]
    if not cands:
        pytest.skip("no gentle clip extracted")
    a = b2d.load_clip_anno(cands[0], params.cameras)
    errs = [float(_round_trip(w, space)[-1])
            for _, w in b2d.iter_clip_windows(a, params, space, with_frames=False)]
    assert errs and max(errs) < 1.0, f"{cands[0].name}: round trip {max(errs):.3f} m at 6.4 s"


def test_labels_and_route(anno, windows, space):
    vocab = {gt_reward.route_hint({"lateral": k}) for k in ("turn_left", "turn_right", "straight")}
    for w in windows:
        assert w.route_hint in vocab and w.route_hint_kin in vocab
        assert w.command in b2d.ROAD_OPTION
        t0 = w.anchor_frame
        assert w.should_brake == bool(anno.should_brake[t0 + 1: t0 + 65].any())
        assert w.should_brake_frac == pytest.approx(anno.should_brake[t0 + 1: t0 + 65].mean())
        assert w.should_brake_t0 == bool(anno.should_brake[t0])
        assert np.allclose(w.expert_controls[:, 1], anno.steer[t0: t0 + 64])
        assert w.ego_speed_t0 == pytest.approx(anno.speed[t0])
        assert w.action_floor_m == pytest.approx(float(_round_trip(w, space)[-1]), abs=1e-4)
        assert w.meta.clip_id == anno.meta.clip_id and w.weather == anno.weather


# ---- cache + dataset -------------------------------------------------------

class _StubContext:
    """Records what the context builder was handed; no tokenizer needed."""
    def __init__(self):
        self.calls = []

    def build(self, window, coc_text=None, traj_bins=None, nav_text=None,
              for_generation=False):
        self.calls.append(dict(coc_text=coc_text, traj_bins=traj_bins, nav_text=nav_text,
                               for_generation=for_generation,
                               cams=sorted(window.frames_student),
                               hist=tuple(window.data["ego_history_xyz"].shape)))
        return {"input_ids": torch.arange(5 + len(self.calls)), "pixel_values": None,
                "image_grid_thw": None, "coc_span": None, "struct_span": None,
                "traj_span": None, "history_span": (0, 1), "n_prompt": 3}


@pytest.fixture(scope="module")
def cache(tmp_path_factory, windows, params):
    root = tmp_path_factory.mktemp("b2d_cache")
    for i, w in enumerate(windows[:2]):
        b2d.save_window(root, i, w, quality=params.jpeg_quality)
    b2d.write_splits(root, [windows[0].clip_id])
    return root


def test_cache_layout_matches_labeler(cache, windows, params):
    cid = windows[0].clip_id
    assert (cache / cid / "00.npz").exists() and (cache / cid / "00_input.npz").exists()
    from distill.data.dataset import discover_shards
    assert [p.name for p in discover_shards(cache)] == ["00.npz", "01.npz"]
    z = b2d.load_targets(cache / cid / "00.npz")
    assert z["gt_traj"].shape == (64, 2) and z["gt_future_xyz"].shape == (64, 3)
    assert str(z["scenario"]) == windows[0].meta.scenario and str(z["town"]) == windows[0].meta.town
    assert int(z["anchor_frame"]) == windows[0].anchor_frame
    back = frames_mod.load_window_input(cache / cid / "00_input.npz", cid)
    assert set(back.frames_student) == set(params.cameras)
    for cam in params.cameras:
        assert back.frames_student[cam].shape == windows[0].frames_student[cam].shape
        # JPEG q92 round trip: same picture, not a channel swap.
        d = np.abs(back.frames_student[cam].astype(int) - windows[0].frames_student[cam].astype(int))
        assert d.mean() < 6.0
    assert tuple(back.data["ego_history_xyz"].shape) == (1, 1, 16, 3)


def test_dataset_and_collate(cache, windows):
    ctx = _StubContext()
    ds = b2d.Bench2DriveDataset(cache, ctx, n_flow_samples=3)
    assert len(ds) == 2
    it = ds[0]
    assert it["route_hint"] == windows[0].route_hint and it["coc_text"] == ""
    assert ctx.calls[-1]["nav_text"] == windows[0].route_hint
    assert ctx.calls[-1]["for_generation"] is True and ctx.calls[-1]["coc_text"] is None
    assert tuple(it["hist_xyz"].shape) == (16, 3) and tuple(it["hist_rot"].shape) == (16, 3, 3)
    assert it["img_drop"] is False
    # Synthesised GT flow tuples in the teacher's convention (D-012).
    t, a_t, v = it["flow_t"], it["flow_a_t"], it["flow_v"]
    a1 = torch.as_tensor(it["gt_traj"])
    a0 = (a_t - t.view(-1, 1, 1) * a1) / (1 - t.view(-1, 1, 1))
    assert torch.allclose(v, a1 - a0, atol=1e-4) and t.min() >= 0 and t.max() < 0.999
    batch = b2d.collate_b2d([ds[0], ds[1]], pad_id=0)
    assert tuple(batch["gt_traj"].shape) == (2, 64, 2)
    assert tuple(batch["gt_future_xyz"].shape) == (2, 64, 3)
    assert tuple(batch["flow_a_t"].shape) == (6, 64, 2) and batch["flow_owner"].tolist() == [0, 0, 0, 1, 1, 1]
    items = [ds[0], ds[1]]
    batch = b2d.collate_b2d(items, pad_id=0)
    lens = [int(it["student"]["input_ids"].shape[0]) for it in items]
    assert tuple(batch["input_ids"].shape) == (2, max(lens))
    assert batch["attention_mask"].sum(1).tolist() == lens
    assert batch["clip_ids"] == [windows[0].clip_id] * 2 and batch["window_idx"] == [0, 1]
    # Kinematic route hint source, and a CoC lookup with teacher forcing.
    ds2 = b2d.Bench2DriveDataset(cache, ctx, route_source="kinematic",
                                 coc_lookup=lambda c, w: "Stay in lane.", for_generation=False)
    it2 = ds2[1]
    assert it2["route_hint"] == windows[1].route_hint_kin and it2["coc_text"] == "Stay in lane."
    assert ctx.calls[-1]["for_generation"] is False


def test_splits_never_share_a_route():
    clips = [f"{s}_Town{t}_Route{r}_Weather{w}"
             for s in ("Accident", "LaneChange", "HighwayExit")
             for t in ("03", "12") for r in range(4) for w in (1, 2)]
    sp = b2d.make_splits(clips, val_fraction=0.10, seed=0)
    assert not set(sp["train"]) & set(sp["val"]) and len(sp["train"]) + len(sp["val"]) == len(clips)
    tr = {b2d.parse_clip_id(c).route_key for c in sp["train"]}
    va = {b2d.parse_clip_id(c).route_key for c in sp["val"]}
    assert not tr & va
    for scn, v in sp["by_scenario"].items():
        assert v["val_routes"] >= 1
    assert sp == b2d.make_splits(clips, 0.10, 0), "not deterministic"
    assert sp["val"] != b2d.make_splits(clips, 0.10, 1)["val"]


@pytest.mark.skipif(not SPEC.exists(), reason="teacher tokenizer spec not on this box")
def test_teacher_tokenizers_accept_the_windows(anno, params, space):
    """The history tokenizer (48 bins) and the future tokenizer (128 bins,
    emission order) run on B2D windows; decoding the GT bins lands within the
    quantisation floor of the future."""
    from distill.teacher.wrapper import swap_action_dims
    spec = torch.load(SPEC, weights_only=False)
    # The steadiest window: the token path shares the action space's regimes.
    t0 = max(b2d.anchor_frames(anno.n, params.stride),
             key=lambda t: anno.speed[t: t + 65].min())
    w = b2d.build_window(anno, t0, params, space, with_frames=False,
                         tokenizer_fn=spec["tokenizer_fn"])
    hb = spec["hist_tokenize_fn"](w.data["ego_history_xyz"], w.data["ego_history_rot"])
    assert tuple(hb.shape) == (1, 48) and int(hb.min()) >= 0 and int(hb.max()) < 1000
    assert w.gt_traj_token_ids.shape == (128,) and w.gt_traj_token_ids.max() < 3000
    toks = swap_action_dims(torch.as_tensor(w.gt_traj_token_ids)).view(1, -1)
    xyz, _, _ = spec["detokenizer_fn"](w.data["ego_history_xyz"][:, -1],
                                       w.data["ego_history_rot"][:, -1], toks)
    err = np.linalg.norm(xyz[0, :, :2].numpy() - w.gt_future_xyz[:, :2], axis=1)
    # Action-space floor (< 0.5 m on a steady window) + the 0.13 m bin floor.
    assert err[-1] < 1.0, f"token round trip {err[-1]:.3f} m at 6.4 s"

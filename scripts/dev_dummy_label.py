"""Dummy-weight harness for TeacherWrapper.label_window (HANDOFF task 1).

Builds a reduced-config, RANDOM-INIT Alpamayo2Super (a few tiny layers, real
tokenizer/processor/config from the released repo - no weight shards touched)
and runs the full label_window path on fabricated windows, then save_shard +
npz reload. Catches shapes, keys, device/dtype, the two-phase generation, and
the labeler's t-orientation (t = data weight, D-012) via noise recovery -
all without a Blackwell allocation, weights, or dataset streaming.

    python scripts/dev_dummy_label.py            # cuda if available (the Ada), else cpu
    python scripts/dev_dummy_label.py --device cpu
    python scripts/dev_dummy_label.py --kill-test   # SIGKILL mid-run + resume (task 2)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# HF env defaults for this box (D-019/D-020) - before any transformers import.
os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / "Karmishth/alpamayo2edge/models"))
_tok_file = Path.home() / ".config/alpamayo2edge/hf_token"
if "HF_TOKEN" not in os.environ and _tok_file.exists():
    os.environ["HF_TOKEN"] = _tok_file.read_text().strip()

sys.path.insert(0, "src")
import numpy as np
import torch

from distill.config import load_config
from distill.data.preprocess import Window
from distill.teacher.labeler import save_shard, shard_path
from distill.teacher.wrapper import TeacherWrapper

FEAT_LAYERS = [1, 3]  # reduced-depth stand-in for cfg feat_layers [7..63]


def build_dummy_teacher(repo: str, device: str):
    """Reduced-config random-init teacher. Real tokenizer + traj tokenizers + processor;
    tiny towers. Expert KV geometry kept equal to the VLM's (the D-004 contract)."""
    from alpamayo2_super.config import Alpamayo2SuperConfig
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    c = Alpamayo2SuperConfig.from_pretrained(repo)
    c._name_or_path = repo  # released config ships "" - tokenizer/processor resolution (D-022)

    t = c.vlm_config.text_config
    t.num_hidden_layers = 4
    t.hidden_size = 512
    t.num_attention_heads = 4      # head_dim stays 128 => mrope_section [24,20,20] still sums to 64
    t.num_key_value_heads = 2
    t.intermediate_size = 1024

    v = c.vlm_config.vision_config
    v.depth = 4
    v.deepstack_visual_indexes = [1, 2, 3]
    v.hidden_size = 128
    v.num_heads = 4
    v.intermediate_size = 256
    v.out_hidden_size = t.hidden_size

    e = c.expert_config.llm_config
    e.num_hidden_layers = t.num_hidden_layers      # KV geometry == VLM (D-004)
    e.num_key_value_heads = t.num_key_value_heads
    e.hidden_size = 256
    e.num_attention_heads = 2      # head_dim stays 128
    e.intermediate_size = 512
    c.expert_config.action_in_proj_cfg["hidden_size"] = 64

    # Keep prompts short: ~16 vision tokens per image instead of ~160.
    c.min_pixels, c.max_pixels = 4096, 16384

    torch.manual_seed(0)
    model = Alpamayo2Super(c).to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def make_wrapper(cfg, model, device: str) -> TeacherWrapper:
    """TeacherWrapper around an already-built model, bypassing from_pretrained."""
    w = TeacherWrapper.__new__(TeacherWrapper)
    w.cfg = cfg
    w.device = device
    w.model = model
    w.tconfig = model.config
    w.tokenizer = model.tokenizer
    w.future_traj_tokenizer = model.future_traj_tokenizer
    w._meta_vocab = None
    return w


def fabricate_window(cfg, clip_id: str, seed: int) -> Window:
    """One synthetic window in the exact load_physical_aiavdataset output format:
    random camera frames + a smooth constant-curvature drive (history ends at the
    ego origin with identity rotation, as traj_to_action assumes)."""
    from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES

    rng = np.random.default_rng(seed)
    v = float(rng.uniform(5.0, 12.0))        # m/s
    curv = float(rng.uniform(-0.02, 0.02))   # 1/m

    def drive(ts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        s = v * ts
        yaw = curv * s
        if abs(curv) > 1e-8:
            x, y = np.sin(yaw) / curv, (1.0 - np.cos(yaw)) / curv
        else:
            x, y = s, np.zeros_like(s)
        xyz = np.stack([x, y, np.zeros_like(x)], -1)
        cos, sin, zero, one = np.cos(yaw), np.sin(yaw), np.zeros_like(yaw), np.ones_like(yaw)
        rot = np.stack([np.stack([cos, -sin, zero], -1),
                        np.stack([sin, cos, zero], -1),
                        np.stack([zero, zero, one], -1)], -2)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32))[None, None]  # (1,1,T,...)
        return to_t(xyz), to_t(rot)

    hist_xyz, hist_rot = drive(np.arange(-15, 1) * 0.1)   # 16 steps ending at t0 = origin
    fut_xyz, fut_rot = drive(np.arange(1, 65) * 0.1)      # 64 future steps

    cameras = sorted(cfg.data.raw["cameras"], key=lambda n: CAMERA_NAMES_TO_INDICES[n])
    n_cam, n_frames = len(cameras), cfg.data.context_frames
    data = {
        "image_frames": torch.from_numpy(
            rng.integers(0, 256, (n_cam, n_frames, 3, 128, 128), dtype=np.uint8)),
        "camera_indices": torch.tensor([CAMERA_NAMES_TO_INDICES[n] for n in cameras]),
        "camera_names": cameras,
        "ego_history_xyz": hist_xyz, "ego_history_rot": hist_rot,
        "ego_future_xyz": fut_xyz, "ego_future_rot": fut_rot,
    }
    return Window(clip_id=clip_id, t0_us=5_000_000, data=data, frames_student={},
                  gt_future_xyz=fut_xyz[0, 0].numpy(),
                  gt_future_rot=fut_rot[0, 0].numpy())


def check_output(out, tconfig, cfg) -> None:
    """Shape/dtype/range contract of TeacherWindowOutput (labeler docstrings + D-012/014)."""
    T, K = tconfig.tokens_per_future_traj, cfg.teacher.topk_logits
    H, A = 64, 2
    kf = cfg.teacher.flow_targets_per_window

    def shp(name, tensor, want):
        assert tuple(tensor.shape) == tuple(want), f"{name}: {tuple(tensor.shape)} != {want}"

    shp("traj_token_ids", out.traj_token_ids, (T,))
    assert out.traj_token_ids.min() >= 0 and out.traj_token_ids.max() < tconfig.future_vocab_size
    shp("traj_topk_idx", out.traj_topk_idx, (T, K))
    assert out.traj_topk_idx.min() >= 0 and out.traj_topk_idx.max() < tconfig.future_vocab_size
    shp("traj_topk_logp", out.traj_topk_logp, (T, K))
    assert out.traj_topk_logp.max() <= 0, "top-k log-probs must be <= 0"
    assert out.coc_token_ids.numel() <= cfg.teacher.max_coc_tokens
    assert isinstance(out.coc_text, str) and isinstance(out.meta_action_text, str)
    assert isinstance(out.meta_action, int)
    assert sorted(out.feats) == FEAT_LAYERS
    for l, f in out.feats.items():
        shp(f"feat_{l}", f, (cfg.teacher.feat_pool_len, tconfig.vlm_config.text_config.hidden_size))
        assert torch.isfinite(f).all()
    shp("flow_t", out.flow_t, (kf,))
    assert 0 <= out.flow_t.min() and out.flow_t.max() < 0.999 + 1e-6
    shp("flow_a_t", out.flow_a_t, (kf, H, A))
    shp("flow_v", out.flow_v, (kf, H, A))
    assert torch.isfinite(out.flow_v).all()
    shp("traj_samples", out.traj_samples, (cfg.teacher.n_traj_samples, H, A))
    shp("gt_traj", out.gt_traj, (H, A))
    shp("gt_future_xyz", out.gt_future_xyz, (H, 3))
    for name in ("traj_token_ids", "traj_topk_idx", "traj_topk_logp", "flow_t", "flow_a_t",
                 "flow_v", "traj_samples", "gt_traj"):
        assert getattr(out, name).device.type == "cpu", f"{name} not on cpu"

    # t-orientation (D-012: x_t = t*data + (1-t)*noise, t = DATA weight). Recovering
    # noise = (x_t - t*a) / (1-t) must give ~N(0,1); a flipped t explodes near t=0.999.
    t = out.flow_t.view(-1, 1, 1).double()
    noise = (out.flow_a_t.double() - t * out.gt_traj.double()[None]) / (1 - t)
    m, s = noise.mean().item(), noise.std().item()
    assert abs(m) < 0.3 and 0.75 < s < 1.3, \
        f"recovered noise not ~N(0,1): mean={m:.3f} std={s:.3f} - t-orientation flipped?"


NPZ_KEYS = {"traj_token_ids": np.int32, "traj_topk_idx": np.int32, "traj_topk_logp": np.float16,
            "coc_token_ids": np.int32, "flow_t": np.float32, "flow_a_t": np.float32,
            "flow_v": np.float16, "traj_samples": np.float32, "gt_traj": np.float32,
            "gt_future_xyz": np.float32, "feat_layers": np.int32}


def check_shard(path: Path, out) -> None:
    with np.load(path) as z:
        for key, dt in NPZ_KEYS.items():
            assert key in z, f"shard missing {key}"
            assert z[key].dtype == dt, f"{key}: {z[key].dtype} != {np.dtype(dt)}"
        assert str(z["coc_text"]) == out.coc_text
        assert int(z["meta_action"]) == out.meta_action
        for l in FEAT_LAYERS:
            assert z[f"feat_{l}"].dtype == np.float16
            assert z[f"feat_{l}"].shape == tuple(out.feats[l].shape)
        np.testing.assert_array_equal(z["traj_token_ids"], out.traj_token_ids.numpy())
        np.testing.assert_allclose(z["flow_a_t"], out.flow_a_t.numpy(), rtol=1e-6)


KILL_CLIPS, KILL_MIN_SHARDS = 4, 3  # 4 clips x 2 windows; SIGKILL after >= 3 shards


def _kill_test_cfg(out_dir: Path):
    cfg = load_config()
    cfg.raw["paths"]["cache_root"] = str(out_dir)
    cfg.raw["teacher"]["feat_layers"] = FEAT_LAYERS
    return cfg


def run_kill_test_child(args) -> None:
    """Child: run the REAL run_labeling loop (skip logic, shard writes, manifest)
    over fabricated clips, with the dummy teacher patched in."""
    import distill.teacher.labeler as labeler

    cfg = _kill_test_cfg(Path(args.out))
    model = build_dummy_teacher(cfg.paths.teacher_repo, args.device)
    labeler.TeacherWrapper = lambda c: make_wrapper(c, model, args.device)
    labeler.iter_windows = lambda c, clip_id: (
        (w, fabricate_window(c, clip_id, seed=1000 + 10 * int(clip_id.rsplit("_", 1)[1]) + w))
        for w in range(c.data.windows_per_clip))
    labeler.run_labeling(cfg, [f"kill_clip_{i}" for i in range(KILL_CLIPS)])


def run_kill_test(args) -> None:
    """Parent: SIGKILL the labeling child mid-run, re-run it, and verify resume
    semantics - completed shards untouched, the rest labeled, manifest written."""
    out_dir = Path(args.out).absolute() / "kill_test"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = [sys.executable, __file__, "--kill-test-child",
           "--out", str(out_dir), "--device", args.device]
    total = KILL_CLIPS * _kill_test_cfg(out_dir).data.windows_per_clip

    print(f"kill test: labeling {KILL_CLIPS} clips ({total} shards), "
          f"SIGKILL after {KILL_MIN_SHARDS} ...")
    child = subprocess.Popen(cmd)
    deadline = time.time() + 600
    while len(list(out_dir.glob("*/*.npz"))) < KILL_MIN_SHARDS:
        assert child.poll() is None, "child exited before reaching the kill threshold"
        assert time.time() < deadline, "timed out waiting for shards"
        time.sleep(0.2)
    child.kill()  # SIGKILL: the no-cleanup worst case of a SLURM preemption
    child.wait()
    pre = {p: p.stat().st_mtime_ns for p in out_dir.glob("*/*.npz")}
    assert not (out_dir / "manifest.json").exists(), "manifest must only exist on completion"
    print(f"  killed with {len(pre)} shards on disk; re-running to completion ...")

    rerun = subprocess.run(cmd)
    assert rerun.returncode == 0, f"resume run failed (rc={rerun.returncode})"
    post = {p: p.stat().st_mtime_ns for p in out_dir.glob("*/*.npz")}
    assert len(post) == total, f"expected {total} shards after resume, got {len(post)}"
    relabeled = [p for p in pre if post[p] != pre[p]]
    assert not relabeled, f"resume re-labeled completed shards: {relabeled}"
    assert not list(out_dir.glob("*/*.tmp")), "orphan .tmp files left behind"
    assert (out_dir / "manifest.json").exists(), "manifest.json not written"
    print(f"PASS - kill test: {len(pre)} pre-kill shards skipped byte-untouched, "
          f"{total - len(pre)} labeled on resume, manifest written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/dummy_label_harness")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--kill-test", action="store_true",
                    help="mid-run SIGKILL + resume test of run_labeling (task 2)")
    ap.add_argument("--kill-test-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.kill_test_child:
        return run_kill_test_child(args)
    if args.kill_test:
        return run_kill_test(args)

    cfg = load_config()
    out_dir = Path(args.out).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.raw["paths"]["cache_root"] = str(out_dir)      # /data/... is unwritable (D-019)
    cfg.raw["teacher"]["feat_layers"] = FEAT_LAYERS    # real [7..63] exceeds the 4-layer dummy

    print(f"device={args.device} | building reduced random-init teacher ...")
    t0 = time.time()
    model = build_dummy_teacher(cfg.paths.teacher_repo, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  built in {time.time() - t0:.1f}s | {n_params / 1e6:.0f}M params (dummy)")
    teacher = make_wrapper(cfg, model, args.device)

    shards = []
    for i in range(args.windows):
        window = fabricate_window(cfg, clip_id=f"dummy_clip_{i}", seed=100 + i)
        t0 = time.time()
        out = teacher.label_window(
            window,
            k_flow=cfg.teacher.flow_targets_per_window,
            topk=cfg.teacher.topk_logits,
            max_coc=cfg.teacher.max_coc_tokens,
            n_traj_samples=cfg.teacher.n_traj_samples,
        )
        dt = time.time() - t0
        check_output(out, model.config, cfg)
        path = shard_path(out_dir, window.clip_id, 0)
        save_shard(path, out)
        check_shard(path, out)
        shards.append((path, out))
        print(f"  window {i}: label_window {dt:.1f}s | shard {path.stat().st_size / 1024:.0f} KB "
              f"| meta_action={out.meta_action!r} | coc {len(out.coc_text)} chars - OK")

    assert (out_dir / "meta_action_vocab.json").exists(), "meta_action_vocab.json not written"
    if len(shards) >= 2:
        assert not torch.equal(shards[0][1].gt_traj, shards[1][1].gt_traj), \
            "windows should differ"
    print("PASS - full label_window path + npz shard roundtrip on dummy weights")


if __name__ == "__main__":
    main()

"""Headroom: how far apart are the teacher and the untrained student?

Everything stage 1 and stage 2 can possibly achieve lives between these two
numbers, on the same clips, with the same metric:

    Alpamayo 1.5 (AV specialist)          -> the ceiling
    Cosmos 3 Edge zero-shot, domain "av"  -> the floor

Measure it BEFORE committing GPU weeks. A narrow gap is not a failure, it is a
finding — but it is one you want before training, not after, because the
response (higher LoRA rank, MLP targets, more data, a different teacher) is a
design change, not a tuning change.

Edge is not a blank slate here: `EMBODIMENT_TO_DOMAIN_ID` ships **"av": 1** with
a 9-D action space, and NVIDIA publishes AV inverse-dynamics cookbooks. So the
floor is a real model doing the real task, which is exactly what makes it the
honest baseline.

The two models never coexist in memory: each phase writes predictions to disk
and `report` compares the files.

CALIBRATION FIRST — read this before trusting any Edge number
--------------------------------------------------------------
The teacher path is fully determined by code we have: the expert's rollout is in
normalized action space and `UnicycleAccelCurvatureActionSpace.action_to_traj`
converts it to metres.

The Edge path has one genuine unknown: **the layout of the 9-D "av" action**.
The open framework ships the domain id and the raw width but no AV dataset
class, so the layout is inferred by analogy with every other pose embodiment in
the repo (`camera_pose` is also 9-D; DROID/UMI are `[Pos(3), Rot6d(6), Gripper]`),
giving frame-relative `[pos(3), rot6d(6)]`. That is an ASSUMPTION, marked
[ASSUMED] at `av_actions_to_xyz`.

It is also a *checkable* assumption, which is why `calibrate()` exists: run Edge
in `inverse_dynamics` mode over the window's PAST video, where the true ego
motion is already known from the dataset, and compare. If the decode is right,
recovered past ≈ known past to well under a metre. If it is wrong, that shows up
as a large calibration ADE and no prediction number should be believed. Run
`calibrate` before `edge`.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .open_loop import ade, min_ade

log = logging.getLogger(__name__)

AV_DOMAIN_NAME = "av"      # EMBODIMENT_TO_DOMAIN_ID["av"] == 1
AV_RAW_ACTION_DIM = 9      # EMBODIMENT_TO_RAW_ACTION_DIM["av"] == 9


# ---------------------------------------------------------------- teacher ----

@torch.no_grad()
def teacher_predictions(cfg, clip_ids: list[str], n_samples: int = 6) -> dict:
    """Alpamayo 1.5 future trajectories, in metres, ego frame.

    Uses the action expert's continuous rollout (`traj_samples`), NOT the
    discrete token path — deliberately. Phase-B discrete emission is the open
    D-022 risk; the expert rollout does not depend on it, so this number stays
    meaningful even if D-022 fails.
    """
    from ..data.preprocess import iter_windows
    from ..teacher.wrapper import TeacherWrapper

    tw = TeacherWrapper(cfg)
    space = tw.future_traj_tokenizer.action_space
    preds, gts, keys = [], [], []
    for clip_id in clip_ids:
        for w_idx, window in iter_windows(cfg, clip_id):
            out = tw.label_window(
                window,
                k_flow=1,                      # flow targets are not needed here
                topk=1,
                max_coc=cfg.teacher.max_coc_tokens,
                n_traj_samples=n_samples,
            )
            hist_xyz = window.data["ego_history_xyz"][0, 0]     # (T, 3)
            hist_rot = window.data["ego_history_rot"][0, 0]     # (T, 3, 3)
            k = out.traj_samples.shape[0]
            xyz, _ = space.action_to_traj(
                out.traj_samples.to(hist_xyz.device),
                hist_xyz.unsqueeze(0).expand(k, -1, -1),
                hist_rot.unsqueeze(0).expand(k, -1, -1, -1),
            )                                                   # (k, H, 3)
            preds.append(xyz.float().cpu().numpy())
            gts.append(np.asarray(window.gt_future_xyz, dtype=np.float32))
            keys.append(f"{clip_id}/{w_idx:02d}")
            log.info("teacher %s (%d/%d)", keys[-1], len(keys),
                     len(clip_ids) * cfg.data.windows_per_clip)
    return {"pred": np.stack(preds), "gt": np.stack(gts), "keys": keys,
            "model": "alpamayo1.5", "n_samples": n_samples}


# ------------------------------------------------------------------ edge ----

def av_actions_to_xyz(actions: torch.Tensor) -> torch.Tensor:
    """[ASSUMED] Edge's 9-D "av" action chunk -> cumulative ego-frame xyz.

    Assumed layout, per step: `[dx, dy, dz, r00, r10, r20, r01, r11, r21]` —
    a frame-relative position delta followed by the first two columns of the
    rotation matrix (rot6d), which is the layout every pose embodiment in
    cosmos-framework uses (`camera_pose` 9-D; DROID/UMI `[Pos, Rot6d, Gripper]`).

    Only the translation half is needed for ADE, and only x/y are scored, so the
    rotation columns are carried through untouched rather than integrated —
    a wrong rot6d convention cannot corrupt the metric, a wrong TRANSLATION
    convention can. `calibrate()` is what catches the latter.

    actions: (..., T, 9) frame-relative deltas. Returns (..., T, 3) cumulative
    positions in the ego frame of the first step.
    """
    if actions.shape[-1] < 3:
        raise ValueError(f"expected >=3 action dims, got {actions.shape[-1]}")
    return torch.cumsum(actions[..., :3].float(), dim=-2)


@dataclasses.dataclass
class EdgeAVRunner:
    """Zero-shot Cosmos 3 Edge on the AV embodiment.

    Follows `scripts/action_policy_server_libero.py::predict_policy_batch`,
    which is the framework's own action-inference path — same batch keys, same
    `generate_samples_from_batch` call, `domain_name="av"` instead of a robot.

    VALIDATE-ON-GPU: `camera`, `fps` and `resolution` below are OUR choices, not
    the framework's. Edge is native 480p; our cached frames are the configured
    `student_resolution`. If quality looks off, that is the first thing to check.
    """
    cfg: Any = None                    # distill.config.Cfg
    num_steps: int = 35
    guidance: float = 7.0
    seed: int = 0
    camera: str = "camera_front_wide_120fov"
    fps: int = 30

    def __post_init__(self):
        from cosmos_framework.data.generator.action.utils.domain_utils import (
            get_action_dim, get_domain_id,
        )
        self.domain_id = get_domain_id(AV_DOMAIN_NAME)
        self.raw_action_dim = get_action_dim(AV_DOMAIN_NAME)
        if self.raw_action_dim != AV_RAW_ACTION_DIM:
            raise RuntimeError(
                f"the shipped 'av' raw action dim is {self.raw_action_dim}, not "
                f"{AV_RAW_ACTION_DIM} — av_actions_to_xyz's layout assumption is "
                "keyed to 9-D and must be revisited")
        self.model = None

    def load(self):
        from huggingface_hub import snapshot_download
        from cosmos_framework.inference.model import Cosmos3OmniModel

        ckpt = snapshot_download(self.cfg.paths.student_repo)
        self.model = Cosmos3OmniModel.from_pretrained_dcp(ckpt).cuda().eval()
        return self

    def _video(self, window) -> torch.Tensor:
        """(C, T, H, W) uint8 from the window's front camera."""
        arr = window.frames_student[self.camera]        # (F, h, w, 3) uint8
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(3, 0, 1, 2)

    @torch.no_grad()
    def run(self, window, mode: str, action_length: int, prompt: str) -> torch.Tensor:
        """One window -> (T, 9) predicted actions.

        mode: "wam" jointly denoises future video and actions from the first
        frame (the forecasting mode, what we want for the gap);
        "inverse_dynamics" recovers actions from video that is fully given
        (what `calibrate` uses).
        """
        from cosmos_framework.data.generator.action.utils.transforms import (
            build_sequence_plan_from_mode,
        )
        from cosmos_framework.data.generator.action.utils.action_processing import (
            ActionProcessingRecord, make_batched_action_processing_fields,
        )

        video = self._video(window)
        plan = build_sequence_plan_from_mode(
            mode=mode,
            video_length=video.shape[1],
            action_length=action_length,
            has_text=True,
        )
        max_dim = int(self.model.model.net.config.action_dim)
        batch = {
            "video": [[video]],
            **make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=self.raw_action_dim,
                                       action_normalizer=None),
                batch_size=1,
            ),
            "action": [[torch.zeros(action_length, max_dim, dtype=torch.float32)]],
            "mode": [mode],
            "ai_caption": [prompt],
            "prompt": [prompt],
            "conditioning_fps": [torch.tensor(self.fps, dtype=torch.long)],
            "image_size": torch.tensor(
                [[video.shape[2], video.shape[3]]], device="cuda"),
            "domain_id": [torch.tensor(self.domain_id, dtype=torch.long)],
            "sequence_plan": [plan],
        }
        samples = self.model.generate_samples_from_batch(
            batch, guidance=self.guidance, seed=[self.seed],
            num_steps=self.num_steps, has_negative_prompt=False,
        )
        return samples["action"][0].float().squeeze(0)[:, : self.raw_action_dim]


DRIVE_PROMPT = "Continue the same driving scene with smooth natural motion."


@torch.no_grad()
def edge_predictions(cfg, clip_ids: list[str], horizon_steps: int = 64,
                     **runner_kw) -> dict:
    """Cosmos 3 Edge zero-shot future trajectories, in metres, ego frame."""
    from ..data.preprocess import iter_windows

    runner = EdgeAVRunner(cfg=cfg, **runner_kw).load()
    preds, gts, keys = [], [], []
    for clip_id in clip_ids:
        for w_idx, window in iter_windows(cfg, clip_id):
            act = runner.run(window, mode="wam", action_length=horizon_steps,
                             prompt=DRIVE_PROMPT)
            preds.append(av_actions_to_xyz(act).cpu().numpy()[None])   # (1, H, 3)
            gts.append(np.asarray(window.gt_future_xyz, dtype=np.float32))
            keys.append(f"{clip_id}/{w_idx:02d}")
            log.info("edge %s (%d)", keys[-1], len(keys))
    return {"pred": np.stack(preds), "gt": np.stack(gts), "keys": keys,
            "model": "cosmos3-edge-zeroshot", "n_samples": 1}


@torch.no_grad()
def calibrate(cfg, clip_ids: list[str], **runner_kw) -> dict:
    """Gate on `av_actions_to_xyz` before any prediction number is believed.

    Inverse dynamics recovers the ego motion that produced video we already
    have, so the answer is known: the window's own past. A correct decode
    reproduces it closely; a wrong translation layout or normalization shows up
    here as a large error, on a run that costs minutes.
    """
    from ..data.preprocess import iter_windows

    runner = EdgeAVRunner(cfg=cfg, **runner_kw).load()
    errs, keys = [], []
    for clip_id in clip_ids:
        for w_idx, window in iter_windows(cfg, clip_id):
            hist = np.asarray(window.data["ego_history_xyz"][0, 0], dtype=np.float32)
            steps = hist.shape[0] - 1
            act = runner.run(window, mode="inverse_dynamics",
                             action_length=steps, prompt=DRIVE_PROMPT)
            rec = av_actions_to_xyz(act).cpu().numpy()          # (steps, 3)
            # Known past, re-expressed as offsets from its first pose.
            truth = hist[1:] - hist[:1]
            n = min(rec.shape[0], truth.shape[0])
            errs.append(float(np.linalg.norm(rec[:n, :2] - truth[:n, :2], axis=-1).mean()))
            keys.append(f"{clip_id}/{w_idx:02d}")
    return {"per_window_ade": errs, "keys": keys,
            "mean_ade": float(np.mean(errs)) if errs else float("nan")}


# ------------------------------------------------------------------ io ------

def save(blob: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=blob["pred"], gt=blob["gt"],
                        keys=np.array(blob["keys"]),
                        meta=np.str_(json.dumps(
                            {k: v for k, v in blob.items()
                             if k not in ("pred", "gt", "keys")})))
    log.info("wrote %s (%d windows)", path, len(blob["keys"]))
    return path


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {"pred": z["pred"], "gt": z["gt"],
                "keys": [str(k) for k in z["keys"]],
                **json.loads(str(z["meta"]))}


def score(blob: dict, k: int | None = None) -> dict:
    """ADE on the first mode, plus minADE over however many modes there are."""
    pred = torch.from_numpy(blob["pred"]).float()      # (N, K, H, 3)
    gt = torch.from_numpy(blob["gt"]).float()          # (N, H, 3)
    h = min(pred.shape[2], gt.shape[1])
    pred, gt = pred[:, :, :h], gt[:, :h]
    if k is not None:
        pred = pred[:, :k]
    a = ade(pred[:, 0], gt)
    m = min_ade(pred, gt)
    return {"model": blob.get("model", "?"), "n": int(a.numel()),
            "k": int(pred.shape[1]),
            "ade": float(a.mean()), "ade_p90": float(a.quantile(0.9)),
            "minade": float(m.mean()), "minade_p90": float(m.quantile(0.9))}


def report(teacher: dict, edge: dict) -> str:
    """The headroom, stated plainly."""
    t, e = score(teacher), score(edge)
    if teacher["keys"] != edge["keys"]:
        common = sorted(set(teacher["keys"]) & set(edge["keys"]))
        raise RuntimeError(
            f"the two runs cover different windows ({len(teacher['keys'])} vs "
            f"{len(edge['keys'])}, {len(common)} shared) — the gap is only "
            "meaningful on identical clips; re-run both on one split")
    gap = e["ade"] - t["ade"]
    rel = gap / e["ade"] * 100 if e["ade"] else float("nan")
    lines = [
        "=== headroom: teacher vs zero-shot student ===",
        f"windows: {t['n']}",
        f"  Alpamayo 1.5      ADE {t['ade']:6.3f} m  (p90 {t['ade_p90']:6.3f})  "
        f"minADE_{t['k']} {t['minade']:6.3f}",
        f"  Cosmos 3 Edge 0-shot ADE {e['ade']:6.3f} m  (p90 {e['ade_p90']:6.3f})  "
        f"minADE_{e['k']} {e['minade']:6.3f}",
        f"  GAP {gap:+.3f} m  ({rel:+.1f}% of the zero-shot error)",
        "",
        "Everything the distillation can achieve lives inside that gap. Narrow",
        "means the intervention needs to change (rank, targets, data), not the",
        "hyperparameters. NOTE minADE is not comparable across different k.",
    ]
    return "\n".join(lines)

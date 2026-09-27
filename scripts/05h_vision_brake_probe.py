"""Do the FROZEN vision features encode that the driver is about to brake hard?

The question phase 2 turns on. D-040 says the trajectory is `f(images)`, and
D-046 says the AR tower reads the frames almost not at all (zeroing them costs
+0.008 nats) - the model extrapolates ego history instead, which is what open-loop
ADE rewards on a mostly-constant-speed dataset. Closed-loop DiffGRPO would punish
that shortcut, but it can only teach the flow head to READ something the frozen
vision pathway actually delivers: `param_groups_stage2` trains the gen tower's
attention over a FROZEN AR tower, with SigLIP2 frozen and the projector never
trained.

So: fit a linear probe on the projected vision tokens the LM is handed
(`prepare_multimodal_reasoner_inputs` -> `inputs_embeds[visual_pos_masks]`) and
ask whether "the driver brakes hard / stops in the next 6.4 s" is linearly
decodable from them.

The comparison that matters is not vision vs chance - it is vision vs EGO
HISTORY, because ego history is what the model already uses. A window where the
car is already slowing is trivially predictable from its own speed trace; only
the lift of (ego + vision) over (ego alone) is evidence that the frames carry the
lead vehicle's motion.

  reads out, and adds over ego     -> phase 2's closed-loop reward can find it;
                                      the blindness is a shortcut, as D-046 read it
  does not add over ego            -> no amount of flow-head LoRA recovers it, and
                                      the projector / encoder has to come unfrozen

Grouped by CLIP (windows of one clip overlap; an ungrouped split leaks).

    python scripts/05h_vision_brake_probe.py --n 1200 --split val
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from sklearn.decomposition import PCA                               # noqa: E402
from sklearn.linear_model import LogisticRegression                 # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from sklearn.model_selection import GroupKFold                      # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402
from distill.eval import gt_reward                                  # noqa: E402

log = logging.getLogger("brake_probe")


def ego_features(ego_xyz: np.ndarray) -> np.ndarray:
    """What the model already has: the ego speed trace, its trend, curvature."""
    g = np.asarray(ego_xyz, dtype=float).reshape(-1, 3)[:, :2]
    v = np.linalg.norm(np.diff(g, axis=0), axis=1) * 10.0
    if len(v) < 4:
        v = np.pad(v, (0, 4 - len(v)), mode="edge")
    d = np.diff(v)
    head = np.degrees(np.arctan2(*np.diff(g, axis=0)[:, ::-1].T))
    return np.concatenate([v, d, [v.mean(), v[-3:].mean() - v[:3].mean(), v.min(), v.max(),
                                  float(np.ptp(head)) if len(head) else 0.0]])


@torch.no_grad()
def vision_features(student, batch) -> np.ndarray:
    """Mean-pooled projected vision tokens, kept per FRAME so motion survives.

    Pooling every token into one vector would average the 4 timesteps together
    and destroy exactly the signal under test. Instead: pool within each frame,
    then keep the frame means and their first differences."""
    from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl import (
        reasoner_multimodal_utils as mm,
    )
    student.lm._ensure_vision_tower()
    embeds, vis_mask, _, _, _ = mm.prepare_multimodal_reasoner_inputs(
        student.lm, input_ids=batch["input_ids"], pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"], attention_mask=batch.get("attention_mask"))
    tok = embeds[0][vis_mask[0].bool()].float()          # (n_vis_tokens, D)
    grid = batch["image_grid_thw"]                       # (n_images, 3) = t,h,w
    merge = (getattr(getattr(student.lm, "visual", None), "spatial_merge_size", None)
             or getattr(getattr(student.lm, "config", None), "spatial_merge_size", None) or 1)
    per_img = [int(t * h * w) // (int(merge) ** 2) for t, h, w in grid.tolist()]
    if sum(per_img) != tok.shape[0]:                     # fall back to an even split
        per_img = [tok.shape[0] // len(per_img)] * len(per_img)
    out, i = [], 0
    for n in per_img:
        out.append(tok[i:i + n].mean(0))
        i += n
    m = torch.stack(out)                                 # (n_images, D) one row per frame
    diff = m[1:] - m[:-1] if m.shape[0] > 1 else torch.zeros_like(m)
    return torch.cat([m.mean(0), m[-1] - m[0], diff.abs().mean(0)]).cpu().numpy()


def fit_report(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str,
               n_pca: int | None) -> dict:
    """Grouped CV; PCA and scaling are fit INSIDE each fold (fitting them on all
    the data would leak the test fold's directions into the probe)."""
    oof = np.zeros(len(y), dtype=float)
    n_splits = max(2, min(5, len(np.unique(groups))))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if n_pca:
            k = min(n_pca, Xtr.shape[0] - 1, Xtr.shape[1])
            p = PCA(n_components=k, random_state=0).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
        clf.fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    # Bootstrap the AUC over CLIPS, the unit the folds respect.
    rng = np.random.default_rng(0)
    uq = np.unique(groups)
    boots = []
    for _ in range(2000):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if 0 < y[idx].sum() < len(idx):
            boots.append(roc_auc_score(y[idx], oof[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {name:<34} AUC {auc:.3f} [{lo:.3f},{hi:.3f}]   AP {ap:.3f} "
          f"(base {y.mean():.3f})   dim {X.shape[1]}")
    return {"name": name, "auc": float(auc), "ci": [float(lo), float(hi)],
            "ap": float(ap), "oof": oof}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="/data/vla/alpamayo2edge/runs/stage1/run-336/best")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--pca", type=int, default=96)
    ap.add_argument("--out", default="runs/vision_brake_probe.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)

    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    checkpoint.load_into(student, Path(a.ckpt))
    student.eval()

    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=load_split(cfg, a.split), cot_generation=True)
    n = min(a.n, len(ds))
    pad_id = student.tokenizer.pad_token_id
    log.info("split %s: %d windows, probing %d", a.split, len(ds), n)

    V, E, Y, G = [], [], [], []
    for i in range(n):
        item = ds[i]
        k = gt_reward.kinematics(item["gt_future_xyz"])
        # The failing row of the CoC eval: a real deceleration from speed.
        Y.append(int(k["braked_from_speed"] and (k["vmin"] < 0.5 or k["dv"] < -2.5)))
        E.append(ego_features(np.asarray(item["hist_xyz"])))
        batch = move_batch(collate_stage1([item], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            V.append(vision_features(student, batch))
        G.append(item["clip_id"])
        if (i + 1) % 100 == 0:
            log.info("  %d/%d", i + 1, n)

    V, E = np.asarray(V, dtype=np.float64), np.asarray(E, dtype=np.float64)
    Y, G = np.asarray(Y), np.asarray(G)
    log.info("\n%d windows, %d clips, %d positive (%.1f%%)\n",
             len(Y), len(np.unique(G)), Y.sum(), 100 * Y.mean())
    print("  target: driver stops or brakes hard (>2.5 m/s lost) within 6.4 s")
    print(f"  {'probe':<34} {'AUC [95% CI over clips]':<30} {'AP':<10}")
    res = [fit_report(E, Y, G, "ego history only (the shortcut)", None),
           fit_report(V, Y, G, "frozen vision only", a.pca),
           fit_report(np.hstack([E, V]), Y, G, "ego + vision", a.pca)]
    lift = res[2]["auc"] - res[0]["auc"]
    print(f"\n  LIFT of vision over ego history: {lift:+.3f} AUC")
    print("  " + ("vision carries braking evidence the ego trace does not -> a frozen-feature\n"
                  "  flow head CAN learn to brake; D-046's blindness is a shortcut, not a bottleneck"
                  if lift > 0.02 else
                  "vision adds nothing over the ego trace -> closed-loop RL on a frozen\n"
                  "  vision pathway cannot learn to brake; the encoder/projector must unfreeze"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"split": a.split, "n": int(len(Y)), "n_clips": int(len(np.unique(G))),
         "base_rate": float(Y.mean()), "lift_auc": float(lift),
         "probes": [{k: v for k, v in r.items() if k != "oof"} for r in res]}, indent=2))
    log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

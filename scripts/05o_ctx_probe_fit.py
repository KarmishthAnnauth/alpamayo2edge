"""Phase 2: is "departs / brakes within 3 s" decodable from what the flow head reads?

Fits probes on the dump of `05o_ctx_probe_extract.py` (train routes) and scores
val routes. The comparison that matters is the LIFT over ego history, because
ego history is what the head demonstrably uses (D-046, 05h): a car already
slowing is predictable from its own speed trace.

Arms (ego features enter linearly in every arm):
  ego          16-step history: per-step speeds, accelerations, v0
  +mean_vis    + mean-pooled vision tokens (05h's setup, phase 1: +0.015 AUC)
  +attn_vis    + learned attention pooling (4 queries) over the vision tokens -
               a lead car is a few tokens of 3,520, which mean-pooling dilutes
  +attn_all    + attention pooling over ALL context tokens (vision, prompt,
               history, the teacher-forced CoC)
  +attn_text   + attention pooling over the 215 non-vision tokens only

Val AUC per arm, the lift over `ego` with a clip-grouped paired bootstrap 95% CI,
and the same on the "cause visible at t0" subset from 05n (positives whose cause
05n judged visible, all negatives). Mean over --seeds.

  decodable, lift clearly > 0  -> the cue is in the context; the head does not use it
                                  (training signal problem: DiffGRPO / conditioning)
  lift ~ 0                     -> the cue does not reach the context (perception /
                                  temporal context problem: unfreeze projector, more frames)

    python scripts/05o_ctx_probe_fit.py --layer 18
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/bulk/users/vla/alpamayo2edge/ctx_probe")
ARMS = ("ego", "mean_vis", "attn_vis", "attn_all", "attn_text")


def ego_features(hist: np.ndarray, v0: float) -> np.ndarray:
    xy = np.asarray(hist)[:, :2]
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.1          # (15,)
    acc = np.diff(sp) / 0.1                                         # (14,)
    return np.r_[sp, acc, v0, acc[-10:].mean(), acc[-5:].mean()].astype(np.float32)


def load(split: str, layer: int, task: str, root: Path = ROOT):
    d = Path(root) / split
    meta = json.loads((d / "meta.json").read_text())
    W = [w for w in meta["windows"] if w[task] >= 0]
    dim = meta["dim"]
    vals = np.memmap(d / f"values_L{layer}.f16", dtype=np.float16, mode="r").reshape(-1, dim)
    vis = np.memmap(d / "vis.u8", dtype=np.uint8, mode="r")
    Lmax = max(w["n_tok"] for w in W)
    X = torch.zeros(len(W), Lmax, dim, dtype=torch.float16)
    V = torch.zeros(len(W), Lmax, dtype=torch.uint8)                # 0 pad, 1 text, 2 vision
    for i, w in enumerate(W):
        o, n = w["offset"], w["n_tok"]
        X[i, :n] = torch.from_numpy(np.array(vals[o:o + n]))
        V[i, :n] = torch.from_numpy(np.asarray(vis[o:o + n]).astype(np.uint8) + 1)
    E = torch.from_numpy(np.stack([ego_features(w["hist_xyz"], w["v0"]) for w in W]))
    y = torch.tensor([w[task] for w in W], dtype=torch.float32)
    return W, X, V, E, y


class Probe(nn.Module):
    def __init__(self, arm: str, dim: int, n_ego: int, q: int = 4):
        super().__init__()
        self.arm = arm
        self.q = nn.Linear(dim, q, bias=False) if arm.startswith("attn") else None
        n_ctx = 0 if arm == "ego" else (dim if arm == "mean_vis" else q * dim)
        self.norm = nn.LayerNorm(n_ctx) if n_ctx else None
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(n_ego + n_ctx, 1)

    def forward(self, X, V, E):
        if self.arm == "ego":
            return self.out(E).squeeze(-1)
        keep = {"mean_vis": V == 2, "attn_vis": V == 2, "attn_all": V > 0, "attn_text": V == 1}[self.arm]
        X = X.float()
        if self.arm == "mean_vis":
            m = keep.float().unsqueeze(-1)
            z = (X * m).sum(1) / m.sum(1).clamp_min(1)
        else:
            s = self.q(X).masked_fill(~keep.unsqueeze(-1), float("-inf"))   # (B, L, q)
            a = s.softmax(dim=1)
            z = torch.einsum("blq,bld->bqd", a, X).flatten(1)
        return self.out(torch.cat([E, self.drop(self.norm(z))], dim=-1)).squeeze(-1)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit(arm, Xtr, Vtr, Etr, ytr, clips_tr, Xva, Vva, Eva, seed, epochs=25, dev="cuda"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    uc = np.array(sorted(set(clips_tr)))
    hold = set(rng.choice(uc, size=max(1, len(uc) // 7), replace=False))   # early-stopping clips
    ih = np.array([c in hold for c in clips_tr])
    it, iv = np.where(~ih)[0], np.where(ih)[0]
    mu, sd = Etr[it].mean(0), Etr[it].std(0) + 1e-3
    norm = lambda E: (E - mu) / sd
    model = Probe(arm, Xtr.shape[-1], Etr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = (1 - ytr[it].mean()) / ytr[it].mean()

    def predict(X, V, E, idx=None):
        model.eval()
        idx = np.arange(len(E)) if idx is None else idx
        out = []
        with torch.no_grad():
            for j in range(0, len(idx), 128):
                b = torch.from_numpy(idx[j:j + 128])
                out.append(model(X[b].to(dev), V[b].to(dev), norm(E[b]).to(dev)).cpu())
        return torch.cat(out).numpy()

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(it)
        for j in range(0, len(perm), 64):
            b = torch.from_numpy(perm[j:j + 64])
            logit = model(Xtr[b].to(dev), Vtr[b].to(dev), norm(Etr[b]).to(dev))
            loss = F.binary_cross_entropy_with_logits(logit, ytr[b].to(dev), pos_weight=pw.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        a = auc(ytr[iv].numpy(), predict(Xtr, Vtr, Etr, iv))
        if best_state is None or a > best:
            best, best_state = a, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(Xva, Vva, Eva)


def paired_boot(y, s_a, s_b, clips, n=1000, seed=0):
    """95% CI of AUC(s_b) - AUC(s_a), resampling val clips."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, c in enumerate(clips):
        by[c].append(i)
    keys = list(by)
    d = []
    for _ in range(n):
        idx = np.concatenate([by[k] for k in rng.choice(keys, size=len(keys))])
        d.append(auc(y[idx], s_b[idx]) - auc(y[idx], s_a[idx]))
    d = np.array([x for x in d if np.isfinite(x)])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--tasks", default="depart3,brake3")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cue", default="runs/coc_probe/cue_visibility_val.json")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--train-split", default="train", help="smoke: val")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cue = {(r["clip"], r["window"]): r for r in json.loads(Path(a.cue).read_text())["rows"]}
    res = {"layer": a.layer}
    for task in a.tasks.split(","):
        Wtr, Xtr, Vtr, Etr, ytr = load(a.train_split, a.layer, task, a.root)
        Wva, Xva, Vva, Eva, yva = load("val", a.layer, task, a.root)
        ctr, cva = [w["clip"] for w in Wtr], [w["clip"] for w in Wva]
        Xtr, Vtr, Xva, Vva = Xtr.cuda(), Vtr.cuda(), Xva.cuda(), Vva.cuda()   # ~4 GB fp16: one copy, not per batch
        yv = yva.numpy()
        # "cause visible" subset: negatives + positives whose 05n cause is visible at t0
        vis_ok = np.array([yv[i] == 0 or bool(cue.get((w["clip"], w["window"]), {}).get("vis"))
                           for i, w in enumerate(Wva)])
        print(f"\n== {task}  layer {a.layer}: train {len(Wtr)} ({int(ytr.sum())} pos), "
              f"val {len(Wva)} ({int(yv.sum())} pos; cause-visible subset {int(yv[vis_ok].sum())} pos)", flush=True)
        scores = {}
        for arm in a.arms.split(","):
            s = np.mean([fit(arm, Xtr, Vtr, Etr, ytr, ctr, Xva, Vva, Eva, seed, epochs=a.epochs) for seed in range(a.seeds)], axis=0)
            scores[arm] = s
        res[task] = {}
        print(f"  {'arm':10s} {'AUC':>6s} {'lift vs ego [95% CI]':>26s} {'AUC vis':>8s} {'lift vis [95% CI]':>26s}")
        for arm, s in scores.items():
            row = {"auc": auc(yv, s), "auc_vis": auc(yv[vis_ok], s[vis_ok])}
            if arm != "ego":
                row["lift"] = row["auc"] - auc(yv, scores["ego"])
                row["lift_ci"] = paired_boot(yv, scores["ego"], s, cva)
                row["lift_vis"] = row["auc_vis"] - auc(yv[vis_ok], scores["ego"][vis_ok])
                row["lift_vis_ci"] = paired_boot(yv[vis_ok], scores["ego"][vis_ok], s[vis_ok],
                                                 [c for c, k in zip(cva, vis_ok) if k])
                l = f"{row['lift']:+.3f} [{row['lift_ci'][0]:+.3f}, {row['lift_ci'][1]:+.3f}]"
                lv = f"{row['lift_vis']:+.3f} [{row['lift_vis_ci'][0]:+.3f}, {row['lift_vis_ci'][1]:+.3f}]"
            else:
                l = lv = "-"
            print(f"  {arm:10s} {row['auc']:6.3f} {l:>26s} {row['auc_vis']:8.3f} {lv:>26s}", flush=True)
            res[task][arm] = row
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Datasets over cached teacher shards. Both stages read the same npz files."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_shard(path: Path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    return d


def discover_shards(cache_root: Path, clip_ids: list[str] | None = None) -> list[Path]:
    if clip_ids is None:
        with open(cache_root / "manifest.json") as f:
            clip_ids = json.load(f)["clips"]
    shards = []
    for cid in clip_ids:
        shards += sorted((cache_root / cid).glob("*.npz"))
    return shards


class DistillShardDataset(Dataset):
    """Returns raw shard dicts; stage-specific collators shape the batch."""

    def __init__(self, cfg, clip_ids: list[str] | None = None):
        self.cfg = cfg
        self.shards = discover_shards(Path(cfg.paths.cache_root), clip_ids)
        if not self.shards:
            raise RuntimeError("No cached shards found - run scripts/02_label.py first")

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i: int) -> dict:
        d = _load_shard(self.shards[i])
        d["clip_id"] = self.shards[i].parent.name
        return d


def collate_stage1(batch: list[dict], pad_id: int) -> dict:
    """Pads token streams; stacks feature targets per teacher layer."""
    def pad(key, dtype=torch.long):
        seqs = [torch.as_tensor(b[key], dtype=dtype) for b in batch]
        L = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), L, *seqs[0].shape[1:]), pad_id, dtype=dtype)
        mask = torch.zeros(len(seqs), L, dtype=torch.bool)
        for j, s in enumerate(seqs):
            out[j, : s.shape[0]] = s
            mask[j, : s.shape[0]] = True
        return out, mask

    traj, traj_mask = pad("traj_token_ids")
    coc, coc_mask = pad("coc_token_ids")
    topk_idx, _ = pad("traj_topk_idx")
    topk_logp, _ = pad("traj_topk_logp", dtype=torch.float32)
    layers = [int(x) for x in batch[0]["feat_layers"]]
    feats = {l: torch.stack([torch.as_tensor(b[f"feat_{l}"], dtype=torch.float32) for b in batch])
             for l in layers}
    return dict(
        traj=traj, traj_mask=traj_mask, coc=coc, coc_mask=coc_mask,
        topk_idx=topk_idx, topk_logp=topk_logp, feats=feats,
        meta_action=torch.as_tensor([int(b["meta_action"]) for b in batch]),
        gt_traj=torch.stack([torch.as_tensor(b["gt_traj"], dtype=torch.float32) for b in batch]),
        clip_ids=[b["clip_id"] for b in batch],
    )


def collate_stage2(batch: list[dict], pad_id: int) -> dict:
    """Flattens the K cached flow targets per window into the batch dim."""
    base = collate_stage1(batch, pad_id)
    t, a_t, v = [], [], []
    owner = []
    for j, b in enumerate(batch):
        k = b["flow_t"].shape[0]
        t.append(torch.as_tensor(b["flow_t"], dtype=torch.float32))
        a_t.append(torch.as_tensor(b["flow_a_t"], dtype=torch.float32))
        v.append(torch.as_tensor(b["flow_v"], dtype=torch.float32))
        owner += [j] * k
    base.update(flow_t=torch.cat(t), flow_a_t=torch.cat(a_t),
                flow_v=torch.cat(v), flow_owner=torch.as_tensor(owner))
    return base

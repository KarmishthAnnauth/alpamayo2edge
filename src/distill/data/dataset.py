"""Datasets over cached teacher shards. Both stages read the same npz files.

The shards hold teacher TARGETS only (`teacher/labeler.py::save_shard`) — no
images and no ego motion, because the teacher consumed those and the cache
exists so the teacher never has to be resident again. The STUDENT still needs
them: its context is the teacher's input (D-028). `Stage1Dataset` therefore
pairs each shard with a re-loaded window, addressing it by
`(clip_id, window_idx)` — the shard filename — through the deterministic anchor
formula in `preprocess.window_t0_us`.
"""
from __future__ import annotations
import json
from pathlib import Path

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from . import frames

log = logging.getLogger(__name__)


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


class Stage1Dataset(Dataset):
    """Cached teacher targets + the student's own view of the same window.

    `context_builder` comes from `EdgeStudent.context_builder()` and is
    deliberately model-free so workers can hold it. Window loading streams from
    the PhysicalAI-AV interface, so this is I/O-bound — give it workers.
    """

    def __init__(self, cfg, context_builder, clip_ids: list[str] | None = None):
        from .preprocess import load_window, window_t0_us

        self.cfg = cfg
        self.ctx = context_builder
        self._load_window = load_window
        self._t0 = window_t0_us
        self.cache_root = Path(cfg.paths.cache_root)
        self.shards = discover_shards(self.cache_root, clip_ids)
        if not self.shards:
            raise RuntimeError("No cached shards found - run scripts/02_label.py first")
        self._warned_uncached = False

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i: int) -> dict:
        path = self.shards[i]
        clip_id, w_idx = path.parent.name, int(path.stem)
        d = _load_shard(path)
        d["clip_id"] = clip_id
        window = self._window(clip_id, w_idx)
        # Region-relative future bins are what the cache stores (D-014), and what
        # the student's appended future rows are indexed by.
        traj_bins = [int(b) for b in d["traj_token_ids"]]
        ctx = self.ctx.build(window,
                             coc_text=str(d["coc_text"]),
                             traj_bins=traj_bins)
        d["student"] = ctx
        return d

    def _window(self, clip_id: str, w_idx: int):
        """Cached student input when the labeling pass wrote one; otherwise
        re-stream the clip.

        The fallback exists so a cache made before `data/frames.py` still
        trains, but it re-reads video every epoch — the log line below is worth
        acting on rather than ignoring.
        """
        cached = frames.input_path(self.cache_root, clip_id, w_idx)
        if cached.exists():
            return frames.load_window_input(cached, clip_id)
        if not self._warned_uncached:
            self._warned_uncached = True
            log.warning(
                "no cached student input at %s — falling back to streaming the "
                "clip for every window, every epoch. Re-run scripts/02_label.py "
                "to backfill (it writes inputs even where targets already exist).",
                cached)
        return self._load_window(self.cfg, clip_id, self._t0(self.cfg, w_idx))


def collate_student(items: list[dict], pad_id: int) -> dict:
    """Right-pad the student contexts and stack the image tensors.

    Right padding specifically: `reasoner_forward` takes no attention mask and is
    causal, so trailing padding cannot reach a real token — left padding would
    silently corrupt every position (D-027).
    """
    ctxs = [it["student"] for it in items]
    lens = [c["input_ids"].shape[0] for c in ctxs]
    L = max(lens)
    B = len(ctxs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.bool)
    coc_mask = torch.zeros(B, L, dtype=torch.bool)
    traj_mask = torch.zeros(B, L, dtype=torch.bool)
    for j, c in enumerate(ctxs):
        n = c["input_ids"].shape[0]
        input_ids[j, :n] = c["input_ids"]
        attention_mask[j, :n] = True
        for span, mask in ((c["coc_span"], coc_mask), (c["traj_span"], traj_mask)):
            if span is not None:
                mask[j, span[0]:span[1]] = True
    out = dict(input_ids=input_ids, attention_mask=attention_mask,
               coc_pos=coc_mask, traj_pos=traj_mask,
               n_prompt=torch.tensor([c["n_prompt"] for c in ctxs]))
    if ctxs[0]["pixel_values"] is not None:
        out["pixel_values"] = torch.cat([c["pixel_values"] for c in ctxs], dim=0)
        out["image_grid_thw"] = torch.cat([c["image_grid_thw"] for c in ctxs], dim=0)
    return out


def move_batch(batch, device="cuda", non_blocking: bool = True):
    """Recursively move a collated batch to `device`.

    Recursive on purpose. `collate_stage1` returns `feats` as a dict of per-layer
    tensors, and the flat `torch.is_tensor(v)` guard the call sites used to carry
    skipped it silently: the teacher's features stayed on the CPU while the
    student's went to the GPU, and `losses.feature_match` - which coerces dtype
    but deliberately not device - raised on the first step of stage 1.

    Non-tensor leaves (`clip_ids`) pass through untouched.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_batch(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch(v, device, non_blocking) for v in batch)
    return batch


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
    # Same positions and the same 128-token geometry as `traj`, so `traj_mask`
    # covers both. Region-relative, like every other bin id in the cache.
    gt_traj_tok, _ = pad("gt_traj_token_ids")
    coc, coc_mask = pad("coc_token_ids")
    topk_idx, _ = pad("traj_topk_idx")
    topk_logp, _ = pad("traj_topk_logp", dtype=torch.float32)
    layers = [int(x) for x in batch[0]["feat_layers"]]
    feats = {l: torch.stack([torch.as_tensor(b[f"feat_{l}"], dtype=torch.float32) for b in batch])
             for l in layers}
    out = dict(
        traj=traj, traj_mask=traj_mask, coc=coc, coc_mask=coc_mask,
        topk_idx=topk_idx, topk_logp=topk_logp, feats=feats,
        gt_traj_tok=gt_traj_tok,
        gt_traj=torch.stack([torch.as_tensor(b["gt_traj"], dtype=torch.float32) for b in batch]),
        clip_ids=[b["clip_id"] for b in batch],
    )
    # Present once the dataset is Stage1Dataset; absent for target-only use
    # (the layer-map CKA probe, offline inspection).
    if "student" in batch[0]:
        out.update(collate_student(batch, pad_id))
    return out


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

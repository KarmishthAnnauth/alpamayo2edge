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
from ..student.context import PREFIX_MASK

log = logging.getLogger(__name__)


def _load_shard(path: Path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    return d


#: `frames.save_window_input` writes the student's inputs as
#: `{window_idx:02d}_input.npz` INTO THE SAME clip directory as the teacher
#: targets `{window_idx:02d}.npz`. A bare `*.npz` glob therefore returns both,
#: which doubles the dataset length and hands `Stage1Dataset` an input file as
#: if it were a shard — `int("00_input")` raises before the first forward pass.
#: `scripts/check_cache.py` already partitions on this suffix; discovery has to
#: agree with it.
INPUT_SUFFIX = "_input.npz"


def discover_shards(cache_root: Path, clip_ids: list[str] | None = None) -> list[Path]:
    if clip_ids is None:
        with open(cache_root / "manifest.json") as f:
            clip_ids = json.load(f)["clips"]
    shards = []
    for cid in clip_ids:
        shards += sorted(p for p in (cache_root / cid).glob("*.npz")
                         if not p.name.endswith(INPUT_SUFFIX))
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

    def __init__(self, cfg, context_builder, clip_ids: list[str] | None = None,
                 for_generation: bool = False, cot_generation: bool = False,
                 image_dropout: float = 0.0, filter_contradictions: bool = False,
                 prefix_noise_bins: float = 0.0, prefix_mask_prob: float = 0.0):
        from .preprocess import load_window, window_t0_us

        self.cfg = cfg
        self.ctx = context_builder
        # CONTEXT FORMAT (D-043) - read from `stage1` so every consumer of this
        # dataset (trainer, gate, val NLL, smoke, inspection scripts, stage 2)
        # builds the sequence the model was trained on:
        #   traj_prefix  "gt" puts the GT trajectory bins in the teacher-forced
        #                sequence (they are also the `gt_ce` target, so the
        #                student learns p(GT bin t | GT bins < t)); "teacher"
        #                is runs 1-6, the teacher's own sampled tokens.
        #   route_hint   fill the `<|route_start|>` slot with the direction the
        #                driver took (turn left / turn right / straight, from
        #                `gt_future_xyz`). The teacher was labelled route-blind
        #                (D-038); at deployment this comes from navigation.
        st = getattr(cfg, "stage1", None)
        self.traj_prefix = str(st.get("traj_prefix", "teacher") if st is not None else "teacher")
        if self.traj_prefix not in ("teacher", "gt"):
            raise ValueError(f"stage1.traj_prefix must be 'teacher' or 'gt', got {self.traj_prefix!r}")
        self.route_hint = bool(st.get("route_hint", False)) if st is not None else False
        # TRAINING-ONLY knobs, explicit so the gate and val paths cannot inherit
        # them by accident:
        #   image_dropout  probability that a window is served with its frames
        #                  zeroed. The collator then clears its CoC/struct loss
        #                  masks (a CoC must not be learned from a blank image)
        #                  and keeps the trajectory mask, so on those samples the
        #                  trajectory has to be read from ego history + route +
        #                  the CoC text. That is what builds the CoC ->
        #                  trajectory dependence stage 1 never had (D-040).
        #   filter_contradictions  drop windows whose teacher CoC flatly
        #                  contradicts the driver's future (data/grounding.py).
        #   prefix_noise_bins  jitter the teacher-forced TRAJECTORY PREFIX (the
        #                  bins in `input_ids`, never the loss target, which the
        #                  trainer reads from the cache). Run 7 as launched put the
        #                  clean GT stream in the prefix and the teacher-forced
        #                  loss sat 0.2 nats above its floor by step 120 while the
        #                  free decode was a random walk (10 m ADE, heading sd
        #                  53 deg, run-332 epoch 0): a 10 Hz curvature/accel trace
        #                  is smooth, so "next = previous" satisfies the loss and
        #                  nothing anchors the plan on the scene. Per window a
        #                  sigma is drawn from U(0, prefix_noise_bins) and every
        #                  prefix bin gets an independent N(0, sigma) offset,
        #                  rounded and clipped to the region, so the model sees
        #                  clean, slightly wrong and badly wrong prefixes and must
        #                  predict the CLEAN next bin from scene + CoC + history.
        #                  Exposure-bias mitigation in the spirit of scheduled
        #                  sampling, without a decode per batch.
        if not 0.0 <= float(image_dropout) <= 1.0:
            raise ValueError(f"image_dropout must be in [0, 1], got {image_dropout}")
        self.image_dropout = float(image_dropout)
        self.filter_contradictions = bool(filter_contradictions)
        #   prefix_mask_prob  probability that a window's WHOLE trajectory prefix
        #                  is replaced by the mask id (`context.PREFIX_MASK`), so
        #                  every one of its 128 bins must be predicted from the
        #                  scene, the CoC, the route and the ego history alone.
        #                  D-045: at sigma-16 jitter the epoch-0 model still took
        #                  0.000 nats from the frames and 0.002 from the CoC - the
        #                  prefix was the only source it read. Jitter makes the
        #                  prefix unreliable; masking makes it absent.
        if float(prefix_noise_bins) < 0.0:
            raise ValueError(f"prefix_noise_bins must be >= 0, got {prefix_noise_bins}")
        self.prefix_noise_bins = float(prefix_noise_bins)
        if not 0.0 <= float(prefix_mask_prob) <= 1.0:
            raise ValueError(f"prefix_mask_prob must be in [0, 1], got {prefix_mask_prob}")
        self.prefix_mask_prob = float(prefix_mask_prob)
        # Eval mode: the CoC is still teacher-forced but the 128 trajectory
        # positions are NOT — the context stops at `<|traj_future_start|>` so the
        # student decodes them itself. Building the teacher-forced context and
        # then generating from it (which is what the gate did) puts the teacher's
        # own answer in the prompt and measures nothing.
        self.for_generation = bool(for_generation)
        # Inspection mode: nothing is teacher-forced. The assistant turn opens
        # `<|cot_start|>` and stops, so the student free-runs the CoC itself
        # (`EdgeStudent.generate_coc_text`). `coc_span`/`traj_span` are then None.
        self.cot_generation = bool(cot_generation)
        self._load_window = load_window
        self._t0 = window_t0_us
        self.cache_root = Path(cfg.paths.cache_root)
        self.shards = discover_shards(self.cache_root, clip_ids)
        if not self.shards:
            raise RuntimeError("No cached shards found - run scripts/02_label.py first")
        self.grounding_stats = None
        if self.filter_contradictions:
            from . import grounding
            self.shards, self.grounding_stats = grounding.grounded_shards(self.shards)
            grounding.log_stats(self.grounding_stats)
            if not self.shards:
                raise RuntimeError("grounding filter dropped every window")
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
        # the student's appended future rows are indexed by. Which stream is the
        # teacher-forced prefix is `stage1.traj_prefix` (see __init__).
        key = "gt_traj_token_ids" if self.traj_prefix == "gt" else "traj_token_ids"
        traj_bins = [int(b) for b in d[key]]
        if not self.for_generation:
            if self.prefix_mask_prob > 0.0 and float(torch.rand(())) < self.prefix_mask_prob:
                traj_bins = [PREFIX_MASK] * len(traj_bins)
            elif self.prefix_noise_bins > 0.0:
                traj_bins = jitter_prefix(traj_bins, self.prefix_noise_bins)
        from . import grounding
        nav = grounding.route_hint(d["gt_future_xyz"]) if self.route_hint else None
        ctx = self.ctx.build(window,
                             coc_text=None if self.cot_generation else str(d["coc_text"]),
                             traj_bins=None if self.for_generation else traj_bins,
                             nav_text=nav,
                             for_generation=self.for_generation)
        d["student"] = ctx
        d["route_hint"] = nav or ""
        # Drawn here, in the worker: each DataLoader worker's torch RNG is seeded
        # per epoch, so the draw is independent across samples and epochs.
        d["img_drop"] = bool(self.image_dropout > 0.0
                             and float(torch.rand(())) < self.image_dropout)
        # The ego history is the frame `detokenize_traj` integrates waypoints
        # from, so the gate needs it alongside the tokens. `[:, -1]` drops the
        # n_traj axis exactly as the teacher's own detokenization does.
        d["hist_xyz"] = window.data["ego_history_xyz"][:, -1][0]
        d["hist_rot"] = window.data["ego_history_rot"][:, -1][0]
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


#: Region-relative future bins live in [0, N_FUTURE_BINS) (D-014: 3000 bins).
N_FUTURE_BINS = 3000


def jitter_prefix(bins: list[int], max_sigma: float, n_bins: int = N_FUTURE_BINS) -> list[int]:
    """Noise a teacher-forced trajectory prefix (see Stage1Dataset.__init__).

    sigma ~ U(0, max_sigma) per call, offsets ~ N(0, sigma) per bin, rounded,
    clipped to [0, n_bins). Uses torch's RNG so DataLoader workers draw
    independently. The LOSS TARGET is untouched: the trainer scores against the
    cache's `gt_traj_token_ids`, not against what sits in `input_ids`.
    """
    sigma = float(torch.rand(())) * float(max_sigma)
    if sigma <= 0.0:
        return list(bins)
    off = torch.round(torch.randn(len(bins)) * sigma).to(torch.long)
    out = (torch.as_tensor(bins, dtype=torch.long) + off).clamp_(0, n_bins - 1)
    return [int(b) for b in out]


def collate_student(items: list[dict], pad_id: int) -> dict:
    """Right-pad the student contexts and stack the image tensors.

    Right padding specifically: `reasoner_forward` takes no attention mask and is
    causal, so trailing padding cannot reach a real token — left padding would
    silently corrupt every position (D-027).
    """
    ctxs = [it["student"] for it in items]
    # Image dropout (D-043). A dropped sample keeps its sequence and its
    # trajectory targets but loses its frames (zeroed pixel values - a constant
    # image, not an absent one, so positions and the mrope layout are unchanged)
    # and its CoC/struct loss masks: the CoC is teacher-forced INPUT on that
    # sample, never a target, because a CoC learned from a blank frame is a
    # hallucination lesson. The trajectory then has to come from ego history,
    # the route hint and the CoC text.
    drop = [bool(it.get("img_drop", False)) for it in items]
    lens = [c["input_ids"].shape[0] for c in ctxs]
    L = max(lens)
    B = len(ctxs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.bool)
    coc_mask = torch.zeros(B, L, dtype=torch.bool)
    struct_mask = torch.zeros(B, L, dtype=torch.bool)
    traj_mask = torch.zeros(B, L, dtype=torch.bool)
    for j, c in enumerate(ctxs):
        n = c["input_ids"].shape[0]
        input_ids[j, :n] = c["input_ids"]
        attention_mask[j, :n] = True
        spans = [(c["traj_span"], traj_mask)]
        if not drop[j]:
            spans += [(c["coc_span"], coc_mask), (c.get("struct_span"), struct_mask)]
        for span, mask in spans:
            if span is not None:
                mask[j, span[0]:span[1]] = True
    out = dict(input_ids=input_ids, attention_mask=attention_mask,
               coc_pos=coc_mask, struct_pos=struct_mask, traj_pos=traj_mask,
               n_prompt=torch.tensor([c["n_prompt"] for c in ctxs]),
               img_drop=torch.tensor(drop, dtype=torch.bool))
    if ctxs[0]["pixel_values"] is not None:
        out["pixel_values"] = torch.cat(
            [torch.zeros_like(c["pixel_values"]) if drop[j] else c["pixel_values"]
             for j, c in enumerate(ctxs)], dim=0)
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
    # ACTION space (accel, curvature) is what `gt_traj` holds; minADE is a
    # POSITIONAL metric, and `gt_future_xyz` is its reference. `open_loop.evaluate`
    # scored against `gt_traj` until 2026-08-24, which is not a distance at all.
    out["gt_future_xyz"] = torch.stack(
        [torch.as_tensor(b["gt_future_xyz"], dtype=torch.float32) for b in batch])
    for key in ("hist_xyz", "hist_rot"):
        if key in batch[0]:
            out[key] = torch.stack(
                [torch.as_tensor(b[key], dtype=torch.float32) for b in batch])
    # Present once the dataset is Stage1Dataset; absent for target-only use
    # (the layer-map CKA probe, offline inspection).
    if "student" in batch[0]:
        out.update(collate_student(batch, pad_id))
    return out


def collate_stage2(batch: list[dict], pad_id: int) -> dict:
    """Flattens the K cached flow targets per window into the batch dim.

    Windows without cached `flow_*` tuples (Bench2Drive, `supervision: gt_flow`)
    pass through with the stage-1 keys only; `train_stage2` then draws its own
    (t, a0) per window."""
    base = collate_stage1(batch, pad_id)
    if "flow_t" not in batch[0]:
        return base
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

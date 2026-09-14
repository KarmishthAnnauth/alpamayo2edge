"""Student checkpoint I/O — the stage-1 -> stage-2 handoff.

Exists because of D-024. With LoRA in the loop a checkpoint has to be written
MERGED: `LoraInjectedLinear` adds `<path>.lora_A.weight` / `<path>.lora_B.weight`
keys, stage 2 reloads into a student with no adapters injected yet, and nothing
in cosmos-framework aliases those keys. An unmerged stage-1 save therefore fails
at the start of stage 2, after the expensive part is already done.

Two more things the old `save_pretrained` / `from_pretrained` pair got wrong and
that only surface on the GPU box:

- The saved tables are LARGER than a fresh student's: `extend_trajectory_vocab`
  appends 3000 rows to embed_tokens and lm_head. Stage 2 must re-extend the
  vocab BEFORE loading, or every embedding key is a shape mismatch. `load_into`
  checks this and says so in one line instead of dumping a shape error per key.
- `student.model.from_pretrained(dir)` (called on the INSTANCE in the old
  stage-2 script) builds a second model from `dir`'s config rather than filling
  the one already on the GPU — briefly doubling resident weights on a box where
  that is the scarce resource. We load into the live module instead.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

WEIGHTS_NAME = "student.safetensors"
META_NAME = "distill_meta.json"


def save(student, path: str | Path, **meta) -> Path:
    """Write LoRA-merged weights + the geometry needed to reload them.

    Non-destructive: the live model keeps its adapters and training continues,
    which is what makes this safe to call from the per-epoch best-checkpoint
    branch.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sd = student.merged_state_dict()

    # safetensors refuses aliased storage; merged tensors are fresh CPU copies
    # but the untouched ones came straight off `state_dict()`.
    seen: dict[int, str] = {}
    out = {}
    for k, v in sd.items():
        v = v.detach().cpu().contiguous()
        ptr = v.data_ptr()
        out[k] = v.clone() if ptr in seen else v
        seen[ptr] = k
    save_file(out, str(path / WEIGHTS_NAME))

    blob = {
        "new_token_range": list(getattr(student, "new_token_range", ())),
        "action_domain_id": student.action_domain_id,
        "lora": student.lora_stats,
        **meta,
    }
    (path / META_NAME).write_text(json.dumps(blob, indent=2))
    cfg = getattr(student.model, "config", None)
    if cfg is not None and hasattr(cfg, "to_json_file"):
        try:
            cfg.to_json_file(str(path / "config.json"))
        except Exception as e:  # config is a convenience here, not the contract
            log.warning("could not serialize model config: %s", e)
    log.info("saved merged student -> %s (%d tensors)", path, len(out))
    return path


def load_into(student, path: str | Path) -> dict:
    """Load a merged checkpoint into the live student. Returns its metadata."""
    from safetensors.torch import load_file

    path = Path(path)
    meta = json.loads((path / META_NAME).read_text()) if (path / META_NAME).exists() else {}

    saved_range = tuple(meta.get("new_token_range") or ())
    live_range = tuple(getattr(student, "new_token_range", ()) or ())
    if saved_range and live_range != saved_range:
        raise RuntimeError(
            f"vocab geometry mismatch: checkpoint was saved with "
            f"new_token_range={saved_range}, live student has {live_range or 'none'}. "
            "Call extend_trajectory_vocab() with the SAME traj_tokenizer_spec.pt "
            "before loading.")

    sd = load_file(str(path / WEIGHTS_NAME))
    lora_keys = [k for k in sd if ".lora_A." in k or ".lora_B." in k]
    if lora_keys:
        raise RuntimeError(
            f"checkpoint carries {len(lora_keys)} unmerged LoRA keys "
            f"(e.g. {lora_keys[0]}) — it was not written through checkpoint.save()")

    missing, unexpected = student.model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    if missing:
        # Buffers (rotary caches, timestep frequencies) are registered
        # non-persistent and legitimately absent; real weights are not.
        weighty = [k for k in missing if k.endswith((".weight", ".bias"))]
        if weighty:
            raise RuntimeError(f"{len(weighty)} weights missing, e.g. {weighty[:3]}")
        log.info("%d non-persistent buffers not in checkpoint (expected)", len(missing))
    log.info("loaded merged student <- %s", path)
    return meta


# ---------------------------------------------------------------- adapters ----
ADAPTERS_NAME = "adapters.pt"


def save_adapters(student, path: str | Path, **meta) -> Path:
    """Adapter-only checkpoint: the `lora_*` tensors of the decoder stack plus
    metadata. ~75 MB for r48 attention-only, against ~8.6 GB for a merged save -
    which is what makes a checkpoint at EVERY eval affordable (D-038: the
    step-75 RL weights were lost because only `best/` was kept, selected on a
    noisy 99-window check). Reload = the run's `init_ckpt` merged weights +
    `load_adapters`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    root = student._decoder_layers()
    sd = {n: p.detach().to("cpu", copy=True) for n, p in root.named_parameters() if "lora_" in n}
    torch.save({"adapters": sd, "meta": {"lora": dict(student.lora_stats), **meta}},
               path / ADAPTERS_NAME)
    return path


def load_adapters(student, path: str | Path) -> dict:
    """Load an adapter-only checkpoint into a student whose adapters are already
    injected (`param_groups_stage1()`) on top of the same `init_ckpt`."""
    blob = torch.load(Path(path) / ADAPTERS_NAME, map_location="cpu", weights_only=False)
    root = student._decoder_layers()
    missing, unexpected = root.load_state_dict(blob["adapters"], strict=False)
    unexpected = [k for k in unexpected]
    if unexpected:
        raise RuntimeError(f"adapter keys not in the model: {unexpected[:3]}...")
    n_lora = sum(1 for n, _ in root.named_parameters() if "lora_" in n)
    if n_lora != len(blob["adapters"]):
        raise RuntimeError(f"model has {n_lora} adapter tensors, file has {len(blob['adapters'])}")
    return blob["meta"]

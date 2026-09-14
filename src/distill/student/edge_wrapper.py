"""Cosmos 3 Edge wrapper: parameter-group separation, trajectory vocab, action head.

Integrated against the local `../cosmos-framework` clone (read 2026-08-17);
see DECISIONS.md D-015..D-017 for the verified facts this relies on:

- Edge is a unified MoT (`Nemotron3DenseVLMoT*`, cosmos_framework/model/generator/
  mot/unified_mot.py). The tower split is expressed PER PARAMETER NAME: every
  generation-tower (diffusion) weight carries the `_moe_gen` suffix
  (q/k/v/o_proj_moe_gen, mlp_moe_gen, *_layernorm_moe_gen, norm_moe_gen); the
  reasoner/AR tower is every language-model weight WITHOUT that suffix. (D-015)
- Action interface: per-frame action tokens enter/leave the MoT through
  `DomainAwareLinear` projections `action2llm` (action_dim->hidden) and
  `llm2action` (hidden->action_dim) with per-embodiment weight rows
  (num_embodiment_domains=32, max_action_dim=64). We register the teacher's
  UnicycleAccelCurvature space as a NEW embodiment domain: 64 waypoint tokens,
  raw dim 2 zero-padded to 64. (D-017, resolves D-009)
- Student rectified flow is SIGN-FLIPPED vs the teacher: x_t = t*noise + (1-t)*data,
  target = noise - data, t = noise weight (`RectifiedFlow.get_interpolation`).
  Mapping from teacher convention: sigma = 1 - t_teacher, v_student* = -u_teacher.
  `flow_forward` does this conversion INTERNALLY and returns velocities in the
  TEACHER convention, so the trainers/losses stay in one convention. (D-016)

Loading: `cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp`
on a local snapshot of nvidia/Cosmos3-Edge (both towers + action head; the HF
`Cosmos3EdgeForConditionalGeneration` in transformers@main is reasoner-only).
Requires the cosmos-framework environment (uv sync per its README).

Blocks marked # VALIDATE-ON-GPU are written against the real APIs but unrun.
"""
from __future__ import annotations
import contextlib
import logging
import re
from pathlib import Path
import torch
import torch.nn as nn

from . import lora, prompt as prompt_mod

# Tower split by parameter name (D-015): gen tower = `_moe_gen` suffix plus the
# VFM network's generation-side modules; AR tower = language-model params
# without the suffix. Embeddings/lm_head are shared-input, handled separately.
DIFF_TOWER_PAT = re.compile(
    r"(_moe_gen|action2llm|llm2action|action_modality_embed|time_embedder)")
AR_TOWER_PAT = re.compile(r"language_model")  # applied AFTER excluding DIFF matches

log = logging.getLogger(__name__)


def _patch_hf_storage_info_for_torch28() -> None:
    """Let cosmos-framework's 3-arg `_HFStorageInfo(...)` work on torch 2.8.

    cosmos-framework targets torch 2.10, where `_HFStorageInfo` carries only
    (relative_path, shape, dtype). torch 2.8's dataclass still has the older
    `offset`/`length` fields and they are REQUIRED, so its `read_metadata`
    raises TypeError before a single weight is read.

    Supplying zeros is safe here, not a guess: on the LOAD path neither reader
    ever touches those two fields. torch 2.8's
    `HuggingFaceStorageReader.read_data` uses only `relative_path`, `dtype` and
    `shape`, taking tensor bytes by FQN out of a full `deserialize()`; and
    cosmos's own `_MmapSafeReadMixin._process_read_request` overrides that with
    `safe_open(...).get_slice(fqn)`. Both slice with `req.storage_offsets` /
    `req.lengths`, which come from the read REQUEST, not from this record.
    (`.length` is read only in torch's WRITE path, which we never take.)

    Applied to cosmos's module namespace rather than torch's, so the blast
    radius is the one construction site that needs it. A no-op once the env
    moves to a torch whose signature already matches.
    """
    import inspect
    from cosmos_framework.inference import model as _cf_model

    real = _cf_model._HFStorageInfo
    params = inspect.signature(real.__init__).parameters
    if "offset" not in params:
        return                                   # torch >= 2.10: nothing to do
    if getattr(real, "_a2e_shimmed", False):
        return

    def _compat(relative_path, shape, dtype, offset=0, length=0, **kw):
        return real(relative_path=relative_path, offset=offset, length=length,
                    shape=shape, dtype=dtype, **kw)

    _compat._a2e_shimmed = True
    _cf_model._HFStorageInfo = _compat
    log.info("patched _HFStorageInfo for torch %s (offset/length unused on load)",
             torch.__version__)


def _point_tokenizer_at_dir(node, ckpt_dir: str) -> bool:
    """Rewrite a `build_processor_lazy` tokenizer node to load from a local dir.

    Mirrors `inference.Inference._point_tokenizer_node_at_dir`, which the
    framework's own entrypoint applies and which `from_pretrained_dcp` does NOT:
    left alone, the node's `repository: nvidia/Cosmos3-Edge` + `revision: main`
    sends `checkpoint_db._hf_download` off to re-fetch the whole repo through a
    nested `uv run --isolated ... hf download`. That defeats the local pin, needs
    the network on every model construction, and is where the first smoke run
    hung. `build_processor_lazy`'s two modes are mutually exclusive, so the
    repository trio has to come OUT as `tokenizer_type` goes in.
    """
    if not isinstance(node, dict):
        return False
    target = str(node.get("_target_", ""))
    if not target.endswith("build_processor_lazy"):
        return False
    if not (node.get("repository") or node.get("tokenizer_type")):
        return False
    for k in ("repository", "revision", "subdir"):
        node.pop(k, None)
    node["tokenizer_type"] = ckpt_dir
    return True


def _patch_tokenizer_nodes(obj, ckpt_dir: str) -> int:
    """Walk the model spec and repoint every processor node at `ckpt_dir`."""
    n = 0
    if isinstance(obj, dict):
        n += _point_tokenizer_at_dir(obj, ckpt_dir)
        for v in obj.values():
            n += _patch_tokenizer_nodes(v, ckpt_dir)
    elif isinstance(obj, list):
        for v in obj:
            n += _patch_tokenizer_nodes(v, ckpt_dir)
    return n


def _omni_config(ckpt_dir: str):
    """The framework's own Cosmos3-Edge architecture spec, as a Cosmos3OmniConfig.

    `deserialize_config_dict` runs `undo_config_replacements`, which rewrites the
    yaml's `cosmos3._src.vfm.*` _target_ paths onto the installed
    `cosmos_framework.*` modules — so the `cosmos3` package itself is never
    imported and does not need to exist.
    """
    import cosmos_framework
    from cosmos_framework.inference.model import Cosmos3OmniConfig
    from cosmos_framework.inference.common.config import deserialize_config_dict
    from cosmos_framework.inference.common.public_model_config import (
        load_model_config_from_hf_config)

    yaml_path = (Path(cosmos_framework.__file__).parent / "inference" / "configs"
                 / "model" / "Cosmos3-Edge.yaml")
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"{yaml_path} is missing — the Cosmos3-Edge architecture spec ships "
            "with cosmos-framework and is what the HF repo's config.json does not "
            "carry. Check the ../cosmos-framework checkout.")
    model_dict = load_model_config_from_hf_config(deserialize_config_dict(yaml_path))
    if not _patch_tokenizer_nodes(model_dict, ckpt_dir):
        log.warning("no build_processor_lazy tokenizer node found in %s — if model "
                    "construction stalls, it is re-downloading the processor from "
                    "the hub", yaml_path)
    return Cosmos3OmniConfig(model=model_dict)


class EdgeStudent(nn.Module):
    def __init__(self, cfg, device: str = "cuda"):
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

        self.cfg = cfg
        # A local checkpoint directory OR an HF repo id. `paths.student_repo` is
        # now the former, mirroring `paths.teacher_repo`: `snapshot_download`
        # resolves `main` at call time, so a push upstream would silently swap the
        # base weights out from under a retention baseline that is only meaningful
        # against the exact checkpoint it was measured on (D-025). A local dir also
        # drops the dependency on HF_HUB_CACHE being exported at load time.
        repo = str(cfg.paths.student_repo)
        ckpt = repo if Path(repo).is_dir() else snapshot_download(repo)
        self._ckpt_dir = ckpt
        # `from_pretrained_dcp` defaults `config` to
        # `Cosmos3OmniConfig.from_pretrained(ckpt)`, which reads the HF repo's
        # config.json — a plain transformers config (model_type cosmos3_edge,
        # text_config/vision_config). That carries no `model:` section, so
        # Cosmos3OmniConfig falls back to `{}` and the Hydra spec the model is
        # instantiated from is empty; construction then dies deep inside
        # `__init__` on `model_dict.config.ema` (the framework's own
        # `load_model_config_dict` documents exactly this "Missing key ema"
        # failure). The architecture has to come from the framework's shipped
        # model config instead; the checkpoint dir supplies only the WEIGHTS,
        # which load fine from the diffusers-style layout via CheckpointType.HF.
        _patch_hf_storage_info_for_torch28()
        self.model = Cosmos3OmniModel.from_pretrained_dcp(
            Path(ckpt), config=_omni_config(ckpt)).to(device)
        # OmniMoTModel -> Cosmos3VFMNetwork -> unified-MoT language model
        self.omni = self.model.model
        self.net = self.omni.net
        self.lm = self.net.language_model
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        # Teacher action space as a new embodiment domain (D-017): reserve a
        # row of the DomainAwareLinear embeddings. VALIDATE-ON-GPU: confirm the
        # chosen id is unused by the shipped checkpoint's domain registry.
        self.action_domain_id = int(cfg.student.get("action_domain_id", 31))
        self.raw_action_dim = 2          # (accel, curvature), D-002
        self.n_action_tokens = 64        # waypoints @ 10 Hz
        self.max_action_dim = self.net.config.action_dim  # 64 on Edge
        self._device = torch.device(device)
        self._mask_handles: list = []   # gradient row-masks, cleared per stage
        self.lora_stats: dict = {}

    def enable_gradient_checkpointing(self) -> None:
        """Turn on activation checkpointing for the AR tower.

        NOT on `self.model`: the Cosmos3OmniModel wrapper reports
        `supports_gradient_checkpointing = False` and transformers raises
        outright. The tower that claims support is the language model underneath
        (`self.lm`), which is also the only part stage 1 backprops through.

        "Claims", because the support flag is inherited and unimplemented: no
        module defines the `gradient_checkpointing` attribute, so this raises
        unless LoRA has already been injected (the adapters happen to carry it).
        And even when it succeeds it does nothing - `_impl_reasoner_forward` calls
        `decoder_layer.reasoner_forward` directly, never through a checkpoint
        function. So this warns rather than raising: a config flag must not be
        able to kill a multi-day unattended run over a feature that is inert.
        """
        try:
            self.lm.gradient_checkpointing_enable()
        except ValueError as e:
            log.warning("gradient checkpointing unavailable on %s (%s); "
                        "continuing without it - see stage1.grad_checkpoint in "
                        "the config for why it is inert here",
                        type(self.lm).__name__, e)
            return
        log.warning("gradient_checkpointing_enable() accepted, but "
                    "reasoner_forward does not route through a checkpoint "
                    "function - expect NO memory saving")

    # ---------------- geometry ----------------
    # `self.model` is the Cosmos3OmniModel wrapper and its config is a
    # Cosmos3OmniConfig, which carries the Hydra `model:` spec and NOT the
    # reasoner's dimensions. The AR tower's own config is the one with them
    # (28 layers x 2048, corroborated by len(lm.model.layers)); reading them off
    # `model.config` raised AttributeError in the stage-1 layer-map setup.

    @property
    def n_layers(self) -> int:
        """Decoder layers in the AR tower — the CKA map's student axis."""
        return int(self.lm.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        """AR-tower hidden width — the FeatureProjections input dim."""
        return int(self.lm.config.hidden_size)

    # ---------------- vocab ----------------

    def extend_trajectory_vocab(self, traj_tok_spec: dict) -> None:
        """Append the teacher's trajectory vocabulary and structural tokens.

        traj_tok_spec comes from TeacherWrapper.probe_trajectory_tokenizer().
        Layout of the appended block, mirroring the teacher's own (D-029):

            [old_n,               old_n + 3000)   FUTURE bins  (region-relative)
            [old_n + 3000,        old_n + 4000)   HISTORY bins (ego motion)
            [old_n + 4000,        old_n + 4009)   the 9 structural specials

        The history region is here because ego motion is DISCRETE: A1.5 fills the
        48 `<|traj_history|>` slots with DeltaTrajectoryTokenizer bins rather than
        with a continuous projection, so the student needs those rows too.

        BOTH tables are resized by hand. `Nemotron3DenseVLTextForCausalLM`
        (unified_mot.py:2510) sets `_tied_weights_keys = []` and defines no
        `get_output_embeddings`, so HF's `resize_token_embeddings` resizes the
        INPUT embedding and silently leaves `lm_head` at the old vocab — the
        appended ids would have no logits at all, and every stage-1 trajectory
        loss would be taken over rows that cannot be produced.

        New rows init: mean of the existing rows + N(0, 0.02).
        """
        emb = self.lm.get_input_embeddings()
        old_n, dim = emb.weight.shape
        n_future = int(traj_tok_spec["vocab_size"])                    # 3000
        n_hist = int(traj_tok_spec.get("hist_vocab_size", 0))          # 1000
        n_bins = int(traj_tok_spec.get("total_bins", n_future + n_hist))
        if n_bins != n_future + n_hist:
            raise ValueError(
                f"trajectory regions do not tile the vocabulary: {n_future} future "
                f"+ {n_hist} history != {n_bins} total")
        specials = list(prompt_mod.SPECIAL_TOKENS)
        n_new = n_bins + len(specials)
        new_n = old_n + n_new
        ekw = {"device": emb.weight.device, "dtype": emb.weight.dtype}

        new_emb = nn.Embedding(new_n, dim, **ekw)
        with torch.no_grad():
            new_emb.weight[:old_n] = emb.weight
            new_emb.weight[old_n:] = (emb.weight.mean(dim=0, keepdim=True)
                                      + 0.02 * torch.randn(n_new, dim, **ekw))
        self.lm.set_input_embeddings(new_emb)

        head = self.lm.lm_head
        hkw = {"device": head.weight.device, "dtype": head.weight.dtype}
        new_head = nn.Linear(dim, new_n, bias=head.bias is not None, **hkw)
        with torch.no_grad():
            new_head.weight[:old_n] = head.weight
            new_head.weight[old_n:] = (head.weight.mean(dim=0, keepdim=True)
                                       + 0.02 * torch.randn(n_new, dim, **hkw))
            if head.bias is not None:
                new_head.bias[:old_n] = head.bias
                new_head.bias[old_n:] = head.bias.mean()
        self.lm.lm_head = new_head

        # Keep the advertised vocab in sync, or `generate` builds its logit
        # processors over the old range and the new ids are unreachable.
        self.lm.vocab_size = new_n
        seen = set()
        for c in (getattr(self.lm, "config", None),
                  getattr(getattr(self.lm, "config", None), "text_config", None),
                  getattr(getattr(self.lm, "model", None), "config", None)):
            if c is None or id(c) in seen:
                continue
            seen.add(id(c))
            if getattr(c, "vocab_size", None) is not None:
                c.vocab_size = new_n

        self.new_token_range = (old_n, new_n)
        self.future_base = old_n
        self.n_future_bins = n_future
        self.hist_base = old_n + n_future
        self.n_hist_bins = n_hist
        self.special_ids = {t: old_n + n_bins + i for i, t in enumerate(specials)}
        # PRIVATE on purpose: `decode` expects `encode`'s (accel, curvature) dim
        # order, and everything in this project — cache, targets, and whatever the
        # student learns to emit — is in the teacher's EMITTED order (D-031). Route
        # through `detokenize_traj` below, never through this attribute.
        self._traj_decode = traj_tok_spec["detokenizer_fn"]
        self.hist_tokenize = traj_tok_spec.get("hist_tokenize_fn")

    def detokenize_traj(self, tokens: torch.Tensor, hist_xyz: torch.Tensor,
                        hist_rot: torch.Tensor) -> torch.Tensor:
        """Region-relative bins in EMISSION order -> (B, H, 3) ego-frame waypoints.

        The student's mirror of `TeacherWrapper.detokenize_traj`, and the student's
        only detokenization entry point. The student is trained on the teacher's
        emitted tokens, so it emits in the teacher's order too — which means it
        needs the same `swap_action_dims` before `decode`, whose convention is
        `encode`'s (D-031). Skipping the swap costs ~20-128 m of ADE and looks
        exactly like a model that failed to learn.

        `hist_xyz` (B, T, 3) / `hist_rot` (B, T, 3, 3) are the window's ego
        history — the reference frame `decode` integrates from.
        """
        from ..teacher.wrapper import swap_action_dims

        toks = swap_action_dims(tokens.reshape(tokens.shape[0], -1).long().cpu())
        # `decode` returns a 3-tuple; the third slot is a timestamp A1.5 never fills.
        fut_xyz, _, _ = self._traj_decode(hist_xyz.float().cpu(),
                                          hist_rot.float().cpu(), toks)
        return fut_xyz

    # ---------------- context assembly (D-028/D-029) ----------------

    def future_bin_id(self, b: int) -> int:
        return self.future_base + int(b)

    def hist_bin_id(self, b: int) -> int:
        return self.hist_base + int(b)

    def special_token_id(self, token: str) -> int:
        return self.special_ids[token]

    def context_builder(self, image_processor=None, cameras=None, n_frames=None):
        """A picklable assembler for this student's context format (D-028).

        Built here because the vocabulary offsets and special-token ids are the
        student's, but handed out as a standalone object because DataLoader
        workers need it and this module owns 9GB of CUDA weights.
        """
        from .context import ContextBuilder, load_image_processor

        if not hasattr(self, "new_token_range"):
            raise RuntimeError("call extend_trajectory_vocab() first")
        if image_processor is None:
            image_processor = load_image_processor(self._ckpt_dir)
        self.lm._ensure_vision_tower()   # plumbs config.image_token_id
        # The VISION TOWER is the authority, not `lm.config` — which does not
        # carry `spatial_merge_size` at all, so the old `getattr(..., 1)` here
        # silently resolved merge=1 and made ContextBuilder reserve 880
        # placeholders per frame where the tower emits 220 (verified on the box:
        # 4 images of 880 raw patches -> image_features (880, 2048)). That 4x
        # mismatch is a masked_scatter size error at best. No default: a wrong
        # merge is worse than a missing one.
        merge = getattr(getattr(self.lm, "visual", None), "spatial_merge_size", None)
        if merge is None:
            merge = getattr(getattr(self.lm, "config", None), "spatial_merge_size", None)
        if merge is None:
            raise RuntimeError(
                "cannot resolve spatial_merge_size from the vision tower or the "
                "LM config; ContextBuilder's placeholder count would be guesswork")
        return ContextBuilder(
            tokenizer=self.tokenizer,
            image_processor=image_processor,
            cameras=list(cameras or self.cfg.data.raw["cameras"]),
            n_frames=int(n_frames or self.cfg.data.context_frames),
            future_base=self.future_base,
            hist_base=self.hist_base,
            special_ids=dict(self.special_ids),
            image_token_id=self._image_token_id,
            hist_tokenize=self.hist_tokenize,
            merge_size=int(merge or 1),
        )

    @property
    def _image_token_id(self) -> int:
        tid = getattr(self.lm.config, "image_token_id", None)
        if tid is None:
            raise RuntimeError(
                "config.image_token_id is unset — it is plumbed by "
                "_ensure_vision_tower(), so touch the vision tower first")
        return int(tid)

    # ---------------- parameter groups ----------------

    def _is_diff(self, name: str) -> bool:
        return bool(DIFF_TOWER_PAT.search(name))

    def _is_ar(self, name: str) -> bool:
        return AR_TOWER_PAT.search(name) is not None and not self._is_diff(name)

    # D-024: both stages adapt through LoRA. What stays full-rank is what has
    # no pretrained weights to preserve — the appended trajectory rows and our
    # embodiment row of the action head — and those are gradient-masked so the
    # pretrained rows sharing the same tensor stay put (TRAINING_STRATEGY §2).

    def _decoder_layers(self) -> nn.Module:
        """The decoder-layer stack: BOTH towers' weights and nothing else.

        Injection scope for both stages. The causal LM is the wrong root: its
        SigLIP2 vision tower (lazily attached at `language_model.visual` by
        `_ensure_vision_tower`) names its attention leaves `q_proj`/`k_proj`/
        `v_proj` too, and the framework's plain-leaf matching would adapt it.
        Within this subtree the `_moe_gen` suffix separates the towers (D-015).
        """
        return self.lm.model.layers

    def _reset_trainable(self) -> None:
        # Materialize the vision tower FIRST. `_ensure_vision_tower` is lazy, so
        # a freeze that ran before it left the 489M SigLIP2 encoder at the torch
        # default requires_grad=True. It is in no optimizer group either way
        # (param_groups_stage1 returns LoRA params + the appended vocab rows), so
        # nothing was training — but autograd still retained activations through
        # all 27 vision layers to build gradients that were then discarded.
        # Measured on the box: 12+ GiB of retained activations vs 0.38 GiB with
        # the tower frozen; it is what made a 7.66 GiB model OOM a 48 GiB card.
        ensure = getattr(self.lm, "_ensure_vision_tower", None)
        if ensure is not None:
            ensure()
        for h in self._mask_handles:
            h.remove()
        self._mask_handles.clear()
        for p in self.model.parameters():
            p.requires_grad_(False)
        visual = getattr(self.lm, "visual", None)
        if visual is not None:                 # not reached by self.model.parameters()
            for p in visual.parameters():      # on every path — belt and braces
                p.requires_grad_(False)

    def _enable_new_vocab_rows(self) -> list[nn.Parameter]:
        """Appended trajectory rows of embed_tokens + lm_head, masked so the
        pretrained rows below `new_token_range[0]` get no gradient."""
        if not hasattr(self, "new_token_range"):
            raise RuntimeError(
                "call extend_trajectory_vocab() before param_groups_stage1()")
        old_n, _ = self.new_token_range
        out = []
        for p in (self.lm.get_input_embeddings().weight, self.lm.lm_head.weight):
            p.requires_grad_(True)
            self._mask_handles.append(lora.mask_rows_below_(p, old_n))
            out.append(p)
        return out

    def _enable_action_domain_rows(self) -> list[nn.Parameter]:
        """Our embodiment row of action2llm/llm2action; the other 31 stay put.

        DomainAwareLinear keeps per-domain parameters in nn.Embedding tables
        (`fc`: [num_domains, out*in], `bias`: [num_domains, out]), so "our row
        only" is a dim-0 row mask on both (domain_aware_linear.py:41).
        """
        out = []
        for proj in (self.net.action2llm, self.net.llm2action):
            for p in (proj.fc.weight, proj.bias.weight):
                p.requires_grad_(True)
                self._mask_handles.append(
                    lora.mask_rows_except_(p, self.action_domain_id))
                out.append(p)
        return out

    def _enable_shared_action_embeds(self) -> list[nn.Parameter]:
        """`action_modality_embed` + `time_embedder`: shared by ALL 32
        embodiments, so training them is a forgetting channel no row mask can
        cover. Off by default; the domain rows already give the teacher's
        action space its own affine interface (D-017)."""
        ps = [self.net.action_modality_embed] + list(self.net.time_embedder.parameters())
        for p in ps:
            p.requires_grad_(True)
        return ps

    def param_groups_stage1(self):
        """AR tower via LoRA + the appended trajectory rows; gen tower untouched.

        Returns AdamW-ready groups carrying an `lr_mult` the trainer applies to
        `stage1.lr`.
        """
        lcfg = self.cfg.student.lora.stage1
        self._reset_trainable()
        if lcfg.get("enabled", True):
            self.lora_stats = self._inject(lcfg, lora.AR_ATTN_TARGETS)
            adapted = lora.lora_parameters(self._decoder_layers())
        else:
            # Full-FT ablation (TRAINING_STRATEGY §2, "the one real cost").
            # ~2.48B trainable = 43.5GB of states before activations: this does
            # NOT fit the 48GB box — it is the 500-clip diagnostic, run elsewhere.
            adapted = [p for n, p in self.lm.model.named_parameters()
                       if not self._is_diff(n) and "embed" not in n]
            for p in adapted:
                p.requires_grad_(True)
            self.lora_stats = {"wrapped": 0, "full_ft": True}
        boosted = self._enable_new_vocab_rows()
        return [{"params": adapted, "lr_mult": 1.0},
                {"params": boosted, "lr_mult": self.cfg.student.new_token_lr_mult}]

    def param_groups_stage2(self):
        """Gen tower via LoRA + our action-head row; AR tower frozen (D-015).

        Stage 1's adapters are already folded into the base weights by the time
        this runs (see distill/checkpoint.py), so there is no stage-1 LoRA left
        to keep training here.
        """
        lcfg = self.cfg.student.lora.stage2
        self._reset_trainable()
        if lcfg.get("enabled", True):
            self.lora_stats = self._inject(lcfg, lora.GEN_ATTN_TARGETS)
            adapted = lora.lora_parameters(self._decoder_layers())
        else:
            adapted = [p for n, p in self.lm.model.named_parameters() if self._is_diff(n)]
            for p in adapted:
                p.requires_grad_(True)
            self.lora_stats = {"wrapped": 0, "full_ft": True}
        new_iface = self._enable_action_domain_rows()
        if self.cfg.student.lora.get("train_shared_action_embeds", False):
            new_iface += self._enable_shared_action_embeds()
        return [{"params": adapted, "lr_mult": 1.0},
                {"params": new_iface, "lr_mult": self.cfg.student.new_token_lr_mult}]

    def _inject(self, lcfg, default_targets: list[str]) -> dict:
        targets = list(lcfg.get("targets", default_targets))
        lora.inject(self._decoder_layers(), targets=targets,
                    rank=lcfg.rank, alpha=lcfg.alpha, device=self._device)
        return lora.describe(self._decoder_layers())

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Everything with a gradient — what the trainer clips."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def merged_state_dict(self) -> dict:
        """LoRA-folded weights for checkpointing; does not disturb training."""
        return lora.merged_state_dict(self.model)

    # ---------------- forward APIs used by the trainers ----------------

    def ar_forward(self, batch, capture_layers=None) -> dict:
        """Teacher-forced reasoner pass over the deployment context (D-027/D-028).

        Runs the REASONER-TOWER api, not the HF causal-LM interface: the causal
        LM's own `forward` takes a `SequencePack` (unified_mot.py:2563) and would
        reject `input_ids=` outright. The AR text path is
        `model.reasoner_forward(...) -> [B, T, hidden]` (final post-norm) plus a
        separate `lm_head`.

        Batching note: `reasoner_forward` takes NO attention mask — the tower is
        causal by construction. Right-padding is therefore safe (real tokens
        never attend to padding that follows them) and left-padding is NOT. The
        collator right-pads; the losses mask.

        Returns {"logits": (B,L,V), "hidden": {student_layer: (B,L,D)},
                 "final_hidden": (B,L,D)}.
        """
        fwd, _ = self._reasoner_inputs(batch)
        with self._capture_layers(capture_layers) as hidden:
            h = self.lm.model.reasoner_forward(cache=None, **fwd)
        return {"logits": self.lm.lm_head(h), "hidden": hidden, "final_hidden": h}

    def _reasoner_inputs(self, batch) -> tuple[dict, torch.Tensor | None]:
        """Prefill kwargs for `reasoner_forward`, plus the mrope deltas.

        Text-only prompts pass `input_ids` straight through. With cameras, the
        images have to be encoded and scattered into `inputs_embeds` first — that
        is what `prepare_multimodal_reasoner_inputs` does, and it also returns the
        mrope `position_ids` the prefill needs and the per-sample deltas the
        decode loop needs.
        """
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            return dict(input_ids=batch["input_ids"]), None
        from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl import (
            reasoner_multimodal_utils as mm,
        )
        self.lm._ensure_vision_tower()
        embeds, vis_mask, deepstack, pos_ids, deltas = mm.prepare_multimodal_reasoner_inputs(
            self.lm,
            input_ids=batch["input_ids"],
            pixel_values=pixel_values,
            image_grid_thw=batch["image_grid_thw"],
            attention_mask=batch.get("attention_mask"),
        )
        return (dict(input_ids=None, inputs_embeds=embeds, position_ids=pos_ids,
                     visual_pos_masks=vis_mask, deepstack_visual_embeds=deepstack),
                deltas)

    @contextlib.contextmanager
    def _capture_layers(self, layers=None):
        """Collect per-layer reasoner outputs for feature KD (D-008).

        `_impl_reasoner_forward` calls `decoder_layer.reasoner_forward(...)`
        DIRECTLY rather than through `__call__`, so `register_forward_hook` never
        fires (D-027). We shadow the bound method on the instance for the
        duration and restore the instance `__dict__` exactly as we found it.

        The captured tensors are activations autograd already retains, so this
        stores references, not copies.
        """
        stack = self._decoder_layers()
        wanted = set(range(len(stack))) if layers is None else {int(i) for i in layers}
        out: dict[int, torch.Tensor] = {}
        patched = []
        for i, layer in enumerate(stack):
            if i not in wanted:
                continue
            original = layer.reasoner_forward
            had_own = "reasoner_forward" in layer.__dict__

            def _wrap(idx, fn):
                def inner(*a, **kw):
                    h = fn(*a, **kw)
                    out[idx] = h
                    return h
                return inner

            layer.reasoner_forward = _wrap(i, original)
            patched.append((layer, original, had_own))
        try:
            yield out
        finally:
            for layer, original, had_own in patched:
                if had_own:
                    layer.reasoner_forward = original
                else:
                    del layer.__dict__["reasoner_forward"]

    def flow_forward(self, batch, a_t, t, context_kv=None) -> torch.Tensor:
        """Gen-tower velocity prediction at cached teacher points, returned in
        the TEACHER convention (D-016 conversion handled here).

        a_t: (N, 64, 2) noised actions, teacher convention (t = data weight).
        t:   (N,) teacher timesteps.
        context_kv: (kv, owner) - frozen AR KV per window + per-sample owner idx.
        """
        n = a_t.shape[0]
        sigma = (1.0 - t).clamp(0.0, 1.0)                 # student noise weight
        # Zero-pad raw 2-dim actions into the 64-wide domain interface (D-017).
        x = a_t.new_zeros(n, self.n_action_tokens, self.max_action_dim)
        x[..., : self.raw_action_dim] = a_t
        domain = torch.full((n,), self.action_domain_id,
                            dtype=torch.long, device=a_t.device)

        # Action tokens -> gen-pathway embeddings: domain projection + timestep
        # embedding + action modality embedding (mirrors Cosmos3VFMNetwork's
        # packing of action tokens). VALIDATE-ON-GPU: timestep scale/shift must
        # match net.config.timestep_scale exactly.
        emb = self.net.action2llm(x, domain)               # (N, 64, hidden)
        ts = self.net.time_embedder(self._embedder_timesteps(sigma))
        emb = emb + ts.unsqueeze(1) + self.net.action_modality_embed

        # Joint two-way attention against the frozen AR context KV: gen-pathway
        # (\*_moe_gen) parameters over the action tokens, und KV from context_kv.
        # VALIDATE-ON-GPU: packing via cosmos_framework pack_input_sequence with
        # action-only gen sequence; reuse und KV across the K flow samples of a
        # window via the owner index.
        h = self._gen_pathway_forward(emb, context_kv)     # (N, 64, hidden)

        v_student = self.net.llm2action(self.net.norm_moe_gen(h)
                                        if hasattr(self.net, "norm_moe_gen") else h,
                                        domain)            # (N, 64, 64)
        v_student = v_student[..., : self.raw_action_dim]  # (N, 64, 2)
        return -v_student  # student target = noise - data; teacher = data - noise

    def _embedder_timesteps(self, sigma: torch.Tensor) -> torch.Tensor:
        """Normalized student noise level -> what `time_embedder` expects.

        Cosmos3VFMNetwork embeds `action.timesteps * net.timestep_scale`
        (cosmos3_vfm_network.py:788), where `action.timesteps` are DISCRETE
        scheduler steps in [0, num_train_timesteps) and
        `timestep_scale = timestep_range / num_train_timesteps`
        (omni_mot_model.py:264). The embedder's own input range is therefore
        [0, timestep_range), and `timestep_range` is 1.0 for Edge
        (edge_model_config.py:63) — so a sigma already in [0, 1] is passed
        THROUGH. The previous code divided by `timestep_scale`, inflating the
        embedder input by 1000x at the shipped 0.001 scale.
        """
        return sigma * self._timestep_range

    @property
    def _timestep_range(self) -> float:
        cfg = getattr(self.omni, "config", None)
        rng = getattr(getattr(cfg, "diffusion_expert_config", None), "timestep_range", None)
        if rng is None:  # derive it back out of the scale the net was built with
            n = getattr(getattr(cfg, "rectified_flow_inference_config", None),
                        "num_train_timesteps", None)
            rng = float(self.net.timestep_scale) * float(n) if n else 1.0
        return float(rng)

    def _gen_pathway_forward(self, gen_embeds, context_kv):
        """Run the MoT gen pathway over action-token embeds with und-context KV.
        VALIDATE-ON-GPU: wire through unified_mot's packed forward
        (set_gen_seq/get_gen_seq + PackedAttentionMoT with cached und K/V)."""
        raise NotImplementedError(
            "Gen-pathway packed forward - wire on the GPU box against "
            "cosmos_framework.model.generator.mot.unified_mot (see D-015).")

    @torch.no_grad()
    def generate_coc_and_traj(self, batch, max_coc_tokens: int = 64,
                              n_future: int | None = None) -> list[dict]:
        """One rollout per batch row: free-running CoC, then the trajectory (D-039).

        The RL recipe NVIDIA ships for Alpamayo rolls out the WHOLE completion -
        chain-of-causation, then trajectory tokens - and rewards the decoded
        trajectory's ADE against the driver's future, so the CoC earns credit
        through the plan it leads to. This is that rollout for the student.

        Why one loop and not `generate_coc_text` followed by `generate_traj_tokens`:
        the G samples of a prompt end their CoC at different lengths, and
        `reasoner_forward` is causal with no attention mask, so a second
        right-padded prefill would decode the short rows from padding
        (eval/coarse_minade.py's batch-size-1 note). Here every row advances one
        token per step through a small state machine, so the cache stays valid:

            0  CoC: sample over the full vocabulary until the `<|cot_end|>`
               subwords appear in the decoded tail (or `max_coc_tokens`: fail)
            1  boundary: the `<|traj_future_start|>` subwords, FORCED - the same
               tokens `struct_ce` teacher-forces, so the layout matches the
               scoring context exactly
            2  trajectory: `n_future` bins, restricted to the future rows
            3  done: emit pad, ignored

        Returns, per row: {"coc_ids", "terminated", "bins" (region-relative
        LongTensor of n_future, or None when the CoC failed)}.
        """
        from cosmos_framework.model.generator.mot.unified_mot import (
            ReasonerKVCache, _sample_next_token,
        )
        model = self.lm.model
        n_fut = int(n_future if n_future is not None else prompt_mod.N_FUTURE_TOKENS)
        boundary = self.tokenizer.encode("<|traj_future_start|>", add_special_tokens=False)
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))
        fwd, deltas = self._reasoner_inputs(batch)
        h = model.reasoner_forward(cache=cache, **fwd)
        base_mrope = (deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)
                      if deltas is not None else None)
        gcfg = self.cfg.teacher
        temperature = float(gcfg.get("gen_temperature", 1.0))
        top_p = float(gcfg.get("gen_top_p", 1.0))
        vocab_lo = self.new_token_range[0]
        stops = ("<|cot_end|>", "</think>")
        pad = int(self.tokenizer.pad_token_id)

        b = h.shape[0]
        phase = [0] * b
        coc: list[list[int]] = [[] for _ in range(b)]
        bins: list[list[int]] = [[] for _ in range(b)]
        forced: list[list[int]] = [[] for _ in range(b)]
        failed = [False] * b
        logits = self.lm.lm_head(h[:, -1, :])
        max_steps = int(max_coc_tokens) + len(boundary) + n_fut
        for step in range(max_steps):
            free = _sample_next_token(logits, do_sample=True, temperature=temperature,
                                      top_k=None, top_p=top_p)
            binned = _sample_next_token(self._restrict_to_future_bins(logits), do_sample=True,
                                        temperature=temperature, top_k=None, top_p=top_p)
            tok = torch.full_like(free, pad)
            for j in range(b):
                if phase[j] == 0:
                    tok[j] = free[j]
                    coc[j].append(int(free[j]))
                    tail = self.tokenizer.decode([t for t in coc[j][-12:] if t < vocab_lo])
                    if any(s in tail for s in stops):
                        phase[j] = 1
                        forced[j] = list(boundary)
                    elif len(coc[j]) >= int(max_coc_tokens):
                        phase[j] = 3
                        failed[j] = True
                elif phase[j] == 1:
                    tok[j] = forced[j].pop(0)
                    if not forced[j]:
                        phase[j] = 2
                elif phase[j] == 2:
                    tok[j] = binned[j]
                    bins[j].append(int(binned[j]) - self.future_base)
                    if len(bins[j]) >= n_fut:
                        phase[j] = 3
            if all(p == 3 for p in phase) or step == max_steps - 1:
                break
            position_ids = None if base_mrope is None else base_mrope + cache.seq_len
            h = model.reasoner_forward(tok.unsqueeze(1), cache=cache, position_ids=position_ids)
            logits = self.lm.lm_head(h[:, -1, :])
        out = []
        for j in range(b):
            ok = (not failed[j]) and len(bins[j]) == n_fut
            out.append({"coc_ids": coc[j], "terminated": not failed[j],
                        "bins": torch.tensor(bins[j], dtype=torch.long) if ok else None})
        return out

    def generate_traj_tokens(self, batch, n_tokens: int | None = None) -> torch.Tensor:
        """Student-sampled trajectory tokens, restricted to the future-bin rows.

        Mirrors the teacher's Phase-B restriction (D-014): only the 3000 appended
        future-bin ids may be emitted. `generate_reasoner_text` cannot express
        that — it has temperature/top-k/top-p but no `suppress_tokens` (D-027) —
        so this is the framework's own decode loop with a per-step logit mask.
        Prefill, cache handling and mrope decode positions follow
        `_impl_generate_reasoner_text` exactly; sampling reuses the framework's
        `_sample_next_token` rather than a second top-p implementation.

        Returns REGION-RELATIVE bins (B, n_tokens), the convention the cache and
        losses use.
        """
        from cosmos_framework.model.generator.mot.unified_mot import (
            ReasonerKVCache, _sample_next_token,
        )

        model = self.lm.model
        n = int(n_tokens if n_tokens is not None else prompt_mod.N_FUTURE_TOKENS)
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))

        fwd, deltas = self._reasoner_inputs(batch)
        h = model.reasoner_forward(cache=cache, **fwd)
        base_mrope = (deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)
                      if deltas is not None else None)

        gcfg = self.cfg.teacher   # same decode settings the teacher labeled with
        temperature = float(gcfg.get("gen_temperature", 1.0))
        top_p = float(gcfg.get("gen_top_p", 1.0))

        emitted = []
        logits = self.lm.lm_head(h[:, -1, :])
        for step in range(n):
            tok = _sample_next_token(
                self._restrict_to_future_bins(logits),
                do_sample=True, temperature=temperature, top_k=None, top_p=top_p)
            emitted.append(tok)
            if step == n - 1:
                break
            position_ids = None if base_mrope is None else base_mrope + cache.seq_len
            h = model.reasoner_forward(tok.unsqueeze(1), cache=cache,
                                       position_ids=position_ids)
            logits = self.lm.lm_head(h[:, -1, :])
        return torch.stack(emitted, dim=1) - self.future_base

    @torch.no_grad()
    def generate_coc_text(self, batch, max_new_tokens: int = 256) -> list[list[int]]:
        """Free-run the chain-of-causation text — no teacher forcing, no restriction.

        The gate (`eval/coarse_minade.py`) teacher-forces the CoC, so it never
        shows what the student would actually reason. This decodes it: the
        context must stop at `<|cot_start|>` (`Stage1Dataset(cot_generation=True)`
        / `ContextBuilder.build(coc_text=None)`), and decoding runs over the FULL
        vocabulary until the CoC terminates or `max_new_tokens`.

        Termination: the student's tokenizer has no `<|cot_end|>` id — the marker
        is written as its six generic subwords (`< | cot _end | >`), which is what
        `struct_ce` supervises (eval_phase1.md §1). So the stop test is a string
        match on the decoded tail, not an id, and it also catches the base model's
        native `</think>` (id 13) in case the student falls back to it.

        Shares the prefill / cache / mrope-decode path with `generate_traj_tokens`
        verbatim. Returns raw id lists (one per row) INCLUDING the terminator
        subwords; the caller splits the decoded text on `<|cot_end|>` / `</think>`.
        """
        from cosmos_framework.model.generator.mot.unified_mot import (
            ReasonerKVCache, _sample_next_token,
        )

        model = self.lm.model
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))
        fwd, deltas = self._reasoner_inputs(batch)
        h = model.reasoner_forward(cache=cache, **fwd)
        base_mrope = (deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)
                      if deltas is not None else None)

        gcfg = self.cfg.teacher
        temperature = float(gcfg.get("gen_temperature", 1.0))
        top_p = float(gcfg.get("gen_top_p", 1.0))
        vocab_lo = self.new_token_range[0]
        stops = ("<|cot_end|>", "</think>")

        b = h.shape[0]
        done = [False] * b
        rows: list[list[int]] = [[] for _ in range(b)]
        logits = self.lm.lm_head(h[:, -1, :])
        for step in range(int(max_new_tokens)):
            tok = _sample_next_token(logits, do_sample=True, temperature=temperature,
                                     top_k=None, top_p=top_p)
            for j in range(b):
                if done[j]:
                    continue
                rows[j].append(int(tok[j]))
                tail = self.tokenizer.decode([t for t in rows[j][-12:] if t < vocab_lo])
                if any(s in tail for s in stops):
                    done[j] = True
            if all(done) or step == int(max_new_tokens) - 1:
                break
            position_ids = None if base_mrope is None else base_mrope + cache.seq_len
            h = model.reasoner_forward(tok.unsqueeze(1), cache=cache,
                                       position_ids=position_ids)
            logits = self.lm.lm_head(h[:, -1, :])
        return rows

    def _restrict_to_future_bins(self, logits: torch.Tensor) -> torch.Tensor:
        """-inf everywhere outside [future_base, future_base + 3000)."""
        lo, hi = self.future_base, self.future_base + self.n_future_bins
        out = torch.full_like(logits, float("-inf"))
        out[..., lo:hi] = logits[..., lo:hi]
        return out

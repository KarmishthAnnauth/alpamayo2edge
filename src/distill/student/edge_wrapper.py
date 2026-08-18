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
import re
import torch
import torch.nn as nn

# Tower split by parameter name (D-015): gen tower = `_moe_gen` suffix plus the
# VFM network's generation-side modules; AR tower = language-model params
# without the suffix. Embeddings/lm_head are shared-input, handled separately.
DIFF_TOWER_PAT = re.compile(
    r"(_moe_gen|action2llm|llm2action|action_modality_embed|time_embedder)")
AR_TOWER_PAT = re.compile(r"language_model")  # applied AFTER excluding DIFF matches


class EdgeStudent(nn.Module):
    def __init__(self, cfg, device: str = "cuda"):
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from cosmos_framework.inference.model import Cosmos3OmniModel

        self.cfg = cfg
        ckpt = snapshot_download(cfg.paths.student_repo)
        self.model = Cosmos3OmniModel.from_pretrained_dcp(ckpt).to(device)
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

    # ---------------- vocab ----------------

    def extend_trajectory_vocab(self, traj_tok_spec: dict) -> None:
        """Append the teacher's trajectory vocabulary verbatim (plan Step 2.1).

        traj_tok_spec comes from TeacherWrapper.probe_trajectory_tokenizer().
        Cache/loss trajectory ids are REGION-RELATIVE bins [0, 3000) (D-014);
        student vocab id = new_token_range[0] + bin.
        New embedding rows init: mean of existing embeddings + N(0, 0.02).
        """
        emb = self.lm.get_input_embeddings()
        old_n, dim = emb.weight.shape
        n_new = traj_tok_spec["vocab_size"]
        new = nn.Embedding(old_n + n_new, dim)
        with torch.no_grad():
            new.weight[:old_n] = emb.weight
            mean = emb.weight.mean(dim=0, keepdim=True)
            new.weight[old_n:] = mean + 0.02 * torch.randn(n_new, dim)
        self.lm.set_input_embeddings(new)
        self.lm.resize_token_embeddings(old_n + n_new)  # ties lm_head if tied
        self.new_token_range = (old_n, old_n + n_new)
        self.traj_detokenize = traj_tok_spec["detokenizer_fn"]

    # ---------------- parameter groups ----------------

    def _is_diff(self, name: str) -> bool:
        return bool(DIFF_TOWER_PAT.search(name))

    def _is_ar(self, name: str) -> bool:
        return AR_TOWER_PAT.search(name) is not None and not self._is_diff(name)

    def param_groups_stage1(self):
        """AR tower + new embeddings trainable; entire gen tower frozen."""
        for n, p in self.model.named_parameters():
            if self._is_diff(n):
                p.requires_grad_(not self.cfg.student.freeze.diffusion_tower_stage1)
        base, boosted = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "embed" in n or "lm_head" in n:
                boosted.append(p)  # includes new trajectory rows
            else:
                base.append(p)
        mult = self.cfg.student.new_token_lr_mult
        return [{"params": base, "lr_mult": 1.0},
                {"params": boosted, "lr_mult": mult}]

    def param_groups_stage2(self):
        """Gen tower (incl. action-domain rows) trainable; AR tower frozen."""
        for n, p in self.model.named_parameters():
            if self._is_ar(n) or "embed" in n or "lm_head" in n:
                p.requires_grad_(False)
        if self.cfg.student.freeze.ar_tower_stage2_lora:
            self._attach_lora(rank=self.cfg.student.freeze.lora_rank)
        for n, p in self.model.named_parameters():
            if self._is_diff(n):
                p.requires_grad_(True)
        return [{"params": [p for p in self.model.parameters() if p.requires_grad],
                 "lr_mult": 1.0}]

    def _attach_lora(self, rank: int) -> None:
        # cosmos-framework has native LoRA injection (OmniMoTModel.add_lora,
        # injected pre-FSDP on meta device); route through it rather than peft.
        # VALIDATE-ON-GPU: confirm target module list for the AR-tower q/k/v/o.
        self.omni.add_lora(rank=rank, alpha=self.cfg.student.get("lora_alpha", 32))

    # ---------------- forward APIs used by the trainers ----------------

    def ar_forward(self, batch) -> dict:
        """Teacher-forced AR pass under the deployment context format:
        [cameras, egomotion, short CoC, trajectory tokens].
        Returns {"logits": (B, L, V), "hidden": {student_layer: (B, L, D_s)},
        "kv_cache": ...}.

        VALIDATE-ON-GPU: uses the reasoner (und) pathway of the unified MoT via
        the standard HF causal-LM interface; input assembly happens in the
        stage-1 collator once the Edge chat template for the deployment context
        is frozen (open design note in README).
        """
        out = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            use_cache=True,
        )
        hidden = {i: h for i, h in enumerate(out.hidden_states[1:])}
        return {"logits": out.logits, "hidden": hidden,
                "kv_cache": out.past_key_values}

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
        ts = self.net.time_embedder(sigma / self.net.timestep_scale
                                    if self.net.timestep_scale != 1.0 else sigma)
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

    def _gen_pathway_forward(self, gen_embeds, context_kv):
        """Run the MoT gen pathway over action-token embeds with und-context KV.
        VALIDATE-ON-GPU: wire through unified_mot's packed forward
        (set_gen_seq/get_gen_seq + PackedAttentionMoT with cached und K/V)."""
        raise NotImplementedError(
            "Gen-pathway packed forward - wire on the GPU box against "
            "cosmos_framework.model.generator.mot.unified_mot (see D-015).")

    @torch.no_grad()
    def generate_traj_tokens(self, batch) -> torch.Tensor:
        """Student-sampled discrete trajectory tokens (scheduled sampling, 4.2).
        Restricted to the appended trajectory vocab rows, mirroring the
        teacher-side Phase-B restriction (D-014)."""
        lo, hi = self.new_token_range
        out = self.lm.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=128, min_new_tokens=128, do_sample=True,
            suppress_tokens=list(range(0, lo)),  # only trajectory rows allowed
        )
        return out[:, batch["input_ids"].shape[1]:] - lo  # region-relative bins

"""TeacherWrapper: the ONLY file that touches the Alpamayo 2 Super codebase.

Everything downstream (labeler, stage-2 training) consumes the dataclasses
defined here. Implemented against the NVlabs/alpamayo2 source (read locally,
2026-08-17); see DECISIONS.md D-012..D-014 for the verified facts this relies on:
  - flow matching: x_t = t*x + (1-t)*noise, target u* = x - noise (D-012)
  - expert conditioning: identity KV pass-through over all VLM layers (D-004)
  - MRoPE expert positions: arange(n) + rope_deltas + kv_seq_len (D-006)
  - released inference stops generation AT <|traj_future_start|>; discrete
    trajectory tokens are emitted only if we continue generation ourselves,
    restricted to the future-token region (D-014)

Requires `pip install -e ../alpamayo2` (plus its deps) and HF access to the
gated nvidia/Alpamayo2-Super checkpoint. Probes run in config-only mode
(load_model=False) without downloading weights.
"""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class TeacherWindowOutput:
    """Everything cached per training window during the offline labeling pass."""
    traj_token_ids: torch.Tensor        # (T_traj,) int - REGION-RELATIVE bin ids [0, 3000)
    traj_topk_idx: torch.Tensor         # (T_traj, K) int - region-relative top-k bin ids
    traj_topk_logp: torch.Tensor        # (T_traj, K) - FULL-softmax log-probs at top-k
                                        # (not renormalized; losses.traj_topk_kl derives
                                        # the tail bucket, incl. out-of-region mass)
    coc_token_ids: torch.Tensor         # (<=max_coc,) int - short-CoC (teacher/Qwen ids)
    coc_text: str                       # decoded CoC text (re-tokenize with Edge tokenizer)
    meta_action: int                    # categorical id (cache_root/meta_action_vocab.json)
    meta_action_text: str
    feats: dict[int, torch.Tensor]      # teacher_layer_idx -> (pool_len, D_t)
    flow_t: torch.Tensor                # (K_flow,) fp32 - t is the DATA weight (D-012)
    flow_a_t: torch.Tensor              # (K_flow, H, A) fp32 - x_t = t*a + (1-t)*noise
    flow_v: torch.Tensor                # (K_flow, H, A) - teacher velocity u(x_t, t | KV)
    traj_samples: torch.Tensor          # (n_samples, H, A) fp32 - sampled actions (sanity)
    gt_traj: torch.Tensor               # (H, A) fp32 - GT future in ACTION space
    gt_future_xyz: torch.Tensor         # (T_fut, 3) fp32 - GT future xyz, ego frame (eval)


class TeacherWrapper:
    def __init__(self, cfg, device: str = "cuda", load_model: bool = True):
        import hydra.utils as hyu
        from alpamayo2_super.config import (
            Alpamayo2SuperConfig,
            build_alpamayo2_super_tokenizer,
            resolve_checkpoint_name_or_path,
        )
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

        self.cfg = cfg
        self.device = device
        repo = cfg.paths.teacher_repo
        if load_model:
            dtype = torch.bfloat16 if cfg.teacher.dtype == "bf16" else torch.float16
            self.model = Alpamayo2Super.from_pretrained(repo, dtype=dtype, device_map=device)
            self.model.eval()
            self.tconfig = self.model.config
            self.tokenizer = self.model.tokenizer
            self.future_traj_tokenizer = self.model.future_traj_tokenizer
        else:  # config-only mode: enough for the Phase-0 probes, no weights
            self.model = None
            self.tconfig = Alpamayo2SuperConfig.from_pretrained(repo)
            self.tokenizer = build_alpamayo2_super_tokenizer(
                resolve_checkpoint_name_or_path(self.tconfig),
                self.tconfig.history_vocab_size,
                self.tconfig.future_vocab_size,
            )
            self.future_traj_tokenizer = hyu.instantiate(
                self.tconfig.future_traj_tokenizer_cfg, load_weights=False)
        self._meta_vocab = None

    # ---------------- Phase 0 verification probes ----------------

    def probe_trajectory_tokenizer(self) -> dict:
        """Discrete future-trajectory tokenizer spec (verified: first-class teacher
        target, D-003). The identical tokenizer is copied verbatim into the student
        (student/edge_wrapper.py::extend_trajectory_vocab)."""
        c = self.tconfig
        tok = self.future_traj_tokenizer
        return {
            "vocab_size": c.future_vocab_size,
            "seq_len": c.tokens_per_future_traj,
            "future_id0": c.traj_ids["future_id0"],
            "traj_ids": dict(c.traj_ids),
            "binning": dict(c.future_traj_tokenizer_cfg),
            "detokenizer_fn": tok.decode,   # decode(hist_xyz, hist_rot, tokens) -> xyz, rot
            "tokenizer_fn": tok.encode,
        }

    def probe_expert_conditioning(self) -> dict:
        """Expert<->reasoner coupling contract, from config + code (D-001/002/004/012)."""
        c = self.tconfig
        vlm_text = c.vlm_config.text_config
        expert_llm = c.expert_config.llm_config
        return {
            # Identity KV pass-through: expert attends EVERY reasoner layer (D-004).
            "attended_layers": list(range(vlm_text.num_hidden_layers)),
            "kv_heads": vlm_text.num_key_value_heads,
            "head_dim": getattr(vlm_text, "head_dim",
                                vlm_text.hidden_size // vlm_text.num_attention_heads),
            "teacher_hidden": vlm_text.hidden_size,
            "expert_hidden": expert_llm.hidden_size,
            "parameterization": "flow_velocity",       # u* = x - noise (D-012)
            "interpolation": "x_t = t*x + (1-t)*noise; t=0 noise, t=1 data",
            "num_steps": self.tconfig.expert_config.diffusion_cfg.get(
                "num_inference_steps", 10),
            "schedule": {"train_timestep_sampler": "beta(1.5,1.0)",
                         "t_transform": "t = 0.999*(1-s), biased toward high noise"},
            "action_space": dict(c.expert_config.action_space_cfg),
            "expert_non_causal_attention": c.expert_config.expert_non_causal_attention,
        }

    def probe_tokenizer_vs(self, student_tokenizer) -> dict:
        """Diff teacher/student text vocabs (Phase 0.3). Trajectory tokens are exempt:
        they are appended to the student verbatim (D-003)."""
        t_vocab = self.tokenizer.get_vocab()
        s_vocab = student_tokenizer.get_vocab()
        shared = set(t_vocab) & set(s_vocab)
        same_id = sum(1 for tok in shared if t_vocab[tok] == s_vocab[tok])
        report = {
            "teacher_size": len(t_vocab), "student_size": len(s_vocab),
            "shared_tokens": len(shared),
            "shared_frac_of_teacher": len(shared) / max(len(t_vocab), 1),
            "same_id_frac_of_shared": same_id / max(len(shared), 1),
        }
        # Cross-tokenizer (Qwen vs Nemotron) => sequence-level KD on re-tokenized
        # CoC text (D-011); cache stores coc_text for exactly this.
        report["vocab_ok"] = report["shared_frac_of_teacher"] > 0.999 and \
            report["same_id_frac_of_shared"] > 0.999
        return report

    # ---------------- Labeling-pass API ----------------

    @torch.no_grad()
    def label_window(self, window, k_flow: int, topk: int,
                     max_coc: int, n_traj_samples: int) -> TeacherWindowOutput:
        """One full teacher pass over a preprocessed window (KV resident throughout).

        1. Phase A: released inference path - generate CoC + meta action, stopping
           right after <|traj_future_start|> (traj-region logits masked).
        2. Phase B: continue generation restricted TO the future-token region for
           exactly tokens_per_future_traj steps, capturing logits -> top-k KD targets.
        3. Pool hidden states of the configured teacher layers (feature KD / CKA).
        4. While KV is resident: k_flow stratified (x_t, t) -> teacher expert
           velocities u_teacher (teacher_flow supervision, D-007), then
           n_traj_samples full Euler rollouts (sanity).
        """
        from alpamayo2_super.helper import prepare_model_inputs, to_device
        from alpamayo2_super.models.alpamayo2_super import (
            MaskDiscreteTrajectoryLogitsProcessor,
        )
        from alpamayo2_super.models.alpamayo2_super import _append_text_eos_mask
        from alpamayo2_super.models.expert_utils import (
            StopAfterEOS,
            build_expert_pos_ids_and_attn_mask,
            find_eos_offset,
        )
        from alpamayo2_super.models.token_utils import extract_text_tokens
        from alpamayo2_super.models.utils import fuse_traj_tokens
        from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
        from transformers.generation.logits_process import LogitsProcessor
        import copy

        cfg, c = self.cfg, self.tconfig
        data = to_device(window.data, self.device)
        inputs = prepare_model_inputs(data, c, self.tokenizer)
        tokenized = dict(inputs["tokenized_data"])
        traj_data = {"ego_history_xyz": inputs["ego_history_xyz"],
                     "ego_history_rot": inputs["ego_history_rot"]}
        tokenized["input_ids"] = fuse_traj_tokens(
            self.model.history_traj_tokenizer, self.model.future_traj_tokenizer,
            tokenized["input_ids"], traj_data, c.traj_ids)

        future_start = c.traj_ids["future_start"]
        future_id0 = c.traj_ids["future_id0"]
        n_future = c.future_vocab_size

        # ---- Phase A: CoC + meta action (traj region masked), KV kept ----
        gen_cfg = copy.deepcopy(self.model.vlm.generation_config)
        gen_cfg.do_sample = True
        gen_cfg.top_p = float(cfg.teacher.get("gen_top_p", 0.98))
        gen_cfg.temperature = float(cfg.teacher.get("gen_temperature", 0.6))
        gen_cfg.max_new_tokens = max(max_coc + 8, 64)
        gen_cfg.return_dict_in_generate = True
        gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        proc = LogitsProcessorList([MaskDiscreteTrajectoryLogitsProcessor(
            traj_token_offset=min(c.traj_ids["history_id0"], future_id0),
            traj_vocab_size=c.traj_vocab_size)])
        _append_text_eos_mask(proc, gen_cfg.eos_token_id, preserved_token_id=future_start)
        out_a = self.model.vlm.generate(
            **tokenized, generation_config=gen_cfg,
            logits_processor=proc,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=future_start)]),
            output_hidden_states=True, use_cache=True)
        rope_deltas = self.model.vlm.model.rope_deltas
        prompt_len = tokenized["input_ids"].shape[1]

        # Feature targets: mean-pool prefill hidden states into fixed segments.
        feat_layers = [int(l) for l in cfg.teacher.get("feat_layers", []) or []]
        pool_len = int(cfg.teacher.get("feat_pool_len", 8))
        feats = {}
        if feat_layers and out_a.hidden_states:
            prefill_hs = out_a.hidden_states[0]  # tuple(layers+1) of (1, L, D)
            for l in feat_layers:
                h = prefill_hs[l + 1][0].float()             # (L, D); +1 skips embeddings
                seg = torch.chunk(h, pool_len, dim=0)
                feats[l] = torch.stack([s.mean(0) for s in seg])  # (pool_len, D)

        # ---- Phase B: emit discrete trajectory tokens, capture logits ----
        class _RestrictToFutureRegion(LogitsProcessor):
            def __call__(self, input_ids, scores):
                mask = torch.full_like(scores, float("-inf"))
                mask[:, future_id0:future_id0 + n_future] = 0.0
                return scores + mask

        gen_cfg_b = copy.deepcopy(gen_cfg)
        gen_cfg_b.min_new_tokens = c.tokens_per_future_traj
        gen_cfg_b.max_new_tokens = c.tokens_per_future_traj
        gen_cfg_b.output_logits = True
        seq_a = out_a.sequences
        out_b = self.model.vlm.generate(
            input_ids=seq_a,
            attention_mask=torch.ones_like(seq_a),
            past_key_values=out_a.past_key_values,
            generation_config=gen_cfg_b,
            logits_processor=LogitsProcessorList([_RestrictToFutureRegion()]),
            use_cache=True)
        # Raw (pre-mask) full-vocab log-probs; top-k taken inside the future region,
        # so the tail bucket absorbs any out-of-region teacher mass (D-014).
        step_logits = torch.stack([l[0] for l in out_b.logits])      # (T_traj, V)
        logp = F.log_softmax(step_logits.float(), dim=-1)
        region_logp = logp[:, future_id0:future_id0 + n_future]      # (T_traj, 3000)
        topk_logp, topk_idx = region_logp.topk(topk, dim=-1)
        traj_token_ids = (out_b.sequences[0, seq_a.shape[1]:] - future_id0).clamp(0, n_future - 1)

        # ---- CoC / meta action text ----
        extra = extract_text_tokens(self.tokenizer, seq_a)
        coc_text = extra["cot"][0]
        meta_text = extra["meta_action"][0]
        gen_a = seq_a[0, prompt_len:]
        gen_a = gen_a[(gen_a != future_start) & (gen_a != (gen_cfg.pad_token_id or -1))]
        coc_token_ids = gen_a[:max_coc].cpu()

        # ---- Flow targets: teacher expert velocities at stratified (x_t, t) ----
        expert = self.model.expert
        action = expert.action_space.traj_to_action(
            traj_history_xyz=data["ego_history_xyz"],
            traj_history_rot=data["ego_history_rot"],
            traj_future_xyz=data["ego_future_xyz"],
            traj_future_rot=data["ego_future_rot"],
        ).reshape(*expert.action_space.get_action_space_dims())          # (H, A)

        kv = out_b.past_key_values
        kv_len = kv.get_seq_length()
        offset = find_eos_offset(seq_a, eos_token_id=future_start,
                                 device=seq_a.device)                     # (1,)
        n_diff = expert.action_space.get_action_space_dims()[0]
        fkw = {"is_causal": False} if expert.config.expert_non_causal_attention else {}

        def expert_v(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Batched expert forward against the resident KV. x_t: (B, H, A),
            t broadcastable; KV must already be batch-repeated to B."""
            b = x_t.shape[0]
            pos, attn = build_expert_pos_ids_and_attn_mask(
                offset=offset.expand(b), rope_deltas=rope_deltas.expand(b, -1),
                kv_cache_seq_len=kv.get_seq_length(), n_diffusion_tokens=n_diff,
                b_star=b, device=x_t.device,
                prefix_mask=tokenized["attention_mask"].expand(b, -1))
            emb = expert.action_in_proj(x_t.to(self.model.dtype), t.to(self.model.dtype))
            if emb.dim() == 2:
                emb = emb.view(b, n_diff, -1)
            eo = expert.expert(inputs_embeds=emb, position_ids=pos, past_key_values=kv,
                               attention_mask=attn, use_cache=True, **fkw)
            kv.crop(kv_len)
            pred = expert.action_out_proj(eo.last_hidden_state)
            return pred.view(b, *expert.action_space.get_action_space_dims())

        eb = int(cfg.teacher.get("expert_batch", 4))
        kv.batch_repeat_interleave(eb)

        t_all = self.stratified_timesteps(k_flow).to(self.device)         # (K,)
        noise = torch.randn(k_flow, *action.shape, device=self.device)
        x_t_all = t_all.view(-1, 1, 1) * action[None] + (1 - t_all.view(-1, 1, 1)) * noise
        v_chunks = []
        for i in range(0, k_flow, eb):
            xt = x_t_all[i:i + eb]
            tt = t_all[i:i + eb].view(-1, 1, 1)
            if xt.shape[0] < eb:  # pad the last chunk to the KV batch size
                pad = eb - xt.shape[0]
                xt = torch.cat([xt, xt[-1:].expand(pad, -1, -1)])
                tt = torch.cat([tt, tt[-1:].expand(pad, -1, -1)])
                v_chunks.append(expert_v(xt, tt)[:eb - pad])
            else:
                v_chunks.append(expert_v(xt, tt))
        flow_v = torch.cat(v_chunks).float()

        # ---- Sampled trajectories (Euler rollout, sanity/eval) ----
        n_s = min(n_traj_samples, eb)
        def step_fn(x, t):
            v = expert_v(x[:eb] if x.shape[0] == eb else
                         torch.cat([x, x[-1:].expand(eb - x.shape[0], -1, -1)]),
                         t if t.shape[0] == eb else
                         torch.cat([t, t[-1:].expand(eb - t.shape[0], *t.shape[1:])]))
            return v[:x.shape[0]]
        traj_samples = expert.diffusion.sample(
            batch_size=n_s, step_fn=step_fn, device=self.device)          # (n_s, H, A)

        return TeacherWindowOutput(
            traj_token_ids=traj_token_ids.cpu(),
            traj_topk_idx=topk_idx.cpu(),
            traj_topk_logp=topk_logp.cpu(),
            coc_token_ids=coc_token_ids,
            coc_text=coc_text,
            meta_action=self._meta_action_id(meta_text),
            meta_action_text=meta_text,
            feats=feats,
            flow_t=t_all.cpu().float(),
            flow_a_t=x_t_all.cpu().float(),
            flow_v=flow_v.cpu(),
            traj_samples=traj_samples.cpu().float(),
            gt_traj=action.cpu().float(),
            gt_future_xyz=torch.as_tensor(window.gt_future_xyz),
        )

    def stratified_timesteps(self, k: int) -> torch.Tensor:
        """Low-discrepancy coverage of the schedule: one uniform draw per bin of
        [0, 0.999) split into k bins. t is the DATA weight (teacher convention,
        D-012); capped at 0.999 like the teacher's beta sampler, which also keeps
        gt_flow a0-recovery well-conditioned (1 - t >= 1e-3)."""
        u = torch.rand(k)
        return 0.999 * (torch.arange(k) + u) / k

    def _meta_action_id(self, text: str) -> int:
        """Persistent text -> categorical id registry (cache_root/meta_action_vocab.json)."""
        path = Path(self.cfg.paths.cache_root) / "meta_action_vocab.json"
        if self._meta_vocab is None:
            self._meta_vocab = json.loads(path.read_text()) if path.exists() else {}
        key = text.strip() or "<none>"
        if key not in self._meta_vocab:
            self._meta_vocab[key] = len(self._meta_vocab)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._meta_vocab, indent=2))
        return self._meta_vocab[key]

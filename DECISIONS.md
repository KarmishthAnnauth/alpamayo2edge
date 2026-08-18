# Decision Log — alpamayo2edge

Running record of verified facts, resolved blockers, and design decisions.
Convention: **[VERIFIED]** = confirmed from primary source (config/code/model card),
**[DECIDED]** = design choice we committed to, **[OPEN]** = still unresolved.

---

## D-001 [VERIFIED] Action expert objective is flow matching, not DDPM-style diffusion

- **Source:** `config.json` → `expert_config.diffusion_cfg._target_ = alpamayo2_super.diffusion.flow_matching.FlowMatching`, `int_method: euler`, `train_timestep_sampler: beta`.
- The model card's "diffusion-based, 10 diffusion steps" is shorthand for iterative denoising; the actual objective is velocity-field regression with Euler ODE integration at sampling.
- **Consequence:** Stage 2 cache stores flow-matching tuples; loss is velocity regression, not epsilon-prediction.
- Resolves the Phase-0 diffusion-vs-flow-matching blocker.

## D-002 [VERIFIED] Regression target space = UnicycleAccelCurvatureActionSpace

- **Source:** `config.json` action_space_cfg + `expert.py::_process_traj_future_training` (calls `action_space.traj_to_action` before noising).
- Flow field lives in normalized (accel, curvature) control space; 64 waypoints, dt=0.1s; XYZ+rot trajectories are a kinematic rollout of the controls.
- Normalization stats are published in config (accel_mean/std, curvature_mean/std, bounds) — use them verbatim, do not re-estimate.
- **Consequence:** Stage 2 cache stores actions in this space. Cache `(action, noise, t)` and reconstruct `x_t` / `u*` on the fly — cheaper than storing `x_t`, and immune to sign-convention errors.

## D-003 [VERIFIED] Stage 1 trajectory-token target = DiscreteTrajectoryTokenizer, 3000 bins

- **Source:** `config.json` → `future_traj_tokenizer_cfg`, `future_vocab_size: 3000`, `tokens_per_future_traj: 128`, traj token IDs 152669–155685 region.
- First-class teacher output; not something we invent.
- **Consequence:** Edge's appended trajectory vocabulary is defined against the *same* discretizer bins → teacher/student trajectory logits are in exact 1:1 correspondence → top-k tail-bucket KL is well-posed despite different text tokenizers (Qwen vs Nemotron).

## D-004 [VERIFIED] Layerwise-KV conditioning is an identity map, standard self-attention

- **Source:** `expert.py` — expert is a vanilla `AutoModel` receiving `past_key_values=vlm_outputs.past_key_values` directly. No remapping, no cross-attention modules, no adapters. `cache_layer_indices: null` is consistent (nothing to remap).
- Coupling contract is KV geometry equality: backbone 64 layers × 8 KV heads × head_dim 128; expert 64 layers × 8 KV heads × head_dim 128 (query widths differ: 5120 vs 1536 hidden — irrelevant to the cache interface).
- **Consequence:** the student's action decoder must match Edge AR tower's per-layer KV geometry. Edge's MoT shared-attention design satisfies this natively. Two-stage decomposition rationale now grounded in code.

## D-005 [VERIFIED] Timestep injection via input projection, not AdaLN

- **Source:** `expert.py` — `action_in_proj(noisy_x, timesteps)` (PerWaypointActionInProjV2, Fourier features); expert body is an unmodified transformer, no DiT-style AdaLN-zero.
- **Consequence:** student head needs an equivalent `(noisy_action, t) → embeddings` projection; no AdaLN grafting into Edge's diffusion tower required.

## D-006 [VERIFIED] Expert attention/positional details for the teacher wrapper

- `expert_non_causal_attention=True`: bidirectional attention among the 64 action tokens, on top of causal VLM prefix.
- MRoPE positions for expert tokens = `arange(n) + rope_deltas + past_key_values.get_seq_length()`. Offline labeling must replicate this offset or cached vs live expert behavior diverges.
- Padding: `cache_attention_mask` concatenated with all-ones expert mask; `padding_side: left` — carry the mask into any batched expert-side forward.
- Loss computed in fp32 (`pred.float()`) — mirror in the student loop.

## D-007 [DECIDED] Stage 2 supervision = (b) teacher-flow distillation, not GT flow matching

- **Options were:**
  - (a) `gt_flow`: replicate teacher's own objective — independent-coupling flow matching on ground-truth actions. Teacher contributes only via Stage 1 + conditioning. No teacher expert forwards needed during labeling.
  - (b) `teacher_flow`: regress the student's velocity field on the *teacher expert's predicted velocities* at sampled `(x_t, t)`. True expert distillation; requires teacher expert forwards during offline labeling.
- **Decision: (b).** Rationale: with the 5k-clip budget as the binding constraint, (b) is where the teacher's extra capacity actually transfers into Stage 2; (a) would just be training a smaller expert from scratch on the same data.
- **Cost accepted:** teacher expert forward passes per cached (x_t, t) sample during the offline labeling run. Expert is 2.3B — cheap relative to the 32B backbone forward that produces the KV cache; amortize by sampling multiple (x_t, t) per backbone forward.
- **Cache schema (updated):** per window, store `(x_t, t, u_teacher)` where `u_teacher = expert(x_t, t | KV)`. Note: with teacher_flow the target must be stored explicitly (it's a network output, not reconstructible from (action, noise, t) — that shortcut only applied to gt_flow).
- Config gains `stage2.supervision: gt_flow | teacher_flow` switch; keep gt_flow implemented as ablation baseline.

## D-008 [DECIDED] Layer mapping: CKA-based correspondence promoted to primary

- **Trigger:** Cosmos 3 Edge was trained **from scratch** (Nemotron-based), NOT initialized from Qwen3-VL weights — unlike Cosmos 3 Nano/Super. Teacher backbone is Qwen3-VL-32B. No shared pretraining lineage → uniform depth interpolation has no justification as a prior.
- Uniform interpolation demoted to sanity-check baseline. If CKA shows no clean correspondence structure, drop feature-level losses entirely rather than force a noisy layer map (feature KD is the dispensable component; output-interface losses carry the plan).

## D-009 [RESOLVED → D-017] Student action-head interface: bypass vs remap

- ~~Options: (i) bolt-on head matching Alpamayo's parameterization; (ii) remap into Edge's native 9D format.~~ — resolved 2026-08-17 after reading the Edge source (local `../cosmos-framework` clone): Edge ships a **designed multi-embodiment extension mechanism** (per-domain `DomainAwareLinear` action projections) that implements option (i) natively. See D-015 (source layout), D-016 (sign convention), D-017 (decision).

## D-010 [RESOLVED → D-012] Remaining Phase-0 code reads (teacher side)

- ~~`diffusion/flow_matching.py`: velocity sign convention, whether `construct_training_data` returns the target explicitly, Beta timestep-sampler parameters.~~ — read 2026-08-17 from the local clone, see D-012.
- (Resolved: ~~`models/expert.py`~~ — read 2026-08-15, see D-004/005/006.)

## D-011 [VERIFIED] Distillation taxonomy — which mechanism per interface, and why

- CoC text: **sequence-level KD** (teacher generations retokenized by Edge's Nemotron tokenizer, plain CE). Forced by cross-tokenizer mismatch; ULD/MinED machinery not worth it when generations are cached anyway.
- Trajectory tokens: **exact-vocab token-level logit KD** (top-k tail-bucket KL). Well-posed only because of the shared discretizer (D-003). Soft distribution carries the multimodal mode structure a hard argmax sequence would discard.
- Flow field: **capacity distillation of the velocity field** via cached supervised regression (D-007). NOT step/consistency distillation — that's a latency optimization, orthogonal, can stack later if 10 Euler steps are too slow on the RTX PRO 6000.
- Feature-level KD: optional, projector-based, CKA-matched (D-008); dispensable.
- Cross-family (Qwen→Nemotron) is normal in VLA distillation (cf. MiniVLA: OpenVLA/Llama-2 → Qwen2.5-0.5B via sequence-level KD); same-family prune-then-distill (Minitron) is the convenient case, unavailable here since Edge is a fixed artifact.

## D-012 [VERIFIED] Flow-matching schedule, target, and timestep sampler (resolves D-010)

- **Source:** `src/alpamayo2_super/diffusion/flow_matching.py` (read 2026-08-17 from the local clone).
- **Interpolation:** `noisy_x = t * x + (1 - t) * noise` — **t is the DATA weight** (t=0 pure noise, t=1 clean action). Sampling integrates t from 0 → 1 (`linspace(0, 1, steps+1)`, Euler, `x += dt * v`, init `x = randn * temperature`).
- **Velocity target:** `u* = x − noise` (data − noise). Matches the scaffold's `a_t = (1−t)a0 + t·a1, v* = a1 − a0` assumption exactly (a0 = noise, a1 = action) — the `train_stage2.py` a0-recovery shortcut is valid for the gt_flow ablation.
- **`construct_training_data` does NOT return the target explicitly.** It returns `{x, noisy_x, timesteps, noise, is_drop_guidance: None}`; the target is formed in `compute_loss_from_pred` as `(x − noise)`, plain MSE, with `pred.float()` upstream (fp32 loss, consistent with D-006).
- **Beta timestep sampler:** `s ~ Beta(1.5, 1.0)`, then `t = 0.999 · (1 − s)` — density of s peaks at 1, so **t is biased toward 0 (the high-noise end)** and capped at 0.999. Labeler stratification stays uniform (config choice) but over `[0, 0.999)` to match the cap.
- CFG machinery exists in the sampler but `ExpertModel.__init__` raises if enabled — irrelevant for us.
- **Last teacher-side Phase-0 blocker resolved.**

## D-013 [VERIFIED] Dataset access is via the `physical_ai_av` package, not raw NCore parsing

- **Source:** `alpamayo2/src/alpamayo2_super/load_physical_aiavdataset.py` + `physical_ai_av/src/physical_ai_av/dataset.py` (local clones).
- The teacher repo consumes PhysicalAI-AV through `physical_ai_av.PhysicalAIAVDatasetInterface` (HF-backed: streaming or local snapshot; `clip_index.parquet` for clip ids, `metadata/data_collection.parquet` for per-clip metadata, `get_clip_feature(clip_id, feature)` returning egomotion interpolators and camera decoders with `decode_images_from_timestamps`).
- `load_physical_aiavdataset(clip_id, t0_us, ...)` already produces the exact model-input dict (7-camera ring, 4 frames/camera at [t0−0.3s … t0], 16-step history / 64-step future at 10 Hz, ego-frame transform at t0). **The scaffold's stubbed "NCore V4 zarr.itar readers" are unnecessary** — `preprocess.py` now wraps this function; curation reads `data_collection.parquet` instead of per-clip JSONs.
- **Correction:** the teacher's S1 camera convention is a **7-camera ring including `camera_rear_tele_30fov`** — the config previously listed 6; fixed in `configs/default.yaml`.

## D-014 [DECIDED] Labeler design: two-phase generation for trajectory-token KD targets

- **Trigger (verified):** the released inference path (`sample_trajectories_from_data`) **never generates discrete trajectory tokens** — it masks the trajectory-token region during CoT and stops right after `<|traj_future_start|>`; the continuous trajectory comes from the expert. The discrete tokens exist as a *trained* target (training loss `future_traj_loss` on fused GT tokens), so the model can emit them if unmasked.
- **Decision:** the labeler runs **Phase A** (released path: CoC + meta action, traj region masked, stop after `future_start`, teacher sampling defaults top_p 0.98 / temp 0.6) then **Phase B**: continue generation from the resident KV for exactly `tokens_per_future_traj` steps with logits restricted to the future-token region, capturing per-step logits.
- **Top-k semantics:** full-vocab log-softmax, top-k taken *within* the 3000-bin future region, indices stored **region-relative** (`id − future_id0`). The tail bucket in `losses.traj_topk_kl` then also absorbs any out-of-region teacher mass. The student's appended trajectory vocab maps 1:1 onto region-relative bins (D-003).
- **Also cached:** `coc_text` + `meta_action_text` (decoded) so Stage 1 can re-tokenize with the Nemotron tokenizer without the teacher in memory (D-011 sequence-level KD); `gt_future_xyz` (ego-frame) for eval minADE; `gt_traj` is stored in **action space** (64×2, D-002) since Stage 2 losses live there. Meta action → categorical id via a grow-on-write registry `cache_root/meta_action_vocab.json`.
- **Expert forwards batched** against the resident KV via `batch_repeat_interleave(expert_batch)` (config, default 4) with `crop(kv_len)` after each call — same pattern as the release `step_fn`. Phase-B tokens appended to the KV are attention-masked out by `build_expert_pos_ids_and_attn_mask` (positions ≥ offset are masked), so expert conditioning matches the released path exactly.

## D-015 [VERIFIED] Cosmos 3 Edge source layout and MoT tower split

- **Source:** local clones `../cosmos` (NVIDIA/cosmos cookbooks) and `../cosmos-framework` (github.com/NVIDIA/cosmos-framework, shallow-cloned 2026-08-17 — this is where the model code lives; the cookbook repo only documents it).
- **Model classes:** the HF-Transformers `Cosmos3EdgeForConditionalGeneration` (transformers@main, no PyPI release yet) loads the **reasoner tower only** — sufficient for Stage 1. The full model (both towers + action head) loads via `cosmos_framework.inference.model.Cosmos3OmniModel.from_pretrained_dcp(<snapshot of nvidia/Cosmos3-Edge>)` → `OmniMoTModel` → `Cosmos3VFMNetwork(language_model=Nemotron3DenseVL unified MoT)`.
- **Tower split is per-parameter-name:** every generation-tower weight carries the **`_moe_gen` suffix** (`q/k/v/o_proj_moe_gen`, `mlp_moe_gen`, `input/post_attention_layernorm_moe_gen`, final `norm_moe_gen`); the reasoner tower is every language-model weight *without* the suffix (source comments state this split explicitly, `unified_mot.py:1518`). Joint attention `two_way`. This directly instantiates the freeze schedule: Stage 1 trains non-`_moe_gen`, Stage 2 trains `_moe_gen` + action projections.
- **Action interface (from `cosmos3_vfm_network.py` + `domain_aware_linear.py`):** actions are per-frame tokens `[T_i, action_dim]` entering via `action2llm` (DomainAwareLinear: action_dim → hidden) and exiting via `llm2action` (hidden → action_dim), each with **per-embodiment weight rows** (`num_embodiment_domains: 32`; Edge `max_action_dim: 64`), plus a learned `action_modality_embed` and per-token timestep embedding. AV native embodiment = 9D ego-pose deltas, 60 frames @ 10 FPS (cookbook action README). Renewed (2026-07-16) Cosmos3-Edge checkpoint ships trained action-head weights (`action_gen: true`); pre-renewal ones lacked them.
- **Student diffusion objective is also rectified flow** (`RectifiedFlow` per modality; `rectified_flow_action` with its own train-time distribution and `independent_action_schedule` support; inference `flow_shift` default 10, timestep shift `sigma = shift·t/(1+(shift−1)t)`).

## D-016 [VERIFIED] Teacher↔student flow conventions are SIGN-FLIPPED — mapping required

- **Teacher (D-012):** `x_t = t·data + (1−t)·noise`, target `u* = data − noise`, t = data weight.
- **Student (`rectified_flow.py::get_interpolation`):** `x_t = t·noise + (1−t)·data`, target `dot_x_t = noise − data`, **t = noise weight (σ)** — "aligned with the rectified-flow community notation", opposite of the teacher's.
- **Mapping (exact, no approximation):** `σ_student = 1 − t_teacher`; the interpolants coincide; `v*_student = −u_teacher`.
- **Consequence:** cache stays in teacher convention end-to-end; `EdgeStudent.flow_forward` converts internally (feeds σ = 1−t to the timestep embedding, negates the head output) and returns teacher-convention velocities, so `losses.flow_distill`/`flow_matching_gt` and `train_stage2.py` never see the flip. Getting this wrong trains on inverted velocity fields and is silent — it's the single most dangerous integration detail in Stage 2.
- The student's `flow_shift`/timestep-shift machinery only reweights which σ are *sampled*; it does not change the velocity field itself, so pointwise distillation at cached (x_t, σ) is unaffected. Student-side inference sampling can keep Edge's native shifted schedule.

## D-017 [DECIDED] Alpamayo action space enters Edge as a NEW embodiment domain (resolves D-009)

- **Decision:** register UnicycleAccelCurvature (64 waypoints × 2 dims, D-002) as a new embodiment domain in Edge's `DomainAwareLinear` action projections: 64 action tokens of raw dim 2, zero-padded into the 64-wide `max_action_dim` interface, `raw_action_dim = 2`, a fresh `domain_id` (config `student.action_domain_id`, default 31; verify the slot is unused in the shipped checkpoint's domain registry at integration).
- **Why this beats both original options:** it *is* option (i)'s bolt-on head, but through the model's designed extension mechanism — per-domain weights are embedding rows, so the new head trains without touching other domains or any architecture surgery, keeps 1:1 correspondence with the cached teacher velocities (no 9D translation error, option (ii)'s flaw), and stays checkpoint-compatible with Edge tooling.
- Trained in Stage 2 together with the `_moe_gen` tower; the new domain rows are exactly the "action head" parameters.
- Native-9D remains available later as a *deployment* conversion (kinematic rollout of accel/curvature → poses is deterministic, D-002), not a training-time remap.

---

## Scaffold changes implied — APPLIED 2026-08-17

1. ~~Stage 2 loss/cache: velocity regression, Euler sampler constants, fp32 loss.~~ (D-001, D-006, D-012)
2. ~~Cache schema: `(x_t, t, u_teacher)` explicit-target format; `supervision` switch in Stage 2 config.~~ (D-007)
3. ~~Labeling pipeline: teacher expert forwards per sampled (x_t, t); MRoPE offset + mask handling.~~ (D-006, D-007, D-014)
4. ~~Layer mapping: `mode: cka` default, `uniform` demoted.~~ (D-008)
5. ~~Teacher-side `# INTEGRATE:` stubs filled from the local alpamayo2 clone~~ (D-004–006, D-012–014). ~~Student-side stubs blocked on D-009~~ — resolved 2026-08-17 (D-015–017): `edge_wrapper.py` integrated against `../cosmos-framework` (load path, `_moe_gen` tower split, domain-aware action head, convention conversion). One remaining `# VALIDATE-ON-GPU` gap: the packed gen-pathway forward (`_gen_pathway_forward`) must be wired against `unified_mot.py`'s packed und/gen sequence utilities on the GPU box.

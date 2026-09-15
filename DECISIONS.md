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
- Flow field: **capacity distillation of the velocity field** via cached supervised regression (D-007). NOT step/consistency distillation — that's a latency optimization, orthogonal, can stack later if 10 Euler steps are too slow on the target GPU (RTX 6000 Ada 48GB).
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
- **Tower split is per-parameter-name:** every generation-tower weight carries the **`_moe_gen` suffix** (`q/k/v/o_proj_moe_gen`, `mlp_moe_gen`, `input/post_attention_layernorm_moe_gen`, final `norm_moe_gen`); the reasoner tower is every language-model weight *without* the suffix (source comments state this split explicitly, `unified_mot.py:1518`). Joint attention `two_way`. This instantiates the **tower split** used by the freeze schedule: Stage 1 touches non-`_moe_gen`, Stage 2 touches `_moe_gen` + action projections. (The *regime* — full FT vs LoRA — is set by D-024, not here.)
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
- Trained in Stage 2 together with the `_moe_gen` tower; the new domain rows are exactly the "action head" parameters. **Full-rank** even under D-024 (they are new parameters), with the other 31 domains' rows gradient-masked.
- Native-9D remains available later as a *deployment* conversion (kinematic rollout of accel/curvature → poses is deterministic, D-002), not a training-time remap.

---

## Scaffold changes implied — APPLIED 2026-08-17

1. ~~Stage 2 loss/cache: velocity regression, Euler sampler constants, fp32 loss.~~ (D-001, D-006, D-012)
2. ~~Cache schema: `(x_t, t, u_teacher)` explicit-target format; `supervision` switch in Stage 2 config.~~ (D-007)
3. ~~Labeling pipeline: teacher expert forwards per sampled (x_t, t); MRoPE offset + mask handling.~~ (D-006, D-007, D-014)
4. ~~Layer mapping: `mode: cka` default, `uniform` demoted.~~ (D-008)
5. ~~Teacher-side `# INTEGRATE:` stubs filled from the local alpamayo2 clone~~ (D-004–006, D-012–014). ~~Student-side stubs blocked on D-009~~ — resolved 2026-08-17 (D-015–017): `edge_wrapper.py` integrated against `../cosmos-framework` (load path, `_moe_gen` tower split, domain-aware action head, convention conversion). One remaining `# VALIDATE-ON-GPU` gap: the packed gen-pathway forward (`_gen_pathway_forward`) must be wired against `unified_mot.py`'s packed und/gen sequence utilities on the GPU box.

---

## D-018 [DECIDED] Teacher switched: Alpamayo 2 Super -> Alpamayo 1.5 (memory constraint)

- **Trigger:** Alpamayo 2 Super (34B: 32B reasoner + 2.3B expert, ~68 GB bf16 weights before KV
  cache and expert-batch activations) does not fit the available RTX GPU. Alpamayo 1.5 is
  **11.08B params / 22.16 GB bf16** (`model.safetensors.index.json` metadata), ~24 GB for
  single-sample inference and ~40 GB at 16 trajectory samples per NVIDIA's table.
- **Decision:** re-target the teacher side of the pipeline at `nvidia/Alpamayo-1.5-10B`
  (local clone `../alpamayo1.5`). Student side (Cosmos 3 Edge) is untouched.
- **What this costs:** the reasoner compression ratio drops from 32B -> 4B to **8B -> 4B**.
  The *expert* compression is unchanged (see D-019: A1.5's expert is ~2.28B, the same capacity
  as A2 Super's 2.3B expert), so **Stage 2's teacher-flow distillation (D-007) is unaffected**;
  only Stage 1's reasoner-capacity gap shrinks. **Settled by D-025:** under a
  capability-transfer framing the ratio is largely irrelevant, so this cost is minor.
- **What this buys, besides memory:** A1.5 is RL post-trained for reasoning/trajectory
  consistency, so cached CoC traces should be better aligned with the cached trajectory than
  A2 Super's. It also adds navigation conditioning and VQA (unused; see D-023).

## D-019 [VERIFIED] Alpamayo 1.5 checkpoint architecture (from config.json + safetensors index)

Source: `../alpamayo1.5/{config.json, model.safetensors.index.json}` (HF snapshot, read 2026-08-18).

- **Backbone:** `vlm_name_or_path: nvidia/Cosmos-Reason2-8B`, `vlm_backend: qwenvl3` (loaded via
  `Qwen3VLConfig`, `base_model.py:376`). **36 language-model layers** + a 27-block ViT.
  Still cross-family vs the student's Nemotron-based Edge -> **D-008 (CKA layer mapping primary)
  stands**; only the layer count changes, 64 -> 36.
- **Expert:** 36 layers, 1:1 with the backbone (`expert.layers.0..35`), `hidden_size 2048`,
  `num_attention_heads 16`, `head_dim 128`, `intermediate_size 8256`, no `embed_tokens`,
  own final `expert.norm`. **~2.28B params - the same expert capacity as A2 Super's 2.3B.**
- **D-004 holds and is now structural:** the expert config is `deepcopy(vlm.config.text_config)`
  with only the four `expert_cfg` keys overridden (`alpamayo1_5.py:94-99`), so
  `num_hidden_layers` and `num_key_value_heads` are inherited -> KV geometry equality is
  guaranteed by construction, not by coincidence. (KV-head count itself comes from the
  Cosmos-Reason2-8B config, not from these two files - read it on the GPU box.)
- **D-002 holds verbatim**, with published normalization stats to use as-is:
  `accel_mean 0.02902694707164455`, `accel_std 0.6810426736454882`,
  `curvature_mean 0.0002692167976330542`, `curvature_std 0.026148280660833106`,
  bounds +/-9.8 and +/-0.33, `dt 0.1`, `n_waypoints 64`.
- **D-003 holds with IDENTICAL numbers** - the earlier worry that the token vocabulary shrank
  was wrong (it came from stale in-code defaults, not the checkpoint):
  `traj_tokenizer_cfg = DiscreteTrajectoryTokenizer` over the same UnicycleAccelCurvature space,
  **`num_bins: 3000`**, `dims_min [-10,-10]`, `dims_max [10,10]`,
  **`tokens_per_future_traj: 128`** (= 64 waypoints x 2 dims).
  Edge's appended trajectory vocabulary (3000 rows) needs **no change**.
- **Token layout - the one real difference, and it is an ORDER SWAP:**
  `traj_token_start_idx: 151669`, `traj_vocab_size: 4000`.
  A1.5 puts the **future bins first**: future = `[151669, 154669)`, history =
  `[154669, 155669)` (1000-bin `DeltaTrajectoryTokenizer`). A2 Super was the other way round
  (history first, future at +1000). So `future_id0 = traj_token_start_idx` here, *not*
  `start + history_vocab_size`. Region-relative bin ids (D-014) are unchanged.
  Specials: `history_start 155674`, `history_end 155676`, `future_start 155681`,
  `future_end 155683`, `history 155684`, `future 155685`; `vocab_size 155697`.
- `tokens_per_history_traj: 48` (16 waypoints x 3 xyz deltas) - matches the hardcoded
  `num_traj_token = 48` placeholder in `helper.create_message`.
- **D-005 holds:** `action_in_proj = PerWaypointActionInProjV2(hidden 512, 20 Fourier feats,
  max_freq 100, 2 enc layers)`; `action_out_proj = nn.Linear(2048 -> 2)`.
- **D-006 holds:** `expert_non_causal_attention: true`, `padding_side: left`,
  `min_pixels 163840 / max_pixels 196608`, `include_camera_ids: true`, `include_frame_nums: true`.
- Diffusion: `FlowMatching(int_method=euler)`, defaults `num_inference_steps=10`, CFG off.
- **Checkpoint layout is flat** (`vlm.*`, `expert.*`, `action_in_proj.*`, `action_out_proj.*`)
  and loads in one call: `Alpamayo1_5.from_pretrained(...)`. No submodule assembly like A2.

## D-020 [VERIFIED] D-012's flow convention survives, but its primary source is gone

- A1.5's released `diffusion/flow_matching.py` ships the **sampler only**: no
  `construct_training_data`, no `compute_loss_from_pred`, no Beta timestep sampler.
- The sampler is byte-equivalent in behaviour to A2's (`linspace(0,1,steps+1)`, Euler
  `x += dt*v`, init `randn*temperature`), which pins the same convention:
  `x_t = t*x + (1-t)*noise`, `u* = x - noise`, **t = data weight**.
- **Consequence:** D-012 downgrades from "verified from training code" to "inferred from the
  sampler". Nothing changes operationally - the labeler stratifies t itself. The `0.999` cap in
  `TeacherWrapper.stratified_timesteps` is now *our* choice (it keeps gt_flow a0-recovery
  well-conditioned), not a mirror of the teacher's sampler.
- **D-016 (student sign flip, sigma = 1 - t, v* = -u) is therefore unchanged** and still the
  single most dangerous integration detail in Stage 2.

## D-021 [VERIFIED] Meta-action does not exist in Alpamayo 1.5 - drop it from the cache

- `config.json` sets `add_special_tokens: true`, so the tokenizer registers
  `base_model.SPECIAL_TOKENS`, whose slots 10-11 are `_padding_2`/`_padding_3` - exactly where
  A2 Super has `meta_action_start`/`meta_action_end`. `token_utils.extract_text_tokens` still
  asks for `"meta_action"` but will always return `""`.
- **Consequence (amends D-014):** drop `meta_action` / `meta_action_text` from
  `TeacherWindowOutput`, the npz shard, and the collator; delete the
  `cache_root/meta_action_vocab.json` registry and the "meta-action auxiliary head" idea from
  the README. `curation.py`'s meta-action strata fall back to metadata-only stratification.

## D-022 [CLOSED -> D-031] Phase-B discrete-trajectory-token emission is unverified for A1.5

- For A2 Super, D-014 rested on a *training* loss (`future_traj_loss`, `alpamayo2_super.py:232`)
  proving the discrete future tokens are a trained target. A1.5's release is inference-only and
  **strips the future-fusion path**: `TrajectoryFusionMixin.fuse_traj_tokens`
  (`base_model.py:172-201`) validates the future-tokenizer attributes and then fuses
  **history only**.
- Evidence the target exists anyway: the checkpoint defines a full 3000-bin future tokenizer,
  `<|traj_future|>`/`<|traj_future_start|>`/`<|traj_future_end|>` ids, and
  `token_utils.extract_traj_tokens` parses exactly that span.
- **This is now the largest risk in the plan.** Validate FIRST on 2-3 debug clips: unmask the
  future region, continue generation for `tokens_per_future_traj` steps, detokenize, and check
  the result is a sane trajectory (compare against the expert's own Euler rollout).
- **Fallback if it fails:** Stage 1 loses its primary target and reduces to sequence-level CoC KD
  (D-011) plus a continuous trajectory target; the coarse-minADE gate would have to run off the
  expert instead of detokenized tokens.

## D-023 [DECIDED] 4-camera configuration (amends D-013)

- A1.5's `load_physical_aiavdataset` defaults to **4 cameras**:
  `[cross_left_120fov, front_wide_120fov, cross_right_120fov, front_tele_30fov]`, and NVIDIA's
  own ablation notebook (`notebooks/inference_cam_num.ipynb`) tops out at 4. The camera-index
  map still covers all 7 (0..6), and A1.5 explicitly supports a variable camera count.
- **Decision:** label with the 4-camera reference set. It is the configuration NVIDIA
  demonstrates, it shortens the prompt and the KV cache (cheaper labeling), and it is closer to
  the student's deployment context than the 7-camera ring.
- This reverses the D-013 correction, which was an A2 Super fact ("7-camera ring including
  `camera_rear_tele_30fov`"). `configs/default.yaml:data.cameras` updated accordingly.

---

## Scaffold changes implied by the teacher swap — APPLIED 2026-08-18

1. `configs/default.yaml`: teacher repo, 4 cameras, `feat_layers` re-spaced over 36 layers,
   meta-action strata removed. (D-018, D-019, D-021, D-023)
2. `data/preprocess.py`: A1.5 loader has no `include_calibration` kwarg and returns no
   `camera_names` key — names now derived from the configured camera list. (D-019)
3. `teacher/wrapper.py`: rewritten against the A1.5 API surface — inlined expert, static
   `_find_eos_offset` / `_build_expert_pos_ids_and_attn_mask`, `ExpertLogitsProcessor`,
   `helper.create_message` + `processor.apply_chat_template` instead of the deleted
   `prepare_model_inputs`/`build_conversation`, `future_id0 = traj_token_start_idx`. (D-019)
4. `teacher/labeler.py`, `data/dataset.py`, `data/curation.py`: meta-action removed. (D-021)
5. Student side (`student/edge_wrapper.py`, `losses.py`, `train_stage2.py`, `eval/`):
   **no changes from the teacher swap** — the trajectory vocabulary is still 3000 bins and the
   flow conventions on both sides are unchanged (D-019, D-020). Note the student side *does*
   change under D-024 (LoRA), which is separate and **still pending** — see
   `TRAINING_STRATEGY.md` §6.

---

## D-024 [DECIDED] LoRA adaptation for both stages, not full fine-tuning

- **Supersedes the freeze schedule's training regime** in D-015/D-017: those entries
  established *which tower belongs to which stage* (still correct); the implicit "the
  active tower is fully trainable" was inherited from the 96GB premise, never a considered
  choice, and is now wrong.
- **Decision:** LoRA on existing pretrained weights — Stage 1 on the AR tower's
  `q/k/v/o_proj` (rank 32-64), Stage 2 on the gen tower via cosmos-framework's own shipped
  default `q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen` (rank 16, alpha 32).
  Full-rank training **only** for genuinely new parameters: the 3000 appended
  trajectory-vocab rows and the new action-embodiment rows (D-017), ~12M params.
- **Rationale:** ~10k training windows is five to six orders of magnitude below a
  distillation pretraining corpus; the gen tower is where Edge's physical prior lives and
  Stage 2 was about to full-FT all 1.94B of it on that data. `cosmos-framework` ships a
  first-class LoRA path (`utils/generator/lora.py`) whose Edge defaults target exactly the
  gen tower — NVIDIA's intended adaptation mechanism, which we were about to ignore.
- **Also required (not optional):** gradient-mask the row-indexed tensors LoRA does not
  cover — embedding/lm_head rows below the appended range, and the 31 non-target embodiment
  rows in `action2llm`/`llm2action`. Otherwise forgetting leaks back in through them.
- **Accepted cost:** LoRA may underfit, making a failed Stage-1 gate ambiguous. Mitigated by
  a full-FT ablation on the 500-clip increment only.
- **Side effect:** resolves the 48GB budget. Stage-1 trainable params drop 2.48B -> ~25M;
  full FT needed 43.5GB before activations and did not fit. The earlier 48GB mitigations
  (8-bit AdamW, frozen-embedding trick, `micro_batch: 1`) are withdrawn. EMA stays off.
- Reasoning and numbers: `TRAINING_STRATEGY.md` §2-§3.

## D-025 [DECIDED] Framing: domain-adaptive capability transfer, not compression

- **Decision:** position the work as capability transfer into a physically-grounded
  generalist, not as compression distillation. Report **task gain and retention as a pair**;
  make **teacher-supervised vs GT-only at matched data/compute** the headline comparison
  rather than an ablation.
- **Consequence for D-018:** the reasoner compression ratio (32B->4B becoming 8B->4B after
  the teacher swap) is close to irrelevant under this framing. D-018's "reframe the thesis
  contribution" note is settled by this entry.
- **Consequence:** a retention eval is required in `eval/` — without one the plan cannot
  detect forgetting at all. Under this framing it is half the result, not hygiene.
- **Known tradeoff to state in the thesis, not hide:** the cached-target design is
  off-policy by construction (the teacher is gone by Stage 2 — that is what makes single-GPU
  work), against a field trend toward on-policy KD. `scheduled_sampling_start_frac: 0.5`
  partially mitigates. Do not change the design.
- Full reasoning: `TRAINING_STRATEGY.md` §1, §4, §5.

## D-026 [VERIFIED] D-024 implemented; four latent bugs found while wiring it

Applied 2026-08-22 against the local `../cosmos-framework` clone. New files:
`student/lora.py`, `checkpoint.py`, `optim.py`, `eval/retention.py`,
`eval/probes/general_probes.json`, `scripts/06_retention.py`,
`tests/test_lora_offline.py`. Changed: `student/edge_wrapper.py`,
`train_stage1.py`, `train_stage2.py`, `configs/default.yaml`.

**Corrections to what TRAINING_STRATEGY §6 assumed:**

- `OmniMoTModel.add_lora` **does exist** (`omni_mot_model.py:5537`); §6 says it does not.
  It is a forwarder to `inject_lora_pre_fsdp(network, lora_rank=, lora_alpha=,
  lora_target_modules=)`. The old `_attach_lora` would still have crashed — it called
  `add_lora(rank=, alpha=)`, a signature that matches neither — so the conclusion held for
  the wrong reason. We call the free function directly because it takes the subtree, and
  the subtree is the part that matters:
- **Injection must be scoped to `language_model.model.layers`.** The framework matches
  plain targets by leaf NAME across whatever root it is handed, and the reasoner's SigLIP2
  vision tower — lazily attached at `language_model.visual` by `_ensure_vision_tower` —
  names its attention leaves `q_proj`/`k_proj`/`v_proj` too. Passing the causal LM as root
  would silently adapt the vision encoder as well. Inside the layer stack the `_moe_gen`
  suffix keeps the two towers apart (D-015), so one scope serves both stages.
- Adapters are meta-device even on an already-materialized model. We `to_empty()` the
  `lora_A`/`lora_B` **submodules** — `to_empty()` on the wrapper would discard the
  pretrained base weight it deliberately preserves.

**Bugs found and fixed while wiring (all would have surfaced only on the GPU box):**

1. **`extend_trajectory_vocab` never resized `lm_head`.** It called HF's
   `resize_token_embeddings`, but `Nemotron3DenseVLTextForCausalLM` sets
   `_tied_weights_keys = []` and defines no `get_output_embeddings`, so only the INPUT
   embedding grew. The 3000 appended trajectory ids would have had **no logits at all** and
   every Stage-1 trajectory loss would have been taken over unreachable rows. Both tables
   are now resized explicitly, and `config.vocab_size` is kept in sync.
2. **Timestep conversion was off by 1000x.** `flow_forward` divided sigma by
   `net.timestep_scale`. The network embeds `action.timesteps * timestep_scale`
   (`cosmos3_vfm_network.py:788`) where timesteps are DISCRETE scheduler steps and
   `timestep_scale = timestep_range / num_train_timesteps` (`omni_mot_model.py:264`), so
   the embedder's input range is `[0, timestep_range)` — and `timestep_range` is **1.0**
   for Edge (`edge_model_config.py:63`). A normalized sigma passes through unscaled.
3. **The Stage-1 cosine schedule never fired.** `if "initial_lr" in g` was never true —
   AdamW does not create that key — so the LR stayed flat for the whole run. `optim.py`
   now stamps `initial_lr` on every group.
4. **Stage 2 could not have loaded Stage 1.** Besides the unmerged-LoRA-keys problem D-024
   predicted, `train_stage2.py` never called `extend_trajectory_vocab`, so the checkpoint's
   extended tables would have shape-mismatched a fresh student; and
   `student.model.from_pretrained(dir)` (called on the instance) builds a SECOND model
   rather than filling the resident one. Now: extend vocab -> `checkpoint.load_into`.

**Checkpoints are merged, non-destructively.** `checkpoint.save` folds `lora_B @ lora_A *
(alpha/rank)` into `<path>.weight` in a CPU copy and drops the adapter keys; the live model
keeps its adapters so per-epoch saves do not end training. `load_into` refuses an unmerged
checkpoint and checks vocab geometry before touching weights. `merge_lora_()` is the
in-place variant, for export only. The merge math and both gradient row-masks are pinned by
`tests/test_lora_offline.py` (8 tests, torch-only, no GPU — all passing).

**Judgment call, flagged:** `action_modality_embed` and `time_embedder` are shared across
all 32 embodiments, so no row mask can protect them. They stay **frozen** by default
(`student.lora.train_shared_action_embeds: false`); the per-domain rows already give the
teacher's action space its own affine interface. Flip it only if Stage 2 underfits.

**Config:** `student.freeze` replaced by `student.lora`; stage-1 rank 48/alpha 96,
stage-2 rank 16/alpha 32 (the shipped Edge default). `micro_batch` doubled in both stages
with `grad_accum` halved, holding the effective batch at 32. LRs left at their full-FT
values — LoRA usually wants 2-5x more, and that is the first knob if Stage 1 underfits.

## D-027 [OPEN] The AR pathway is not the HF interface — `ar_forward` cannot run as written

Found while wiring D-026's retention eval, which needed the same pathway.

`Nemotron3DenseVLTextForCausalLM.forward(pack: SequencePack, attention_mask, position_ids,
...)` takes a **packed sequence**, not `input_ids`. So both of these, written against the
HF causal-LM interface, are wrong:

- `EdgeStudent.ar_forward` — `self.lm(input_ids=..., output_hidden_states=True,
  use_cache=True)`
- `EdgeStudent.generate_traj_tokens` — `self.lm.generate(...)`, which routes through that
  same pack-based `forward`

The real reasoner-tower API (`unified_mot.py`) is:

- `language_model.model.reasoner_forward(input_ids, cache, position_ids=None,
  inputs_embeds=None, ...) -> [B, T, hidden]` — final **post-norm** hidden states, so
  `lm_head(hidden)` gives logits;
- `language_model.generate_reasoner_text(input_ids, max_new_tokens, *, do_sample,
  temperature, top_k, top_p, eos_token_id, pad_token_id, seed, return_only_new_tokens, ...)`
  — greedy or sampled decode with a `ReasonerKVCache`, and the image/video-conditioned
  prefill path;
- per-layer KV lives in `ReasonerKVCache(keys, values)` in BSHD layout — not an HF
  `past_key_values`, which is what `train_stage2.py` currently forwards as `context_kv`.

`eval/retention.py` is already written against these and is correct. **Two things block a
straight port of `ar_forward`:**

1. `reasoner_forward` returns only the FINAL hidden states, but feature KD (D-008) needs
   the 8 mapped intermediate layers. `MoTDecoderLayer.reasoner_forward` is called directly
   rather than through `__call__`, so ordinary `nn.Module` forward hooks will NOT fire —
   capturing per-layer output means wrapping the bound method per layer.
2. Trajectory-token sampling needs the vocab restriction Phase B uses (D-014).
   `generate_reasoner_text` exposes `top_k`/`top_p`/`temperature` but **no
   `suppress_tokens`**, which is what `generate_traj_tokens` relied on. Either post-mask
   logits through a custom decode loop over `reasoner_forward` + `ReasonerKVCache`, or
   accept unrestricted sampling and reject off-vocabulary draws.

Neither is blocked on hardware — but (1) touches the frozen-context/chat-template question
that Stage-1 input assembly is already waiting on, so do them together.

## D-028 [VERIFIED] Student input format = the teacher's, verbatim

New sibling repo `../alpamayo-recipes` (github.com/NVlabs/alpamayo-recipes, cloned
2026-08-22) — NVIDIA's post-training recipes. It carries the **training-time** chat
template that the release only implied: `src/alpamayo/chat_template/{r1,r1_5}.py` +
`components.py`, with the component order pinned by
`recipes/alpamayo1_5_sft/configs/vla_processor/default.yaml` and by
`tests/test_recipe_static_contracts.py`.

**Decision:** the student's deployment context mirrors the teacher's input exactly —
cameras, ego motion, short text instruction — and the student produces CoC then
trajectory tokens. This closes the "freeze Edge's deployment-context chat template"
open item.

**The format, cross-validated by two independent sources that agree** — the released
`alpamayo1_5/helper.py::create_message` (which `teacher/wrapper.py` already uses for
labeling) and the recipes training template:

    system     "You are a driving assistant that generates safe and accurate actions."
    user       per camera, ASCENDING camera index:
                 "<Display name>: " then per frame "frame {i} " + <image>
               "<|traj_history_start|>" + "<|traj_history|>" x48 + "<|traj_history_end|>"
               ["<|route_start|>" nav "<|route_end|>"]        # unused for now
               "output the chain-of-thought reasoning of the driving process,
                then output the future trajectory."
    assistant  "<|cot_start|>" cot "<|cot_end|>"
               "<|traj_future_start|>" + 128 bins + "<|traj_future_end|>"

Component order is `image -> traj_history -> [route] -> prompt`; in generation mode the
assistant turn is opened with `<|cot_start|>` and nothing else. History is **48** slots
(`helper.py: num_traj_token = 48`), future is 128 bins (D-014). Camera display names and
indices are `alpamayo.common.constants` (index orders the prompt — `construct_image`
asserts ascending; our 4-camera set is 0,1,2,6, already ascending, D-023).

**Two things are the student's own and are NOT copied:**

1. **Role scaffolding** — Edge has its own chat template and role tokens. We mirror the
   content and its order, not the teacher's turn markup. Hence `student/prompt.py`
   emits SEGMENTS, not one pre-rendered string.
2. **Image placeholder count per frame** — a property of Edge's SigLIP2 tower and
   processor (`_ensure_vision_tower` plumbs `config.image_token_id`), not of the
   teacher. `assemble()` takes it as a callable so the real processor decides it on the
   GPU box and the module stays testable offline.

**Ego motion rides in the 48 reserved slots, as in the teacher** — A1.5 overwrites those
slot embeddings with a fused projection of the continuous ego history
(`model.fuse_traj_tokens`). Edge has no such module, so we owe it one
(`EgoHistoryEncoder`, still to write): a new full-rank trainable module, which is
exactly what D-024 says new interfaces get. **Rejected:** serializing ego motion as text
numerals — lossy, far more tokens, and it stops being "the teacher's input".

**Also appended to the student vocabulary** alongside the 3000 trajectory bins: the 9
structural special tokens above. The student's tokenizer knows none of them, and the
CoC/trajectory losses key off the spans they delimit.

`student/prompt.py` implements this; `tests/test_prompt_offline.py` pins the rendered
context against the expected literal (12 tests, no GPU).

**Gap this exposes:** `collate_stage1` produces no `input_ids` and no images at all — the
cached shards are teacher TARGETS only (`labeler.py`'s `savez_compressed`). Stage-1
training must pair each shard with a re-loaded window (`preprocess.load_window` already
returns student-resolution frames per camera plus `ego_history_xyz`/`rot`). That
plumbing, `EgoHistoryEncoder`, and the D-027 port are one connected piece of work.

## D-029 [VERIFIED] Stage-1 input path built; ego motion is DISCRETE, not a projection

Implements D-028 and closes D-027. New: `student/prompt.py`, `student/context.py`,
`data/dataset.py::Stage1Dataset` + `collate_student`, `losses.gather_targets`,
`tests/test_{prompt,context}_offline.py`. Changed: `student/edge_wrapper.py`
(`ar_forward`, `generate_traj_tokens`, `extend_trajectory_vocab`), `teacher/wrapper.py`,
`data/preprocess.py`, both trainers, `eval/coarse_minade.py`.

**Correction to D-028: no `EgoHistoryEncoder` is needed, and none was written.**
`fuse_traj_tokens` sounds like an embedding fusion but is not. It calls
`tokenize_history_trajectory` (`base_model.py:95`), which runs a SECOND tokenizer —
`DeltaTrajectoryTokenizer`, 1000 bins — over the ego history and then
`replace_pad_token`s the resulting ids into the `<|traj_history|>` slots. Ego motion is
therefore discrete token ids, exactly like the future trajectory, and the student needs
no continuous side-channel at all. D-028's "we owe Edge a projection module" was wrong.

That tokenizer is called with the history passed as the FUTURE argument and only the
first pose kept as the reference — an inversion that is easy to get backwards, so the
teacher wrapper now hands the student a ready-made `hist_tokenize_fn` closure rather
than raw `encode`.

**Vocabulary layout** (checkpoint `config.json`, so [VERIFIED]): `traj_vocab_size = 4000`,
`traj_token_start_idx = 151669`, `tokens_per_history_traj = 48`,
`tokens_per_future_traj = 128`. Future occupies the first 3000 bins, history the next
1000. The student appends **4000 bins + 9 structural special tokens = 4009 rows**, laid
out `[future | history | specials]`, with `future_base` / `hist_base` / `special_ids`
recorded on the wrapper. Previously it appended only 3000, which would have left the ego
slots unrepresentable.

**D-027 resolved.** `ar_forward` now runs `lm.model.reasoner_forward(...)` + `lm_head`.
Per-layer hidden states for feature KD come from `_capture_layers`, which shadows each
layer's bound `reasoner_forward` for the duration — `_impl_reasoner_forward` calls that
method DIRECTLY, so `register_forward_hook` never fires. `generate_traj_tokens` is the
framework's own decode loop (`ReasonerKVCache`, mrope decode positions, its
`_sample_next_token`) plus a per-step logit mask restricting emission to the future-bin
rows, since `generate_reasoner_text` has no `suppress_tokens`.

**Right-padding is now a contract, not a preference.** `reasoner_forward` takes NO
attention mask — the tower is causal by construction. Trailing padding can never reach a
real token; leading padding would corrupt every position. `collate_student` right-pads
and the losses mask.

**Two alignment rules in `losses.gather_targets`,** both silent failures if wrong:
the logit that predicts position `p` is at `p-1`; and CoC targets are read back out of
`input_ids` rather than from the cache, because the cached ids are in the TEACHER's
vocabulary while the student's own ids for the same text are already in the assembled
sequence. That sidesteps the D-011 vocab-match question entirely. Trajectory top-k
indices still need `+ future_base`, since the cache stores region-relative bins (D-014).

**Window addressing.** Shards are named `{window_idx:02d}.npz` and store no timestamp, so
Stage-1 recovers a window's `t0_us` from its index through
`preprocess.window_t0s_us(cfg)` — now the single home for that formula. **Changing it
silently repoints every existing cache at different video.**

**Still VALIDATE-ON-GPU:** which image processor the Cosmos3-Edge snapshot ships
(`context.load_image_processor` tries `AutoProcessor` then `AutoImageProcessor`) and
whether its `pixel_values`/`image_grid_thw` match what
`prepare_multimodal_reasoner_inputs` expects. Placeholder counts per frame are derived
from the processor's own returned grid, so nothing guesses a constant.

**Unchanged and still open:** `_gen_pathway_forward` (the packed gen-pathway forward).
Stage 2 now passes the reasoner's final hidden states as context instead of the
non-existent `past_key_values`, but the packed und/gen wiring is still the remaining
GPU-box piece.

## D-030 [VERIFIED] The teacher/student pairing is vendor-endorsed; no recipe exists for it

Two independent NVIDIA statements bracket this project, and the gap between them is
the contribution.

**Cosmos 3 Edge release article** (user-sourced, 2026-08-22): the model "kann als
Student-Backbone für die Destillation automobiler Richtlinienmodelle eingesetzt werden,
unter anderem mit NVIDIA Alpamayo Vision-Language-Action-Modellen" — usable as a student
backbone for distilling automotive policy models, expressly including with Alpamayo VLA
models. That is this thesis, named by the vendor.

**`../alpamayo-recipes/README.md`** describes Alpamayo 2 Super's uses as including "a
teacher model for distillation and quantization into student models that meet in-vehicle
latency and safety requirements on DRIVE AGX Thor". Both ends of the pairing are
positioned for exactly this.

**But neither repo ships a recipe for it.** Verified 2026-08-22:
- `alpamayo-recipes` advertises "post-train, quantize or distill Alpamayo VLA models" in
  its purpose table, yet `recipes/` contains only `alpamayo1_5_quant`, `alpamayo1_5_sft`,
  `alpamayo1_sft`, `alpamayo1_x_rl`. There is no distillation recipe and no roadmap entry
  promising one.
- The only distillation cookbooks in `../cosmos` are **DMD2 step distillation**
  (Cosmos3-Super T2I/I2V -> 4-step students). That is TRAINING_STRATEGY §1's bucket 3 —
  compressing sampling STEPS, not transferring capability — and D-011 already ruled it
  out as a different mechanism. Confirmed now that it is the ONLY distillation NVIDIA
  publishes for Cosmos 3.

**What this settles:** the "why this pairing" question, which is now answered by the
vendor rather than argued by us. Motivation should cite both statements.

**What it does NOT settle, and should not be read as settling:**
- The teacher is **Alpamayo 1.5, not 2 Super** (D-018, a memory-driven swap). NVIDIA's
  explicit "teacher for distillation" language attaches to the 34B Super. The capability-
  transfer framing (TRAINING_STRATEGY §1) is what justifies the smaller teacher; expect
  an examiner to ask, and answer with D-018 plus the headroom number from
  `scripts/07_measure_gap.py`, not with the vendor quote.
- The world-model forgetting risk is unchanged. "Use it as a policy student" is an
  intended-use statement, not a guarantee that action-only adaptation leaves the
  generation tower intact — and the tower's attention weights are demonstrably shared
  across vision and action tokens (D-029 discussion, `get_gen_seq` = all generating
  tokens). Vendor intent does not remove the mechanism.

**Consequence for the framing:** NVIDIA says Edge can be an automotive policy student;
nobody has published what that costs its world model. The retention pair (D-025) answers
a question the vendor left open, which makes it a contribution rather than hygiene.

## D-031 [VERIFIED] Phase B works, but A1.5 emits the action dims transposed — and greedy decoding of it is a trap

Closes D-022. Ten curated windows, `02a_probe_phaseb.py --clips 10 --sampled`, 2026-08-24.

**The decisive reading: `region_mass` 0.9889** (pass >= 0.5, fail < 0.05). The teacher puts
98.9% of the whole ~155k-vocabulary softmax on its top-32 future bins. The 3000-row future
region is a *trained* target, not allocated-and-never-supervised vocabulary. Measured on raw
logits, so it is independent of every decode and detokenization choice below it.

| reading | value | note |
|---|---|---|
| `region_mass` | 0.9889 | decisive; three orders of magnitude clear of the fail band |
| `argmax_match` | 1.000 | logits/sequence alignment correct |
| floor ADE | 0.01 m | codec and coordinate frame correct |
| sampled ADE | 1.79 m | what `02_label.py` caches |
| expert ADE | 1.34 m | ratio 1.34, gate allows 2.0 |
| greedy ADE | 2.82 m | diagnostic only — NOT what is cached |

### 1. The teacher emits (curvature, accel); its own tokenizer reads (accel, curvature)

The 128-token future stream is 64 waypoints x 2 dims interleaved.
`DiscreteTrajectoryTokenizer.encode` builds it from `UnicycleAccelCurvatureActionSpace` in
(accel, curvature) order and `decode` reshapes identically — mutual inverses, so the
quantization floor round-trips to ~0.01 m **no matter what the model does**. A1.5 emits the
two dims the other way round.

Feeding the emitted stream to `decode` untouched gave ADE 20-128 m; swapping first gave
0.62-5.31 m. Independent of ADE, the per-dim statistics say it outright: on a straight clip
the *emitted* stream's first dim is the one pinned at bin ~1500 (value 0, zero curvature)
while `encode`'s first dim is the one that varies. The dim-major transpose was tested and
ruled out (9.97-77 m).

Nothing in the release round-trips a model-emitted future token through `decode` — the
future-fusion path is stripped, which is what made D-022 open in the first place — so this is
invisible to A1.5's own tests. It is a property of the checkpoint, not a bug in us.

**The half no probe number would have caught:** `gt_traj_token_ids` comes out of `encode`, so
it lands in the OPPOSITE order to `traj_token_ids` and `traj_topk_*`. Cached that way, stage
1's `gt_ce` anchor would train the student toward the per-waypoint transpose of what
`traj_kl` distils — two supervisions fighting, on a term weighted 0.25, low enough to degrade
a run without breaking it. Caught only because the cached-GT field (added 2026-08-23) put
both orders in the same shard where they could be compared.

Fix: `swap_action_dims` in `wrapper.py`, applied at exactly two points — the GT encode in
`label_window`, and `TeacherWrapper.detokenize_traj`, now the only detokenization entry
point. Pinned by `tests/test_traj_dim_order_offline.py`.

### 2. Score the sampled decode. Greedy is a decode artifact, not a teacher property

`02_label.py` samples (top_p 0.98, temperature 0.6); `greedy_traj=True` exists only so a
human can read a waypoint table without stochastic noise. Greedy decoding of a head whose
top-1 sits near 0.5 latches onto one bin and repeats it:

| | greedy | sampled |
|---|---|---|
| median ADE | 2.82 m | **1.79 m** |
| windows with a collapsed curvature dim | 6/10 | **1/10** |

Every window that emitted a single curvature bin under greedy recovered under sampling
(`[1,16]`->`[5,45]`, `[1,15]`->`[13,48]`, `[1,20]`->`[13,46]`). One window still flags:
`170f2756`, one curvature bin against GT's six, on a road straight to within 0.7 m over
141 m — six bins is ~0.001 1/m, a 1000 m radius. Its ADE is longitudinal, from the accel dim,
which is not collapsed.

The probe now measures token health on the sampled stream and gates on it. Gating on greedy
fails a cache that is fine.

### 3. Probe corrections made along the way (thresholds untouched)

`MASS_PASS 0.5`, `ADE_PASS_RATIO 2.0`, `ADE_PASS_SLACK 1.0` are unchanged. What changed is
*which quantity* is measured:

- **`argmax_match` counts ties.** bf16 logits tie exactly between adjacent bins, and
  `torch.topk` and generate's `argmax` break those ties differently. Comparing ids alone read
  0.906-0.953 and cried BROKEN over an alignment that was never wrong: every mismatch sat at
  rank 1 with a top1-minus-emitted log-prob gap of exactly 0.0000.
- **Degeneracy is referenced against GT's own per-dim counts.** The absolute form is
  unreadable on the case it was written for — on a straight road a correct teacher MUST emit
  near-constant curvature.
- **`max_run` is printed, not triggered.** It compares a smooth model against noisy measured
  GT, so any model smoother than its target scores worse. It flagged the window with *more*
  distinct bins than GT and the best ADE of the run.
- **A flat dim needs GT to have variation** before it counts as collapsed, or a stationary
  vehicle (GT `[1, 3]`) false-positives.

### 4. Environment facts this run established

- A1.5's expert mask is built float32 (`alpamayo1_5.py:198`) while the expert runs bf16, and
  torch 2.8's SDPA rejects a bias whose dtype differs from the query's. Every released entry
  point wraps generation in `torch.autocast`, which casts the mask with q/k/v; `label_window`
  now does the same. The expert is forced to sdpa deliberately (`alpamayo1_5.py:103` — "the
  diffusion expert does not support FlashAttention 2"), so this path is unavoidable.
- flash-attn IS required (`config.json` sets `flash_attention_2`) and must match
  cu12/torch2.8/cxx11abiTRUE/cp312.
- `requirements.txt` had open-ended bounds; uv resolved torch 2.13 + transformers 5.x against
  A1.5's hard pins of 2.8.0 / 4.57.1. Now pinned.

### Verdict

The probe returns MARGINAL on 1/10 degenerate windows. Every other criterion passes, and the
one flag is a straight road being described as straight. **Proceeding to `02_label.py --n
500`** — the increment is itself the hedge (nested subset, resumable, additive), so an early
exit stays cheap. D-022 closes.

**Still open:** `run_labeling` is serial, so the GPU idles through every fetch. At ~10 s per
camera and 4 cameras that is ~40 s/window, ~11 h of streaming for the 500 increment. Fits
`main`'s 3-day cap here; does NOT fit for 5000 (~110 h). Parallel per-camera fetch plus a
prefetch queue is a prerequisite for the full run, not for this one.

## D-032 [VERIFIED] The stage-1 consumer side was unrunnable — six defects, plus three more found while fixing them

Found 2026-08-24 while the 500-clip labeling run was in flight, by reading the path the
cache feeds rather than the path that writes it. Nothing here is a teacher-side problem;
the shards being written are fine. All six are fixed, all fixes are pinned by offline
tests, none of it needed a GPU.

The pattern is worth naming: every one of these sits at a seam between two modules that
were written weeks apart and never executed together. The labeler was tested against the
labeler; the losses were tested against the losses.

### The two that would have crashed

**1. Feature KD compared unpooled states against pooled ones.** `label_window` caches
`(feat_pool_len, D_t)` per layer — `torch.chunk(h, 8)` over the teacher's PREFILL — while
`ar_forward` returns `(B, L, D)` per-token states, and nothing reduced them.
`feature_match` would have raised at step 1. Fixed: `layer_map.pool_prompt_segments`
mirrors `torch.chunk` exactly (not an even split — chunk sizes are `ceil(L/n)`, so an
even split pools different token ranges than the cache did) over `[0, n_prompt)`, which
also excludes the right-padding and the teacher-forced answer the teacher never saw.

**2. No `split_*.json` existed.** `coarse_minade` reads `split_challenging.json`,
`05_eval.py` and `07_measure_gap.py` read others, and nothing in the repo wrote any of
them — `07_measure_gap.py` carried a NOTE saying so. The gate runs at the END of epoch 1,
so this surfaces hours into a run. New `data/splits.py` + `scripts/01b_splits.py`, also
called from `01_curate.py`. **Membership is a per-clip hash, not a list position**, so the
splits nest the way the curated increments do (`val_500 ⊂ val_2000`) — splitting by index
would reassign every clip when the increment grows and the data-scaling curve would be
comparing three different validation sets.

`eval.challenging_split: challenging_v1` is aspirational: the PhysicalAI-AV metadata
carries no such flag. `split_challenging.json` is the long-tail STRATA of the val split
(`curation.stratum_of` ≠ "default"), with a documented fallback to the whole val split
when fewer than 5 clips qualify. Config comment updated to say so rather than implying
NVIDIA's list is wired.

### The four that would not have

**3. Training trained on the gate's clips.** `train_stage1` built `Stage1Dataset` over
every shard on disk; the gate evaluated a subset of the same. Now `split_train`.

**4. minADE was computed against action space.** `open_loop.evaluate` scored
`batch["gt_traj"]` — the GT future in the teacher's `UnicycleAccelCurvature` space — with
`min_ade`, whose `[..., :2]` means *x, y in metres*. On `gt_traj` those two slots are
accel and curvature. It runs, it returns a plausible float, and stage 1 early-stops on it.
The reference is `gt_future_xyz`, which the collator did not even pass through; it does
now, and `evaluate` raises rather than falling back.

**5. The gate generated from the teacher-forced context.** `Stage1Dataset` built the full
sequence — teacher CoC *and* the teacher's 128 trajectory bins — and `coarse_minade` then
called `generate_traj_tokens` on it, i.e. asked the student to produce an answer already
present in its prompt. New `for_generation` mode (`prompt.assistant_segments` →
`ContextBuilder.build` → `Stage1Dataset`) stops the context at `<|traj_future_start|>`.
Note the trap it sidesteps: `assemble` skips a `bins` segment whose values are None but
still emits the CLOSING token, so the naive "just pass `traj_bins=None`" prefill ends on
`<|traj_future_end|>`.

**Decision, flagged as a judgment call:** the CoC stays teacher-forced in the epoch gate.
It keeps the gate comparable across epochs, matches the training context exactly, and
keeps the cost at 128 decode steps rather than ~192. The price is that the gate does not
measure the student's own reasoning — the free-running two-phase decode (CoC, then
trajectory) belongs in the final eval. Batch size is 1 there for the same class of reason
`collate_student` right-pads: right-padding is correct for a teacher-forced forward and
wrong for a prefill, since every shorter sample would decode its first token after a run
of pad tokens.

**6. `student.traj_detokenize` was the teacher's raw `tok.decode`** — wrong arity for how
`coarse_minade` called it, and, worse, it bypassed `swap_action_dims`. D-031 says the
swap belongs at exactly two points and that "anything that decodes trajectory tokens by
calling the tokenizer directly is a bug waiting to happen"; this was that. The attribute
is now `_traj_decode` (private) behind `EdgeStudent.detokenize_traj`, the student's mirror
of `TeacherWrapper.detokenize_traj`. **The student inherits the teacher's transposed
emission order because it is trained on the teacher's emitted tokens** — the swap is not
teacher-only.

### Also: the layer map was neither injective nor over the cached layers

`uniform_map` returned one entry per STUDENT layer — 28 of them onto 8 teacher layers —
and `FeatureProjections.forward` keys its output by TEACHER layer. Twenty of the twenty-
eight projections were therefore overwritten in a dict, received no gradient, and which
student layer fed each teacher layer was decided by iteration order. It was also called
with `expert_layers["attended_layers"]`, all 36 layers (the expert attends every one,
D-004), while the cache holds 8.

Both fixed: the map is built over `cfg.teacher.feat_layers`, one student layer per teacher
layer, and `FeatureProjections` now refuses a non-injective map. And
`student.layer_map.mode: cka` — D-008's primary — was dead config: the trainer hardcoded
`uniform_map`. CKA now runs where it can: a few batches of the UNTRAINED student before
step 0, inside `train_stage1._build_layer_map`, since the projections do not exist yet and
the probe is a handful of forward passes. `cka_map` was rewritten to iterate the teacher
layers (the side that must be covered exactly once) and is monotone by construction.
`cka_probe_clips: 2000` — which nothing read — becomes `cka_probe_batches: 4`.

### Three more, found while writing the smoke test

`FeatureProjections` was constructed with `expert_layers["kv_dim"]` as its output
dimension. `probe_expert_conditioning` returns no such key — it returns `kv_heads`,
`head_dim` and `teacher_hidden` — so stage 1 died on a `KeyError` before step 0. Loud,
but also wrong in intent: the projection has to land on the dimension the CACHE was
written at, and the cached features are pooled *hidden states*, not KV. It now reads
D_t off a shard header directly (`_cached_feature_dim`) and warns if that disagrees
with the probed `teacher_hidden`. This is also the cheapest answer to D-019's open
question — the number A1.5's config.json does not carry.

**The text-KD target was sized in the teacher's vocabulary.** `gather_targets` was
called with `n_targets = batch["coc"].shape[1]` — the padded length of the TEACHER's
CoC token ids — while the positions it gathers are the STUDENT's. The two counts differ
by construction: cross-family tokenizers are exactly why the cache stores `coc_text` and
the context re-tokenizes it (D-011). Any student CoC that tokenized longer than the
teacher's had its tail silently dropped from the loss, and the mask then intersected two
different length conventions. Now sized from `coc_pos` itself and masked by `coc_ok`
alone.

Same class, same day: `torch.load` of `traj_tokenizer_spec.pt` needs
`weights_only=False`. The spec pickles the teacher tokenizer's bound `decode` and a
`HistoryTokenize` instance; torch >= 2.6 defaults `weights_only=True` and refuses both,
and `requirements.txt` pins torch 2.8 (D-031). Fixed in both trainers and
`scripts/06_retention.py`.

### Environment consequence to check on the box

`traj_tokenizer_spec.pt` pickles the teacher tokenizer's bound `decode` method (D-029's
`HistoryTokenize` note is the same issue from the other side). Stage 1 runs in the
**cosmos-framework** env and `torch.load`s that file, so **`alpamayo1.5` must be importable
there too**, not only in the labeling env. Cheapest check: `python -c "import
torch;torch.load('<cache>/traj_tokenizer_spec.pt')"` in the training env, before anything
else.

### Not fixed, deliberately

`scripts/05_eval.py` has the same class of defects (shard-only dataset, no student
context, `sample_refined_trajectory` against the still-open `_gen_pathway_forward`). It is
stage-2 infrastructure and is not on the critical path until stage 1 passes its gate.

---

## D-033 [RESOLVED 2026-08-31] Feature KD is inert as configured, and the CKA map it feeds is collapsed — decide after phase 1

**Resolution (run 1 / job 183, see `eval_phase1.md`): feature KD was not load-bearing.**
The epoch gate fell from 10.92 m (untrained baseline, job 199) to 2.35 m at epoch 3 — a
78% reduction — with `feat` at 0.00 for the warmup and ≤1.4% of the loss thereafter. That
is option 1: feature KD is written up as a minor regulariser, and D-008's CKA mapping did
not do the work the plan claimed for it. The map stays as-is for now (`mode: cka`, weight
0.5) because changing it in isolation is the unsafe move described below.

Not fully closed as a *lever*: run 1 also plateaued by epoch 3, so §3 option 2 (raise
`feat` to ~5.0 **and** switch `layer_map.mode` to `uniform`, together) remains the second
thing to try for run 2 if the `stage1.lr` bump does not break the plateau. If run 2 clears
the plateau without touching `feat`, this becomes final.

Original analysis below.

---

## D-033 [was OPEN] Feature KD is inert as configured, and the CKA map it feeds is collapsed — decide after phase 1

Two findings from the first stage-1 smoke run (2026-08-28, Blackwell, micro_batch 4) and
the startup of the first real run (job 183). They are separate defects that happen to
cancel each other out, which is why neither is urgent and why fixing one alone is wrong.

### 1. `feat` contributes ~1.4% of the loss

The smoke test's loss table at step 0:

```
  traj_kl   12.5911   x1.00  = 12.5911
  text_kl    6.9430   x0.50  =  3.4715
  feat       0.3518   x0.00  =  0.0000     <- warmup, not the steady-state weight
  gt_ce     14.5620   x0.25  =  3.6405
```

The `x0.00` is `feat_warmup_frac: 0.1` not having ramped at step 0 — expected, not a bug.
The number that matters is the raw value: at its configured `loss_weights.feat: 0.5` it
settles at **0.176 against traj_kl's 12.59, about 1.4% of the total**. NEXT_STEPS §4 asked
this question in exactly these terms ("whether `feat: 0.5` is a contribution or a rounding
error") and the answer is: a rounding error. Making feature KD a real mechanism needs
roughly a 10x weight increase, to put it on par with `text_kl`.

### 2. The CKA layer map is collapsed onto adjacent student layers

From job 183's startup, the probe on the untrained student:

```
layer map (student -> teacher): {4: 3, 21: 8, 22: 12, 23: 17, 24: 21, 25: 26, 26: 30, 27: 35}
```

Injective and monotone, so it passes the basic check — but seven of the eight student
layers are CONSECUTIVE (21..27) at the top of a 28-layer stack, with one outlier at layer
4. The teacher side is spread evenly across its 36. NEXT_STEPS §6 names this case: "if it
collapses onto adjacent student layers, the mapping is not finding structure and `uniform`
is the honest baseline; set `student.layer_map.mode: uniform` and say so."

### Why this is [OPEN] rather than fixed

The two compound in a way that makes the current run safe but the obvious fix unsafe:
because `feat` is ~1.4% of the loss, the layer map barely influences training, so a
collapsed map costs almost nothing **as configured**. Raise `loss_weights.feat` on its own
and that stops being true — a collapsed CKA map carrying real weight is worse than a
uniform one, because it concentrates the feature-matching signal on seven adjacent layers
near the output rather than distributing it through the stack.

So the two knobs move together or not at all:

* **leave both** — feature KD is a minor regulariser, D-008's CKA mapping is not doing the
  work the plan claims for it, and the writeup says so plainly; or
* **raise `loss_weights.feat` to ~5.0 AND set `student.layer_map.mode: uniform`** — feature
  KD becomes a real mechanism on an honest baseline mapping; or
* **raise the weight and keep `cka`** — only with evidence that the collapse is a property
  of the untrained student rather than of the probe, e.g. by re-running the probe on the
  stage-1 checkpoint and seeing whether the map spreads.

**Decision deferred to the end of phase 1** (user, 2026-08-28): judge it on the epoch-gate
curve from job 183. If coarse minADE improves steadily with feat inert, feature KD was
never load-bearing and option 1 is the honest write-up. If the gate plateaus early, option
2 is the first thing to try before touching `stage1.lr`.

Note when reading that curve: the gate reports "challenging coarse-minADE" but
`split_challenging == split_val` (264 clips, identical sets) because the PhysicalAI-AV
metadata carries none of the fields `stratum_of` reads. It is val minADE, not hard-val.

---

## D-034 [RESOLVED 2026-08-31] The CoC terminator was never supervised — the student cannot end its chain-of-causation

Found by free-running the epoch-3 student's CoC on 8 windows (job 202,
`scripts/05a_inspect_coc.py`; the gate teacher-forces the CoC so it never showed this).
All 8 samples ran to the token cap: the student **never emits `<|cot_end|>`**. It closes
reasoning with a literal `</think>` (token id 13, the base Cosmos3-Edge reasoner's native
delimiter) and then drifts into the appended trajectory-id range with no
`<|traj_future_start|>` structure.

### Why

The student tokenizer has **no** `<|cot_end|>` / `<|traj_future_start|>` id — unlike the
teacher (`teacher/wrapper.py:172` adds them), `extend_trajectory_vocab` appends only
embedding/lm_head rows, not tokenizer entries. So the markers are written as their six /
nine generic subwords (`< | cot _end | >`). And `coc_span` ended *before* the `<|cot_end|>`
subwords (`prompt.py`), so `gather_targets`'s AR-shifted CE (`pos-1 -> input_ids[pos]`)
never had a target inside them. Nothing trained the student to emit the terminator, so
free-running it falls back to the base model's `</think>`.

### Fix (run 2)

New `struct_span` in `prompt.assemble` = the `<|cot_end|><|traj_future_start|>` subword
tokens; `struct_pos` mask in `collate_student`; a `struct_ce` term in `train_stage1.py`
(same `input_ids`-as-target CE as the CoC), `loss_weights.struct_ce: 0.5`. Smoke (job 203)
confirms 60 supervised structural positions/batch (15 per sample × 4), raw CE 4.2 untrained.
`generate_coc_text` stops on a string match for `<|cot_end|>` **or** `</think>` since the
id-level stop is not available.

### Considered and rejected

**Adding the 4009 trajectory/special strings to the student tokenizer** (mirroring the
teacher) would make `<|cot_end|>` a single learnable id and is arguably the "correct"
architecture. Rejected for run 2: `len(tokenizer) == 131072 == old embedding size`, so it
is *feasible* cleanly, but it changes every sequence length and the vocab contract, needs
its own review, and the subword-CE fix is sufficient to teach termination. Revisit if the
subwords prove hard to learn (watch `struct_ce` in the run-2 log and re-run job 202's
inspection).

---

## D-035 [VERIFIED 2026-09-06] The teacher's trajectory KL is worth keeping on curvature and not on accel — and the GT anchor was the wrong SHAPE, not the wrong weight

Runs 2 and 3 both treated `traj_kl` vs `gt_ce` as a weight problem and moved the dial in
opposite directions (run 2 raised `gt_ce` 0.25 -> 0.5, run 3 put it back and annealed it
*down* to 0.06). Neither helped, because neither was the problem. Two cache-only
measurements settle it — no GPU, no model, 400 clips / 102,400 trajectory positions.

### 1. The fight is almost entirely in the accel half of the token stream

The 128-token future is 64 waypoints x 2 action dims, interleaved, in the teacher's
emission order: **even = curvature, odd = accel** (D-031's swap). Re-verified
independently here by correlating the cached bins against `gt_traj`'s columns — token
dim0 vs col1 r = 0.9992, token dim1 vs col0 r = 1.0000 — so D-031 is holding in the
10k-clip cache and this is not the transpose bug recurring.

Teacher probability mass within +/-w bins of the GT bin, over the 3000-bin future region:

| window | curvature (dim0) | accel (dim1) |
|--------|------------------|--------------|
| +/-2   | 0.420            | 0.028        |
| +/-10  | 0.708            | 0.094        |
| +/-20  | 0.806            | 0.170        |
| +/-100 | 0.931            | 0.577        |

The teacher's **path shape is good** and worth distilling. Its **speed profile is close to
uninformative about GT** — even a +/-100-bin tolerance captures only 58% of its mass.
Path shape is far more predictable than a speed profile from a 4-frame context, so this is
the expected shape of the result, not an artifact. Half the stream was fighting `gt_ce`
and the other half was not, and a global weight cannot express that.

### 2. `gt_ce` was flat because a one-hot over 3000 bins is nearly unlearnable

Job 243's raw `gt_ce` sat at ~6.0 nats for all 8 epochs while `traj_kl` fell 4.4 -> 0.38.
Uniform over the region is ln(3000) = 8.01, so the "flat, therefore dominant" reading runs
2-3 acted on was wrong — the term had barely moved off uniform. Nothing in a one-hot CE
knows that bin 1498 and bin 1502 are the same trajectory to within centimetres, so a single
GT sample per context carries no local shape and the model must average many samples to
recover it. For scale, the teacher's own exact-bin NLL on GT is ~7.8 nats, and its
full-sequence `gt_ce` bounds *optimistically* at 12.3 nats — worse than uniform, i.e.
sharply peaked and displaced. Neither model was ever going to drive this term down.

### Fix (run 4)

* **`traj_kl` split by action dim.** `loss_weights.traj_kl: 1.0` on even/curvature
  positions, new `loss_weights.traj_kl_accel: 0.1` on odd/accel. Two `traj_topk_kl` calls
  over complementary masks; logged separately so the halves are attributable.
  Accel is downweighted 10x, not zeroed: it still regularises against the one-sample
  variance of the GT anchor.
* **`losses.gt_traj_soft_ce`** — CE against a Gaussian over the bins neighbouring GT
  (`stage1.gt_soft_sigma_bins: 6.0`, truncated at +/-3 sigma, renormalised at the region
  edges so it cannot leak onto the history bins immediately above). Restores the metric the
  bin index already carries. It does not change the loss *value* separation on a Gaussian
  student — it changes where the gradient goes, from 1 bin in 3000 to 37.
* **`gt_ce` now ramps UP**, 0.5 -> 1.5 (`stage1.gt_ce_end`, was `gt_ce_min` 0.25 -> 0.06;
  the old key is still read). Teacher leads early while the student is still learning the
  token geometry at all; GT is the primary trajectory signal by the end.
* **CoC untouched.** `text_kl: 1.0`, `struct_ce: 0.5`. This is the teacher signal being
  *kept* — its minADE is 1.7 m and not worth inheriting whole, its reasoning is. Do not
  raise them: 243 drove train `text` to 0.003 while val CoC NLL **rose** 0.648 -> 0.921,
  so the CoC is already overfitting and more weight makes that worse.
* **`stage1.epochs: 8 -> 12`**, `--time 2-00:00:00 -> 2-20:00:00`. A ceiling, not a
  target: on full data `early_stop_patience` ends the run, and job 232 early-stopped at
  epoch 6 with its best at epoch 3, so 8 was already not binding. 12 exists because run 4
  reshapes the trajectory supervision and the overfit onset may move later. Budget: 9477
  train clips -> 593 steps/epoch at ~3.8 h/epoch (measured, job 232) -> 45.6 h inside 68 h.
* **LoRA `rank: 96` stays, but the capacity question is still OPEN** - see the correction
  below. Kept because it is the better gate number we have, not because it was shown to
  cause it.

`gt_soft` floors at the target's own entropy, ~= ln(sigma * sqrt(2*pi*e)) = 3.21 nats at
sigma 6 — **not** 0, and not comparable to 243's ~6.0. The old one-hot CE is still computed
under `no_grad` and logged as `gt` for exactly that comparison. `scripts/wandb_tail.py`
now parses the breakdown by term NAME rather than capture-group position, which the two new
terms would otherwise have silently mislabelled.

### Confounded on purpose

Run 4 changes the trajectory supervision *and* the epoch budget at once. That is deliberate
— it is the final phase-1 run, not an ablation — but it means a gain cannot be attributed
between the two without a follow-up. The per-dim and soft-target changes are separable
(`traj_kl_accel: 1.0` and `gt_soft_sigma_bins: 0.01` recover run-3 behaviour) if it matters
later.

### Correction, same day: 232 and 243 are not comparable, and one of them was a 500-clip run

The first version of this entry (commit 8a8ba0e) argued that LoRA rank 96 was validated by
job 243 beating job 232, and justified `epochs: 12` by 243 never early-stopping. Both
claims were wrong, and `sacct` is what exposed it — 243's elapsed time was 3h22m against
232's 26h36m, which is not a rank difference.

| | total steps | steps/epoch | elapsed | h/epoch | outcome |
|---|---|---|---|---|---|
| 232 (r48, attention-only, 19.3M adapter) | 4744 | 593 | 26:36 | 3.8 | early stop ep 6, best ep 3 = 2.463 m |
| 243 (r96, +MLP targets, 99.1M adapter)   | 240  | 30  | 03:22 | 0.4 | hit epoch cap = 2.414 m |

30 steps/epoch x 8 accum x 4 micro = 960 windows ~= 500 clips x 2 — job 243 was the
**500-clip diagnostic**, exactly what the `rank: 96` config comment said it was before that
comment got rewritten into a conclusion. The split files were regenerated 2026-09-05 17:29,
*after* 243 finished, and now hold 9477 train clips (593 steps/epoch, matching 232).

So the two runs differ in three ways — rank, LoRA targets, and a 20x difference in both
data and optimizer steps. 243 reaching a better gate from 1/20th of each is interesting and
worth an actual ablation; it is not evidence for rank 96 specifically. **Run 4 is the first
full-data run at r96 + MLP targets.**

What this does NOT touch, checked rather than assumed:

* The cache measurements above (curvature vs accel mass, teacher NLL on GT) are taken from
  the label cache directly. No training run is involved and they are unaffected.
* **The flat-`gt_ce` finding reproduces on full data.** Job 232's log: `gt` starts at 8.301
  and then bounces 5.1–7.5 for four epochs with no downward trend, while `traj` falls
  7.207 -> 0.368. That is the same pattern 243 showed, on 20x the data. The one-hot target's
  shape is the problem, not the run size.
* **The CoC overfit reproduces on full data.** 232's val CoC NLL bottoms at 0.455 (epoch 2)
  and rises to 0.761 by epoch 6, so holding `text_kl` at 1.0 rather than raising it is
  right on the full split too.

Lesson for the next entry: `sacct -j <ids> --format=Elapsed` before comparing two runs.
Both of these logs record the gate number prominently and the step count only in passing,
which is how a 20x data difference got read as a rank effect.

## D-036 [DECIDED 2026-09-13] Stage 1's failure is in the CoC, and it is two failures: capacity overfit and maneuver mode-collapse

Run 5 (job 313) looked bad on the gate - best saved minADE_6 3.549 m against run 4's
2.232 m - and that reading was wrong three ways. Everything below is from the cache and
from `scripts/09_teacher_ceiling.py` / `scripts/05b_eval_coc.py`, both new.

### 1. minADE is not a stage-1 metric

On the gate's own 60 windows (`runs/teacher_ceiling.json`): codec floor 0.054 m; **teacher
token path 3.767 m ADE_1** (p90 9.376 - heavy-tailed, which is why D-031's 10-window
1.79 m was so far off); teacher action expert 0.998 m minADE_4. The student at 2.937 m
(run 5, epoch 2) is at its target's ceiling, not far above it. Hybrids: teacher curvature
+ GT accel 1.324 m, GT curvature + teacher accel 3.031 m - the accel half carries ~65% of
the token path's error, confirming D-035 in metres. Pushing minADE lower now means
diverging from the teacher.

Also settled: A1.5's action expert **masks the emitted future trajectory tokens out of
its attention** (`alpamayo1_5.py:201-204`, `offset : -n_diffusion_tokens` = -inf; the
expert has no `embed_tokens` at all). It conditions on the KV of prompt + CoC. So the CoC
is the phase-1 deliverable and the trajectory tokens are a side output. NB our
`train_stage2.py:73-84` does the opposite - `ar_forward` over a context that includes the
traj tokens, no mask - which is an open divergence to resolve before phase 2.

### 2. The CoC, free-running and scored (300-500 windows/arm, val and train)

`05b_eval_coc.py` parses maneuver / ego direction / named objects from both traces; the
teacher's CoC is a near-regular language (94.6% causal connective, 95.8% single clause,
99.9% of leading tokens from a 25-word vocabulary; parser assigns a maneuver to 98.7%).

| arm                        | maneuver acc | direction: silent / wrong | obj recall | false-clear |
|----------------------------|-------------:|--------------------------:|-----------:|------------:|
| run-253 4cam train (ep 3)  | 0.787        | 32.1% / 6.2%              | -          | 0.107       |
| run-253 4cam val           | 0.640        | 57.9% / 7.9%              | 0.792      | 0.095       |
| run-253 **1cam** val       | 0.529        | 65.8% / 11.4%             | 0.649      | 0.205       |
| run-313 1cam train (ep 1)  | 0.695        | 48.9% / 15.6%             | 0.773      | 0.105       |
| run-313 1cam val           | 0.580        | 47.4% / 14.5%             | 0.726      | 0.123       |

Termination is 1.000 everywhere (D-034 holds). Two separable failures:

- **Overfit**: maneuver accuracy drops 0.115-0.147 train->val, and run-253's direction
  *silence* goes 32% -> 58%. Lines up with val CoC NLL peaking at epoch 1-2 and with the
  capacity history: best NLL 0.455 at LoRA r48 attention-only (job 232) vs 0.509 / 0.536
  at r96+MLP (253 / 313), 19.3M -> 99.1M adapter params.
- **Mode collapse, no train/val gap**: both students over-emit FOLLOW/KEEP and
  under-emit every committed maneuver; run-313 emits LANE_CHANGE at 11% of the teacher's
  rate (1 vs 9). The "low direction accuracy" is mostly this - the student is *silent* on
  direction ~48% of the time, and when it does commit it is right 70-90%.
- **Camera** (within-model, same weights): 4->1 camera costs object recall 0.792 -> 0.649
  and doubles false-clear 0.095 -> 0.205. Training on 1 camera recovers most of that
  (run-313) but the 1cam-native wrong-side rate is ~2x the 4cam-native (14.9% vs 7.2%
  pooled, p~0.03 on 18 vs 14 events - suggestive, to be tightened at larger n before a
  training slot goes on it).

### 3. Decisions for run 6 (one change per failure, both in `configs/default.yaml`)

1. `student.lora.stage1`: rank 96 -> **48**, alpha 192 -> 96, targets back to attention-only
   - 232's exact setting, reverted together so it is one change. Tests the capacity reading.
2. `stage1.maneuver_sampling: {enabled: true, alpha: 0.5}` (`data/sampling.py`): draw
   probability ~ (1/freq of the teacher's maneuver class)^0.5, class parsed from the cached
   `coc_text`. LANE_CHANGE 2.1% -> 5.1% of draws, FOLLOW 32.9% -> 20.3%. Targets the
   collapse. This is the teacher-label bootstrap D-021 disabled, with a signal that exists.

Judge run 6 on `05b_eval_coc.py`, not the gate: val maneuver accuracy up, the train->val
gap down, LANE_CHANGE / ACCELERATE rates toward the teacher's, false-clear down - and
object precision NOT down (the over-correction signature). `select_on: coc_nll` stays.

### D-036 outcome — run 6 (job 314, commit 32fb7b1, 2026-09-13)

Early-stopped after epoch 4; best = epoch 1 (`run-314/best`, val CoC NLL 0.566). Gate:

| epoch | minADE_6 | val CoC NLL |
|------:|---------:|------------:|
| 0 | 3.764 | 0.568 |
| 1 | 3.616 | **0.566** |
| 2 | 3.347 | 0.624 |
| 3 | 3.254 | 0.688 |
| 4 | 3.476 | 0.703 |

Free-running CoC (`05b_eval_coc.py`, 500 windows/split, 1 camera), best epoch vs run 5's:

| | run-313 ep1 | run-314 ep0 | **run-314 ep1** | teacher |
|---|---:|---:|---:|---:|
| maneuver acc, val | 0.580 | 0.465 | **0.605** | |
| maneuver acc, train | 0.695 | 0.528 | 0.656 | |
| **train - val gap** | 0.115 | 0.063 | **0.051** | |
| token F1 | 0.624 | 0.561 | **0.652** | |
| object recall / precision | 0.726 / 0.818 | 0.767 / 0.728 | 0.758 / 0.819 | |
| false-clear | 0.123 | 0.092 | 0.113 | |
| LANE_CHANGE emitted | 1 | 44 | 4 | 16 |
| FOLLOW / KEEP emitted | 86 / 62 (n=298) | 105 / 73 | 181 / 136 | 145 / 94 |
| direction silent / wrong | 47% / 14% | 46% / 18% | 67% / 8% | |

**Decision 1 (LoRA r48 attention-only): confirmed.** At the same epoch the train->val
maneuver gap halved (0.115 -> 0.051) with val *higher* and train *lower* - the
capacity/data-mismatch signature. Every aggregate beats run 5's best. Keep r48.

**Decision 2 (maneuver sampler, alpha 0.5): did not work, and not for a reason alpha
fixes.** Epoch 0 over-emitted the up-weighted classes 2.5-3.5x (LANE_CHANGE 44 vs 16,
TURN 61 vs 21, YIELD 39 vs 11; precision 0.818 -> 0.728) - the reweighted *marginal*,
learned first. By epoch 1 the conditional had taken over and the model collapsed back
onto FOLLOW/KEEP, further than run 5 (1.25x / 1.45x the teacher's rate), with direction
silence at 67%. Confusions at epoch 1 are all committed -> passive on semantically close
pairs (ADAPT_SPEED->KEEP 19, NUDGE->FOLLOW 15, NUDGE->KEEP 13, LANE_CHANGE->FOLLOW 10):
the student is not seeing the cue that separates them, so the conditional mode is the
generic answer whatever the training marginal. The only arm that ever emitted
LANE_CHANGE near the teacher's rate is run-253 at 4 cameras (50%). The collapse is a
perception limit to test with input (cameras, resolution), not a frequency to reweight.
Set `maneuver_sampling.enabled: false` for run 7; keep the module for a later ablation.

**Also learned:** val CoC NLL turned at epoch 1 and rose faster than in run 5
(+0.058/epoch) while free-running content at epoch 1 was *better* - NLL is token-level
and moves on phrasing; it is not the same thing as content overfit and is a poor
early-stop signal for it. And because only `best/` is saved, epochs 2-4 could not be
free-run scored at all. Run 7 needs maneuver accuracy in the epoch gate (~4 min/epoch at
200 windows) and per-epoch checkpoints, so the selection and the stop see the deliverable.

## D-037 [DECIDED 2026-09-14] The CoC collapse is an under-sharpened conditional; fix it with GRPO on the AR tower, not a better CE

### Why the text loss cannot fix it

`losses.text_kl_or_ce` is plain hard-label CE on ONE sampled teacher trace per window -
Phase A caches no logits (`labeler.py:184-185`), so the "text_kl" the log prints is not a
KL. The trajectory path gets the teacher's top-32 logits and a real KL; the CoC gets a coin
flip per window. Measured on run-314 epoch 1 (`05c_coc_ce_by_position.py`, val, 200 windows):

| teacher's first word | student p on it | n |
|---|---:|---:|
| keep | 0.75 | 85 |
| stop | 0.56 | 20 |
| turn | 0.25 | 9 |
| nudge | 0.18 | 27 |
| adapt | 0.15 | 17 |
| change (lane) | 0.02 | 5 |

The maneuver verb is 19.4% of all CoC cross-entropy at 7.2% of tokens - the heaviest
position the optimiser sees, so "template drowns it out" is not the story. The student holds
0.1-0.25 on the committed verb and the rest on "keep". Decoding the same checkpoint at T=1.0
(`05b --temperature`) brings LANE_CHANGE emission from 0.25x to 1.11x the teacher's rate
while maneuver accuracy falls 0.605 -> 0.538: the mass is there, the conditional is not
sharp. CE learns the expectation of a noisy target and cannot sharpen it.

### What the vendors' recipes say (alpamayo-recipes, cosmos-framework)

- Alpamayo's public SFT supervises `traj_future` only; the CoC is NOT a CE target. The
  model card: "RL post-trained with reasoning reward applied to the chain-of-causation".
  `recipes/alpamayo1_x_rl`: GRPO, `n_generation` 12, T=0.6/top_p 0.98, gated ADE reward,
  joint mode with a pluggable `BaseReasoningGrader`.
- Cosmos3-Edge's own reasoner SFT (`videophy2_sft_edge.py`): FULL-parameter, lr 1e-6,
  weight decay 0.05, SigLIP2 frozen, projector + LM trained. Ours is LoRA on decoder
  attention only; the projector never trains. Untested alternative, fits the Blackwell.
- `loss/cross_entropy.py:weighted_cross_entropy_loss(exponent)`: per-sample vs per-token
  normalisation. Ours is per-token. Minor.

### The teacher on one camera (`09b_teacher_coc_diversity.py`, val, 40 windows, K=8)

Run 5 set `data.cameras` to one camera and the teacher's loader follows it
(`preprocess.py:84`), so this probe is the 4-camera-labelled teacher re-run on ONE camera:
mean majority share 0.806, 27.5% split scenes; where the cached label is a committed
maneuver (n=24) the fresh samples agree 0.48 and say FOLLOW/KEEP 0.40. The 11B teacher,
handed the student's input, exhibits the student's collapse. The 4-camera rerun (queued)
separates the camera effect from sampling diversity. Whatever it says, RL on one camera can
sharpen the conditional only up to what one view supports.

### Decision: phase 1.5, GRPO on the CoC (`train_grpo_coc.py`, `stage1_rl`, `03c_grpo_coc.sh`)

Policy = `run-314/best` merged + fresh zero-init LoRA (r48, attention); reference = the same
weights with `_lora_alpha` set to 0 for the forward (no second model). Rollouts via
`generate_coc_text` at T=0.6/top_p 0.98 - the setting the student is judged at. Reward =
`eval/coc_score` against the cached trace: +1 maneuver, +0.5 direction, +0.5 x object
recall, -1 false-clear, -1 no terminator or >48 tokens. Group-normalised advantages,
zero-variance groups skipped; per-token PG + 0.03 x KL (k3) + 0.1 x the existing CE as an
anchor against reward-hacking the rule grader. lr 1e-5 on the adapters only. Free-running
val check (100 windows) every 25 steps; best-by-maneuver-accuracy saved.

Job 315 (G=8, uniform prompts): 35 s/step, but 69-81% of groups zero-variance - the
documented "group collapse" on easy FOLLOW scenes. Restarted as job 316 with G=16 and
prompts drawn by teacher maneuver class (D-036's weights, `prompt_alpha` 0.5; for GRPO this
is efficiency, not bias - advantages are within-group): skip 0.44 -> 0.31, 85-103 s/step,
~12 h for 500 steps. Step-0 baseline on the 99-window val check: 0.62-0.64 maneuver
accuracy (the +/-0.02 between two runs of the same checkpoint is that check's noise).

Not phase 2: DiffGRPO on the flow head with closed-loop reward is unchanged. This is the
AR tower's CoC, which is what the expert's KV attends.

## D-038 [MEASURED 2026-09-14] The teacher's CoC is an opinion: against the driver's own future it is route-blind, conservative, and its hazard mentions carry no information

All from the cache - `gt_future_xyz` (6.4 s, ego frame, +y = left) against the parsed cached
trace, 19,930 windows - after looking at the frames (user, 2026-09-14): cones the teacher
"nudges" for are beside the road, obstacles it names are visible only in the 30-degree tele
camera, and "turn left/right" at an intersection depends on a goal nobody gave it.

| teacher says | matches driver | opposite | driver went straight |
|---|---:|---:|---:|
| TURN left/right (n=663) | 56% | **17%** | 27% |
| LANE_CHANGE left/right (n=399, offset >2 m) | 43% | **21%** | 36% |

Labelling ran the teacher with no route (`nav_text` never passed). One "turn left" in six,
the car turned right.

| label | stops | slows | holds | speeds up |
|---|---:|---:|---:|---:|
| STOP | 74% | 15% | 5% | 6% |
| SLOW | 4% | 47% | 28% | 21% |
| YIELD | 30% | 12% | 30% | 28% |
| ADAPT_SPEED | 2% | 24% | 51% | 23% |

STOP is grounded; the cautionary classes are not - the driver held or gained speed about
half the time the teacher counselled slowing.

| teacher names a hazard? | n | driver stops / brakes >2.5 m/s / moves >1.5 m within 6.4 s |
|---|---:|---:|
| yes | 13,235 | 70% |
| no | 6,695 | 69% |

Identical: the hazard mentions are scene narration. Two of run 1's reward terms (object recall,
false-clear) were defined by them.

**What run 1 (job 316, teacher-match reward) did with that**, 500 val windows:

| | SFT | RL 75 | RL 400 | teacher |
|---|---:|---:|---:|---:|
| object recall / precision | 0.758 / 0.819 | 0.792 / 0.792 | 0.821 / 0.757 | |
| false-clear (teacher-defined) | 0.113 | 0.071 | 0.027 | |
| GT false-clear (driver braked) | 22% | 15% | 14% | 16% |
| GT-consistent maneuvers | 56% | 57% | 54% | 64% |
| SLOW / YIELD / TURN vs teacher rate | 0.6x/1.1x/1.4x | 1.1x/1.7x/1.7x | 1.4x/1.7x/1.8x | |
| train - val maneuver gap | 0.051 | 0.067 | 0.081 | |

By step 75 the collapse was repaired (FOLLOW/KEEP 1.25x/1.45x -> 1.10x/1.09x, direction
silence 67% -> 56%). After it the run optimised the reward it was given: more hazards named
from a view that cannot see them, less precision, more caution, no more grounding. Not a
failure of GRPO - a wrong objective. The step-75 weights were overwritten by the noisy
99-window selector (only `best/` was kept); every eval now saves its adapters.

## D-039 [DECIDED 2026-09-14] RL run 2 grades the CoC through the trajectory it leads to - the recipe's shape - with the driver's future as the reference

`recipes/alpamayo1_x_rl` (read, not paraphrased): the rollout is the whole completion, CoC
then trajectory tokens; reward = -w_l2 * ADE/3 (+ comfort, + optional Lingo-Judge score vs a
GT reasoning label), and -1 outright when ADE >= 3 m, the CoT is missing, or the judge
fails. `kl_beta = 0.0`, PPO eps 0.2/0.28, lr 2e-6 full-parameter, n_generation 12 at
T=0.6/top_p 0.98, validation on ADE. The reasoning is never graded by rules; it earns credit
through the plan that follows it. Cosmos-RL's advantage normalisation could not be checked
(package not installed); group mean/std assumed.

Run 2 (`train_grpo_coc.py`, `stage1_rl.reward.mode: traj`, job 317):
- `EdgeStudent.generate_coc_and_traj`: one batched decode, per-row state machine (free CoC
  -> forced `<|traj_future_start|>` subwords -> 128 bins restricted to the future rows), so
  G rows with different CoC lengths share one causal cache (no second right-padded prefill).
- reward = -min(ADE, 8)/8 + 0.25 kin + 0.25 dir; -2 for an unterminated CoC or an
  undecodable trajectory. Cap, not gate, and 8 m not 3: the token path lives at 3-5 m
  (teacher's own 3.77 m ADE_1 on the gate windows; SFT single-sample 4.5 m), and a 3 m cap
  saturated the term at -1 for most samples in the smoke.
- scoring layout = CoC + sampled bins; PG and KL per span with separate means (128 trajectory
  tokens must not drown 14 CoC tokens); chunked scoring with grad accumulation.
- route hint `<|route_start|>Turn left ahead<|route_end|>` etc. from `gt_future_xyz`
  (direction only) in every context; the student is told what the teacher was not.
- kept from run 1: LoRA r48 on `run-314/best` as policy and reference, kl_beta 0.03 + 0.1 CE
  anchor (the recipe has neither; we train adapters at 1e-5, not weights at 2e-6), G=16,
  maneuver-weighted prompts. 8 prompts/step, 200 steps, val 200 windows / 25 steps on ADE.
- single front camera, by decision (user, 2026-09-14): the reward pays only for what the
  driver did, so it never asks the student to name what a side camera saw.
- `gt_reward.gt_metrics` (consistency with the driver's speed/path, GT false-clear, ungrounded
  hazards, direction taken) is the teacher-free evaluation; `05b --route-hint --adapters`
  scores any saved step on full n.

Judge run 2 on val ADE (its objective), then on the teacher-free CoC metrics against run 1's
step-75 checkpoint (GT-consistent 57%, GT false-clear 15%) and the teacher's own (64%, 16%).

### D-039 outcome — run 2 (job 317) and the route-hint defect

Run 2 (1 camera, `run-314/best`, 200 steps, 6h53m): on the 200-window val, ADE 5.26 -> 4.54 m
(-14%, monotone after step 25, every rollout decodable), the route hint went from ignored
(0.08) to read half the time (0.50), the CoC followed the trajectory from ~step 100
(FOLLOW 79 -> 49, TURN 7 -> 21), termination 0.995. On the full 500 windows, teacher-free:
GT-consistent 0.73 (SFT 0.70, run 1 0.71, teacher 0.69), direction taken stated correctly
0.60 (SFT 0.08), GT false-clear 0.23 (unchanged - the trajectory reward barely constrains the
words on braking scenes; run 1's text reward did move this, 0.22 -> 0.15). Train - val gap on
teacher agreement 0.036, the smallest of any checkpoint.

**Defect, found by that scoring:** LANE_CHANGE emitted at 3.3x the teacher's rate (TURN
2.4x). The route hint fired "change to the <side> lane" on 33% of val windows - any 2 m
lateral offset 60-100 m ahead, i.e. ordinary road curvature - against the teacher's 3.2%,
and the +0.25 direction term paid the student to echo it. No 6.4 s path detector recovers
the teacher's LANE_CHANGE windows (arc-residual variants: 0/16), and Alpamayo's nav mode
carries turns, not lane changes. Fixed in `31d27fd`: the hint is turn-left / turn-right /
straight, turns read off the final heading (>40 deg; the teacher under-labels turns because
it narrates the approach), a volunteered lane-change direction checked only against the
lateral offset. Val trigger rates now: straight 88%, turn left 6.2%, turn right 5.8%.

Run 2's ADE, route-reading and GT-consistency results stand (the hint does not enter the
ADE term); its maneuver mix and anything about "what the student learned to say" do not.
It is a superseded experiment, not a candidate checkpoint. Run 3 (job 319, 4 cameras) had
trained 2 h on the defective hint and was cancelled; job 320 is the same run on the fix.

## D-040 [MEASURED 2026-09-15] The student's trajectory does not depend on its own chain-of-causation - so a trajectory reward cannot train the reasoning

`scripts/05d_coc_intervention.py`: 40 val windows, 7 forced CoCs each ("turn left" /
"turn right" / "stop" / "accelerate" / "keep lane" / "follow" / the cached teacher trace),
trajectory tokens decoded after each (2 samples, T=0.6), stats of the decoded plan.

| | run-253 @ 4 cam | run-314 @ 1 cam |
|---|---:|---:|
| final heading, "turn left" vs "turn right" | +2.0 vs +4.5 deg | +5.2 vs -0.7 deg |
| end speed, "accelerate" minus "stop" | +1.35 m/s | -0.05 m/s |
| ADE across the 7 CoCs | 3.75 - 4.91 m | 5.69 - 7.28 m |

The per-CoC means are the same plan; the within-window spread (heading sd ~20 deg) is
sampling noise. Stage 1 trained the trajectory path with the teacher's CoC teacher-forced
and the teacher's trajectory as the target regardless of the text, so the student learned
trajectory = f(images). Two parallel heads, not a chain.

Consequences: run 3 (job 320, 4 cameras, ADE reward) moved ADE (3.79 -> 3.44 m by step
125) and did not move the CoC at all (FOLLOW 53 at every check, route read 0.18 flat) -
and could not have: the credit GRPO assigns to CoC tokens through ADE is noise. Run 2's
CoC movement came from the 0.25 kin/dir text terms and the hint echo, not the ADE. The
recipe's reasoning-through-trajectory mechanism presupposes a coupling this SFT student
does not have.

Decision for run 4 (see HANDOFF-run4.md): per-span advantages - an ADE reward applied
to the trajectory tokens, a GT-grounded text reward applied to the CoC tokens - plus a
self-consistency term on both spans (the stated maneuver scored against the kinematics of
the student's OWN decoded plan: teacher-free, GT-free, and the only term that builds the
chain rather than assuming it), and the CoC span sampled at T=1.0 in rollouts so the
sharp 4-camera student's groups contain different reasonings to choose between.

## D-041 [LAUNCHED 2026-09-15] GRPO run 4 = the D-040 design, as job 322

Implemented in `3207af6` exactly as `HANDOFF-run4.md` specifies: `reward.mode:
perspan` (two rewards per joint rollout, `gt_reward.perspan_reward`; each span's
advantage group-normalised on its own, a flat span gets zero advantage, a group
is dropped only when both spans are flat), `self_consistency: 0.5` on both spans
(`gt_reward.self_consistency_term`: kinematic + direction rules against the
student's own decoded plan), `coc_temperature: 1.0` in rollouts only, CoC-span
weights `kin 1.0 / dir 0.5 / hazard 0.5 / teacher 0.25`, `select_on: gt_score`,
init `run-253/best` at 4 cameras, G=16, 8 prompts/step, 200 steps.

Two departures from the handoff, both operational: (1) no Ada smoke - the Ada
held a 45 GB vLLM engine of another user's, so `03c_grpo_coc.sh` now runs the
2-step smoke on the Blackwell under `rl-run-4-perspan-smoke` (own `steps.jsonl`,
via the new `--run-name`) and `set -e` ends the job if it fails; (2) the sbatch
memory floor is 60 GiB (job 320 sat at 58.2 GiB resident) and the wall 32 h.

Queued behind job 320 (run 3, ~3 h from done at submission) and job 321 (user
`vqa`, partition `debug`). Log `logs/a2e-grpo-coc-322.out`; checkpoints
`runs/stage1_rl/rl-run-4-perspan/`. Success criteria are the handoff's: the
05d intervention gaps grow (heading gap from ~3 deg, end-speed gap from ~1 m/s)
AND the CoC mix moves off FOLLOW 53 / TURN ~3, with 500-window val ADE not worse
than run 3's.

**Update 2026-09-15 (later):** job 322 ran the smoke only (passed: peak 53.9 GiB,
both spans rewarded, per-span skip exercised) and was cancelled by the user's
choice before the real run. Run 3 (job 320) was cancelled at step 165 on the
flat curve above. **Run 4 is job 325**, queued behind another user's 3-day job
323 (started 11:24) and a 1 h job 324; log `logs/a2e-grpo-coc-325.out`.

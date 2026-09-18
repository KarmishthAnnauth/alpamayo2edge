# Handoff Brief — alpamayo2edge (for Claude Code session)

**Project:** Distill **Alpamayo 1.5** (11.08B teacher: Cosmos-Reason2-8B backbone, 36 layers,
+ 2.28B flow-matching action expert) into Cosmos 3 Edge (4B MoT student, Nemotron-based, trained
from scratch) for AV edge deployment on a single RTX GPU. Data: 5k curated clips from
nvidia/PhysicalAI-Autonomous-Vehicles, nested increments 500 → 2,000 → 5,000.

**Teacher swapped 2026-08-18** from Alpamayo 2 Super (34B, ~68GB bf16 — did not fit the
available GPU) to Alpamayo 1.5 (22.16GB bf16). Read D-018..D-023 first: the expert is the
same size, the 3000-bin trajectory vocabulary is unchanged, and the whole student side was
untouched. What changed: the API surface, the token *order* (future bins first), the camera
set (4, not 7), the layer count (36, not 64), and meta-action is gone.

**Next action: `NEXT_STEPS.md`** — the stage-1 runbook, written 2026-08-24 while the
500-clip labeling pass was running. Order is splits -> offline tests -> retention
baseline -> `scripts/03a_smoke_stage1.py` (one batch through the whole stage-1 path,
which has never executed) -> the headroom measurement (`COMPARISON.md`) -> stage 1.

**Read D-032 before touching the stage-1 code.** Nine defects on the CONSUMER side of
the cache were found and fixed that day: unpooled features against the cache's pooled
ones, no `split_*.json` written by anything, training on the gate's own clips, minADE
scored against action space, the gate generating from a context that already contained
the teacher's answer, a detokenizer that bypassed D-031's dim swap, a projection built
from a config key that does not exist, a `torch.load` that torch >= 2.6 refuses, and a
text-KD target sized in the teacher's vocabulary instead of the student's. Three would
have crashed; the rest would have produced a number that looked fine.

**Read `DECISIONS.md` first** (append-only log, D-001 … D-032), then
`TRAINING_STRATEGY.md` (why the training regime is LoRA and how the project is framed).
The log holds verified
architecture facts, committed design choices with rationale, and open items. Treat [VERIFIED]
entries as ground truth; do not re-derive them.

**Sibling repos (local clones in the parent directory, since 2026-08-17):**
- `../alpamayo1.5` — full NVlabs/alpamayo1.5 source **plus the checkpoint's `config.json` and
  `model.safetensors.index.json`** (placed there 2026-08-18; they are what D-019 is verified
  against). Teacher-side integration is DONE against it.
- `../alpamayo2` — the previous teacher, kept for reference only. Nothing imports it any more;
  `reference/teacher_expert.py` is redundant with both clones.
- `../physical_ai_av` — the PhysicalAI-AV dataset tooling. `preprocess.py` uses its
  `PhysicalAIAVDatasetInterface` (via `load_physical_aiavdataset`); no raw NCore parsing (D-013).
- `../alpamayo-recipes` — NVlabs/alpamayo-recipes (cloned 2026-08-22): NVIDIA's
  post-training recipes. Carries the **training-time chat template**
  (`src/alpamayo/chat_template/`) that pins the teacher's input format, the SFT
  label-masking helper (`utils/get_label_mask.py`), and the PAI data adapters.
  This is what D-028 is verified against.
- `../cosmos` — NVIDIA/cosmos cookbooks (docs only; the model code is NOT here).
- `../cosmos-framework` — github.com/NVIDIA/cosmos-framework: the actual Cosmos 3 Edge
  model source (unified MoT, diffusion tower, action head). Student-side integration is
  against this (D-015–017). Env: its own `uv sync` setup (see its README), heavier than pip.
- Teacher side: `pip install -e ../alpamayo1.5 -e ../physical_ai_av`.

**Repo state (2026-08-22):** Scaffold changes 1–5 APPLIED, plus the teacher-swap changes 1–5
(end of DECISIONS.md). Teacher side fully re-integrated against Alpamayo 1.5
(`teacher/wrapper.py`: config-only Phase-0 probes + full `label_window`; D-012/D-014/D-019).
**D-024's LoRA rework is now APPLIED (D-026)** — `student/lora.py`, `checkpoint.py`,
`optim.py`, `eval/retention.py`, `tests/test_lora_offline.py` are new; `edge_wrapper.py`,
both trainers and `configs/default.yaml` are updated. `student.freeze` is gone. Wiring it
turned up four bugs that would only have surfaced on the GPU box (unresized `lm_head`, a
1000x timestep error, a cosine schedule that never fired, an impossible stage1->stage2
reload) — all fixed, all written up in D-026.
Student side integrated against cosmos-framework (`student/edge_wrapper.py`): load path
(`Cosmos3OmniModel.from_pretrained_dcp`), tower freeze via the `_moe_gen` parameter-suffix
split (D-015), teacher action space as a new embodiment domain in the DomainAwareLinear
action head (D-017), and the CRITICAL sign/timestep conversion σ = 1−t, v* = −u between
teacher and student rectified-flow conventions (D-016) handled inside `flow_forward`.

**Open items:**
- Stage 1 has never RUN against real weights. `scripts/03a_smoke_stage1.py` is the
  one-batch dry run that settles the remaining VALIDATE-ON-GPU items in one go (image
  processor, D_t, action-domain slot, gradient flow, peak memory).
- **Stage-1 input path is BUILT (D-028/D-029).** The student's context mirrors the
  teacher's exactly — cameras, ego motion, short instruction — pinned against two
  agreeing NVIDIA sources. D-027 is closed: `ar_forward` runs
  `lm.model.reasoner_forward` + `lm_head` with per-layer capture, and
  `generate_traj_tokens` is a masked decode loop. Ego motion turned out to be
  DISCRETE bins (48 of them), not a projection, so the student appends 4009 rows,
  not 3000. `Stage1Dataset` pairs each cached shard with a re-loaded window.
- Retention eval (D-025): the reasoner half is DONE and runnable
  (`scripts/06_retention.py baseline` must run on the UNTRAINED checkpoint, before
  stage 1). The gen tower's denoising loss on generic clips is still unwired — it
  needs a real framework `training_step` data batch. Relative weight drift per tower
  covers it for now.
- ~~`label_window` Phase B is the top risk (D-022).~~ **CLOSED by D-031** (2026-08-24):
  region_mass 0.9889, argmax_match 1.000, sampled ADE 1.79 m against the expert's 1.34 m.
  The future region IS a trained target. Two things came out of it that are permanent:
  A1.5 emits the action dims TRANSPOSED (`swap_action_dims`, two call sites only), and the
  cached decode is SAMPLED — greedy collapses the curvature dim on 6 windows in 10.
- `EdgeStudent._gen_pathway_forward` (# VALIDATE-ON-GPU): wire the packed gen-pathway
  forward against `unified_mot.py`'s und/gen packed-sequence utilities on the GPU box.
- Image processing is the last unverified link in the Stage-1 path: which processor
  the Cosmos3-Edge snapshot ships, and whether its `pixel_values`/`image_grid_thw`
  match `prepare_multimodal_reasoner_inputs`. See `student/context.py`.
- Verify `student.action_domain_id: 31` is an unused embodiment slot in the shipped
  Cosmos3-Edge checkpoint (use the renewed 2026-07-16 checkpoint — earlier ones ship
  untrained action heads and NaN, per cosmos-framework's own config comments).
- Confirm `nvidia/Cosmos-Reason2-8B`'s `num_key_value_heads` on the GPU box — it is inherited
  by the expert and is the one D-019 number not derivable from the two files we have locally.
- Runtime prerequisites: HF auth. **All four gated repos are licence-accepted (confirmed
  2026-08-22): Alpamayo-1.5-10B, Cosmos-Reason2-8B, PhysicalAI-AV, Cosmos3-Edge.**
  **Hardware is 1x RTX 6000 Ada 48GB.** Teacher labeling ~30GB
  at `expert_batch: 4` (headroom to raise it to 8); student training fits only under D-024's
  LoRA regime — full FT needed 43.5GB before activations. Wrapper code is
  written-but-unrun on real weights — validate on 2–3 debug clips first. The torch-only
  parts (LoRA merge math, gradient row-masks) ARE tested:
  `python -m pytest tests/ -q` — 26 + the resume/prefetch/dim-order sets + ~25 added
  with D-032 (splits, layer map, pooling, gate arithmetic), no GPU needed.

**Conventions:** append new decisions to DECISIONS.md as D-033+ with [VERIFIED]/[DECIDED]/[OPEN]
tags. Terminology precision matters: "flow matching" ≠ "diffusion" — the distinction propagates
into supervision-signal design.

**2026-09-19 — phase 1 is one command.** Read `PHASE1_RUNBOOK.md`, then
`bash scripts/run_phase1.sh`. The recipe, the selector (driver-grounded CoC score),
the evaluation (05b + 05f 500-window table) and the pitfalls are all there;
D-043..D-050 hold the evidence.

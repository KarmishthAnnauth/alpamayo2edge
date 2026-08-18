# Handoff Brief — alpamayo2edge (for Claude Code session)

**Project:** Distill Alpamayo 2 Super (34B teacher: 32B Qwen3-VL backbone + 2.3B flow-matching
action expert) into Cosmos 3 Edge (4B MoT student, Nemotron-based, trained from scratch) for
AV edge deployment on a single RTX PRO 6000 96GB. Data: 5k curated clips from
nvidia/PhysicalAI-Autonomous-Vehicles, nested increments 500 → 2,000 → 5,000.

**Read `DECISIONS.md` first.** It is the append-only decision log (D-001 … D-017): verified
architecture facts, committed design choices with rationale, and open items. Treat [VERIFIED]
entries as ground truth; do not re-derive them.

**Sibling repos (local clones in the parent directory, since 2026-08-17):**
- `../alpamayo2` — full NVlabs/alpamayo2 source. Teacher-side integration is DONE against it
  (`reference/teacher_expert.py` is now redundant with `../alpamayo2/src/.../models/expert.py`).
- `../physical_ai_av` — the PhysicalAI-AV dataset tooling. `preprocess.py` uses its
  `PhysicalAIAVDatasetInterface` (via `load_physical_aiavdataset`); no raw NCore parsing (D-013).
- `../cosmos` — NVIDIA/cosmos cookbooks (docs only; the model code is NOT here).
- `../cosmos-framework` — github.com/NVIDIA/cosmos-framework: the actual Cosmos 3 Edge
  model source (unified MoT, diffusion tower, action head). Student-side integration is
  against this (D-015–017). Env: its own `uv sync` setup (see its README), heavier than pip.
- Teacher side: `pip install -e ../alpamayo2 -e ../physical_ai_av`.

**Repo state (2026-08-17):** Scaffold changes 1–5 APPLIED. Teacher side fully integrated
(`teacher/wrapper.py`: config-only Phase-0 probes + full `label_window`; D-012/D-014).
Student side integrated against cosmos-framework (`student/edge_wrapper.py`): load path
(`Cosmos3OmniModel.from_pretrained_dcp`), tower freeze via the `_moe_gen` parameter-suffix
split (D-015), teacher action space as a new embodiment domain in the DomainAwareLinear
action head (D-017), and the CRITICAL sign/timestep conversion σ = 1−t, v* = −u between
teacher and student rectified-flow conventions (D-016) handled inside `flow_forward`.

**Open items (no code blockers left):**
- `EdgeStudent._gen_pathway_forward` (# VALIDATE-ON-GPU): wire the packed gen-pathway
  forward against `unified_mot.py`'s und/gen packed-sequence utilities on the GPU box.
- Stage-1 input assembly: freeze Edge's deployment-context chat template
  [cameras, egomotion, short CoC, traj tokens] and finish `ar_forward` batch plumbing.
- Verify `student.action_domain_id: 31` is an unused embodiment slot in the shipped
  Cosmos3-Edge checkpoint (use the renewed 2026-07-16 checkpoint — earlier ones ship
  untrained action heads and NaN, per cosmos-framework's own config comments).
- Runtime prerequisites: HF auth + accepted licenses (Alpamayo2-Super, PhysicalAI-AV,
  Cosmos3-Edge); the RTX PRO 6000 96GB box for anything past `00_verify.py`. All new
  wrapper code is written-but-unrun — validate on 2–3 debug clips first.

**Conventions:** append new decisions to DECISIONS.md as D-018+ with [VERIFIED]/[DECIDED]/[OPEN]
tags. Terminology precision matters: "flow matching" ≠ "diffusion" — the distinction propagates
into supervision-signal design.

# alpamayo2edge

Distills **Alpamayo 1.5** (11.08B: Cosmos-Reason2-8B reasoner + 2.28B flow-matching
action expert) into **Cosmos 3 Edge** (4B MoT) on a **single GPU** using
**5,000 curated clips** from NVIDIA PhysicalAI-AV.

> **Teacher swapped 2026-08-18.** The original target was Alpamayo 2 Super (34B,
> ~68 GB bf16), which does not fit the available RTX GPU. Alpamayo 1.5 is 22.16 GB
> in bf16 and keeps an expert of the *same* capacity (2.28B vs 2.3B), so Stage 2 is
> unaffected; only the reasoner compression ratio shrinks (32B->4B becomes 8B->4B).
> See DECISIONS.md D-018..D-023.

Companion to `alpamayo2super-to-cosmos3edge-distillation-plan.md` (v2).

## Design decisions encoded here
- **Flow-target caching, not KV caching.** The labeler queries the 2.3B expert
  at K stratified noise levels *while the 32B reasoner's KV is resident*, and
  stores only `(t, a_t, v_teacher)` tuples (~kB/window). Stage 2 is then pure
  supervised regression - no teacher in memory.
- **Discrete trajectory tokens as the primary stage-1 target** (top-k KL),
  copied verbatim from the teacher's tokenizer into Edge's AR vocab.
- **Nested curated increments** (500 -> 2000 -> 5000): deterministic weighted
  sampling means each smaller set is a subset of each larger one, so the
  data-scaling curve is apples-to-apples. Long-tail scenarios oversampled.
- **Coarse minADE as the stage-1 gate**: detokenized discrete tokens, no
  diffusion tower. Fails fast if the plan quality is off.
- **Joint distill + GT losses** in both stages (the Orion-Lite finding), with
  feature-loss warmup (stage 1) and annealing toward GT (stage 2). The
  teacher-supervised vs GT-only comparison is the headline result, not an
  ablation (D-025).
- **LoRA, not full fine-tuning** (D-024): the student was picked for its physical
  prior, so adaptation is low-rank on pretrained weights and full-rank only on
  genuinely new parameters. Task gain and retention are reported as a pair.

## What we tried (details in `DECISIONS.md`)
SFT and RL runs are numbered separately.

**Phase 1: SFT of the reasoning (CoC) + trajectory tokens**

| Run | Change | Outcome |
|---|---|---|
| 1 | Baseline distillation | Gate minADE 10.9 -> 2.35 m; feature KD not load-bearing (D-033) |
| 2-3 | Reweight teacher trajectory KL vs GT | No gain: the wrong knob (D-035) |
| 4 | Teacher KL on curvature only, not accel | Best gate, 2.23 m (D-035) |
| 5 | Single front camera, no GT, select on CoC | minADE is not a stage-1 metric; CoC overfits and collapses to one maneuver (D-036) |
| 6 | Smaller LoRA, maneuver-balanced sampling | Aimed at the collapse (D-036) |
| 7 / 7b / 7c | Teacher CoC + driver trajectory; prefix jitter; prefix masking | Copied the trajectory prefix and ignored frames/CoC; 7c partly fixed it (D-043..D-047) |
| 8 | 7c recipe, selected on a driver-grounded CoC score, lr 1e-4 | Current SFT init (D-051, D-056, D-057) |

Takeaway: SFT can't do better than the teacher's CoC, and the teacher's CoC is route-blind
and often contradicts what the driver actually did (D-038, D-047).

**Phase 1.5: GRPO on the CoC**

| RL run | Reward | Outcome |
|---|---|---|
| 1 | Match the teacher's CoC | Learns the teacher's flaws (D-038) |
| 2-3 | Grade the CoC through its trajectory (NVIDIA's recipe) | ADE moves, CoC doesn't: the trajectory ignores the CoC (D-040) |
| 4-5 | Separate rewards for CoC and trajectory | 4-camera init barely moves; stopped (D-041, D-042) |
| 6 / 6b / 6c | CoC only, scored against the driver's real future | Beat the teacher at step 25, then gamed the reward ("slow down"); penalties made it honest (D-048..D-055) |
| 7 / 7b | 6c recipe from SFT run 8 | Step 25 beats the teacher on braking (best checkpoint), but a rerun got half the gain; need >= 2 seeds (D-058, D-059) |

## Status: teacher side integrated; student side pending the LoRA rework
Read `TRAINING_STRATEGY.md` before touching the student — the training regime
changed on 2026-08-19 from full fine-tuning to LoRA (D-024/D-025) and the code
does **not** reflect that yet.
- `src/distill/teacher/wrapper.py` - IMPLEMENTED against the local `../alpamayo1.5`
  clone (probes + full labeling pass; see DECISIONS.md D-012..D-014, D-019..D-022).
- `src/distill/data/preprocess.py` - IMPLEMENTED via the `../physical_ai_av`
  dataset interface (no raw NCore parsing needed, D-013); 4-camera set (D-023).
- `src/distill/student/edge_wrapper.py` - PARTIAL. Correct against
  `../cosmos-framework` (D-015..D-017): `_moe_gen` tower split, teacher action
  space as a new embodiment domain, teacher<->student flow-convention conversion
  (sigma = 1-t, v* = -u) inside `flow_forward`. Two known gaps:
  (a) `_gen_pathway_forward` - the packed gen-pathway forward, `# VALIDATE-ON-GPU`;
  (b) `_attach_lora` calls `omni.add_lora(...)`, **which does not exist** - the
  whole param-group/LoRA path is being rewritten per D-024
  (`TRAINING_STRATEGY.md` §6).
All wrapper code is written-but-unrun - validate on a few debug clips first.
Target hardware: 1x RTX 6000 Ada 48GB (teacher labeling ~30GB; student training
fits only under D-024's LoRA regime).
Setup: `pip install -r requirements.txt -e ../alpamayo1.5 -e ../physical_ai_av`
(teacher side); cosmos-framework uses its own `uv sync` env (student side).
HF auth with accepted licenses for the gated teacher/student checkpoints + dataset
(Alpamayo-1.5-10B, Cosmos3-Edge, PhysicalAI-AV).
`scripts/00_verify.py` runs the Phase-0 probes config-only (no weight download).

## Order of operations (2026-09-19: see `PHASE1_RUNBOOK.md` for phase 1)
```
bash scripts/run_phase1.sh           # phase 1 SFT: smoke + train, selects on the driver-grounded CoC score
sbatch scripts/03c_grpo_coc.sh       # phase 1.5: CoC-only GRPO from runs/stage1/best (D-049/D-050)
```
Historical sequence:
```
python scripts/00_verify.py          # Phase 0: verify assumptions, write specs
python scripts/01_curate.py --n 500  # first increment
python scripts/02_label.py  --n 500  # teacher labeling (resumable)
bash   scripts/03_train_stage1.sh    # AR tower distillation
bash   scripts/04_train_stage2.sh    # diffusion tower distillation
python scripts/05_eval.py            # minADE on challenging + geo-holdout splits
# then repeat 01/02 with --n 2000, retrain, plot the scaling curve; then 5000
```

## Known simplifications to revisit during integration
- ~~`train_stage2.py` a0-recovery assumes a rectified-flow schedule~~ - VERIFIED
  correct against the teacher's `flow_matching.py` (D-012); labeler caps t at
  0.999 so the recovery stays well-conditioned.
- `collate_stage1` pads token streams jointly; align position/segment ids with
  Edge's actual message format when wiring `ar_forward`.
- ~~Meta-action as an auxiliary head~~ - dropped: Alpamayo 1.5 has no meta-action
  output (D-021).
- **Phase-B trajectory-token emission is unverified on Alpamayo 1.5 (D-022).**
  Validate on 2-3 debug clips before committing to a labeling run: it is the
  primary Stage-1 target and the release strips the future-fusion code path.

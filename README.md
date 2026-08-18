# alpamayo2edge

Distills **Alpamayo 2 Super** (32B Cosmos 3 Super Reasoner + 2.3B flow-matching
action expert) into **Cosmos 3 Edge** (4B MoT) on a **single 96 GB GPU** using
**5,000 curated clips** from NVIDIA PhysicalAI-AV.

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
  feature-loss warmup (stage 1) and annealing toward GT (stage 2).

## Status: both sides integrated; validate on GPU
- `src/distill/teacher/wrapper.py` - IMPLEMENTED against the local `../alpamayo2`
  clone (probes + full labeling pass; see DECISIONS.md D-012..D-014).
- `src/distill/data/preprocess.py` - IMPLEMENTED via the `../physical_ai_av`
  dataset interface (no raw NCore parsing needed, D-013).
- `src/distill/student/edge_wrapper.py` - IMPLEMENTED against the local
  `../cosmos-framework` clone (D-015..D-017): `_moe_gen` tower split, teacher
  action space as a new embodiment domain, teacher<->student flow-convention
  conversion (sigma = 1-t, v* = -u) inside `flow_forward`. One
  `# VALIDATE-ON-GPU` gap: the packed gen-pathway forward.
All wrapper code is written-but-unrun - validate on a few debug clips first.
Setup: `pip install -r requirements.txt -e ../alpamayo2 -e ../physical_ai_av`
(teacher side); cosmos-framework uses its own `uv sync` env (student side).
HF auth with accepted licenses for the gated teacher/student checkpoints + dataset.
`scripts/00_verify.py` runs the Phase-0 probes config-only (no weight download).

## Order of operations
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
- Meta-action is cached but not yet consumed as an auxiliary head - cheap win,
  add once `ar_forward` lands.

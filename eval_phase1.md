# Evaluating phase 1

Running log of stage-1 training attempts, plus the method for judging each one.
Companion to `NEXT_STEPS.md` §6; the deferred D-033 decision is settled here too.
The **Run log** below is the record; sections 0–6 are the method it applies.

---

## Run log

### Run 1 — job 183, commit `015625e` (2026-08-28)

**Baseline** (job 199, commit `ddfd19b`, 2026-08-31): untrained `coarse minADE_6` =
**10.92 m** (p90 23.5, n=60), from `03a_smoke_stage1.py --gate 60` — model untouched,
same 60-window prefix the epoch gate uses. Loss table at step 0:

| term | raw | weight | contribution |
|---|---|---|---|
| traj_kl | 12.55 | 1.00 | 12.55 |
| text_kl | 6.95 | 0.50 | 3.48 |
| feat | 0.35 | 0.00¹ | 0.00 |
| gt_ce | 14.38 | 0.25 | 3.59 |

¹ `feat_warmup_frac` not ramped at step 0; steady-state ≤1.4% of the loss (D-033).
Peak GPU 58.3 GiB @ micro_batch 4, |g| 89.7.

**Curve** (gate: challenging = val, n=60, CoC teacher-forced):

| epoch | minADE_6 | note |
|---|---|---|
| — (untrained) | 10.92 | baseline |
| 0 | 3.284 | −70% vs baseline |
| 1 | 2.760 | |
| 2 | 2.644 | |
| 3 | **2.352** | best → checkpointed |
| 4 | 2.680 | gate turns |
| 5 | 2.779 | early stop (patience 2) |

Best checkpoint: epoch 3, `/data/vla/alpamayo2edge/runs/stage1/best`.
`seff 183`: 9h19m wall, CPU 14.5% of 8 cores, 36 GB / 64 GB RAM.

**Verdict: worked, but plateaus early.** A 78% error reduction vs the untrained
baseline — not the `<~10% better than untrained` underfit row in §2. Two caveats:

- saturates by epoch 3, then the gate rises through epochs 4–5 while training loss
  sits flat at a noisy ~2.0–2.7 → mild overfit to the ~10k-window increment;
- best 2.35 m is *best-of-6*; the teacher's discrete token path scores 1.79 m
  *single-sample* ADE (§1) — a laxer metric still losing to a stricter one, so
  distillation signal is left on the table.

Sits between "worked → stage 2" and "underfit → §3". Proceeding with §3.

**D-033 outcome:** feature KD was **not load-bearing**. `feat` was 0.00 at step 0
and ≤1.4% of the loss thereafter, and the curve improved 78% regardless. Written up
as a minor regulariser; the CKA map (collapsed onto student layers 21–27) did not do
the work D-008 claimed for it. The early plateau keeps §3 option 2 on the table for
run 2 anyway. Recorded under D-033 in `DECISIONS.md`.

**CoC inspection** (job 202, `scripts/05a_inspect_coc.py`, 8 challenging windows,
free-running — nothing teacher-forced). The gate is blind to this (§5); it is an
eyeball, not a metric. Full output: `logs/a2e-inspect-coc-202.out`.

- **Content is well-distilled.** The opening clause tracks the teacher closely
  ("Keep distance to the lead vehicle since it is moving slowly ahead in our
  lane" vs teacher "…since it is slowing ahead"); red lights, cones/construction,
  cut-in vehicles, intersections are all named correctly. One genuine miss
  (window 4: student "proceed straight, light is green" vs teacher "complete a
  left turn because of a clear gap").
- **Control is not.** All 8 samples ran to the 640-token cap — the student
  **never emits `<|cot_end|>`**. It closes reasoning with a literal `</think>`
  (window 8), the *base model's* convention, then drifts ~575 tokens into the
  appended trajectory/special id range with no `<|traj_future_start|>` structure.
- **Root cause is a training gap, not just under-distillation.** `coc_span` ends
  *before* `<|cot_end|>` (`prompt.py:224`), and `gather_targets` supervises
  `pos-1 → input_ids[pos]` only for `pos` inside that span — so the last trained
  CoC target is the last text token, and **nothing ever teaches the student to
  emit `<|cot_end|>` or the CoC→trajectory boundary**. Free-running, it has no
  learned stop.
- Also visible: over-generation — the terse teacher target (~10 tokens) is padded
  with ungrounded specifics ("backed up by slow trucks and stopped cars"),
  consistent with `text_kl: 0.5` being weak against a model that wants paragraphs.

Fixed for run 2: `struct_ce` now supervises the `<|cot_end|><|traj_future_start|>`
subwords (D-034); `text_kl` 0.5 → 1.0. Still owed: the held-out CoC NLL metric
(§4) — the gate cannot see any of this, so re-run `05a_inspect_coc.py` on the
run-2 checkpoint to confirm the CoC now terminates.

### Run 2 — config staged, waiting on the data increment

Changes applied to `configs/default.yaml` + the training path (2026-08-31):

| change | from → to | why |
|---|---|---|
| `stage1.lr` | 1.0e-4 → **3.0e-4** | plateau at epoch 3; LoRA wants 2–5× (§3 #1, D-026). Primary gate-facing lever — the rest are things the gate can't see, so they don't confound the reading. |
| `loss_weights.text_kl` | 0.5 → **1.0** | run-1 CoC over-generated, but content was fine (job 202) — 2×, not more; `struct_ce` is the real brevity lever. |
| `loss_weights.gt_ce` | 0.25 → **0.5** | token path still above the teacher's coarse ADE; harder GT anchor. |
| `loss_weights.struct_ce` | — → **0.5** (new) | CE on the `<|cot_end|><|traj_future_start|>` subword positions. Run 1 never supervised these → the student could not terminate its CoC (§1, D-034). New `struct_span` in `prompt.py` → `struct_pos` in the collator → term in `train_stage1.py`. |
| `stage1.early_stop_patience` | 2 → **3** | run-1 peak was epoch 3/6 on a noisy gate. |

Left as-is: `feat` / `layer_map.mode: cka` (D-033 option 1). Only escalate to
`feat ~5.0` + `mode: uniform` if the lr bump doesn't break the plateau.

**Still open before launch:**
- ~~per-term loss logging + held-out CoC NLL (§4)~~ — done 2026-08-31
  (`train_stage1.py`). Every 20 steps: `loss 8.31 [traj 4.2 text 1.9 struct 0.3
  feat 0.1 gt 1.8]`. Each epoch, next to the gate: `| val CoC NLL 1.83 struct NLL
  0.11`.
- **the data increment.** Run-1's epoch-4/5 degradation is overfit to ~7.2k train
  windows. `tomorrow.txt` has the next labelling batch queued
  (`sbatch_label.sh 5000`, ~+10k windows). Wait for it — more data outweighs any
  weight change here.

Sbatch for run 2: `--cpus-per-task=4 --mem=48G` (run 1 used 14.5% CPU, 36 GB).

### Run 7 — D-043, queued 2026-09-17 (teacher CoC + GT trajectory, coupled)

Runs 2-6 are recorded in `DECISIONS.md` D-035..D-036 (curves: 232 best 2.463 m ep 3;
253 best 2.232 m ep 3; 313 / 314 at 1 camera, `gt_ce` off, 2.9-3.7 m). Run 7 restarts
phase 1 on a different premise - see D-043 for the four measured causes.

| change | from -> to |
|---|---|
| trajectory prefix + target | teacher tokens -> **GT tokens** (`traj_prefix: gt`, `gt_ce` 1.0 flat, `traj_kl*` 0.0) |
| context | + **route hint** (driver's direction) in every context |
| coupling | **image dropout 0.3** on train: frames zeroed, CoC/struct masked, trajectory read from CoC + history + route |
| data | train windows whose CoC flatly contradicts the GT future **dropped** (~18%) |
| cameras | 1 -> **4** |
| selection | `coc_nll` -> **minade**; adapters saved every epoch |
| sampler | maneuver sampling off |

Baseline for the gate stays job 199's untrained 10.92 m (the hint changes the context,
so strictly a new untrained number is owed; the smoke's `--gate 2` is not it). Reference:
run 4's 2.232 m (last GT-anchored run), teacher expert 0.998 m minADE_4 on the same windows.

Judge on three things, in this order: `05d_coc_intervention.py --hint match` (coupling:
heading gap and end-speed gap must open), the gate curve (below 2.23 m), and
`05b_eval_coc.py --route-hint` teacher-free CoC metrics against run 6.

---

## 0. Method — get the baseline. The curve is meaningless without it

*(Done for run 1 — see the Run log. Keep this for every future run.)*

Nothing recorded an untrained coarse minADE at the gate's own settings, and
"2.4 m at epoch 6" says nothing on its own. This needs no new code:

```bash
python scripts/03a_smoke_stage1.py --gate 60
```

`03a_smoke_stage1.py` never calls `opt.step()` — forward, backward, stop — so the
model is untouched, and `--gate 60` runs `coarse_minade` over the same
deterministic 60-window prefix the epoch gate uses (`eval.gate_max_windows`).
The last line it prints is the number every epoch gets compared against.

It also reprints the per-term loss table, which is what §3 needs if the answer
turns out to be "re-run with different weights".

## 1. Then read the curve

```bash
grep "coarse minADE" logs/a2e-stage1-183.out    # the learning curve
grep -E "layer map|LoRA:|early stop" logs/a2e-stage1-183.out
cat /data/vla/alpamayo2edge/runs/stage1/best/distill_meta.json   # which epoch won
seff 183                                        # whether 8 CPUs / 64 G was right
```

Reference points, from D-031 (10 windows, teacher-forced, **single-sample ADE**,
so a soft ceiling — not directly comparable to minADE_6):

| | ADE |
|---|---|
| quantization floor (codec round-trip) | 0.01 m |
| teacher, action expert | 1.34 m |
| teacher, discrete token path — what stage 1 copies | 1.79 m |

## 2. The decision

| curve | reading | action |
|---|---|---|
| drops steadily, best checkpoint at epoch ≥4 | worked | score retention, then stage 2 |
| flattens by epoch 2, early stop at 3–4, <~10% better than untrained | **underfit** | §3, one change at a time |
| never moves; total loss flat in the first 20 steps | broken, not mistuned | check the `LoRA:` and `layer map` lines before touching any knob |
| gate worsens while training loss falls | overfitting ~10k windows | more data increment; weights will not fix it |
| < ~0.5 m | suspect the protocol, not success | verify the gate ran `for_generation=True` and that train/gate splits are disjoint (both were live bugs on 2026-08-24) |

## 3. If it underfit — in this order, one at a time

1. **`stage1.lr: 1.0e-4 → 3e-4`.** Inherited from the full-FT regime; LoRA
   typically wants 2–5x more (D-026, `NEXT_STEPS.md` §6). Cheapest run, most
   likely cause.
2. **`loss_weights.feat: 0.5 → ~5.0` *and* `student.layer_map.mode: cka → uniform`.**
   These move together or not at all (D-033): `feat` is currently ~1.4% of the
   loss, and raising it alone would concentrate real weight on a CKA map that
   collapsed onto seven adjacent layers (21–27 of 28), which is worse than
   uniform.
3. **LoRA capacity** — rank 48 → higher, or path-qualified MLP targets
   (`mlp.up_proj`, `mlp.down_proj`; the bare leaf names are shared with the gen
   tower).

TRAINING_STRATEGY §2's "one real cost" is that a failed gate is ambiguous between
"LoRA too weak" and "the distillation signal is weak". The designed
disambiguation is full FT on the 500-clip increment only
(`student.lora.stage1.enabled: false`) — 43.5 GB of optimizer states before
activations, so it needs the Blackwell, not the Ada.

## 4. Before any second run: the log cannot currently attribute the curve

`train_stage1.py` logs only the **total** loss every 20 steps, so job 183's log
cannot tell you whether `traj_kl` fell while `text_kl` stalled — which is exactly
what a loss-weight decision needs. Two small additions, both worth having before
re-running:

- **per-term loss logging** at the existing `log.info` site;
- **held-out CoC NLL per epoch** — teacher-forced `text_kl` on the val split,
  logged next to the gate. No generation needed, and it closes the hole in §5.

## 5. What the gate does not measure — do not over-read it

- **It is blind to the CoC.** The gate teacher-forces the reasoning text
  (D-033 discussion, `eval/coarse_minade.py`), so the student's own CoC
  contributes nothing to the number. A run where `text_kl` collapsed but the
  trajectory head learned looks identical to a healthy one.
- **It is val, not hard-val.** `split_challenging == split_val` (264 identical
  clips) because the PhysicalAI-AV metadata carries none of the fields
  `stratum_of` reads. The log line says "challenging"; the thesis should not.
- **minADE_6 ≠ ADE.** Best-of-6 is a laxer metric than the single-sample teacher
  numbers in §1. Always quote k alongside it.
- **It is the token path only** — no expert, no diffusion tower. That is the
  point (a bad coarse plan is an input to stage 2, so you want it caught early),
  but it is not the headline number. That one needs the free-running two-phase
  decode in `05_eval.py`, which is still unwritten.

## 6. Record the D-033 outcome either way

The curve is the evidence the feature-KD decision was deferred to (user,
2026-08-28):

- **improves steadily with `feat` inert** → feature KD was never load-bearing;
  write it up as a minor regulariser and say the CKA mapping did not do the work
  the plan claimed for it;
- **plateaus early** → option 2 in §3 is the first thing to try, before `lr`.

Append the answer to `DECISIONS.md` under D-033 when the run is judged.

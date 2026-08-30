# Evaluating phase 1 (job 183)

How to decide whether stage-1 training worked, and what to change if it did not.
Companion to `NEXT_STEPS.md` §6; the deferred D-033 decision is settled here too.

---

## 0. First — get the baseline. The curve is meaningless without it

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

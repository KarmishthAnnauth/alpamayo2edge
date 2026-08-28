# After the labeling run — stage-1 runbook

Written 2026-08-24, while the 500-clip labeling pass was still going. Companion to
`INSTRUCTIONS.md` (the labeling runbook) and `COMPARISON.md` (the headroom
measurement). `DECISIONS.md` is why; this is what to type, in order, and what
each step is allowed to tell you.

**The short version:** nine defects on the consumer side of the cache were found and
fixed today (**D-032**) — three would have crashed, six would have trained or
early-stopped on a number that meant nothing. Stage 1 has still never executed
against real weights, so step 4 is the one that matters.

---

## 0. When the labeling run ends

```bash
python scripts/check_cache.py                  # every shard opens? --fix deletes bad ones
du -sh $CACHE_ROOT
cat $CACHE_ROOT/failed_windows.json 2>/dev/null | head -40
ls $CACHE_ROOT/manifest.json                   # only written on a completed run
```

A missing `manifest.json` means the run stopped early — re-run `02_label.py --n 500`,
it resumes for free. Scattered failures are the network; a solid run of them tripped
the circuit breaker and is systemic.

**Record in `DECISIONS.md` as D-033:** throughput (win/s from the log), wall clock,
final cache size, how many windows failed and why. `INSTRUCTIONS.md` asked for this
under D-031; D-031 is already written, so the labeling numbers get their own entry.

---

## 1. Splits — 30 seconds, no GPU, and stage 1 cannot start without them

```bash
python scripts/01b_splits.py --n 500
```

Writes `split_train.json`, `split_val.json`, `split_challenging.json`. Until today
nothing wrote them and three consumers read them; the gate that reads
`split_challenging.json` runs at the **end of epoch 1**.

Check the printed counts. At `val_fraction: 0.05` and 500 clips that is ~25 val
clips (~50 windows). If `split_challenging` came back equal to `split_val`, fewer
than 5 long-tail clips landed in val and the fallback fired — fine to proceed, but
the gate is then "val", not "hard val", and the thesis should say so.

## 2. Offline tests and an environment check — still no GPU

```bash
python -m pytest tests/ -q          # 26 + 5 resume + prefetch + ~25 new today
python -c "import torch; torch.load('$CACHE_ROOT/traj_tokenizer_spec.pt', weights_only=False)"
```

Run that second line **in the cosmos-framework env**, not the labeling env. The spec
pickles the teacher tokenizer's bound `decode` method, so `alpamayo1.5` has to be
importable where stage 1 runs:

```bash
pip install -e ../alpamayo1.5           # in the cosmos env, if it fails
```

## 3. Retention baseline — before any training touches the checkpoint

```bash
python scripts/06_retention.py baseline
```

Writes `runs/retention_base`. This is half the headline result (D-025), it must be
measured on the **untrained** Edge checkpoint, and it is unrecoverable afterwards.
The gen-tower denoising half is still unwired; relative weight drift covers it.

## 4. THE STEP THAT MATTERS — one batch through stage 1

```bash
python scripts/03a_smoke_stage1.py                # forward + backward, 1 batch
python scripts/03a_smoke_stage1.py --gate 2       # ...plus 2 windows of the epoch gate
```

Everything between the cache and the optimizer step is written-but-unrun: the image
processor, `prepare_multimodal_reasoner_inputs`, the extended vocabulary,
`reasoner_forward` with per-layer capture, four losses, a LoRA backward. This runs
all of it once and prints every shape.

Read four things off it:

| line | what it settles |
|---|---|
| `pixel_values` / `image_grid_thw` | which processor the snapshot ships and whether the reasoner accepts it (D-029's last VALIDATE-ON-GPU item) |
| `D_t=` | the teacher hidden size the cache was written at — A1.5's `config.json` does not carry it (D-019). 4096 makes `INSTRUCTIONS.md`'s disk table hold |
| the four loss rows | whether `feat: 0.5` is a contribution or a rounding error next to `traj_kl: 1.0`. Re-weight here, not after six epochs |
| `peak GPU` | whether `micro_batch` can go above 4 |

`--gate 2` additionally reports the **untrained** coarse minADE. That is the number
epoch 1 gets compared against, and it is worth having: an untrained student that
already scores well means the gate is not measuring what you think.

If it fails, it fails in the first minute and the traceback names the seam.

## 5. Headroom — teacher vs zero-shot student (`COMPARISON.md`)

```bash
python scripts/07_measure_gap.py calibrate --clips 5     # GATE: <1 m or stop
python scripts/07_measure_gap.py teacher --clips 50 --samples 6
python scripts/07_measure_gap.py edge    --clips 50 --samples 6
python scripts/07_measure_gap.py report
```

Needs no cache and runs on raw windows, which is why it can come after step 4
without blocking it. It answers "how much room is there between the specialist and
the zero-shot generalist" — a narrow gap is a **design** signal (raise LoRA rank, add
MLP targets, more data), and you want it before the GPU weeks, not after. It is also
the number that answers an examiner's "the teacher is only 8B" question (D-030).

Note the env split: `teacher` wants the alpamayo env, `calibrate`/`edge` the cosmos
env, `report` either.

## 6. Stage 1

```bash
sbatch scripts/03_train_stage1.sh          # Blackwell, via SLURM
squeue -u $USER                            # then follow slurm-a2e-stage1-<jobid>.out
```

What to watch, in the order it appears:

- **`layer map (student -> teacher)`** — the CKA probe now actually runs (it was dead
  config until today, D-032). Eight pairs, injective, monotone. If it collapses onto
  adjacent student layers, the mapping is not finding structure and `uniform` is the
  honest baseline; set `student.layer_map.mode: uniform` and say so.
- **The first 20 steps.** `traj_kl` should move. If nothing moves, the LR inherited
  from the full-FT regime is the first suspect — LoRA usually wants 2-5x more
  (D-026), and that is `stage1.lr`.
- **The epoch gate.** ~60 windows at batch 1, `minade_k` modes each, so minutes.
  Compare against the untrained number from step 4.
- **Early stop** is 2 evals without improvement.

### If the gate does not improve

TRAINING_STRATEGY §2's "one real cost": a failed gate is ambiguous between "LoRA is
too weak" and "the distillation signal is weak". The disambiguation is already
designed — full fine-tuning on the 500-clip increment only, `student.lora.stage1.
enabled: false`, run on a bigger box (43.5 GB of optimizer states before
activations, so not this one).

---

## Still open, and knowingly not addressed today

- **`scripts/05_eval.py` does not run.** Shard-only dataset with no student context,
  and it calls `sample_refined_trajectory` through the still-open
  `_gen_pathway_forward`. Stage-2 infrastructure; annotated in place.
- **The final eval should free-run the CoC.** The epoch gate teacher-forces it
  deliberately (comparable across epochs, matches the training context, 128 decode
  steps instead of ~192). The two-phase decode — text to `<|cot_end|>`, then the
  restricted trajectory decode — is written nowhere yet, and the headline number
  should not come from the gate.
- **`holdout_geo` has no shards.** `curation.curate` drops JPN/ZAF before the teacher
  sees them, by design. A minADE number there needs GT and student frames only, no
  teacher — a cheap input-only pass, unwritten.
- **`_gen_pathway_forward`** (stage 2's packed forward) is unchanged and still
  VALIDATE-ON-GPU.
- **Camera ordering is correct by luck** — `[0,1,2,6]` is already ascending
  (`INSTRUCTIONS.md`, known rough edges). Reordering `data.cameras` breaks it silently.

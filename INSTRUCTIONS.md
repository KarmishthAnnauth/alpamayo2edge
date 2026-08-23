# Labeling run — GPU runbook

Written 2026-08-23 for the first real teacher pass. Companion to `COMPARISON.md`
(the headroom measurement, which needs no cache and can run first). Read
`DECISIONS.md` for *why* any of this is shaped the way it is; this file is only
*what to type* and *what the numbers mean*.

**Hardware:** 1x RTX 6000 Ada 48GB. Teacher labeling sits around 30GB at
`expert_batch: 4`.

---

## What changed for this run (2026-08-23)

**Resume is now free.** `run_labeling` decides whether it needs a window before
loading it, rather than streaming and decoding every window on the way to skipping
the ones already cached. Matters for the increments (§2) and for restarting an
interrupted run. Five new tests in `tests/test_labeler_resume_offline.py`.

**Ground-truth trajectory tokens are now cached.** `stage1.loss_weights.gt_ce`
previously read as an anchor against teacher error — `losses.gt_traj_ce` is
documented as *"Anchor CE on ground-truth trajectory tokens (guards against teacher
errors)"* — but `train_stage1.py` fed it `traj_tgt`, which `gather_targets` reads
back out of `input_ids`, and those carry the **teacher's** emitted tokens. It was a
hard-label restatement of the target `traj_topk_kl` already distills softly. Nothing
in the pipeline disagreed with the teacher.

Fixed across five files: `label_window` now runs the GT future through the teacher's
own future tokenizer and returns `gt_traj_token_ids`; `save_shard` persists it;
`collate_stage1` pads it as `gt_traj_tok`; `train_stage1` feeds *that* to `gt_ce`
with the same appended-row offset the top-k indices get. It is free at label time —
pure arithmetic on tensors already in memory — and it could **only** be captured
during this pass. `gt_ce: 0.25` now means what it says.

The probe reads the cached field rather than recomputing it, so its quantization
floor now also validates exactly what stage 1 will train on.

---

## 0. Preflight

```bash
cd ~/…/alpamayo2edge
pip install -e ../alpamayo1.5 -e ../physical_ai_av
pip install -r requirements.txt
huggingface-cli whoami          # all four gated repos accepted 2026-08-22
nvidia-smi                      # expect ~48GB free
```

Point the three paths in `configs/default.yaml` at real directories — they are
`/data/...` placeholders today:

```yaml
paths:
  dataset_root: …   # PhysicalAI-AV clips (streamed; this is the HF local_dir)
  cache_root:   …   # shards land here — budget below
  runs_root:    …   # probe output, later checkpoints
```

Disk budget, computed from the config rather than guessed:

| | per window | 500 clips (1k windows) | 5 000 clips (10k windows) |
|---|---|---|---|
| teacher targets `NN.npz` | ~550 KB | ~0.6 GB | ~5.5 GB |
| student frames `NN_input.npz` | ~700 KB | ~0.7 GB | ~7 GB |

The targets shard is **dominated by feature KD**: 8 layers x 8 pooled segments x
D_t at fp16 is ~512 KB of the 550. D_t is Cosmos-Reason2-8B's hidden size, which
A1.5's `config.json` does not carry (D-019) — if it is 4096 the table holds; confirm
on the box. Dropping `teacher.feat_layers` to 4 entries roughly halves the cache.

Then, offline, no GPU needed:

```bash
python -m pytest tests/ -q      # 26 + 5 new resume tests; catches a broken env fast
```

---

## 1. Verify the token geometry

```bash
python scripts/00_verify.py
```

Config-only — no 22GB download, but it does need HF auth for the checkpoint config
and the Cosmos-Reason2-8B tokenizer. It writes `traj_tokenizer_spec.pt`,
`expert_layers.pt` and `vocab_report.json` into `cache_root`, and asserts the
geometry the whole pipeline is built on: 3000 future bins x 128 tokens, 1000 history
bins x 48 slots, the two regions tiling `traj_vocab_size`.

**If an assert fires, stop.** It means the checkpoint moved under us and the
student's appended vocabulary no longer lines up. Nothing downstream is salvageable
until that is reconciled.

Expect `vocab_ok: false` in the vocab report. That is not a failure — it is D-011
confirming the teacher and student tokenizers are different families, which is why
CoC distillation is sequence-level on re-tokenized text.

---

## 2. Curate the increment

```bash
python scripts/01_curate.py --n 500
```

Writes `cache_root/curated_500.json`. Metadata-only stratification (the
meta-action bootstrap died with the teacher swap, D-021), holdout countries JPN/ZAF
excluded, deterministic per `(seed, clip_id)`.

### Why 500 and not 5 000 — the increments are not a smaller dataset

**5 000 is the target** (`data.num_clips: 5000`). The increments are *nested subsets
of that same 5 000*: `curated_500 ⊂ curated_2000 ⊂ curated_5000`, guaranteed by the
deterministic per-`(seed, clip_id)` ranking in `curation.py`. And labeling is
resumable and additive — `02_label.py --n 2000` re-reads the larger list and skips
every shard already on disk. So **500 → 2 000 → 5 000 costs the same total teacher
compute as going straight to 5 000.** You are not paying twice for the first 500.

What the increments buy, all three already designed into the project:

1. **The data-scaling curve** — README lists it as an output, and the nesting is what
   makes it apples-to-apples rather than three unrelated samples.
2. **The full-FT ablation slot.** TRAINING_STRATEGY §2 runs full fine-tuning on the
   **500-clip increment only**, to disambiguate a failed stage-1 gate ("LoRA too
   weak" vs "the distillation signal is weak"). That ablation needs a 500-clip cache
   to exist as its own point.
3. **An early exit** if the labeling turns out wrong.

This is true in *data streaming* as well as teacher compute, as of 2026-08-23:
`run_labeling` now decides whether it needs a window **before** loading it. It used
to re-stream and re-decode every already-labeled window on the way to skipping it,
which made each increment re-download its predecessor's clips. Pinned by
`tests/test_labeler_resume_offline.py`.

Separately from all of that: my advice to run 500 **tomorrow** is operational, not
scientific. It is how you find out the real throughput and the real shard size before
committing to an overnight run. If the probe passes and the first 500 look healthy,
going straight to `--n 5000` is a perfectly reasonable next command.

No GPU for this step, but it touches the dataset index, so it is also your first real
check that `dataset_root` and HF auth work.

---

## 3. THE GATE — D-022 probe

```bash
python scripts/02a_probe_phaseb.py --clips 3
```

**Do not skip this, and do not start labeling if it fails.** ~30GB GPU, three
teacher passes, a few minutes. Full reasoning in §"What we're measuring" below.

Exit codes: `0` PASS → go. `1` MARGINAL → widen to ~10 clips and decide
deliberately. `3` FAIL → take D-022's fallback, do not label. `4` BROKEN → a bug in
our plumbing, not the teacher; fix before reading anything else.

Writes `<runs_root>/phaseb_probe/probe.json` plus one npz per window (`gt`,
`discrete`, `floor`, `expert` trajectories) so you can plot them if the verdict is
ambiguous.

---

## 4. Label

```bash
python scripts/02_label.py --n 500 2>&1 | tee ~/label_500.log
```

Resumable — existing shards are skipped, so an interrupted run is just re-run. Logs
`labeled N (skipped M) | X win/s | eta H h` every 50 windows.

**Start with 500** — see §"Why 500 and not 5 000" above. The first 500 tell you the
real throughput, the real per-shard size, and whether hour 6 looks like hour 1; the
work is then reused, not redone, when you go wider.

**Throughput is genuinely unmeasured.** Per window: a 16-image prefill, ~64 CoC
decode steps, 128 trajectory decode steps, then ~13 expert forwards at batch 4. The
ETA the script prints after the first 50 windows is your only real estimate — treat
any number I gave you before that as noise.

---

## What we're actually measuring tomorrow

Two activities, different in kind. Worth keeping separate in your head.

### The probe is a measurement. The labeling run is manufacturing.

Only step 3 has a result. Steps 1, 2 and 4 either work or fail loudly; they produce
artifacts, they don't produce findings. So the honest answer to "what are we
measuring tomorrow" is: **one binary question, with four independent readings on
it.**

### The question: is the teacher's discrete trajectory vocabulary a *trained* target?

The whole stage-1 design assumes Alpamayo 1.5 can emit discrete future-trajectory
tokens that mean something. The evidence *for* is circumstantial but real: the
checkpoint allocates a full 3000-bin future tokenizer, defines
`<|traj_future_start|>` / `<|traj_future_end|>`, and ships a parser
(`token_utils.extract_traj_tokens`) for exactly that span. The evidence *against* is
that A1.5's release is inference-only and its fusion path
(`TrajectoryFusionMixin.fuse_traj_tokens`) fuses **history only** — the future half
was stripped. For the previous teacher, A2 Super, D-014 rested on a training loss
that proved the target was supervised. That proof is gone.

So it is possible that those 3000 rows were allocated and never trained, and that
our restricted decode is sampling structured noise. `label_window` will emit 128
bin ids either way and nothing downstream would notice.

Four readings, weakest to strongest:

**`region_mass` — the decisive one.** The probability mass the teacher puts on its
top-32 future bins, out of the entire ~155k vocabulary. This falls out for free
because `traj_topk_logp` caches *full-softmax* log-probs, deliberately not
renormalized (`losses.traj_topk_kl` derives its tail bucket from that). So
`exp(logp).sum(-1)` is directly readable. A trained head concentrates there and
reads ~0.9. An untrained one leaves the mass on text tokens and reads ~1e-3. This
is three orders of magnitude, not a judgment call — and it is measured on raw
logits, so it is independent of every decode and detokenization choice downstream.

**ADE, against two references.** Absolute ADE means nothing here; the ratios do.
The *expert Euler rollout* is the trusted released path — the teacher's own accuracy
on this window, the number the discrete head has to approach. The *quantization
floor* is GT encoded to bins and decoded straight back: the best any 128-token
sequence could possibly achieve, and therefore the floor under the discrete head.
The floor doubles as a frame check — if it comes back large, the detokenized
trajectory is living in a different coordinate frame than `ego_future_xyz`, and you
want to know that before blaming Phase B for it.

**Per-dim token health.** The failure ADE can miss. The 128 tokens interleave accel
and curvature (64 waypoints x 2 dims), so a curvature dim pinned to one bin decodes
to a perfectly smooth, entirely straight line — which ADE scores as merely mediocre
on a straight road. Distinct bins *per dim* catches it; the same statistic on the
interleaved stream does not.

**`argmax_match` — this one tests us, not the teacher.** In greedy mode the emitted
token must equal `traj_topk_idx[:, 0]`. If it doesn't, there is an off-by-one
between the `out_b.logits` steps and the `out_b.sequences` slice, which would
misalign every KD target in the cache against its own token — silently, and in a way
that would show up months later as stage 1 simply not converging. A failure here
invalidates the other three readings.

### If the probe says FAIL

D-022's fallback, unchanged: stage 1 loses its primary target and reduces to
sequence-level CoC KD (D-011) plus a continuous trajectory target, and the
coarse-minADE gate runs off the expert rather than off detokenized tokens. That is
a smaller thesis, not a dead one — but it is a design change, and it is much cheaper
to make it tomorrow morning than after a week of labeling.

### What the labeling run produces (and which loss eats it)

Not measurements, but worth knowing what you are paying GPU-hours for. Per window,
one `NN.npz`:

| field | consumer | why it exists |
|---|---|---|
| `traj_topk_idx`, `traj_topk_logp` | `traj_topk_kl` → stage1 `traj_kl: 1.0` | the primary signal; K+1 buckets with the tail absorbing out-of-region mass |
| `coc_text` | `text_kl_or_ce` → stage1 `text_kl: 0.5` | re-tokenized with the Edge tokenizer — cross-family vocabs (D-011) |
| `feat_{3,8,…,35}` | `feature_match` → stage1 `feat: 0.5` | 8 layers x 8 pooled segments, mapped by CKA (D-008) |
| `gt_traj_token_ids` | `gt_traj_ce` → stage1 `gt_ce: 0.25` | GT future through the teacher's own tokenizer — the independent anchor, new today |
| `traj_token_ids` | teacher-forced into the student's context | the sequence the top-k targets are aligned to |
| `flow_t`, `flow_a_t`, `flow_v` | `flow_distill` → stage2 `flow_distill: 1.0` | 12 stratified `(t, x_t, u_teacher)` tuples; **caching these instead of KV is what makes single-GPU stage 2 pure supervised regression** (D-007/D-012) |
| `gt_traj` | `flow_matching_gt` → stage2 `fm_gt: 0.5` | GT in action space; stage 2 anneals toward it over the last 20% |
| `traj_samples`, `gt_future_xyz` | sanity / eval | — |

Alongside each, an `NN_input.npz`: 16 JPEG frames plus ego history, the student's own
view of the same window. Written during labeling because the pass already streamed
the clip, and without it stage-1 training re-streams every clip once per epoch, six
epochs deep. **Watch the color channels** — `frames.py` flips RGB↔BGR on both sides
of `cv2.imencode`, and `test_frames_offline.py` pins the round trip, because a silent
swap trains the student on blue cars and costs a full run to notice.

---

## While it runs

- **First 50 windows.** The log line gives you win/s and an ETA. Sanity-check the
  ETA against how long you are willing to wait before walking away.
- **`nvidia-smi`.** If it sits well under 30GB, raise `teacher.expert_batch` from 4
  toward 8 — the flow targets and Euler rollouts both batch against it. Restart is
  free; the run is resumable.
- **Shard size.** `du -sh cache_root` after ~100 windows, extrapolate, compare to the
  table above. A large surprise means the feature cache is bigger than assumed.
- **A few `coc_text` values.** `python -c "import numpy as np;
  print(np.load('…/00.npz')['coc_text'])"`. Should read like driving commentary. If
  it is empty or degenerate on every window, Phase A is misfeeding the model and the
  `text_kl` term is worthless — worth catching in hour one rather than hour eight.

## Known rough edges

**Camera ordering is correct by luck.** `preprocess.py:89` names student frames by
*config* order; the loader actually returns them sorted by camera index
(`image_frames[argsort(camera_indices)]`). Your four cameras are indices `[0,1,2,6]`,
already ascending, so it works. Reorder `data.cameras` in the config and every
student frame gets silently mislabeled by camera.

**`kv.batch_repeat_interleave`** (`wrapper.py:427`) is the one API I could not verify
offline against the transformers version A1.5 pins (4.57.1) — it has been
deprecated-then-moved before. It would fail on window one, so you will know
immediately.

## What to record

Append to `DECISIONS.md` as **D-031**: the probe verdict and its four numbers
(region_mass, discrete/expert/floor ADE), the clip ids used, and the labeling run's
throughput and final cache size. D-022 closes on a PASS — mark it, since it has been
the top risk since the teacher swap.

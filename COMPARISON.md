# Headroom measurement — GPU runbook

**Goal:** one number. How far apart are the teacher and the untrained student on the
same driving clips, same metric?

    Alpamayo 1.5 (AV specialist)          ADE  ->  the ceiling
    Cosmos 3 Edge zero-shot, domain "av"  ADE  ->  the floor

Everything stage 1 and stage 2 can achieve lives inside that gap. Measure it **before**
the labeling run and before any training: a narrow gap is a design signal (raise LoRA
rank, add MLP targets, more data, reconsider the teacher), not a tuning signal, and you
want it now rather than six epochs in.

Background: D-030 (the pairing is vendor-endorsed, no recipe exists), D-029 (Cosmos 3
ships a trained `av` embodiment at domain id 1).

---

## 0. Before you start

**Two environments, and they are not the same one.** The phases run as separate
processes precisely so this is fine:

| Phase | Needs |
|---|---|
| `teacher` | `pip install -e ../alpamayo1.5 -e ../physical_ai_av` |
| `calibrate`, `edge` | the cosmos-framework env (`uv sync` per its README) |
| `report` | numpy only — either env |

**Also required:**
- HF auth, with the four gated repos accepted (done, 2026-08-22).
- `configs/default.yaml` → `paths.dataset_root`, `paths.cache_root`, `paths.runs_root`
  pointing at real directories on the box.
- The PhysicalAI-AV clips reachable (this streams them; nothing is cached yet).
- **Use the renewed 2026-07-16 Cosmos3-Edge checkpoint.** Earlier ones ship untrained
  action heads that produce NaN — the framework's own config comments say so, and this
  measurement is entirely an action-head measurement.

Nothing from `02_label.py` is needed. This runs on raw windows by design, so it can
happen first.

---

## 1. GATE — calibrate the action decode

```bash
python scripts/07_measure_gap.py calibrate --clips 5
```

**Do not skip this.** Cosmos 3 ships the `av` embodiment (domain 1, 9-D actions) but no
AV action dataset, so the 9-D layout is *inferred* by analogy with every other pose
embodiment in the framework (`camera_pose` is also 9-D; DROID/UMI are
`[Pos(3), Rot6d(6), Gripper]`). It is marked `[ASSUMED]` at
`distill/eval/gap.py::av_actions_to_xyz`.

It is checkable: inverse dynamics recovers the ego motion that produced video you
already have, so the answer is known — the window's own past.

| Result | Meaning | Action |
|---|---|---|
| **< ~1 m** mean ADE | decode is sane | go to step 2 |
| 1–2 m | marginal | eyeball a couple of windows before trusting anything |
| **> 2 m** | decode is wrong | stop; see Troubleshooting |

Writes `<runs_root>/gap/calibration.json`. `report` warns loudly if it is missing.

---

## 2. The two numbers

Separate processes, so 22 GB and 9 GB never coexist. Start small to prove the path
runs end to end, then scale up.

```bash
# smoke test first — 5 clips, ~10 windows
python scripts/07_measure_gap.py teacher --clips 5
python scripts/07_measure_gap.py edge    --clips 5

# then the real thing
python scripts/07_measure_gap.py teacher --clips 50 --samples 6
python scripts/07_measure_gap.py edge    --clips 50
```

**Both phases must score the same windows.** `report` refuses to compare mismatched key
sets — a gap measured on different clips is not a gap. Since no `split_*.json` exists
yet (`01_curate.py` writes `curated_<n>.json`, not splits), the script falls back to the
raw clip index and takes the first `--clips`. That is deterministic, so both phases
agree. If you want it pinned explicitly:

```bash
python scripts/07_measure_gap.py teacher --clips-file /data/distill_cache/curated_500.json --clips 50
python scripts/07_measure_gap.py edge    --clips-file /data/distill_cache/curated_500.json --clips 50
```

Each phase prints its own score and writes `<runs_root>/gap/{teacher,edge}.npz`.

---

## 3. Read the result

```bash
python scripts/07_measure_gap.py report
```

```
=== headroom: teacher vs zero-shot student ===
windows: 100
  Alpamayo 1.5         ADE  1.xxx m  (p90 ...)  minADE_6 ...
  Cosmos 3 Edge 0-shot ADE  2.xxx m  (p90 ...)  minADE_1 ...
  GAP +x.xxx m  (+xx.x% of the zero-shot error)
```

**How to act on it:**

- **Wide gap (Edge error ≥ ~1.5x teacher).** The plan stands. Proceed to the D-022
  Phase-B probe, then labeling.
- **Narrow gap (within ~20%).** Stop and rethink before spending GPU weeks. Rank-16 LoRA
  on ~10k windows is a modest intervention and will not show much against a small
  ceiling. Responses, in order of cost: raise stage-1 rank / add MLP targets, scale data
  past 5k clips, or reconsider the teacher (D-018 swapped away from the 34B Super for
  memory reasons — Super is the model NVIDIA actually positions as a distillation
  teacher).
- **Edge better than the teacher.** Almost certainly a bug, not a result. Re-check
  calibration and that `edge` really ran in `wam` mode rather than recovering the past.

**Two caveats to carry into the write-up:**

1. `minADE_k` is not comparable across different `k`. The teacher gets 6 samples here,
   Edge gets 1. **Compare the ADE column**, which is single-mode for both; the minADE
   figures are informational.
2. The teacher may have seen these clips. Alpamayo 1.5 trained on 110k+ hours of driving
   data and PhysicalAI-AV is NVIDIA's own public set, so overlap is plausible and would
   flatter the ceiling. It does not invalidate the gap as a headroom bound — it makes it
   an optimistic one. Say so in the thesis rather than letting it be found.

---

## Troubleshooting

**Calibration > 2 m.** In order of likelihood:
1. *Action layout wrong.* Try the other plausible orderings in `av_actions_to_xyz` —
   rotation-first `[rot6d(6), pos(3)]`, or absolute rather than cumulative positions
   (drop the `cumsum`). Only translation is integrated, so this is a small edit.
2. *Actions need denormalizing.* `EdgeAVRunner` passes `action_normalizer=None`. If the
   checkpoint expects normalized actions there will be an AV normalizer json under the
   framework's `datasets/normalizers/`.
3. *Wrong axes convention.* Ego-frame x-forward vs y-forward would show as a large error
   that shrinks if you swap columns 0 and 1.

**NaN from the Edge phase.** Almost certainly a pre-renewal checkpoint with an untrained
action head. Confirm you have the 2026-07-16 snapshot.

**Edge output looks like the past, not the future.** Check `mode="wam"` in
`gap.edge_predictions`. `inverse_dynamics` recovers actions from given video — that is
the calibration mode, and using it here would produce a flatteringly wrong number.

**`camera` / `fps` / resolution.** `EdgeAVRunner` defaults to `camera_front_wide_120fov`
at 30 fps and whatever `data.student_resolution` gives (360x640). Edge is native 480p.
These are our choices, not the framework's; if quality looks off after calibration
passes, they are the first knobs.

**OOM on the teacher phase.** `--samples 6` is the driver. Drop to 4.

---

## What to record

Put in `DECISIONS.md` as D-031: the two ADE numbers, the window count, the calibration
ADE, the checkpoint date, and which clip selection was used. The gap is the number every
later result gets measured against — it needs to be reproducible, not remembered.

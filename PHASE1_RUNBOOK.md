# Phase 1 runbook — what "run phase 1" means (as of 2026-09-19)

One command. Everything below is what it does and why; the decisions are
D-043..D-050 in `DECISIONS.md`.

```bash
bash scripts/run_phase1.sh          # smoke on the Blackwell, then the SFT run behind it
```

Refuses to submit if one of our jobs is already on the card (`squeue -u vla`).
The Blackwell is arbitrated by Slurm; announce a `scancel` of someone else's
job first, never of ours without asking.

## What phase 1 delivers

An AR-tower checkpoint whose **chain-of-causation reasons from the video like
the teacher AND agrees with what the driver then did** (user, 2026-09-18):
hazard in the frames, "stop" in the text, a stop in `gt_future_xyz`. The
trajectory tokens are a side output; the real trajectory comes from the phase-2
flow head. Token-path-to-CoC coupling (D-040/D-045/D-046) is NOT a phase-1 goal.

Output: `/data/vla/alpamayo2edge/runs/stage1/run-<job>/best` (merged) = the
epoch with the best driver-grounded CoC score, plus every epoch under
`epoch-NN/` (adapters + trained vocab rows; reload with `checkpoint.load_adapters`
on top of the untrained student).

## The recipe (`configs/default.yaml`, `stage1`)

| setting | value | why |
|---|---|---|
| `traj_prefix: gt`, `gt_ce: 1.0`, `traj_kl*: 0` | driver's trajectory tokens are prefix and target | teacher's token path is a 3.8 m ceiling (D-043) |
| `prefix_noise_bins: 64`, `prefix_mask_prob: 0.25` | corrupt / hide the prefix | otherwise the head copies its prefix and reads nothing else (D-045) |
| `route_hint: true` | driver's direction in `<|route_start|>` | teacher was labelled route-blind, 17% of its turns are the wrong way (D-038) |
| `filter_contradictions: true` | drop train windows whose CoC contradicts GT (~19%) | "say one thing, do another" pairs (D-043) |
| `image_dropout: 0.0` | every CoC target is used | it only served the token head (D-047) |
| `select_on: coc_gt`, `coc_gt_windows: 200` | pick the epoch on the free-running driver-grounded score | minADE picked the worst CoC epoch of run 7c (D-047) |
| `save_every_epoch: true` | keep every epoch | 7c's best epoch was overwritten and lost (D-046) |
| `epochs: 6`, `early_stop_patience: 2` | short | the CoC peaks at epoch 1-2, later epochs drift to the teacher's phrasing |
| 4 cameras, LoRA r48 attention-only, lr 3e-4 | unchanged | D-036 |

## Reading the log

```
grep -E 'coarse-minADE|vs DRIVER|early stop' logs/a2e-stage1-<job>.out
```

Per epoch: `val CoC vs DRIVER (n=200): gt_score X | consistent .. false_clear ..
direction_ok .. maneuver_acc .. | mix {...}`. `gt_score` = consistent minus
false-clear. Reference points on the 500-window table (below): run 7c epoch 1
had GT false-clear 0.164 / direction 0.417; the teacher scores 0.157 / 0.267.

## Final judgement — the 500-window table

```bash
python scripts/05b_eval_coc.py --ckpt /data/vla/alpamayo2edge/runs/stage1/run-<job>/best \
       --split val --n 500 --route-hint --dump runs/coc-run<job>-val.jsonl --out runs/coc-run<job>-val.json
python scripts/05f_coc_gt_score.py --dump runs/coc-run<job>-val.jsonl --split val
```

(both on the Ada with the shim: `CUDA_VISIBLE_DEVICES=<Ada uuid> A2E_MEM_FRAC=0.4
PYTHONPATH=src:scripts/_ada_shim`, as the earlier eval logs show). Rows that
matter, student vs teacher on the same windows: stopped -> says stop/slow/yield;
braked hard -> says slow/stop; GT false-clear; direction stated ok; and the
over-claim row (speed claim on a hold-speed window) which must stay near the
teacher's 9%. SFT cannot beat the teacher's own grounding (0.65 on stops, 0.16 on
hard brakes; D-047) - that is what phase 1.5 is for.

**Noise floor (D-052 addendum 2):** the student's CoC is sampled (T 0.6), and two draws of
the same checkpoint differed by 0.06 on false-clear and 0.09 on the 23-window stop row.
Run `05b` twice (different dumps) and average before calling a difference under ~0.1 real;
or compare checkpoints on the trainer's own 400-window `vs DRIVER` line, which is one draw
each but the same protocol across epochs.

## Then phase 1.5

`stage1_rl.init_ckpt` -> the new `best`; `sbatch scripts/03c_grpo_coc.sh`
(CoC-only rollouts, driver-grounded strict reward with the hold-speed and
NUDGE charges, D-049/D-050). Judge on the same 500-window table; a checkpoint
is better only if the driver rows rise WITHOUT the over-claim / NUDGE / precision
rows going the wrong way.

## Known pitfalls

- The per-epoch `epoch-NN/` dirs of jobs 332/334/336 are LoRA-only (pre-D-046)
  and NOT reloadable; only their `best/` is.
- `runs/stage1_rl/rl-run-6-coc-grounded/best` is the reward-hacked step 75;
  the honest checkpoint of that run is `step-0025` (adapters on `run-336/best`).
- Launcher exit code 141 = the nvidia-smi/awk SIGPIPE race; fixed 2026-09-18
  in all launchers (`awk ... !d {print; d=1}`), resubmit if it ever recurs.

#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 1.5: GRPO on the student's chain-of-causation (D-037), on the Blackwell.
#
#     cd ~/projects/alpamayo2edge && mkdir -p logs && sbatch scripts/03c_grpo_coc.sh
#
# Built from slurm_tutorial/job-template.sbatch. Verify interactively with
# salloc FIRST (05-interactive.md): "interactive sessions are for debugging,
# not for training" - but equally, don't debug inside a batch job.
# ---------------------------------------------------------------------------
#SBATCH --job-name=a2e-grpo-coc
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1    # 96GB Blackwell: the student is 7.7GB resident but
                                   # the vision tower's activations are the real cost
#SBATCH --cpus-per-task=8          # -> DataLoader workers, via $SLURM_CPUS_PER_TASK
#SBATCH --mem=64G                  # total for the job; exceeding it KILLS the job
#SBATCH --time=1-08:00:00          # under main's 3-day cap. Exceeding it KILLS the job.
                                   # RL run 3 (job 320): 200 steps at ~280 s + 9 evals
                                   # = ~16.5 h at 4 cameras; run 4 adds a ~15 min smoke
                                   # and longer CoCs at T=1.0. 32 h leaves ~2x headroom.
                                   # Per-epoch "best" checkpointing means a kill costs at
                                   # most the current epoch, not the run.
                                   # Run 4: was 2-00:00:00. 9477 train clips -> 593
                                   # steps/epoch at ~3.8 h/epoch (measured, job 232), so
                                   # stage1.epochs 12 needs 45.6 h and would have hit the
                                   # old 48 h wall around epoch 12. 68 h leaves ~50%
                                   # headroom; early stop should end it near epoch 6.
#SBATCH --output=logs/%x-%j.out    # logs/ MUST already exist or the job dies silently
#
# NOTE: deliberately NO --requeue, unlike the template and sbatch_label.sh.
# Labeling is resumable (run_labeling skips existing shards); train_stage1 is not
# - it starts at epoch 0 every time, so a requeue would silently restart training
# from scratch with only the epoch counter in the log to show for it.
set -euo pipefail

# --- provenance: makes an old log interpretable -----------------------------
echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname), started $(date -Is)"
echo "Submitted from: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"
echo "Commit:         $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "Scratch:        ${SLURM_SCRATCH:-<none>}"
echo "----------------------------------------------------------------"

# --- environment ------------------------------------------------------------
source ~/envs/alpamayo2edge/bin/activate      # cosmos-framework's supported stack:
                                              # py3.13 + torch 2.10+cu128. The Blackwell
                                              # is sm_120 and needs >= 2.7/cu128 (07).
                                              # alpamayo1.5 is installed --no-deps purely
                                              # to unpickle traj_tokenizer_spec.pt.
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"   # D-020: NOT the default token
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- don't oversubscribe the CPU -------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- pin the Blackwell, deviating from the cheatsheet, on purpose ----------
# The awk below must READ ALL of nvidia-smi's output: an `exit` on the first match
# closes the pipe while nvidia-smi may still be writing the second card's line,
# nvidia-smi dies of SIGPIPE, and under `pipefail` + `set -e` that ends the job
# with exit 141 before anything is logged (job 342, 2026-09-18 - a race, so it
# usually works). Same fix in 03_train_stage1.sh / 03a_smoke_stage1.sh.
# slurm_tutorial/CHEATSHEET.md says CUDA_VISIBLE_DEVICES=0 inside a 1-GPU job is
# an index into your allocation and must not be "fixed". That holds when the
# mapping is correct. On 2026-08-28 it is not: gres.conf declares exactly one GPU
# (File=/dev/nvidia0, which /proc/driver/nvidia/gpus confirms is the Blackwell),
# cgroup.conf sets ConstrainDevices=yes, yet inside
# `salloc --gres=gpu:rtxpro6000:1` both cards are visible and CUDA_VISIBLE_DEVICES=0
# resolves to the 48GB Ada. Measured peak for this job is 58.3 GiB, above the Ada's
# 47.4 GiB TOTAL, so an unpinned run cannot succeed - it OOMs in the decoder MLP.
#
# Since gres holds only one GPU and Slurm has allocated it to this job, pinning
# the Blackwell aligns the process with the allocation rather than circumventing
# it; no other Slurm job can hold that card. REMOVE THIS once slurmd's device
# mapping is fixed (reported to the admin) - it is a workaround, not a design.
BW_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX PRO 6000/ && !d {print $1; d=1}')"
if [ -z "$BW_UUID" ]; then
    echo "no RTX PRO 6000 visible - is this a lab account?" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$BW_UUID"

python3 - <<'GUARD'
import os, sys, torch
print("   CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("   CPUs allocated       =", os.environ.get("SLURM_CPUS_PER_TASK"))
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible to torch")
p = torch.cuda.get_device_properties(0)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
print(f"   device 0             = {p.name}  {total:.1f} GiB total, {free:.1f} GiB free")
if "RTX PRO 6000" not in p.name:
    sys.exit(f"pinned the wrong card: {p.name}")
# Run 5 (1 camera) needs far less than the 4-camera run: image tokens dominate
# activations and there are 4x fewer. The old 65 GiB floor was calibrated for
# the 4-camera peak of 58.3 GiB and would abort this job whenever a co-tenant
# holds ~40 GiB of the shared card, which on this box is routine.
#
# Size it against RESIDENT memory, not torch's allocator counter. 03a's
# "peak GPU" is max_memory_allocated() - live tensors only - and reported
# 26.9 GiB, but job 313 shows 36.7 GiB in nvidia-smi once the CUDA context,
# the caching allocator's reserve and fragmentation are counted. A floor set
# from the 26.9 figure would admit a job that then OOMs. 44 GiB = observed
# resident + ~20%. Raise back toward 65 if data.cameras returns to 4 cameras.
# RL at 4 cameras (stage1_rl.cameras overrides data.cameras): job 320 sat at
# 58.2 GiB resident in nvidia-smi for its whole run. 60 GiB = that + a margin;
# a co-tenant holding more than ~35 GiB outside SLURM means the job cannot fit.
if free < 60:
    sys.exit(f"only {free:.1f} GiB free on {p.name}; need ~60 GiB "
             f"(job 320 resident 58.2 GiB: GRPO, G=16, 4 cameras, score_chunk 4).")
GUARD

# --- W&B sidecar ----------------------------------------------------------
# Tails THIS job's stdout/err log to wandb.ai so the run is watchable from
# anywhere (phone included) without a process babysitting the node. Fully
# best-effort: backgrounded, never error-checked, and scripts/wandb_tail.py is
# a pure log parser (no torch, no CUDA) - nothing here can perturb training.
# Creds come from ~/.netrc (one-time `wandb login`). Skips itself if wandb is
# missing so a bare `bash scripts/03_train_stage1.sh` still runs.
JOB_LOG="logs/${SLURM_JOB_NAME:-a2e-stage1}-${SLURM_JOB_ID:-manual}.out"
WANDB_RUN="stage1-job${SLURM_JOB_ID:-manual}"
if python3 -c "import wandb" 2>/dev/null; then
    python3 scripts/wandb_tail.py "$JOB_LOG" --name "$WANDB_RUN" &
    WANDB_TAIL_PID=$!
    finish_wandb() {
        kill "$WANDB_TAIL_PID" 2>/dev/null || true
        wait "$WANDB_TAIL_PID" 2>/dev/null || true
        # final sweep: flush the last gate row + the "Finished"/"early stop" line
        python3 scripts/wandb_tail.py "$JOB_LOG" --name "$WANDB_RUN" --once || true
    }
    trap finish_wandb EXIT
    echo "wandb sidecar: run $WANDB_RUN (pid $WANDB_TAIL_PID)"
else
    echo "wandb sidecar: skipped (wandb not importable)"
fi

echo "----------------------------------------------------------------"
# Smoke first (G=4, 2 prompts, 2 steps, 8 val windows; ~15 min at 4 cameras),
# under its own run_name so its rows never land in the real run's steps.jsonl.
# The Ada is the usual smoke card, but at 4 cameras the smoke needs ~28 GiB
# and the Ada is often not that free; `set -e` makes a smoke failure end the
# job here, at the cost of minutes instead of a resubmit after the first step.
RUN_NAME="$(python3 -c "import yaml;print(yaml.safe_load(open('configs/default.yaml'))['stage1_rl']['run_name'])")"
echo "smoke: run_name ${RUN_NAME}-smoke"
PYTHONPATH=src python3 -m distill.train_grpo_coc --config configs/default.yaml \
    --smoke --run-name "${RUN_NAME}-smoke"
echo "smoke OK - starting ${RUN_NAME} $(date -Is)"
echo "----------------------------------------------------------------"
PYTHONPATH=src python3 -m distill.train_grpo_coc --config configs/default.yaml
echo "----------------------------------------------------------------"
echo "Finished $(date -Is)"
echo "Check what this actually used:  seff ${SLURM_JOB_ID:-<jobid>}"

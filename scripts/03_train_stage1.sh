#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 1: distill the teacher reasoner into Edge's AR tower (HANDOFF task 3).
#
#     cd ~/projects/alpamayo2edge && mkdir -p logs && sbatch scripts/03_train_stage1.sh
#
# Built from slurm_tutorial/job-template.sbatch. Verify interactively with
# salloc FIRST (05-interactive.md): "interactive sessions are for debugging,
# not for training" - but equally, don't debug inside a batch job.
# ---------------------------------------------------------------------------
#SBATCH --job-name=a2e-stage1
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1    # 96GB Blackwell: the student is 7.7GB resident but
                                   # the vision tower's activations are the real cost
#SBATCH --cpus-per-task=8          # -> DataLoader workers, via $SLURM_CPUS_PER_TASK
#SBATCH --mem=64G                  # total for the job; exceeding it KILLS the job
#SBATCH --time=2-20:00:00          # under main's 3-day cap. Exceeding it KILLS the job.
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
# Run 7 is back on 4 cameras (D-043): measured peak 58.3 GiB at micro_batch 4
# (job 232, r48 attention-only, same setting). Sized against RESIDENT memory,
# not torch's allocator counter - the CUDA context, the caching allocator's
# reserve and fragmentation sit on top of max_memory_allocated(). 65 GiB =
# observed peak + ~10%; the run-5/6 single-camera floor was 44.
if free < 65:
    sys.exit(f"only {free:.1f} GiB free on {p.name}; need ~65 GiB "
             f"(job 232 peak 58.3 GiB at micro_batch 4, 4 cameras).")
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
PYTHONPATH=src python3 -m distill.train_stage1 --config configs/default.yaml
echo "----------------------------------------------------------------"
echo "Finished $(date -Is)"
echo "Check what this actually used:  seff ${SLURM_JOB_ID:-<jobid>}"

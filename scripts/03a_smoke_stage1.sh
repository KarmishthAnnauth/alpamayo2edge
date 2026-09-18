#!/bin/bash
# Pre-flight for stage 1, under SLURM. Submit with:
#     sbatch scripts/03a_smoke_stage1.sh          # forward + backward, 1 batch
#     sbatch scripts/03a_smoke_stage1.sh --gate 2 # ...plus 2 windows of the epoch gate
#
# WHY THIS EXISTS. NEXT_STEPS.md §4 says to run the smoke test as a bare
# `python scripts/03a_smoke_stage1.py`. That is not safe on this node: per
# 03_train_stage1.sh's pinning note, slurmd's device mapping is broken and
# torch's device 0 resolves to the 48 GiB **Ada**, not the Blackwell. The smoke
# test's measured peak is the same 58.3 GiB as training at micro_batch 4, above
# the Ada's 47.4 GiB TOTAL - so a bare run cannot succeed, and on a shared node
# it takes other people's processes down with it on the way to its own OOM.
# Going through SLURM also means the scheduler arbitrates rather than us racing
# whoever holds the card.
#
# ---------------------------------------------------------------------------
#SBATCH --job-name=a2e-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00            # one batch + an optional 2-window gate. If this
                                   # is not done in 30 min something is wrong and
                                   # the queue should get the card back.
#SBATCH --output=logs/%x-%j.out    # logs/ MUST already exist or the job dies silently
#
set -euo pipefail

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname), started $(date -Is)"
echo "Commit:         $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "Args:           $*"
echo "----------------------------------------------------------------"

source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Same UUID pin as 03_train_stage1.sh, and for the same reason - see the long
# comment there. Remove from BOTH scripts once slurmd's mapping is fixed.
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
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible to torch")
p = torch.cuda.get_device_properties(0)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
print(f"   device 0             = {p.name}  {total:.1f} GiB total, {free:.1f} GiB free")
if "RTX PRO 6000" not in p.name:
    sys.exit(f"pinned the wrong card: {p.name}")
if free < 65:            # same peak as training: this batch is the real micro_batch
    sys.exit(f"only {free:.1f} GiB free on {p.name}; need ~60 GiB.")
GUARD

echo "----------------------------------------------------------------"
PYTHONPATH=src python3 scripts/03a_smoke_stage1.py --config configs/default.yaml "$@"
echo "----------------------------------------------------------------"
echo "Finished $(date -Is)"

#!/bin/bash
# Teacher labeling on the Blackwell via SLURM (HANDOFF task 2).
#
#   sbatch scripts/sbatch_label.sh                 # full increment, N=500
#   sbatch scripts/sbatch_label.sh 2000            # N=2000
#   sbatch --partition=debug scripts/sbatch_label.sh 500 3   # smoke: first 3 clips, 1h cap
#
# Resumable by design: run_labeling skips existing shards (atomic writes, so a
# kill/requeue mid-shard cannot poison the cache) - requeued or resubmitted jobs
# just continue. No --time set: each partition's limit applies (main 3d, debug 1h).
#
#SBATCH --job-name=a2e-label
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

N="${1:-500}"
LIMIT="${2:-}"

cd "$HOME/Karmishth/alpamayo2edge/alpamayo2edge"
source .venv-a2e/bin/activate                                    # A1.5 teacher env; the old
                                                                 # .venv was pre-teacher-swap
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"        # D-020: NOT the default stored token
export HF_HUB_CACHE="$HOME/Karmishth/alpamayo2edge/models"       # D-019: else HF re-downloads 67 GB

echo "== $(date -Is) job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0} N=$N LIMIT=${LIMIT:-none}"
nvidia-smi -L

CACHE=$(python -c "import sys; sys.path.insert(0,'src')
from distill.config import load_config; print(load_config().paths.cache_root)")
if [ ! -f "$CACHE/curated_$N.json" ]; then
    echo "== curated_$N.json missing - running curation (deterministic, seed=0)"
    python scripts/01_curate.py --n "$N"
fi

python scripts/02_label.py --n "$N" ${LIMIT:+--limit "$LIMIT"}
echo "== $(date -Is) labeling done"

#!/bin/bash
# Teacher labeling via SLURM (HANDOFF task 2).
#
#   sbatch scripts/sbatch_label.sh 5000           # label curated_5000 (skips the ~3.9k already done)
#   sbatch scripts/sbatch_label.sh 5000 3         # smoke: first 3 clips of the deterministic list
#
# Resumable by design: run_labeling skips existing shards (atomic writes, so a
# kill/requeue mid-shard cannot poison the cache) - requeued or resubmitted jobs
# just continue. No --time set: each partition's limit applies (main 3d, debug 1h).
#
# Runs on ProArt. The node exposes ONE gres slot but both cards are visible
# inside the job. This pins the 48GB Ada and leaves the Blackwell for stage-1
# training (scripts/03_train_stage1.sh), so labeling and a training run can go
# at once on separate cards. The teacher is ~22GB and labeling peaks ~30GB, so
# it fits the Ada with headroom.
#
# NOTE: because SLURM only has one gres slot, an sbatch labeling job and an
# sbatch training job will SERIALIZE (second one queues). To run them together,
# launch labeling directly instead (scripts/run_label.sh) - it is resumable, so
# it does not need SLURM's requeue protection - and keep the Blackwell training
# under sbatch.
#
#SBATCH --job-name=a2e-label
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8          # -> the prefetch ThreadPool + av video decode
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out    # logs/ MUST already exist
set -euo pipefail

N="${1:-5000}"
LIMIT="${2:-}"

cd "${SLURM_SUBMIT_DIR:-$HOME/projects/alpamayo2edge}"
mkdir -p logs
# py3.12 A1.5 teacher env, built with uv per the recipe in requirements.txt:
#   uv venv --python 3.12 .venv-a2e
#   uv pip install -e ../physical_ai_av
#   uv pip install -e ../alpamayo1.5 --no-deps
#   uv pip install -r requirements.txt
#   uv pip install accelerate hydra-colorlog pillow matplotlib seaborn
#   uv pip install --no-deps <flash-attn 2.8.3 cu12/torch2.8/cp312 wheel>
source .venv-a2e/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"        # D-020: NOT the default stored token
export HF_HUB_CACHE=/bulk/users/vla/alpamayo2edge/hf-cache       # Cosmos-Reason2-8B processor/tokenizer live here
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- pin the Ada (see header) ---------------------------------------------
ADA_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX 6000 Ada/{print $1; exit}')"
if [ -z "$ADA_UUID" ]; then
    echo "no RTX 6000 Ada visible" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$ADA_UUID"

echo "== $(date -Is) job=${SLURM_JOB_ID:-?} host=$(hostname) restarts=${SLURM_RESTART_COUNT:-0} N=$N LIMIT=${LIMIT:-none}"
echo "== commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi -L
python -c "import torch,sys
p=torch.cuda.get_device_properties(0); free,tot=(x/2**30 for x in torch.cuda.mem_get_info())
print(f'   pinned {p.name}  {tot:.0f} GiB total, {free:.0f} GiB free')
ok = 'Ada' in p.name and free > 23
sys.exit(0 if ok else f'wrong card or low mem: {p.name}, {free:.0f} GiB free')"

CACHE=$(python -c "import sys; sys.path.insert(0,'src')
from distill.config import load_config; print(load_config().paths.cache_root)")
if [ ! -f "$CACHE/curated_$N.json" ]; then
    echo "== curated_$N.json missing - running curation (deterministic, seed=0)"
    python scripts/01_curate.py --n "$N"
fi

python scripts/02_label.py --n "$N" ${LIMIT:+--limit "$LIMIT"}
echo "== $(date -Is) labeling done"

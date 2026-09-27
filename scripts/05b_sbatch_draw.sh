#!/bin/bash
# One free-running CoC draw on the Blackwell via SLURM: full split, route hint,
# the runbook's 500-window protocol widened to every window in the split.
#
#     TAG=<name> [AD=<adapters dir>] [N=1046] [SPLIT=val] sbatch scripts/05b_sbatch_draw.sh
#
#SBATCH --job-name=a2e-eval-coc
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${TAG:?set TAG=<output name>}"
source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '/Blackwell/ && !d {print $1; d=1}')"
export PYTHONPATH=src
EXTRA=()
[ -n "${AD:-}" ] && EXTRA+=(--adapters "$AD")
echo "Job ${SLURM_JOB_ID} $(date -Is)  adapters=${AD:-none} split=${SPLIT:-val} n=${N:-1046} tag=$TAG"
python3 scripts/05b_eval_coc.py \
    --ckpt /data/vla/alpamayo2edge/runs/stage1/run-336/best "${EXTRA[@]}" \
    --split "${SPLIT:-val}" --n "${N:-1046}" --route-hint \
    --dump "runs/$TAG.jsonl" --out "runs/$TAG.json"
echo "Finished $(date -Is)"

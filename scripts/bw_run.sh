#!/bin/bash
# Run a python script on the SLURM-managed Blackwell (partition main). Every
# submission is announced to the user first (memory: gpu-etiquette).
#
#     sbatch [--time=HH:MM:SS --job-name=...] scripts/bw_run.sh scripts/<x>.py args...
#
#SBATCH --job-name=a2e-bw
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname), started $(date -Is); commit $(git rev-parse --short HEAD)"
source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
BW_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX PRO 6000/ && !d {print $1; d=1}')"
[ -n "$BW_UUID" ] || { echo "no RTX PRO 6000 visible" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$BW_UUID"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$@"

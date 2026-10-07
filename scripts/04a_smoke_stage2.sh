#!/bin/bash
# Stage-2 smoke on the Blackwell: 6 micro-batches of the bench2drive trainer
# (full FT, own (t, a0) draws, CoC cache), a 16-window gate, one bf16 save.
#     sbatch [--dependency=afterany:<job>] scripts/04a_smoke_stage2.sh
#SBATCH --job-name=a2e-smoke-stage2
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
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
export CUDA_VISIBLE_DEVICES="$BW_UUID"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m distill.train_stage2 --config configs/default.yaml --smoke 6

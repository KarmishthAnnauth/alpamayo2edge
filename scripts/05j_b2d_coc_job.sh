#!/bin/bash
# Step 2+3 prerequisites on the Blackwell: open-loop sanity of the flow
# evaluator on the untrained init (2 min), then the full CoC cache.
#SBATCH --job-name=a2e-b2d-coc
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
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
echo "=== 05k sanity: untrained init, 32 val windows, no CoC ==="
python3 scripts/05k_flow_minade.py --split val --n 32 --k 6 --batch 8 --coc-cache none \
        --out runs/flow-minade-init-val32-nococ.json
echo "=== 05j generate: CoC cache over every window (resumable) ==="
python3 scripts/05j_b2d_coc.py generate
echo "=== 05j score ==="
python3 scripts/05j_b2d_coc.py score --json runs/b2d-coc-rl7s25-score-all.json
python3 scripts/05j_b2d_coc.py score --split val --json runs/b2d-coc-rl7s25-score-val.json

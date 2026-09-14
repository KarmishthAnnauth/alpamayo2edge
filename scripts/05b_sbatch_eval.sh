#!/bin/bash
# Score a checkpoint (merged, or init + adapters) on the Blackwell via SLURM,
# for when the Ada is taken. ~10 min for 500 val + 500 train windows.
#
#     INIT=<merged ckpt> [AD=<adapters dir>] TAG=<name> [ROUTE=1] sbatch scripts/05b_sbatch_eval.sh
#
#SBATCH --job-name=a2e-eval-coc
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${INIT:?set INIT=<checkpoint dir>}"; : "${TAG:?set TAG=<output name>}"
source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '/Blackwell/{print $1; exit}')"
export PYTHONPATH=src
OUT=/data/vla/alpamayo2edge/runs/coc_eval
EXTRA=()
[ -n "${AD:-}" ] && EXTRA+=(--adapters "$AD")
[ "${ROUTE:-0}" = "1" ] && EXTRA+=(--route-hint)
echo "Job ${SLURM_JOB_ID} $(date -Is)  init=$INIT adapters=${AD:-none} route=${ROUTE:-0} tag=$TAG"
for SPLIT in val train; do
  echo "=================== $TAG $SPLIT ==================="
  python3 scripts/05b_eval_coc.py --ckpt "$INIT" "${EXTRA[@]}" --split "$SPLIT" --n 500 \
      --out "$OUT/$TAG-$SPLIT.json" --dump "$OUT/$TAG-$SPLIT.jsonl"
done
echo "Finished $(date -Is)"

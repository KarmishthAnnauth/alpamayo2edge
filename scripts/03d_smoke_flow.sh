#!/bin/bash
# Phase-2 flow-head smoke on the Ada (LoRA arm; --full-ft needs the Blackwell).
# Same guard as 03b_ada_rehearsal.sh: pin the Ada by UUID, refuse if others hold
# more than OTHERS_MAX_MIB, cap our share via the sitecustomize shim.
set -euo pipefail
MEM_FRAC=${MEM_FRAC:-0.70}
OTHERS_MAX_MIB=${OTHERS_MAX_MIB:-4000}

source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

ADA_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
            | awk -F', ' '/RTX 6000 Ada/ && !d {print $1; d=1}')"
[ -n "$ADA_UUID" ] || { echo "no RTX 6000 Ada visible" >&2; exit 1; }
OTHERS_MIB="$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader \
              | awk -F', ' -v u="$ADA_UUID" '$1==u {s += $2} END {print s+0}')"
echo "others currently on the Ada: ${OTHERS_MIB} MiB"
if [ "$OTHERS_MIB" -gt "$OTHERS_MAX_MIB" ]; then
    echo "REFUSING: others hold ${OTHERS_MIB} MiB (> ${OTHERS_MAX_MIB})." >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$ADA_UUID"
export A2E_MEM_FRAC="$MEM_FRAC"
export PYTHONPATH="src:scripts/_ada_shim${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/03d_smoke_flow.py "$@"

#!/bin/bash
# Run any python script on the shared Ada behind the standard guard: pin by
# UUID, refuse if others hold > OTHERS_MAX_MIB, cap our share (MEM_FRAC) via the
# sitecustomize shim. Never touches the Blackwell.
#
#     bash scripts/ada_run.sh scripts/05j_b2d_coc.py generate --n 100
set -euo pipefail
MEM_FRAC=${MEM_FRAC:-0.60}
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
exec python3 "$@"

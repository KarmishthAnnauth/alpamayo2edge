#!/bin/bash
# Stage-1 REHEARSAL on the Ada, outside SLURM. Not for the real run.
#
#     bash scripts/03b_ada_rehearsal.sh --smoke          # one batch, ~5 min
#     bash scripts/03b_ada_rehearsal.sh                  # train on the current split
#
# WHY A SEPARATE SCRIPT. 03_train_stage1.sh pins the Blackwell by UUID and
# refuses below 65 GiB free, both correct for the real run and both fatal here.
# This is its opposite: pin the Ada, size the batch down, and cap what we take.
#
# THE ADA IS SHARED AND UNARBITRATED. Nothing schedules it - no SLURM, no
# reservation. At the time of writing it carries two of roberto's watch_task.py
# demo servers with 25- and 22-day uptimes. If we take the whole card they die,
# and nothing will tell us we did it. So:
#   * MEM_FRAC caps this process below the card, so an underestimate OOMs US
#     rather than starving them.
#   * The preflight refuses to start if someone else's footprint has grown
#     since we sized the run.
# Re-check `nvidia-smi` before launching. If a third party has appeared, the
# right move is to wait, not to shrink the cap until it fits.
#
set -euo pipefail

SMOKE=0
[ "${1:-}" = "--smoke" ] && { SMOKE=1; shift; }

# micro_batch 2 x grad_accum 16 = 32, the config's effective batch. Keep the
# product at 32 or the gate stops being comparable to runs 232/243.
MB=${MB:-2}
GA=${GA:-16}
# Fraction of the Ada this process may allocate.
#
# MEASURED 2026-09-06 (run-4 smoke, this script): peak 38.9 GiB at micro_batch 2.
# The first attempt at 0.80 (= 37.9 GiB) OOM'd in backward needing 1.94 GiB more,
# so 0.80 is BELOW the real requirement, not above it - do not "fix" an OOM here
# by lowering this. 0.88 = 41.7 GiB clears the measured peak by 2.8 GiB and still
# leaves ~5.7 GiB of the card for the demo servers.
#
# Decomposed against the Blackwell's 58.3 GiB at micro_batch 4: fixed cost
# ~19.5 GiB (weights + optimizer states + grads, card-independent) and ~9.7 GiB
# of activations per sample. So micro_batch 1 is ~29 GiB if the card gets busier.
MEM_FRAC=${MEM_FRAC:-0.88}
# Refuse to start if others already hold more than this (MiB). Sized so the cap
# above is actually honourable: 0.80 * 49140 + 4000 < 49140.
OTHERS_MAX_MIB=${OTHERS_MAX_MIB:-4000}

echo "Ada rehearsal, started $(date -Is)"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "Batch:   micro_batch $MB x grad_accum $GA = $((MB * GA))"
echo "Cap:     ${MEM_FRAC} of the card"
echo "----------------------------------------------------------------"

source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

ADA_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
            | awk -F', ' '/RTX 6000 Ada/{print $1; exit}')"
if [ -z "$ADA_UUID" ]; then
    echo "no RTX 6000 Ada visible" >&2
    exit 1
fi

# Everyone else's footprint on the Ada, before we add ours.
OTHERS_MIB="$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader \
              | awk -F', ' -v u="$ADA_UUID" '$1==u {s += $2} END {print s+0}')"
echo "others currently on the Ada: ${OTHERS_MIB} MiB"
if [ "$OTHERS_MIB" -gt "$OTHERS_MAX_MIB" ]; then
    echo "REFUSING: others hold ${OTHERS_MIB} MiB (> ${OTHERS_MAX_MIB})." >&2
    echo "Someone else's job arrived after this run was sized. Wait for the" >&2
    echo "Blackwell instead of squeezing them - see the header." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$ADA_UUID"
export A2E_MEM_FRAC="$MEM_FRAC"

python3 - <<'GUARD'
import os, sys, torch
p = torch.cuda.get_device_properties(0)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
print(f"   device 0 = {p.name}  {total:.1f} GiB total, {free:.1f} GiB free")
if "RTX 6000 Ada" not in p.name:
    sys.exit(f"pinned the wrong card: {p.name}")
GUARD

# The cap has to be set inside the training process, so it goes in via sitecustomize
# rather than the launcher. PYTHONSTARTUP does not apply to -m, and a wrapper module
# is less surprising than patching the trainer with a card-specific flag.
export PYTHONPATH="src:scripts/_ada_shim${PYTHONPATH:+:$PYTHONPATH}"

echo "----------------------------------------------------------------"
if [ "$SMOKE" = "1" ]; then
    python3 scripts/03a_smoke_stage1.py --config configs/default.yaml \
            --micro-batch "$MB" "$@"
else
    python3 -m distill.train_stage1 --config configs/default.yaml \
            --micro-batch "$MB" --grad-accum "$GA" "$@"
fi
echo "----------------------------------------------------------------"
echo "Finished $(date -Is)"

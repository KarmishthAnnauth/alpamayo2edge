#!/bin/bash
# ---------------------------------------------------------------------------
# PHASE 1 in one command: smoke on the Blackwell, then the SFT run chained
# behind it (`--dependency=afterok`), so the run only starts if the smoke
# passes. Everything the run does is in configs/default.yaml `stage1`; the
# reasoning is in PHASE1_RUNBOOK.md and DECISIONS.md D-043..D-050.
#
#     bash scripts/run_phase1.sh            # submit smoke + train
#     bash scripts/run_phase1.sh --no-smoke # train only (the smoke already passed today)
#
# What it delivers: runs/stage1/run-<job>/best (merged; the epoch with the best
# DRIVER-grounded free-running CoC score) plus every epoch under epoch-NN/
# (adapters + the trained vocab rows, reloadable with checkpoint.load_adapters).
# The log prints, per epoch:
#     epoch N challenging coarse-minADE ... | val CoC NLL ...
#     epoch N val CoC vs DRIVER (n=200): gt_score ... consistent ... false_clear ...
# Judge it on the second line and, for the final call, on the 500-window table:
#     python scripts/05b_eval_coc.py --ckpt runs/stage1/run-<job>/best --split val \
#            --n 500 --route-hint --dump runs/coc-<name>-val.jsonl
#     python scripts/05f_coc_gt_score.py --dump runs/coc-<name>-val.jsonl --split val
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

if squeue -h -o "%j" | grep -qE "a2e-stage1|a2e-grpo-coc|a2e-smoke"; then
    echo "a job of ours is already queued or running on the Blackwell:" >&2
    squeue -u "$USER" -o "%.6i %.12j %.3t %.8M %R" >&2
    echo "cancel it first (scancel <id>) or wait; not submitting." >&2
    exit 1
fi

echo "config (stage1):"
python3 - <<'PY'
import yaml
c = yaml.safe_load(open("configs/default.yaml"))["stage1"]
for k in ("traj_prefix", "route_hint", "filter_contradictions", "image_dropout",
          "prefix_noise_bins", "prefix_mask_prob", "select_on", "coc_gt_windows",
          "epochs", "early_stop_patience", "save_every_epoch"):
    print(f"   {k:<22} {c.get(k)}")
PY

if [ "${1:-}" = "--no-smoke" ]; then
    TRAIN=$(sbatch --parsable scripts/03_train_stage1.sh)
    echo "train job $TRAIN  -> logs/a2e-stage1-$TRAIN.out"
else
    SMOKE=$(sbatch --parsable scripts/03a_smoke_stage1.sh --gate 2)
    TRAIN=$(sbatch --parsable --dependency=afterok:"$SMOKE" scripts/03_train_stage1.sh)
    echo "smoke job $SMOKE  -> logs/a2e-smoke-$SMOKE.out"
    echo "train job $TRAIN  -> logs/a2e-stage1-$TRAIN.out  (starts when the smoke passes)"
fi
echo
echo "watch:   grep -E 'coarse-minADE|vs DRIVER|early stop' logs/a2e-stage1-$TRAIN.out"
echo "result:  /data/vla/alpamayo2edge/runs/stage1/run-$TRAIN/best  (+ epoch-NN/)"

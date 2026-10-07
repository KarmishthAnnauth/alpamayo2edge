#!/bin/bash
# Phase 2: closed-loop Bench2Drive eval of a stage-2 checkpoint (PHASE2_DECISIONS T3).
#
#   bash scripts/06_b2d_closed_loop.sh --run-id r4-smoke --routes-subset 0          # one route
#   bash scripts/06_b2d_closed_loop.sh --run-id r4-b2d220                           # all 220, resumable
#   bash scripts/06_b2d_closed_loop.sh --run-id r4-k6mean --k 6 --temperature 1 --select mean
#
# Two processes (src/distill/closed_loop/a2e_common.py says why):
#   1. the model server, in OUR env on the Ada behind the usual guard
#      (scripts/ada_run.sh; SERVER_LAUNCHER=... to change), holding the student;
#   2. the stock Bench2Drive evaluator from ~/projects/TriTrack, in ITS env
#      (python 3.10 + carla 0.9.15), which starts its own CARLA server and
#      imports src/distill/closed_loop/a2e_agent.py.
# Re-running with the same --run-id resumes from results.json.
#
# Output: runs/closed_loop/<run-id>/{results.json, live.txt, eval.log, server.log,
#         viz/<route>/{meta/*.json, metric_info.json}}
#
# Ports: CARLA binds --port, +1, +2 (default 2500) and the traffic manager
# --tm-port (default 8000). Check `ss -lntp` first on this shared box.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD

TRITRACK_ROOT=${TRITRACK_ROOT:-$HOME/projects/TriTrack/TriTrack}
CKPT=runs/stage2/sft-run-4-b2d-arlora-traj-cocce/best
ROUTES=$TRITRACK_ROOT/Bench2Drive/leaderboard/data/bench2drive220.xml
RUN_ID=; SUBSET=; PORT=2500; TM_PORT=8000; GPU_RANK=0; TIMEOUT=600
K=1; TEMPERATURE=0; SELECT=first
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt) CKPT=$2; shift 2 ;;
        --routes) ROUTES=$2; shift 2 ;;
        --routes-subset) SUBSET=$2; shift 2 ;;
        --run-id) RUN_ID=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --tm-port) TM_PORT=$2; shift 2 ;;
        --gpu-rank) GPU_RANK=$2; shift 2 ;;          # CARLA's -graphicsadapter
        --timeout) TIMEOUT=$2; shift 2 ;;
        --k) K=$2; shift 2 ;;
        --temperature) TEMPERATURE=$2; shift 2 ;;
        --select) SELECT=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$RUN_ID" ]] || { echo "--run-id is required" >&2; exit 2; }
[[ "$CKPT" == init || -d "$CKPT" ]] || { echo "no checkpoint dir $CKPT (or --ckpt init)" >&2; exit 2; }
[[ -f "$ROUTES" ]] || { echo "no routes file $ROUTES" >&2; exit 2; }

OUT=$REPO/runs/closed_loop/$RUN_ID
mkdir -p "$OUT/viz"
SOCK=$OUT/model.sock

for p in $PORT $((PORT + 1)) $((PORT + 2)) $TM_PORT; do
    if ss -lnt | awk '{print $4}' | grep -q ":$p\$"; then
        echo "port $p is in use; pass --port / --tm-port" >&2; exit 1
    fi
done

# ---- 1. model server ---------------------------------------------------------
SERVER_LAUNCHER=${SERVER_LAUNCHER:-scripts/ada_run.sh}
CKPT_ARG=(--ckpt "$CKPT"); [[ "$CKPT" == init ]] && CKPT_ARG=()
rm -f "$SOCK"
bash "$SERVER_LAUNCHER" -m distill.closed_loop.server --socket "$SOCK" "${CKPT_ARG[@]}" \
    --k "$K" --temperature "$TEMPERATURE" --select "$SELECT" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; rm -f "$SOCK"; }
trap cleanup EXIT
echo "[a2e] model server pid $SERVER_PID, log $OUT/server.log"
for _ in $(seq 600); do
    [[ -S "$SOCK" ]] && grep -q "READY" "$OUT/server.log" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[a2e] model server died:" >&2; tail -20 "$OUT/server.log" >&2; exit 1
    fi
    sleep 1
done
[[ -S "$SOCK" ]] || { echo "[a2e] model server not ready after 600 s" >&2; exit 1; }
echo "[a2e] model server ready"

# ---- 2. evaluator (environment as TriTrack's run_eval.sh sets it) -------------
source "$TRITRACK_ROOT/tools/load_env.sh"
_tritrack_load_env "${TRITRACK_ENV_FILE:-$TRITRACK_ROOT/tritrack.env}"
[[ -x "${TRITRACK_CARLA_ROOT:-}/CarlaUE4.sh" ]] || { echo "no CARLA at TRITRACK_CARLA_ROOT=${TRITRACK_CARLA_ROOT:-<unset>}" >&2; exit 1; }
export CARLA_ROOT=$TRITRACK_CARLA_ROOT
export TRITRACK_ROOT
BENCH2DRIVE_ROOT=${TRITRACK_BENCH2DRIVE:-$TRITRACK_ROOT/Bench2Drive}
export LEADERBOARD_ROOT=$BENCH2DRIVE_ROOT/leaderboard
export SCENARIO_RUNNER_ROOT=$BENCH2DRIVE_ROOT/scenario_runner
PYTHON=$TRITRACK_PYTHON
CONDA_SITE=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PY_PREFIX=$("$PYTHON" -c 'import sys; print(sys.prefix)')
export DISPLAY=${TRITRACK_DISPLAY:-${DISPLAY:-:1}}
unset SDL_VIDEODRIVER || true
export LD_LIBRARY_PATH="$PY_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$TRITRACK_ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla:$CONDA_SITE
export TRITRACK_CARLA_ARGS=${TRITRACK_CARLA_ARGS:-}
export IS_BENCH2DRIVE=True
export SAVE_PATH=$OUT/viz

ARGS=(--routes "$ROUTES" --repetitions 1 --track SENSORS
      --agent "$REPO/src/distill/closed_loop/a2e_agent.py" --agent-config "$SOCK"
      --checkpoint "$OUT/results.json" --debug-checkpoint "$OUT/live.txt"
      --port "$PORT" --traffic-manager-port "$TM_PORT" --gpu-rank "$GPU_RANK"
      --timeout "$TIMEOUT" --debug 0 --resume True)
[[ -n "$SUBSET" ]] && ARGS+=(--routes-subset "$SUBSET")

echo "[a2e] ckpt $CKPT | routes $ROUTES ${SUBSET:+(subset $SUBSET) }| k=$K T=$TEMPERATURE select=$SELECT"
echo "[a2e] results $OUT/results.json"
set +e
"$PYTHON" -u "$REPO/src/distill/closed_loop/run_b2d.py" "${ARGS[@]}" 2>&1 | tee -a "$OUT/eval.log"
code=${PIPESTATUS[0]}
set -e
echo "[a2e] evaluator exit $code; summary (driving score, success rate, per ability):"
echo "    (cd $TRITRACK_ROOT && $PYTHON -m tritrack.report $OUT/results.json)"
exit "$code"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Run a local robot-policy simulator showcase.

Usage:
  ./scripts/run_showcase.sh [options]
  ./scripts/run_interactive_showcase.sh [options]

Backend, model, and task options:
  --backend libero|robocasa       Simulator backend (default: libero)
  --model pi05|fastwam|groot-n1.5 Local policy plugin (default: pi05)
  --world-model NAME              none (default) or robocasa-sim oracle baseline
  --compare-world-model           Compare prediction after each real action prefix
  --no-compare-world-model        Run normally without comparison (default)
  --task-suite NAME               LIBERO suite or RoboCasa task set
  --task-set NAME                 Alias for --task-suite
  --task-id ID                    Initial task interactively, or one batch task
  --task-ids IDS                  Batch task IDs (comma-separated or all)
  --split pretrain|target         RoboCasa scene/object split (default: target)
  --trials COUNT                  Automatic attempts for a batch run
  --seed SEED                     Reproducible simulator seed
  --replan-steps COUNT            Actions executed before querying the policy again
  --max-policy-steps COUNT        LIBERO test-only rollout cap; 0 uses suite default
  --viewer-width PX               RoboCasa dashboard render width (default: 960)
  --viewer-height PX              RoboCasa dashboard render height (default: 540)
  --viewer-fps FPS                Maximum dashboard render rate (default: 6)
  --prompt TEXT                   Initial instruction override
  --evaluation-mode MODE          scored or exploratory
  --budget 1|2|3                  Rollout budget multiplier

Session options:
  --interactive                   Wait for browser commands
  --batch                         Start automatically and exit after --trials
  --auto-start                    Start the initial interactive rollout immediately
  --realtime-delay-ms MS          Delay after each simulator action
  --policy-port PORT              Local policy server port
  --dashboard-port PORT           Browser dashboard port
  --hold-open / --no-hold-open    Keep or close the dashboard after completion
  --open / --no-open              Enable or disable automatic browser opening
  --network-audit / --no-network-audit
  -h, --help                      Show this help

Examples:
  ./scripts/run_interactive_showcase.sh --backend robocasa --task-set atomic_seen
  ./scripts/run_interactive_showcase.sh --backend robocasa --model groot-n1.5
  ./scripts/run_showcase.sh --backend robocasa --batch --task-id 2 --trials 3
  ./scripts/run_showcase.sh --backend libero --task-suite libero_spatial --task-ids 0,1
  ./scripts/run_interactive_showcase.sh --backend libero --model fastwam --task-id 2

Environment-variable controls remain supported for backward compatibility.
EOF
}

BACKEND="${BACKEND:-libero}"
MODEL="${MODEL:-pi05}"
WORLD_MODEL="${WORLD_MODEL:-}"
COMPARE_WORLD_MODEL="${COMPARE_WORLD_MODEL:-0}"
TASK_SUITE="${TASK_SUITE:-}"
TASK_IDS="${TASK_IDS:-}"
ROBOCASA_SPLIT="${ROBOCASA_SPLIT:-target}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
REPLAN_STEPS="${REPLAN_STEPS:-}"
MAX_POLICY_STEPS="${MAX_POLICY_STEPS:-0}"
VIEWER_WIDTH="${VIEWER_WIDTH:-960}"
VIEWER_HEIGHT="${VIEWER_HEIGHT:-540}"
VIEWER_FPS="${VIEWER_FPS:-6}"
POLICY_PORT="${POLICY_PORT:-${PI05_PORT:-8000}}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8085}"
REALTIME_DELAY_MS="${REALTIME_DELAY_MS:-35}"
NETWORK_AUDIT="${NETWORK_AUDIT:-1}"
AUTO_OPEN="${AUTO_OPEN:-1}"
HOLD_OPEN="${HOLD_OPEN:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
AUTO_START="${AUTO_START:-0}"
INITIAL_PROMPT="${INITIAL_PROMPT:-}"
EVALUATION_MODE="${EVALUATION_MODE:-}"
ROLLOUT_BUDGET_MULTIPLIER="${ROLLOUT_BUDGET_MULTIPLIER:-0}"
LOCAL_LLM_URL="${LOCAL_LLM_URL:-}"
LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-}"
LOCAL_LLM_NUM_GPU="${LOCAL_LLM_NUM_GPU:-0}"
SESSION_DIR="${SESSION_DIR:-}"
FASTWAM_DIR="${FASTWAM_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend requires a value}"; shift 2 ;;
    --model) MODEL="${2:?--model requires a value}"; shift 2 ;;
    --world-model) WORLD_MODEL="${2:?--world-model requires a value}"; shift 2 ;;
    --preview-steps|--preview-approval)
      echo "Warning: $1 is deprecated; comparison length follows --replan-steps." >&2
      : "${2:?$1 requires a value}"
      shift 2
      ;;
    --compare-world-model) COMPARE_WORLD_MODEL=1; shift ;;
    --no-compare-world-model) COMPARE_WORLD_MODEL=0; shift ;;
    --task-suite|--task-set) TASK_SUITE="${2:?$1 requires a value}"; shift 2 ;;
    --task-id|--task-ids) TASK_IDS="${2:?$1 requires a value}"; shift 2 ;;
    --split) ROBOCASA_SPLIT="${2:?--split requires a value}"; shift 2 ;;
    --trials) TRIALS_PER_TASK="${2:?--trials requires a value}"; shift 2 ;;
    --seed) SEED="${2:?--seed requires a value}"; shift 2 ;;
    --replan-steps) REPLAN_STEPS="${2:?--replan-steps requires a value}"; shift 2 ;;
    --max-policy-steps) MAX_POLICY_STEPS="${2:?--max-policy-steps requires a value}"; shift 2 ;;
    --viewer-width) VIEWER_WIDTH="${2:?--viewer-width requires a value}"; shift 2 ;;
    --viewer-height) VIEWER_HEIGHT="${2:?--viewer-height requires a value}"; shift 2 ;;
    --viewer-fps) VIEWER_FPS="${2:?--viewer-fps requires a value}"; shift 2 ;;
    --prompt) INITIAL_PROMPT="${2:?--prompt requires a value}"; shift 2 ;;
    --evaluation-mode) EVALUATION_MODE="${2:?--evaluation-mode requires a value}"; shift 2 ;;
    --budget) ROLLOUT_BUDGET_MULTIPLIER="${2:?--budget requires a value}"; shift 2 ;;
    --policy-port) POLICY_PORT="${2:?--policy-port requires a value}"; shift 2 ;;
    --dashboard-port) DASHBOARD_PORT="${2:?--dashboard-port requires a value}"; shift 2 ;;
    --realtime-delay-ms) REALTIME_DELAY_MS="${2:?--realtime-delay-ms requires a value}"; shift 2 ;;
    --session-dir) SESSION_DIR="${2:?--session-dir requires a value}"; shift 2 ;;
    --interactive) INTERACTIVE=1; shift ;;
    --batch) INTERACTIVE=0; shift ;;
    --auto-start) AUTO_START=1; shift ;;
    --hold-open) HOLD_OPEN=1; shift ;;
    --no-hold-open) HOLD_OPEN=0; shift ;;
    --open) AUTO_OPEN=1; shift ;;
    --no-open) AUTO_OPEN=0; shift ;;
    --network-audit) NETWORK_AUDIT=1; shift ;;
    --no-network-audit) NETWORK_AUDIT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

BACKEND="${BACKEND,,}"
MODEL="${MODEL,,}"
WORLD_MODEL="${WORLD_MODEL,,}"
case "$MODEL" in
  pi|pi0.5|pi-0.5) MODEL="pi05" ;;
  groot|gr00t|gr00t-n1.5|groot_n1.5|gr00t_n1.5) MODEL="groot-n1.5" ;;
  fast-wam|fast_wam) MODEL="fastwam" ;;
esac
case "$WORLD_MODEL" in
  off|direct) WORLD_MODEL="none" ;;
  sim|simulator|simulator-oracle) WORLD_MODEL="robocasa-sim" ;;
esac

case "$BACKEND" in
  libero)
    OPENPI_DIR="${LIBERO_OPENPI_DIR:-$PROJECT_DIR/upstream-openpi}"
    # A git worktree does not share ignored virtualenvs with the primary
    # checkout. Reuse that checkout's pinned LIBERO client when available.
    if [[ -z "${LIBERO_OPENPI_DIR:-}" ]] && \
      [[ ! -x "$OPENPI_DIR/examples/libero/.venv/bin/python" ]]; then
      candidate="$PROJECT_DIR/../../embodied-policy-lab/upstream-openpi"
      if [[ -x "$candidate/examples/libero/.venv/bin/python" ]]; then
        OPENPI_DIR="$(cd "$candidate" && pwd)"
      fi
    fi
    LIBERO_OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_DIR/cache/openpi}"
    primary_openpi_cache="$(dirname "$OPENPI_DIR")/cache/openpi"
    if [[ ! -d "$LIBERO_OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" ]] && \
      [[ -d "$primary_openpi_cache/openpi-assets/checkpoints/pi05_libero" ]]; then
      LIBERO_OPENPI_DATA_HOME="$primary_openpi_cache"
    fi
    case "$MODEL" in
      pi05)
        RUNTIME_PYTHON="$OPENPI_DIR/.venv/bin/python"
        CLIENT_PYTHON="${LIBERO_CLIENT_PYTHON:-$OPENPI_DIR/examples/libero/.venv/bin/python}"
        ;;
      fastwam)
        if [[ -z "$FASTWAM_DIR" ]]; then
          for candidate in "$PROJECT_DIR/../upstream-fastwam" \
            "$PROJECT_DIR/../../upstream-fastwam"; do
            if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
              FASTWAM_DIR="$(cd "$candidate" && pwd)"
              break
            fi
          done
        fi
        if [[ -z "$FASTWAM_DIR" ]]; then
          echo "Cannot find upstream-fastwam; set FASTWAM_DIR explicitly." >&2
          exit 1
        fi
        RUNTIME_PYTHON="$FASTWAM_DIR/.venv/bin/python"
        CLIENT_PYTHON="${LIBERO_CLIENT_PYTHON:-$OPENPI_DIR/examples/libero/.venv/bin/python}"
        ;;
      *)
        echo "Model $MODEL does not support LIBERO; choose pi05 or fastwam." >&2
        exit 2
        ;;
    esac
    TASK_SUITE="${TASK_SUITE:-libero_spatial}"
    TASK_IDS="${TASK_IDS:-0}"
    WORLD_MODEL="${WORLD_MODEL:-none}"
    if [[ "$WORLD_MODEL" != "none" ]]; then
      echo "LIBERO currently supports only --world-model none." >&2
      exit 2
    fi
    ;;
  robocasa)
    case "$MODEL" in
      pi05)
        OPENPI_DIR="$PROJECT_DIR/upstream-robocasa-openpi"
        RUNTIME_PYTHON="$OPENPI_DIR/.venv/bin/python"
        CLIENT_PYTHON="$RUNTIME_PYTHON"
        ;;
      groot-n1.5)
        GROOT_DIR="$PROJECT_DIR/upstream-robocasa-groot"
        RUNTIME_PYTHON="$GROOT_DIR/.venv/bin/python"
        CLIENT_PYTHON="$RUNTIME_PYTHON"
        ;;
      *)
        echo "Unsupported model: $MODEL (choose pi05 or groot-n1.5)" >&2
        exit 2
        ;;
    esac
    TASK_SUITE="${TASK_SUITE:-${ROBOCASA_TASK_SET:-atomic_seen}}"
    TASK_IDS="${TASK_IDS:-${ROBOCASA_TASK_ID:-0}}"
    WORLD_MODEL="${WORLD_MODEL:-none}"
    if [[ "$INTERACTIVE" == "1" ]] && [[ ! "$TASK_IDS" =~ ^[0-9]+$ ]]; then
      echo "Interactive RoboCasa mode requires one numeric --task-id." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported backend: $BACKEND (choose libero or robocasa)" >&2
    exit 2
    ;;
esac

read -r DEFAULT_REPLAN_STEPS ACTION_HORIZON < <(
  python3 - "$PROJECT_DIR" "$BACKEND" "$MODEL" <<'PY'
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from showcase import backend_registry

profile = backend_registry.get_profile(sys.argv[2], sys.argv[3])
print(profile.default_replan_steps, profile.action_horizon)
PY
)
REPLAN_STEPS="${REPLAN_STEPS:-$DEFAULT_REPLAN_STEPS}"
if [[ "$MODEL" == "fastwam" ]]; then
  NUM_STEPS_WAIT=30
else
  NUM_STEPS_WAIT=10
fi

if [[ "$MODEL" == "groot-n1.5" ]]; then
  # Albumentations otherwise performs an online version check during import,
  # which would pollute the local-inference network audit.
  export NO_ALBUMENTATIONS_UPDATE=1
fi

for integer_value in "$TRIALS_PER_TASK" "$SEED" "$REPLAN_STEPS" "$MAX_POLICY_STEPS" "$POLICY_PORT" \
  "$DASHBOARD_PORT" "$REALTIME_DELAY_MS" "$VIEWER_WIDTH" "$VIEWER_HEIGHT"; do
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    echo "Expected a non-negative integer, got: $integer_value" >&2
    exit 2
  fi
done
if [[ "$TRIALS_PER_TASK" == "0" ]] || [[ "$REPLAN_STEPS" == "0" ]]; then
  echo "--trials and --replan-steps must be positive." >&2
  exit 2
fi
if (( REPLAN_STEPS > ACTION_HORIZON )); then
  echo "--replan-steps cannot exceed the $ACTION_HORIZON-action horizon for $BACKEND/$MODEL." >&2
  exit 2
fi
if [[ "$COMPARE_WORLD_MODEL" != "0" && "$COMPARE_WORLD_MODEL" != "1" ]]; then
  echo "COMPARE_WORLD_MODEL must be 0 or 1." >&2
  exit 2
fi
if [[ "$VIEWER_WIDTH" == "0" ]] || [[ "$VIEWER_HEIGHT" == "0" ]] || \
  [[ ! "$VIEWER_FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
  [[ "$VIEWER_FPS" == "0" ]] || [[ "$VIEWER_FPS" == "0.0" ]]; then
  echo "Viewer dimensions and FPS must be positive." >&2
  exit 2
fi
if [[ "$ROLLOUT_BUDGET_MULTIPLIER" != "0" ]] && \
  [[ ! "$ROLLOUT_BUDGET_MULTIPLIER" =~ ^[123]$ ]]; then
  echo "--budget must be 1, 2, or 3." >&2
  exit 2
fi
if [[ "$BACKEND" != "libero" ]] && [[ "$MAX_POLICY_STEPS" != "0" ]]; then
  echo "--max-policy-steps is currently a LIBERO-only test control." >&2
  exit 2
fi
if [[ -n "$EVALUATION_MODE" ]] && \
  [[ "$EVALUATION_MODE" != "scored" && "$EVALUATION_MODE" != "exploratory" ]]; then
  echo "--evaluation-mode must be scored or exploratory." >&2
  exit 2
fi
if [[ "$ROBOCASA_SPLIT" != "pretrain" && "$ROBOCASA_SPLIT" != "target" ]]; then
  echo "--split must be pretrain or target." >&2
  exit 2
fi
case "$BACKEND/$WORLD_MODEL" in
  libero/none|robocasa/none|robocasa/robocasa-sim) ;;
  *)
    echo "World model $WORLD_MODEL does not support backend $BACKEND." >&2
    exit 2
    ;;
esac
if [[ "$WORLD_MODEL" == "none" ]]; then
  COMPARE_WORLD_MODEL=0
fi
if [[ ! -x "$RUNTIME_PYTHON" ]] || [[ ! -x "$CLIENT_PYTHON" ]]; then
  if [[ "$MODEL" == "groot-n1.5" ]]; then
    echo "GR00T is not set up. Run ./scripts/setup_groot.sh first." >&2
  else
    echo "$BACKEND/$MODEL runtime is not set up; see the README setup instructions." >&2
  fi
  exit 1
fi

if [[ "$LOCAL_LLM_URL" == http://127.0.0.1:11434/* ]] && \
  [[ -n "$LOCAL_LLM_MODEL" ]] && command -v ollama >/dev/null 2>&1; then
  echo "Reserving GPU memory for the robot policy before loading the optional prompt model..."
  ollama stop "$LOCAL_LLM_MODEL" >/dev/null 2>&1 || true
fi

# One heavyweight session at a time is the safe default on a 24 GB GPU. This
# lock is independent from ALLOW_GPU_OVERSUBSCRIPTION so that flag cannot
# accidentally start a second copy of this lab and exhaust cuBLAS resources.
mkdir -p "$PROJECT_DIR/showcase-runs"
LAB_LOCK_PATH="$PROJECT_DIR/showcase-runs/.active-session.lock"
if [[ "${ALLOW_CONCURRENT_LAB_RUNS:-0}" != "1" ]]; then
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required for the single-session safety guard." >&2
    exit 1
  fi
  exec {LAB_LOCK_FD}>"$LAB_LOCK_PATH"
  if ! flock -n "$LAB_LOCK_FD"; then
    echo "Another Embodied Policy Lab session is already active:" >&2
    sed -n '1,4p' "$LAB_LOCK_PATH" >&2 2>/dev/null || true
    echo "Finish that session before launching another one." >&2
    echo "Expert override: ALLOW_CONCURRENT_LAB_RUNS=1 (requires separate ports and sufficient VRAM)." >&2
    exit 1
  fi
  printf 'pid=%s\nstarted=%s\nbackend=%s\nmodel=%s\n' \
    "$$" "$(date --iso-8601=seconds)" "$BACKEND" "$MODEL" > "$LAB_LOCK_PATH"
else
  echo "Warning: concurrent lab session guard explicitly disabled." >&2
fi

GPU_COMPUTE_APPS="$(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "$GPU_COMPUTE_APPS" && "${ALLOW_GPU_OVERSUBSCRIPTION:-0}" != "1" ]]; then
  echo "The GPU is already occupied by another compute process:" >&2
  echo "$GPU_COMPUTE_APPS" >&2
  echo "Local robot policies can use 21–22 GB on this 24 GB GPU." >&2
  echo "Stop the other workload, or set ALLOW_GPU_OVERSUBSCRIPTION=1 if intentional." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_DIR="${SESSION_DIR:-$PROJECT_DIR/showcase-runs/$TIMESTAMP}"
mkdir -p "$SESSION_DIR/frames" "$SESSION_DIR/videos" "$SESSION_DIR/previews" \
  "$PROJECT_DIR/showcase-runs"
ln -sfn "$SESSION_DIR" "$PROJECT_DIR/showcase-runs/latest"

cleanup() {
  for process_group in "${DASHBOARD_PID:-}" "${SERVER_PID:-}"; do
    if [[ -n "$process_group" ]]; then
      kill -- "-$process_group" 2>/dev/null || true
      wait "$process_group" 2>/dev/null || true
    fi
  done
  if [[ -n "${GPU_PID:-}" ]]; then
    kill "$GPU_PID" 2>/dev/null || true
    wait "$GPU_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

http_ready() {
  "$RUNTIME_PYTHON" - "$1" "$2" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{int(sys.argv[1])}{sys.argv[2]}", timeout=0.3
    ) as response:
        raise SystemExit(response.status != 200)
except (OSError, urllib.error.URLError):
    raise SystemExit(1)
PY
}

wait_for_http() {
  local port="$1"
  local route="$2"
  local owner_pid="$3"
  for _ in $(seq 1 900); do
    if ! kill -0 "$owner_pid" 2>/dev/null; then
      return 1
    fi
    if http_ready "$port" "$route"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

update_fastwam_startup_status() {
  local elapsed_seconds="$1"
  local detail
  if (( elapsed_seconds < 120 )); then
    detail="Deserializing the 12 GB Fast-WAM checkpoint on CPU - ${elapsed_seconds}s elapsed; typical startup is 90-120 seconds. GPU activity starts with the first policy request."
  else
    detail="Fast-WAM is still loading on CPU - ${elapsed_seconds}s elapsed. Competing CPU or RAM workloads can extend startup; the loader process is still active."
  fi
  echo "$detail"
  "$RUNTIME_PYTHON" - "$SESSION_DIR/state.json" "$elapsed_seconds" "$detail" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
elapsed_seconds = int(sys.argv[2])
detail = sys.argv[3]
state = json.loads(path.read_text(encoding="utf-8"))
if state.get("phase") != "initializing":
    raise SystemExit(0)
state["command_message"] = detail
state["startup_elapsed_seconds"] = elapsed_seconds
state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
temporary = path.with_suffix(".startup.tmp")
temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
temporary.replace(path)
PY
}

wait_for_policy_http() {
  local port="$1"
  local route="$2"
  local owner_pid="$3"
  local started_at=$SECONDS
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "$owner_pid" 2>/dev/null; then
      return 1
    fi
    if http_ready "$port" "$route"; then
      return 0
    fi
    if [[ "$MODEL" == "fastwam" ]] && (( attempt % 10 == 0 )); then
      update_fastwam_startup_status "$((SECONDS - started_at))"
    fi
    sleep 1
  done
  return 1
}

wait_for_tcp_listener() {
  local port="$1"
  local owner_pid="$2"
  for _ in $(seq 1 900); do
    if ! kill -0 "$owner_pid" 2>/dev/null; then
      return 1
    fi
    if ss -ltn "sport = :$port" | awk 'NR > 1 {found=1} END {exit !found}'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# Publish the selected profile before starting the heavyweight policy process so
# the browser can explain what is loading instead of showing an empty dashboard.
"$RUNTIME_PYTHON" - \
  "$PROJECT_DIR" "$SESSION_DIR/state.json" "$BACKEND" "$MODEL" "$TASK_SUITE" \
  "$TASK_IDS" "$POLICY_PORT" "$INTERACTIVE" "$NETWORK_AUDIT" \
  "$VIEWER_WIDTH" "$VIEWER_HEIGHT" "$WORLD_MODEL" \
  "$COMPARE_WORLD_MODEL" "$REPLAN_STEPS" <<'PY'
import datetime
import json
import pathlib
import sys

project_dir = pathlib.Path(sys.argv[1])
state_path = pathlib.Path(sys.argv[2])
backend, model, suite, task_ids = sys.argv[3:7]
port = int(sys.argv[7])
interactive = sys.argv[8] == "1"
network_audit = sys.argv[9] == "1"
viewer_width, viewer_height = map(int, sys.argv[10:12])
world_model_key = sys.argv[12]
compare_world_model = sys.argv[13] == "1"
replan_steps = int(sys.argv[14])
sys.path.insert(0, str(project_dir))

from showcase import backend_registry
from showcase import world_model_registry

simulator, policy = backend_registry.require_compatible(backend, model)
profile = backend_registry.get_profile(backend, model)
world_model = world_model_registry.require_world_model(backend, world_model_key)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
transport_scheme = {"websocket": "ws", "http": "http"}.get(
    policy.transport, "tcp"
)
try:
    task_id = int(task_ids)
except ValueError:
    task_id = 0

if model == "fastwam":
    startup_message = (
        "Deserializing the 12 GB Fast-WAM checkpoint on CPU; typical startup is "
        "90-120 seconds. An idle GPU is expected until the first policy request."
    )
else:
    startup_message = "Loading policy weights into local accelerator memory"

state = {
    "phase": "initializing",
    "message": "Loading the selected local policy and simulator",
    "command_message": startup_message,
    "startup_elapsed_seconds": 0,
    "backend": simulator.key,
    "simulator": simulator.simulator,
    "model_plugin": policy.key,
    "model": profile.model_name,
    "model_display_name": policy.display_name,
    "runtime": policy.runtime,
    "policy_transport": policy.transport,
    "policy_endpoint": f"{transport_scheme}://127.0.0.1:{port}",
    "world_model": world_model.key,
    "world_model_display_name": world_model.display_name,
    "world_model_runtime": world_model.runtime,
    "world_model_prediction_kind": world_model.prediction_kind,
    "world_model_description": world_model.description,
    "available_world_models": world_model_registry.catalog(backend),
    "preview_steps": replan_steps,
    "compare_world_model": compare_world_model,
    "comparison_status": "initializing" if compare_world_model else "disabled",
    "suite": suite,
    "task_ids": task_ids,
    "task_id": task_id,
    "interactive": interactive,
    "network_audit": network_audit,
    "state_dimension": simulator.state_dimension,
    "action_dimension": simulator.action_dimension,
    "action_horizon": profile.action_horizon,
    "camera_count": len(simulator.cameras),
    "model_image_width": 224,
    "model_image_height": 224,
    "viewer_width": viewer_width,
    "viewer_height": viewer_height,
    "episodes": 0,
    "successes": 0,
    "started_at": now,
    "updated_at": now,
}
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
PY

DASHBOARD_COMMAND=(
  "$RUNTIME_PYTHON" "$PROJECT_DIR/showcase/dashboard_server.py"
  --session-dir "$SESSION_DIR"
  --static-dir "$PROJECT_DIR/showcase/static"
  --port "$DASHBOARD_PORT"
)
if [[ -n "$LOCAL_LLM_URL" || -n "$LOCAL_LLM_MODEL" ]]; then
  if [[ -z "$LOCAL_LLM_URL" || -z "$LOCAL_LLM_MODEL" ]]; then
    echo "LOCAL_LLM_URL and LOCAL_LLM_MODEL must be set together." >&2
    exit 2
  fi
  DASHBOARD_COMMAND+=(
    --local-llm-url "$LOCAL_LLM_URL"
    --local-llm-model "$LOCAL_LLM_MODEL"
    --local-llm-num-gpu "$LOCAL_LLM_NUM_GPU"
  )
fi
setsid "${DASHBOARD_COMMAND[@]}" > "$SESSION_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!

if ! wait_for_http "$DASHBOARD_PORT" /api/state "$DASHBOARD_PID"; then
  echo "Dashboard failed. See $SESSION_DIR/dashboard.log" >&2
  exit 1
fi

DASHBOARD_URL="http://127.0.0.1:$DASHBOARD_PORT"
echo "Dashboard: $DASHBOARD_URL"
echo "Session: $SESSION_DIR"
echo "Backend: $BACKEND · Policy: $MODEL · Predictor: $WORLD_MODEL · Task collection: $TASK_SUITE · Task ID: $TASK_IDS"
echo "Loading the local policy; startup progress is now visible in the dashboard."
if [[ "$INTERACTIVE" == "1" ]]; then
  echo "Interactive mode: use the browser to run attempts and end the session."
fi
if [[ "$AUTO_OPEN" == "1" ]] && [[ -n "${DISPLAY:-}" ]]; then
  xdg-open "$DASHBOARD_URL" >/dev/null 2>&1 || true
fi

echo "Starting local $MODEL server for $BACKEND..."
if [[ "$NETWORK_AUDIT" == "1" ]]; then
  setsid strace -f -e trace=network -s 256 -o "$SESSION_DIR/network-server.log" \
    env BACKEND="$BACKEND" MODEL="$MODEL" POLICY_PORT="$POLICY_PORT" \
      LIBERO_OPENPI_DIR="${OPENPI_DIR:-}" \
      OPENPI_DATA_HOME="${LIBERO_OPENPI_DATA_HOME:-}" \
      FASTWAM_DIR="$FASTWAM_DIR" FASTWAM_ARTIFACT_DIR="$SESSION_DIR/policy-inference" \
      "$SCRIPT_DIR/run_server.sh" \
    > "$SESSION_DIR/server.log" 2>&1 &
else
  setsid env BACKEND="$BACKEND" MODEL="$MODEL" POLICY_PORT="$POLICY_PORT" \
    LIBERO_OPENPI_DIR="${OPENPI_DIR:-}" \
    OPENPI_DATA_HOME="${LIBERO_OPENPI_DATA_HOME:-}" \
    FASTWAM_DIR="$FASTWAM_DIR" FASTWAM_ARTIFACT_DIR="$SESSION_DIR/policy-inference" \
    "$SCRIPT_DIR/run_server.sh" \
    > "$SESSION_DIR/server.log" 2>&1 &
fi
SERVER_PID=$!

if [[ "$MODEL" == "groot-n1.5" ]]; then
  POLICY_READY=0
  wait_for_tcp_listener "$POLICY_PORT" "$SERVER_PID" || POLICY_READY=$?
else
  POLICY_READY=0
  wait_for_policy_http "$POLICY_PORT" /healthz "$SERVER_PID" || POLICY_READY=$?
fi
if [[ "$POLICY_READY" != "0" ]]; then
  echo "Policy server failed. See $SESSION_DIR/server.log" >&2
  tail -100 "$SESSION_DIR/server.log" >&2
  exit 1
fi

"$RUNTIME_PYTHON" - "$SESSION_DIR/state.json" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state["command_message"] = "Policy ready; loading the simulator and task catalog"
state["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
temporary.replace(path)
PY

echo "timestamp,name,memory.used [MiB],utilization.gpu [%],power.draw [W],temperature.gpu" > "$SESSION_DIR/gpu.csv"
(
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$SESSION_DIR/gpu.csv"
    sleep 1
  done
) &
GPU_PID=$!

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ "$BACKEND" == "libero" ]]; then
  export LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero"
  export PYTHONPATH="$OPENPI_DIR/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
  if [[ "$INTERACTIVE" == "1" ]]; then
    if [[ ! "$TASK_IDS" =~ ^[0-9]+$ ]]; then
      echo "Interactive LIBERO mode requires one numeric --task-id." >&2
      exit 2
    fi
    CLIENT_COMMAND=(
      "$CLIENT_PYTHON" "$PROJECT_DIR/showcase/interactive_libero.py"
      --model "$MODEL"
      --host 127.0.0.1
      --port "$POLICY_PORT"
      --replan-steps "$REPLAN_STEPS"
      --num-steps-wait "$NUM_STEPS_WAIT"
      --max-policy-steps "$MAX_POLICY_STEPS"
      --task-suite-name "$TASK_SUITE"
      --task-id "$TASK_IDS"
      --video-out-path "$SESSION_DIR/videos"
      --session-dir "$SESSION_DIR"
      --seed "$SEED"
      --realtime-delay-ms "$REALTIME_DELAY_MS"
      --initial-prompt "$INITIAL_PROMPT"
    )
  else
    CLIENT_COMMAND=(
      "$CLIENT_PYTHON" "$PROJECT_DIR/showcase/instrumented_libero.py"
      --model "$MODEL"
      --host 127.0.0.1
      --port "$POLICY_PORT"
      --replan-steps "$REPLAN_STEPS"
      --num-steps-wait "$NUM_STEPS_WAIT"
      --max-policy-steps "$MAX_POLICY_STEPS"
      --task-suite-name "$TASK_SUITE"
      --task-ids "$TASK_IDS"
      --num-trials-per-task "$TRIALS_PER_TASK"
      --video-out-path "$SESSION_DIR/videos"
      --session-dir "$SESSION_DIR"
      --seed "$SEED"
      --realtime-delay-ms "$REALTIME_DELAY_MS"
    )
  fi
else
  CLIENT_COMMAND=(
    "$CLIENT_PYTHON" "$PROJECT_DIR/showcase/interactive_robocasa.py"
    --model "$MODEL"
    --world-model "$WORLD_MODEL"
    --host 127.0.0.1
    --port "$POLICY_PORT"
    --replan-steps "$REPLAN_STEPS"
    --viewer-width "$VIEWER_WIDTH"
    --viewer-height "$VIEWER_HEIGHT"
    --viewer-fps "$VIEWER_FPS"
    --task-set-name "$TASK_SUITE"
    --split "$ROBOCASA_SPLIT"
    --video-out-path "$SESSION_DIR/videos"
    --session-dir "$SESSION_DIR"
    --seed "$SEED"
    --realtime-delay-ms "$REALTIME_DELAY_MS"
    --initial-prompt "$INITIAL_PROMPT"
    --initial-evaluation-mode "$EVALUATION_MODE"
    --initial-rollout-budget-multiplier "$ROLLOUT_BUDGET_MULTIPLIER"
  )
  if [[ "$COMPARE_WORLD_MODEL" == "1" ]]; then
    CLIENT_COMMAND+=(--compare-world-model)
  else
    CLIENT_COMMAND+=(--no-compare-world-model)
  fi
  if [[ "$INTERACTIVE" == "1" ]]; then
    CLIENT_COMMAND+=(--interactive --task-id "$TASK_IDS")
    if [[ "$AUTO_START" == "1" ]]; then
      CLIENT_COMMAND+=(--auto-start)
    fi
  else
    CLIENT_COMMAND+=(
      --no-interactive
      --auto-start
      --task-ids "$TASK_IDS"
      --num-trials-per-task "$TRIALS_PER_TASK"
    )
  fi
fi
if [[ "$NETWORK_AUDIT" == "1" ]]; then
  CLIENT_COMMAND+=(--network-audit)
else
  CLIENT_COMMAND+=(--no-network-audit)
fi

set +e
if [[ "$NETWORK_AUDIT" == "1" ]]; then
  strace -f -e trace=network -s 256 -o "$SESSION_DIR/network-client.log" \
    "${CLIENT_COMMAND[@]}" 2>&1 | tee "$SESSION_DIR/client.log"
  CLIENT_STATUS=${PIPESTATUS[0]}
else
  "${CLIENT_COMMAND[@]}" 2>&1 | tee "$SESSION_DIR/client.log"
  CLIENT_STATUS=${PIPESTATUS[0]}
fi
set -e

# If the client was interrupted outside its normal shutdown path, do not leave
# the browser/report claiming that a dead rollout is still running.
"$RUNTIME_PYTHON" - "$SESSION_DIR/state.json" "$CLIENT_STATUS" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = int(sys.argv[2])
state = json.loads(path.read_text(encoding="utf-8"))
if state.get("phase") in {"waiting", "initializing", "preparing_task", "running"}:
    interrupted = status in {130, 143}
    clean_exit = status == 0
    state["phase"] = "stopped" if interrupted or clean_exit else "error"
    state["stop_reason"] = (
        "interrupted"
        if interrupted
        else ("client_exit_without_finalize" if clean_exit else "client_exit")
    )
    state["command_message"] = (
        "Session interrupted; partial artifacts were saved"
        if interrupted
        else (
            "Simulator client ended; partial artifacts were saved"
            if clean_exit
            else f"Simulator client exited with status {status}"
        )
    )
    if not interrupted and not clean_exit:
        state["error"] = f"Simulator client exited with status {status}"
    state["finished_at"] = datetime.datetime.now(
        datetime.timezone.utc
    ).astimezone().isoformat()
    state["updated_at"] = state["finished_at"]
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)
PY

kill -- "-$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
kill "$GPU_PID" 2>/dev/null || true
wait "$GPU_PID" 2>/dev/null || true
GPU_PID=""

if [[ -f "$SESSION_DIR/state.json" ]]; then
  "$RUNTIME_PYTHON" "$PROJECT_DIR/showcase/generate_report.py" "$SESSION_DIR"
fi
if [[ "$CLIENT_STATUS" == "0" ]]; then
  echo "Session saved successfully."
else
  echo "Simulator exited with status $CLIENT_STATUS." >&2
fi
echo "Report: $SESSION_DIR/report.md"
echo "Dashboard snapshot data: $SESSION_DIR/state.json"

if [[ "$HOLD_OPEN" == "1" ]]; then
  echo "Dashboard is now in review mode at $DASHBOARD_URL."
  echo "Press Ctrl+C here when you are finished reviewing the results."
  wait "$DASHBOARD_PID"
fi

exit "$CLIENT_STATUS"

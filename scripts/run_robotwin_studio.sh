#!/usr/bin/env bash
# Launch the shared browser studio around a native RoboTwin policy adapter.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

die() { echo "run_robotwin_studio: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: scripts/run_robotwin_studio.sh [OPTIONS]

Options:
  --model NAME          fastwam (default) or flexpi
  --task NAME           initial RoboTwin task (default: click_bell)
  --phase NAME          demo_clean (default) or demo_randomized
  --flexpi-mode MODE    full-joint (default) or action-only
  --replan-steps COUNT  executed actions per model request (profile default)
  --seed INTEGER        reproducible upstream seed offset (default: 0)
  --gpu ID              CUDA device index (default: 0)
  --dashboard-port PORT loopback browser port (default: 8085)
  --realtime-delay-ms N optional delay after each simulator action (default: 35)
  --auto-start          prepare and run the initial task after model load
  --open / --no-open    enable or disable automatic browser opening
  --hold-open / --no-hold-open
                        keep the dashboard available after the adapter exits
  --network-audit / --no-network-audit
  --session-dir PATH    explicit artifact directory
  --dry-run             verify paths and print the native adapter command
  -h, --help            show this help

The browser opens before the model is loaded and reports startup progress.
Task execution begins only after Start is pressed. RoboTwin's head, left-wrist,
and right-wrist RGB observations are displayed without changing orientation or
mapping them onto the LIBERO action schema.
EOF
}

find_checkout() {
  local override_name="$1" directory_name="$2" override_value="" candidate
  override_value="${!override_name:-}"
  if [[ -n "$override_value" ]]; then
    printf '%s\n' "$override_value"
    return
  fi
  for candidate in "$PROJECT_DIR/../$directory_name" "$PROJECT_DIR/../../$directory_name"; do
    if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
      (cd "$candidate" && pwd)
      return
    fi
  done
  return 1
}

model="fastwam"
task="click_bell"
phase="demo_clean"
flexpi_mode="full-joint"
replan_steps=""
seed="0"
gpu_id="0"
dashboard_port="8085"
realtime_delay_ms="35"
auto_start=0
auto_open=1
hold_open=1
network_audit=1
session_dir=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="${2:?--model requires a value}"; shift 2 ;;
    --task|--task-id) task="${2:?$1 requires a value}"; shift 2 ;;
    --phase|--task-set|--task-suite) phase="${2:?$1 requires a value}"; shift 2 ;;
    --flexpi-mode) flexpi_mode="${2:?--flexpi-mode requires a value}"; shift 2 ;;
    --replan-steps) replan_steps="${2:?--replan-steps requires a value}"; shift 2 ;;
    --seed) seed="${2:?--seed requires a value}"; shift 2 ;;
    --gpu) gpu_id="${2:?--gpu requires a value}"; shift 2 ;;
    --dashboard-port) dashboard_port="${2:?--dashboard-port requires a value}"; shift 2 ;;
    --realtime-delay-ms) realtime_delay_ms="${2:?--realtime-delay-ms requires a value}"; shift 2 ;;
    --auto-start) auto_start=1; shift ;;
    --open) auto_open=1; shift ;;
    --no-open) auto_open=0; shift ;;
    --hold-open) hold_open=1; shift ;;
    --no-hold-open) hold_open=0; shift ;;
    --network-audit) network_audit=1; shift ;;
    --no-network-audit) network_audit=0; shift ;;
    --session-dir) session_dir="${2:?--session-dir requires a value}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$model" == "fastwam" || "$model" == "flexpi" ]] || die "--model must be fastwam or flexpi"
[[ "$phase" == "demo_clean" || "$phase" == "demo_randomized" ]] || die "--phase must be demo_clean or demo_randomized"
[[ "$task" =~ ^[a-z0-9_]+$ ]] || die "--task must be a snake_case RoboTwin task"
[[ "$seed" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
for value in "$gpu_id" "$dashboard_port" "$realtime_delay_ms"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "expected a non-negative integer, got: $value"
done
if [[ "$model" == "flexpi" ]]; then
  [[ "$flexpi_mode" == "full-joint" || "$flexpi_mode" == "action-only" ]] || die "--flexpi-mode must be full-joint or action-only"
else
  [[ "$flexpi_mode" == "full-joint" ]] || die "--flexpi-mode applies only to --model flexpi"
  flexpi_mode="action-only"
fi

if [[ "$model" == "fastwam" ]]; then
  model_root="$(find_checkout FASTWAM_DIR upstream-fastwam || true)"
  [[ -n "$model_root" ]] || die "Fast-WAM checkout missing; run scripts/setup_robotwin.sh --model fastwam"
  checkpoint="$model_root/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
  statistics="$model_root/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
  task_profile="robotwin_uncond_3cam_384_1e-4"
  replan_steps="${replan_steps:-24}"
else
  model_root="$(find_checkout FLEXPI_DIR upstream-flexpi || true)"
  [[ -n "$model_root" ]] || die "Flex-π checkout missing; run scripts/setup_robotwin.sh --model flexpi"
  checkpoint="$model_root/runs/flexpi-robotwin/checkpoints/weights/step_048060.pt"
  statistics="$model_root/runs/flexpi-robotwin/dataset_stats.json"
  task_profile="robotwin_unified_flex_3cam_384_1e-4"
  replan_steps="${replan_steps:-32}"
fi
runtime_python="$model_root/.venv-robotwin/bin/python"
robotwin_root="$model_root/third_party/RoboTwin"

[[ "$replan_steps" =~ ^[1-9][0-9]*$ ]] || die "--replan-steps must be positive"
(( replan_steps <= 32 )) || die "--replan-steps cannot exceed the 32-action horizon"
for required in "$runtime_python" "$checkpoint" "$statistics" \
  "$robotwin_root/task_config/$phase.yml" "$robotwin_root/task_config/_eval_step_limit.yml" \
  "$robotwin_root/envs/$task.py" "$robotwin_root/assets/embodiments"; do
  [[ -e "$required" ]] || die "missing prerequisite: $required (run scripts/setup_robotwin.sh --check)"
done
session_dir="${session_dir:-$PROJECT_DIR/showcase-runs/$(date +%Y%m%d-%H%M%S)-robotwin}"
runtime_checkpoint="$checkpoint"
if [[ "$model" == "flexpi" ]]; then
  runtime_checkpoint="$session_dir/flexpi-inference-release/checkpoints/weights/$(basename "$checkpoint")"
fi
runner=(
  "$runtime_python" "$PROJECT_DIR/showcase/interactive_robotwin.py"
  --model "$model"
  --model-root "$model_root"
  --robotwin-root "$robotwin_root"
  --checkpoint "$runtime_checkpoint"
  --dataset-stats "$statistics"
  --task-profile "$task_profile"
  --task "$task"
  --phase "$phase"
  --session-dir "$session_dir"
  --replan-steps "$replan_steps"
  --seed "$seed"
  --realtime-delay-ms "$realtime_delay_ms"
  --flexpi-mode "$flexpi_mode"
)
[[ "$auto_start" == "1" ]] && runner+=(--auto-start)
[[ "$network_audit" == "1" ]] && runner+=(--network-audit) || runner+=(--no-network-audit)

printf 'RoboTwin browser studio\n'
printf '  model: %s · mode: %s\n  task: %s · phase: %s\n' "$model" "$flexpi_mode" "$task" "$phase"
printf '  contract: head + left wrist + right wrist · 14D qpos · horizon 32 · replan %s\n' "$replan_steps"
printf '  session: %s\n' "$session_dir"
printf 'Command:'
printf ' %q' "${runner[@]}"
printf '\n'
[[ "$dry_run" == "1" ]] && exit 0

"$runtime_python" -W ignore::UserWarning -c "import pkg_resources, sapien" \
  >/dev/null 2>&1 \
  || die "RoboTwin SAPIEN runtime is incompatible; rerun scripts/setup_robotwin.sh --model $model"

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is required for rollout artifacts"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for local GPU telemetry"
if [[ "$network_audit" == "1" ]]; then
  command -v strace >/dev/null 2>&1 || die "strace is required with --network-audit"
fi

mkdir -p "$session_dir/frames" "$session_dir/videos" "$session_dir/controls" "$PROJECT_DIR/showcase-runs"
ln -sfn "$session_dir" "$PROJECT_DIR/showcase-runs/latest"
if [[ "$model" == "flexpi" ]]; then
  prepared_checkpoint="$(
    "$runtime_python" "$PROJECT_DIR/showcase/prepare_flexpi_inference_release.py" \
      --checkpoint "$checkpoint" \
      --config "$model_root/runs/flexpi-robotwin/config.yaml" \
      --destination "$session_dir/flexpi-inference-release"
  )"
  [[ "$prepared_checkpoint" == "$runtime_checkpoint" ]] \
    || die "unexpected Flex-π runtime checkpoint path: $prepared_checkpoint"
fi

mkdir -p "$PROJECT_DIR/showcase-runs"
lock_path="$PROJECT_DIR/showcase-runs/.active-session.lock"
if [[ "${ALLOW_CONCURRENT_LAB_RUNS:-0}" != "1" ]]; then
  exec {lock_fd}>"$lock_path"
  flock -n "$lock_fd" || die "another Embodied Policy Lab session is active"
  printf 'pid=%s\nstarted=%s\nbackend=robotwin\nmodel=%s\n' \
    "$$" "$(date --iso-8601=seconds)" "$model" > "$lock_path"
fi

cleanup() {
  for process in "${dashboard_pid:-}" "${gpu_pid:-}"; do
    [[ -z "$process" ]] || kill "$process" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

python3 - "$session_dir/state.json" "$model" "$phase" "$task" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
model, phase, task = sys.argv[2:5]
display = "Fast-WAM" if model == "fastwam" else "Flex-π"
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
path.write_text(json.dumps({
    "phase": "initializing",
    "interactive": True,
    "backend": "robotwin",
    "simulator": "RoboTwin 2.0 / SAPIEN / Vulkan",
    "model_display_name": display,
    "model": model,
    "suite": phase,
    "task_id": task,
    "policy_transport": "in-process native",
    "policy_endpoint": f"in-process://robotwin/{model}",
    "camera_count": 3,
    "command_message": "Launching the isolated RoboTwin policy runtime",
    "started_at": now,
    "updated_at": now,
}, indent=2), encoding="utf-8")
PY

python3 "$PROJECT_DIR/showcase/dashboard_server.py" \
  --session-dir "$session_dir" --static-dir "$PROJECT_DIR/showcase/static" \
  --port "$dashboard_port" > "$session_dir/dashboard.log" 2>&1 &
dashboard_pid=$!

dashboard_url="http://127.0.0.1:$dashboard_port"
for _ in $(seq 1 100); do
  curl -fsS "$dashboard_url/api/state" >/dev/null 2>&1 && break
  kill -0 "$dashboard_pid" 2>/dev/null || die "dashboard failed; see $session_dir/dashboard.log"
  sleep 0.1
done
echo "Dashboard: $dashboard_url"
if [[ "$auto_open" == "1" && -n "${DISPLAY:-}" ]]; then
  xdg-open "$dashboard_url" >/dev/null 2>&1 || true
fi

echo "timestamp,name,memory.used [MiB],utilization.gpu [%],power.draw [W],temperature.gpu" > "$session_dir/gpu.csv"
(
  while :; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$session_dir/gpu.csv" 2>/dev/null || true
    sleep 1
  done
) &
gpu_pid=$!

export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$model_root/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

set +e
if [[ "$network_audit" == "1" ]]; then
  strace -f -e trace=network -s 256 -o "$session_dir/network-client.log" \
    "${runner[@]}" 2>&1 | tee "$session_dir/client.log"
  status=${PIPESTATUS[0]}
else
  "${runner[@]}" 2>&1 | tee "$session_dir/client.log"
  status=${PIPESTATUS[0]}
fi
set -e

kill "$gpu_pid" 2>/dev/null || true
wait "$gpu_pid" 2>/dev/null || true
gpu_pid=""
python3 "$PROJECT_DIR/showcase/generate_report.py" "$session_dir"
echo "Report: $session_dir/report.md"

if [[ "$status" != "0" ]]; then
  echo "RoboTwin studio adapter failed. See $session_dir/client.log" >&2
  exit "$status"
fi
if [[ "$hold_open" == "1" ]]; then
  echo "Session complete. Dashboard remains open for review; press Ctrl-C to close it."
  wait "$dashboard_pid"
fi

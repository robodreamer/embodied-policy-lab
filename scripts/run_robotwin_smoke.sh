#!/usr/bin/env bash
# Exercise RoboTwin reset -> observations -> one 14D qpos no-op without a model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

task="click_bell"
phase="demo_clean"
seed="0"
runtime_model=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="${2:?--task requires a value}"; shift 2 ;;
    --phase) phase="${2:?--phase requires a value}"; shift 2 ;;
    --seed) seed="${2:?--seed requires a value}"; shift 2 ;;
    --runtime-model) runtime_model="${2:?--runtime-model requires a value}"; shift 2 ;;
    --output) output="${2:?--output requires a value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--task click_bell] [--phase demo_clean|demo_randomized] [--seed 0] [--runtime-model fastwam|flexpi] [--output FILE]"
      exit 0 ;;
    *) echo "run_robotwin_smoke: unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$task" =~ ^[a-z0-9_]+$ ]] || { echo "invalid snake_case task: $task" >&2; exit 2; }
[[ "$phase" == "demo_clean" || "$phase" == "demo_randomized" ]] \
  || { echo "phase must be demo_clean or demo_randomized" >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }
[[ -z "$runtime_model" || "$runtime_model" == "fastwam" || "$runtime_model" == "flexpi" ]] \
  || { echo "runtime model must be fastwam or flexpi" >&2; exit 2; }

find_root() {
  local override_name="$1" name="$2" candidate override=""
  override="${!override_name:-}"
  if [[ -n "$override" ]]; then printf '%s\n' "$override"; return; fi
  for candidate in "$PROJECT_DIR/../$name" "$PROJECT_DIR/../../$name"; do
    if [[ -d "$candidate" ]]; then (cd "$candidate" && pwd); return; fi
  done
  return 1
}

fastwam_root="$(find_root FASTWAM_DIR upstream-fastwam || true)"
flexpi_root="$(find_root FLEXPI_DIR upstream-flexpi || true)"
if [[ -z "$runtime_model" ]]; then
  if [[ -x "$fastwam_root/.venv-robotwin/bin/python" ]]; then
    runtime_model="fastwam"
  elif [[ -x "$flexpi_root/.venv-robotwin/bin/python" ]]; then
    runtime_model="flexpi"
  else
    echo "No isolated RoboTwin runtime found; run scripts/setup_robotwin.sh first." >&2
    exit 1
  fi
fi

if [[ "$runtime_model" == "fastwam" ]]; then
  model_root="$fastwam_root"
else
  model_root="$flexpi_root"
fi
[[ -n "$model_root" ]] || { echo "$runtime_model checkout not found" >&2; exit 1; }
python="$model_root/.venv-robotwin/bin/python"
robotwin_root="$model_root/third_party/RoboTwin"
[[ -x "$python" ]] || { echo "missing runtime: $python" >&2; exit 1; }
[[ -e "$robotwin_root/assets/embodiments" ]] \
  || { echo "missing RoboTwin assets; rerun setup with --download-assets" >&2; exit 1; }

if [[ -z "$output" ]]; then
  output="$PROJECT_DIR/results/robotwin-smoke/$(date +%Y%m%d-%H%M%S)/result.json"
fi

export PYTHONNOUSERSITE=1
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
exec "$python" "$PROJECT_DIR/showcase/robotwin_smoke.py" \
  --robotwin-root "$robotwin_root" --task "$task" --phase "$phase" \
  --seed "$seed" --output "$output"

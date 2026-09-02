#!/usr/bin/env bash
# Run one bounded RoboTwin task through the upstream model-native evaluator.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

die() { echo "run_robotwin_evaluation: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: scripts/run_robotwin_evaluation.sh [OPTIONS]

Options:
  --model NAME          fastwam (default) or flexpi
  --task NAME           RoboTwin snake_case task (default: click_bell)
  --phase NAME          demo_clean (default) or demo_randomized
  --trials COUNT        accepted evaluation episodes (default: 1)
  --seed INTEGER        upstream seed offset (default: 0)
  --replan-steps COUNT  executed actions per model call (profile default)
  --flexpi-mode MODE    full-joint (default) or action-only
  --gpu ID              CUDA device index (default: 0)
  --fast-render         skip intermediate RGB renders inside an action prefix
  --dry-run             verify paths and print the exact command
  -h, --help            show this help

RoboTwin assets, checkpoints, and the isolated runtime are prepared by
scripts/setup_robotwin.sh. The evaluator writes native evidence beneath the
selected model checkout's evaluate_results/robotwin directory.
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
trials="1"
seed="0"
replan_steps=""
flexpi_mode="full-joint"
gpu_id="0"
skip_get_obs="false"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) model="${2:?--model requires a value}"; shift 2 ;;
    --task) task="${2:?--task requires a value}"; shift 2 ;;
    --phase) phase="${2:?--phase requires a value}"; shift 2 ;;
    --trials) trials="${2:?--trials requires a value}"; shift 2 ;;
    --seed) seed="${2:?--seed requires a value}"; shift 2 ;;
    --replan-steps) replan_steps="${2:?--replan-steps requires a value}"; shift 2 ;;
    --flexpi-mode) flexpi_mode="${2:?--flexpi-mode requires a value}"; shift 2 ;;
    --gpu) gpu_id="${2:?--gpu requires a value}"; shift 2 ;;
    --fast-render) skip_get_obs="true"; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$model" == "fastwam" || "$model" == "flexpi" ]] \
  || die "--model must be fastwam or flexpi"
[[ "$phase" == "demo_clean" || "$phase" == "demo_randomized" ]] \
  || die "--phase must be demo_clean or demo_randomized"
[[ "$task" =~ ^[a-z0-9_]+$ ]] || die "--task must be a snake_case task name"
[[ "$trials" =~ ^[1-9][0-9]*$ ]] || die "--trials must be a positive integer"
[[ "$seed" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
[[ "$gpu_id" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
if [[ -n "$replan_steps" ]]; then
  [[ "$replan_steps" =~ ^[1-9][0-9]*$ ]] \
    || die "--replan-steps must be a positive integer"
  (( replan_steps <= 32 )) || die "--replan-steps cannot exceed the 32-action horizon"
fi
if [[ "$model" == "flexpi" ]]; then
  [[ "$flexpi_mode" == "full-joint" || "$flexpi_mode" == "action-only" ]] \
    || die "--flexpi-mode must be full-joint or action-only"
elif [[ "$flexpi_mode" != "full-joint" ]]; then
  die "--flexpi-mode applies only to --model flexpi"
fi

if [[ "$model" == "fastwam" ]]; then
  model_root="$(find_checkout FASTWAM_DIR upstream-fastwam || true)"
  [[ -n "$model_root" ]] || die "Fast-WAM checkout missing; run scripts/setup_robotwin.sh --model fastwam"
  runtime_python="$model_root/.venv-robotwin/bin/python"
  robotwin_root="$model_root/third_party/RoboTwin"
  entrypoint="$model_root/experiments/robotwin/eval_robotwin_single.py"
  checkpoint="$model_root/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
  statistics="$model_root/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
  task_profile="robotwin_uncond_3cam_384_1e-4"
  output_tag="robotwin_uncond_3cam_384"
  replan_steps="${replan_steps:-24}"
else
  model_root="$(find_checkout FLEXPI_DIR upstream-flexpi || true)"
  [[ -n "$model_root" ]] || die "Flex-π checkout missing; run scripts/setup_robotwin.sh --model flexpi"
  runtime_python="$model_root/.venv-robotwin/bin/python"
  robotwin_root="$model_root/third_party/RoboTwin"
  entrypoint="$model_root/experiments/robotwin/eval_robotwin_single.py"
  checkpoint="$model_root/runs/flexpi-robotwin/checkpoints/weights/step_048060.pt"
  statistics="$model_root/runs/flexpi-robotwin/dataset_stats.json"
  task_profile="robotwin_unified_flex_3cam_384_1e-4"
  output_tag="flexpi-robotwin_checkpoints"
  replan_steps="${replan_steps:-32}"
fi

for required in "$runtime_python" "$entrypoint" "$checkpoint" "$statistics" \
  "$robotwin_root/task_config/$phase.yml" "$robotwin_root/envs/$task.py" \
  "$robotwin_root/assets/embodiments"; do
  [[ -e "$required" ]] || die "missing prerequisite: $required (run scripts/setup_robotwin.sh --check)"
done
run_stamp="$(date +%Y%m%d-%H%M%S)"
output_hint="$model_root/evaluate_results/robotwin/$output_tag/$run_stamp"
runtime_checkpoint="$checkpoint"
if [[ "$model" == "flexpi" ]]; then
  runtime_checkpoint="$output_hint/flexpi-inference-release/checkpoints/weights/$(basename "$checkpoint")"
fi
command=(
  "$runtime_python" "$entrypoint"
  "task=$task_profile"
  "ckpt=$runtime_checkpoint"
  "EVALUATION.robotwin_root=$robotwin_root"
  "EVALUATION.dataset_stats_path=$statistics"
  "EVALUATION.task_name=$task"
  "EVALUATION.task_config=$phase"
  "EVALUATION.eval_num_episodes=$trials"
  "EVALUATION.replan_steps=$replan_steps"
  "EVALUATION.skip_get_obs_within_replan=$skip_get_obs"
  "EVALUATION.output_dir=$run_stamp"
  "seed=$seed"
  "gpu_id=$gpu_id"
)

if [[ "$model" == "flexpi" ]]; then
  joint="true"
  [[ "$flexpi_mode" == "action-only" ]] && joint="false"
  command+=(
    "+EVALUATION.infer_joint_video=$joint"
    "+EVALUATION.infer_joint_dino=$joint"
    "+EVALUATION.infer_joint_pointmap=$joint"
    "EVALUATION.offload_text_encoder=${FLEXPI_OFFLOAD_TEXT_ENCODER:-true}"
  )
fi

printf 'RoboTwin 2.0 native evaluation\n'
printf '  model: %s\n  task: %s\n  phase: %s\n  trials: %s\n' \
  "$model" "$task" "$phase" "$trials"
printf '  action contract: 14D ALOHA-AgileX qpos · horizon 32 · replan %s\n' "$replan_steps"
printf '  cameras: head + left wrist + right wrist\n'
printf '  evidence: %s\n' "$output_hint"

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$dry_run" == "1" ]]; then
  exit 0
fi

"$runtime_python" -W ignore::UserWarning -c \
  "from importlib.metadata import version; assert version('nvidia-curobo') == '0.7.8'; assert version('warp-lang') == '1.12.0'; import pkg_resources, sapien, warp as wp, curobo.types.math, curobo.types.robot; assert hasattr(wp, 'torch')" \
  >/dev/null 2>&1 \
  || die "RoboTwin simulator runtime is incompatible; rerun scripts/setup_robotwin.sh --model $model"

if [[ "$model" == "flexpi" ]]; then
  prepared_checkpoint="$(
    "$runtime_python" "$PROJECT_DIR/showcase/prepare_flexpi_inference_release.py" \
      --checkpoint "$checkpoint" \
      --config "$model_root/runs/flexpi-robotwin/config.yaml" \
      --destination "$output_hint/flexpi-inference-release"
  )"
  [[ "$prepared_checkpoint" == "$runtime_checkpoint" ]] \
    || die "unexpected Flex-π runtime checkpoint path: $prepared_checkpoint"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
if [[ "$model" == "flexpi" ]]; then
  export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$model_root/checkpoints}"
  export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
fi

cd "$model_root"
exec "${command[@]}"

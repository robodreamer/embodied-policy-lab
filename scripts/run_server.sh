#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BACKEND="${BACKEND:-libero}"
MODEL="${MODEL:-pi05}"
PORT="${POLICY_PORT:-${PI05_PORT:-8000}}"

if [[ "$BACKEND/$MODEL" == "robocasa/pi05" ]]; then
  # Leave headroom for RoboCasa's EGL presentation context and cuBLAS startup.
  # JAX's former 0.75 default could leave the 24 GB validation GPU effectively
  # full before the first jitted inference initialized its BLAS handle.
  DEFAULT_XLA_MEM_FRACTION=0.70
else
  DEFAULT_XLA_MEM_FRACTION=0.75
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-$DEFAULT_XLA_MEM_FRACTION}"

case "$BACKEND" in
  libero)
    case "$MODEL" in
      pi05)
        OPENPI_DIR="${LIBERO_OPENPI_DIR:-$PROJECT_DIR/upstream-openpi}"
        if [[ ! -f "$OPENPI_DIR/scripts/serve_policy.py" ]]; then
          candidate="$PROJECT_DIR/../../embodied-policy-lab/upstream-openpi"
          if [[ -f "$candidate/scripts/serve_policy.py" ]]; then
            OPENPI_DIR="$(cd "$candidate" && pwd)"
          fi
        fi
        if [[ ! -f "$OPENPI_DIR/scripts/serve_policy.py" ]]; then
          echo "Cannot find the LIBERO OpenPI server checkout." >&2
          echo "Set LIBERO_OPENPI_DIR to a checkout containing scripts/serve_policy.py." >&2
          exit 1
        fi
        OPENPI_CACHE="${OPENPI_DATA_HOME:-$PROJECT_DIR/cache/openpi}"
        primary_cache="$(dirname "$OPENPI_DIR")/cache/openpi"
        if [[ ! -d "$OPENPI_CACHE/openpi-assets/checkpoints/pi05_libero" ]] && \
          [[ -d "$primary_cache/openpi-assets/checkpoints/pi05_libero" ]]; then
          OPENPI_CACHE="$primary_cache"
        fi
        export OPENPI_DATA_HOME="$OPENPI_CACHE"
        cd "$OPENPI_DIR"
        OPENPI_UV_COMMAND="${OPENPI_UV_COMMAND:-uv}"
        exec "$OPENPI_UV_COMMAND" run scripts/serve_policy.py --env LIBERO --port "$PORT"
        ;;
      fastwam)
        if [[ -z "${FASTWAM_DIR:-}" ]]; then
          for candidate in "$PROJECT_DIR/../upstream-fastwam" \
            "$PROJECT_DIR/../../upstream-fastwam"; do
            if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
              FASTWAM_DIR="$(cd "$candidate" && pwd)"
              break
            fi
          done
        fi
        if [[ -z "${FASTWAM_DIR:-}" ]]; then
          echo "Cannot find upstream-fastwam; set FASTWAM_DIR explicitly." >&2
          exit 1
        fi
        PYTHON="$FASTWAM_DIR/.venv/bin/python"
        CHECKPOINT="${FASTWAM_CHECKPOINT:-$FASTWAM_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
        STATS="${FASTWAM_STATS:-$FASTWAM_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
        if [[ ! -x "$PYTHON" ]] || [[ ! -f "$CHECKPOINT" ]] || [[ ! -f "$STATS" ]]; then
          echo "Fast-WAM runtime or release artifacts are missing." >&2
          echo "Run ./scripts/setup_fastwam_libero.sh --check for details." >&2
          exit 1
        fi
        COMMAND=(
          "$PYTHON" "$PROJECT_DIR/showcase/serve_fastwam_policy.py"
          --port "$PORT"
          --upstream-root "$FASTWAM_DIR"
          --checkpoint "$CHECKPOINT"
          --stats "$STATS"
        )
        if [[ -n "${FASTWAM_ARTIFACT_DIR:-}" ]]; then
          COMMAND+=(--artifact-dir "$FASTWAM_ARTIFACT_DIR")
        fi
        exec "${COMMAND[@]}"
        ;;
      flexpi)
        if [[ -z "${FLEXPI_DIR:-}" ]]; then
          for candidate in "$PROJECT_DIR/../upstream-flexpi" \
            "$PROJECT_DIR/../../upstream-flexpi"; do
            if git -C "$candidate" rev-parse --git-dir >/dev/null 2>&1; then
              FLEXPI_DIR="$(cd "$candidate" && pwd)"
              break
            fi
          done
        fi
        if [[ -z "${FLEXPI_DIR:-}" ]]; then
          echo "Cannot find upstream-flexpi; set FLEXPI_DIR explicitly." >&2
          exit 1
        fi
        PYTHON="$FLEXPI_DIR/.venv/bin/python"
        RELEASE="$FLEXPI_DIR/runs/flexpi-libero"
        CHECKPOINT="${FLEXPI_CHECKPOINT:-$RELEASE/checkpoints/weights/step_010860.pt}"
        CONFIG="${FLEXPI_CONFIG:-$RELEASE/config.yaml}"
        STATS="${FLEXPI_STATS:-$RELEASE/dataset_stats.json}"
        INTRINSICS="${FLEXPI_INTRINSICS:-$FLEXPI_DIR/data/libero-intrinsics/libero_spatial_no_noops_lerobot/meta/camera_intrinsics.json}"
        if [[ ! -x "$PYTHON" ]] || [[ ! -f "$CHECKPOINT" ]] || \
          [[ ! -f "$CONFIG" ]] || [[ ! -f "$STATS" ]] || [[ ! -f "$INTRINSICS" ]]; then
          echo "Flex-π runtime or release artifacts are missing." >&2
          echo "Run ./scripts/setup_flexpi_libero.sh --check for details." >&2
          exit 1
        fi
        export PYTHONNOUSERSITE=1
        export DIFFSYNTH_MODEL_BASE_PATH="$FLEXPI_DIR/checkpoints"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$FLEXPI_DIR/.libero-config}"
        export PYTHONPATH="$FLEXPI_DIR/third_party/LIBERO:$FLEXPI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
        COMMAND=(
          "$PYTHON" "$PROJECT_DIR/showcase/serve_flexpi_policy.py"
          --port "$PORT"
          --upstream-root "$FLEXPI_DIR"
          --checkpoint "$CHECKPOINT"
          --config "$CONFIG"
          --stats "$STATS"
          --intrinsics "$INTRINSICS"
        )
        if [[ -n "${FLEXPI_ARTIFACT_DIR:-}" ]]; then
          COMMAND+=(--artifact-dir "$FLEXPI_ARTIFACT_DIR")
        fi
        exec "${COMMAND[@]}"
        ;;
      *)
        echo "Unsupported LIBERO model: $MODEL (choose pi05, fastwam, or flexpi)" >&2
        exit 2
        ;;
    esac
    ;;
  robocasa)
    case "$MODEL" in
      pi05)
        OPENPI_DIR="$PROJECT_DIR/upstream-robocasa-openpi"
        PYTHON="$OPENPI_DIR/.venv/bin/python"
        CHECKPOINT="${ROBOCASA_CHECKPOINT:-$PROJECT_DIR/cache/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
        CONFIG="${ROBOCASA_POLICY_CONFIG:-pi05_pretrain_human300}"
        if [[ ! -x "$PYTHON" ]]; then
          echo "RoboCasa is not set up. Run ./scripts/setup_robocasa.sh first." >&2
          exit 1
        fi
        if [[ ! -f "$CHECKPOINT/assets/norm_stats.json" ]] || \
          [[ ! -f "$CHECKPOINT/params/_METADATA" ]]; then
          echo "The RoboCasa π0.5 checkpoint is incomplete at $CHECKPOINT" >&2
          echo "Download it with: ROBOCASA_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_robocasa.sh" >&2
          exit 1
        fi
        export OPENPI_DATA_HOME="$PROJECT_DIR/cache/robocasa-openpi"
        exec "$PYTHON" "$PROJECT_DIR/showcase/serve_robocasa_policy.py" \
          --port "$PORT" \
          --config "$CONFIG" \
          --checkpoint "$CHECKPOINT"
        ;;
      groot-n1.5)
        GROOT_DIR="$PROJECT_DIR/upstream-robocasa-groot"
        PYTHON="$GROOT_DIR/.venv/bin/python"
        CHECKPOINT="${GROOT_CHECKPOINT:-$PROJECT_DIR/cache/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000}"
        if [[ ! -x "$PYTHON" ]]; then
          echo "GR00T is not set up. Run ./scripts/setup_groot.sh first." >&2
          exit 1
        fi
        if [[ ! -f "$CHECKPOINT/config.json" ]] || \
          [[ ! -f "$CHECKPOINT/model.safetensors.index.json" ]] || \
          [[ ! -f "$CHECKPOINT/experiment_cfg/metadata.json" ]]; then
          echo "The RoboCasa GR00T checkpoint is incomplete at $CHECKPOINT" >&2
          echo "Download it with: GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh" >&2
          exit 1
        fi
        export NO_ALBUMENTATIONS_UPDATE=1
        exec "$PYTHON" "$PROJECT_DIR/showcase/serve_groot_policy.py" \
          --port "$PORT" \
          --checkpoint "$CHECKPOINT"
        ;;
      *)
        echo "Unsupported RoboCasa model: $MODEL (choose pi05 or groot-n1.5)" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unsupported backend: $BACKEND (choose libero or robocasa)" >&2
    exit 2
    ;;
esac

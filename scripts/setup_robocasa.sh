#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROBOCASA_DIR="$PROJECT_DIR/upstream-robocasa"
ROBOSUITE_DIR="$PROJECT_DIR/upstream-robosuite"
OPENPI_DIR="$PROJECT_DIR/upstream-robocasa-openpi"
PYTHON="$OPENPI_DIR/.venv/bin/python"
DOWNLOAD_ASSETS="${ROBOCASA_DOWNLOAD_ASSETS:-0}"
DOWNLOAD_CHECKPOINT="${ROBOCASA_DOWNLOAD_CHECKPOINT:-0}"
CHECKPOINT_ROOT="$PROJECT_DIR/cache/robocasa365_checkpoints"
CHECKPOINT_RELATIVE="pi05_pretrain_human300/multitask_learning/75000"

for command in git uv cc curl nvidia-smi ffmpeg; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command" >&2
    exit 1
  fi
done

git -C "$PROJECT_DIR" submodule update --init \
  upstream-robocasa upstream-robocasa-openpi upstream-robosuite

echo "Creating the isolated RoboCasa/OpenPI Python 3.11 runtime..."
(
  cd "$OPENPI_DIR"
  # The fork pins PyAV 14.4, which has no CPython 3.11 Linux wheel and tries to
  # compile against FFmpeg 7 headers. Video decoding is not used for inference,
  # so install the current compatible wheel instead of adding system libraries.
  # The fork's lockfile predates uv's current lock format. --frozen consumes it
  # as published and keeps this pinned submodule byte-for-byte clean.
  GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-package av
)

uv pip install --python "$PYTHON" "av>=18" --only-binary=:all:

uv pip install --python "$PYTHON" \
  -e "$ROBOSUITE_DIR" \
  -e "$ROBOCASA_DIR" \
  -e "$OPENPI_DIR/packages/openpi-client"

# RoboCasa's LeRobot requirement otherwise selects huggingface-hub 1.x, while
# the fork's pinned Transformers release explicitly requires a pre-1.0 Hub API.
uv pip install --python "$PYTHON" "huggingface-hub>=0.34.2,<1.0"

if [[ ! -f "$ROBOCASA_DIR/robocasa/macros_private.py" ]]; then
  "$PYTHON" -m robocasa.scripts.setup_macros
fi
if [[ ! -f "$ROBOSUITE_DIR/robosuite/macros_private.py" ]]; then
  "$PYTHON" -m robosuite.scripts.setup_macros
fi

if [[ "$DOWNLOAD_ASSETS" == "1" ]]; then
  echo "Downloading the approximately 10 GB RoboCasa kitchen asset bundle..."
  printf 'y\n' | "$PYTHON" -m robocasa.scripts.download_kitchen_assets --type all
else
  echo "RoboCasa assets were not requested. Download later with:"
  echo "  ROBOCASA_DOWNLOAD_ASSETS=1 ./scripts/setup_robocasa.sh"
fi

if [[ "$DOWNLOAD_CHECKPOINT" == "1" ]]; then
  echo "Downloading only inference parameters and normalization assets (~12 GB)..."
  mkdir -p "$CHECKPOINT_ROOT"
  HF_XET_HIGH_PERFORMANCE=1 \
  "$OPENPI_DIR/.venv/bin/hf" download robocasa/robocasa365_checkpoints \
    --include "$CHECKPOINT_RELATIVE/assets/**" \
    --include "$CHECKPOINT_RELATIVE/params/**" \
    --local-dir "$CHECKPOINT_ROOT"
else
  echo "The π0.5 RoboCasa checkpoint was not requested. Download later with:"
  echo "  ROBOCASA_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_robocasa.sh"
fi

"$PYTHON" - <<'PY'
import jax
import robocasa
import robosuite
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

print("JAX devices:", jax.devices())
print("RoboCasa:", robocasa.__path__[0])
print("robosuite:", robosuite.__version__)
print("Task sets:", ", ".join(sorted(TASK_SET_REGISTRY)))
PY

echo "RoboCasa setup complete."
echo "Simulator check: ./scripts/run_robocasa_smoke.sh"
echo "Checkpoint path: $CHECKPOINT_ROOT/$CHECKPOINT_RELATIVE"

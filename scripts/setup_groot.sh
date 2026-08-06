#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GROOT_DIR="$PROJECT_DIR/upstream-robocasa-groot"
ROBOCASA_DIR="$PROJECT_DIR/upstream-robocasa"
ROBOSUITE_DIR="$PROJECT_DIR/upstream-robosuite"
PYTHON="$GROOT_DIR/.venv/bin/python"
DOWNLOAD_ASSETS="${ROBOCASA_DOWNLOAD_ASSETS:-0}"
DOWNLOAD_CHECKPOINT="${GROOT_DOWNLOAD_CHECKPOINT:-0}"
CHECKPOINT_ROOT="$PROJECT_DIR/cache/robocasa365_checkpoints"
CHECKPOINT_RELATIVE="gr00t_n1-5/multitask_learning/checkpoint-120000"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

for command in git uv nvidia-smi ffmpeg; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command" >&2
    exit 1
  fi
done

if [[ "$(uname -s)" != "Linux" ]] || [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This GR00T setup profile currently supports Linux x86_64 NVIDIA workstations." >&2
  exit 1
fi

git -C "$PROJECT_DIR" submodule update --init \
  upstream-robocasa-groot upstream-robocasa upstream-robosuite

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating the isolated GR00T Python 3.12 runtime..."
  uv venv --python 3.12 "$GROOT_DIR/.venv"
fi

echo "Installing a Blackwell-compatible PyTorch/CUDA and FlashAttention runtime..."
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" \
  --default-index https://pypi.org/simple \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  "torch==2.9.0+cu128" "torchvision==0.24.0+cu128"
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" "$FLASH_ATTN_WHEEL"

# The RoboCasa N1.5 fork does not put torch in its core dependency set. Keeping
# accelerator packages above explicit lets this machine use CUDA 12.8/sm_120
# rather than the fork's older optional torch 2.5 profile.
# The fork's monolithic dependency list includes ONNX 1.15, which has no
# CPython 3.12 wheel and is only used by the optional TensorRT export path.
# Install the pinned source without dependencies, then the inference/runtime
# subset explicitly. This keeps the upstream submodule unmodified.
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" --no-deps -e "$GROOT_DIR"
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" \
  "albumentations==1.4.18" "av==12.3.0" "blessings==1.7" "dm_tree==0.1.8" \
  "einops==0.8.1" "gymnasium==1.0.0" "h5py" "hydra-core==1.3.2" \
  "imageio[ffmpeg]>=2.34" "kornia==0.7.4" "matplotlib==3.10.0" \
  "numpy==2.2.5" "numpydantic==1.6.7" "omegaconf==2.3.0" \
  "opencv-python-headless==4.11.0.86" "pandas==2.2.3" "pydantic==2.10.6" \
  "PyYAML==6.0.2" "Requests==2.32.3" "timm==1.0.14" "tqdm==4.67.1" \
  "transformers==4.51.3" "typing_extensions>=4.14" "accelerate" \
  "protobuf==3.20.3" "peft==0.17.0" "diffusers==0.30.2" "pyzmq" "pytest" \
  "tyro" "huggingface-hub[cli]>=0.34.2,<1.0"

UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" --no-deps -e "$ROBOSUITE_DIR"
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" \
  "qpsolvers[quadprog]>=4.3.1" termcolor

# RoboCasa pins tianshou 0.4.10 for an optional parallel-dataset utility while
# the official GR00T fork pins tianshou 0.5.1. Neither package is used by this
# single-environment inference path. Install RoboCasa without those training
# extras and retain the GR00T version, avoiding an artificial resolver conflict.
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" --no-deps -e "$ROBOCASA_DIR"
UV_HTTP_TIMEOUT=600 uv pip install --python "$PYTHON" \
  "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
  pygame pyyaml h5py lxml hidapi

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
  echo "Reusing installed RoboCasa assets. To download/recheck them, run:"
  echo "  ROBOCASA_DOWNLOAD_ASSETS=1 ./scripts/setup_groot.sh"
fi

if [[ "$DOWNLOAD_CHECKPOINT" == "1" ]]; then
  echo "Downloading the GR00T N1.5 inference checkpoint (~7.6 GB; no optimizer state)..."
  mkdir -p "$CHECKPOINT_ROOT"
  HF_XET_HIGH_PERFORMANCE=1 \
  "$GROOT_DIR/.venv/bin/hf" download robocasa/robocasa365_checkpoints \
    --include \
      "$CHECKPOINT_RELATIVE/config.json" \
      "$CHECKPOINT_RELATIVE/model-*.safetensors" \
      "$CHECKPOINT_RELATIVE/model.safetensors.index.json" \
      "$CHECKPOINT_RELATIVE/experiment_cfg/metadata.json" \
    --local-dir "$CHECKPOINT_ROOT"
else
  echo "The GR00T checkpoint was not requested. Download it with:"
  echo "  GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh"
fi

"$PYTHON" - <<'PY'
import flash_attn
import robocasa
import robosuite
import torch
from gr00t.eval.robot import RobotInferenceClient
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
major, minor = torch.cuda.get_device_capability()
assert (major, minor) >= (8, 0), f"FlashAttention requires sm_80+, got sm_{major}{minor}"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(), "capability:", (major, minor))
print("FlashAttention:", flash_attn.__version__)
print("RoboCasa:", robocasa.__path__[0])
print("robosuite:", robosuite.__version__)
print("GR00T client:", RobotInferenceClient.__name__)
print("Task sets:", ", ".join(sorted(TASK_SET_REGISTRY)))
PY

echo "GR00T setup complete."
echo "Checkpoint path: $CHECKPOINT_ROOT/$CHECKPOINT_RELATIVE"
echo "Run: ./scripts/run_interactive_showcase.sh --backend robocasa --model groot-n1.5"

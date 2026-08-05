#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_DIR/upstream-openpi"
EXPECTED_COMMIT="$(tr -d '[:space:]' < "$PROJECT_DIR/UPSTREAM_COMMIT")"

if ! git -C "$OPENPI_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  if git -C "$PROJECT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    git -C "$PROJECT_DIR" submodule update --init --recursive
  else
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
    git -C "$OPENPI_DIR" checkout "$EXPECTED_COMMIT"
    git -C "$OPENPI_DIR" submodule update --init --recursive
  fi
fi

ACTUAL_COMMIT="$(git -C "$OPENPI_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Warning: upstream-openpi is at $ACTUAL_COMMIT; documented baseline is $EXPECTED_COMMIT" >&2
fi

for command in git git-lfs uv cc nvidia-smi strace ffmpeg; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command" >&2
    echo "On Ubuntu install: sudo apt-get install -y linux-libc-dev build-essential git-lfs strace ffmpeg" >&2
    exit 1
  fi
done

mkdir -p "$PROJECT_DIR/cache/openpi" "$PROJECT_DIR/config/libero" \
  "$PROJECT_DIR/logs" "$PROJECT_DIR/run" "$PROJECT_DIR/videos"

cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync

if [[ ! -x examples/libero/.venv/bin/python ]]; then
  uv venv --python 3.8 examples/libero/.venv
fi

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --python examples/libero/.venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python \
  -e packages/openpi-client -e third_party/libero

if [[ ! -f "$PROJECT_DIR/config/libero/config.yaml" ]]; then
  printf 'n\n' | env \
    LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero" \
    PYTHONPATH="$OPENPI_DIR/third_party/libero" \
    examples/libero/.venv/bin/python -c 'import libero.libero'
fi

uv run python -c "import jax; print('JAX devices:', jax.devices())"
env \
  LIBERO_CONFIG_PATH="$PROJECT_DIR/config/libero" \
  PYTHONPATH="$OPENPI_DIR/third_party/libero" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  examples/libero/.venv/bin/python -c \
  "import mujoco; from libero.libero import benchmark; print('MuJoCo:', mujoco.__version__); print('LIBERO suites:', sorted(benchmark.get_benchmark_dict()))"

echo "Setup complete. Run: $PROJECT_DIR/scripts/run_smoke_test.sh"

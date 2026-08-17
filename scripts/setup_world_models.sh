#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
UPSTREAM_DIR="$PROJECT_DIR/upstream-jepa-wms"
RUNTIME_PYTHON="$UPSTREAM_DIR/.venv/bin/python"
MODEL="${JEPA_WORLD_MODEL:-dino_wm_droid}"

case "$MODEL" in
  dino_wm_droid|jepa_wm_droid) ;;
  *) echo "JEPA_WORLD_MODEL must be dino_wm_droid or jepa_wm_droid" >&2; exit 2 ;;
esac

git -C "$PROJECT_DIR" submodule update --init upstream-jepa-wms
uv venv --python 3.10 "$UPSTREAM_DIR/.venv"
uv pip install --python "$RUNTIME_PYTHON" -e "$UPSTREAM_DIR"
uv pip install --python "$RUNTIME_PYTHON" "huggingface_hub>=0.20.0"

if [[ "${JEPA_DOWNLOAD_WEIGHTS:-0}" == "1" ]]; then
  echo "Downloading $MODEL into the authenticated Hugging Face cache..."
  uvx hf@latest download facebook/jepa-wms "$MODEL.pth.tar"
fi

cat <<EOF
JEPA-WMs isolated runtime is ready.

The learned adapter remains diagnostic-only until its 7D action and temporal
mapping are validated against the 12D RoboCasa policy output.

Start the persistent worker with:
  $RUNTIME_PYTHON $PROJECT_DIR/showcase/serve_jepa_world_model.py \\
    --model $MODEL --device cpu \\
    --allowed-root $PROJECT_DIR/showcase-runs \\
    --upstream-dir $UPSTREAM_DIR
EOF

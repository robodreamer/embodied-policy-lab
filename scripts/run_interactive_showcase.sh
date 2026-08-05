#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export INTERACTIVE=1
export HOLD_OPEN=0
exec "$SCRIPT_DIR/run_showcase.sh"

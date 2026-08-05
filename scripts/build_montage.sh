#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION_DIR="${1:-$PROJECT_DIR/showcase-runs/latest}"
VIDEO_DIR="$SESSION_DIR/videos"
OUTPUT_PATH="${2:-$SESSION_DIR/showcase-reel.mp4}"
CONCAT_FILE="$SESSION_DIR/video-list.txt"

if ! find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' -print -quit | grep -q .; then
  echo "No MP4 rollouts found under $VIDEO_DIR" >&2
  exit 1
fi

find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' -print0 | sort -z | \
  while IFS= read -r -d '' video; do
    printf "file '%s'\n" "${video//\'/\'\\\'\'}"
  done > "$CONCAT_FILE"

ffmpeg -hide_banner -loglevel warning -y \
  -f concat -safe 0 -i "$CONCAT_FILE" \
  -vf "scale=960:-2:flags=lanczos,format=yuv420p" \
  -c:v libx264 -preset medium -crf 20 -an "$OUTPUT_PATH"

echo "Wrote $OUTPUT_PATH"

"""Dependency-light Flex-π release and presentation contracts."""

from __future__ import annotations

from typing import Any


FLEXPI_MODES = (
    {
        "key": "action-only",
        "display_name": "Action only",
        "description": "Fast KV-cache path; visual futures are not generated.",
    },
    {
        "key": "full-joint",
        "display_name": "World-action co-generation",
        "description": (
            "Co-generate RGB, DINO, pointmap and end-effector action futures."
        ),
    },
)

# Pinned flexpi-libero composite: wrist on top, external at bottom-left, and a
# synthetic black third-camera slot at bottom-right.
COMPOSITE_WIDTH = 512
COMPOSITE_HEIGHT = 448
WRIST_WIDTH = 512
WRIST_HEIGHT = 288
EXTERNAL_WIDTH = 256
EXTERNAL_HEIGHT = 160
DEFAULT_FRAME_INTERVAL_ACTIONS = 4

# Presentation speed only. This is intentionally not labeled as simulator or
# model time; prefix/action offsets are recorded separately in timeline JSON.
REPLAY_FPS = 2


def normalize_flexpi_mode(mode: str) -> str:
    key = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "action": "action-only",
        "fast": "action-only",
        "joint": "full-joint",
        "full": "full-joint",
    }
    key = aliases.get(key, key)
    supported = {item["key"] for item in FLEXPI_MODES}
    if key not in supported:
        raise ValueError(
            f"Unknown Flex-π mode {mode!r}; choose one of: "
            f"{', '.join(sorted(supported))}"
        )
    return key


def prediction_layout() -> dict[str, Any]:
    return {
        "name": "tshape_libero_2cam_448x512",
        "composite_width": COMPOSITE_WIDTH,
        "composite_height": COMPOSITE_HEIGHT,
        "wrist": {"top": 0, "left": 0, "width": WRIST_WIDTH, "height": WRIST_HEIGHT},
        "external": {
            "top": WRIST_HEIGHT,
            "left": 0,
            "width": EXTERNAL_WIDTH,
            "height": EXTERNAL_HEIGHT,
        },
        "synthetic_black": {
            "top": WRIST_HEIGHT,
            "left": EXTERNAL_WIDTH,
            "width": EXTERNAL_WIDTH,
            "height": EXTERNAL_HEIGHT,
        },
    }


def split_prediction_frame(frame: Any) -> tuple[Any, Any]:
    import numpy as np

    array = np.asarray(frame, dtype=np.uint8)
    expected = (COMPOSITE_HEIGHT, COMPOSITE_WIDTH, 3)
    if array.shape != expected:
        raise ValueError(
            f"Flex-π prediction composite must have shape {expected}, got {array.shape}"
        )
    wrist = array[:WRIST_HEIGHT, :WRIST_WIDTH]
    external = array[
        WRIST_HEIGHT : WRIST_HEIGHT + EXTERNAL_HEIGHT,
        :EXTERNAL_WIDTH,
    ]
    return np.ascontiguousarray(external), np.ascontiguousarray(wrist)

# README media provenance

The media in this directory is derived from local Embodied Policy Lab runs and
contains no downloaded model weights or third-party marketing artwork.

| Asset | Source run | Source commit | Profile and state | Transform |
|---|---|---|---|---|
| `environment-libero.png` | `showcase-runs/20260826-100813` | `2859c61` | Flex-π full-joint, LIBERO spatial task 0; completed action-aligned comparison | Representative frame extracted at 6 s from `flexpi-world-action-comparison.gif`; no crop or content edit |
| `environment-robocasa.png` | `showcase-runs/20260819-121848` | pre-public-launch stack | Successful local π0.5 RoboCasa rollout: put both moka pots on the stove | Representative frame extracted at 18 s from `robocasa-pi05-rollout.gif`; no crop or content edit |
| `studio-overview.png` | `showcase-runs/20260903-114826-robotwin` | `9fa46a7` | Fast-WAM action-only, RoboTwin `beat_block_hammer`; successful functional check and ready for another rollout | Browser capture at a 1440 px publication viewport; cropped at the experiment-panel border before the camera grid |
| `studio-live-cameras.png` | `showcase-runs/20260903-114826-robotwin` | `9fa46a7` | Fast-WAM action-only; completed front-observer, head, left-wrist, and right-wrist views | Browser capture cropped to the four-camera grid; no content edits |
| `studio-world-action-comparison.png` | `showcase-runs/20260826-100813` | `2859c61` | Flex-π full-joint; post-rollout actual/generated external and wrist comparison | Unaltered browser capture |
| `studio-evidence.png` | `showcase-runs/20260903-114826-robotwin` | `9fa46a7` | Fast-WAM action-only; applied prompt, completed telemetry, 14D action, runtime, and single-run result panels | Browser capture cropped to the evidence panels; no content edits |
| `flexpi-world-action-comparison.gif` | `showcase-runs/20260826-100813` | `2859c61` | Flex-π full-joint; completed 24-frame external/wrist prediction timeline | Scaled and arranged from the run's four aligned MP4 streams at 2 fps; labels only |
| `robocasa-pi05-rollout.gif` | `showcase-runs/20260819-121848` | pre-public-launch stack | Successful local π0.5 RoboCasa rollout: put both moka pots on the stove | Resized presentation derivative |

The studio PNGs are current product captures with only the crops documented
above; the animations are presentation derivatives of local run videos. Raw
session data, model inputs, checkpoints, and machine-specific logs remain
ignored. Model, simulator, and asset licensing is documented in
[`../external-assets.md`](../external-assets.md) and the upstream repositories.

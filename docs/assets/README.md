# README media provenance

The media in this directory is derived from local Embodied Policy Lab runs and
contains no downloaded model weights or third-party marketing artwork.

| Asset | Source run | Source commit | Profile and state | Transform |
|---|---|---|---|---|
| `environment-libero.png` | `showcase-runs/20260824-120749` | pre-public-launch stack | Flex-π, LIBERO Spatial tasks 0, 5, and 6; front/external camera at rollout start | Three frames extracted at 0.2 s, scaled to 240×240, centered in equal 320×240 cells, then joined without cropping |
| `environment-robocasa.png` | `showcase-runs/20260817-164751`, `20260806-112709`, and `20260806-125457` | pre-public-launch stack | Genuine RoboCasa365 backend; blender-lid, toaster-oven, and stove task scenes from the front/external camera | Three frames extracted at 0.2 s, scaled to 240×240, centered in equal 320×240 cells, then joined without cropping |
| `environment-robotwin.png` | `showcase-runs/20260903-114826-robotwin`, `20260902-160259-robotwin`, plus deterministic `place_bread_basket` fixture at seed `500000` | `9fa46a7`, pre-merge RoboTwin stack, and `12430e4` | RoboTwin front observer; hammer, lift-pot, and bread-basket task scenes | Native 320×240 observer frames joined horizontally without cropping; bread-basket frame captured after a successful upstream expert check |
| `studio-overview.png` | `showcase-runs/20260903-114826-robotwin` | `9fa46a7` | Fast-WAM action-only, RoboTwin `beat_block_hammer`; successful functional check and ready for another rollout | Browser capture at a 1440 px publication viewport; cropped at the experiment-panel border before the camera grid |
| `studio-live-cameras.png` | `showcase-runs/20260903-114826-robotwin` | `9fa46a7` | Fast-WAM action-only; completed front-observer, head, left-wrist, and right-wrist views | Browser capture cropped to the four-camera grid; no content edits |
| `studio-world-action-comparison.png` | `showcase-runs/20260826-100813` | `2859c61` | Flex-π full-joint; post-rollout actual/generated external and wrist comparison | Unaltered browser capture |
| `studio-evidence.png` | `showcase-runs/20260903-114826-robotwin` | `9fa46a7` | Fast-WAM action-only; applied prompt, completed telemetry, 14D action, runtime, and single-run result panels | Browser capture cropped to the evidence panels; no content edits |
| `flexpi-world-action-comparison.gif` | `showcase-runs/20260826-100813` | `2859c61` | Flex-π full-joint; completed 24-frame external/wrist prediction timeline | Scaled and arranged from the run's four aligned MP4 streams at 2 fps; labels only |
| `libero-fastwam-rollout.gif` | `showcase-runs/20260819-121848` | pre-public-launch stack | Successful local Fast-WAM LIBERO-10 rollout: put both moka pots on the stove | Resized presentation derivative |

The studio PNGs are current product captures with only the crops documented
above; the animations are presentation derivatives of local run videos. Raw
session data, model inputs, checkpoints, and machine-specific logs remain
ignored. Model, simulator, and asset licensing is documented in
[`../external-assets.md`](../external-assets.md) and the upstream repositories.

# RoboTwin 2.0 environment foundation

Status: **experimental native batch integration**. This is an executable wiring
foundation, not a reproduction of either model's published RoboTwin score.

## Why this is a separate environment

RoboTwin is a bimanual SAPIEN/Vulkan benchmark, not another MuJoCo task suite.
The released Fast-WAM and Flex-π profiles use an ALOHA-AgileX 14D joint-state /
14D joint-position action contract with three cameras (head, left wrist, right
wrist). LIBERO uses a 7D end-effector action contract and two cameras. Treating
those paths as interchangeable would invalidate the checkpoints.

RoboTwin's SAPIEN stack also requires `opencv-python`, while Flex-π's LIBERO
runtime pins `opencv-python-headless`. Installing both into one environment can
silently replace the shared `cv2` files. The lab therefore creates
`.venv-robotwin` beside each model and never changes its existing `.venv`.

## Exact source boundary

| Component | Revision | Role |
|---|---|---|
| RoboTwin stable `release` tag | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` | simulator/task contract used by both released model evaluators |
| Fast-WAM | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | native RoboTwin action evaluator |
| Flex-π | `20c1b2b71ea35a415d5d47c39b04443cfadad7a1` | native selectable action-only/full-joint evaluator |
| Curobo | `8e734f3ced1df898990bcd92de40abce475907db` | motion-planning dependency used by scene initialization/expert seed checks |
| PyTorch3D stable | `75ebeeaea0908c5527e7b1e305fbc7681382db47` | RoboTwin geometry dependency |

RoboTwin's newer `main` branch is not substituted here. The model repositories
vendor the stable release contract and their published checkpoints were built
for it. Migrating to the newer XPolicyLab path is a separate compatibility and
benchmark project.

The Fast-WAM revision omits the upstream `task_config/` directory from its
vendored copy. Setup fills that missing untracked runtime input from the exact
RoboTwin release above. Flex-π keeps its locally adapted vendor tree; both
vendors share only the downloaded official assets.

## Setup without surprise downloads

Source and isolated runtimes can be prepared without fetching weights or the
asset bundle:

```bash
./scripts/setup_robotwin.sh --model both
```

Large downloads are explicit:

```bash
# Official meshes, textures, and embodiment files: approximately 16 GB.
./scripts/setup_robotwin.sh --model both --download-assets

# Release weights: approximately 12 GB per selected model.
# Flex-π also fetches its required VAE, T5/tokenizer, and DINOv3 assets.
./scripts/setup_robotwin.sh --model fastwam --download-checkpoints
./scripts/setup_robotwin.sh --model flexpi --download-checkpoints

# Non-mutating integrity/readiness report. This hashes present checkpoints.
./scripts/setup_robotwin.sh --model both --check
```

The asset download uses RoboTwin's own pinned downloader. Model weights come
from exact Hugging Face revisions and are not redistributed by this repository.
SAPIEN renders through Vulkan; on a multi-GPU Linux host, set
`VK_ICD_FILENAMES` explicitly if automatic device selection is wrong.

## Validate the simulator before loading a model

The model-free smoke path resets one task, validates all three RGB observations
and the 14D joint state, executes the current 14D qpos as a one-step no-op, and
writes JSON evidence:

```bash
./scripts/run_robotwin_smoke.sh --task click_bell --phase demo_clean --seed 0
```

Evidence lands under `results/robotwin-smoke/<timestamp>/result.json`. A pass is
a simulator/interface wiring check; it says nothing about policy success.

## Run one bounded model evaluation

RoboTwin currently uses each model publisher's native single-task evaluator.
This is deliberately batch-only; it does not yet claim the shared browser
studio, HTTP policy boundary, or matched headless benchmark.

```bash
# Fast-WAM action inference; 24 actions executed from each 32-action chunk.
./lab --backend robotwin --model fastwam --mode batch \
  --task-set demo_clean --task-id click_bell --trials 1 --default

# Flex-π full world-action co-generation; all 32 actions are the native default.
./lab --backend robotwin --model flexpi --mode batch \
  --task-set demo_clean --task-id click_bell --trials 1 --default

# The same Flex-π checkpoint on its lower-compute action-only path.
./lab --backend robotwin --model flexpi --flexpi-mode action-only --mode batch \
  --task-set demo_randomized --task-id turn_switch --trials 1 --default
```

The runner keeps intermediate camera observations by default so saved videos
are inspectable. `-- --fast-render` restores the upstream speed-oriented mode
that skips RGB rendering inside an executed action prefix. Flex-π defaults to
text-encoder CPU offload to leave room for SAPIEN on a 24 GB GPU; override with
`FLEXPI_OFFLOAD_TEXT_ENCODER=false` on a larger card.

Native results remain beneath the selected model checkout's
`evaluate_results/robotwin/` tree. They include the upstream config, log,
videos, and clean/random result file.

## What can and cannot be compared yet

Both registered profiles now agree on simulator release, embodiment, three
cameras, 14D qpos action space, 32-action prediction horizon, task, phase, seed,
and accepted episode count. Their native defaults still differ materially:

| Setting | Fast-WAM | Flex-π |
|---|---:|---:|
| Replan interval | 24 | 32 |
| Inference output | actions | selectable actions or joint RGB/DINO/pointmap/actions |
| Default denoising configuration | publisher profile | four-step publisher profile |

Consequently, two one-off commands are useful integration checks but not a
matched benchmark. A later benchmark PR should declare shared tasks and seeds,
equalize the replan schedule and stopping rules, run both clean and randomized
phases, retain failures, and report uncertainty. It should keep action-only and
full-joint Flex-π as separate conditions.

## Known limitations

- No shared interactive studio or live camera dashboard yet.
- No normalized lab session schema for native RoboTwin artifacts yet.
- No local GPU/model rollout is claimed by this foundation until assets and
  checkpoints are present and the commands above are recorded.
- RoboTwin's official full protocol is 100 episodes per task per phase across
  50 tasks. The default one-trial command is only a bounded wiring check.

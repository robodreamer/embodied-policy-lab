# RoboTwin 2.0 environment foundation

Status: **experimental browser-studio and native batch integration**. This is
an executable wiring foundation, not a reproduction of either model's
published RoboTwin score.

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

## Run the browser studio

The browser adapter keeps the policy and SAPIEN environment in one native
process. This avoids serializing three RGB/depth observations or translating
absolute 14D qpos actions into the unrelated LIBERO HTTP schema:

```bash
./lab --backend robo_twin --list-tasks
./lab --backend robo_twin --model flexpi  # searchable phase/task picker
./lab --backend robo_twin --model fastwam --mode interactive \
  --task-set demo_clean --task click_bell --default
./lab --backend robo_twin --model flexpi --mode interactive \
  --task-set demo_randomized --task-id turn_switch --default
```

The page is available while the checkpoint loads. Once ready, it offers all 50
named tasks, live head/left-wrist/right-wrist RGB views, prompt and rollout
controls, 14D action charts, GPU telemetry, and per-camera MP4 artifacts.
Flex-π's full-joint and action-only modes are switchable without reloading the
checkpoint. The UI calls the policy link `IN-PROCESS NATIVE · LOCAL`; it does
not claim a network policy service.

The full-joint switch enables the released Flex-π joint denoising path, but the
current RoboTwin adapter does not request, decode, or retain generated-future
media. Its studio evidence is the executed three-camera rollout plus the 14D
action chunks. Do not interpret the absence of a comparison panel as
action-only inference; the selected mode remains recorded in `state.json` and
`report.json`.

The released Flex-π checkpoint contains complete video and action experts. Its
saved training config nevertheless requests the Wan and ActionDiT initialization
bases, while the upstream RoboTwin inference preset correctly sets
`skip_dit_load_from_pretrain: true`. Loading the saved config used to overwrite
that inference-only setting and either trigger an unnecessary roughly 20 GB Wan
download or fail with an empty `wan_video_dit` path when downloads were disabled.
The launchers now verify that both expert key sets are complete and create a
session-local hard-linked checkpoint view with the inference overrides. No
checkpoint bytes are copied, publisher files remain unchanged, and
`inference-release.json` records the decision in the session evidence.

Each rollout intentionally keeps RoboTwin's upstream expert seed check and
unseen-instruction generator. Preparation is therefore slower than merely
resetting a scene, but the resulting score follows the native evaluator's seed
validity and language path.

## Run one bounded native model evaluation

Batch mode uses each model publisher's native single-task evaluator. It remains
the reference path for native logs and result naming; the browser studio adds
interactive control and lab-normalized session artifacts without claiming an
HTTP policy boundary or a matched cross-model benchmark.

```bash
# Fast-WAM action inference; 24 actions executed from each 32-action chunk.
./lab --backend robo_twin --model fastwam --mode batch \
  --task-set demo_clean --task-id click_bell --trials 1 --default

# Flex-π full world-action co-generation; all 32 actions are the native default.
./lab --backend robo_twin --model flexpi --mode batch \
  --task-set demo_clean --task-id click_bell --trials 1 --default

# The same Flex-π checkpoint on its lower-compute action-only path.
./lab --backend robo_twin --model flexpi --flexpi-mode action-only --mode batch \
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

- The studio records a normalized lab session, while native batch output keeps
  the upstream result schema; those two artifact layouts are intentionally not
  merged.
- No local GPU/model rollout is claimed by this foundation until assets and
  checkpoints are present and the commands above are recorded.
- RoboTwin's official full protocol is 100 episodes per task per phase across
  50 tasks. The default one-trial command is only a bounded wiring check.

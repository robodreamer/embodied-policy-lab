# RoboCasa backend development notes

Status: unified CLI, automatic rollout evaluator, and interactive dashboard
adapter validated on 2026-08-05. LIBERO remains the default backend; RoboCasa
is selected explicitly.

## Why RoboCasa is a separate runtime

The released `pi05_libero` checkpoint cannot be treated as a simulator-neutral
policy. LIBERO sends two cameras, a 7-dimensional end-effector state/action, and
uses a fixed-arm workspace. The official RoboCasa π0.5 integration sends three
cameras, a 16-dimensional state, and returns a 12-dimensional action for a
`PandaOmron` mobile manipulator. It also has different normalization statistics
and success predicates.

For this reason the project pins three additional upstream repositories as Git
submodules:

- `upstream-robocasa`: environments, task registry, assets, and success checks;
- `upstream-robosuite`: MuJoCo robot and controller runtime;
- `upstream-robocasa-openpi`: RoboCasa policy transforms, training config, and
  the official π0.5 evaluator/server integration.

They use an isolated Python 3.11 environment under
`upstream-robocasa-openpi/.venv`. The existing LIBERO environment is unchanged.

## Current validation ladder

1. Install imports and enumerate the RoboCasa task registry.
2. Reset one target-split environment through Gymnasium.
3. Render and step it headlessly with a zero 12D action.
4. Start the official RoboCasa OpenPI server with the matching checkpoint.
5. Audit one model request, execute its converted action, and record video.
6. Connect reset/task/prompt controls to the existing interactive dashboard.

All six steps passed on this workstation on 2026-08-05. Steps 1–3 deliberately
do not load a policy: a passing simulator smoke test alone is not evidence that
π0.5 inference ran. The next milestone is a statistically meaningful seeded
evaluation across multiple tasks, not additional launcher plumbing.

## Unified showcase CLI

The original entrypoints now select the complete runtime profile with
`--backend`. They choose the correct Python environment, checkpoint, policy
server, observation/action adapter, task registry, horizon, and success signal.

Launch a persistent browser-controlled session:

```bash
./scripts/run_interactive_showcase.sh \
  --backend robocasa \
  --task-set atomic_seen \
  --split target
```

Run automatic trials and generate a report without browser interaction:

```bash
./scripts/run_showcase.sh \
  --backend robocasa \
  --batch \
  --task-set atomic_seen \
  --task-ids 0,2 \
  --trials 3 \
  --budget 1 \
  --no-open
```

Use `./scripts/run_showcase.sh --help` for all flags. Existing environment
variables remain compatible; for example:

```bash
BACKEND=robocasa TASK_SUITE=atomic_seen TASK_IDS=0 \
  ./scripts/run_interactive_showcase.sh
```

The live viewer uses independent MuJoCo renders at 960×540 and up to 6 FPS by
default. This keeps the two dashboard cameras sharp and widescreen without
changing RoboCasa's square 256×256 observations or the 224×224 images delivered
to π0.5. Use `--viewer-width`, `--viewer-height`, and `--viewer-fps` to tune
presentation quality or rendering cost.

In interactive mode, task selection prepares the chosen kitchen scene and
publishes its exact scene-dependent canonical instruction before the local-LLM
button is used. Typed or locally generated text clears the queued action plan,
is sent on the next synchronous policy request, and is recorded with prompt and
action SHA-256 digests. Scored success uses RoboCasa's `info["success"]`;
exploratory and mixed-prompt attempts are saved but excluded from success rates.

The CLI `--task-id` is only the initially selected dropdown entry. It does not
restrict task switching. `atomic_seen` supplies 18 entries by default; use, for
example, `--task-set all_atomic_tasks` for 65 atomic entries or `--task-set
composite_seen` for 16 composite entries in the same dashboard control.

## Setup and simulator smoke test

Install the Python runtime without downloading large artifacts:

```bash
./scripts/setup_robocasa.sh
```

Download the approximately 10 GB kitchen assets and the approximately 12 GB
inference-only portion of the public π0.5 checkpoint:

```bash
ROBOCASA_DOWNLOAD_ASSETS=1 \
ROBOCASA_DOWNLOAD_CHECKPOINT=1 \
./scripts/setup_robocasa.sh
```

The full Hugging Face checkpoint repository is larger because it also contains
optimizer/training state. The setup script intentionally downloads only
`assets/` and `params/`, which are the parts required by OpenPI inference.

The upstream OpenPI lock file pins PyAV 14.4, for which PyPI does not provide a
CPython 3.11 Linux wheel. The setup substitutes a newer binary PyAV wheel. The
showcase does not use LeRobot's dataset video decoder, so this avoids requiring
system FFmpeg development headers without changing policy or simulator code.

Run a model-free reset/render/step check:

```bash
./scripts/run_robocasa_smoke.sh
```

Useful overrides:

```bash
ROBOCASA_TASK_SET=atomic_seen ROBOCASA_TASK_ID=1 \
ROBOCASA_SPLIT=target ROBOCASA_SMOKE_STEPS=20 \
./scripts/run_robocasa_smoke.sh
```

Results are written under `results/robocasa-smoke/<timestamp>/` as a video and
`result.json`. The JSON explicitly records `policy_used: false` to prevent a
simulator-only check from being mistaken for local π0.5 inference.

After the checkpoint download, run one real local π0.5 request and execute the
first five actions from its predicted chunk:

```bash
./scripts/run_robocasa_policy_smoke.sh
```

This produces `result.json`, `inference-audit.jsonl`, GPU telemetry, server and
client logs, and an MP4 under `results/robocasa-policy-smoke/<timestamp>/`.
Unlike the simulator smoke result, this result records `policy_used: true` and
the prompt/action hashes, dimensions, checkpoint, and measured model latency.

### Checkpoint-statistics launcher workaround

The upstream `pi05_pretrain_human300` config retains all 300 training-dataset
paths so it can calculate fallback normalization statistics during training.
At inference startup, the upstream policy factory currently tries those paths
before loading the statistics shipped with the public checkpoint. A machine
with only the inference artifacts therefore gets a `FileNotFoundError` for a
`lerobot/meta/stats.json` training file, even though
`assets/norm_stats.json` is present in the checkpoint.

`showcase/serve_robocasa_policy.py` is a narrow, project-owned workaround. It
loads the checkpoint statistics first, clears `data_dirs` from an in-memory
copy of the config, and passes the statistics explicitly to the unchanged
upstream policy factory. Model weights, RoboCasa observation/action transforms,
and the WebSocket protocol remain upstream implementations. No submodule source
is patched, so the workaround is easy to remove once upstream fixes the load
order.

## Local validation record — 2026-08-05

The model-free check reset `atomic_seen` task 0 (`CloseBlenderLid`) on the
`target` split, rendered the kitchen and `PandaOmron`, and completed five zero
action steps. Its result is explicitly marked `policy_used: false`.

The policy check then restored the public checkpoint entirely from the local
cache, prepared the official three-camera/16D observation, and sent this exact
environment annotation to the local WebSocket policy:

> Close the lid blender by securely placing the lid on top.

The model returned a `[50, 12]` action chunk. The runner recorded SHA-256 hashes
for both prompt and action bytes, executed the first five actions in RoboCasa,
and wrote a valid six-frame H.264 video. Measured first-request latency was
10.48 seconds, including JAX compilation; peak observed GPU memory was
21,121 MiB and peak utilization was 43%. The checkpoint restore read 6.2 GiB
of parameters in 5.21 seconds.

`success_after_smoke_actions` was false, which is expected and is not a policy
failure: this check intentionally stops after five actions while the task
horizon is 900. The smoke result alone is not a RoboCasa success-rate
measurement. It proves local model loading, correct policy-shaped input/output,
simulator action conversion, rendering, and cleanup; the unified evaluator
validated below handles complete episodes and rate accounting.

Local artifacts from the passing run are under
`results/robocasa-policy-smoke/20260805-171258/`. They are ignored by Git because
logs contain machine paths and repeated videos can become large. This measured
summary is the publishable trail.

### Unified batch and interactive validation

Session `20260805-173321` exercised the new CLI rather than a smoke script:

```bash
./scripts/run_showcase.sh --backend robocasa --batch \
  --task-set atomic_seen --task-id 2 --split target --trials 1 \
  --budget 1 --replan-steps 5 --realtime-delay-ms 0 \
  --no-open --no-hold-open --no-network-audit
```

It ran the complete 450-action `CloseToasterOvenDoor` horizon, issued 90 local
π0.5 requests, and generated a 226-frame H.264 rollout, dashboard state,
telemetry, audit log, summary, and Markdown report. Cold/JIT latency was 9.59 s;
warm median was 134.49 ms and warm P95 was 141.16 ms. Peak GPU memory was
21,045 MiB and peak utilization was 99%. The policy did not complete this seed,
so the result is correctly recorded as **0/1**, not presented as a successful
benchmark.

Session `20260805-173442` validated the interactive control protocol. It began
idle, prepared task 0, switched to task 2, and published the scene's exact
canonical instruction. A local `gemma3:1b` generation produced
`Make sure to close the toaster oven door.` The exact text and SHA-256 digest
`2f237a093b5be094e23daf9b19db4c040f1aa5e70d474da122539427a26bca5b`
were acknowledged by π0.5 for three requests before the test deliberately sent
Finish. The attempt is correctly recorded as aborted and excluded from the
success denominator. A follow-up generator test produced three distinct safe
scored variants while rejecting repeated, token-list, and action-changing text.

Network auditing was disabled for these adapter tests, so their reports say
`not_audited`; they do not make a local-only network claim. Enable the default
audit for publication runs.

Session `20260805-174340` then exercised the default syscall audit through the
unified RoboCasa entrypoint. Two π0.5 requests were acknowledged before a
deliberate stop. The report verdict was `loopback_only`: the sole observed
destination was `127.0.0.1:8004`, with no remote IP destination recorded.

Finally, session `20260805-174230` ran the rewritten entrypoint with
`--backend libero`; the original `libero_spatial` task 0 path still completed
successfully (**1/1**). This regression check confirms that backend routing did
not replace or silently alter the default LIBERO workflow.

## Policy profile

The initial profile follows the official RoboCasa π0.5 submission:

| Field | Value |
|---|---|
| OpenPI config | `pi05_pretrain_human300` |
| Checkpoint | `pi05_pretrain_human300/multitask_learning/75000` |
| Default task set | `atomic_seen` |
| Default scene/object split | `target` |
| Policy cameras | left agent view, right agent view, eye-in-hand |
| State dimension | 16 |
| Action dimension | 12 |
| Success signal | `info["success"]` |

The checkpoint lives in `cache/robocasa365_checkpoints/` and is excluded from
Git. The downloaded kitchen assets live in the ignored asset directories of the
RoboCasa submodule. Neither artifact should be committed.

RoboCasa may print optional warnings about `robosuite_models`, Mink, or
MimicGen. Those packages provide robots and dataset-generation features that
the `PandaOmron` π0.5 evaluation path does not use.

## Backend boundary

`showcase/backend_registry.py` is the dependency-free compatibility manifest.
`showcase/robocasa_runtime.py` implements task discovery, create/reset, policy
observation preparation, action validation/conversion, rendering, horizon
selection, and session state. `showcase/interactive_robocasa.py` implements both
browser-controlled and automatic multi-task/multi-trial rollouts. The shared
dashboard/report layer consumes the same state contract for both backends.

Do not allow the UI to choose an incompatible model profile. Selecting RoboCasa
must select the RoboCasa config/checkpoint and display `12D actions`; selecting
LIBERO must select `pi05_libero` and display `7D actions`.

## Upstream references

- RoboCasa: <https://github.com/robocasa/robocasa>
- RoboCasa OpenPI fork: <https://github.com/robocasa-benchmark/openpi>
- Public checkpoint: <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000>
- VLA Evaluation Harness: <https://github.com/allenai/vla-evaluation-harness>

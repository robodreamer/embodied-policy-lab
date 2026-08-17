# Embodied Policy Lab

> A local, publishable workbench for plugging vision-language-action policies
> into robot simulators with live rollouts, prompt experiments, success scoring,
> GPU telemetry, and network-audit evidence.

Simulator and model are independent CLI choices. The current compatibility
matrix is:

| Simulator | `pi05` | `groot-n1.5` |
|---|---:|---:|
| LIBERO | ✓ | — |
| RoboCasa365 | ✓ | ✓ |

RoboCasa also has an independent world-model selector. The default
`robocasa-sim` model can compare each real action prefix with a prediction from
a matching cloned MuJoCo environment. Comparison is off by default and never
gates execution; when enabled, the paired clips appear only after the real
prefix finishes. Use `--world-model none` for direct-only execution. The DINO-WM and
JEPA-WM DROID/RoboCasa checkpoints are pinned as diagnostic candidates, but are
not offered as execution gates until their 7D arm-action and temporal contracts
are validated against this lab's 12D mobile-manipulator actions. See
[the world-model guide](docs/world-model-plugins.md).

The policy adapter boundary is documented in
[the model-plugin guide](docs/model-plugins.md). RoboCasa implementation and
validation history remains in [the development notes](docs/robocasa-backend.md),
with GR00T-specific setup and measured results in
[the N1.5 notes](docs/groot-n1.5.md).

Quick RoboCasa checks:

```bash
./scripts/run_robocasa_smoke.sh         # simulator only
./scripts/run_robocasa_policy_smoke.sh  # one real local π0.5 request
./scripts/run_groot_policy_smoke.sh     # one real local GR00T N1.5 request
```

Interactive and automatic RoboCasa rollouts use the original showcase scripts:

```bash
./scripts/run_interactive_showcase.sh --backend robocasa
./scripts/run_interactive_showcase.sh --backend robocasa --model groot-n1.5
./scripts/run_interactive_showcase.sh --backend robocasa \
  --world-model robocasa-sim --compare-world-model
./scripts/run_showcase.sh --backend robocasa --batch --task-id 0 --trials 3
```

For the simplest every-run workflow, use the terminal picker. RoboCasa + π0.5
is listed first and used by `--default`; the picker only offers compatible
model/simulator pairs:

```bash
./lab                   # fzf arrow-key picker, or numbered fallback
./lab --default         # immediately launch RoboCasa + π0.5
./lab --list            # compatibility matrix
./lab --default --dry-run
```

Interactive mode opens task 0 immediately; choose another task from the
dashboard or pass `--task-id ID`. Batch mode asks for a task ID and trial count
unless those values were supplied as flags.

You can preselect part of the configuration and let the picker fill the rest,
or bypass it entirely:

```bash
./lab --backend robocasa --model groot-n1.5
./lab --backend robocasa --model groot-n1.5 --mode batch \
  --task-id 2 --trials 3 --default
```

This repository is a reproducible local deployment of π0.5 and NVIDIA Isaac
GR00T N1.5 in the LIBERO and RoboCasa robot-manipulation simulators. It uses
pinned upstream integrations, robosuite, MuJoCo, and the workstation's NVIDIA
GPU. Simulation rendering is headless through EGL; each rollout is saved as an
MP4 for inspection.

The project is structured for publication: `upstream-openpi` is a pinned Git
submodule, while model weights and generated machine-specific artifacts are
excluded from version control.

## Status

- Upstream sources: Physical Intelligence OpenPI and RoboCasa's NVIDIA Isaac GR00T fork
- Pinned revisions: Git submodules plus `UPSTREAM_COMMIT` for the original OpenPI baseline
- Models: `pi05_libero`, `pi05_pretrain_human300`, and `gr00t_n1.5_robocasa365_120k`
- Policy runtimes: isolated JAX/CUDA and PyTorch/CUDA environments
- Simulator runtimes: Python 3.8 LIBERO, Python 3.11 RoboCasa/OpenPI, and Python 3.12 RoboCasa/GR00T
- Terminal-picker default: RoboCasa + π0.5; the legacy low-level CLI still defaults to LIBERO

Initial LIBERO validation on August 5, 2026 completed all 10/10 smoke-test episodes.
See `results/README.md` for measured runtime and GPU telemetry.

The first LIBERO run downloads approximately 11.6 GiB into `cache/openpi`.
RoboCasa assets use approximately 23 GB in the current release. The π0.5
inference checkpoint is approximately 12 GB; the inference-only GR00T download
is approximately 7.6 GB. Later runs reuse all caches.

## Live showcase

The showcase opens a local browser console displaying:

- an active-profile bar with the selected model, simulator, task collection,
  and local transport;
- an animated startup banner while policy weights, the simulator, and the scene
  are loading;
- full-resolution external and wrist-camera MuJoCo views, rendered separately
  from the lower-resolution observations used by the policy pipeline;
- the active language command and episode progress;
- the profile-specific action chunk (10×7D LIBERO π0.5, 50×12D RoboCasa
  π0.5, or 16×12D RoboCasa GR00T);
- cold/startup and warm inference latency;
- GPU utilization, VRAM, power, and temperature;
- task success and the local policy endpoint;
- optional post-execution prediction/actual clips with before/after live-state
  and predicted/actual final-state hashes;
- a syscall-level audit of model and simulator network destinations.

Set up once, then launch:

```bash
git submodule update --init --recursive
./scripts/setup.sh
./scripts/capture_system_info.sh
./scripts/run_showcase.sh
```

For RoboCasa, run its one-time setup instead of `setup.sh`:

```bash
ROBOCASA_DOWNLOAD_ASSETS=1 ROBOCASA_DOWNLOAD_CHECKPOINT=1 \
  ./scripts/setup_robocasa.sh
./scripts/run_interactive_showcase.sh --backend robocasa
```

For NVIDIA Isaac GR00T N1.5, use its isolated setup and the same launcher:

```bash
GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh
./scripts/run_interactive_showcase.sh \
  --backend robocasa --model groot-n1.5
```

The two large policies should be run one at a time on this 24 GB GPU. The
launcher owns the selected server lifecycle, so ordinary runs already do this.

The browser opens at <http://127.0.0.1:8085>. By default, the showcase runs task
0 from `libero_spatial`, records one rollout, and audits network syscalls from
the policy server and simulator. A normal cached run should report only the
local policy connection at `127.0.0.1:8000` and no remote IP destination.

Every run is retained under a timestamped `showcase-runs/` directory, and
`showcase-runs/latest` points to the newest one. Important artifacts include:

- `report.md` and `summary.json`: portable performance and network results;
- `state.json`: dashboard snapshot, actions, task state, and latency samples;
- `gpu.csv`: one-second NVIDIA telemetry;
- `network-*`: raw `strace` network syscall evidence;
- `videos/*.mp4`: simulator rollouts;
- `previews/*.mp4` and `preview-audit.jsonl`: paired prediction/actual outcomes,
  non-mutation evidence, and final-state comparisons;
- `client.log`, `server.log`, and `dashboard.log`: runtime trails.

Review the last session or build an MP4 reel with:

```bash
./scripts/view_latest_showcase.sh
./scripts/build_montage.sh
```

Useful showcase controls:

| Environment variable | Default | Purpose |
|---|---:|---|
| `BACKEND` | `libero` | Select `libero` or `robocasa` |
| `MODEL` | `pi05` | Select `pi05` or `groot-n1.5` |
| `POLICY_PORT` | `8000` | Local policy port (`PI05_PORT` remains a legacy alias) |
| `TASK_SUITE` | backend default | Select a LIBERO suite or RoboCasa task set |
| `TASK_IDS` | `0` | Comma-separated task IDs, or `all` |
| `TRIALS_PER_TASK` | `1` | Episodes per selected task |
| `ROBOCASA_SPLIT` | `target` | Select the RoboCasa `target` or `pretrain` split |
| `VIEWER_WIDTH` | `960` | RoboCasa high-resolution dashboard width |
| `VIEWER_HEIGHT` | `540` | RoboCasa high-resolution dashboard height |
| `VIEWER_FPS` | `6` | Maximum RoboCasa dashboard render rate |
| `REALTIME_DELAY_MS` | `35` | Slow simulator steps for easier live viewing |
| `AUTO_OPEN` | `1` | Open the dashboard in the desktop browser |
| `HOLD_OPEN` | `0` | Keep the dashboard running after completion |
| `NETWORK_AUDIT` | `1` | Trace and report network destinations |
| `INITIAL_PROMPT` | empty | Initial override in interactive mode |
| `LOCAL_LLM_URL` | empty | Loopback Ollama or chat-completions endpoint |
| `LOCAL_LLM_MODEL` | empty | Model served by the local LLM endpoint |
| `LOCAL_LLM_NUM_GPU` | `0` | Ollama GPU layers; CPU is the safe default while a VLA owns VRAM |
| `ALLOW_GPU_OVERSUBSCRIPTION` | `0` | Override the safety stop when another GPU compute process is present |

For example, run three tasks and keep the completed dashboard open:

```bash
TASK_IDS=0,1,2 HOLD_OPEN=1 ./scripts/run_showcase.sh
```

The scripts also expose a flag-based CLI. Run
`./scripts/run_showcase.sh --help` for the full reference. Examples:

```bash
# Persistent browser-controlled RoboCasa session
./scripts/run_interactive_showcase.sh \
  --backend robocasa --task-set atomic_seen --split target

# Three automatic trials for each selected RoboCasa task
./scripts/run_showcase.sh \
  --backend robocasa --batch --task-ids 0,2 --trials 3 --budget 1

# Existing LIBERO behavior remains available
./scripts/run_showcase.sh \
  --backend libero --task-suite libero_spatial --task-ids 0,1
```

RoboCasa's browser views default to independent 960×540 MuJoCo renders at up to
6 FPS. Counterfactual EGL contexts are active only while a prediction is being
rendered, preventing them from changing the live cameras. The official square
camera observations and model-side 224×224 transforms remain unchanged.
Override presentation quality without modifying code:

```bash
./scripts/run_interactive_showcase.sh \
  --backend robocasa \
  --viewer-width 1280 --viewer-height 720 --viewer-fps 8
```

The dashboard labels the MuJoCo presentation, policy-camera, and model-input
resolutions separately.

The network audit produces evidence; it is not a firewall or network namespace.
Unprivileged network namespaces are disabled on the validation workstation, so
the report records every observed IP destination instead of claiming hard kernel
isolation. A `loopback_only` verdict means no remote inference connection was
observed during that run.

## Interactive experiment console

Launch a persistent session when you want to reset the simulator, switch tasks,
edit the selected policy's instruction while it is running, and accumulate success rates:

```bash
./scripts/run_interactive_showcase.sh
```

Add `--backend robocasa` to use RoboCasa and its matching local checkpoint. The
task dropdown prepares a selected kitchen scene far enough to publish its exact
scene-dependent canonical instruction before prompt generation or execution.
`--task-id` selects only the scene shown when the session opens; it does not
restrict the dropdown. The default `atomic_seen` collection exposes 18 tasks.
Choose a broader registered RoboCasa collection at launch when useful:

```bash
# Populate the same browser dropdown with all 65 atomic tasks.
./scripts/run_interactive_showcase.sh \
  --backend robocasa --task-set all_atomic_tasks

# Or use the 16-task seen composite collection.
./scripts/run_interactive_showcase.sh \
  --backend robocasa --task-set composite_seen
```

The browser now waits for you instead of starting automatically. Its workflow is:

1. Choose the **evaluation task**, which sets the scene and success scoring.
2. Review or edit the **instruction sent to the local policy**.
3. Choose **result handling** and a **rollout budget**. Use Scored only when the
   instruction still means exactly the selected evaluation goal. Standard uses
   the upstream benchmark limit, Extended allows 2× as many actions, and Long
   allows 3×.
4. Press **Start a fresh scored rollout**.

Extended is the default and is recommended for language variants. A manual text
edit defaults to a Long, unscored custom experiment; switch it back to Scored
only for a true paraphrase of the selected goal. Locally generated scored
variations automatically stage Extended; exploratory commands stage Long and
unscored. A scored rollout still stops early when the selected simulator detects
its goal, so a fast successful variation is expected. Results and success-rate
grouping record the step budget, preventing Standard and Extended attempts from
being treated as identical trials. Custom experiments ignore that unrelated
simulator completion signal and run their full selected budget unless you stop
them; their history result is `unscored` rather than a misleading success or
failure against the original goal.

Changing the dropdown only stages a choice; it never resets a running attempt.
During a rollout, **Apply draft to this rollout** changes the instruction and
forces a replan without resetting the scene. Because that creates a mixed-prompt
attempt, the dashboard excludes it from per-prompt rates. **Abort & start a fresh
rollout** resets the simulator with the staged task and instruction. **Finish &
save report** stops inference, writes artifacts, and leaves the dashboard open
in review mode until you press Ctrl+C in the terminal.

The dashboard keeps a submitted draft visible until the simulator acknowledges
the command *and* publishes the matching prompt, task, and rollout budget. This
avoids briefly restoring the previous instruction while a fresh environment is
being constructed.

The dashboard retains completed, failed, mixed-prompt, and aborted attempts.
`summary.json` records prompt timelines and per-task/prompt totals.

An important interpretation detail: selecting an evaluation task chooses the world,
initial state, and hidden success predicate. Editing only the prompt does not
change that predicate. This makes prompt edits useful for testing paraphrase,
ambiguity, or adversarial-instruction robustness against a fixed goal.

The default optional prompt generator is the compact `gemma3:1b` model served by
Ollama. Once Ollama is installed, prepare it with:

```bash
./scripts/setup_local_llm.sh
```

The interactive launcher detects the local server and model automatically. The
dashboard then shows **Local generator ready: gemma3:1b** and enables **Generate
random draft**. Choose either:

- **Random scored variation** to produce a new wording while preserving the
  selected task's exact goal; or
- **Random exploratory command** to invent a different plausible pick/place
  action using objects named in the scene. Exploratory and mixed-prompt attempts
  are excluded from success rates because the simulator's hidden goal has not changed.

The generator records and avoids prior outputs from the session. Generated text
remains a draft until you explicitly start a new rollout or apply it to the
current rollout.

Ollama prompt generation defaults to CPU (`LOCAL_LLM_NUM_GPU=0`) so a compact
LLM cannot evict or exhaust the 24 GB GPU while π0.5 or GR00T is resident. This
is slower than GPU prompt generation but keeps policy inference stable. Override
the value only on a machine with enough spare VRAM or a separate prompt GPU.

For every synchronous inference request, the client records the exact prompt,
its SHA-256 digest, the returned action-chunk digest, attempt, step, and latency
in `inference-audit.jsonl`. The dashboard shows the last acknowledged prompt
digest. This proves which text was serialized to the local policy server even
when the robot's visible behavior does not change. Each released checkpoint is
trained around its backend's task distribution; a novel or physically
unsupported command may still produce familiar behavior despite receiving
different text.

To use another loopback-only Ollama or OpenAI-compatible server, override:

```bash
LOCAL_LLM_URL=http://127.0.0.1:11434/api/generate \
LOCAL_LLM_MODEL=your-model \
./scripts/run_interactive_showcase.sh
```

For an OpenAI-compatible local server such as LM Studio, use its local
`/v1/chat/completions` URL and model name. The launcher rejects non-loopback LLM
URLs so prompt generation cannot silently become a hosted call. Ollama/LM Studio
remains optional; policy inference does not require either one.

## Full-suite smoke test

The original evaluator remains available for a straightforward 10-task check:

```bash
./scripts/run_smoke_test.sh
```

The smoke test automatically starts and stops the policy server. Results are in:

- `videos/libero_spatial/*.mp4`: rollouts named with `success` or `failure`;
- `logs/client-*.log`: task descriptions and cumulative success counts;
- `logs/server-*.log`: checkpoint and policy-server messages;
- `logs/gpu-*.csv`: GPU memory, utilization, and power samples;
- `logs/system-info.txt`: hardware, OS, runtime, and source revision.

## Run server and client separately

This is useful when experimenting or watching server output live.

Terminal 1:

```bash
./scripts/run_server.sh
```

Terminal 2:

```bash
./scripts/run_client.sh
```

The client defaults to one trial for every task in `libero_spatial`. Override its
settings with environment variables:

```bash
TASK_SUITE=libero_10 TRIALS_PER_TASK=1 SEED=7 ./scripts/run_client.sh
```

Supported suites in the official evaluator are `libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`, and `libero_90`.

## Watch a rollout

Use any MP4 player, for example:

```bash
ffplay videos/libero_spatial/rollout_*.mp4
```

Shell wildcard expansion may open multiple files. Pass one exact filename when
you want a single rollout.

## Reproducibility notes

- The two Python environments live under `upstream-openpi/.venv` and
  `upstream-openpi/examples/libero/.venv`.
- LIBERO's path configuration is kept in `config/libero`, not in `~/.libero`.
- Model assets are kept in `cache/openpi`, not in `~/.cache/openpi`.
- The upstream checkout is intentionally left unmodified. Local orchestration
  lives in this folder's `scripts` directory.
- `results/README.md` is the publishable location for sanitized validation
  summaries; raw runtime artifacts stay local by default.
- A 10/10 one-rollout smoke test demonstrates that the stack works; it is not a
  statistically meaningful benchmark. The official evaluator defaults to 50
  rollouts per task.

## System prerequisites installed during initial setup

Ubuntu packages required by the upstream dependency pins:

```bash
sudo apt-get install -y linux-libc-dev build-essential git-lfs curl strace ffmpeg
```

Docker is not required for this installation. The official Docker Compose route
remains available in `upstream-openpi/examples/libero/README.md` if Docker and
the NVIDIA Container Toolkit are installed later.

## Troubleshooting

If EGL initialization fails, try GLX from a desktop session:

```bash
MUJOCO_GL=glx PYOPENGL_PLATFORM=glx ./scripts/run_client.sh
```

If port 8000 is occupied, use the same alternate port for both processes:

```bash
PI05_PORT=8010 ./scripts/run_server.sh
PI05_PORT=8010 ./scripts/run_client.sh
```

If a first checkpoint download was interrupted, rerun the server. The official
downloader uses a `.partial` directory and finalizes it after all files arrive.

Older installations may print the same `datasets path ... does not exist`
warning repeatedly. That directory contains optional training demonstrations and
is not used by this evaluator. Rerun `./scripts/setup.sh`; it now creates the
configured directory so the harmless warning does not obscure rollout results.

## Sources

- Official openpi repository: <https://github.com/Physical-Intelligence/openpi>
- Official LIBERO example: <https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero>
- Recent local reproduction: <https://note.com/npaka/n/n1eb56d6be1c7?hl=en>

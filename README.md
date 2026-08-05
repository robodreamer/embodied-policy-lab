# π0.5 Sim Lab

This folder is a reproducible local deployment of Physical Intelligence's official
`pi05_libero` checkpoint in the LIBERO robot-manipulation simulator. It uses the
official `openpi` policy server, LIBERO/robosuite, MuJoCo, and the workstation's
NVIDIA GPU. Simulation rendering is headless through EGL; each rollout is saved
as an MP4 for inspection.

The project is structured for publication: `upstream-openpi` is a pinned Git
submodule, while model weights and generated machine-specific artifacts are
excluded from version control.

## Status

- Upstream source: `Physical-Intelligence/openpi`
- Pinned commit: see `UPSTREAM_COMMIT`
- Model: `gs://openpi-assets/checkpoints/pi05_libero`
- Policy runtime: Python 3.11 + JAX/CUDA
- Simulator runtime: Python 3.8 + LIBERO + robosuite + MuJoCo
- Default smoke test: one seeded rollout for each of 10 `libero_spatial` tasks

Initial validation on August 5, 2026 completed all 10/10 smoke-test episodes.
See `results/README.md` for measured runtime and GPU telemetry.

The first run downloads approximately 11.6 GiB of checkpoint data into
`cache/openpi`. Later runs reuse that cache.

## Live showcase

The showcase opens a local browser console displaying:

- live external and wrist-camera model inputs;
- the active language command and episode progress;
- the predicted 10-step, 7-dimensional action chunk;
- cold/JIT and warm inference latency;
- GPU utilization, VRAM, power, and temperature;
- task success and the local policy endpoint;
- a syscall-level audit of model and simulator network destinations.

Set up once, then launch:

```bash
git submodule update --init --recursive
./scripts/setup.sh
./scripts/capture_system_info.sh
./scripts/run_showcase.sh
```

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
- `client.log`, `server.log`, and `dashboard.log`: runtime trails.

Review the last session or build an MP4 reel with:

```bash
./scripts/view_latest_showcase.sh
./scripts/build_montage.sh
```

Useful showcase controls:

| Environment variable | Default | Purpose |
|---|---:|---|
| `TASK_SUITE` | `libero_spatial` | Select a supported LIBERO suite |
| `TASK_IDS` | `0` | Comma-separated task IDs, or `all` |
| `TRIALS_PER_TASK` | `1` | Episodes per selected task |
| `REALTIME_DELAY_MS` | `35` | Slow simulator steps for easier live viewing |
| `AUTO_OPEN` | `1` | Open the dashboard in the desktop browser |
| `HOLD_OPEN` | `0` | Keep the dashboard running after completion |
| `NETWORK_AUDIT` | `1` | Trace and report network destinations |
| `INITIAL_PROMPT` | empty | Initial override in interactive mode |
| `LOCAL_LLM_URL` | empty | Loopback Ollama or chat-completions endpoint |
| `LOCAL_LLM_MODEL` | empty | Model served by the local LLM endpoint |

For example, run three tasks and keep the completed dashboard open:

```bash
TASK_IDS=0,1,2 HOLD_OPEN=1 ./scripts/run_showcase.sh
```

The network audit produces evidence; it is not a firewall or network namespace.
Unprivileged network namespaces are disabled on the validation workstation, so
the report records every observed IP destination instead of claiming hard kernel
isolation. A `loopback_only` verdict means no remote inference connection was
observed during that run.

## Interactive experiment console

Launch a persistent session when you want to reset the simulator, switch tasks,
edit the π0.5 instruction while it is running, and accumulate success rates:

```bash
./scripts/run_interactive_showcase.sh
```

Use the browser controls to apply a typed prompt immediately, reset into a new
rollout, change tasks, and stop cleanly to generate the final report. Applying a
prompt discards the queued action chunk, so π0.5 replans from the next
observation. The dashboard retains completed, failed, and aborted attempts and
shows the running success rate; `summary.json` also records per-prompt totals.

An important interpretation detail: selecting a LIBERO task chooses the world,
initial state, and hidden success predicate. Editing only the prompt does not
change that predicate. This makes prompt edits useful for testing paraphrase,
ambiguity, or adversarial-instruction robustness against a fixed goal.

To enable the **Generate locally** button, point it at a loopback-only Ollama or
OpenAI-compatible server. For Ollama, for example:

```bash
LOCAL_LLM_URL=http://127.0.0.1:11434/api/generate \
LOCAL_LLM_MODEL=gemma3:1b \
./scripts/run_interactive_showcase.sh
```

For an OpenAI-compatible local server such as LM Studio, use its local
`/v1/chat/completions` URL and model name. The launcher rejects non-loopback LLM
URLs so prompt generation cannot silently become a hosted call. Ollama/LM Studio
is optional and is not installed by setup.

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
sudo apt-get install -y linux-libc-dev build-essential git-lfs strace ffmpeg
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

## Sources

- Official openpi repository: <https://github.com/Physical-Intelligence/openpi>
- Official LIBERO example: <https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero>
- Recent local reproduction: <https://note.com/npaka/n/n1eb56d6be1c7?hl=en>

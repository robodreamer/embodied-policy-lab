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

## Quick start

From this directory:

```bash
git submodule update --init --recursive
./scripts/setup.sh
./scripts/capture_system_info.sh
./scripts/run_smoke_test.sh
```

The smoke test automatically starts and stops the policy server. Results are in:

- `videos/libero_spatial/*.mp4`: rollout videos, named with `success` or `failure`
- `logs/client-*.log`: task descriptions and cumulative success counts
- `logs/server-*.log`: checkpoint loading and policy-server messages
- `logs/gpu-*.csv`: one-second GPU memory, utilization, and power samples
- `logs/system-info.txt`: hardware, OS, runtime, and source revision

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
sudo apt-get install -y linux-libc-dev build-essential git-lfs
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

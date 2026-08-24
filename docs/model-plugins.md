# Model plugins

Embodied Policy Lab keeps simulator behavior separate from policy inference.
The CLI selects both axes explicitly:

```bash
./scripts/run_interactive_showcase.sh \
  --backend robocasa \
  --model groot-n1.5
```

The interactive `./lab` terminal picker reads the same compatibility choices
and defaults to RoboCasa + π0.5. It does not implement a second execution path;
it prints and invokes the shared showcase command.

`showcase/backend_registry.py` is the dependency-free compatibility manifest.
It describes simulators, policies, aliases, action horizons, transports, and
valid pairings. `showcase/robocasa_policy_plugins.py` and
`showcase/libero_policy_plugins.py` are the adapter boundaries.
Every adapter implements three operations:

1. identify its endpoint and policy metadata;
2. prepare a native simulator observation and exact prompt for its wire format;
3. infer and return the simulator's canonical `[horizon, action_dimension]`
   action matrix.

The shared rollout engine owns tasks, resets, rendering, prompt changes,
success scoring, latency, audit hashes, and reports. It does not import OpenPI
or GR00T directly.

## Current plugins

| Plugin key | Simulator | Transport | Observation | Action chunk |
|---|---|---|---|---:|
| `pi05` | LIBERO | WebSocket | 2 cameras + 8D state + prompt | 10 × 7D |
| `fastwam` | LIBERO | localhost HTTP | 2 cameras + 8D state + prompt | 32 × 7D |
| `flexpi` | LIBERO | localhost HTTP | 2 RGB + 2 metric-depth cameras + 8D state + prompt | 32 × 7D |
| `pi05` | RoboCasa | WebSocket | 3 cameras + 16D state + prompt | 50 × 12D |
| `groot-n1.5` | RoboCasa | ZeroMQ | 3 cameras + named state fields + language annotation | 16 × 12D |

The GR00T adapter follows RoboCasa's official `panda_omron` data config. It
sends left scene, right scene, and wrist observations with a one-frame time
dimension and the instruction as the one-item language history expected by the
N1.5 inference helper. Its five named action outputs are concatenated in
official order: end-effector position, end-effector rotation, gripper, mobile
base, and control mode. That yields the 12D action accepted by RoboCasa's
`convert_action`.

Fast-WAM intentionally runs in a separate Python 3.10 PyTorch process. Its
localhost HTTP adapter sends lossless uint8 camera arrays and float32 state,
then validates a finite 32×7 response. The server applies the released
two-camera preprocessing and dataset statistics. It caches text embeddings,
but re-encodes the current image at every policy query. This distinction is a
required correctness invariant for closed-loop validation.

Flex-π also runs in its own pinned Python 3.10 / CUDA 12.8 / MuJoCo 3.3.2
environment. Its lossless HTTP boundary adds aligned uint16 millimetre depth.
The client defaults to world-action co-generation (`full-joint` internally)
and can switch to `action-only` between rollouts. Co-generation returns future
RGB along with the DINO, pointmap, and end-effector action streams. The
dashboard buffers every matched prefix privately, compiles a full-rollout
sample timeline, and publishes external-camera clips above wrist-camera clips,
with per-view RGB PSNR, below the live cameras only after the complete rollout
ends. Exact 448×512 composites and per-prefix offsets remain as artifacts. This
is evidence about the checkpoint's learned dynamics, not an independent model
comparison.

## Integrated WAM policy tradeoffs

Fast-WAM and Flex-π are the two executable world-action policy plugins in this
repository. π0.5 and GR00T N1.5 are VLA policy baselines; the RoboCasa simulator
oracle is a deterministic diagnostic; none of those three should be described
as an integrated learned WAM.

| Dimension | Fast-WAM | Flex-π |
|---|---|---|
| Test-time path | Uses the video-trained representation to denoise actions, but skips explicit future-video generation | Selectable action-only or joint denoising of actions and future world streams |
| World output visible here | None from the validated released action path | RGB future plus internal DINO semantic and 3D pointmap streams in `full-joint` mode |
| Observation contract | Two 224×224 RGB cameras + 8D state + language | Two 256×256 RGB cameras, aligned metric depth and intrinsics + 8D state + language |
| Action contract | 32×7 end-effector/gripper chunk; 10 denoising steps | 32×7 end-effector/gripper chunk; 4 Euler steps |
| Local 24 GB result | 14,288 MiB process peak; warm live requests about 1.6–1.86 s | 14,671 MiB action-only or 17,961 MiB full-joint process peak; measured model pass 1.291 s or 2.980 s |
| Strongest use in this lab | Lower-compute WAM-derived control baseline and closed-loop task evaluation | Same-checkpoint action-only/joint ablation plus post-rollout dynamics inspection |
| Main limitation | No generated future to inspect or score at inference | More sensors, assets, memory, and latency; generated future is correlated evidence, not an independent action scorer |

The [Fast-WAM paper](https://arxiv.org/abs/2603.16666) asks whether the cost of
future imagination is necessary at test time and retains video co-training
while removing future rendering from the deployed policy. The
[Flex-π paper](https://arxiv.org/abs/2608.10860) instead uses stream dropout and
cross-modality forcing so one checkpoint can operate with different subsets of
RGB, DINO, pointmap, and action streams. In full-joint mode, Flex-π does **not**
render several candidate actions and select the best-looking video. Its action
and future tokens are correlated outputs of one denoising process.

The local numbers above are bounded hardware/contract gates, not a fair policy
leaderboard. Fast-WAM has one recorded successful full-budget initial state;
Flex-π still needs matched full-suite rollouts before behavior or success-rate
claims are justified. Use Fast-WAM when control throughput and minimal runtime
surface are the priority. Use Flex-π full-joint when the experiment needs
visible learned-dynamics evidence, and compare it with Flex-π action-only to
isolate the added cost and behavioral value of co-generation.

For a matched closed-loop comparison, use
`./scripts/benchmark_wam_libero.py --profile smoke|pilot|paper`. The runner
shares the Flex-π LIBERO/MuJoCo 3.3.2 simulator runtime across both policies,
freezes task IDs, seeds, 10-action replanning, and suite budgets, and reports
success confidence intervals, warm batch-1 latency, and peak VRAM. Details and
claim limits are in
[the dated benchmark note](../notes/2026-08-24-fastwam-flexpi-headless-libero-benchmark.md).

## Adding another model

1. Add a `PolicySpec` and aliases in `showcase/backend_registry.py`.
2. Add the valid simulator pairing to `PROFILES`.
3. Implement an adapter in `showcase/robocasa_policy_plugins.py` (or the
   corresponding simulator plugin module).
4. Add a server launch case in `scripts/run_server.sh` and an isolated setup
   script. Keep model-specific imports inside the adapter/runtime.
5. Add contract tests for observation keys, prompt serialization, output order,
   horizon, dimensions, and non-finite actions.
6. Run one audited local request and one short rollout before claiming support.

Checkpoint weights and virtual environments remain ignored. In-tree upstream
integrations are pinned Git submodules. Fast-WAM and Flex-π use
revision-checked sibling checkouts because their independent runtimes and
release assets are not part of this repository.

## GR00T setup

```bash
GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh
```

The setup downloads only the N1.5 inference files, excluding roughly 8.5 GB of
optimizer/trainer state. It creates `upstream-robocasa-groot/.venv` with Python
3.12, CUDA 12.8 PyTorch, and a prebuilt FlashAttention wheel compatible with
this workstation's Blackwell GPU.

See [the GR00T N1.5 notes](groot-n1.5.md) for pinned sources, packaging choices,
and the latest measured local validation.

Then use the normal interactive or batch CLI:

```bash
# Interactive dashboard
./scripts/run_interactive_showcase.sh --backend robocasa --model groot-n1.5

# One automatic task attempt
./scripts/run_showcase.sh --backend robocasa --model groot-n1.5 \
  --batch --task-id 0 --trials 1
```

GR00T defaults to executing its full 16-action prediction before replanning.
Override with `--replan-steps`, up to 16, when comparing closed-loop frequency.

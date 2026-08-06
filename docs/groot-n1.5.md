# NVIDIA Isaac GR00T N1.5 + RoboCasa notes

## Scope

This integration uses the official RoboCasa benchmark fork of NVIDIA Isaac
GR00T at pinned commit `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` and the
public RoboCasa365 `checkpoint-120000`. It is an N1.5 compatibility plugin, not
an assertion that newer GR00T checkpoints share the same architecture or data
contract.

Sources:

- RoboCasa GR00T fork: <https://github.com/robocasa-benchmark/Isaac-GR00T>
- RoboCasa foundation-model docs: <https://robocasa.ai/docs/build/html/benchmarking/foundation_model_learning.html>
- N1.5 checkpoint: <https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000>
- Current NVIDIA Isaac GR00T: <https://github.com/NVIDIA/Isaac-GR00T>

## Runtime decisions

The RoboCasa N1.5 fork predates this workstation's Blackwell GPU. Its optional
desktop dependency set pins PyTorch 2.5.1, which does not provide the required
`sm_120` support here. The isolated environment therefore uses:

- system CPython 3.12;
- PyTorch 2.9.0 + CUDA 12.8;
- FlashAttention 2.8.3's prebuilt CPython 3.12 / torch 2.9 / CUDA 12 wheel;
- the fork's pinned Transformers 4.51.3 and model source.

The measured policy peak is close to the machine's 24 GB limit. The launcher
therefore refuses to start while an unrelated GPU compute process is resident.
It stops only the configured Ollama prompt model, never arbitrary workloads;
`ALLOW_GPU_OVERSUBSCRIPTION=1` is available for intentional expert use. Prompt
generation asks Ollama for CPU execution by default.

The setup intentionally omits ONNX 1.15 and TensorRT export tooling. ONNX 1.15
has no CPython 3.12 wheel and is not imported by PyTorch policy inference. It
also omits optional RoboCasa dataset conversion and keyboard-device packages
whose `tianshou`/`evdev` pins conflict with the policy environment. Simulator,
policy server, dashboard, report, and video paths are installed and verified.

Checkpoint download includes only `config.json`, both safetensor shards, the
weight index, and `experiment_cfg/metadata.json`. Optimizer, scheduler, RNG,
and trainer state are excluded. The resulting local inference directory is
about 7.1 GiB on disk.

## Observation and action contract

The plugin follows the official `panda_omron` configuration:

- three RGB observations: left scene, right scene, and eye-in-hand;
- end-effector position/rotation, gripper, base position, and base rotation;
- exact active instruction as `annotation.human.task_description`;
- embodiment tag `new_embodiment`;
- four denoising steps;
- 16 predicted actions per request.

The named GR00T outputs are concatenated in the official order to form the
RoboCasa 12D control accepted by `convert_action`.

## Local validation — August 6, 2026

The standalone smoke test loaded the real checkpoint, reset RoboCasa task 0
(`CloseBlenderLid`, target split, seed 7), sent the exact canonical prompt and
three views, received a finite `[16, 12]` chunk, and executed its first action.
Initial request latency was 917.26 ms.

The shared top-level CLI then completed a full 900-step automatic rollout:

```bash
./scripts/run_showcase.sh --backend robocasa --model groot-n1.5 \
  --batch --task-id 0 --trials 1 --budget 1 \
  --realtime-delay-ms 0 --no-open --no-hold-open --network-audit
```

Measured results:

| Measurement | Result |
|---|---:|
| Policy requests | 57 |
| Action response shape | `16 × 12` |
| Cold request | 826.85 ms |
| Warm mean / median / P95 | 227.04 / 165.44 / 415.67 ms |
| Peak GPU memory | 22,092 MiB |
| Peak GPU utilization | 100% |
| Session duration | 59.58 s |
| Network audit | `loopback_only` |
| Task result | 0/1 (failure) |

One failed rollout is a functional integration measurement, not a model-quality
benchmark. It confirms local model loading, repeated closed-loop requests,
correct dimensions, simulator stepping, artifact generation, and absence of a
remote inference connection. Use the official evaluation protocol and many
seeds/tasks for a meaningful success-rate comparison.

## Commands

```bash
# Install runtime and fetch inference weights
GROOT_DOWNLOAD_CHECKPOINT=1 ./scripts/setup_groot.sh

# One request + one simulator action
./scripts/run_groot_policy_smoke.sh

# Interactive dashboard
./scripts/run_interactive_showcase.sh --backend robocasa --model groot-n1.5

# Automatic scored evaluation
./scripts/run_showcase.sh --backend robocasa --model groot-n1.5 \
  --batch --task-ids 0,1,2 --trials 3 --budget 1
```

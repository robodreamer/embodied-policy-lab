# π0.5 on RoboCasa validation

## Scope

The RoboCasa profile runs the public `pi05_pretrain_human300` checkpoint with
the official RoboCasa observation/action transforms and success predicates. It
is an isolated runtime rather than a projection of the 7D LIBERO profile onto
RoboCasa's mobile-manipulator contract.

This document records the public behavior contract, setup, and bounded local
validation. Historical implementation and debugging narratives are maintained
outside the public repository.

## Run it

```bash
# Runtime only
./scripts/setup_robocasa.sh

# Public simulator assets and inference-only checkpoint files
ROBOCASA_DOWNLOAD_ASSETS=1 ROBOCASA_DOWNLOAD_CHECKPOINT=1 \
  ./scripts/setup_robocasa.sh

# Model-free simulator gate
./scripts/run_robocasa_smoke.sh

# One real request and a short executed prefix
./scripts/run_robocasa_policy_smoke.sh

# Interactive or automatic evaluation
./lab --backend robocasa --model pi05
./scripts/run_showcase.sh --backend robocasa --model pi05 \
  --batch --task-set atomic_seen --task-id 2 --trials 1 --budget 1
```

## Observation and action contract

| Field | RoboCasa π0.5 profile |
|---|---|
| OpenPI config | `pi05_pretrain_human300` |
| Checkpoint | `pi05_pretrain_human300/multitask_learning/75000` |
| Cameras | left scene, right scene, eye-in-hand |
| State | 16D mobile-manipulator state |
| Returned action | 50 × 12D chunk |
| Default robot | `PandaOmron` |
| Default task collection | `atomic_seen` |
| Success authority | RoboCasa `info["success"]` |

The simulator, robosuite controller, policy transforms, and checkpoint are
pinned upstream components. The policy and simulator communicate through the
same local policy boundary used by the shared dashboard.

## Inference-only checkpoint workaround

The upstream training config retains dataset paths used to compute fallback
normalization statistics. On an inference-only machine those training datasets
are absent even though the checkpoint already contains `assets/norm_stats.json`.

`showcase/serve_robocasa_policy.py` loads the checkpoint statistics first,
clears the unavailable training paths in an in-memory config copy, and passes
the statistics to the unchanged upstream policy factory. It does not patch the
submodule, model weights, transforms, or transport protocol.

## Bounded local validation — 2026-08-05

Validated on an RTX PRO 5000 Blackwell Laptop GPU with approximately 24 GB of
VRAM:

| Gate | Result |
|---|---|
| Model-free reset/render/step | passed on `CloseBlenderLid` |
| Real policy smoke | finite 50×12 action chunk; first five actions executed |
| Complete automatic episode | 450 actions and 90 local policy requests; `0/1` |
| Interactive control | task switch, prompt acknowledgement, abort accounting, report and cleanup passed |
| Network audit | `loopback_only`; only the configured local policy endpoint observed |
| LIBERO regression after backend addition | existing π0.5 LIBERO path completed `1/1` |

The complete RoboCasa episode had a 134.49 ms warm median request latency,
141.16 ms warm P95, and 21,045 MiB peak observed GPU memory. The unsuccessful
single episode is integration evidence, not a policy-quality estimate.

## Interpretation boundary

- A simulator smoke does not prove that policy inference ran.
- A five-action policy smoke is not a task attempt.
- One complete seed cannot establish a success rate.
- RoboCasa results cannot be ranked directly against LIBERO results because
  the cameras, state, action space, robot, task distribution, and success
  predicates differ.

## Sources

- [RoboCasa](https://github.com/robocasa/robocasa)
- [RoboCasa OpenPI fork](https://github.com/robocasa-benchmark/openpi)
- [Public RoboCasa checkpoints](https://huggingface.co/robocasa/robocasa365_checkpoints)
- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)

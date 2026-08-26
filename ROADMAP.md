# Embodied Policy Lab roadmap

Embodied Policy Lab is focused on local evaluation and observability for robot
VLAs and world-action models. The project complements upstream training
frameworks and simulators by making released models comparable, inspectable,
and reproducible on local hardware.

The central research question is:

> When is future imagination worth its latency and memory cost?

## Current foundation

- One compatibility-aware launcher for π0.5, GR00T N1.5, Fast-WAM, and Flex-π.
- LIBERO and RoboCasa execution with profile-specific observation and action
  contracts.
- Delayed, action-aligned Flex-π prediction-versus-execution replay for both
  external and wrist cameras.
- Durable prompts, actions, task metadata, latency, GPU telemetry, videos,
  source revisions, checkpoint identities, and network-audit evidence.
- Headless Fast-WAM/Flex-π evaluation profiles with resumable result manifests.

## Next

1. **Complete the imagination-ablation matrix.** Add the current Fast-WAM
   first-frame and Optional-IDM modes beside Flex-π action-only and full-joint
   inference under one task, seed, budget, and simulator schedule.
2. **Publish a compact reproducible result bundle.** Include exact revisions,
   checkpoint digests, tasks, seeds, confidence intervals, machine-readable
   results, and representative aligned media.
3. **Add replay-only onboarding.** Let visitors inspect a representative studio
   session without downloading checkpoints or owning a compatible GPU.
4. **Version the compatibility matrix.** Record checkpoint, upstream revision,
   simulator version, validation date, and validation depth for every profile.

## Then

- Add perceptual, semantic, geometry, temporal, and task-relevant future
  metrics instead of relying on RGB PSNR alone.
- Add a cross-run view for success intervals, latency/VRAM tradeoffs, per-task
  failures, inference regimes, and links to exact session evidence.
- Stabilize a versioned artifact schema and demonstrate one externally
  discoverable model plugin.
- Reuse LeRobot policy, processor, dataset, and environment integrations where
  that reduces duplication without weakening the evidence boundary.
- Refresh the deliberately pinned GR00T N1.5 profile toward current GR00T
  releases and expand simulator coverage only when matched evaluation remains
  possible.
- Add `./lab doctor`, download/storage/VRAM estimates, resumable setup, and
  container recipes for stable profiles.

## Scope boundaries

This project is not intended to replace model training repositories, robot
dataset platforms, simulators, or their official leaderboards. Upstream
projects remain authoritative for model architecture and publisher claims.

Public results remain labelled as functional checks, provisional local
comparisons, or complete matched local protocols according to the actual task,
seed, and trial coverage. Different simulators and action contracts are not
ranked as though they were a controlled benchmark.

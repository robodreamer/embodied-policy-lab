# Embodied Policy Lab community positioning and roadmap — 2026-08-26

## Purpose

This note records how Embodied Policy Lab fits into the current open robotics
ecosystem, what is genuinely differentiated, what should not be claimed, and
which improvements would make the project more useful and defensible.

The conclusion is deliberately narrower than “a model-agnostic robotics
platform.” Broad model support is already available elsewhere. The stronger
position is:

> Embodied Policy Lab is a local evaluation and observability studio for VLAs
> and world-action models: run heterogeneous policies under matched conditions,
> compare predicted futures with actual execution, and preserve evidence-grade
> artifacts.

## Community landscape

| Layer | Representative projects | What they primarily own | Embodied Policy Lab's role |
|---|---|---|---|
| General robot-learning platform | [LeRobot](https://github.com/huggingface/lerobot), [LeLab](https://github.com/huggingface/leLab) | datasets, training, many policies, hardware, simulation evaluation, and an accessible GUI | specialize in post-training validation, observability, and comparisons |
| Model-family implementation | [OpenPI](https://github.com/Physical-Intelligence/openpi), [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) | canonical checkpoints, fine-tuning, inference, and deployment for one family | run released models through a common experiment and evidence contract |
| Simulator and benchmark | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [RoboCasa](https://github.com/robocasa/robocasa), [ManiSkill](https://github.com/mani-skill/ManiSkill), [RoboTwin](https://github.com/robotwin-Platform/robotwin) | tasks, environments, datasets, success criteria, and leaderboards | normalize the operator and reporting experience across simulators |
| WAM research implementation | [Fast-WAM](https://github.com/yuantianyuan01/FastWAM), [Flex-π](https://github.com/geyan21/flex-pi) | new architectures, training code, released checkpoints, and publisher evaluation | independently exercise checkpoints and expose their prediction semantics |

LeRobot is the closest functional overlap. Its current repository covers
imitation learning, reinforcement learning, many VLAs, several world models
(including FastWAM), reward models, real-robot control, LeRobotDataset, and a
unified evaluation command. LeLab adds a browser workflow for calibration,
teleoperation, recording, training, inference, and replay, currently centered
on SO-ARM101. Competing with that breadth would dilute this project's strongest
work.

The model repositories and simulators are not direct competitors. They are the
authoritative upstreams that this lab should make easier to validate together.

## What is genuinely distinctive

### Prediction versus completed reality

The Flex-π integration aligns generated external and wrist futures with the
action prefixes actually executed. Predictions are collected silently and
revealed only after execution, so they cannot be mistaken for the live camera
or a planner that selected the action. The result is a replayable diagnostic,
not merely an attractive generated clip.

### One evidence contract around incompatible runtimes

π0.5, GR00T, Fast-WAM, and Flex-π require different Python, CUDA, simulator,
camera, state, and action contracts. The lab preserves those upstream-specific
transforms in isolated runtimes while producing one dashboard and artifact
layout.

### Claims separated from evidence

Publisher numbers, bounded functional checks, matched local comparisons, and
full paper protocols are labeled separately. A one-rollout success is not
presented as benchmark accuracy. This distinction is a product feature because
it makes the output safer to interpret and easier to review.

### Operational observability

Each session can retain the exact instruction, task, seed, action chunk, prompt
hash, request latency, GPU telemetry, network destinations, simulator video,
generated future, source revisions, checkpoint identity, and scored outcome.
The dashboard connects the model's behavior to the evidence required to debug
or reproduce it.

### A coherent WAM research question

Fast-WAM argues that video co-training provides most of the benefit and that
explicit future generation can be removed at inference. Flex-π offers a
runtime frontier from action-only inference to multimodal joint generation.
Putting those paths behind a matched task/seed/budget protocol lets the lab ask:

> When is future imagination worth its latency and memory cost?

The current integration supports the released Fast-WAM direct action path and
Flex-π action-only/full-joint modes. Current Fast-WAM upstream also documents a
compiled action path and an Optional-IDM checkpoint that switches between
first-frame and imagine-then-act inference. Adding those modes would make the
question considerably stronger.

## Selling point

Primary audience: researchers and engineers evaluating a newly released robot
policy or WAM on their own workstation.

Primary promise:

> Test whether a released embodied model actually works on your hardware—and
> inspect not only whether it succeeded, but what it predicted would happen.

Short public phrasing:

> Run the policy. See the future it predicted. Compare it with what actually
> happened.

Developer phrasing:

> One local experiment and evidence contract around incompatible embodied-model
> runtimes.

The README should lead with this workflow rather than a list of supported model
names, because individual models and versions will change quickly.

## Claims to avoid

Do not describe the project as:

- broader than LeRobot;
- a model-training framework;
- a new simulator, benchmark, or leaderboard;
- fully model-agnostic until an external plugin can be added without editing
  central registry code;
- a paper reproduction when using different hardware, budgets, runtimes, or
  checkpoints;
- evidence of model quality when only a smoke test or a handful of trials ran.

“Matched local evaluation” and “research preview” are accurate descriptions of
the present state.

## Improvement roadmap

### Priority 0 — establish the evaluation wedge

1. **Complete the imagination-ablation matrix.** Update the pinned Fast-WAM
   integration and support direct/first-frame, Optional-IDM imagination,
   Flex-π action-only, and Flex-π full-joint modes under one schedule.
2. **Publish a small reproducible result bundle.** Include exact revisions,
   checkpoint digests, tasks, seeds, machine-readable results, confidence
   intervals, aligned media, and the generated comparison report.
3. **Add a replay-only path.** `./lab --replay-demo` should let a visitor inspect
   a representative session without a GPU or model download.
4. **Version the compatibility matrix.** Record checkpoint, upstream revision,
   simulator version, validation date, and validation depth (`load`, `smoke`,
   `rollout`, or `benchmark`) for every profile.

### Priority 1 — deepen the evidence

1. **Use task-relevant future metrics.** RGB PSNR is not sufficient because it
   may reward blur. Add perceptual similarity, DINO feature similarity,
   pointmap/depth error, end-effector or object displacement error, temporal
   consistency, and prediction error versus task outcome.
2. **Add a cross-run comparison view.** Show success with confidence intervals,
   latency/VRAM/success frontiers, per-task failures, inference regimes, and
   direct links to the underlying sessions.
3. **Stabilize the plugin and artifact contracts.** Add a small external example
   plugin and a versioned JSON schema so “model-agnostic” becomes demonstrable.
4. **Add a LeRobot policy adapter.** Reuse its model, processor, dataset, and
   environment ecosystem instead of duplicating its training or hardware
   surface. Keep this project focused on evaluation and observability.

### Priority 2 — expand and distribute

1. Refresh the GR00T path from the deliberately pinned N1.5 compatibility
   integration toward current N1.7 workflows.
2. Expand beyond the current LIBERO and RoboCasa task exposure toward
   RoboCasa365, RoboTwin, or another environment only when the matched evidence
   contract can be preserved.
3. Add `./lab doctor`, storage/VRAM/download estimates, resumable setup, and
   container recipes for the most stable profiles.
4. Publish a tagged research-preview release, rendered documentation, and a
   current end-to-end demo.

## Strategic boundary

LeRobot should own datasets, training, policy breadth, and hardware adapters.
OpenPI, GR00T, Fast-WAM, and Flex-π should remain authoritative for their model
implementations and publisher claims. LIBERO, RoboCasa, ManiSkill, and RoboTwin
should own their environments and benchmark protocols.

Embodied Policy Lab should own the layer that is otherwise easy to neglect:

- repeatable local orchestration;
- predicted-versus-actual execution diagnostics;
- comparable evidence across heterogeneous upstreams;
- auditable, versioned experiment artifacts;
- a public corpus of matched validation results.

The dashboard alone is not a durable moat. The combination of stable adapters,
a trusted artifact schema, reproducible comparisons, and accumulated public
results could become one.

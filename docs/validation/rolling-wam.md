# Rolling-WAM candidate — not registered

Status: **unpublished candidate**. Rolling-WAM is not a selectable model.
`./lab --list` and the compatibility registry do not offer it. The paper's
success rates and latencies below are publisher-reported figures, not lab
measurements, and this note is not a paper reproduction.

Primary references, inspected 2026-09-30:

- [project page](https://rolling-wam.github.io/)
- [paper, arXiv:2609.30247v1](https://arxiv.org/abs/2609.30247) (submitted 2026-09-24, CC BY 4.0)
- [upstream repository](https://github.com/zyinghua/Rolling-WAM), commit `9dbe8abcf36cbd11db2fc8d392a27c4a8a2a9c9a` (2026-09-25, Apache-2.0)

That commit contains the README, license, and image assets. The README says
code and checkpoints are still being prepared. No sampler, checkpoint, or
statistics file is available to pin.

The machine-readable record is `showcase/rolling_wam_candidate.py`.
Registration stays blocked while `ready_to_register()` is false.

## What the paper specifies

Rolling-WAM keeps a sliding window of video-action chunks at staggered noise
levels. Each cycle fully denoises the next chunk, executes it, and carries the
partially denoised future into the next cycle with a newly noised chunk. The
default schedule in Section IV-B is:

| Quantity | Value |
|---|---:|
| Denoising steps per chunk, `N` | 10 |
| Window chunks, `W` | 5 |
| Actions executed per cycle, `K` | 16 |
| Steady-state denoising steps, `N/W` | 2 |
| Window horizon, `W×K` | 80 actions |
| Classifier-free guidance | 1 |
| Noise shift `ρ` | 5 |

The first cycle is an initialization pass of `N` steps. Later cycles are the
steady-state pass of `N/W` steps. The window is model state. A stateless
request that denoises one action chunk from noise, which is how the current
Fast-WAM HTTP client behaves, does not implement this schedule.

The video expert starts from Wan2.2-TI2V-5B. The action expert has 30 layers
and hidden size 1024, initialized by interpolating the video expert. Action
tokens attend to visual tokens across the window. Action-to-action attention
stays inside one chunk. Video tokens do not attend to actions. Language and
proprioception enter by cross-attention.

## Simulator fit

| Setting | Stated observation | Stated evaluation | Open contract |
|---|---|---|---|
| LIBERO | external and wrist RGB | one policy, four suites, 50 rollouts per task; publisher average 98.1% | image size, state ordering, action dimension, normalization |
| RoboTwin 2.0 | head and two wrist RGB | one policy, 50 tasks, Clean and Randomized, 100 rollouts per task per setting; publisher average 93.3% | action dimension and normalization. Latency measurement uses 384×320 |
| RoboCasa | not evaluated | — | out of scope |
| Unitree G1 | one 320×224 egocentric RGB, 43D state, 78D action | 20 trials per real-world task; publisher average 85.0% | outside this lab's simulators |

The 384×320 RoboTwin latency setting is not a license to reuse the lab's
Fast-WAM RoboTwin 14D qpos profile. The paper says its Fast-WAM and Joint-WAM
baselines use matched training and evaluation settings where applicable. It
does not state Rolling-WAM's LIBERO or RoboTwin action width. Those dimensions
stay unknown until a checkpoint config states them.

On one A100, the paper reports steady-state replanning of 215 ms for
Rolling-WAM, 548 ms for its Fast-WAM baseline, and 978 ms for its Joint-WAM
baseline, at 384×320 with 16 executed actions. That is the publisher's
latency comparison. It is not a measurement on this lab's hardware, and it
does not transfer to the lab's Fast-WAM action-only profile.

## Generated futures

The paper shows imagined video beside later observations. In this lab, that
media would be a generated future: action-aligned, revealed after the chunk
executes, and excluded from action selection. The rolling window itself is
denoising state carried between cycles. It is not a set of scored alternative
plans.

## Admission checklist

Register a profile only after all of the following are true:

1. Inference code is published and pinned by commit.
2. A checkpoint and its statistics have identities and digests.
3. LIBERO and RoboTwin each have an explicit camera list, image transform,
   state ordering, action dimension, normalization, horizon, and simulator
   revision. Do not copy the 7D or 14D Fast-WAM contracts by assumption.
4. The runtime is an isolated sibling checkout with a stateful policy boundary
   that retains the rolling window across replans. One heavyweight model stays
   resident.
5. A CPU-safe contract test covers the wire format, and one audited local
   request plus one short rollout are recorded before any support claim.

Until then, `rollingwam`, `rolling-wam`, and `rolling_wam` remain unknown
model names.

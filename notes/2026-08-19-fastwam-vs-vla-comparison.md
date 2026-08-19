# Fast-WAM compared with pi0.5 and GR00T N1.5 — 2026-08-19

## Working conclusion

Fast-WAM is currently the most interesting research direction, pi0.5 is the
strongest practical baseline in this lab, and GR00T N1.5 provides the broadest
open robotics and embodiment-adaptation ecosystem. This is not yet a local
model-quality ranking: the three integrations have not completed the same
tasks, seeds, environments, and evaluation budgets.

The WAM/VLA boundary is also less distinct than the names imply. Fast-WAM does
not generate a future video during its normal action-inference path. Its world
modeling advantage comes from video co-training, after which its video DiT is
used as a single-pass world encoder for direct action prediction. Conversely,
GR00T N1.5 is called a VLA but adds Future LAtent Representation Alignment
(FLARE), a world-model-like future representation objective without generative
future-frame rendering.

## Architectural comparison

| Model | Main emphasis | Runtime action path | World/temporal supervision | Best current role in this lab |
|---|---|---|---|---|
| Fast-WAM | Dynamics-aware representations and data efficiency | Current cameras, state, and language to a 32x7 action chunk | Video prediction during training; no required future-video generation at test time | Test the value of video co-training and future counterfactual extensions |
| pi0.5 | Semantic generalization and hierarchical language-conditioned control | VLM high-level subtask inference plus a flow-matching action expert | Heterogeneous co-training across robots, web data, language, and semantic subtasks | Stable control baseline and the most mature local LIBERO path |
| GR00T N1.5 | Generalist embodiment adaptation and an open NVIDIA robotics stack | Eagle VLM embeddings cross-attended by an action/state DiT | FLARE future-latent alignment plus diverse robot, simulation, and synthetic data | RoboCasa, humanoid, post-training, and embodiment experiments |

Fast-WAM is therefore not an online visual planner in the currently released
LIBERO evaluation path. Operationally it behaves like an action policy. The
important experimental distinction is its training objective and world encoder,
not a visible imagined rollout before every action.

## How Fast-WAM produces actions

The released Fast-WAM checkpoint used here does **not** generate several future
videos and choose the action associated with the best one. Its normal inference
path generates one continuous action chunk directly from the current cameras,
language instruction, and robot state.

During training, the model receives the clean current frame, future
ground-truth frames, language, state, and demonstrated actions. The future
video latents and action tokens have separate flow-matching losses:

```text
current frame -------------------+
future ground-truth frames ------+-- video flow-matching loss
language + robot state ----------+
demonstrated action chunk -------+-- action flow-matching loss
```

The video objective forces the Wan2.2 video backbone to learn motion, contact,
object persistence, and interaction structure. A structured attention mask is
important to interpreting the result: action tokens can attend to the clean
first-frame tokens, but cannot attend to future-video tokens. The first-frame
tokens also cannot absorb future tokens. Future information therefore cannot
leak into the action branch during training.

The base checkpoint's inference path is:

```text
external + wrist cameras
          |
          v
center crop, resize, concatenate to 224x448
          |
          v
VAE encoding of the current frame only
          |
          v
one video-DiT pass and cached per-layer visual K/V features
          |
          +---- T5 instruction embedding + normalized 8D state
          |
          v
32x7 Gaussian action noise
          |
          v  ten action flow-matching steps
one normalized 32x7 action chunk
          |
          v
dataset-statistics denormalization and gripper conversion
          |
          v
execute the first 10 actions, observe again, and replan
```

No future-video tokens are instantiated or denoised in `infer_action`. The
fixed model seed of 42 makes identical inputs produce identical action samples
in this experiment. Sampling one action trajectory from a learned conditional
distribution is not action selection through planning.

Specifically, Fast-WAM currently has no:

- candidate set of action sequences;
- predicted future rollout for each candidate;
- reward, success, collision, or safety critic;
- candidate ranking, tree search, or trajectory optimization.

A true world-model model-predictive-control loop would instead propose multiple
action sequences, predict the future resulting from each, score those futures,
choose the highest-scoring sequence, execute a short prefix, and then replan
from a new observation. This is a possible future experiment, not a behavior of
the released base Fast-WAM policy.

### Output contracts versus the local VLA profiles

At their outer interfaces all three policies return continuous action chunks;
the WAM/VLA distinction is not a special action datatype:

| Local profile | Output | Generator | Representation conditioning the action generator |
|---|---:|---|---|
| Fast-WAM LIBERO | `32x7` | Ten-step action flow matching | Wan2.2 video DiT shaped by future-video co-training |
| pi0.5 LIBERO | `10x7` | Ten-step action flow matching | PaliGemma-derived VLM shaped by heterogeneous semantic and robot co-training |
| GR00T N1.5 RoboCasa | `16x12` | Action DiT / flow matching | Eagle VLM with FLARE future-latent alignment |

Each Fast-WAM LIBERO step contains end-effector translation deltas, axis-angle
rotation deltas, and one gripper value:

```text
delta-x, delta-y, delta-z,
delta-rotation-x, delta-rotation-y, delta-rotation-z,
gripper
```

The gripper output is converted from the training convention to LIBERO's
convention, inverted, and sign-binarized after action denormalization.

### What the paper is trying to establish

The paper deliberately separates four cases:

- **Fast-WAM:** future-video co-training, but current-frame-to-action inference;
- **Fast-WAM-Joint:** future video and actions are denoised together;
- **Fast-WAM-IDM:** a future video is generated first and actions are then
  conditioned on its representation;
- **without video co-training:** the same direct action interface without the
  future-video training objective.

Its reported LIBERO averages are 97.6% for Fast-WAM, 98.5% for Joint, 98.0%
for IDM, and 93.5% without video co-training. The larger degradation from
removing video co-training supports the paper's thesis: future prediction is a
valuable representation-learning objective, while explicitly rendering a
future at every control step provides much less benefit relative to its cost.

This is evidence for temporal/video supervision as a useful training signal. It
does not by itself prove that Fast-WAM performs explicit causal planning or has
learned a general-purpose simulator. The repository's optional `infer_joint`
path can expose predicted frames, but the base model's structured mask prevents
those generated future tokens from selecting or ranking the base action. The
Joint and IDM variants are separate experimental architectures.

Relevant implementation paths:

- [`../showcase/fastwam_policy.py`](../showcase/fastwam_policy.py): exact local
  preprocessing, current-frame encoding, action generation, denormalization,
  and staging behavior;
- [`../../../upstream-fastwam/src/fastwam/models/wan22/fastwam.py`](../../../upstream-fastwam/src/fastwam/models/wan22/fastwam.py): pinned upstream
  training mask plus `infer_action` and `infer_joint` implementations.

## Evidence measured on this workstation

| Measurement | Fast-WAM / LIBERO | pi0.5 / LIBERO | GR00T N1.5 / RoboCasa |
|---|---:|---:|---:|
| Warm request latency | 1.86 s mean | approximately 132 ms median | 227 ms mean / 165 ms median |
| Peak observed GPU memory | 14,288 MiB | 19,098 MiB | 22,092 MiB |
| Action response | 32x7 | 10-action LIBERO profile | 16x12 |
| Behavioral evidence | 1/1 on Spatial task 2 | 10/10 on one-seed Spatial suite | 0/1 on a different RoboCasa task |

These numbers require careful interpretation:

- Fast-WAM's 24 GB profile repeatedly stages components between CPU and GPU.
  Its warm 1.86-second latency is dominated by that memory-saving adaptation;
  the paper reports 190 ms with a resident model on a 32 GB RTX 5090D.
- The pi0.5 `10/10` result and Fast-WAM `1/1` result measure different amounts
  of validation. They do not establish that one model has higher success.
- The GR00T attempt used RoboCasa, a different observation/action contract and
  a more difficult task family. Its single failure cannot be compared directly
  with either LIBERO result.
- The current shared LIBERO client reports MuJoCo 3.2.3, while Fast-WAM asks for
  MuJoCo 3.3.2 to match its training data. This remains a comparability caveat.

Detailed local evidence is recorded in:

- [`2026-08-19-fastwam-libero-validation.md`](2026-08-19-fastwam-libero-validation.md)
- [`../results/README.md`](../results/README.md)
- [`../docs/groot-n1.5.md`](../docs/groot-n1.5.md)

## Published results and interpretation

The Fast-WAM paper reports a 97.6% average success rate over the four LIBERO
suites, compared with 96.9% for its pi0.5 baseline. It also reports a much
larger drop when video co-training is removed than when explicit test-time
future imagination is removed. This supports the hypothesis that temporal
supervision may be most valuable while learning representations, rather than
as mandatory video generation during execution. The same paper reports that
pretrained pi0.5 remained the strongest method in its real-world towel-folding
test, so the result is not a universal Fast-WAM advantage.

The RoboCasa365 leaderboard currently reports GR00T N1.5 above pi0.5 on its
50-task multi-task benchmark. That is relevant evidence for RoboCasa, but the
leaderboard explicitly warns against treating configurations with different
architectures and training setups as controlled comparisons.

Primary references:

- [Fast-WAM paper](https://arxiv.org/abs/2603.16666)
- [Fast-WAM project](https://yuantianyuan01.github.io/FastWAM/)
- [pi0.5 technical report](https://www.pi.website/download/pi05.pdf)
- [GR00T N1.5 research overview](https://research.nvidia.com/labs/gear/gr00t-n1_5/)
- [RoboCasa365 leaderboard](https://robocasa.ai/leaderboard.html)

## Practical recommendation

1. Keep pi0.5 as the control baseline because it is fast, stable, and already
   validated across the complete local LIBERO-Spatial suite.
2. Use Fast-WAM to test data efficiency, temporal representation learning, and
   whether optional predictions improve counterfactual action analysis.
3. Use GR00T N1.5 for RoboCasa and embodiment/post-training experiments where
   its open NVIDIA tooling is more valuable than a lightweight runtime.

The likely long-term architecture is a hybrid: a VLM for language and visual
grounding, world-model objectives during training, a fast action head for
routine closed-loop control, and optional future simulation only when task
complexity or uncertainty justifies the added latency. Fast-WAM and GR00T N1.5
approach this design from different directions.

## Next fair validation gate

Run Fast-WAM and pi0.5 with the same LIBERO environment, suites, task IDs,
initial-state seeds, prompt text, action budgets, and scoring predicates:

1. Repeat all ten `libero_spatial` tasks at seed 7 to match the existing pi0.5
   functional run.
2. Extend to `libero_object`, `libero_goal`, and `libero_10` with at least three
   seeds per task.
3. Compare task success, policy steps to success, end-to-end episode time,
   cold/warm request latency, peak VRAM, and sensitivity to paraphrased language.
4. Report the MuJoCo version and any model-specific memory staging separately.

Only after this matched evaluation should the lab claim a behavioral comparison
between Fast-WAM and pi0.5. GR00T should be compared with pi0.5 separately on
identical RoboCasa tasks, scenes, seeds, horizons, and checkpoint-training
conditions.

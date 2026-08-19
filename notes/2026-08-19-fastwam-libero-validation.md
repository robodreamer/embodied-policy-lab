# Fast-WAM closed-loop LIBERO validation — 2026-08-19

## Objective

Validate the released Fast-WAM LIBERO policy in Embodied Policy Lab without
changing the existing π0.5 or RoboCasa runtimes. This branch is deliberately an
isolated worktree experiment: `experiment/fastwam-libero-validation`.

Primary references:

- [Fast-WAM project page](https://yuantianyuan01.github.io/FastWAM/)
- [paper, arXiv:2603.16666](https://arxiv.org/abs/2603.16666)
- [official source](https://github.com/yuantianyuan01/FastWAM)
- [released checkpoints](https://huggingface.co/yuanty/fastwam)

Pinned source revision:
`45d8e1458921d83f8ad6cf9ce993d371208dabd0`.

## Why this belongs in Embodied Policy Lab

Fast-WAM's released LIBERO checkpoint directly predicts robot actions from the
current observation and language command. That makes closed-loop task success,
replan behavior, latency, and action traces the meaningful validation—not a
standalone image-to-video demo. The separate `world-model-experiment` repo
proved that one action-only inference can fit this workstation using staging;
this branch tests whether the same policy can control the existing LIBERO
simulator repeatedly.

The release also contains joint/video modeling code, but the published LIBERO
evaluation defaults to `infer_action` with future visualization disabled. A
future-video comparison is therefore a separate follow-up gate and must not be
presented as part of this action-only validation.

## Frozen behavior contract

- Released checkpoint: `libero_uncond_2cam224.pt`
- Released dataset statistics: `libero_uncond_2cam224_dataset_stats.json`
- Two 224×224 RGB observations: external then wrist, concatenated horizontally
  into 224×448
- State: 8D end-effector position, axis-angle orientation, and gripper qpos
- Action: 32×7, denormalized with released stats
- Gripper: converted to LIBERO convention, inverted, then sign-binarized
- BF16, 10 inference steps, model RNG seed 42, CPU noise RNG
- Execute 10 actions and then replan from new observations
- 30 initial no-op steps
- Maximum policy steps: 400 for spatial/object/goal; 700 for LIBERO-10/90

## Runtime architecture

The LIBERO client and Fast-WAM policy remain in different processes and
environments:

```text
LIBERO / MuJoCo client
  current external + wrist + 8D state + instruction
                    |
                    | lossless localhost HTTP
                    v
Fast-WAM Python 3.10 service
  T5 prompt cache -> current-frame VAE encode -> MoT action diffusion
                    |
                    v
             finite 32×7 action chunk
```

The 24 GB service stages model components:

1. Load the full checkpoint on CPU.
2. On a prompt-cache miss, keep MoT on CPU, move T5 to CUDA, cache the context
   on CPU, then offload T5.
3. Keep MoT/proprio resident on CUDA during repeated replans.
4. For **every** request, move VAE to CUDA and encode that request's combined
   current camera image, then offload VAE.
5. Run the unchanged 10-step action diffusion using the current latent and
   cached prompt context.

Caching or replaying an image latent across replans would invalidate the
closed-loop experiment and is explicitly not done.

## How to run

```bash
cd /home/andypark/Projects/playground/git-worktrees/embodied-policy-lab-fastwam
./scripts/setup_fastwam_libero.sh --check

# Interactive dashboard; task 2 matches the bounded fixture used earlier.
./scripts/run_interactive_showcase.sh \
  --backend libero --model fastwam --task-id 2

# One automatic scored rollout.
./scripts/run_showcase.sh \
  --backend libero --model fastwam \
  --batch --task-suite libero_spatial --task-id 2 --trials 1

# Wiring smoke only: execute two 10-action replans, not a behavioral score.
./scripts/run_showcase.sh \
  --backend libero --model fastwam \
  --batch --task-suite libero_spatial --task-id 2 --trials 1 \
  --max-policy-steps 20
```

If the worktree cannot find the already configured simulator, set
`LIBERO_OPENPI_DIR` and `LIBERO_CLIENT_PYTHON` to the primary checkout's
`upstream-openpi` and LIBERO virtualenv. `FASTWAM_DIR` overrides discovery of
the sibling Fast-WAM checkout.

## Review artifacts and acceptance gates

Every replan writes `policy-inference/request-NNNNN.json` and its exact
`request-NNNNN-input.png` under the session directory. Review these alongside
`state.json`, `inference-audit.jsonl`, `gpu.csv`, the rollout MP4, and the final
report.

Pass the experiment in stages:

1. **Contract:** current input is 224×448, state is finite 8D, output is finite
   32×7, and gripper values are sign-binarized.
2. **Dynamic observation:** two requests using different camera images must
   have different input hashes; both must complete without OOM. Prompt cache
   should be false then true for the same instruction.
3. **Closed loop:** one bounded `libero_spatial` rollout reaches the simulator,
   replans every 10 actions, and saves all artifacts without remote inference.
4. **Behavior:** run the official suite budgets and multiple initial states.
   Report task success rather than single-frame action similarity.
5. **Only after action validation:** assess `infer_joint` future frames with
   temporally aligned ground truth, PSNR/LPIPS-style metrics, and qualitative
   prediction/actual clips. This may require more memory than the action-only
   profile and is not yet claimed to fit.

Known limitation: the current simulator environment was originally built for
the π0.5 LIBERO stack. Fast-WAM's source asks for MuJoCo 3.3.2 to match its
training data. Record the actual MuJoCo version in results and treat a mismatch
as a comparability caveat until a dedicated 3.3.2 simulator environment is
validated.

## Local result on 2026-08-19

The first implementation passed its bounded gates on the local RTX PRO 5000
Blackwell Generation Laptop GPU (24,463 MiB reported VRAM):

- checkpoint CPU load: 88.9–105.2 seconds across two clean server starts;
- first request with a T5 prompt-cache miss: 15.52 seconds total;
- repeated prompt with a distinct shifted camera input: 1.78 seconds total;
- two live LIBERO replans after the prompt was cached: 1.60 and 1.53 seconds;
- CUDA peak allocated: 13,613,923,840 bytes; launcher telemetry peak: 14,288 MiB;
- all responses: finite 32×7 actions with binarized gripper values;
- four successive requests had four distinct combined-image hashes;
- the same live observation, state, prompt, and model seed produced bit-exact
  normalized actions across independent server starts;
- one standard-launcher smoke produced the dashboard state, one exact input
  artifact, action trace, GPU CSV, rollout MP4, summary, and report, then freed
  the GPU;
- shell picker/setup checks and 17 Python tests passed.

The short simulator trials were capped at 10 or 20 policy steps and ended
`0/1`. That was expected and is **not** a behavior score. It proved transport,
dynamic observation encoding, action execution, telemetry, and artifact
generation.

The subsequent full-budget `libero_spatial` task-2 run passed **1/1**. The
simulator's real success predicate fired at policy step 87 of 400 after nine
replans. All nine current combined-camera hashes were unique. The first request
took 16.54 seconds; the eight warm requests averaged 1.86 seconds end to end
(median 1.82 seconds, P95 2.06 seconds). Launcher telemetry peaked at 14,288 MiB
and 41% GPU utilization. The syscall audit observed only
`127.0.0.1:8000` and no remote destination. Nine inference-audit events, nine
exact input/action artifacts, and a 48 KiB successful rollout MP4 were saved.

This is a real behavioral success, but one initial state is not a benchmark.
Multiple initial states and broader suites remain the next behavior gate.

The validated simulator client reports MuJoCo 3.2.3, so the 3.3.2 training-data
compatibility caveat remains open.

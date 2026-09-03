# Flex-π LIBERO world-action validation — 2026-08-24

## Objective and public validation boundary

Add the released Flex-π LIBERO checkpoint to Embodied Policy Lab without
disturbing the validated Fast-WAM experiment. Both appear as model plugins in
the same dashboard and launcher, while a single-session GPU guard prevents
simultaneous heavyweight servers. This page records the released-asset
contract and bounded checks that support the public integration claims.

Primary references:

- [project page](https://flex-pi.github.io/)
- [paper, arXiv:2608.10860](https://arxiv.org/abs/2608.10860)
- [official source](https://github.com/geyan21/flex-pi)
- [released LIBERO checkpoint](https://huggingface.co/flex-pi/flexpi-libero)

Pinned source revision:
`20c1b2b71ea35a415d5d47c39b04443cfadad7a1`.

## What is being tested

Flex-π is a multi-stream world-action model. Its released stream-dropout
checkpoint can use several inference regimes without retraining. This
integration exposes two endpoints of that spectrum:

| Mode | Generated streams | Purpose |
|---|---|---|
| `action-only` | 32×7 action chunk | Explicit lower-memory closed-loop policy baseline |
| World-action co-generation (`full-joint` internally) | RGB + DINOv3 + pointmap + 32×7 actions | Default world-action behavior and visual-dynamics validation |

The co-generation path jointly denoises future stream tokens. Here, “joint” is
an upstream inference term for joint multimodal generation; the actions remain
7D end-effector deltas plus gripper control, not robot joint-space commands.
It does not generate
many action candidates, render each candidate, and choose the best one. The
future RGB and action chunk are correlated outputs of one model pass, so the
UI calls this a policy-owned world-action prediction rather than an independent
world-model comparison.

## Frozen released contract

- Checkpoint: `flex-pi/flexpi-libero`, `step_010860.pt`
- Input: two 256×256 simulator RGB views plus aligned metric depth
- Composite: wrist on top at 288×512, exterior bottom-left at 160×256, black
  synthetic camera bottom-right; final 448×512
- State: 8D end-effector position, axis-angle orientation, and gripper qpos
- Internal universal action/state layout: 32 channels
- LIBERO action channels: `[0, 1, 2, 3, 4, 5, 18]`
- LIBERO state channels: `[0, 1, 2, 3, 4, 5, 18, 19]`
- Output: denormalized finite 32×7 actions; gripper converted, inverted, and
  sign-binarized for LIBERO
- Four Euler denoising steps, model seed 42, CPU random-noise device
- Execute ten actions, then replan from a new observation
- Future-frame interval: four simulator actions
- Exact upstream MuJoCo version: 3.3.2

Depth is not guessed from RGB. The client enables robosuite depth buffers,
converts normalized z-buffer values to metric metres with the live MuJoCo
model, stores clipped uint16 millimetres, rotates depth exactly like RGB, and
uses the published per-camera intrinsics. Only the four 399-byte intrinsics
files are needed from the 12 GB training dataset.

## Runtime and memory boundary

The upstream environment requires Python 3.10, PyTorch 2.7.1 + CUDA 12.8, and
MuJoCo 3.3.2. It cannot honestly share the older OpenPI LIBERO client. Both the
simulator client and model server therefore use the isolated sibling
`upstream-flexpi/.venv`; requests cross only a loopback HTTP boundary.

The setup downloads roughly 25 GB of inference assets: the 12 GB Flex-π
checkpoint, 1.4 GB Wan VAE, approximately 11 GB frozen T5 encoder, tokenizer,
and DINOv3. The demonstration dataset is not downloaded. At model startup the
checkpoint is deserialized on CPU; T5 remains on CPU when the model moves to
CUDA, and prompt contexts are cached. Action-only is the first hardware gate.
Full-joint is attempted only after measuring that baseline because upstream
reports a 16–26 GB inference range and this workstation has 24,463 MiB.

## Dashboard validation protocol

Mode changes are accepted only while the simulator is waiting between
rollouts. For full-joint inference:

1. Save the generated composite at action indices 0, 4, and 8 for the default
   ten-action replan prefix.
2. Execute the returned actions normally; prediction never gates execution.
3. Capture actual composites at the same indices.
4. After each prefix completes, buffer its matched samples privately without
   changing the live execution layout.
5. Once the entire rollout ends, concatenate all completed prefixes into one
   non-looping replay timeline and reveal it below the external and wrist live
   views.
6. Present an aspect-correct 256×160 external comparison above the 512×288
   wrist comparison; retain the exact 448×512 model composites and a JSON map
   of frame/action offsets for audit.
7. Report external, wrist, and raw-composite mean pixel PSNR as reproducible
   alignment smoke metrics.

PSNR is intentionally not treated as a robot-success metric. The main policy
measure remains closed-loop LIBERO task success. The visual comparison checks
temporal alignment and obvious dynamics failures; future work can add LPIPS,
DINO feature similarity, and pointmap geometry metrics.

## Rollout budgets and fair comparison

The paper/model documentation reports the conventional LIBERO budgets
220/280/300/520 for spatial/object/goal/LIBERO-10. The current upstream source
uses 400/400/400/700. This lab keeps its standard cross-policy defaults
220/280/300/520 for Flex-π and records the budget in every audit event. Any
published comparison must use the same task IDs, initial states, simulator
version, executed-action count, and budget for every policy. Fast-WAM's existing
reproduction remains on its explicitly documented upstream 400/400/400/700
contract.

## How to run and review

```bash
cd /path/to/embodied-policy-lab

# Full setup/download, then a non-mutating diagnostic:
./scripts/setup_flexpi_libero.sh
./scripts/setup_flexpi_libero.sh --check

# Default world-action co-generation:
./scripts/run_interactive_showcase.sh \
  --backend libero --model flexpi --task-suite libero_spatial --task-id 0 \
  --max-policy-steps 20

# Explicit lower-memory action-only baseline:
./scripts/run_interactive_showcase.sh \
  --backend libero --model flexpi --flexpi-mode action-only \
  --task-suite libero_spatial --task-id 0 --max-policy-steps 20

# Terminal picker: select WAM, Flex-π, then joint or action-only.
./scripts/run_interactive_showcase.sh

# Non-interactive selection through the same friendly CLI.
./lab --backend libero --model flexpi --flexpi-mode action-only --default
```

Review `showcase-runs/latest/state.json`, `inference-audit.jsonl`, `gpu.csv`,
`server.log`, the rollout under `videos/`, per-request metadata under
`policy-inference/`, the wrist-view full-rollout comparison at
`previews/latest_policy_{prediction,actual}.mp4`, the external-view comparison
at `previews/latest_policy_{prediction,actual}_external.mp4`, the exact model layouts at
`previews/latest_policy_{prediction,actual}_raw_composite.mp4`, and prefix
offsets in `previews/latest_policy_timeline.json`.

The matched Fast-WAM/Flex-π headless evaluation is documented separately in
[the benchmark protocol](../benchmarks/fastwam-flexpi-libero.md).

## Current local status

Validated locally on an NVIDIA RTX PRO 5000 Blackwell Laptop GPU with 24,463
MiB VRAM. The isolated Python 3.10 environment, release checkpoint, VAE,
UMT5/tokenizer, DINOv3 weights, and four published intrinsics files are present;
`./scripts/setup_flexpi_libero.sh --check` passes.

| Gate | Result | Model timing | CUDA peak |
|---|---|---:|---:|
| Action-only, one query + one executed action | Pass; finite 32×7 output | 1.291 s inference; 4.240 s total with 2.884 s cold prompt encoding | 13.13 GiB allocated / 13.24 GiB reserved |
| Full-joint, one query + one executed action | Pass; finite 32×7 output + 9 RGB future frames | 2.980 s inference; 5.796 s total with 2.783 s cold prompt encoding | 15.33 GiB allocated / 16.45 GiB reserved |
| Delayed 10-action visual comparison | Pass; reveal only after execution | 2.866 s inference; 5.727 s total | 3 aligned frames, mean RGB PSNR 31.164 dB |

Cached model startup measured 95.5–102.8 seconds. The dashboard now reports
the active initialization phase and this expected range while it waits; the
resident server is reused across rollouts. One-second `nvidia-smi`
telemetry observed 14,671 MiB process-level memory for action-only and 17,961
MiB for full-joint; those totals include CUDA context/driver allocations that
PyTorch's allocator counters do not. Full-joint therefore fits this machine
with useful headroom and does not require a reduced stream regime.

The bounded action-only and full-joint episodes were deliberately capped at
one policy action, so their recorded task failures are not policy-quality
results. The ten-action run exists to validate temporal capture/reveal, not
LIBERO success. A fair quality claim still requires complete task-suite
rollouts under the frozen budgets above.

Validated sessions (kept outside version control):

- `showcase-runs/20260824-113450`: action-only gate
- `showcase-runs/20260824-113722`: full-joint hardware-fit gate
- `showcase-runs/20260824-114021`: delayed predicted-vs-actual comparison

### Final release-path gate

After the stacked PR review, the pinned clean checkout was loaded again through
the hardened `weights_only=True` server path. Startup completed in 87.6 seconds.
A direct action-only request returned a finite 32×7 action chunk in 3.748
seconds (including the first prompt encode). A subsequent full-joint request
returned a finite 32×7 action chunk in 2.403 seconds plus the exact 448×512
input composite and nine losslessly transported 448×512 generated frames at a
four-action interval. The interactive client now requests only the prefix it
can compare after its configured rollout, avoiding serialization of unused
future frames.

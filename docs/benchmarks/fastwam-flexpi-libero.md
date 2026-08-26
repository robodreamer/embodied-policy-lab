# Fast-WAM vs Flex-π matched headless LIBERO benchmark — 2026-08-24

## Purpose

Validate the subset of published Fast-WAM and Flex-π claims that the released
LIBERO checkpoints and this workstation can actually test. The comparison has
three configurations:

1. Fast-WAM's released action-only inference path;
2. Flex-π action-only stream dropout;
3. Flex-π full-joint world-action co-generation.

Primary references:

- [Fast-WAM project page](https://yuantianyuan01.github.io/FastWAM/)
- [Fast-WAM paper](https://arxiv.org/abs/2603.16666)
- [Flex-π project page](https://flex-pi.github.io/)
- [Flex-π paper](https://arxiv.org/abs/2608.10860)

## What the papers claim

The published standard LIBERO averages are 97.6% for Fast-WAM, 98.4% for
Flex-π action-only, and 98.5% for Flex-π full-joint. Their per-suite values
are embedded in `showcase/wam_benchmark.py` and printed beside local results.
Flex-π evaluates all 40 LIBERO tasks with 50 rollouts per task, suite budgets
of 220/280/300/520 actions, a 32-action prediction horizon, 10 executed actions
per replan, and four Euler denoising steps.

The latency claims are not directly portable to this laptop. Fast-WAM reports
190 ms on an RTX 5090D; Flex-π reports roughly 60 ms action-only and 193 ms
joint on an RTX 5090 and reports a separately compiled 90 ms Fast-WAM baseline.
Those numbers use different 5090-class hardware and optimized serving stacks.
Our runner measures the released local integrations and treats direction and
relative deployment cost as evidence, not equality with the published values.

## Matched local protocol

The benchmark runner uses the common Flex-π Python/LIBERO/MuJoCo 3.3.2 client
runtime for all three configurations. Fast-WAM retains its pinned model server;
only its simulator client moves to the common runtime. Every configuration uses
the same task IDs, initialization seed, number of trials, suite action budget,
and 10-action execution prefix. Each checkpoint retains its required native
camera size, depth requirement, state normalization, and prompt preprocessing.
It also retains the released evaluator's initial no-op settling period
(30 steps for Fast-WAM and 10 for Flex-π); this is a documented protocol
difference, not a perfectly identical observation stream.

Before the first rollout, the runner repeats one unchanged observation for a
configurable number of warmups and timed calls. It records end-to-end local
round-trip time, server total time, the server's denoising/model-core time, and
CUDA allocation counters. Closed-loop episodes record simulator success and
the report includes 95% Wilson confidence intervals. Benchmark mode disables
the browser, network tracing, per-step JPEGs, real-time sleeps, and rollout MP4s
so measurement overhead is bounded.

## Profiles

```bash
# Three 20-action wiring checks (one per configuration)
./scripts/benchmark_wam_libero.py --profile smoke

# 4 suites × 2 tasks × 3 trials × 3 configurations
./scripts/benchmark_wam_libero.py --profile pilot

# 4 suites × 10 tasks × 50 trials × 3 configurations = 6,000 episodes
./scripts/benchmark_wam_libero.py --profile paper

# Inspect the exact 12-session paper plan without starting a model
./scripts/benchmark_wam_libero.py --profile paper --plan-only
```

Use `--configs fastwam,flexpi-action,flexpi-joint` to select configurations.
Use `--output-dir PATH --resume` to continue an interrupted experiment. Every
run writes `benchmark-manifest.json`, `benchmark-results.json`, and
`benchmark-report.md`, plus the normal state, inference audit, policy artifacts,
and GPU telemetry inside each session directory.

## Claim matrix

| Claim | Can this runner test it? | Evidence |
|---|---|---|
| Flex-π action-only/full-joint standard LIBERO success vs Fast-WAM | Yes, as a matched local comparison with the complete `paper` profile | Same 40 tasks, 50 trials, shared local budgets, seed policy, simulator runtime, and action prefix |
| Flex-π action-only is faster than Fast-WAM locally | Yes, directionally | Repeated warm calls through each released local server |
| Cost of Flex-π joint generation vs action-only | Yes | Same checkpoint/runtime with only inference mode changed |
| Published absolute RTX 5090 latency | No | Different GPU and optimization/compilation stack |
| Fast-WAM video-co-training and data-scaling ablations | No | Requires retraining checkpoints, not inference-only evaluation |
| Flex-π cross-modality training/data-efficiency ablations | No | Requires controlled retraining |
| RoboTwin, LIBERO-Plus, and real-robot results | No | They are outside this LIBERO simulator setup |

A smoke result is marked **wiring only**, and a pilot result is marked
**provisional**. A complete 12-session `paper` run with 500 episodes per
session is labelled a **complete matched local protocol**. It is not a direct
paper reproduction: this harness uses Flex-π's shared 220/280/300/520 suite
budgets, whereas Fast-WAM's native evaluator uses 400/400/400/700. A failure
to match a published rate is evidence about this local integration, not proof
that a paper's result is false; upstream commits, assets, hardware numerics,
and initial-state handling must be audited before interpreting a discrepancy.

## Bounded local validation

The `smoke` profile completed on this workstation on 2026-08-24. Each
configuration executed one 20-action `libero_spatial` task-2 episode using the
same MuJoCo 3.3.2 simulator source. All episodes ended before success, which is
expected at this deliberately short horizon and is not behavioral evidence.

| Configuration | Server total | Denoise core | Round trip | Peak GPU memory |
|---|---:|---:|---:|---:|
| Fast-WAM | 1,784.3 ms | 378.9 ms | 1,824.6 ms | 14,319 MiB |
| Flex-π action-only | 311.9 ms | 295.8 ms | 324.4 ms | 14,671 MiB |
| Flex-π full-joint | 2,325.1 ms | 2,304.2 ms | 2,355.6 ms | 17,961 MiB |

On this uncompiled local stack, Flex-π action-only was 5.72× faster than the
Fast-WAM server-total path. Flex-π full-joint cost 7.45× action-only. These
two ratios validate that the harness can measure the expected deployment
tradeoff, but only two timed calls were collected and the absolute values must
not be substituted for the papers' RTX 5090 measurements. The ignored local
report is at `benchmark-runs/wam-libero-smoke-20260824/benchmark-report.md`.

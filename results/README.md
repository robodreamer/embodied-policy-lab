# Validated local results

This directory contains concise, sanitized evidence from local Embodied Policy
Lab runs. Raw session logs, model inputs, videos, caches, and machine-specific
paths remain ignored.

Unless a row explicitly says otherwise, these are functional integration checks
or bounded local measurements—not statistically meaningful model rankings or
paper reproductions.

## Validation workstation

- NVIDIA RTX PRO 5000 Blackwell Laptop GPU
- 24,463 MiB reported VRAM
- Local model inference and simulation
- Exact model, simulator, checkpoint, and source revisions retained by each
  generated session

## Policy validation summary

| Profile | Environment | Coverage | Result | Warm latency | Peak observed GPU memory | Interpretation |
|---|---|---:|---:|---:|---:|---|
| π0.5 | LIBERO Spatial | 10 tasks × 1 seed | 10/10 | ~132 ms median | 19,098 MiB | complete one-seed functional suite, not benchmark accuracy |
| π0.5 | RoboCasa | 1 complete 450-action episode | 0/1 | 134.49 ms median | 21,045 MiB | closed-loop integration evidence |
| GR00T N1.5 | RoboCasa | 1 complete 900-action episode | 0/1 | 165.44 ms median | 22,092 MiB | closed-loop integration evidence |
| Fast-WAM | LIBERO Spatial task 2 | 1 complete initial state | 1/1 | 1.86 s mean | 14,288 MiB | real bounded success; insufficient for a rate |

Different environment rows are not directly comparable. RoboCasa uses a
mobile-manipulator 12D contract and different tasks and success predicates;
LIBERO uses a fixed-arm 7D contract.

## Flex-π hardware and prediction gates

| Mode | Output | Gate result | Model timing | CUDA peak reservation |
|---|---|---|---:|---:|
| Action-only | finite 32×7 actions | one query and one executed action passed | 1.291 s | 13.24 GiB |
| Full-joint | 32×7 actions plus RGB/DINO/pointmap futures | one query and one executed action passed | 2.980 s | 16.45 GiB |
| Full-joint delayed comparison | aligned actual/generated external and wrist frames | capture, delayed reveal, and replay passed | 2.866 s | 17,961 MiB process-level telemetry |

These short gates validate loading, transport, action execution, future capture,
and local hardware fit. They are not closed-loop task-success measurements.

## Matched WAM smoke

The headless smoke profile ran Fast-WAM, Flex-π action-only, and Flex-π
full-joint on the same deliberately short 20-action LIBERO task. All three
episodes ended before success as expected; the measurements validate the
benchmark wiring and deployment-cost instrumentation.

| Configuration | Server total | Denoise core | Round trip | Peak GPU memory |
|---|---:|---:|---:|---:|
| Fast-WAM | 1,784.3 ms | 378.9 ms | 1,824.6 ms | 14,319 MiB |
| Flex-π action-only | 311.9 ms | 295.8 ms | 324.4 ms | 14,671 MiB |
| Flex-π full-joint | 2,325.1 ms | 2,304.2 ms | 2,355.6 ms | 17,961 MiB |

Only two timed calls were collected. These values must not replace publisher
latency claims measured on different GPUs and optimized serving stacks.

## Reproduce and review

- [π0.5 RoboCasa validation](../docs/validation/robocasa-pi05.md)
- [GR00T N1.5 RoboCasa validation](../docs/validation/groot-n1.5-robocasa.md)
- [Fast-WAM LIBERO validation](../docs/validation/fastwam-libero.md)
- [Flex-π LIBERO validation](../docs/validation/flexpi-libero.md)
- [Matched Fast-WAM/Flex-π protocol](../docs/benchmarks/fastwam-flexpi-libero.md)

Generate a fresh local session with `./lab` or the commands in the linked
validation documents. Publish a new result only with its task IDs, seeds,
trial count, revisions, checkpoint identity, hardware, and interpretation
boundary.

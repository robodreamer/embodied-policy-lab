# Reproduction results

## 2026-08-05 — RTX PRO 5000 Blackwell Laptop GPU

- Source revision: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Model: official `pi05_libero` checkpoint
- Suite: `libero_spatial`
- Seed: `7`
- Trials: one per task, 10 total
- Result: **10/10 successful episodes (100%)**
- Evaluator time: **1 minute 13 seconds** after server readiness
- Peak observed GPU memory: **19,098 MiB**
- Peak observed GPU utilization: **99%**
- Peak observed GPU power: **93.35 W**
- Generated rollouts: 10 MP4 files, all marked `success`
- Client exit status: `0`

The first episode includes JAX compilation and took approximately 16 seconds;
later episodes generally completed in roughly 4–6 seconds. This is a functional
smoke test, not a statistically meaningful success-rate benchmark.

The pinned robosuite/MuJoCo environment emitted `EGL_NOT_INITIALIZED` while
destroying the rendering context after evaluation. This occurred after all
videos and results were written and did not change the successful exit status.

## Local artifacts

Run `scripts/run_smoke_test.sh` to generate a new local result. The runner retains:

- policy-server and simulator logs under `logs/`;
- one-second GPU telemetry under `logs/`;
- MP4 rollouts under `videos/<task-suite>/`.

These artifacts are intentionally ignored because logs contain machine-specific
paths and videos can grow large. Add a dated, sanitized summary here when
publishing additional results.

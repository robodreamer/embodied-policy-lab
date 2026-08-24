# Stacked PR maintainability review — 2026-08-24

This note records the focused pre-merge review of the Fast-WAM parent and
Flex-π child branches. Three independent review passes covered correctness,
release/security risk, dead code, duplication, hard-coded values, and stacked
PR readiness.

## Addressed before merge

- Removed stale DINO-WM and V-JEPA rows from the public model registry and
  added registry/launcher parity tests.
- Removed unused observation and preview plumbing, including the obsolete
  LIBERO `resize_size` option.
- Moved Flex-π modes, image layout, frame interval, and replay presentation
  rate into one dependency-light contract module shared by launcher, client,
  server, replay builder, and tests.
- Stopped recreating the Flex-π input image in the simulator process. The
  pinned policy runtime now returns its exact upstream-preprocessed composite,
  and comparison frames use lossless transport before PSNR is calculated.
- Full-joint prediction pixels are serialized only for the interactive replay;
  headless benchmarks retain metadata and actions without transporting unused
  multi-megabyte frame payloads.
- Pinned Flex-π source/model dependencies to immutable revisions and verify
  their SHA-256 digests. The policy process restores `weights_only=True` after
  the upstream evaluator changes PyTorch's loader behavior.
- Resume logic now retains completed retry directories and validates task IDs,
  seed, replan interval, suite budget, model, mode, and episode count before a
  session can be skipped.
- Renamed the full benchmark result to a **matched local protocol**. Shared
  local budgets make it useful for side-by-side testing, but not a direct
  reproduction of Fast-WAM's native paper evaluator.
- Replaced fixed workstation paths and linked-worktree assumptions with
  repository discovery or portable examples.

## Reasonable follow-up refactors

These are non-blocking because they span already-working VLA and WAM paths and
would increase the risk of this release cleanup if combined with it:

- Extract the duplicated Fast-WAM/Flex-π loopback HTTP request/response
  scaffolding into a small transport module while retaining policy-specific
  validation.
- Consolidate repeated runtime/worktree discovery in the shell launchers.
- Share the dashboard `LiveState`/control-inbox mechanics between the LIBERO
  and RoboCasa runners after their state schemas stabilize.
- Move published claim/reference values out of the benchmark implementation
  into a versioned data file with source citations.

The remaining numeric constants describe upstream checkpoint contracts or
explicit presentation choices; they are named and documented rather than
being scattered through control flow.

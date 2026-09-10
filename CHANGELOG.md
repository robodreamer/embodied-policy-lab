# Changelog

All notable changes to Embodied Policy Lab are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the series is `0.y.z`, the public CLI, dashboard, and model/simulator
contracts may still change. Git tags use a `v` prefix (`v0.1.0`).

## [Unreleased]

### Changed

- README version badge now tracks the latest GitHub Release.

## [0.1.0] - 2026-09-08

First tagged snapshot of the local studio: run a released VLA or world-action
model, watch the rollout, and keep the evidence.

### Added

- Interactive `./lab` launcher with a compatibility-aware picker, `--default`
  path, and `./lab --version`.
- Browser studio for live cameras, instruction, actions, latency, GPU
  telemetry, and per-attempt history.
- Compatible profiles for π0.5 (LIBERO, RoboCasa), GR00T N1.5 (RoboCasa),
  Fast-WAM (LIBERO, RoboTwin 2.0), and Flex-π (LIBERO, RoboTwin 2.0).
- Live instruction edits, prompt variants, and unscored exploratory commands
  for probing generalization beyond the example task wording.
- Action-aligned Flex-π prediction-versus-execution replay on LIBERO after
  the rollout finishes.
- Session artifacts under `showcase-runs/` (report, audit log, telemetry,
  videos, and optional aligned previews).
- Headless Fast-WAM / Flex-π LIBERO benchmark runner with `smoke`, `pilot`,
  and `paper` profiles. The `paper` profile is a matched local 6,000-episode
  schedule, not a paper reproduction.
- Isolated model runtimes, a GPU occupancy guard, and the optional RoboCasa
  simulator-oracle baseline.

[Unreleased]: https://github.com/robodreamer/embodied-policy-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/robodreamer/embodied-policy-lab/releases/tag/v0.1.0

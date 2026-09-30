# Changelog

All notable changes to Embodied Policy Lab are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the series is `0.y.z`, the public CLI, dashboard, and model/simulator
contracts may still change. Git tags use a `v` prefix (for example, `v0.7.0`).

The public contracts are the CLI flags, policy/simulator profiles, plugin
interfaces, dashboard controls, and saved evidence schema. During initial
`0.y.z` development, new capabilities or incompatible contract changes advance
the minor version; compatible fixes and documentation changes advance the patch
version. A stable public contract is required before `1.0.0`.

The history below was reconstructed from Git commits. Versions through `0.6.0`
identify development milestones; their dates and links identify the corresponding
source changes. Those milestones are not Git tags. The published tag `v0.1.0`
is `81053ae` (2026-09-10), a later snapshot than the retrospective `[0.1.0]`
milestone at `f0c574d`. The current release is `v0.7.0`.

## [Unreleased]

## [0.7.0] - 2026-09-30

Explicit CLI version reporting and reconciled version history.

### Added

- `./lab --version` and `./lab -V`, backed by the canonical `VERSION` file.
- A changelog for the public studio, and a version field on the existing citation metadata.

### Changed

- Reconstructed feature milestones using minor versions and documentation
  cleanup using a patch version, with links to the corresponding Git history.
- Synchronized source version, citation, README badge, and security policy.
- Rewrote the README as a direct product overview with quick-start instructions,
  supported profiles, and evidence boundaries.

## [0.6.0] - 2026-09-03

Retrospective milestone: RoboTwin studio and native batch evaluation.

### Added

- RoboTwin 2.0 batch evaluation foundation and a browser studio adapter for
  Fast-WAM and Flex-π, preserving the native bimanual 14D qpos contract.
- RoboTwin task selection, seed validation, and three-camera rollout capture.
- README studio captures and a gallery of the supported environments.

### Fixed

- RoboTwin startup, planner dependencies, rendering, and Flex-π asset pinning.
- Fast-WAM RoboTwin memory use to support the local 24 GB GPU path.

## [0.5.1] - 2026-08-26

Retrospective milestone: public documentation and evidence curation.

### Changed

- Public operator guidance, contribution and security guidance, license
  boundaries, citation metadata, and a roadmap.
- Curated validation protocols and sanitized evidence summaries for local
  model integrations.
- Refreshed studio media and clarified the lab's evaluation and observability
  scope, including the distinction between local validation and paper claims.
- Organized model contracts, benchmark protocols, and validation guides into
  the public documentation.

## [0.5.0] - 2026-08-24

Retrospective milestone: Flex-π world-action replay and matched WAM benchmarks.

### Added

- Flex-π LIBERO integration with action-only and full-joint modes; full-joint
  inference became the default Flex-π mode.
- Aligned prediction-versus-execution comparisons with dual-view replay,
  revealed after the rollout finishes.
- Headless Fast-WAM / Flex-π LIBERO benchmarks with `smoke`, `pilot`, and
  `paper` schedules and recorded provenance. These are local comparison
  protocols, not claims of paper reproduction.

### Fixed

- Model picker behavior, Flex-π mode selection, and dashboard ready controls.
- Fast-WAM source revision checks and Flex-π auxiliary asset validation.

### Removed

- Unvalidated learned world-model candidates from the supported integrations.

## [0.4.0] - 2026-08-19

Retrospective milestone: staged Fast-WAM integration on LIBERO.

### Added

- An experimental Fast-WAM LIBERO action inference path with startup progress
  and documented suite selection.
- Documentation of the Fast-WAM action contract and its comparison with VLA
  policies.

## [0.3.0] - 2026-08-17

Retrospective milestone: action previews and post-execution comparisons.

### Added

- A RoboCasa world-model preview plugin boundary and simulator-oracle action
  previews for comparison with executed actions.
- Prediction diagnostics and post-execution comparison artifacts. Learned
  world-model candidates remained experimental and were later removed from
  the supported integrations in `0.5.0`.

### Fixed

- RoboCasa camera framing, viewport sizing, rendering, and GPU startup.
- Preview failure handling and action-contract diagnostics.

## [0.2.0] - 2026-08-06

Retrospective milestone: multiple simulators and policy plugins.

### Added

- RoboCasa backend and interactive viewer alongside the LIBERO workflow.
- Model-agnostic policy plugins, compatibility registry, and GR00T N1.5
  RoboCasa support.
- Interactive `./lab` launcher and isolated policy setup and serving paths.

## [0.1.0] - 2026-08-05

Retrospective milestone: initial local π0.5 LIBERO lab.

### Added

- Reproducible local π0.5 LIBERO simulation and an interactive browser console.
- Local prompt generation, prompt delivery auditing, and randomized
  exploratory runs.
- Saved rollout evidence and documented local results.

### Fixed

- Prompt handoff during interactive runs and rollout budget handling.

[Unreleased]: https://github.com/robodreamer/embodied-policy-lab/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/robodreamer/embodied-policy-lab/compare/24c2e00...v0.7.0
[0.6.0]: https://github.com/robodreamer/embodied-policy-lab/compare/caaef86...24c2e00
[0.5.1]: https://github.com/robodreamer/embodied-policy-lab/compare/69a7ecd...caaef86
[0.5.0]: https://github.com/robodreamer/embodied-policy-lab/compare/a0edca7...69a7ecd
[0.4.0]: https://github.com/robodreamer/embodied-policy-lab/compare/eec20f9...a0edca7
[0.3.0]: https://github.com/robodreamer/embodied-policy-lab/compare/a1eeb3c...eec20f9
[0.2.0]: https://github.com/robodreamer/embodied-policy-lab/compare/f0c574d...a1eeb3c
[0.1.0]: https://github.com/robodreamer/embodied-policy-lab/tree/f0c574d

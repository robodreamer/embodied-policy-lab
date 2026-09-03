# Embodied Policy Lab agent guide

These instructions apply to the entire repository. Embodied Policy Lab is a
local evaluation and observability layer for robot vision-language-action
(VLA) policies and world-action models (WAMs). A change is complete only when
runtime behavior, model and simulator contracts, evidence, tests, and public
documentation agree.

## Mission and claim boundary

The lab answers three practical questions:

1. What did the model receive and predict?
2. What did the robot execute in the simulator?
3. What durable evidence supports the comparison?

Upstream projects remain authoritative for architecture, training, and
publisher benchmark claims. Do not imply that a local integration check
reproduces a paper, that unlike simulator results are directly comparable, or
that this research software is a robot-safety system.

Use these labels precisely:

- **Wiring check:** loading, transport, tensor/action shape, or short execution.
- **Bounded local validation:** a declared task, seed, budget, and trial count on
  stated hardware; insufficient for a general success rate unless designed as
  such.
- **Matched local benchmark:** models share the declared simulator, tasks,
  seeds, budgets, replanning schedule, metrics, and stopping rules.
- **Paper reproduction:** the primary protocol has been matched materially.
  Otherwise call the work an adaptation or local comparison.
- **Generated future:** a model prediction aligned to actual execution. It is
  not a counterfactual planner unless alternatives are generated, scored, and
  used to choose an action.

## Sources of truth

| Concern | Canonical source |
|---|---|
| Model aliases and compatibility | `showcase/backend_registry.py` |
| Policy adapter contracts | `showcase/robocasa_policy_plugins.py` and related plugin modules |
| Predictor semantics | `showcase/world_model_plugins.py` |
| Launch lifecycle and GPU guard | `scripts/run_showcase.sh`, `scripts/run_server.sh` |
| Interactive entry point | `lab`, `bin/embodied-lab` |
| Operator behavior | `docs/operator-guide.md` |
| Model extension boundary | `docs/model-plugins.md` |
| External revisions and licenses | `docs/external-assets.md` |
| Public validation protocols | `docs/validation/` and `docs/benchmarks/` |
| Sanitized public evidence | `results/README.md` |

Update executable or machine-readable behavior before derived documentation.
Do not add a selector option only in HTML, shell text, or Markdown.

## Compatibility guardrails

- Preserve the released checkpoint's cameras, state ordering, image transforms,
  action ordering, horizon, normalization, and simulator version.
- Never project LIBERO's fixed-arm 7D contract onto RoboCasa's
  mobile-manipulator 12D or RoboTwin's bimanual 14D qpos contract. Add an
  explicit adapter/profile instead.
- Valid public pairings are currently π0.5 with LIBERO or RoboCasa, Fast-WAM
  with LIBERO or RoboTwin, Flex-π with LIBERO or RoboTwin, and GR00T N1.5 with
  RoboCasa. RoboTwin WAM profiles expose the shared studio through an explicit
  in-process native adapter and retain the publisher-native batch path. Update
  the registry, tests, compatibility table, and operator guide together when
  this changes.
- Keep heavyweight or conflicting CUDA/Python stacks in isolated runtimes.
  Import model-specific dependencies lazily and communicate through an
  explicit local policy boundary. RoboTwin keeps policy and SAPIEN together in
  its native adapter to avoid serializing three RGB/depth observations.
- Keep one heavyweight model resident at a time unless the user explicitly
  opts into GPU oversubscription.
- Flex-π action-only and full-joint world-action inference are distinct
  experimental modes. Record the selected mode in session evidence.
- Prediction-versus-execution media must be action-aligned and revealed after
  execution. It must not flash, loop automatically, cover the live cameras, or
  be described as control authority.
- Do not register a learned predictor until its inputs, outputs, checkpoint,
  simulator compatibility, latency, failure behavior, and validation gate are
  explicit.

## Change workflow

1. Read the relevant public guide, registry, adapter, tests, and upstream
   license/revision record.
2. State the observation, action, simulator, and evidence contracts before
   changing a model path.
3. Prefer focused adapter or registry changes over cross-runtime rewrites.
4. Add deterministic CPU-safe contract tests. Run GPU/model checks only when
   the matching assets are already available.
5. Update the compatibility matrix and operator documentation for user-visible
   behavior.
6. For dashboard changes, inspect the complete rendered studio and include a
   current screenshot or short clip.
7. Review the staged diff for secrets, personal paths, raw logs, caches,
   checkpoints, and unsupported empirical claims.

## Validation

The default checks must remain CPU-safe and must not download model weights:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
tests/test_lab_cli.sh
bash -n bin/embodied-lab scripts/*.sh tests/*.sh
git diff --check
```

Add a regression test for user-visible bugs. For model-specific validation,
record the lab and upstream revisions, checkpoint identity, hardware/runtime,
simulator and tasks, seeds, action budget, trial count, results, and known
protocol differences. Preserve failures and negative outcomes.

## Artifacts, privacy, and licenses

- Keep model weights, virtual environments, caches, raw traces, and uncurated
  run directories out of Git.
- Public evidence should be concise and sanitized. Retain enough provenance to
  reproduce it without committing personal filesystem paths or credentials.
- Maintainer investigation logs, branch-cleanup history, private integration
  details, and strategy drafts live outside this public repository. Promote
  only stable behavior, protocols, decisions, and claim boundaries here.
- Never commit Hugging Face tokens, SSH keys, private URLs, or copied terminal
  output that may contain them. Follow `SECURITY.md` for disclosures.
- Lab-authored code is Apache-2.0. Upstream repositories, simulator assets, and
  checkpoints keep their own licenses and access terms; do not redistribute
  weights or imply the repository license covers them.

## Definition of done

- Relevant CPU-safe tests and shell checks pass.
- Model/simulator/action contracts remain explicit and tested.
- No heavyweight download was added to CI or the default test path.
- User-facing changes are documented; visual changes have current media.
- Empirical language matches the actual protocol and sample size.
- The staged diff contains no secrets, local paths, raw artifacts, or unrelated
  user changes.

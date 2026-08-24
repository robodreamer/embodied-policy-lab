# 2026-08-17 — Robot VLA consequence-preview and world-model roadmap

## Implementation review and correction

The first implementation called the paired MuJoCo branch a world model. That
overstated the capability: it is a deterministic simulator oracle, so exact
predicted/actual MuJoCo-state matches are expected. It also allowed preview and
media failures to abort the real policy rollout, and an interrupted process
could leave the dashboard marked as running.

The corrected behavior:

- defaults to direct policy execution and labels the optional branch as a
  simulator-oracle diagnostic baseline;
- records prediction setup/inference/finalization failures in
  `preview-audit.jsonl` while continuing the policy rollout;
- distinguishes completed comparisons from failed diagnostic events;
- reconciles interrupted client state and prevents accidental concurrent lab
  sessions independently of the GPU-oversubscription switch;
- keeps policy inference audits independent from optional predictor evidence.

Recent π0.5 evidence showed nonzero base-motion values and a separate binary
control-mode channel. That evidence ruled out silently projecting the lab's
12D mobile-manipulator output into an unrelated 7D arm-only action contract.

## Implemented simulator-oracle baseline

- Keep policy and world-model selection independent.
- Use a paired RoboCasa environment as a non-destructive deterministic oracle.
- Keep comparison opt-in and never make it an execution gate.
- Execute the real action prefix normally, then reveal the matching prediction
  and actual clips side by side.
- Save separate policy, prediction, actual-prefix, and rollout evidence with
  predicted/actual final-state hashes.
- Preserve π0.5 and GR00T N1.5 behind their existing policy plugins.

## Promotion boundary

The earlier DINO-WM / JEPA-WM prototype surface was removed before promotion:
neither candidate had a validated mapping from the lab's 12D RoboCasa action
contract, and disabled selector entries plus an unused submodule made the main
experience noisier without providing a runnable capability. Their research
questions, along with Cosmos and future visual/world-generation candidates,
belong in isolated experiment branches until an exact simulator action and
observation contract passes the same evidence protocol.

World models provide proposals, rankings, or diagnostics. Simulator success and
robot safety remain independent authorities.

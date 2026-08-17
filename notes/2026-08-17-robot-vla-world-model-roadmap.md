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
- writes action-contract diagnostics for every policy response.

Recent π0.5 evidence showed nonzero base-motion values (maximum absolute value
approximately `1.8e-3`) and a binary control-mode channel near `-1`. The old
diagnostic worker treated all five trailing dimensions as base motion, so it
would reject every observed chunk. A learned 7D adapter must separate those
semantics and validate rotation and temporal conventions before activation.

## Implemented simulator-oracle baseline

- Keep policy and world-model selection independent.
- Use a paired RoboCasa environment as a non-destructive deterministic oracle.
- Keep comparison opt-in and never make it an execution gate.
- Execute the real action prefix normally, then reveal the matching prediction
  and actual clips side by side.
- Save separate policy, prediction, actual-prefix, and rollout evidence with
  predicted/actual final-state hashes.
- Preserve π0.5 and GR00T N1.5 behind their existing policy plugins.

## Learned-model sequence

1. Start with DINO-WM DROID/RoboCasa because its predictor is the smaller
   candidate and can run in an isolated Python 3.10 worker.
2. Validate its 7D arm action projection and temporal packing against the
   paired-simulator outcome; never discard 12D base actions silently.
3. Promote DINO-WM into the dashboard only after the compatibility test passes.
4. Repeat the same evidence protocol with the larger JEPA-WM checkpoint.
5. Keep Cosmos as a separate visual/world-generation judge until a checkpoint
   with the exact RoboCasa action domain is validated.
6. Consider V-JEPA2-AC for Franka-style fixed-base experiments and GE-Sim2 only
   through explicit embodiment adapters.

World models provide proposals, rankings, or diagnostics. Simulator success and
robot safety remain independent authorities.

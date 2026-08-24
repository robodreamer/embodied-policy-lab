# Predictor diagnostics and post-execution comparisons

The policy and predictor are deliberately separate selections:

| Policy | Proposes | Predictor / baseline | Produces |
|---|---|---|---|
| π0.5 or GR00T N1.5 | a RoboCasa 12D action chunk | `robocasa-sim` | deterministic ground-truth replay in a cloned MuJoCo state |
| π0.5 or GR00T N1.5 | a RoboCasa 12D action chunk | `none` | nothing; execute directly |

`robocasa-sim` is a **simulator oracle baseline, not a learned world model**.
The launcher creates the live and
counterfactual environments from the same construction RNG state, copies the
current MuJoCo state into the counterfactual branch, and executes only the
configured action prefix there. The real policy prefix then executes normally.
Only after those real actions finish does the dashboard reveal the prediction
beside the recorded execution. It also records predicted/actual final-state
hashes and before/after live-state hashes in `preview-audit.jsonl`.

This is useful for validating state cloning, action handoff, media alignment,
and comparison bookkeeping. Because the same deterministic simulator produces
both branches, exact state agreement is expected and is not evidence of learned
prediction quality. RoboCasa still owns task success. A predictor observes and
compares; it does not approve, reject, or gate policy execution. If predictor
setup, inference, rendering, or comparison fails, the event is audited and the
policy rollout continues without comparisons for that attempt.

## Run it

```bash
./lab # direct execution is the default

# Equivalent explicit launch
./scripts/run_interactive_showcase.sh \
  --backend robocasa \
  --model pi05 \
  --world-model robocasa-sim \
  --compare-world-model
```

Select the simulator oracle and enable the browser checkbox to run the baseline.
The workflow is
**propose → predict privately → execute the real prefix → reveal both clips**.
The comparison length always matches `--replan-steps`. World-model selection and
the checkbox are available between rollouts. Choose `Direct execution (no
preview)` or leave comparison unchecked for ordinary policy execution.

Artifacts are retained under the session directory:

- `previews/*prediction*.mp4` and `*actual*.mp4`: paired finite clips plus
  `latest_prediction.mp4` and `latest_actual.mp4` for the UI;
- `preview-audit.jsonl`: completed, failed, and discarded diagnostic events;
  completed oracle comparisons include action hash, timing, branch outcome,
  live-state non-mutation evidence, predicted/actual final-state hashes, and
  numerical qpos/qvel error checked at an absolute tolerance of `1e-9`;
- `inference-audit.jsonl`: the independent policy request/action record;
- `videos/*.mp4`: actions that were actually executed in the live episode.

## Learned-predictor admission boundary

Learned predictor candidates are intentionally absent from the runnable
registry and default dependencies. Add one only after its camera, temporal, and
action contracts match a supported simulator profile, its output is compared
with saved simulator outcomes, and its latency and memory fit beside the
selected policy. A predictor remains an opt-in, non-gating diagnostic until
those checks pass; simulator success remains authoritative.

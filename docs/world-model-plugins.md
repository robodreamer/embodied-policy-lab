# Post-execution world-model comparisons for robot policies

The policy and world model are deliberately separate selections:

| Policy | Proposes | World model | Predicts |
|---|---|---|---|
| π0.5 or GR00T N1.5 | a RoboCasa 12D action chunk | `robocasa-sim` | rendered consequences in a cloned MuJoCo state |
| π0.5 or GR00T N1.5 | a RoboCasa 12D action chunk | `none` | nothing; execute directly |
| either policy (future adapter) | 7D arm-only projection | DINO-WM / JEPA-WM | a learned latent trajectory |

`robocasa-sim` is the runnable baseline. The launcher creates the live and
counterfactual environments from the same construction RNG state, copies the
current MuJoCo state into the counterfactual branch, and executes only the
configured action prefix there. The real policy prefix then executes normally.
Only after those real actions finish does the dashboard reveal the prediction
beside the recorded execution. It also records predicted/actual final-state
hashes and before/after live-state hashes in `preview-audit.jsonl`.

This is a consequence preview, not a safety certificate. RoboCasa still owns
task success. The world model observes and compares; it does not approve, reject,
or gate policy execution.

## Run it

```bash
./lab

# Equivalent explicit launch
./scripts/run_interactive_showcase.sh \
  --backend robocasa \
  --model pi05 \
  --world-model robocasa-sim \
  --compare-world-model
```

The browser checkbox is off by default. When enabled, the workflow is
**propose → predict privately → execute the real prefix → reveal both clips**.
The comparison length always matches `--replan-steps`. World-model selection and
the checkbox are available between rollouts. Choose `Direct execution (no
preview)` or leave comparison unchecked for ordinary policy execution.

Artifacts are retained under the session directory:

- `previews/*prediction*.mp4` and `*actual*.mp4`: paired finite clips plus
  `latest_prediction.mp4` and `latest_actual.mp4` for the UI;
- `preview-audit.jsonl`: action hash, timing, branch outcome, live-state
  non-mutation evidence, predicted/actual final-state hashes, and numerical
  qpos/qvel error checked at an absolute tolerance of `1e-9`;
- `inference-audit.jsonl`: the independent policy request/action record;
- `videos/*.mp4`: actions that were actually executed in the live episode.

## Learned DINO-WM / JEPA-WM path

The pinned `upstream-jepa-wms` source advertises DROID/RoboCasa checkpoints, but
those checkpoints consume 7D arm actions (XYZ, Euler rotation, gripper). This
lab's policy plugins expose a 12D mobile-manipulator contract. The learned
options therefore remain visibly unavailable in the main selector until all of
the following are measured:

1. the five base/control dimensions are zero or have an explicit learned map;
2. simulator control steps are calibrated to the world model's temporal rate;
3. camera crop and context-window packing match checkpoint training;
4. latent predictions are compared with the saved simulator counterfactual;
5. latency and memory fit alongside the selected policy.

Set up the isolated Python 3.10 runtime without disturbing either policy env:

```bash
JEPA_DOWNLOAD_WEIGHTS=1 JEPA_WORLD_MODEL=dino_wm_droid \
  ./scripts/setup_world_models.sh
```

`showcase/serve_jepa_world_model.py` is a persistent, loopback-only diagnostic
worker. It rejects nonzero base actions rather than silently discarding them and
can report latent displacement plus latent error against a simulator outcome.
Its result is diagnostic evidence only; it is not yet an execution gate.

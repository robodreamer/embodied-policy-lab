# Reproduction results

## 2026-08-05 — prompt handoff and rollout-budget diagnosis

Session `20260805-153435` reproduced both reported symptoms. The dashboard
acknowledged a fresh-start command before the new environment published its
prompt, allowing the previous state to repaint the draft briefly. Failed custom
commands also stopped at the fixed `libero_spatial` cap of 220 actions, which
took about 17.7–18.0 seconds on this workstation. A successful scored variation
ended after 105 policy steps (8.95 seconds) because LIBERO detected the selected
goal; that early stop is correct for a scored attempt.

The dashboard now retains a submitted draft until the acknowledged state also
matches its prompt, task, evaluation mode, and budget. Fresh rollouts support
Standard (1×), Extended (2×), and Long (3×) budgets, with Extended as the
default. Manual edits default to a Long custom experiment. Custom experiments
run the full budget and are reported as unscored; true goal-preserving variants
can be marked Scored and still stop as soon as the selected goal succeeds.

Session `20260805-154934` validated the fix against the real local policy. A
browser-driven fresh start sampled the instruction field 97 times during the
handoff; every sample retained the newly submitted prompt and the pending state
was visible. A custom 220-step run reached the hidden LIBERO goal but continued
through all 44 policy requests and finished `unscored` after 28.12 seconds. The
same scene then ran a true scored wording variation, succeeded after 23 policy
requests in 9.22 seconds, and was recorded separately as 1/1 scored plus one
unscored custom attempt.

## 2026-08-05 — prompt delivery and randomized generation audit

A live interactive session was inspected after four task-4 rollouts. The command
queue, client log, state, and per-attempt prompt timelines all agreed on the
exact step-0 inputs: a `throw it` instruction, two `put it in the drawer`
variants, and `close the drawer`. Results were one success followed by three
failures. This confirms the edits reached the client model input; visually
similar trajectories are policy behavior for novel LIBERO instructions, not the
dashboard silently reusing the canonical text.

The next-run instrumentation now records every synchronous request in
`inference-audit.jsonl`, including the exact prompt and SHA-256 digests of both
the prompt and returned action chunk. The dashboard displays the acknowledged
prompt digest. A full validation session recorded 58 audited requests across
three completed attempts. Its two scored attempts succeeded; the exploratory
`Position the ramekin near the plate.` attempt also triggered the original
LIBERO success condition, showing the policy's familiar-task fallback even
though all 16 requests in that attempt carried the exploratory text. That
attempt was correctly excluded, leaving a scored result of 2/2 and one
unscored exploration.

Random local generation was tested repeatedly with `gemma3:1b`. Accepted scored
variations retain the ordered object/location/destination terms. Exploratory
outputs included distinct commands such as `Pick up the ramekin`, `Position the
ramekin near the plate`, and `Pick up the black bowl`; these are explicitly
excluded from LIBERO success-rate denominators. After adding cue validation and
a local-model fallback for repeated output, a final batch produced eight unique
exploratory drafts in eight requests without an error.

## 2026-08-05 — local-LLM interactive workflow

The revised three-step UI and combined GPU workflow were validated with Ollama
`gemma3:1b` and the local π0.5 checkpoint:

- the simulator remained idle until an explicit **Start a fresh scored rollout**
  action;
- the dashboard generated `Pick up the black bowl and place it on the plate.`
  locally through `127.0.0.1:11434`;
- the generated prompt was recorded with source `local_llm`;
- the resulting task-0 rollout completed successfully: **1/1**;
- warm median π0.5 inference: approximately **135 ms** per action chunk;
- combined peak GPU memory: **20,483 MiB**;
- traced π0.5/simulator network verdict: **`loopback_only`**.

Launch ordering was also tested: the runner unloads an already-resident Ollama
model before π0.5 checkpoint restoration, then allows the small prompt model to
load on demand. This avoids JAX checkpoint-load memory pressure on the 24 GB GPU.

## 2026-08-05 — interactive prompt/reset validation

The persistent browser-controlled workflow was exercised through its HTTP
control path with the real local checkpoint and simulator:

- canonical task-0 rollout: **success**;
- typed paraphrase followed by an interactive reset: **success**;
- completed-attempt success rate: **2/2 (100%)**;
- task switching and clean session stop: verified;
- warm mean inference: approximately **127 ms** per action chunk;
- peak observed GPU memory: **19,098 MiB**;
- network audit: **`loopback_only`**, with only `127.0.0.1:8000` observed;
- local-LLM adapter: enabled/disabled behavior and non-loopback URL rejection
  verified against a temporary local test endpoint.

This is an integration validation, not a statistically meaningful prompt
robustness result. Run many seeded attempts before comparing prompt variants.

## 2026-08-05 — instrumented local showcase

The browser-console workflow was validated on `libero_spatial` task 1 with the
cached official checkpoint:

- result: **1/1 successful episode**;
- runtime: local JAX/CUDA on the RTX PRO 5000 Blackwell Laptop GPU;
- warm median action-chunk inference: approximately **132 ms**;
- warm p95 action-chunk inference: approximately **141 ms**;
- peak observed GPU memory: **19,096 MiB**;
- peak observed GPU utilization: **94%**;
- network audit: **`loopback_only`**;
- observed IP destination: `127.0.0.1:8000`;
- observed remote IP destinations: **none**.

The instrumented runner completed successfully and generated its Markdown/JSON
report, raw syscall traces, GPU telemetry, dashboard state, and MP4 rollout.

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

# Contributing to Embodied Policy Lab

Thanks for helping make robot-model experiments easier to reproduce and harder
to misinterpret. Small fixes, new policy/simulator adapters, dashboard
improvements, validation protocols, and documentation are all welcome.

## Before opening a change

1. Search existing issues and pull requests.
2. Keep heavyweight upstream code, checkpoints, caches, and run artifacts out
   of this repository.
3. Add new models through the documented
   [policy adapter boundary](docs/model-plugins.md), and add predictors through
   the [world-model boundary](docs/world-model-plugins.md).
4. Preserve each released checkpoint's observation and action contract. If an
   experiment intentionally changes it, label the result as an ablation.

## Local checks

The default suite is CPU-safe:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
tests/test_lab_cli.sh
bash -n bin/embodied-lab scripts/*.sh tests/*.sh
git diff --check
```

Run model-specific smoke tests only when the matching simulator, checkpoint,
and GPU runtime are already available. Do not make CI download gated or
multi-gigabyte assets.

## Experimental evidence checklist

Pull requests that add or change empirical claims should record:

- the lab commit and every upstream source revision;
- model/checkpoint identity and integrity digest when publishers provide one;
- OS, GPU, driver/CUDA, and relevant Python runtime;
- simulator, task collection, task IDs, seeds, action budget, and trial count;
- whether numbers are a wiring check, provisional pilot, matched local
  benchmark, or paper reproduction;
- raw machine-readable results plus a sanitized human-readable summary;
- known differences from the publisher's evaluation protocol.

Never commit Hugging Face tokens, private URLs, personal paths, or raw logs that
may contain credentials. The repository's secret scan is a backstop, not a
substitute for reviewing the staged diff.

## Pull requests

Keep changes focused and explain the user-visible behavior first. Include
screenshots or short clips for dashboard changes and exact commands for new
workflows. Update the compatibility matrix and operator guide when a new
profile becomes runnable.

By submitting a contribution, you agree that it is licensed under the
repository's [Apache License 2.0](LICENSE).

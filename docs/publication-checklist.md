# Public repository checklist

Use this after the stacked pull requests are merged. Repository settings are
intentionally not changed by the documentation branch.

## GitHub About panel

Suggested description:

> Run and compare robot VLAs and world-action models in LIBERO and RoboCasa,
> with live rollouts, predicted-future replay, and reproducible local evidence.

Suggested topics:

```text
robotics embodied-ai vision-language-action world-action-model
libero robocasa mujoco reproducible-research
```

Upload [`assets/studio-overview.png`](assets/studio-overview.png) in
**Settings → General → Social preview**. The current product capture is a
near-2:1 image that keeps the studio identity, active model, task controls, and
local-runtime status legible on GitHub, X, and LinkedIn.

## Before changing visibility

- Merge the Fast-WAM, Flex-π, and public-launch stack in order.
- Confirm `main` passes both CI jobs: CPU-safe tests and secret scanning.
- Confirm GitHub recognizes `LICENSE` as Apache-2.0 and renders
  `CITATION.cff` under **Cite this repository**.
- Review the complete Git history with a secret scanner, not only the tip.
- Check issue forms and the private Security Advisory link.
- Verify every README image through GitHub's rendered view.
- Keep caches, checkpoints, raw sessions, personal paths, and access tokens
  untracked.

## First release

Use a research-preview version such as `v0.1.0` only after the documented setup
has been tested from a fresh clone. Release notes should identify validated
profiles, known experimental paths, hardware used, and exact upstream
revisions. Do not attach third-party checkpoints or simulator assets.

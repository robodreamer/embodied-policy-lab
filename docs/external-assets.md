# External code and model assets

The repository's [Apache-2.0 license](../LICENSE) covers lab-authored code and
documentation only. It does not replace the licenses shipped by Git submodules,
revision-checked sibling repositories, simulator assets, or model publishers.

## Source components

| Component | How it is consumed | Upstream source license |
|---|---|---|
| Physical Intelligence OpenPI | pinned Git submodule | Apache-2.0 |
| RoboCasa | pinned Git submodule | MIT |
| RoboCasa OpenPI fork | pinned Git submodule | Apache-2.0 |
| robosuite | pinned Git submodule | MIT |
| RoboCasa Isaac-GR00T fork | pinned Git submodule | Apache-2.0 |
| Fast-WAM | revision-checked sibling checkout | MIT |
| Flex-π | revision-checked sibling checkout | MIT |
| RoboTwin 2.0 | exact-revision sibling checkout; model vendors retain local adapters | MIT |

Each component's own `LICENSE` file is authoritative. Preserve it when copying
or modifying upstream code.

## Model assets

Embodied Policy Lab does not redistribute model weights. Setup scripts fetch
the artifacts below from their publishers and verify the revisions and hashes
used by the validated profiles.

| Profile | Publisher | Source license | Weight license | Local verification |
|---|---|---|---|---|
| Fast-WAM LIBERO | `yuanty/fastwam` | MIT | Not specified on the publisher's Hugging Face model card as of 2026-08-24 | Pinned Hub revision plus SHA-256 for checkpoint and statistics |
| Flex-π LIBERO | `flex-pi/flexpi-libero` | MIT | MIT | Pinned Hub/source revisions plus SHA-256 for checkpoint, config, statistics, intrinsics, VAE, T5/tokenizer, and DINOv3 assets |
| Fast-WAM RoboTwin | `yuanty/fastwam` | MIT | Not separately specified on the model card as of 2026-09-02 | Pinned Hub/source/RoboTwin revisions plus SHA-256 for checkpoint and statistics |
| Flex-π RoboTwin | `flex-pi/flexpi-robotwin` | MIT | MIT | Pinned Hub/source/RoboTwin revisions plus SHA-256 for checkpoint, config, statistics, VAE, T5/tokenizer, and DINOv3 assets |

The source license does not automatically establish a license for separately
published weights. Confirm the publisher's current terms before redistributing
any downloaded artifact.

RoboTwin's roughly 16 GB simulator asset bundle is downloaded from the
publisher by `scripts/setup_robotwin.sh --download-assets` and remains outside
Git. The lab pins the stable `release` source commit
`bf44be51cf5717a5595ce59447f2cf5263d2aa95`; see the
[foundation note](validation/robotwin-foundation.md) for the complete contract.

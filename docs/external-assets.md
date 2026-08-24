# External model assets

Embodied Policy Lab does not redistribute model weights. Setup scripts fetch
the artifacts below from their publishers and verify the revisions and hashes
used by the validated profiles.

| Profile | Publisher | Source license | Weight license | Local verification |
|---|---|---|---|---|
| Fast-WAM LIBERO | `yuanty/fastwam` | MIT | Not specified on the publisher's Hugging Face model card as of 2026-08-24 | Pinned Hub revision plus SHA-256 for checkpoint and statistics |
| Flex-π LIBERO | `flex-pi/flexpi-libero` | MIT | MIT | Pinned Hub/source revisions plus SHA-256 for checkpoint, config, statistics, intrinsics, VAE, T5/tokenizer, and DINOv3 assets |

The source license does not automatically establish a license for separately
published weights. Confirm the publisher's current terms before redistributing
any downloaded artifact.

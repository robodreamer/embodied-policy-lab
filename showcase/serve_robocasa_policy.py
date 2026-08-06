#!/usr/bin/env python3
"""Serve a RoboCasa OpenPI checkpoint without requiring training datasets.

The upstream RoboCasa training config contains the paths of every dataset used
to calculate aggregate normalization statistics. Those datasets are not needed
for inference because the published checkpoint includes its own norm stats, but
the upstream policy factory currently tries to open the training paths before
it reads the checkpoint assets. This launcher disables only that fallback and
passes the checkpoint's statistics explicitly.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import socket

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_pretrain_human300")
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    stats_path = checkpoint / "assets" / "norm_stats.json"
    params_path = checkpoint / "params" / "_METADATA"
    if not stats_path.is_file() or not params_path.is_file():
        raise FileNotFoundError(
            f"Incomplete inference checkpoint at {checkpoint}; expected "
            "assets/norm_stats.json and params/_METADATA"
        )

    train_config = config.get_config(args.config)
    if not hasattr(train_config.data, "data_dirs"):
        raise TypeError(
            f"Config {args.config!r} uses {type(train_config.data).__name__}, "
            "not the expected RoboCasa data config"
        )

    dataset_count = len(train_config.data.data_dirs or [])
    inference_data_config = dataclasses.replace(train_config.data, data_dirs=None)
    inference_config = dataclasses.replace(train_config, data=inference_data_config)
    norm_stats = normalize.load(checkpoint / "assets")

    logging.info(
        "Using checkpoint normalization stats from %s; skipped %d training dataset paths",
        stats_path,
        dataset_count,
    )
    policy = policy_config.create_trained_policy(
        inference_config,
        checkpoint,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
    )

    hostname = socket.gethostname()
    logging.info(
        "Creating local policy server (host: %s, port: %d)", hostname, args.port
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

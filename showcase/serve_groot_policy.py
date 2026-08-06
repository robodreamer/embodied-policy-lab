"""Serve the official RoboCasa GR00T N1.5 checkpoint over local ZeroMQ."""

from __future__ import annotations

import argparse

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=args.denoising_steps,
    )
    print(
        f"Serving NVIDIA Isaac GR00T N1.5 on tcp://{args.host}:{args.port}",
        flush=True,
    )
    RobotInferenceServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()

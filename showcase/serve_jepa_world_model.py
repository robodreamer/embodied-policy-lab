"""Persistent diagnostic worker for the pinned JEPA-WMs runtime.

This worker intentionally exposes latent prediction diagnostics only. It does
not claim to gate the 12D mobile-manipulator policy contract until the 7D
projection and temporal calibration are validated against saved RoboCasa
counterfactuals.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("dino_wm_droid", "jepa_wm_droid"), default="dino_wm_droid"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--upstream-dir", default="upstream-jepa-wms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_root = pathlib.Path(args.allowed_root).resolve()
    upstream_dir = pathlib.Path(args.upstream_dir).resolve()
    sys.path.insert(0, str(upstream_dir))

    import torch
    import hubconf

    started = time.perf_counter()
    model, _ = getattr(hubconf, args.model)(pretrained=True, device=args.device)
    load_duration_ms = round((time.perf_counter() - started) * 1000.0, 2)

    def read_image(raw_path: str):
        path = pathlib.Path(raw_path).resolve()
        if not path.is_relative_to(allowed_root):
            raise ValueError(f"Image must be under allowed root {allowed_root}")
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None, None]

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self.send_json(
                    {
                        "ready": True,
                        "model": args.model,
                        "device": args.device,
                        "load_duration_ms": load_duration_ms,
                        "contract": "diagnostic-7d-manip-v1",
                    }
                )
            else:
                self.send_json({"error": "unknown route"}, 404)

        def do_POST(self):
            if self.path != "/predict-latent":
                self.send_json({"error": "unknown route"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 2_000_000:
                    raise ValueError("Request exceeds 2 MB")
                body = json.loads(self.rfile.read(length))
                actions_12d = np.asarray(body["actions"], dtype=np.float32)
                if actions_12d.ndim != 2 or actions_12d.shape[1] != 12:
                    raise ValueError("actions must have shape [T, 12]")
                base_magnitude = float(np.abs(actions_12d[:, 7:]).max(initial=0.0))
                tolerance = float(body.get("base_action_tolerance", 1e-6))
                if base_magnitude > tolerance:
                    raise ValueError(
                        "DROID checkpoint accepts 7D arm actions only; this chunk "
                        f"contains base/control magnitude {base_magnitude:.6g}"
                    )
                context = read_image(str(body["context_image_path"]))
                actions = torch.from_numpy(actions_12d[:, :7])[:, None].to(
                    model.device, dtype=torch.float32
                )
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    encoded = model.encode(context)
                    predicted = model.unroll(encoded, act_suffix=actions)
                    first = predicted[0].float()
                    final = predicted[-1].float()
                    latent_displacement = float((final - first).pow(2).mean().sqrt())
                    actual_error = None
                    if body.get("actual_image_path"):
                        actual = model.encode(
                            read_image(str(body["actual_image_path"]))
                        )[0]
                        actual_error = float((final - actual.float()).pow(2).mean())
                self.send_json(
                    {
                        "model": args.model,
                        "contract": "diagnostic-7d-manip-v1",
                        "predicted_steps": len(actions),
                        "latent_displacement_rms": latent_displacement,
                        "simulator_outcome_latent_mse": actual_error,
                        "inference_duration_ms": round(
                            (time.perf_counter() - inference_started) * 1000.0, 2
                        ),
                        "diagnostic_only": True,
                        "calibration_status": "temporal mapping not yet validated",
                    }
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, message, *values):
            print("jepa-world-model: " + message % values, flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"JEPA world-model diagnostic worker: http://127.0.0.1:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

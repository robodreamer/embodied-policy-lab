#!/usr/bin/env python3
"""Serve the staged Fast-WAM adapter over localhost HTTP."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import pathlib
import traceback
from typing import Any

from fastwam_policy import CHECKPOINT_FILENAME, STATS_FILENAME, StagedFastWamPolicy
from libero_policy_plugins import decode_uint8_payload


MAX_REQUEST_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-root", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--stats", type=pathlib.Path)
    parser.add_argument("--artifact-dir", type=pathlib.Path)
    return parser.parse_args()


def handler_for(policy: StagedFastWamPolicy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FastWAMPolicy/1"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/healthz":
                self._json(
                    200, {"status": "ok", "requests": policy.metadata["requests"]}
                )
            elif self.path == "/metadata":
                self._json(200, policy.metadata)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/infer":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError(
                        f"Content-Length must be in [1,{MAX_REQUEST_BYTES}], got {length}"
                    )
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if payload.get("schema_version") != 1:
                    raise ValueError("Unsupported request schema_version")
                result = policy.infer(
                    external=decode_uint8_payload(payload["external"], name="external"),
                    wrist=decode_uint8_payload(payload["wrist"], name="wrist"),
                    state=payload["state"],
                    task=payload["prompt"],
                )
                self._json(200, result)
            except (KeyError, TypeError, ValueError) as error:
                self._json(400, {"error": f"{type(error).__name__}: {error}"})
            except Exception as error:  # Keep server alive for inspectable failures.
                traceback.print_exc()
                self._json(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"fastwam-http {self.address_string()} {format % args}", flush=True)

    return Handler


def main() -> None:
    args = parse_args()
    upstream = args.upstream_root.resolve()
    checkpoint = (
        args.checkpoint
        or upstream / "checkpoints" / "fastwam_release" / CHECKPOINT_FILENAME
    )
    stats = args.stats or upstream / "checkpoints" / "fastwam_release" / STATS_FILENAME
    print("Loading staged Fast-WAM checkpoint on CPU...", flush=True)
    policy = StagedFastWamPolicy(
        upstream_root=upstream,
        checkpoint=checkpoint,
        stats=stats,
        artifact_dir=args.artifact_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_for(policy))
    server.daemon_threads = True
    print(
        f"Fast-WAM ready at http://{args.host}:{args.port} "
        f"(CPU load {policy.load_seconds:.1f}s)",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Dependency-free local web server for the π0.5 showcase dashboard."""

import argparse
import csv
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"phase": "waiting", "message": "Waiting for the simulator"}


def read_telemetry(path):
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except FileNotFoundError:
        rows = []
    if not rows:
        return {"samples": 0}

    def number(row, key):
        try:
            return float(row[key].strip())
        except (KeyError, TypeError, ValueError):
            return 0.0

    latest = rows[-1]
    return {
        "samples": len(rows),
        "timestamp": latest.get("timestamp", "").strip(),
        "gpu": latest.get("name", "").strip(),
        "memory_mib": number(latest, "memory.used [MiB]"),
        "utilization_pct": number(latest, "utilization.gpu [%]"),
        "power_w": number(latest, "power.draw [W]"),
        "temperature_c": number(latest, "temperature.gpu"),
        "max_memory_mib": max(number(row, "memory.used [MiB]") for row in rows),
        "max_utilization_pct": max(number(row, "utilization.gpu [%]") for row in rows),
        "max_power_w": max(number(row, "power.draw [W]") for row in rows),
    }


def write_json_atomic(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def require_loopback_url(value):
    if not value:
        return None
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in (
        "localhost",
        "127.0.0.1",
        "::1",
    ):
        raise ValueError("Local LLM URL must use localhost or a loopback IP")
    return value


def generate_prompt(url, model, instruction):
    if "chat/completions" in url:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the requested robot manipulation goal as one short, "
                        "unambiguous imperative sentence. Return only that sentence."
                    ),
                },
                {"role": "user", "content": instruction},
            ],
            "temperature": 0.4,
            "max_tokens": 64,
            "stop": ["\n"],
        }
    else:
        payload = {
            "model": model,
            "prompt": (
                "Rewrite this robot manipulation goal as one short, unambiguous "
                "imperative sentence. Return only the sentence:\n" + instruction
            ),
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 48,
                "stop": ["\n"],
            },
        }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "chat/completions" in url:
        return str(result["choices"][0]["message"]["content"]).strip()
    return str(result["response"]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--static-dir", required=True)
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--local-llm-url", default="")
    parser.add_argument("--local-llm-model", default="")
    args = parser.parse_args()
    session_dir = pathlib.Path(args.session_dir).resolve()
    static_dir = pathlib.Path(args.static_dir).resolve()
    local_llm_url = require_loopback_url(args.local_llm_url.strip())
    local_llm_model = args.local_llm_model.strip()
    llm_event_lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(static_dir), **handler_kwargs)

        def _json(self, payload, status=200):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _file(self, path, content_type):
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route == "/api/state":
                self._json(read_json(session_dir / "state.json"))
            elif route == "/api/telemetry":
                self._json(read_telemetry(session_dir / "gpu.csv"))
            elif route == "/api/config":
                self._json(
                    {
                        "local_llm_enabled": bool(local_llm_url and local_llm_model),
                        "local_llm_model": local_llm_model or None,
                    }
                )
            elif route == "/frames/external.jpg":
                self._file(session_dir / "frames" / "external.jpg", "image/jpeg")
            elif route == "/frames/wrist.jpg":
                self._file(session_dir / "frames" / "wrist.jpg", "image/jpeg")
            else:
                super().do_GET()

        def do_POST(self):
            route = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    self._json({"error": "Request is too large"}, 413)
                    return
                body = json.loads(self.rfile.read(length) or b"{}")
                if route == "/api/control":
                    allowed = {
                        "reset",
                        "start",
                        "start_rollout",
                        "stop",
                        "set_prompt",
                        "set_task",
                    }
                    if body.get("action") not in allowed:
                        self._json({"error": "Unsupported control action"}, 400)
                        return
                    command = dict(body)
                    command["id"] = time.time_ns()
                    command["created_at"] = time.time()
                    write_json_atomic(session_dir / "control.json", command)
                    controls_dir = session_dir / "controls"
                    controls_dir.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(
                        controls_dir / "{}.json".format(command["id"]), command
                    )
                    self._json({"accepted": True, "command": command})
                elif route == "/api/generate-prompt":
                    if not local_llm_url or not local_llm_model:
                        self._json({"error": "No local LLM is configured"}, 503)
                        return
                    instruction = str(body.get("instruction", "")).strip()
                    if not instruction:
                        self._json({"error": "Instruction is empty"}, 400)
                        return
                    started = time.perf_counter()
                    prompt = generate_prompt(
                        local_llm_url, local_llm_model, instruction
                    )
                    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
                    event = {
                        "created_at": time.time(),
                        "model": local_llm_model,
                        "endpoint_host": urllib.parse.urlparse(local_llm_url).hostname,
                        "instruction": instruction,
                        "generated_prompt": prompt,
                        "duration_ms": duration_ms,
                    }
                    with llm_event_lock:
                        with (session_dir / "llm-generations.jsonl").open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write(json.dumps(event) + "\n")
                    self._json(
                        {
                            "prompt": prompt,
                            "model": local_llm_model,
                            "duration_ms": duration_ms,
                            "local": True,
                        }
                    )
                else:
                    self._json({"error": "Unknown endpoint"}, 404)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                self._json({"error": str(error)}, 400)
            except (urllib.error.URLError, TimeoutError) as error:
                self._json({"error": "Local LLM request failed: {}".format(error)}, 502)

        def log_message(self, message, *values):
            if not self.path.startswith(("/api/", "/frames/")):
                print("dashboard: " + message % values)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("Dashboard: http://127.0.0.1:{}".format(args.port), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

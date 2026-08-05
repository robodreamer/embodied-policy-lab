"""Dependency-free local web server for the π0.5 showcase dashboard."""

import argparse
import csv
import json
import os
import pathlib
import re
import secrets
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


def _normalize_prompt(value):
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _clean_prompt(value):
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    return re.sub(r"^(?:[-*•]+|\d+[.)])\s*", "", first_line).strip(' "')


IGNORED_GOAL_WORDS = {
    "a",
    "an",
    "and",
    "it",
    "pick",
    "place",
    "put",
    "set",
    "the",
    "up",
}
ACTION_WORDS = {"lift", "move", "pick", "place", "position", "put", "relocate", "set"}
NON_OBJECT_WORDS = IGNORED_GOAL_WORDS | {
    "between",
    "black",
    "blue",
    "bottom",
    "brown",
    "gold",
    "green",
    "in",
    "inside",
    "left",
    "next",
    "of",
    "on",
    "red",
    "right",
    "silver",
    "to",
    "top",
    "white",
    "wooden",
    "yellow",
}


def _goal_terms(goal):
    return [
        word
        for word in re.findall(r"[a-z0-9]+", goal.lower())
        if word not in IGNORED_GOAL_WORDS
    ]


def _scene_objects(goal):
    return list(
        dict.fromkeys(
            word
            for word in re.findall(r"[a-z0-9]+", goal.lower())
            if word not in NON_OBJECT_WORDS
        )
    )


def _exploration_cues(goal):
    scene_objects = _scene_objects(goal)
    cues = []
    for target in scene_objects:
        cues.extend(
            [
                "pick up the {}".format(target),
                "lift the {}".format(target),
            ]
        )
        for neighbor in scene_objects:
            if neighbor == target:
                continue
            cues.extend(
                [
                    "move the {} next to the {}".format(target, neighbor),
                    "relocate the {} away from the {}".format(target, neighbor),
                    "position the {} near the {}".format(target, neighbor),
                    "place the {} beside the {}".format(target, neighbor),
                ]
            )
    return cues


def _ordered_terms(words, terms, start=0):
    position = start
    for term in terms:
        try:
            position = words.index(term, position) + 1
        except ValueError:
            return None
    return position


def _preserves_scored_goal(candidate, goal):
    goal_words = re.findall(r"[a-z0-9]+", goal.lower())
    candidate_words = re.findall(r"[a-z0-9]+", candidate.lower())
    try:
        split_at = goal_words.index("place")
    except ValueError:
        return all(term in candidate_words for term in _goal_terms(goal))
    source_terms = [
        word for word in goal_words[2:split_at] if word not in {"a", "an", "and", "the"}
    ]
    destination_terms = [
        word
        for word in goal_words[split_at + 1 :]
        if word not in {"a", "an", "it", "the"}
    ]
    source_end = _ordered_terms(candidate_words, source_terms)
    return (
        source_end is not None
        and _ordered_terms(candidate_words, destination_terms, source_end) is not None
    )


def _valid_generated_prompt(candidate, goal, mode, required_objects=None):
    candidate_words = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    if mode == "scored_variation":
        return _preserves_scored_goal(candidate, goal)
    object_terms = set(_scene_objects(goal))
    return (
        bool(candidate_words & object_terms)
        and bool(candidate_words & ACTION_WORDS)
        and not _preserves_scored_goal(candidate, goal)
        and set(required_objects or ()).issubset(candidate_words)
    )


def _generation_instruction(goal, mode, avoid):
    random_styles = [
        "lead with the destination",
        "lead with the object location",
        "use a concise operator-style command",
        "use natural conversational wording",
        "use a spatially precise command",
        "use a different verb and sentence structure",
    ]
    style = secrets.choice(random_styles)
    exclusions = "\n".join("- {}".format(item) for item in avoid[-8:])
    nonce = secrets.token_hex(4)
    required_terms = ", ".join(_goal_terms(goal))
    exploration_cues = _exploration_cues(goal)
    exploration_cue = (
        secrets.choice(exploration_cues) if exploration_cues else "move the object"
    )
    if mode == "exploratory":
        request = (
            "Invent one different, physically plausible tabletop robot command for "
            "the same scene. Use only objects explicitly mentioned in the reference "
            "goal. Prefer safe pick, place, or move actions. Do not throw objects, "
            "invent objects, use vague pronouns, or repeat the scored goal. Name at "
            "least one concrete scene object and start with an action verb. This is "
            "an unscored exploration. Follow the exploration cue closely and include "
            "its named objects verbatim."
        )
    else:
        request = (
            "Create a randomized alternative wording of the reference goal. Preserve "
            "the exact object, spatial relationship, action, and destination so the "
            "same simulator success condition remains valid. Do not add or remove "
            "steps. Every required term listed below must appear verbatim in the result."
        )
    instruction = """You write commands for a tabletop robot.
{request}
Variation direction: {style}.
Reference goal: {goal}
Required reference terms: {required_terms}
Exploration cue (use only in exploratory mode): {exploration_cue}
Do not return any of these previous commands:
{exclusions}
Random variation token: {nonce}
Return exactly one short imperative sentence and nothing else.""".format(
        request=request,
        style=style,
        goal=goal,
        required_terms=required_terms,
        exploration_cue=exploration_cue,
        exclusions=exclusions or "- none",
        nonce=nonce,
    )
    return instruction, exploration_cue


def _request_generation(url, model, instruction):
    if "chat/completions" in url:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the generation instructions exactly.",
                },
                {"role": "user", "content": instruction},
            ],
            "temperature": 0.85,
            "top_p": 0.9,
            "max_tokens": 64,
            "stop": ["\n"],
        }
    else:
        payload = {
            "model": model,
            "prompt": instruction,
            "stream": False,
            "options": {
                "temperature": 0.85,
                "top_p": 0.9,
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


def generate_prompt(url, model, goal, mode, avoid):
    normalized_avoid = {_normalize_prompt(item) for item in [goal, *avoid]}
    candidate = ""
    last_exploration_cue = ""
    attempts = 12 if mode == "exploratory" else 8
    for _ in range(attempts):
        instruction, last_exploration_cue = _generation_instruction(goal, mode, avoid)
        candidate = _clean_prompt(_request_generation(url, model, instruction))
        if (
            candidate
            and _normalize_prompt(candidate) not in normalized_avoid
            and _valid_generated_prompt(
                candidate,
                goal,
                mode,
                _scene_objects(last_exploration_cue) if mode == "exploratory" else None,
            )
        ):
            return candidate
        if candidate:
            avoid.append(candidate)
            normalized_avoid.add(_normalize_prompt(candidate))
    if mode == "exploratory":
        cues = _exploration_cues(goal)
        secrets.SystemRandom().shuffle(cues)
        if last_exploration_cue:
            cues.insert(0, last_exploration_cue)
        for cue in cues:
            if _normalize_prompt(cue) in normalized_avoid:
                continue
            forced_instruction = (
                "Return exactly the robot command below as one imperative sentence. "
                "Do not explain or change its objects:\n" + cue
            )
            candidate = _clean_prompt(
                _request_generation(url, model, forced_instruction)
            )
            if (
                candidate
                and _normalize_prompt(candidate) not in normalized_avoid
                and _valid_generated_prompt(candidate, goal, mode, _scene_objects(cue))
            ):
                return candidate
    raise ValueError("The local model repeated previous prompts; try Generate again")


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
                        "prompt_generation_modes": ["scored_variation", "exploratory"],
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
                    goal = str(body.get("goal", body.get("instruction", ""))).strip()
                    mode = str(body.get("mode", "scored_variation"))
                    if not goal:
                        self._json({"error": "Selected task goal is empty"}, 400)
                        return
                    if mode not in ("scored_variation", "exploratory"):
                        self._json({"error": "Unsupported generation mode"}, 400)
                        return
                    event_path = session_dir / "llm-generations.jsonl"
                    previous = []
                    try:
                        for line in event_path.read_text(encoding="utf-8").splitlines():
                            event = json.loads(line)
                            if event.get("goal", event.get("instruction")) == goal:
                                previous.append(event.get("generated_prompt", ""))
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    current_draft = str(body.get("instruction", "")).strip()
                    if current_draft:
                        previous.append(current_draft)
                    started = time.perf_counter()
                    prompt = generate_prompt(
                        local_llm_url, local_llm_model, goal, mode, previous
                    )
                    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
                    event = {
                        "created_at": time.time(),
                        "model": local_llm_model,
                        "endpoint_host": urllib.parse.urlparse(local_llm_url).hostname,
                        "goal": goal,
                        "mode": mode,
                        "instruction": current_draft,
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
                            "mode": mode,
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

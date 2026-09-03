"""Generate a portable Markdown/JSON report for a showcase session."""

import argparse
import csv
import datetime
import glob
import ipaddress
import json
import pathlib
import re


IPV4_PATTERN = re.compile(r'inet_addr\("([^"]+)"\)')
IPV6_PATTERN = re.compile(r'inet_pton\([^,]+, "([^"]+)"')
PORT_PATTERN = re.compile(r"(?:htons\((\d+)\)|sin6?_port=(\d+))")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def telemetry(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def number(row, key):
        try:
            return float(row[key].strip())
        except (KeyError, ValueError):
            return 0.0

    return {
        "samples": len(rows),
        "max_memory_mib": max(
            (number(row, "memory.used [MiB]") for row in rows), default=0
        ),
        "max_utilization_pct": max(
            (number(row, "utilization.gpu [%]") for row in rows), default=0
        ),
        "max_power_w": max((number(row, "power.draw [W]") for row in rows), default=0),
        "max_temperature_c": max(
            (number(row, "temperature.gpu") for row in rows), default=0
        ),
        "gpu": rows[-1].get("name", "").strip() if rows else "unknown",
    }


def network_destinations(session_dir):
    destinations = set()
    for filename in glob.glob(str(session_dir / "network-*")):
        text = pathlib.Path(filename).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "connect(" not in line and "sendto(" not in line:
                continue
            ip_match = IPV4_PATTERN.search(line) or IPV6_PATTERN.search(line)
            if not ip_match:
                continue
            address = ip_match.group(1)
            if not address:
                continue
            port_match = PORT_PATTERN.search(line)
            port = (
                next((value for value in port_match.groups() if value), "?")
                if port_match
                else "?"
            )
            destinations.add((address, port))
    remote = []
    local = []
    for address, port in sorted(destinations):
        try:
            is_loopback = ipaddress.ip_address(address).is_loopback
        except ValueError:
            is_loopback = False
        (local if is_loopback else remote).append("{}:{}".format(address, port))
    verdict = "remote_detected" if remote else "loopback_only"
    return {"verdict": verdict, "loopback": local, "remote": remote}


def local_llm_events(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    args = parser.parse_args()
    session_dir = pathlib.Path(args.session_dir).resolve()
    state_path = session_dir / "state.json"
    state = load_json(state_path)
    gpu = telemetry(session_dir / "gpu.csv")
    network = (
        network_destinations(session_dir)
        if state.get("network_audit")
        else {"verdict": "not_audited", "loopback": [], "remote": []}
    )

    started = datetime.datetime.fromisoformat(state["started_at"])
    finished = datetime.datetime.fromisoformat(
        state.get("finished_at", state["updated_at"])
    )
    duration = (finished - started).total_seconds()
    videos = sorted((session_dir / "videos").glob("*.mp4"))
    previews = sorted(
        path
        for path in (session_dir / "previews").glob("*.mp4")
        if not path.name.startswith("latest_") and path.name != "latest.mp4"
    )
    llm_events = local_llm_events(session_dir / "llm-generations.jsonl")
    inference_events = local_llm_events(session_dir / "inference-audit.jsonl")
    policy_inference_artifacts = sorted(
        (session_dir / "policy-inference").glob("request-*.json")
    )
    preview_events = local_llm_events(session_dir / "preview-audit.jsonl")
    completed_preview_events = [
        event
        for event in preview_events
        if event.get("status") == "completed" or "predicted_matches_actual" in event
    ]
    failed_preview_events = [
        event for event in preview_events if event.get("status") == "failed"
    ]

    summary = {
        "backend": state.get("backend", "libero"),
        "simulator": state.get("simulator", "LIBERO / robosuite / MuJoCo"),
        "model": state.get("model"),
        "runtime": state.get("runtime"),
        "world_model": state.get("world_model", "none"),
        "world_model_runtime": state.get("world_model_runtime", "disabled"),
        "suite": state.get("suite"),
        "interactive": state.get("interactive", False),
        "task_ids": state.get("task_ids"),
        "seed": state.get("seed"),
        "episodes": state.get("episodes"),
        "successes": state.get("successes"),
        "success_rate": state.get("success_rate"),
        "completed_attempts": state.get("completed_attempts", state.get("episodes")),
        "unscored_attempts": state.get("unscored_attempts", 0),
        "aborted_attempts": state.get("aborted_attempts", 0),
        "duration_seconds": round(duration, 2),
        "cold_inference_latency_ms": state.get("cold_inference_latency_ms"),
        "mean_inference_latency_ms": state.get("mean_inference_latency_ms"),
        "warm_mean_inference_latency_ms": state.get("warm_mean_inference_latency_ms"),
        "median_inference_latency_ms": state.get("median_inference_latency_ms"),
        "p95_inference_latency_ms": state.get("p95_inference_latency_ms"),
        "gpu": gpu,
        "network": network,
        "videos": [video.name for video in videos],
        "previews": [preview.name for preview in previews],
        "attempt_history": state.get("attempt_history", []),
        "prompt_stats": state.get("prompt_stats", {}),
        "local_llm_generations": llm_events,
        "inference_audit_events": len(inference_events),
        "policy_inference_artifacts": [
            path.name for path in policy_inference_artifacts
        ],
        "preview_audit_events": len(preview_events),
        "completed_prediction_comparisons": len(completed_preview_events),
        "prediction_failures": len(failed_preview_events),
    }
    (session_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    remote_text = ", ".join(network["remote"]) if network["remote"] else "None observed"
    loopback_text = (
        ", ".join(network["loopback"]) if network["loopback"] else "None observed"
    )
    report = """# Embodied Policy Lab local showcase report

| Measurement | Result |
|---|---:|
| Backend | `{backend}` |
| Simulator | {simulator} |
| Model | `{model}` |
| Runtime | {runtime} |
| Predictor / baseline | `{world_model}` ({world_model_runtime}) |
| Task collection / IDs | `{suite}` / `{task_ids}` |
| Episodes successful | **{successes}/{episodes}** |
| Unscored exploratory / mixed attempts | {unscored_attempts} |
| Aborted attempts | {aborted_attempts} |
| Session duration | {duration:.2f} s |
| Cold/startup inference latency | {cold_latency} ms |
| Warm mean inference latency | {warm_mean_latency} ms |
| Warm median inference latency | {median_latency} ms |
| Warm P95 inference latency | {p95_latency} ms |
| Peak GPU memory | {memory:.0f} MiB |
| Peak GPU utilization | {utilization:.0f}% |
| Peak GPU power | {power:.2f} W |
| Peak GPU temperature | {temperature:.0f} °C |
| Local prompt generations | {llm_generation_count} |
| Audited policy requests | {inference_audit_count} |
| Completed prediction/actual comparisons | {preview_audit_count} |
| Non-gating predictor failures | {preview_failure_count} |

## Local-inference network audit

**Verdict: `{verdict}`**

- Loopback destinations: {loopback}
- Remote IP destinations: {remote}

The audit was produced from `strace -f -e trace=network` logs for both the
policy server and simulator. A `loopback_only` verdict means the traced model
run made no connection to a remote inference service. The simulator communicated
with the local policy server over loopback.

## Artifacts

- `state.json`: final simulator and inference state
- `gpu.csv`: per-second NVIDIA telemetry
- `network-*`: raw network system-call traces
- `client.log` and `server.log`: runtime logs
- `videos/`: rollout MP4 files ({video_count})
- `previews/`: predictor/actual media, including retained discarded predictions ({preview_count})
- `llm-generations.jsonl`: local prompt-generation provenance ({llm_generation_count})
- `inference-audit.jsonl`: prompt and action hashes for synchronous policy requests ({inference_audit_count})
- `policy-inference/`: exact Fast-WAM inputs, actions, and staging metrics ({policy_inference_count})
- `preview-audit.jsonl`: completed comparisons and non-gating predictor failures ({preview_event_count} total events)
""".format(
        backend=summary["backend"],
        simulator=summary["simulator"],
        model=summary["model"],
        runtime=summary["runtime"],
        world_model=summary["world_model"],
        world_model_runtime=summary["world_model_runtime"],
        suite=summary["suite"],
        task_ids=summary["task_ids"],
        successes=summary["successes"],
        episodes=summary["episodes"],
        unscored_attempts=summary["unscored_attempts"],
        aborted_attempts=summary["aborted_attempts"],
        duration=summary["duration_seconds"],
        cold_latency=summary["cold_inference_latency_ms"],
        warm_mean_latency=summary["warm_mean_inference_latency_ms"],
        median_latency=summary["median_inference_latency_ms"],
        p95_latency=summary["p95_inference_latency_ms"],
        memory=gpu["max_memory_mib"],
        utilization=gpu["max_utilization_pct"],
        power=gpu["max_power_w"],
        temperature=gpu["max_temperature_c"],
        llm_generation_count=len(llm_events),
        inference_audit_count=len(inference_events),
        policy_inference_count=len(policy_inference_artifacts),
        preview_audit_count=len(completed_preview_events),
        preview_failure_count=len(failed_preview_events),
        preview_event_count=len(preview_events),
        verdict=network["verdict"],
        loopback=loopback_text,
        remote=remote_text,
        video_count=len(videos),
        preview_count=len(previews),
    )
    (session_dir / "report.md").write_text(report, encoding="utf-8")

    state["network_verdict"] = network["verdict"]
    state["network_loopback_destinations"] = network["loopback"]
    state["network_remote_destinations"] = network["remote"]
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(session_dir / "report.md")


if __name__ == "__main__":
    main()

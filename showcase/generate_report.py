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
    llm_events = local_llm_events(session_dir / "llm-generations.jsonl")

    summary = {
        "model": state.get("model"),
        "runtime": state.get("runtime"),
        "suite": state.get("suite"),
        "interactive": state.get("interactive", False),
        "task_ids": state.get("task_ids"),
        "seed": state.get("seed"),
        "episodes": state.get("episodes"),
        "successes": state.get("successes"),
        "success_rate": state.get("success_rate"),
        "duration_seconds": round(duration, 2),
        "cold_inference_latency_ms": state.get("cold_inference_latency_ms"),
        "mean_inference_latency_ms": state.get("mean_inference_latency_ms"),
        "warm_mean_inference_latency_ms": state.get("warm_mean_inference_latency_ms"),
        "median_inference_latency_ms": state.get("median_inference_latency_ms"),
        "p95_inference_latency_ms": state.get("p95_inference_latency_ms"),
        "gpu": gpu,
        "network": network,
        "videos": [video.name for video in videos],
        "attempt_history": state.get("attempt_history", []),
        "prompt_stats": state.get("prompt_stats", {}),
        "local_llm_generations": llm_events,
    }
    (session_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    remote_text = ", ".join(network["remote"]) if network["remote"] else "None observed"
    loopback_text = (
        ", ".join(network["loopback"]) if network["loopback"] else "None observed"
    )
    report = """# π0.5 local showcase report

| Measurement | Result |
|---|---:|
| Model | `{model}` |
| Runtime | {runtime} |
| LIBERO suite / task IDs | `{suite}` / `{task_ids}` |
| Episodes successful | **{successes}/{episodes}** |
| Session duration | {duration:.2f} s |
| Cold/JIT inference latency | {cold_latency} ms |
| Warm mean inference latency | {warm_mean_latency} ms |
| Warm median inference latency | {median_latency} ms |
| Warm P95 inference latency | {p95_latency} ms |
| Peak GPU memory | {memory:.0f} MiB |
| Peak GPU utilization | {utilization:.0f}% |
| Peak GPU power | {power:.2f} W |
| Peak GPU temperature | {temperature:.0f} °C |
| Local prompt generations | {llm_generation_count} |

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
- `llm-generations.jsonl`: local prompt-generation provenance ({llm_generation_count})
""".format(
        model=summary["model"],
        runtime=summary["runtime"],
        suite=summary["suite"],
        task_ids=summary["task_ids"],
        successes=summary["successes"],
        episodes=summary["episodes"],
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
        verdict=network["verdict"],
        loopback=loopback_text,
        remote=remote_text,
        video_count=len(videos),
    )
    (session_dir / "report.md").write_text(report, encoding="utf-8")

    state["network_verdict"] = network["verdict"]
    state["network_loopback_destinations"] = network["loopback"]
    state["network_remote_destinations"] = network["remote"]
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(session_dir / "report.md")


if __name__ == "__main__":
    main()

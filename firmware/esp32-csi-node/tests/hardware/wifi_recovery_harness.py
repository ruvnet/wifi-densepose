#!/usr/bin/env python3
"""Evidence-gated ESP32 Wi-Fi recovery checks.

This harness observes RuView and an optional AP status endpoint. It never
changes AP, helper, USB, serial, or firmware state; the operator performs the
named fault between phase captures. A passing recovery requires monotonic node
uptime so a USB/power reset cannot masquerade as automatic recovery.
"""

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


SCENARIOS = {"ap-outage", "ap-reboot-channel", "helper-restart", "stale-active"}
PHASES = ("before", "during", "after")
MAX_HTTP_BYTES = 512 * 1024
MAX_SESSION_BYTES = 2 * 1024 * 1024


def _nodes_by_id(capture):
    return {
        int(item["node_id"]): item
        for item in capture.get("nodes", [])
        if isinstance(item, dict) and isinstance(item.get("node_id"), int)
    }


def evaluate_scenario(scenario, evidence, *, expected_nodes, minimum_outage_seconds=90, expected_channel=None):
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    failures = []
    for phase in PHASES:
        if phase not in evidence:
            failures.append(f"missing {phase} capture")
    if failures:
        return {"passed": False, "scenario": scenario, "failures": failures}

    before = evidence["before"]
    during = evidence["during"]
    after = evidence["after"]
    before_nodes = _nodes_by_id(before)
    during_nodes = _nodes_by_id(during)
    after_nodes = _nodes_by_id(after)

    for node_id in sorted(expected_nodes):
        before_node = before_nodes.get(node_id)
        after_node = after_nodes.get(node_id)
        if not before_node or before_node.get("status") != "active":
            failures.append(f"node {node_id} was not active before fault")
            continue
        if not after_node or after_node.get("status") != "active":
            failures.append(f"node {node_id} did not recover active")
            continue
        before_health = before_node.get("health") or {}
        after_health = after_node.get("health") or {}
        if before_health.get("extended") is not True or after_health.get("extended") is not True:
            failures.append(f"node {node_id} lacks extended HEALTH evidence")
            continue
        before_uptime = before_health.get("uptime_ms")
        after_uptime = after_health.get("uptime_ms")
        if not isinstance(before_uptime, int) or not isinstance(after_uptime, int):
            failures.append(f"node {node_id} uptime is unavailable")
        elif after_uptime < before_uptime:
            failures.append(f"node {node_id} uptime moved backwards")

        before_epoch = before_health.get("association_epoch")
        after_epoch = after_health.get("association_epoch")
        if scenario == "helper-restart" and after_epoch != before_epoch:
            failures.append(f"node {node_id} association changed during helper restart")
        if scenario in {"ap-outage", "ap-reboot-channel"} and (
            not isinstance(before_epoch, int) or not isinstance(after_epoch, int) or after_epoch <= before_epoch
        ):
            failures.append(f"node {node_id} did not report a new association epoch")
        if expected_channel is not None and after_health.get("channel") != expected_channel:
            failures.append(f"node {node_id} did not recover on channel {expected_channel}")

    if scenario != "helper-restart":
        for node_id in sorted(expected_nodes):
            if during_nodes.get(node_id, {}).get("status") != "stale":
                failures.append(f"node {node_id} was not observed stale during fault")
    elif during.get("server_reachable") is not False:
        failures.append("helper was not observed unavailable during restart")

    if scenario == "ap-outage":
        elapsed_ms = during.get("captured_at_unix_ms", 0) - before.get("captured_at_unix_ms", 0)
        if elapsed_ms < minimum_outage_seconds * 1_000:
            failures.append(f"AP outage evidence is shorter than {minimum_outage_seconds} seconds")

    if scenario == "ap-reboot-channel":
        before_ap = before.get("ap_uptime_seconds")
        after_ap = after.get("ap_uptime_seconds")
        if not isinstance(before_ap, int) or not isinstance(after_ap, int) or after_ap >= before_ap:
            failures.append("AP uptime did not prove a reboot")
        if expected_channel is None:
            failures.append("expected channel is required for AP reboot/channel test")

    return {"passed": not failures, "scenario": scenario, "failures": failures}


def _read_json_url(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_HTTP_BYTES:
            raise ValueError("HTTP response exceeds evidence cap")
        data = response.read(MAX_HTTP_BYTES + 1)
    if len(data) > MAX_HTTP_BYTES:
        raise ValueError("HTTP response exceeds evidence cap")
    return json.loads(data)


def capture_phase(server_url, ap_info_url=None):
    capture = {
        "captured_at_unix_ms": int(time.time() * 1_000),
        "server_reachable": False,
        "nodes": [],
        "ap_uptime_seconds": None,
    }
    try:
        payload = _read_json_url(f"{server_url.rstrip('/')}/api/v1/nodes")
        nodes = payload.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) > 256:
            raise ValueError("invalid node inventory")
        capture["nodes"] = nodes
        capture["server_reachable"] = True
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
        pass
    if ap_info_url:
        try:
            payload = _read_json_url(ap_info_url)
            uptime = payload.get("data", {}).get("uptime")
            if isinstance(uptime, int) and uptime >= 0:
                capture["ap_uptime_seconds"] = uptime
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
            pass
    return capture


def _load_session(session_path):
    if not session_path.exists():
        return {"schema": "ruview.hardware.wifi-recovery.v1", "captures": {}}
    if session_path.stat().st_size > MAX_SESSION_BYTES:
        raise ValueError("session evidence exceeds 2 MiB")
    value = json.loads(session_path.read_text(encoding="utf-8"))
    if value.get("schema") != "ruview.hardware.wifi-recovery.v1" or not isinstance(value.get("captures"), dict):
        raise ValueError("invalid session evidence schema")
    return value


def _write_session(session_path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(data) > MAX_SESSION_BYTES:
        raise ValueError("session evidence exceeds 2 MiB")
    session_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".wifi-recovery-", dir=str(session_path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, session_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--phase", choices=PHASES, required=True)
    capture_parser.add_argument("--session", type=Path, required=True)
    capture_parser.add_argument("--server-url", default="http://127.0.0.1:3000")
    capture_parser.add_argument("--ap-info-url")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    verify_parser.add_argument("--session", type=Path, required=True)
    verify_parser.add_argument("--nodes", required=True, help="comma-separated node IDs")
    verify_parser.add_argument("--minimum-outage-seconds", type=int, default=90)
    verify_parser.add_argument("--expected-channel", type=int)
    arguments = parser.parse_args()

    if arguments.command == "capture":
        session = _load_session(arguments.session)
        session["captures"][arguments.phase] = capture_phase(arguments.server_url, arguments.ap_info_url)
        _write_session(arguments.session, session)
        print(json.dumps(session["captures"][arguments.phase], sort_keys=True))
        return

    expected_nodes = {int(value) for value in arguments.nodes.split(",") if value.strip()}
    if not expected_nodes or any(node_id < 1 or node_id > 255 for node_id in expected_nodes):
        parser.error("--nodes must contain IDs in 1..255")
    session = _load_session(arguments.session)
    result = evaluate_scenario(
        arguments.scenario,
        session["captures"],
        expected_nodes=expected_nodes,
        minimum_outage_seconds=arguments.minimum_outage_seconds,
        expected_channel=arguments.expected_channel,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

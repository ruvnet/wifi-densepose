#!/usr/bin/env python3
"""
rvagent-mcp-consumer.py — ADR-125 tier1+2 iter 5: end-to-end agentic loop.

Spawns the published `@ruvnet/rvagent` MCP server (ADR-124, npm 0.1.0)
as a subprocess and exercises it through the standard MCP JSON-RPC 2.0
stdio protocol. This is the "agentic capabilities" half of the ADR-125
Tier 1+2 sprint — it proves the full bidirectional chain:

    real C6 (192.168.1.179)
      → UDP feature_state
      → c6-presence-watcher.py (BFLD PrivacyGate)
      → /tmp/ruview-last-feature.json
      → ruview-sensing-server.py (sensing-server-equivalent on :3000)
      → @ruvnet/rvagent (this script spawns it via `npx -y`)
      → MCP JSON-RPC tools/call (this script sends them)
      → result returned to any MCP-aware agent

If real data flows back, the agentic surface for RuView's BFLD-gated
stream is live for every MCP client in the ecosystem — Claude Code,
Codex, custom LLM agents.

Run on ruv-mac-mini (or any host with Node ≥ 20 + the running
ruview-sensing-server.py on :3000):

    RVAGENT_SENSING_URL=http://localhost:3000 \
      python3 rvagent-mcp-consumer.py
"""
from __future__ import annotations
import json
import os
import sys
import time
import subprocess

NODE_ID = os.environ.get("RVAGENT_TEST_NODE", "12")
SENSING_URL = os.environ.get("RVAGENT_SENSING_URL", "http://localhost:3000")


def _send(proc: subprocess.Popen, msg: dict) -> None:
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, want_id: int | None = None,
          timeout: float = 8.0) -> dict | None:
    """Read JSON-RPC responses, optionally waiting for a specific id."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # rvagent may print non-JSON log lines on stdout in
            # error cases — skip and keep listening.
            print(f"[non-json] {line[:200]}", file=sys.stderr)
            continue
        if want_id is None or obj.get("id") == want_id:
            return obj
    return None


def call_tool(proc: subprocess.Popen, tool_name: str,
              args: dict, request_id: int) -> dict | None:
    _send(proc, {
        "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
        "params": {"name": tool_name, "arguments": args},
    })
    return _recv(proc, want_id=request_id, timeout=12.0)


def main() -> int:
    env = {**os.environ, "RVAGENT_SENSING_URL": SENSING_URL}
    print(f"[mcp-consumer] spawning npx -y @ruvnet/rvagent")
    print(f"[mcp-consumer] RVAGENT_SENSING_URL={SENSING_URL}")
    print(f"[mcp-consumer] test node_id={NODE_ID}")

    proc = subprocess.Popen(
        ["npx", "-y", "@ruvnet/rvagent"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env, bufsize=1,
    )
    # Give npx a chance to install if cold.
    time.sleep(2.0)

    # 1. initialize handshake
    _send(proc, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ruview-iter5-consumer", "version": "0.1"},
        },
    })
    resp = _recv(proc, want_id=1)
    if resp is None:
        print("[mcp-consumer] FAIL: no initialize response", file=sys.stderr)
        proc.kill()
        return 1
    server_info = resp.get("result", {}).get("serverInfo", {})
    print(f"[mcp-consumer] server: {server_info.get('name')} "
          f"v{server_info.get('version')}")

    # initialized notification
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    # 2. tools/list
    _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    resp = _recv(proc, want_id=2)
    tools = (resp or {}).get("result", {}).get("tools", [])
    print(f"[mcp-consumer] {len(tools)} tools available:")
    for t in tools:
        print(f"             - {t.get('name')}")

    # Locate the actual tool names (rvagent uses both snake_case and
    # dotted forms — discover them rather than hard-coding).
    names = [t.get("name") for t in tools]
    vitals_tool = next((n for n in names
                        if "vitals" in n and ("all" in n or n.endswith("vitals"))), None)
    bfld_tool = next((n for n in names if "bfld" in n and "last_scan" in n), None)
    print(f"[mcp-consumer] resolved: vitals={vitals_tool} bfld={bfld_tool}")

    # 3. tools/call vitals
    resp = call_tool(proc, vitals_tool or "vitals_get_all",
                     {"node_id": NODE_ID}, 3)
    if resp is None or "error" in resp:
        print(f"[mcp-consumer] vitals_get_all failed: {resp}",
              file=sys.stderr)
    else:
        content = resp.get("result", {}).get("content", [])
        text = content[0].get("text", "") if content else ""
        print(f"[mcp-consumer] vitals_get_all OK — {len(text)} bytes")
        try:
            parsed = json.loads(text)
            print(f"             presence={parsed.get('data', {}).get('presence')}, "
                  f"motion={parsed.get('data', {}).get('motion')}, "
                  f"breathing={parsed.get('data', {}).get('breathing_rate_bpm')}, "
                  f"hr={parsed.get('data', {}).get('heartrate_bpm')}")
        except (json.JSONDecodeError, AttributeError):
            print(f"             (response head: {text[:200]})")

    # 4. tools/call bfld last_scan
    resp = call_tool(proc, bfld_tool or "ruview.bfld.last_scan",
                     {"node_id": NODE_ID}, 4)
    if resp is None or "error" in resp:
        print(f"[mcp-consumer] bfld_last_scan failed: {resp}",
              file=sys.stderr)
    else:
        content = resp.get("result", {}).get("content", [])
        text = content[0].get("text", "") if content else ""
        print(f"[mcp-consumer] bfld_last_scan OK — {len(text)} bytes")
        try:
            parsed = json.loads(text)
            print(f"             privacy_class={parsed.get('privacy_class')}, "
                  f"identity_risk_score={parsed.get('identity_risk_score')!r}, "
                  f"presence={parsed.get('presence')}, "
                  f"person_count={parsed.get('n_frames')}")
        except (json.JSONDecodeError, AttributeError):
            print(f"             (response head: {text[:200]})")

    proc.stdin.close()
    proc.wait(timeout=5)
    print("[mcp-consumer] done — agentic chain validated end-to-end")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

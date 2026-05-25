#!/usr/bin/env python3
"""
ruview-sensing-server.py — ADR-125 Tier 1+2 iter 2.

A tiny HTTP server that speaks the subset of the RuView sensing-server
HTTP API that @ruvnet/rvagent (ADR-124, npm v0.1.0) expects, sourced
from the BFLD-gated state files written by c6-presence-watcher.py.

This is the "sensing-server-equivalent" the cron stop condition names,
and it lets any MCP agent (Claude Code via `claude mcp add rvagent`,
Codex with the matching MCP config, custom LLM client) consume the
real ESP32-C6 stream through the same MCP tool surface that the Rust
sensing-server exposes — without needing the Rust binary to be running.

Endpoints (matched against tools/ruview-mcp/src/tools/*.ts):

  GET  /health                                 — liveness
  GET  /api/v1/sensing/latest                  — ADR-102 schema v2
  GET  /api/v1/edge/registry                   — node enumeration
  GET  /api/v1/vitals/<node_id>/latest         — EdgeVitalsMessage
  GET  /api/v1/bfld/<node_id>/last_scan        — BfldScanResponse
  POST /api/v1/bfld/<node_id>/subscribe?duration_s=N — { subscription_id }

The source-of-truth file is `/tmp/ruview-last-feature.json` written
by the watcher on every BFLD-gated feature_state packet. If absent
or stale (> STALENESS_S seconds old), endpoints return 503 with a
hint so the rvagent tool emits a graceful warn shape.

Bearer-token auth is intentionally OFF in this dev surface — the
Rust sensing-server adds it via the #443 middleware; that path is
out of scope for the demo bridge.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

FEATURE_FILE = os.environ.get("RUVIEW_FEATURE_JSON",
                              "/tmp/ruview-last-feature.json")
STALENESS_S = 10.0
DEFAULT_PORT = int(os.environ.get("PORT", "3000"))


def _load_feature() -> dict | None:
    try:
        with open(FEATURE_FILE, "r") as fh:
            d = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(d, dict):
        return None
    age = time.time() - float(d.get("ts", 0))
    if age > STALENESS_S:
        return None
    return d


def vitals_for(node_id: str) -> dict | None:
    f = _load_feature()
    if f is None or f.get("node_id") != node_id:
        return None
    return {
        "node_id": f["node_id"],
        "timestamp_ms": int(f.get("timestamp_ms",
                                  int(time.time() * 1000))),
        "presence": bool(f.get("presence", False)),
        "n_persons": int(f.get("n_persons", 0)),
        "confidence": float(f.get("confidence", 0.0)),
        "breathing_rate_bpm": f.get("breathing_rate_bpm"),
        "heartrate_bpm": f.get("heartrate_bpm"),
        "motion": float(f.get("motion", 0.0)),
    }


def bfld_scan_for(node_id: str) -> dict | None:
    f = _load_feature()
    if f is None or f.get("node_id") != node_id:
        return None
    # ADR-125 §2.1.d: identity_risk_score never crosses the HAP
    # boundary. We mirror that here — even though rvagent's schema
    # has a nullable identity_risk_score slot, we deliberately
    # always return None for it on this bridge.
    return {
        "node_id": f["node_id"],
        "identity_risk_score": None,        # ADR-125 §2.1.d invariant
        "privacy_class": int(f.get("privacy_class", 2)),
        "person_count": int(f.get("n_persons", 0)),
        "confidence": float(f.get("confidence", 0.0)),
        "presence": bool(f.get("presence", False)),
        # timestamp_ns matches BFLD wire format (BfldEvent.timestamp_ns)
        "timestamp_ns": int(f.get("ts", time.time()) * 1_000_000_000),
    }


_PATH_VITALS = re.compile(r"^/api/v1/vitals/([^/]+)/latest$")
_PATH_BFLD_SCAN = re.compile(r"^/api/v1/bfld/([^/]+)/last_scan$")
_PATH_BFLD_SUBSCRIBE = re.compile(r"^/api/v1/bfld/([^/]+)/subscribe$")
_PATH_SEMANTIC = re.compile(r"^/api/v1/semantic-events/([^/]+)/latest$")


def semantic_events_for(node_id: str) -> dict | None:
    """ADR-125 §2.1.d semantic-event surface.

    The three named events that cross the HAP boundary. Each one is a
    boolean + last-fire timestamp. Agents subscribe to this endpoint
    rather than reasoning over raw scores — the naming is the contract.
    """
    f = _load_feature()
    if f is None or f.get("node_id") != node_id:
        return None
    presence = bool(f.get("presence", False))
    anomaly = float(f.get("anomaly_score") or 0.0)
    return {
        "node_id": f["node_id"],
        "privacy_class": int(f.get("privacy_class", 2)),
        "events": {
            "unknown_presence": {
                "active": presence,
                "source": "BFLD presence_score (rolling 3s avg ≥ 0.30)",
                "ts": f["ts"],
            },
            "unexpected_occupancy": {
                # Placeholder: schedule-aware gating is future work.
                # For now we surface raw occupancy and mark the gate
                # as `schedule_aware=False` so agents know not to
                # equate this with the full §2.1.d intent yet.
                "active": presence,
                "schedule_aware": False,
                "ts": f["ts"],
            },
            "unrecognized_activity_pattern": {
                "active": anomaly >= 0.7,
                "anomaly_threshold": 0.7,
                "anomaly_score": anomaly,
                "ts": f["ts"],
            },
        },
        # ADR-125 §2.1.d invariant restated at the HTTP boundary:
        # identity_risk_score, soul_match_probability, and rf_signature_hash
        # are NEVER published from this endpoint.
        "redacted_fields": [
            "identity_risk_score",
            "soul_match_probability",
            "rf_signature_hash",
        ],
    }


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt: str, *args) -> None:
        # Quiet the default per-request log; print on a single line.
        sys.stdout.write(
            f"[{self.log_date_time_string()}] {self.command} "
            f"{self.path} -> {args[1] if len(args) > 1 else '?'}\n"
        )

    def _json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            f = _load_feature()
            self._json(200, {
                "ok": True,
                "feature_age_s": (None if f is None
                                  else round(time.time() - f["ts"], 2)),
                "source": FEATURE_FILE,
            })
            return

        if path == "/api/v1/edge/registry":
            f = _load_feature()
            nodes = ([{"node_id": f["node_id"], "kind": "esp32-c6",
                       "online": True}] if f else [])
            self._json(200, {"nodes": nodes})
            return

        if path == "/api/v1/sensing/latest":
            f = _load_feature()
            if f is None:
                self._json(503, {"error": "no recent feature_state",
                                 "hint": "is c6-presence-watcher running?"})
                return
            # ADR-102 sensing/latest schema v2 — the rvagent
            # csi-latest tool ingests this shape.
            self._json(200, {
                "schema_version": 2,
                "node_id": f["node_id"],
                "timestamp_ms": f["timestamp_ms"],
                "presence": f["presence"],
                "n_persons": f["n_persons"],
                "confidence": f["confidence"],
                "motion": f["motion"],
                "breathing_rate_bpm": f.get("breathing_rate_bpm"),
                "heartrate_bpm": f.get("heartrate_bpm"),
                "privacy_class": f.get("privacy_class", 2),
            })
            return

        m = _PATH_VITALS.match(path)
        if m:
            node_id = m.group(1)
            v = vitals_for(node_id)
            if v is None:
                self._json(503, {"error": f"no recent vitals for {node_id}",
                                 "hint": "watcher running? node_id correct?"})
                return
            self._json(200, v)
            return

        m = _PATH_BFLD_SCAN.match(path)
        if m:
            node_id = m.group(1)
            r = bfld_scan_for(node_id)
            if r is None:
                self._json(503, {"error": f"no recent BFLD scan for {node_id}",
                                 "hint": "watcher running? node_id correct?"})
                return
            self._json(200, r)
            return

        m = _PATH_SEMANTIC.match(path)
        if m:
            node_id = m.group(1)
            r = semantic_events_for(node_id)
            if r is None:
                self._json(503, {"error": f"no recent semantic events for {node_id}",
                                 "hint": "watcher running? node_id correct?"})
                return
            self._json(200, r)
            return

        self._json(404, {"error": "not found", "path": path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        m = _PATH_BFLD_SUBSCRIBE.match(parsed.path)
        if m:
            qs = parse_qs(parsed.query)
            duration_s = float(qs.get("duration_s", ["10"])[0])
            sub_id = f"sub-{int(time.time() * 1000)}-{m.group(1)}"
            self._json(200, {
                "subscription_id": sub_id,
                "node_id": m.group(1),
                "duration_s": duration_s,
                "endpoint_hint": (f"poll GET /api/v1/bfld/{m.group(1)}"
                                  "/last_scan every 1 s for the window"),
            })
            return
        self._json(404, {"error": "not found", "path": parsed.path})


def main() -> int:
    port = DEFAULT_PORT
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[sensing-server] listening on 0.0.0.0:{port}", flush=True)
    print(f"[sensing-server] feature source: {FEATURE_FILE}", flush=True)
    print(f"[sensing-server] staleness limit: {STALENESS_S} s", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

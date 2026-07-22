"""ADR-117 P4 — End-to-end test for SensingClient against an in-process
WS server.

We spin up a real `websockets.serve()` server in the same event loop,
send the four message types defined in ADR-115 §1, and assert the
client decodes them into the right dataclasses. No mocks — the only
moving part this test does NOT exercise is the actual sensing-server
binary, but the wire protocol is the contract under test here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import websockets

from wifi_densepose.client import (
    ConnectionEstablishedMessage,
    EdgeVitalsMessage,
    PoseDataMessage,
    SensingClient,
    SensingMessage,
)


# ─── In-process WS server fixture ────────────────────────────────────


_FIXTURE_MESSAGES = [
    {
        "type": "connection_established",
        "node_id": "test-node-001",
        "version": "0.7.4",
        "capabilities": ["edge_vitals", "pose_data"],
    },
    {
        "type": "edge_vitals",
        "node_id": "test-node-001",
        "presence": True,
        "fall_detected": False,
        "motion": 0.21,
        "breathing_rate_bpm": 14.5,
        "heartrate_bpm": 72.3,
        "n_persons": 1,
        "motion_energy": 0.034,
        "presence_score": 0.91,
        "rssi": -42.0,
    },
    {
        "type": "pose_data",
        "node_id": "test-node-001",
        "timestamp": 1700000000.5,
        "persons": [{"id": 1, "keypoints": []}],
        "confidence": 0.88,
    },
    # Unknown type — should NOT crash the stream; should yield a plain
    # SensingMessage.
    {
        "type": "future_message_type_not_yet_modelled",
        "extra": "data",
    },
]


def _request_headers(websocket: Any) -> Any:
    """Return the upgrade-request headers regardless of websockets version:
    the legacy server protocol (<14, still reachable via >=12.0's floor)
    exposes `request_headers`; the newer asyncio-based server exposes
    `request.headers` instead."""
    legacy = getattr(websocket, "request_headers", None)
    return legacy if legacy is not None else websocket.request.headers


async def _handler(websocket: Any) -> None:
    for msg in _FIXTURE_MESSAGES:
        await websocket.send(json.dumps(msg))
    # Send one malformed frame to assert the client logs+drops it
    # rather than crashing the stream.
    await websocket.send("{not valid json")
    # And one final "real" message so the test can confirm the stream
    # survived the malformed one.
    await websocket.send(json.dumps({"type": "edge_vitals", "node_id": "post-bad-frame"}))


@pytest.fixture
async def ws_server() -> Any:
    """Start a websocket server on a random port; yield the bound URL."""
    server = await websockets.serve(_handler, "127.0.0.1", 0)
    # Get the bound port (host="127.0.0.1" returns one socket).
    port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        yield f"ws://127.0.0.1:{port}/ws/sensing"
    finally:
        server.close()
        await server.wait_closed()


# ─── End-to-end stream test ──────────────────────────────────────────


async def test_sensing_client_decodes_all_message_types(ws_server: str) -> None:
    received: list[SensingMessage] = []
    async with SensingClient(ws_server) as client:
        async for msg in client.stream():
            received.append(msg)
            if len(received) >= len(_FIXTURE_MESSAGES) + 1:  # +1 for post-bad-frame
                break

    # connection_established → typed
    assert isinstance(received[0], ConnectionEstablishedMessage)
    assert received[0].node_id == "test-node-001"
    assert received[0].version == "0.7.4"
    assert "edge_vitals" in received[0].capabilities

    # edge_vitals → typed with full fields
    assert isinstance(received[1], EdgeVitalsMessage)
    assert received[1].presence is True
    assert received[1].fall_detected is False
    assert received[1].breathing_rate_bpm == 14.5
    assert received[1].heartrate_bpm == 72.3
    assert received[1].n_persons == 1
    assert received[1].rssi == -42.0

    # pose_data → typed
    assert isinstance(received[2], PoseDataMessage)
    assert received[2].timestamp == 1700000000.5
    assert len(received[2].persons) == 1
    assert received[2].confidence == 0.88

    # Unknown type → plain SensingMessage (forward-compat)
    assert type(received[3]) is SensingMessage  # exact base class
    assert received[3].type == "future_message_type_not_yet_modelled"
    assert received[3].raw["extra"] == "data"

    # After the malformed frame: the stream should have survived and
    # yielded the post-bad-frame message.
    assert isinstance(received[4], EdgeVitalsMessage)
    assert received[4].node_id == "post-bad-frame"


async def test_sensing_client_recv_one(ws_server: str) -> None:
    async with SensingClient(ws_server) as client:
        msg = await client.recv_one(timeout=2.0)
    assert isinstance(msg, ConnectionEstablishedMessage)


async def test_sensing_client_raises_when_used_without_context() -> None:
    client = SensingClient("ws://127.0.0.1:1/")  # never connects
    with pytest.raises(RuntimeError, match="not connected"):
        await client.recv_one(timeout=0.1)
    with pytest.raises(RuntimeError, match="not connected"):
        async for _ in client.stream():
            pass


async def test_sensing_client_close_is_idempotent(ws_server: str) -> None:
    client = SensingClient(ws_server)
    await client.__aenter__()
    await client.close()
    await client.close()  # second close is a no-op


def test_sensing_client_decoder_directly() -> None:
    """The decoder is pure — exercise it without bringing up a WS
    server, so we have a fast unit test for the type mapping."""
    from wifi_densepose.client.ws import _decode

    msg = _decode(json.dumps({
        "type": "edge_vitals",
        "node_id": "x",
        "presence": True,
        "fall_detected": False,
        "motion": 1.5,
    }))
    assert isinstance(msg, EdgeVitalsMessage)
    assert msg.presence is True
    assert msg.motion == 1.5
    assert msg.breathing_rate_bpm is None  # not present → None, not 0.0
    assert msg.heartrate_bpm is None
    assert msg.rssi is None


async def test_sensing_client_sends_bearer_token_on_upgrade() -> None:
    """#1395: a token passed to the constructor must reach the server as
    an `Authorization: Bearer <token>` header on the WS upgrade request."""
    captured_auth: dict[str, Any] = {}

    async def handler(websocket: Any) -> None:
        captured_auth["value"] = _request_headers(websocket).get("Authorization")
        await websocket.send(json.dumps({"type": "connection_established", "node_id": "auth-test"}))
        await websocket.wait_closed()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        async with SensingClient(f"ws://127.0.0.1:{port}/ws/sensing", token="s3cr3t") as client:
            await client.recv_one(timeout=2.0)
    finally:
        server.close()
        await server.wait_closed()

    assert captured_auth["value"] == "Bearer s3cr3t"


async def test_sensing_client_omits_auth_header_when_no_token(monkeypatch: Any) -> None:
    """No token passed and RUVIEW_API_TOKEN unset: no Authorization header
    at all, not an empty one — matches the HTTP client's no-op behaviour."""
    monkeypatch.delenv("RUVIEW_API_TOKEN", raising=False)
    captured_auth: dict[str, Any] = {}

    async def handler(websocket: Any) -> None:
        captured_auth["present"] = "Authorization" in _request_headers(websocket)
        await websocket.send(
            json.dumps({"type": "connection_established", "node_id": "no-auth-test"})
        )
        await websocket.wait_closed()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        async with SensingClient(f"ws://127.0.0.1:{port}/ws/sensing") as client:
            await client.recv_one(timeout=2.0)
    finally:
        server.close()
        await server.wait_closed()

    assert captured_auth["present"] is False


async def test_sensing_client_reads_token_from_env(monkeypatch: Any) -> None:
    """Constructor token defaults to RUVIEW_API_TOKEN when not passed explicitly."""
    monkeypatch.setenv("RUVIEW_API_TOKEN", "from-env-token")
    captured_auth: dict[str, Any] = {}

    async def handler(websocket: Any) -> None:
        captured_auth["value"] = _request_headers(websocket).get("Authorization")
        await websocket.send(
            json.dumps({"type": "connection_established", "node_id": "env-token-test"})
        )
        await websocket.wait_closed()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        async with SensingClient(f"ws://127.0.0.1:{port}/ws/sensing") as client:
            await client.recv_one(timeout=2.0)
    finally:
        server.close()
        await server.wait_closed()

    assert captured_auth["value"] == "Bearer from-env-token"


def test_sensing_client_decoder_handles_None_subfields() -> None:
    """When the sensing-server explicitly emits null for HR/BR (no
    measurement yet), the client should propagate None, not crash."""
    from wifi_densepose.client.ws import _decode

    msg = _decode(json.dumps({
        "type": "edge_vitals",
        "node_id": "x",
        "presence": False,
        "fall_detected": False,
        "motion": 0.0,
        "breathing_rate_bpm": None,
        "heartrate_bpm": None,
        "rssi": None,
    }))
    assert isinstance(msg, EdgeVitalsMessage)
    assert msg.breathing_rate_bpm is None
    assert msg.heartrate_bpm is None
    assert msg.rssi is None

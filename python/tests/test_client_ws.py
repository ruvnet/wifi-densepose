"""ADR-117 P4 — End-to-end test for SensingClient against an in-process
WS server.

We spin up a real `websockets.serve()` server in the same event loop,
send the four message types defined in ADR-115 §1, and assert the
client decodes them into the right dataclasses. No mocks — the only
moving part this test does NOT exercise is the actual sensing-server
binary, but the wire protocol is the contract under test here.
"""

from __future__ import annotations

import inspect
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


async def _handler(websocket: Any) -> None:
    for msg in _FIXTURE_MESSAGES:
        await websocket.send(json.dumps(msg))
    # Send one malformed frame to assert the client logs+drops it
    # rather than crashing the stream.
    await websocket.send("{not valid json")
    # And one final "real" message so the test can confirm the stream
    # survived the malformed one.
    await websocket.send(json.dumps({"type": "edge_vitals", "node_id": "post-bad-frame"}))


async def _auth_handler(websocket: Any) -> None:
    request = getattr(websocket, "request", None)
    headers = request.headers if request is not None else websocket.request_headers
    if headers.get("Authorization") != "Bearer test-token":
        await websocket.close(code=1008, reason="missing or invalid bearer token")
        return
    await websocket.send(json.dumps(_FIXTURE_MESSAGES[0]))


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


@pytest.fixture
async def authenticated_ws_server() -> Any:
    """Start a WS server that rejects clients without the expected bearer."""
    server = await websockets.serve(_auth_handler, "127.0.0.1", 0)
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


@pytest.mark.parametrize("explicit_token", [False, True])
async def test_sensing_client_authenticates_with_bearer_token(
    authenticated_ws_server: str,
    monkeypatch: pytest.MonkeyPatch,
    explicit_token: bool,
) -> None:
    if explicit_token:
        monkeypatch.setenv("RUVIEW_API_TOKEN", "wrong-environment-token")
        client = SensingClient(authenticated_ws_server, token="test-token")
    else:
        monkeypatch.setenv("RUVIEW_API_TOKEN", "test-token")
        client = SensingClient(authenticated_ws_server)

    async with client:
        msg = await client.recv_one(timeout=2.0)

    assert isinstance(msg, ConnectionEstablishedMessage)


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


@pytest.mark.parametrize("header_parameter", ["extra_headers", "additional_headers"])
def test_authorization_header_supports_websockets_versions(
    monkeypatch: pytest.MonkeyPatch, header_parameter: str
) -> None:
    """websockets 14 renamed extra_headers to additional_headers."""
    from wifi_densepose.client import ws as ws_module

    if header_parameter == "additional_headers":

        async def connect_with_additional_headers(
            *, additional_headers: Any = None
        ) -> None:
            pass

        connect = connect_with_additional_headers
    else:

        async def connect_with_extra_headers(*, extra_headers: Any = None) -> None:
            pass

        connect = connect_with_extra_headers

    monkeypatch.setattr(ws_module.websockets, "connect", connect)
    kwargs = ws_module._authorization_header_kwargs("configured-token")

    assert set(inspect.signature(connect).parameters) == {header_parameter}
    assert kwargs == {header_parameter: {"Authorization": "Bearer configured-token"}}


def test_authorization_header_is_omitted_without_token() -> None:
    from wifi_densepose.client.ws import _authorization_header_kwargs

    assert _authorization_header_kwargs(None) == {}
    assert _authorization_header_kwargs("") == {}

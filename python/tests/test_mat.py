"""ADR-185 P3 — MAT binding tests, incl. the §4.1 bit-for-bit parity gate.

The parity test drives the same committed CSI stream through the binding's
DisasterResponse pipeline and asserts the survivor count + triage classes
(as a SHA-256 of a canonical string) match the native-Rust golden. A
mismatch is a release blocker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wifi_densepose import mat

GOLDEN = Path(__file__).parent / "golden"


def fixture() -> dict:
    return json.loads((GOLDEN / "mat_input.json").read_text())


def build_response() -> mat.DisasterResponse:
    cfg = mat.DisasterConfig(
        mat.DisasterType.Earthquake,
        sensitivity=0.9,
        confidence_threshold=0.1,
        max_depth=5.0,
    )
    resp = mat.DisasterResponse(cfg)
    resp.initialize_event(0.0, 0.0, "parity-fixture")
    resp.add_zone(mat.ScanZone.rectangle("Zone A", 0.0, 0.0, 50.0, 30.0))
    return resp


def run_scan(resp: mat.DisasterResponse) -> None:
    for frame in fixture()["stream"]:
        resp.push_csi_data(frame["amplitude"], frame["phase"])
    resp.scan_once()


# ─── enums / config ──────────────────────────────────────────────────

def test_triage_priority_order() -> None:
    assert mat.TriageStatus.Immediate.priority == 1
    assert mat.TriageStatus.Delayed.priority == 2
    assert mat.TriageStatus.Unknown.priority == 5


def test_disaster_config_fields() -> None:
    cfg = mat.DisasterConfig(mat.DisasterType.Flood, sensitivity=1.5, confidence_threshold=0.3)
    assert cfg.sensitivity == 1.0  # clamped to [0, 1]
    assert abs(cfg.confidence_threshold - 0.3) < 1e-9


# ─── pipeline behaviour ──────────────────────────────────────────────

def test_scan_requires_event() -> None:
    resp = mat.DisasterResponse(mat.DisasterConfig(mat.DisasterType.Unknown))
    # No initialize_event / add_zone -> scan_cycle errors "No active event".
    with pytest.raises(ValueError):
        resp.scan_once()


def test_push_csi_rejects_mismatched_lengths() -> None:
    resp = build_response()
    with pytest.raises(ValueError):
        resp.push_csi_data([1.0, 2.0], [1.0])


def test_scan_detects_survivor_from_breathing_stream() -> None:
    resp = build_response()
    run_scan(resp)
    survivors = resp.survivors()
    # The synthetic breathing-modulated stream trips one detection (matches
    # the native-Rust reference).
    assert len(survivors) == 1
    s = survivors[0]
    assert isinstance(s.id, str) and len(s.id) > 0
    assert s.triage_status == mat.TriageStatus.Delayed
    assert 0.0 <= s.confidence <= 1.0
    # survivors_by_triage is consistent with the survivor's own class.
    assert len(resp.survivors_by_triage(mat.TriageStatus.Delayed)) == 1
    assert len(resp.survivors_by_triage(mat.TriageStatus.Immediate)) == 0


# ─── §4.1 bit-for-bit parity gate (release-blocking) ─────────────────

def test_bit_for_bit_parity_with_native_rust() -> None:
    resp = build_response()
    run_scan(resp)
    survivors = resp.survivors()
    priorities = sorted(s.triage_status.priority for s in survivors)
    canon = f"count={len(survivors)};triage_priorities={priorities}"
    got = hashlib.sha256(canon.encode()).hexdigest()
    expected = (GOLDEN / "mat_result.sha256").read_text().strip()
    assert got == expected, (
        f"Python MAT result diverged from native-Rust golden "
        f"(canonical form: {canon}; {got} != {expected})"
    )


def test_base_wheel_import_error_message() -> None:
    src = Path(mat.__file__).read_text()
    assert "--features mat" in src
    assert "pip install wifi-densepose[mat]" not in src

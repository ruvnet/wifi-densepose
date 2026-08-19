"""MAT — Mass Casualty Assessment Tool (ADR-024 crate, ADR-185 P3).

WiFi-based disaster-survivor detection and START-protocol triage from CSI:
ingest CSI frames, run a scan cycle, and query detected survivors by triage.

Not in the binary wheels yet (see ruvnet/RuView#1412 — the P6 SOTA
bindings are shipped source-build-only for now to keep the base wheel
small). Build from source with ``maturin ... --features mat`` (or
``--features sota`` for all three P6 subsystems).

Quick start::

    from wifi_densepose.mat import DisasterConfig, DisasterResponse, DisasterType, ScanZone

    cfg = DisasterConfig(DisasterType.Earthquake, sensitivity=0.9, confidence_threshold=0.1)
    resp = DisasterResponse(cfg)
    resp.initialize_event(0.0, 0.0, "Building A")          # required before scanning
    resp.add_zone(ScanZone.rectangle("North Wing", 0.0, 0.0, 50.0, 30.0))
    for amp, phase in csi_stream:
        resp.push_csi_data(amp, phase)
    resp.scan_once()                                        # one detection cycle
    for s in resp.survivors():
        print(s.id, s.triage_status, s.confidence, s.location)

Honest scope (ADR-185 §3.4): the ADR's Rust-side `scan_once()` wrapper was
unnecessary — this binding drives one cycle of the public async
`start_scanning()` (with `continuous_monitoring` forced off) on an internal
runtime. `initialize_event` + `add_zone` are required before `scan_once`.
`Survivor.latest_vitals` returns the latest reading (the Rust accessor is a
history). The detection pipeline is real but unvalidated on live rubble.
"""

from __future__ import annotations

from wifi_densepose import _native

# MAT symbols are compiled into `_native` only under the Rust `mat` feature.
if not hasattr(_native, "DisasterResponse"):
    raise ImportError(
        "wifi_densepose.mat is not in the binary wheels yet "
        "(see ruvnet/RuView#1412). Build from source with "
        "`maturin ... --features mat` (or `--features sota`)."
    )

DisasterType = _native.DisasterType
TriageStatus = _native.TriageStatus
DisasterConfig = _native.DisasterConfig
DisasterResponse = _native.DisasterResponse
ScanZone = _native.ScanZone
Survivor = _native.Survivor
VitalSignsReading = _native.VitalSignsReading

__all__ = [
    "DisasterType",
    "TriageStatus",
    "DisasterConfig",
    "DisasterResponse",
    "ScanZone",
    "Survivor",
    "VitalSignsReading",
]

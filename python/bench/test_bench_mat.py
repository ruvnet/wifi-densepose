"""ADR-185 §4.2 — MAT scan micro-benchmark.

Measures the cost of one full ingest + `scan_once()` cycle over the
committed 256-frame CSI stream. The per-cycle cost should stay comfortably
below the configured scan interval (default 500 ms) so the binding is not
the bottleneck.

Run with:
    pytest python/bench/test_bench_mat.py --benchmark-only

Validated on a RELEASE wheel; a debug wheel will be several× slower.
"""

from __future__ import annotations

import json
from pathlib import Path

from wifi_densepose import mat

_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "golden" / "mat_input.json"


def _stream() -> list[dict]:
    return json.loads(_FIXTURE.read_text())["stream"]


def test_scan_cycle_cost(benchmark) -> None:
    stream = _stream()

    def _run() -> int:
        cfg = mat.DisasterConfig(
            mat.DisasterType.Earthquake, sensitivity=0.9, confidence_threshold=0.1
        )
        resp = mat.DisasterResponse(cfg)
        resp.initialize_event(0.0, 0.0, "bench")
        resp.add_zone(mat.ScanZone.rectangle("Zone A", 0.0, 0.0, 50.0, 30.0))
        for frame in stream:
            resp.push_csi_data(frame["amplitude"], frame["phase"])
        resp.scan_once()
        return len(resp.survivors())

    survivors = benchmark(_run)
    assert survivors == 1

"""MAT disaster-survivor triage from CSI (ADR-185 P3, `[mat]` extra).

Ingest a CSI stream, run one detection cycle, and list detected survivors
by START triage class.

    pip install wifi-densepose[mat]
    python examples/mat_triage.py

Note: the stream here is synthetic (breathing-modulated) — it demonstrates
the API and pipeline, not validated detection accuracy on real rubble.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from wifi_densepose.mat import DisasterConfig, DisasterResponse, DisasterType, ScanZone


def breathing_stream(
    frames: int = 256, subc: int = 56, fs: float = 20.0
) -> Iterator[tuple[list[float], list[float]]]:
    for t in range(frames):
        tt = t / fs
        breath = 2.0 * math.sin(2 * math.pi * 0.3 * tt)
        amp = [10.0 + 0.05 * k + breath for k in range(subc)]
        phase = [0.01 * k + 0.1 * math.sin(2 * math.pi * 0.3 * tt) for k in range(subc)]
        yield amp, phase


def main() -> None:
    cfg = DisasterConfig(DisasterType.Earthquake, sensitivity=0.9, confidence_threshold=0.1)
    resp = DisasterResponse(cfg)
    resp.initialize_event(0.0, 0.0, "Collapsed Building A")
    resp.add_zone(ScanZone.rectangle("North Wing", 0.0, 0.0, 50.0, 30.0))

    for amp, phase in breathing_stream():
        resp.push_csi_data(amp, phase)
    resp.scan_once()

    survivors = resp.survivors()
    print(f"detected {len(survivors)} survivor(s)")
    for s in survivors:
        print(f"  {s.id[:8]}  triage={s.triage_status}  confidence={s.confidence:.3f}")


if __name__ == "__main__":
    main()

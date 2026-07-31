"""ADR-185 §4.2 — MERIDIAN micro-benchmarks.

Targets (release build, ADR-027 §4.1/§4.3 ×2 headroom): `normalize()`
< 200 µs/frame, `encode()` < 200 µs.

Run with:
    pytest python/bench/test_bench_meridian.py --benchmark-only

Validated on a RELEASE wheel; a debug wheel will be several× slower.
"""

from __future__ import annotations

from wifi_densepose import meridian as mer


def test_normalize_per_frame(benchmark) -> None:
    norm = mer.HardwareNormalizer()
    amp = [10.0 + 0.05 * k for k in range(64)]
    phase = [0.01 * k for k in range(64)]
    out = benchmark(lambda: norm.normalize(amp, phase, mer.HardwareType.Esp32S3))
    assert len(out.amplitude) == 56


def test_geometry_encode(benchmark) -> None:
    enc = mer.GeometryEncoder(mer.MeridianGeometryConfig())
    aps = [[0.0, 0.0, 2.5], [5.0, 0.0, 2.5], [0.0, 4.0, 2.5]]
    out = benchmark(lambda: enc.encode(aps))
    assert len(out) == 64

"""ADR-185 §4.2 — AETHER embed() micro-benchmarks.

Target (release build, ADR-024 §2.8 FP32 <1 ms with headroom): steady-state
`embed()` < 2 ms/window, and batched `embed()` scales roughly linearly (no
accidental O(n²)).

Run with:
    pytest python/bench/test_bench_aether.py --benchmark-only

Skipped by default (they live in `bench/`, outside `testpaths`). Timing
targets are validated on a RELEASE wheel (`maturin develop --release
--features sota`); a debug wheel will be several× slower.
"""

from __future__ import annotations

import math

import pytest

from wifi_densepose import aether


def _window(frames: int = 8, subc: int = 56) -> list[list[float]]:
    return [[math.sin(0.1 * t + 0.03 * k) for k in range(subc)] for t in range(frames)]


def _extractor() -> aether.EmbeddingExtractor:
    return aether.EmbeddingExtractor(n_subcarriers=56, config=aether.AetherConfig())


def test_embed_per_window(benchmark) -> None:
    ext = _extractor()
    window = _window()
    out = benchmark(lambda: ext.embed(window))
    assert len(out) == 128


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_embed_batch_scaling(benchmark, batch: int) -> None:
    ext = _extractor()
    windows = [_window() for _ in range(batch)]
    out = benchmark(lambda: [ext.embed(w) for w in windows])
    assert len(out) == batch

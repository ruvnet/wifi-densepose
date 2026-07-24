"""AETHER — contrastive CSI embeddings & re-identification (ADR-024, ADR-185 P1).

Self-supervised 128-dim L2-normalized embeddings for WiFi CSI: room
fingerprinting, person re-identification, and anomaly scoring, computed
entirely offline by the Rust core (no server, no network).

Not in the binary wheels yet (see ruvnet/RuView#1412 — the P6 SOTA
bindings are shipped source-build-only for now to keep the base wheel
small). Build from source with ``maturin ... --features aether`` (or
``--features sota`` for all three P6 subsystems).

Quick start::

    from wifi_densepose.aether import AetherConfig, EmbeddingExtractor, cosine_similarity

    ext = EmbeddingExtractor(n_subcarriers=56, config=AetherConfig())
    a = ext.embed(window_a)          # list[float], length == config.d_proj (128)
    b = ext.embed(window_b)
    score = cosine_similarity(a, b)  # re-ID similarity in [-1, 1]
"""

from __future__ import annotations

from wifi_densepose import _native

# The AETHER symbols are compiled into `_native` only under the Rust `aether`
# feature. The binary wheels do NOT enable it yet (ruvnet/RuView#1412);
# it is available from a source build with the feature. Name that fix, not
# a pip extra, which cannot add compiled code to a built wheel.
if not hasattr(_native, "AetherConfig"):
    raise ImportError(
        "wifi_densepose.aether is not in the binary wheels yet "
        "(see ruvnet/RuView#1412). Build from source with "
        "`maturin ... --features aether` (or `--features sota`)."
    )

AetherConfig = _native.AetherConfig
CsiAugmenter = _native.CsiAugmenter
EmbeddingExtractor = _native.EmbeddingExtractor
info_nce_loss = _native.info_nce_loss
cosine_similarity = _native.cosine_similarity

__all__ = [
    "AetherConfig",
    "CsiAugmenter",
    "EmbeddingExtractor",
    "info_nce_loss",
    "cosine_similarity",
]

"""AETHER re-identification from CSI (ADR-185 P1, `[aether]` extra).

Compute 128-dim contrastive embeddings for CSI windows and score them by
cosine similarity — the primitive behind room fingerprinting and person
re-identification.

    pip install wifi-densepose[aether]
    python examples/reid_from_csi.py
"""

from __future__ import annotations

import math

from wifi_densepose.aether import AetherConfig, EmbeddingExtractor, cosine_similarity


def make_window(phase_shift: float, frames: int = 8, subc: int = 56) -> list[list[float]]:
    """A synthetic CSI window; `phase_shift` stands in for a different scene."""
    return [
        [math.sin(0.1 * t + 0.03 * k + phase_shift) for k in range(subc)]
        for t in range(frames)
    ]


def main() -> None:
    ext = EmbeddingExtractor(n_subcarriers=56, config=AetherConfig())

    same_a = ext.embed(make_window(0.0))
    same_b = ext.embed(make_window(0.0))  # same scene
    other = ext.embed(make_window(1.5))   # different scene

    print(f"embedding dim:        {len(same_a)}")
    print(f"same-scene similarity:  {cosine_similarity(same_a, same_b):.4f}")
    print(f"cross-scene similarity: {cosine_similarity(same_a, other):.4f}")


if __name__ == "__main__":
    main()

"""MERIDIAN cross-room calibration (ADR-185 P2, `[meridian]` extra).

Hardware-invariant CSI normalization, AP-geometry encoding, and few-shot
rapid adaptation — the tch-free domain-generalization path.

    pip install wifi-densepose[meridian]
    python examples/cross_room_calibrate.py
"""

from __future__ import annotations

import math

from wifi_densepose.meridian import (
    GeometryEncoder,
    HardwareNormalizer,
    HardwareType,
    MeridianGeometryConfig,
    RapidAdaptation,
)


def main() -> None:
    # 1. Normalize a 64-subcarrier ESP32 frame to the canonical 56-tone grid.
    norm = HardwareNormalizer()
    amp = [10.0 + 0.05 * k for k in range(64)]
    phase = [0.01 * k for k in range(64)]
    frame = norm.normalize(amp, phase, HardwareType.detect(64))
    print(f"canonical subcarriers: {len(frame.amplitude)} (hw={frame.hardware_type})")

    # 2. Encode AP positions into a permutation-invariant geometry embedding.
    enc = GeometryEncoder(MeridianGeometryConfig())
    geometry = enc.encode([[0.0, 0.0, 2.5], [5.0, 0.0, 2.5], [0.0, 4.0, 2.5]])
    print(f"geometry embedding dim: {len(geometry)}")

    # 3. Few-shot rapid adaptation over a handful of unlabeled frames.
    ra = RapidAdaptation(min_calibration_frames=10, lora_rank=4)
    for i in range(12):
        ra.push_frame([math.sin(0.1 * i + 0.05 * d) for d in range(16)])
    result = ra.adapt()
    print(f"adapted over {result.frames_used} frames, final_loss={result.final_loss:.4f}")


if __name__ == "__main__":
    main()

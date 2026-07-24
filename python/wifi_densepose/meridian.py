"""MERIDIAN — cross-environment domain generalization (ADR-027, ADR-185 P2).

Hardware-invariant CSI normalization, geometry-conditioned deployment,
few-shot room adaptation, and cross-domain evaluation — the tch-free
inference/adaptation path of Project MERIDIAN, computed by the Rust core.

Not in the binary wheels yet (see ruvnet/RuView#1412 — the P6 SOTA
bindings are shipped source-build-only for now to keep the base wheel
small). Build from source with ``maturin ... --features meridian`` (or
``--features sota`` for all three P6 subsystems).

Quick start::

    from wifi_densepose.meridian import HardwareNormalizer, HardwareType

    norm = HardwareNormalizer()                      # canonical 56 subcarriers
    hw = HardwareType.detect(64)                     # -> HardwareType.Esp32S3
    frame = norm.normalize(amplitude, phase, hw)     # -> CanonicalCsiFrame
    print(len(frame.amplitude), frame.hardware_type)

Note (honest scope, ADR-185 §3.3): the ADR's ``RapidAdaptation.calibrate``
/ ``AdaptationResult.converged`` do not exist in the Rust core — use
``push_frame(...)`` then ``adapt()``; the result exposes ``final_loss``,
``frames_used``, ``adaptation_epochs``. Training-time types
(DomainFactorizer, GradientReversalLayer, VirtualDomainAugmentor) are
out of scope for P6 (they need the deferred libtorch training tier).
"""

from __future__ import annotations

from wifi_densepose import _native

# MERIDIAN symbols are compiled into `_native` only under the Rust
# `meridian` feature; absent in a base wheel (ADR-185 §6 acceptance).
if not hasattr(_native, "HardwareNormalizer"):
    raise ImportError(
        "wifi_densepose.meridian is not in the binary wheels yet "
        "(see ruvnet/RuView#1412). Build from source with "
        "`maturin ... --features meridian` (or `--features sota`)."
    )

HardwareType = _native.HardwareType
CanonicalCsiFrame = _native.CanonicalCsiFrame
HardwareNormalizer = _native.HardwareNormalizer
MeridianGeometryConfig = _native.MeridianGeometryConfig
GeometryEncoder = _native.GeometryEncoder
RapidAdaptation = _native.RapidAdaptation
AdaptationResult = _native.AdaptationResult
CrossDomainEvaluator = _native.CrossDomainEvaluator
mpjpe = _native.mpjpe

__all__ = [
    "HardwareType",
    "CanonicalCsiFrame",
    "HardwareNormalizer",
    "MeridianGeometryConfig",
    "GeometryEncoder",
    "RapidAdaptation",
    "AdaptationResult",
    "CrossDomainEvaluator",
    "mpjpe",
]

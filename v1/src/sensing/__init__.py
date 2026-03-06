"""
Commodity WiFi Sensing Module (ADR-013)
=======================================

RSSI-based presence and motion detection using standard Linux WiFi metrics.
This module provides real signal processing from commodity WiFi hardware,
extracting presence and motion features from RSSI time series.

Components:
    - rssi_collector: Data collection from Linux WiFi interfaces
    - feature_extractor: Time-domain and frequency-domain feature extraction
    - classifier: Presence and motion classification from features
    - backend: Common sensing backend interface

Capabilities:
    - PRESENCE: Detect whether a person is present in the sensing area
    - MOTION: Classify motion level (absent / still / active)

Note: This module uses RSSI only. For higher-fidelity sensing (respiration,
pose estimation), CSI-capable hardware and the full DensePose pipeline
are required.
"""

from src.sensing.rssi_collector import (
    LinuxWifiCollector,
    SimulatedCollector,
    WindowsWifiCollector,
    WifiSample,
)
from src.sensing.feature_extractor import (
    RssiFeatureExtractor,
    RssiFeatures,
)
from src.sensing.classifier import (
    PresenceClassifier,
    SensingResult,
    MotionLevel,
)
from src.sensing.backend import (
    SensingBackend,
    CommodityBackend,
    Capability,
)

__all__ = [
    "LinuxWifiCollector",
    "SimulatedCollector",
    "WindowsWifiCollector",
    "WifiSample",
    "RssiFeatureExtractor",
    "RssiFeatures",
    "PresenceClassifier",
    "SensingResult",
    "MotionLevel",
    "SensingBackend",
    "CommodityBackend",
    "Capability",
]

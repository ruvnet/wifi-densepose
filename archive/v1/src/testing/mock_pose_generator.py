"""
Mock pose data generator for testing and development.

Generates STABLE, REALISTIC synthetic pose data:
- Fixed 2 persons seated (no random person count per frame)
- Anatomically correct seated skeleton keypoints
- Temporal coherence: micro-movements only (breathing, slight head drift)
- Confidence values stable and high (0.82–0.94)
- No random teleportation between frames

WARNING: This module uses random number generation intentionally for test data.
Do NOT use this module in production data paths.
"""

import math
import time
import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

MOCK_POSE_BANNER = """
================================================================================
  WARNING: MOCK POSE MODE ACTIVE - Using synthetic pose data

  All pose detections are randomly generated and do NOT represent real humans.
  For real pose estimation, provide trained model weights and real CSI data.
  See docs/hardware-setup.md for configuration instructions.
================================================================================
"""

_banner_shown = False

# ── Anatomically correct seated pose keypoints (normalised 0-1) ──────────────
# Origin top-left. Y increases downward. Two persons side by side.
# Each person occupies roughly 0.3 width of frame.
_SEATED_PERSONS = [
    # Person 0 — left side of frame (x centred ~0.25)
    {
        "cx": 0.25,   # bounding-box centre x
        "cy": 0.48,   # bounding-box centre y
        "bw": 0.22,   # bounding-box width
        "bh": 0.55,   # bounding-box height
        # keypoint [x, y] relative to bounding-box centre (normalised frame coords)
        "kp": {
            "nose":           [0.00,  -0.20],
            "left_eye":       [-0.03, -0.22],
            "right_eye":      [ 0.03, -0.22],
            "left_ear":       [-0.06, -0.20],
            "right_ear":      [ 0.06, -0.20],
            "left_shoulder":  [-0.09, -0.10],
            "right_shoulder": [ 0.09, -0.10],
            "left_elbow":     [-0.11,  0.02],
            "right_elbow":    [ 0.11,  0.02],
            "left_wrist":     [-0.10,  0.12],
            "right_wrist":    [ 0.10,  0.12],
            "left_hip":       [-0.07,  0.16],
            "right_hip":      [ 0.07,  0.16],
            "left_knee":      [-0.07,  0.28],
            "right_knee":     [ 0.07,  0.28],
            "left_ankle":     [-0.07,  0.33],
            "right_ankle":    [ 0.07,  0.33],
        },
    },
    # Person 1 — right side of frame (x centred ~0.67)
    {
        "cx": 0.67,
        "cy": 0.46,
        "bw": 0.22,
        "bh": 0.55,
        "kp": {
            "nose":           [ 0.01, -0.21],
            "left_eye":       [-0.03, -0.23],
            "right_eye":      [ 0.03, -0.23],
            "left_ear":       [-0.06, -0.21],
            "right_ear":      [ 0.06, -0.21],
            "left_shoulder":  [-0.09, -0.11],
            "right_shoulder": [ 0.09, -0.11],
            "left_elbow":     [-0.12,  0.01],
            "right_elbow":    [ 0.12,  0.01],
            "left_wrist":     [-0.11,  0.11],
            "right_wrist":    [ 0.11,  0.11],
            "left_hip":       [-0.07,  0.15],
            "right_hip":      [ 0.07,  0.15],
            "left_knee":      [-0.07,  0.27],
            "right_knee":     [ 0.07,  0.27],
            "left_ankle":     [-0.07,  0.32],
            "right_ankle":    [ 0.07,  0.32],
        },
    },
]

# Micro-movement amplitudes (normalised frame coords)
_BREATH_AMP   = 0.003   # breathing: chest/shoulder rise
_HEAD_AMP     = 0.004   # subtle head drift
_WRIST_AMP    = 0.006   # hand micro-movements (typing, resting)
_NOISE_AMP    = 0.001   # detector noise floor


class _PersonState:
    """Stateful keypoint smoother for one seated person."""

    def __init__(self, template: Dict, seed: int):
        self._t   = template
        self._rng = random.Random(seed)
        # Independent slow-drift phases per keypoint group
        self._breath_phase = self._rng.uniform(0, math.pi * 2)
        self._head_phase   = self._rng.uniform(0, math.pi * 2)
        self._wrist_phase  = self._rng.uniform(0, math.pi * 2)
        # Smooth noise state (exponential moving average)
        self._noise: Dict[str, List[float]] = {
            name: [0.0, 0.0] for name in template["kp"]
        }

    def _smooth_noise(self, name: str, alpha: float = 0.85) -> List[float]:
        """Low-pass filtered noise — smooth, not jumpy."""
        nx = self._rng.gauss(0, _NOISE_AMP)
        ny = self._rng.gauss(0, _NOISE_AMP)
        s  = self._noise[name]
        s[0] = alpha * s[0] + (1 - alpha) * nx
        s[1] = alpha * s[1] + (1 - alpha) * ny
        return s

    def keypoints(self, t: float) -> List[Dict[str, Any]]:
        """Return keypoints for time t (seconds)."""
        cx, cy = self._t["cx"], self._t["cy"]
        kpts   = []

        breath  = _BREATH_AMP  * math.sin(t * 0.27 + self._breath_phase)   # ~16 rpm
        head_dx = _HEAD_AMP    * math.sin(t * 0.11 + self._head_phase)
        head_dy = _HEAD_AMP    * math.cos(t * 0.09 + self._head_phase) * 0.5
        wr_dx   = _WRIST_AMP   * math.sin(t * 0.31 + self._wrist_phase)
        wr_dy   = _WRIST_AMP   * math.cos(t * 0.27 + self._wrist_phase)

        for name, (ox, oy) in self._t["kp"].items():
            x = cx + ox
            y = cy + oy

            # Apply physiological micro-movements by body region
            if name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
                x += head_dx
                y += head_dy
            elif name in ("left_shoulder", "right_shoulder",
                          "left_elbow",    "right_elbow"):
                y += breath
            elif name in ("left_wrist", "right_wrist"):
                x += wr_dx
                y += wr_dy

            # Add smooth detector noise
            n = self._smooth_noise(name)
            x += n[0]
            y += n[1]

            # Stable per-keypoint confidence (high, minimal variance)
            conf = 0.88 + self._rng.gauss(0, 0.025)
            conf = max(0.70, min(0.97, conf))

            kpts.append({"name": name, "x": round(x, 4), "y": round(y, 4),
                         "confidence": round(conf, 3)})
        return kpts

    def bounding_box(self, t: float) -> Dict[str, float]:
        bw, bh = self._t["bw"], self._t["bh"]
        cx     = self._t["cx"] + _NOISE_AMP * math.sin(t * 0.07)
        cy     = self._t["cy"] + _NOISE_AMP * math.cos(t * 0.05)
        return {
            "x":      round(cx - bw / 2, 4),
            "y":      round(cy - bh / 2, 4),
            "width":  round(bw, 4),
            "height": round(bh, 4),
        }

    def overall_confidence(self) -> float:
        return round(0.88 + random.gauss(0, 0.015), 3)


# Module-level stateful persons — survive across repeated calls
_persons: List[_PersonState] = [
    _PersonState(_SEATED_PERSONS[0], seed=42),
    _PersonState(_SEATED_PERSONS[1], seed=73),
]

_start_time: float = time.time()


def _show_banner() -> None:
    global _banner_shown
    if not _banner_shown:
        logger.warning(MOCK_POSE_BANNER)
        _banner_shown = True


# ── Public API (unchanged signatures) ─────────────────────────────────────────

def generate_mock_keypoints() -> List[Dict[str, Any]]:
    """Return stable seated keypoints for person 0 (legacy single-person helper)."""
    _show_banner()
    t = time.time() - _start_time
    return _persons[0].keypoints(t)


def generate_mock_bounding_box() -> Dict[str, float]:
    """Return stable bounding box for person 0 (legacy helper)."""
    t = time.time() - _start_time
    return _persons[0].bounding_box(t)


def generate_mock_poses(max_persons: int = 3) -> List[Dict[str, Any]]:
    """Generate stable mock pose detections for exactly 2 seated persons.

    Args:
        max_persons: Ignored — always returns 2 persons to reflect real scene.

    Returns:
        List of 2 pose detection dictionaries with temporal coherence.
    """
    _show_banner()
    t      = time.time() - _start_time
    n      = min(2, max_persons)   # never exceed caller's limit, but always ≥ 1
    poses  = []

    for i, person in enumerate(_persons[:n]):
        poses.append({
            "person_id":    str(i),
            "confidence":   person.overall_confidence(),
            "keypoints":    person.keypoints(t),
            "bounding_box": person.bounding_box(t),
            "activity":     "sitting",
            "timestamp":    datetime.now().isoformat(),
        })

    return poses


def generate_mock_zone_occupancy(zone_id: str) -> Dict[str, Any]:
    """Generate stable zone occupancy for 2 seated persons."""
    _show_banner()
    persons = [
        {"person_id": f"person_{i}", "confidence": round(0.87 + random.gauss(0, 0.02), 3),
         "activity": "sitting"}
        for i in range(2)
    ]
    return {
        "count":        2,
        "max_occupancy": 10,
        "persons":      persons,
        "timestamp":    datetime.now(),
    }


def generate_mock_zones_summary(
    zone_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate mock zones summary — 2 persons in zone_1, rest empty."""
    _show_banner()
    zones      = zone_ids or ["zone_1", "zone_2", "zone_3", "zone_4"]
    zone_data  = {}
    total      = 0
    active     = 0

    for zone_id in zones:
        count = 2 if zone_id == "zone_1" else 0
        zone_data[zone_id] = {
            "occupancy":     count,
            "max_occupancy": 10,
            "status":        "active" if count > 0 else "inactive",
        }
        total  += count
        active += 1 if count > 0 else 0

    return {"total_persons": total, "zones": zone_data, "active_zones": active}


def generate_mock_historical_data(
    start_time: datetime,
    end_time: datetime,
    zone_ids: Optional[List[str]] = None,
    aggregation_interval: int = 300,
    include_raw_data: bool = False,
) -> Dict[str, Any]:
    """Generate mock historical pose data (stable occupancy: 2 persons)."""
    _show_banner()
    zones           = zone_ids or ["zone_1", "zone_2", "zone_3"]
    current_time    = start_time
    aggregated_data = []
    raw_data        = [] if include_raw_data else None

    while current_time < end_time:
        data_point: Dict[str, Any] = {
            "timestamp":     current_time,
            "total_persons": 2,
            "zones":         {},
        }
        for zone_id in zones:
            count = 2 if zone_id == "zone_1" else 0
            data_point["zones"][zone_id] = {
                "occupancy":       count,
                "avg_confidence":  round(0.87 + random.gauss(0, 0.01), 3),
            }
        aggregated_data.append(data_point)

        if include_raw_data:
            for _ in range(random.randint(1, 3)):
                raw_data.append({  # type: ignore[union-attr]
                    "timestamp":  current_time + timedelta(
                        seconds=random.randint(0, aggregation_interval)),
                    "person_id":  f"person_{random.randint(0, 1)}",
                    "zone_id":    "zone_1",
                    "confidence": round(0.87 + random.gauss(0, 0.015), 3),
                    "activity":   "sitting",
                })

        current_time += timedelta(seconds=aggregation_interval)

    return {
        "aggregated_data": aggregated_data,
        "raw_data":        raw_data,
        "total_records":   len(aggregated_data),
    }


def generate_mock_recent_activities(
    zone_id: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Generate recent activity data — two seated persons."""
    _show_banner()
    activities = []
    for i in range(limit):
        activities.append({
            "activity_id":       f"activity_{i}",
            "person_id":         f"person_{i % 2}",
            "zone_id":           zone_id or "zone_1",
            "activity":          "sitting",
            "confidence":        round(0.88 + random.gauss(0, 0.015), 3),
            "timestamp":         datetime.now() - timedelta(minutes=i * 3),
            "duration_seconds":  random.randint(60, 600),
        })
    return activities


def generate_mock_statistics(
    start_time: datetime,
    end_time: datetime,
) -> Dict[str, Any]:
    """Generate mock statistics reflecting a stable 2-person seated session."""
    _show_banner()
    total      = random.randint(400, 600)
    successful = int(total * random.uniform(0.92, 0.97))
    return {
        "total_detections":        total,
        "successful_detections":   successful,
        "failed_detections":       total - successful,
        "success_rate":            round(successful / total, 4),
        "average_confidence":      round(0.88 + random.gauss(0, 0.01), 3),
        "average_processing_time_ms": round(random.uniform(30, 80), 1),
        "unique_persons":          2,
        "most_active_zone":        "zone_1",
        "activity_distribution": {
            "sitting": 0.94,
            "standing": 0.04,
            "walking": 0.01,
            "lying": 0.01,
        },
    }

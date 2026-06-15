//! The per-room specialist bank (ADR-151 Stage 4).
//!
//! A versioned collection of small models scoped to one `room_id`, fit from the
//! enrollment anchors and tied to the ADR-135 baseline it was trained against.
//! When the baseline drifts (room rearranged, AP moved), the bank is marked
//! STALE rather than emitting confident-but-wrong readings — the calibration
//! analogue of the firmware's honest `DEGRADED` flag.

use serde::{Deserialize, Serialize};

use crate::error::{CalibrationError, Result};
use crate::extract::AnchorFeature;
use crate::geometry::NodeGeometry;
use crate::specialist::{
    AnomalySpecialist, BreathingSpecialist, HeartbeatSpecialist, PostureSpecialist,
    PresenceSpecialist, RestlessnessSpecialist, SpecialistKind,
};

/// A versioned bank of room-calibrated specialists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistBank {
    /// Room scope.
    pub room_id: String,
    /// ADR-135 baseline id this bank was trained against (drift → STALE).
    pub baseline_id: String,
    /// Training time (unix seconds).
    pub trained_at_unix_s: i64,
    /// Number of anchors used.
    pub anchor_count: usize,
    /// Transceiver geometry snapshot the bank was trained under (ADR-152
    /// §2.1.1). Empty both for banks persisted before geometry existed (serde
    /// default — same pattern as `PresenceSpecialist::mean_dist_threshold`) and
    /// for enrollments where no geometry was recorded. Statistical specialists
    /// ignore it; the ADR-151 P6 LoRA heads will consume it (ADR-152 §2.1.2).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub geometry: Vec<NodeGeometry>,

    /// Presence gate (requires the `empty` + an occupied anchor).
    pub presence: Option<PresenceSpecialist>,
    /// Posture classifier (requires posture anchors).
    pub posture: Option<PostureSpecialist>,
    /// Breathing (band-limited periodicity; stateless).
    pub breathing: BreathingSpecialist,
    /// Heartbeat (band-limited periodicity; stateless).
    pub heartbeat: HeartbeatSpecialist,
    /// Restlessness (requires calm + active anchors).
    pub restlessness: Option<RestlessnessSpecialist>,
    /// Anomaly novelty detector (requires ≥2 anchors).
    pub anomaly: Option<AnomalySpecialist>,
}

impl SpecialistBank {
    /// Train a bank from enrollment anchor features.
    ///
    /// Requires at least one anchor; specialists whose prerequisite anchors are
    /// missing are simply left `None` (a partial bank still works for the
    /// signals it could fit).
    pub fn train(
        room_id: impl Into<String>,
        baseline_id: impl Into<String>,
        anchors: &[AnchorFeature],
        at_unix_s: i64,
    ) -> Result<Self> {
        if anchors.is_empty() {
            return Err(CalibrationError::InsufficientSamples {
                kind: "bank".into(),
                have: 0,
                need: 1,
            });
        }
        Ok(Self {
            room_id: room_id.into(),
            baseline_id: baseline_id.into(),
            trained_at_unix_s: at_unix_s,
            anchor_count: anchors.len(),
            geometry: Vec::new(),
            presence: PresenceSpecialist::train(anchors),
            posture: PostureSpecialist::train(anchors),
            breathing: BreathingSpecialist::default(),
            heartbeat: HeartbeatSpecialist::default(),
            restlessness: RestlessnessSpecialist::train(anchors),
            anomaly: AnomalySpecialist::train(anchors),
        })
    }

    /// Attach the enrollment's transceiver-geometry snapshot (ADR-152 §2.1.1),
    /// builder style — typically `EnrollmentSession::geometry()` at train time.
    pub fn with_geometry(mut self, geometry: Vec<NodeGeometry>) -> Self {
        self.geometry = geometry;
        self
    }

    /// The fixed-length geometry embedding of the bank's snapshot (ADR-152
    /// §2.1.2) — the conditioning vector the ADR-151 P6 LoRA heads concatenate
    /// with the backbone embedding. Derived on demand from [`Self::geometry`]
    /// (it is a pure function of the snapshot), so it adds no schema surface;
    /// a geometry-free bank yields the well-defined all-zero embedding.
    pub fn geometry_embedding(&self) -> crate::geometry_embedding::GeometryEmbedding {
        crate::geometry_embedding::GeometryEmbedding::from_nodes(&self.geometry)
    }

    /// `true` if the bank was trained against a different baseline (it is STALE).
    pub fn is_stale(&self, current_baseline_id: &str) -> bool {
        self.baseline_id != current_baseline_id
    }

    /// Error out if stale.
    pub fn check_fresh(&self, current_baseline_id: &str) -> Result<()> {
        if self.is_stale(current_baseline_id) {
            Err(CalibrationError::StaleBaseline {
                trained: self.baseline_id.clone(),
                current: current_baseline_id.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Which specialists were successfully fit.
    pub fn trained_kinds(&self) -> Vec<SpecialistKind> {
        let mut v = vec![SpecialistKind::Breathing, SpecialistKind::Heartbeat];
        if self.presence.is_some() {
            v.push(SpecialistKind::Presence);
        }
        if self.posture.is_some() {
            v.push(SpecialistKind::Posture);
        }
        if self.restlessness.is_some() {
            v.push(SpecialistKind::Restlessness);
        }
        if self.anomaly.is_some() {
            v.push(SpecialistKind::Anomaly);
        }
        v
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| CalibrationError::Serde(e.to_string()))
    }

    /// Deserialize from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| CalibrationError::Serde(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchor::AnchorLabel;
    use crate::extract::Features;

    fn af(label: AnchorLabel, variance: f32, motion: f32) -> AnchorFeature {
        AnchorFeature {
            room_id: "living-room".into(),
            label,
            features: Features {
                mean: 1.0,
                variance,
                motion,
                breathing_score: 0.0,
                breathing_hz: 0.0,
                heart_score: 0.0,
                heart_hz: 0.0,
            },
        }
    }

    fn full_anchors() -> Vec<AnchorFeature> {
        vec![
            af(AnchorLabel::Empty, 1.0, 0.1),
            af(AnchorLabel::StandStill, 10.0, 0.2),
            af(AnchorLabel::Sit, 6.0, 0.2),
            af(AnchorLabel::LieDown, 3.0, 0.2),
            af(AnchorLabel::SmallMove, 4.0, 1.2),
            af(AnchorLabel::SleepPosture, 3.0, 0.1),
        ]
    }

    #[test]
    fn train_full_bank() {
        let bank = SpecialistBank::train("living-room", "base-1", &full_anchors(), 1000).unwrap();
        let kinds = bank.trained_kinds();
        assert!(kinds.contains(&SpecialistKind::Presence));
        assert!(kinds.contains(&SpecialistKind::Posture));
        assert!(kinds.contains(&SpecialistKind::Restlessness));
        assert!(kinds.contains(&SpecialistKind::Anomaly));
        assert_eq!(bank.anchor_count, 6);
    }

    #[test]
    fn empty_anchors_error() {
        assert!(SpecialistBank::train("r", "b", &[], 0).is_err());
    }

    #[test]
    fn json_roundtrip() {
        let bank = SpecialistBank::train("r", "base-1", &full_anchors(), 1000).unwrap();
        let json = bank.to_json().unwrap();
        let back = SpecialistBank::from_json(&json).unwrap();
        assert_eq!(back.room_id, "r");
        assert_eq!(back.anchor_count, 6);
    }

    #[test]
    fn geometry_snapshot_roundtrips() {
        let geometry = vec![
            NodeGeometry::new(1, "tape-measure").with_position(0.0, 0.0, 1.0),
            NodeGeometry::unknown(2),
        ];
        let bank = SpecialistBank::train("r", "base-1", &full_anchors(), 1000)
            .unwrap()
            .with_geometry(geometry.clone());
        let json = bank.to_json().unwrap();
        let back = SpecialistBank::from_json(&json).unwrap();
        assert_eq!(back.geometry, geometry);
    }

    /// ADR-152 §2.1.2: the embedding is derived from the snapshot — present
    /// geometry conditions it, absent geometry yields the all-zero vector.
    #[test]
    fn geometry_embedding_derives_from_snapshot() {
        let bare = SpecialistBank::train("r", "base-1", &full_anchors(), 1000).unwrap();
        assert_eq!(
            bare.geometry_embedding(),
            crate::geometry_embedding::GeometryEmbedding::default(),
            "no geometry → all-zero embedding"
        );

        let geometry = vec![
            NodeGeometry::new(1, "tape-measure").with_position(0.0, 0.0, 1.0),
            NodeGeometry::new(2, "tape-measure").with_position(3.0, 0.0, 1.0),
        ];
        let bank = bare.with_geometry(geometry.clone());
        let emb = bank.geometry_embedding();
        assert_eq!(
            emb,
            crate::geometry_embedding::GeometryEmbedding::from_nodes(&geometry),
            "embedding is a pure function of the snapshot"
        );
        assert!(emb.as_slice().iter().any(|&x| x != 0.0));
    }

    /// ADR-152 schema-compat fixture: bank JSON persisted BEFORE the geometry
    /// field existed (captured from the pre-ADR-152 serializer shape) must
    /// deserialize cleanly with an empty geometry snapshot.
    #[test]
    fn pre_geometry_bank_json_loads() {
        let old_json = r#"{
            "room_id": "living-room",
            "baseline_id": "base-1",
            "trained_at_unix_s": 1000,
            "anchor_count": 2,
            "presence": {"threshold": 5.5, "occupied_var": 10.0},
            "posture": null,
            "breathing": {"min_score": 0.0},
            "heartbeat": {"min_score": 0.0},
            "restlessness": null,
            "anomaly": null
        }"#;
        let bank = SpecialistBank::from_json(old_json).unwrap();
        assert!(bank.geometry.is_empty(), "old banks carry no geometry");
        assert_eq!(bank.room_id, "living-room");
        assert!(bank.presence.is_some());
        // And a geometry-free bank serializes without the field (old shape).
        assert!(!bank.to_json().unwrap().contains("geometry"));
    }

    #[test]
    fn staleness() {
        let bank = SpecialistBank::train("r", "base-1", &full_anchors(), 1000).unwrap();
        assert!(!bank.is_stale("base-1"));
        assert!(bank.is_stale("base-2"));
        assert!(bank.check_fresh("base-2").is_err());
    }
}

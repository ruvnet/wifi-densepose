//! Guided anchors + event-sourced enrollment session (ADR-151 Stage 2).
//!
//! Enrollment teaches the room a small set of *clean anchors* — not hours of
//! data. Each anchor is a short labelled capture (stand / sit / lie / breathe /
//! move / sleep) layered on top of the ADR-135 empty-room baseline. The session
//! is event-sourced so re-enrollment is incremental and auditable (per CLAUDE.md
//! state rules).

use serde::{Deserialize, Serialize};

use crate::geometry::NodeGeometry;

/// Coarse posture an anchor establishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Posture {
    /// Standing.
    Standing,
    /// Sitting.
    Sitting,
    /// Lying down.
    Lying,
}

/// The fixed guided-anchor sequence (ADR-151 §2.2).
///
/// Serializes as snake_case (`empty`, `stand_still`, …) to match
/// [`AnchorLabel::as_str`] and the documented JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorLabel {
    /// Empty room reference (reuses the ADR-135 baseline).
    Empty,
    /// Person standing still, in view of the sensor.
    StandStill,
    /// Person sitting.
    Sit,
    /// Person lying down.
    LieDown,
    /// Slow respiration (~0.1–0.15 Hz).
    BreatheSlow,
    /// Normal respiration (~0.2–0.3 Hz).
    BreatheNormal,
    /// Small limb movement.
    SmallMove,
    /// Quiescent sleep posture (lying, still).
    SleepPosture,
}

impl AnchorLabel {
    /// The canonical enrollment order.
    pub const SEQUENCE: [AnchorLabel; 8] = [
        AnchorLabel::Empty,
        AnchorLabel::StandStill,
        AnchorLabel::Sit,
        AnchorLabel::LieDown,
        AnchorLabel::BreatheSlow,
        AnchorLabel::BreatheNormal,
        AnchorLabel::SmallMove,
        AnchorLabel::SleepPosture,
    ];

    /// Stable string id (used in persistence / API).
    pub fn as_str(&self) -> &'static str {
        match self {
            AnchorLabel::Empty => "empty",
            AnchorLabel::StandStill => "stand_still",
            AnchorLabel::Sit => "sit",
            AnchorLabel::LieDown => "lie_down",
            AnchorLabel::BreatheSlow => "breathe_slow",
            AnchorLabel::BreatheNormal => "breathe_normal",
            AnchorLabel::SmallMove => "small_move",
            AnchorLabel::SleepPosture => "sleep_posture",
        }
    }

    /// Parse from the stable string id.
    pub fn from_str(s: &str) -> Option<AnchorLabel> {
        AnchorLabel::SEQUENCE
            .iter()
            .copied()
            .find(|a| a.as_str() == s)
    }

    /// Operator-facing prompt shown by the CLI / UI.
    pub fn prompt(&self) -> &'static str {
        match self {
            AnchorLabel::Empty => "Leave the room empty and still…",
            AnchorLabel::StandStill => "Stand still, in view of the sensor…",
            AnchorLabel::Sit => "Sit down and stay still…",
            AnchorLabel::LieDown => "Lie down and stay still…",
            AnchorLabel::BreatheSlow => "Lie or sit still and breathe slowly…",
            AnchorLabel::BreatheNormal => "Stay still and breathe normally…",
            AnchorLabel::SmallMove => "Make small movements (wave a hand, shift)…",
            AnchorLabel::SleepPosture => "Lie in your sleep posture and relax…",
        }
    }

    /// Suggested capture duration (seconds).
    pub fn duration_s(&self) -> u32 {
        match self {
            AnchorLabel::BreatheSlow | AnchorLabel::BreatheNormal | AnchorLabel::SleepPosture => 30,
            _ => 20,
        }
    }

    /// Whether a person is expected to be present for this anchor.
    pub fn expects_presence(&self) -> bool {
        !matches!(self, AnchorLabel::Empty)
    }

    /// Whether the subject is expected to be (largely) still.
    pub fn expects_still(&self) -> bool {
        !matches!(self, AnchorLabel::SmallMove)
    }

    /// Posture this anchor establishes, if any.
    pub fn posture(&self) -> Option<Posture> {
        match self {
            AnchorLabel::StandStill => Some(Posture::Standing),
            AnchorLabel::Sit => Some(Posture::Sitting),
            AnchorLabel::LieDown | AnchorLabel::SleepPosture => Some(Posture::Lying),
            _ => None,
        }
    }
}

/// Quality assessment of a captured anchor (from the enrollment quality gate).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnchorQuality {
    /// Median amplitude z-score vs the empty-room baseline (presence strength).
    pub presence_z: f32,
    /// Fraction of frames flagged as motion.
    pub motion_rate: f32,
    /// Number of frames captured.
    pub frames: u32,
    /// Whether the anchor passed the gate.
    pub accepted: bool,
}

/// A captured, accepted anchor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    /// Which anchor in the sequence.
    pub label: AnchorLabel,
    /// Capture time (unix seconds).
    pub captured_at_unix_s: i64,
    /// Quality metrics.
    pub quality: AnchorQuality,
}

/// Event log entry for an enrollment session (event sourcing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnrollmentEvent {
    /// Session opened.
    Started {
        /// Room scope.
        room_id: String,
        /// Baseline id the enrollment layers on.
        baseline_id: String,
        /// Unix seconds.
        at: i64,
    },
    /// An anchor passed the gate and was accepted.
    AnchorAccepted {
        /// The accepted anchor.
        anchor: Anchor,
    },
    /// Transceiver geometry recorded for the session's nodes (ADR-152 §2.1.1).
    /// Typically appended right after `Started`; a later event supersedes an
    /// earlier one (latest wins), so a geometry correction is an append, not a
    /// rewrite. Sessions persisted before this variant existed replay cleanly —
    /// the variant is additive to the externally-tagged event encoding.
    GeometryRecorded {
        /// Per-node geometry records.
        geometry: Vec<NodeGeometry>,
        /// Unix seconds.
        at: i64,
    },
    /// An anchor failed the gate (re-prompt).
    AnchorRejected {
        /// Which anchor.
        label: AnchorLabel,
        /// Human-readable reason.
        reason: String,
        /// Unix seconds.
        at: i64,
    },
    /// All required anchors accepted.
    Completed {
        /// Unix seconds.
        at: i64,
    },
}

/// Event-sourced enrollment session for one room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollmentSession {
    /// Room scope.
    pub room_id: String,
    /// Baseline id this session layers on.
    pub baseline_id: String,
    /// Append-only event log.
    pub events: Vec<EnrollmentEvent>,
}

impl EnrollmentSession {
    /// Open a new session.
    pub fn new(room_id: impl Into<String>, baseline_id: impl Into<String>, at: i64) -> Self {
        let room_id = room_id.into();
        let baseline_id = baseline_id.into();
        let mut s = Self {
            room_id: room_id.clone(),
            baseline_id: baseline_id.clone(),
            events: Vec::new(),
        };
        s.events.push(EnrollmentEvent::Started {
            room_id,
            baseline_id,
            at,
        });
        s
    }

    /// Append an event (event sourcing — state is derived, never mutated in place).
    pub fn apply(&mut self, event: EnrollmentEvent) {
        self.events.push(event);
    }

    /// The set of accepted anchors (latest acceptance per label wins).
    pub fn accepted_anchors(&self) -> Vec<Anchor> {
        let mut out: Vec<Anchor> = Vec::new();
        for ev in &self.events {
            if let EnrollmentEvent::AnchorAccepted { anchor } = ev {
                if let Some(slot) = out.iter_mut().find(|a| a.label == anchor.label) {
                    *slot = anchor.clone();
                } else {
                    out.push(anchor.clone());
                }
            }
        }
        out
    }

    /// Record the session's transceiver geometry (ADR-152 §2.1.1) — appends a
    /// [`EnrollmentEvent::GeometryRecorded`] event; the latest recording wins.
    pub fn record_geometry(&mut self, geometry: Vec<NodeGeometry>, at: i64) {
        self.apply(EnrollmentEvent::GeometryRecorded { geometry, at });
    }

    /// The geometry snapshot in effect (latest `GeometryRecorded` event), if
    /// any was recorded. Derived from the event log, never stored separately.
    pub fn geometry(&self) -> Option<&[NodeGeometry]> {
        self.events.iter().rev().find_map(|ev| match ev {
            EnrollmentEvent::GeometryRecorded { geometry, .. } => Some(geometry.as_slice()),
            _ => None,
        })
    }

    /// The next anchor in the canonical sequence not yet accepted, if any.
    pub fn next_anchor(&self) -> Option<AnchorLabel> {
        let accepted = self.accepted_anchors();
        AnchorLabel::SEQUENCE
            .iter()
            .copied()
            .find(|label| !accepted.iter().any(|a| a.label == *label))
    }

    /// `(accepted, total)` progress.
    pub fn progress(&self) -> (usize, usize) {
        (self.accepted_anchors().len(), AnchorLabel::SEQUENCE.len())
    }

    /// Whether every anchor in the sequence has been accepted.
    pub fn is_complete(&self) -> bool {
        self.next_anchor().is_none()
    }

    /// Labels still required.
    pub fn missing(&self) -> Vec<AnchorLabel> {
        let accepted = self.accepted_anchors();
        AnchorLabel::SEQUENCE
            .iter()
            .copied()
            .filter(|label| !accepted.iter().any(|a| a.label == *label))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(label: AnchorLabel) -> Anchor {
        Anchor {
            label,
            captured_at_unix_s: 1,
            quality: AnchorQuality {
                presence_z: 3.0,
                motion_rate: 0.1,
                frames: 400,
                accepted: true,
            },
        }
    }

    #[test]
    fn label_roundtrip() {
        for l in AnchorLabel::SEQUENCE {
            assert_eq!(AnchorLabel::from_str(l.as_str()), Some(l));
        }
        assert_eq!(AnchorLabel::from_str("nope"), None);
    }

    #[test]
    fn label_serde_is_snake_case_matching_as_str() {
        // The JSON wire format must equal as_str() (the documented contract).
        for l in AnchorLabel::SEQUENCE {
            let json = serde_json::to_string(&l).unwrap();
            assert_eq!(json, format!("\"{}\"", l.as_str()));
            let back: AnchorLabel = serde_json::from_str(&json).unwrap();
            assert_eq!(back, l);
        }
    }

    #[test]
    fn sequence_order_and_next() {
        let mut s = EnrollmentSession::new("living-room", "base-1", 0);
        assert_eq!(s.next_anchor(), Some(AnchorLabel::Empty));
        s.apply(EnrollmentEvent::AnchorAccepted {
            anchor: anchor(AnchorLabel::Empty),
        });
        assert_eq!(s.next_anchor(), Some(AnchorLabel::StandStill));
        assert_eq!(s.progress(), (1, 8));
        assert!(!s.is_complete());
    }

    #[test]
    fn completion_and_missing() {
        let mut s = EnrollmentSession::new("r", "b", 0);
        for l in AnchorLabel::SEQUENCE {
            s.apply(EnrollmentEvent::AnchorAccepted { anchor: anchor(l) });
        }
        assert!(s.is_complete());
        assert!(s.missing().is_empty());
        assert_eq!(s.progress(), (8, 8));
    }

    #[test]
    fn reaccept_replaces_not_duplicates() {
        let mut s = EnrollmentSession::new("r", "b", 0);
        s.apply(EnrollmentEvent::AnchorAccepted {
            anchor: anchor(AnchorLabel::Sit),
        });
        s.apply(EnrollmentEvent::AnchorAccepted {
            anchor: anchor(AnchorLabel::Sit),
        });
        assert_eq!(
            s.accepted_anchors()
                .iter()
                .filter(|a| a.label == AnchorLabel::Sit)
                .count(),
            1
        );
    }

    #[test]
    fn geometry_recorded_latest_wins_and_roundtrips() {
        let mut s = EnrollmentSession::new("r", "b", 0);
        assert!(s.geometry().is_none(), "no geometry before recording");

        s.record_geometry(vec![NodeGeometry::unknown(1)], 5);
        let corrected = vec![
            NodeGeometry::new(1, "tape-measure").with_position(0.0, 0.0, 1.0),
            NodeGeometry::new(2, "tape-measure")
                .with_position(3.0, 0.0, 1.0)
                .with_distance(1, 3.0),
        ];
        s.record_geometry(corrected.clone(), 10);

        // Latest recording wins, derived from the event log.
        assert_eq!(s.geometry(), Some(corrected.as_slice()));

        // The whole session (incl. geometry events) survives persistence.
        let json = serde_json::to_string(&s).unwrap();
        let back: EnrollmentSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.geometry(), Some(corrected.as_slice()));
        assert_eq!(back.events.len(), s.events.len());
    }

    /// Sessions persisted BEFORE the GeometryRecorded variant existed must
    /// deserialize cleanly and report no geometry (ADR-152 schema-compat rule).
    #[test]
    fn pre_geometry_session_json_loads() {
        let old_json = r#"{
            "room_id": "r",
            "baseline_id": "b",
            "events": [
                {"Started": {"room_id": "r", "baseline_id": "b", "at": 0}},
                {"AnchorRejected": {"label": "sit", "reason": "no person", "at": 1}}
            ]
        }"#;
        let s: EnrollmentSession = serde_json::from_str(old_json).unwrap();
        assert!(s.geometry().is_none());
        assert_eq!(s.next_anchor(), Some(AnchorLabel::Empty));
    }

    #[test]
    fn posture_mapping() {
        assert_eq!(AnchorLabel::StandStill.posture(), Some(Posture::Standing));
        assert_eq!(AnchorLabel::LieDown.posture(), Some(Posture::Lying));
        assert_eq!(AnchorLabel::SmallMove.posture(), None);
        assert!(!AnchorLabel::SmallMove.expects_still());
        assert!(!AnchorLabel::Empty.expects_presence());
    }
}

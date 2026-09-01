//! Deterministic software acceptance. All inputs are SYNTHETIC/L0; this proves
//! contract and fusion behavior, not the ADR-331 hardware reproduction gate.

use ruview_nlos::protocol::{EvidenceLevel, FrameSource};
use ruview_nlos::{SyntheticScene, TrackEnvelope};

#[test]
fn golden_track_contract_round_trips_byte_semantics() {
    let fixture = include_str!("fixtures/track_synthetic.json");
    let envelope: TrackEnvelope = serde_json::from_str(fixture).unwrap();
    envelope.validate().unwrap();
    let encoded = serde_json::to_string(&envelope).unwrap();
    let decoded: TrackEnvelope = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn synthetic_fusion_reduces_lost_tracks_by_at_least_25_percent() {
    let report = SyntheticScene::benchmark(120, 512);
    assert_eq!(report.evidence, "SYNTHETIC_L0");
    assert!(!report.hardware_reproduction_gate_passed);
    assert!(
        report.lost_track_reduction_percent >= 25.0,
        "SYNTHETIC architecture gate: LiDAR-only lost rate={}, fused lost rate={}, reduction={}%; this is not hardware evidence",
        report.lidar_only_lost_track_rate,
        report.fused_lost_track_rate,
        report.lost_track_reduction_percent,
    );
}

#[test]
fn generated_contract_can_never_alias_to_live_evidence() {
    let mut scene = SyntheticScene::default();
    let frame = scene.frame(None, 0.0, 1);
    assert_eq!(frame.source, FrameSource::Synthetic);
    assert_eq!(frame.evidence_level, EvidenceLevel::L0Synthetic);
    assert_eq!(frame.calibration_hash, "0".repeat(64));
    assert_eq!(frame.provenance.transport, "replay");
    frame.validate().unwrap();
}

//! ADR-307 scenario tests: continuity, no-swap, spawn/expire, ambiguity,
//! id opacity, and determinism. All fixtures are synthetic and in-code; time is
//! injected (no wall clock); no randomness.

use ruview_ontology::{Container, EvidenceLevel, SpaceId};
use ruview_track::*;

fn space(id: &str) -> Container {
    Container::Space {
        id: SpaceId::new(id).unwrap(),
    }
}

fn feat(v: &[f64]) -> CoarseFeature {
    CoarseFeature::quantize(v).unwrap()
}

/// kitchen ▸ hallway ▸ bedroom, wired as a corridor.
fn corridor() -> Topology {
    let mut t = Topology::new();
    t.connect(&space("kitchen"), &space("hallway"));
    t.connect(&space("hallway"), &space("bedroom"));
    t
}

#[test]
fn single_target_continuity_across_zones() {
    let mut mgr = TrackManager::new(TrackerConfig::default(), corridor());
    let f = feat(&[0.2, 0.6, 0.3]);

    let o1 = mgr
        .ingest(Detection::new(space("kitchen"), [1.0, 1.0], f.clone(), 1_000).unwrap())
        .unwrap();
    let o2 = mgr
        .ingest(Detection::new(space("hallway"), [1.3, 1.1], f.clone(), 1_500).unwrap())
        .unwrap();
    let o3 = mgr
        .ingest(Detection::new(space("bedroom"), [1.6, 1.0], f, 2_000).unwrap())
        .unwrap();

    // One persistent entity, one pseudonym across all three rooms.
    assert_eq!(mgr.len(), 1);
    assert_eq!(o1.person, o2.person);
    assert_eq!(o2.person, o3.person);
    assert!(matches!(o2.association, Association::Matched { .. }));
    assert!(matches!(o3.association, Association::Matched { .. }));

    // History reads kitchen -> hallway -> bedroom.
    let traj = mgr.trajectory(&o1.track).unwrap();
    assert_eq!(
        traj,
        vec![space("kitchen"), space("hallway"), space("bedroom")]
    );
}

#[test]
fn topology_blocks_non_adjacent_handoff() {
    // kitchen and bedroom are NOT adjacent (no hallway hop recorded here).
    let mut t = Topology::new();
    t.connect(&space("kitchen"), &space("hallway"));
    let mut mgr = TrackManager::new(TrackerConfig::default(), t);
    let f = feat(&[0.2, 0.6, 0.3]);

    let a = mgr
        .ingest(Detection::new(space("kitchen"), [1.0, 1.0], f.clone(), 1_000).unwrap())
        .unwrap();
    let b = mgr
        .ingest(Detection::new(space("bedroom"), [1.0, 1.0], f, 1_200).unwrap())
        .unwrap();

    // Non-adjacent: a fresh pseudonym rather than a false join.
    assert_ne!(a.person, b.person);
    assert!(matches!(
        b.association,
        Association::Unknown {
            reason: UnknownReason::TopologyBlocked,
            ..
        }
    ));
    assert_eq!(mgr.len(), 2);
}

#[test]
fn two_targets_no_swap_under_separation() {
    let mut mgr = TrackManager::with_defaults(); // single space, empty topology
    let fa = feat(&[0.1, 0.1, 0.1]);
    let fb = feat(&[0.9, 0.9, 0.9]);

    // Frame 1: two well-separated detections spawn two tracks.
    let f1 = mgr
        .ingest_frame(&[
            Detection::new(space("kitchen"), [0.0, 0.0], fa.clone(), 1_000).unwrap(),
            Detection::new(space("kitchen"), [10.0, 0.0], fb.clone(), 1_000).unwrap(),
        ])
        .unwrap();
    let (pa, pb) = (f1[0].person.clone(), f1[1].person.clone());
    let (ta, tb) = (f1[0].track.clone(), f1[1].track.clone());
    assert_ne!(pa, pb);

    // Several frames of parallel motion, staying separated.
    for k in 1..=5 {
        let t = 1_000 + k * 200;
        let x = k as f64 * 0.1;
        let out = mgr
            .ingest_frame(&[
                Detection::new(space("kitchen"), [x, 0.0], fa.clone(), t).unwrap(),
                Detection::new(space("kitchen"), [10.0 + x, 0.0], fb.clone(), t).unwrap(),
            ])
            .unwrap();
        // Each detection stays with its own original track — no swap.
        assert_eq!(out[0].track, ta);
        assert_eq!(out[1].track, tb);
        assert_eq!(out[0].person, pa);
        assert_eq!(out[1].person, pb);
    }
    assert_eq!(mgr.len(), 2);
}

#[test]
fn track_spawn_and_expire() {
    let mut mgr = TrackManager::with_defaults();
    let f = feat(&[0.5]);
    let out = mgr
        .ingest(Detection::new(space("kitchen"), [0.0, 0.0], f.clone(), 1_000).unwrap())
        .unwrap();
    assert_eq!(mgr.len(), 1);
    assert!(matches!(
        out.association,
        Association::Unknown {
            reason: UnknownReason::NoCandidate,
            ..
        }
    ));
    assert_eq!(mgr.state(&out.track), Some(TrackState::Tentative));

    // A second hit confirms the track (default confirm_after = 2).
    let out2 = mgr
        .ingest(Detection::new(space("kitchen"), [0.1, 0.0], f, 1_100).unwrap())
        .unwrap();
    assert_eq!(out2.track, out.track);
    assert_eq!(mgr.state(&out.track), Some(TrackState::Active));

    // Within the horizon: idle-but-alive (lost), not expired.
    let horizon = mgr.config().max_coast_ms;
    let expired = mgr.tick(1_100 + horizon);
    assert!(expired.is_empty());
    assert_eq!(mgr.len(), 1);
    assert_eq!(mgr.state(&out.track), Some(TrackState::Lost));

    // Past the horizon: expired and dropped.
    let expired = mgr.tick(1_100 + horizon + 1);
    assert_eq!(expired, vec![out.track.clone()]);
    assert_eq!(mgr.len(), 0);
    assert_eq!(mgr.state(&out.track), None);
}

#[test]
fn beyond_horizon_mints_fresh_pseudonym() {
    let mut mgr = TrackManager::with_defaults();
    let f = feat(&[0.5, 0.5]);
    let a = mgr
        .ingest(Detection::new(space("kitchen"), [0.0, 0.0], f.clone(), 1_000).unwrap())
        .unwrap();
    let horizon = mgr.config().max_coast_ms;
    // Same place and feature, but long after the horizon: under-link, do not join.
    let b = mgr
        .ingest(Detection::new(space("kitchen"), [0.0, 0.0], f, 1_000 + horizon + 500).unwrap())
        .unwrap();
    assert_ne!(a.person, b.person);
    assert!(matches!(b.association, Association::Unknown { .. }));
}

#[test]
fn ambiguous_detection_stays_tentative_not_misassigned() {
    // Confirm immediately so the two seed tracks are active and equal-footing.
    let cfg = TrackerConfig {
        confirm_after: 1,
        ..TrackerConfig::default()
    };
    let mut mgr = TrackManager::new(cfg, Topology::new());
    let f = feat(&[0.5, 0.5]);

    // Two tracks with identical features, symmetric about the origin. Their
    // separation (3.0) exceeds the position gate (2.0) so they stay distinct,
    // yet each sits within the gate of the midpoint.
    let a = mgr
        .ingest(Detection::new(space("kitchen"), [-1.5, 0.0], f.clone(), 1_000).unwrap())
        .unwrap();
    let b = mgr
        .ingest(Detection::new(space("kitchen"), [1.5, 0.0], f.clone(), 1_000).unwrap())
        .unwrap();
    assert_eq!(mgr.len(), 2);

    // A detection exactly between them, same feature: equidistant → ambiguous.
    let mid = mgr
        .ingest(Detection::new(space("kitchen"), [0.0, 0.0], f, 1_100).unwrap())
        .unwrap();

    assert!(matches!(
        mid.association,
        Association::Unknown {
            reason: UnknownReason::Ambiguous,
            ..
        }
    ));
    // It was NOT attached to either existing track — a third pseudonym.
    assert_ne!(mid.person, a.person);
    assert_ne!(mid.person, b.person);
    assert_eq!(mgr.len(), 3);
}

#[test]
fn pseudonyms_are_opaque_and_rotatable_with_no_civil_fields() {
    let mut mgr = TrackManager::with_defaults();
    let out = mgr
        .ingest(Detection::new(space("kitchen"), [0.0, 0.0], feat(&[0.3]), 1_000).unwrap())
        .unwrap();

    // Opaque synthetic form, no civil identifier embedded.
    let pid = out.person.as_str().to_string();
    assert!(pid.starts_with("person_"));

    // Rotate: new opaque id, same track and history preserved.
    let before = mgr.trajectory(&out.track).unwrap();
    let rotated = mgr.rotate_pseudonym(&out.track).unwrap();
    assert_ne!(rotated.as_str(), pid);
    assert!(rotated.as_str().starts_with("person_"));
    assert_eq!(mgr.person_of(&out.track), Some(&rotated));
    assert_eq!(mgr.trajectory(&out.track).unwrap(), before);

    // The emitted canonical Person/Track carry no civil-identity field.
    let person = mgr.to_person(&out.track).unwrap();
    let track = mgr.to_track(&out.track).unwrap();
    let pj = serde_json::to_string(&person).unwrap();
    let tj = serde_json::to_string(&track).unwrap();
    for forbidden in ["name", "mac", "email", "phone", "account", "ssid"] {
        assert!(!pj.contains(forbidden), "person leaked `{forbidden}`: {pj}");
        assert!(!tj.contains(forbidden), "track leaked `{forbidden}`: {tj}");
    }
    // Tentative track is floored to L0; feature never appears in the node.
    assert_eq!(person.evidence_level, EvidenceLevel::L0);
    assert!(!pj.contains("bins"));
}

#[test]
fn ingest_is_deterministic() {
    fn run() -> Vec<(String, String)> {
        let mut mgr = TrackManager::new(TrackerConfig::default(), corridor());
        let script = [
            (space("kitchen"), [0.0, 0.0], vec![0.1, 0.2], 1_000i64),
            (space("kitchen"), [5.0, 0.0], vec![0.8, 0.9], 1_000),
            (space("hallway"), [0.3, 0.1], vec![0.1, 0.2], 1_400),
            (space("hallway"), [5.3, 0.1], vec![0.8, 0.9], 1_400),
            (space("bedroom"), [0.6, 0.0], vec![0.1, 0.2], 1_800),
        ];
        let mut trace = Vec::new();
        for (c, p, v, t) in script {
            let o = mgr
                .ingest(Detection::new(c, p, feat(&v), t).unwrap())
                .unwrap();
            let kind = match o.association {
                Association::Matched { .. } => "matched",
                Association::Unknown { .. } => "unknown",
            };
            trace.push((o.person.as_str().to_string(), kind.to_string()));
        }
        trace
    }
    assert_eq!(run(), run());
}

#[test]
fn malformed_position_is_a_boundary_error_not_unknown() {
    // NaN position is rejected at the boundary — distinct from association UNKNOWN.
    let err = Detection::new(space("kitchen"), [f64::NAN, 0.0], feat(&[0.5]), 1_000);
    assert_eq!(err.unwrap_err(), TrackError::NonFinitePosition);
}

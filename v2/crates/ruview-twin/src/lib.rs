//! # `ruview-twin` — a digital RF twin (ADR-312, ADR-297 phase 3)
//!
//! **SYNTHETIC / L0 — a simulation scaffold, not a measurement system.**
//!
//! This crate is a *research-forward primitive*: a persistent, versioned,
//! per-deployment **model** of an RF environment. A twin **predicts** an expected
//! observable; it never **measures** one. Every distribution it produces and any
//! propagation it simulates is a model at evidence level `L0` (ADR-282),
//! labelled `SYNTHETIC`. Nothing in this crate is a hardware, `MEASURED`, or
//! accuracy claim, and it asserts **no** detection-accuracy number (ADR-312
//! evidence discipline). Following ADR-297 rule 1, *insufficient information* is
//! a first-class value ([`ExpectedDistribution::Unknown`] /
//! [`LinkDeltaStatus::Unknown`]), never an error and never a confident default.
//!
//! ## What the twin holds
//!
//! - **Radio node positions** in coarse metric coordinates ([`RadioNode`],
//!   [`Point3`]).
//! - **Geometry references** into the canonical ontology ([`SpaceId`],
//!   [`Container`]) — the twin *annotates* the ADR-303 scene, it does not invent
//!   a second geometry.
//! - A **simple documented propagation model** — log-distance path loss with
//!   optional wall attenuation ([`PropagationParams`], [`crate::predict`]),
//!   clearly a SYNTHETIC model, not real RF.
//! - **Recorded multipath / calibration state** ([`MultipathRecord`],
//!   [`RfTwin::calibration_version`]).
//! - An **[`ExpectedDistribution`] per link** — the mean/variance of an
//!   observable under the twin.
//!
//! ## The load-bearing operation
//!
//! [`RfTwin::delta`] compares a supplied observation set to the twin's
//! predictions and returns a typed [`TwinDelta`] with an overall magnitude and
//! *which* links deviate, each scored against the twin's own modelled variance.
//! A physical change becomes a *measurable delta against the twin* — a candidate
//! change to corroborate, never a confident detection.
//!
//! ## Determinism
//!
//! Everything is deterministic. Synthetic scenes are varied by an explicit
//! [`seed`](DeploymentDescription::seed) via [`synthetic_deployment`]; there is
//! no wall-clock, no unseeded randomness, and no I/O anywhere in the crate.
//!
//! ```
//! use ruview_twin::*;
//!
//! // A reproducible synthetic deployment, then its zero-delta reference.
//! let twin = RfTwin::build(synthetic_deployment(7)).unwrap();
//! let observed = ObservationSet::from_twin_prediction(&twin);
//! let delta = twin.delta(&observed);
//! assert!(delta.is_zero()); // observation matches prediction ⇒ zero delta
//! assert_eq!(twin.evidence_level, ruview_ontology::EvidenceLevel::L0);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod delta;
mod predict;
mod twin;

pub use delta::{
    compute_delta, compute_delta_with_threshold, LinkDelta, LinkDeltaStatus, LinkObservation,
    ObservationSet, TwinDelta, DEFAULT_SIGNIFICANCE_THRESHOLD,
};
pub use predict::{
    path_loss_db, predict_all, predict_link, wall_attenuation_db, ExpectedDistribution, Observable,
    UnknownReason,
};
pub use twin::{
    DeploymentDescription, LinkId, MultipathRecord, Point3, PropagationParams, RadioNode, RfTwin,
    TwinError, VersionEvent, Wall, MAX_NODES, MAX_WALLS,
};

// Re-export the canonical ontology vocabulary the twin references, so consumers
// speak one semantics (ADR-297 rule 3, ADR-303).
pub use ruview_ontology::{Container, EvidenceLevel, SensorId, SpaceId};

impl RfTwin {
    /// Predict the expected distribution for a link. See [`predict_link`].
    #[must_use]
    pub fn predict(&self, link: &LinkId) -> ExpectedDistribution {
        predict_link(self, link)
    }

    /// Compute the delta of a supplied observation set against this twin, using
    /// the default significance threshold. See [`compute_delta`].
    #[must_use]
    pub fn delta(&self, observed: &ObservationSet) -> TwinDelta {
        compute_delta(self, observed)
    }
}

/// Build a deterministic **SYNTHETIC** deployment from an explicit `seed`.
///
/// Four radios are placed in a `5 m × 4 m` room with one interior wall. Node
/// positions are jittered by a seeded `splitmix64` stream so distinct seeds give
/// distinct-but-reproducible scenes; the same seed always yields the same scene.
/// This is a simulation fixture, not a model of any real room.
#[must_use]
pub fn synthetic_deployment(seed: u64) -> DeploymentDescription {
    let mut state = seed;
    // Deterministic jitter helper in [-0.5, 0.5] metres.
    let jitter = |s: &mut u64| -> f64 { splitmix64_unit(s) - 0.5 };

    let base = [(0.5, 0.5), (4.5, 0.5), (4.5, 3.5), (0.5, 3.5)];
    let nodes: Vec<RadioNode> = base
        .iter()
        .enumerate()
        .map(|(i, (bx, by))| {
            let x = (bx + jitter(&mut state)).clamp(0.0, 5.0);
            let y = (by + jitter(&mut state)).clamp(0.0, 4.0);
            RadioNode {
                id: SensorId::new(format!("node-{i}")).expect("static id is valid"),
                position: Point3::new(x, y, 1.0),
                located_in: Container::Space {
                    id: SpaceId::new(format!("space-{seed}")).expect("static id is valid"),
                },
                tx_power_dbm: 20.0,
            }
        })
        .collect();

    let walls = vec![Wall {
        id: "interior-wall".to_string(),
        a: (2.5, 0.0),
        b: (2.5, 4.0),
        attenuation_db: 6.0,
    }];

    DeploymentDescription {
        space: SpaceId::new(format!("space-{seed}")).expect("static id is valid"),
        nodes,
        walls,
        params: PropagationParams::default_indoor(),
        multipath: Vec::new(),
        calibration_version: "synthetic-cal-v0".to_string(),
        seed,
    }
}

/// One `splitmix64` step mapped to a unit `f64` in `[0, 1)`. Deterministic; the
/// only source of scene variation in the crate.
fn splitmix64_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 53 bits → [0, 1).
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensor(id: &str) -> SensorId {
        SensorId::new(id).unwrap()
    }

    #[test]
    fn expected_distribution_is_deterministic() {
        // Same seed ⇒ identical twin ⇒ identical predictions, exactly.
        let a = RfTwin::build(synthetic_deployment(42)).unwrap();
        let b = RfTwin::build(synthetic_deployment(42)).unwrap();
        assert_eq!(a, b);

        for link in a.links() {
            let da = a.predict(&link);
            let db = b.predict(&link);
            assert_eq!(da, db);
            // Every link in this fixture is predictable (SYNTHETIC/L0).
            let (mean, var) = da.known().expect("known distribution");
            assert!(mean.is_finite());
            assert!(var > 0.0);
        }

        // Distinct seeds give distinct-but-reproducible scenes.
        let c = RfTwin::build(synthetic_deployment(43)).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn zero_delta_when_observation_matches_prediction() {
        let twin = RfTwin::build(synthetic_deployment(1)).unwrap();
        let observed = ObservationSet::from_twin_prediction(&twin);
        let delta = twin.delta(&observed);

        assert!(delta.is_zero());
        assert_eq!(delta.total_magnitude, 0.0);
        assert!(delta.deviating_links.is_empty());
        assert!(delta.unknown_links.is_empty());
        assert_eq!(delta.baseline_version, 1);
        // Every per-link deviation is exactly zero.
        for ld in &delta.per_link {
            match &ld.status {
                LinkDeltaStatus::Evaluated { deviation, significance, .. } => {
                    assert_eq!(*deviation, 0.0);
                    assert_eq!(*significance, Some(0.0));
                }
                LinkDeltaStatus::Unknown { .. } => panic!("unexpected unknown link"),
            }
        }
    }

    #[test]
    fn delta_is_nonzero_and_localized_to_a_moved_node() {
        // Baseline twin and its self-consistent observation set.
        let twin = RfTwin::build(synthetic_deployment(2)).unwrap();
        let baseline_obs = ObservationSet::from_twin_prediction(&twin);

        // Build a moved-node world: shift exactly one node, predict from it, and
        // treat those predictions as the "observed" set against the baseline.
        let mut moved = synthetic_deployment(2);
        let moved_id = moved.nodes[0].id.clone();
        moved.nodes[0].position.x += 2.0; // a clear physical relocation
        let moved_twin = RfTwin::build(moved).unwrap();
        let observed = ObservationSet::from_twin_prediction(&moved_twin);

        let delta = twin.delta(&observed);

        // A physical change produces a non-zero, significant delta.
        assert!(delta.total_magnitude > 0.0);
        assert!(!delta.deviating_links.is_empty());
        assert!(delta.max_significance >= delta.significance_threshold);

        // The change is localized: every deviating link touches the moved node,
        // and links not touching it match the baseline exactly.
        for link in &delta.deviating_links {
            assert!(link.a == moved_id || link.b == moved_id, "deviation off the moved node");
        }
        for ld in &delta.per_link {
            let touches_moved = ld.link.a == moved_id || ld.link.b == moved_id;
            if let LinkDeltaStatus::Evaluated { deviation, .. } = &ld.status {
                if !touches_moved {
                    assert_eq!(*deviation, 0.0, "untouched link should not deviate");
                }
            }
        }

        // Sanity: the untouched baseline observations still yield zero delta.
        assert!(twin.delta(&baseline_obs).is_zero());
    }

    #[test]
    fn delta_is_localized_to_a_new_reflector() {
        let twin = RfTwin::build(synthetic_deployment(3)).unwrap();

        // Add a new reflector that crosses exactly the node-0 ↔ node-1 path
        // (both near y≈0.5) without crossing the far links.
        let mut with_reflector = synthetic_deployment(3);
        let n0 = with_reflector.nodes[0].id.clone();
        let n1 = with_reflector.nodes[1].id.clone();
        let (x0, _) = with_reflector.nodes[0].position.xy();
        let (x1, _) = with_reflector.nodes[1].position.xy();
        let mid_x = (x0 + x1) / 2.0;
        with_reflector.walls.push(Wall {
            id: "new-reflector".into(),
            a: (mid_x, 0.0),
            b: (mid_x, 1.2),
            attenuation_db: 12.0,
        });
        let reflector_twin = RfTwin::build(with_reflector).unwrap();
        let observed = ObservationSet::from_twin_prediction(&reflector_twin);

        let delta = twin.delta(&observed);
        assert!(delta.total_magnitude > 0.0);
        let target = LinkId::new(n0, n1);
        // The n0-n1 link deviates; it is the crossed path.
        let target_delta = delta
            .per_link
            .iter()
            .find(|ld| ld.link == target)
            .expect("target link present");
        match &target_delta.status {
            LinkDeltaStatus::Evaluated { deviation, .. } => assert!(deviation.abs() > 0.0),
            LinkDeltaStatus::Unknown { .. } => panic!("target should be evaluable"),
        }
    }

    #[test]
    fn unknown_is_first_class_not_an_error() {
        let twin = RfTwin::build(synthetic_deployment(5)).unwrap();

        // Predicting a link to a node that does not exist ⇒ UNKNOWN, not panic.
        let ghost = LinkId::new(sensor("node-0"), sensor("ghost"));
        assert!(matches!(
            twin.predict(&ghost),
            ExpectedDistribution::Unknown { reason: UnknownReason::MissingNode }
        ));

        // Observing an out-of-twin link surfaces as an unknown link in the delta.
        let observed = ObservationSet::new().with(ghost.clone(), -50.0);
        let delta = twin.delta(&observed);
        assert_eq!(delta.unknown_links, vec![ghost]);
        assert_eq!(delta.total_magnitude, 0.0);
        assert!(delta.deviating_links.is_empty());

        // A self-link is UNKNOWN too, never a divide-by-zero.
        let self_link = LinkId::new(sensor("node-0"), sensor("node-0"));
        assert!(matches!(
            twin.predict(&self_link),
            ExpectedDistribution::Unknown { reason: UnknownReason::SelfLink }
        ));
    }

    #[test]
    fn boundary_validation_rejects_malformed_input_without_panic() {
        // Non-finite coordinate.
        let mut d = synthetic_deployment(9);
        d.nodes[0].position.x = f64::NAN;
        assert!(matches!(
            RfTwin::build(d),
            Err(TwinError::NonFiniteCoordinate { .. })
        ));

        // Duplicate node id.
        let mut d = synthetic_deployment(9);
        let dup = d.nodes[0].id.clone();
        d.nodes[1].id = dup;
        assert!(matches!(RfTwin::build(d), Err(TwinError::DuplicateNode { .. })));

        // Invalid propagation parameter.
        let mut d = synthetic_deployment(9);
        d.params.path_loss_exponent = 0.0;
        assert!(matches!(
            RfTwin::build(d),
            Err(TwinError::InvalidParameter { .. })
        ));

        // Too many nodes (bounded allocation). Construct a minimal over-limit
        // description directly to avoid allocating a huge scene twice.
        let mut nodes = Vec::new();
        for i in 0..(MAX_NODES + 1) {
            nodes.push(RadioNode {
                id: sensor(&format!("n{i}")),
                position: Point3::new(0.0, 0.0, 0.0),
                located_in: Container::Space { id: SpaceId::new("s").unwrap() },
                tx_power_dbm: 20.0,
            });
        }
        let over = DeploymentDescription {
            space: SpaceId::new("s").unwrap(),
            nodes,
            walls: Vec::new(),
            params: PropagationParams::default_indoor(),
            multipath: Vec::new(),
            calibration_version: "v0".into(),
            seed: 0,
        };
        assert!(matches!(RfTwin::build(over), Err(TwinError::TooManyNodes { .. })));
    }

    #[test]
    fn versioning_advances_on_events() {
        let mut twin = RfTwin::build(synthetic_deployment(11)).unwrap();
        assert_eq!(twin.version, 1);
        assert_eq!(twin.advance_version(VersionEvent::Calibration), 2);
        assert_eq!(twin.advance_version(VersionEvent::GeometryEdit), 3);
        assert_eq!(twin.advance_version(VersionEvent::AcceptedChange), 4);
        assert_eq!(twin.version, 4);
    }

    #[test]
    fn serde_round_trip_is_lossless() {
        let mut twin = RfTwin::build(synthetic_deployment(13)).unwrap();
        twin.multipath.push(MultipathRecord {
            link: LinkId::new(sensor("node-0"), sensor("node-1")),
            extra_variance_db2: 9.0,
        });

        let json = serde_json::to_string_pretty(&twin).unwrap();
        let back: RfTwin = serde_json::from_str(&json).unwrap();
        assert_eq!(twin, back);

        // Evidence discipline is on the wire: L0 / SYNTHETIC.
        assert!(json.contains("\"evidence_level\": \"L0\""));

        // The delta result also round-trips.
        let observed = ObservationSet::from_twin_prediction(&twin).with(
            LinkId::new(sensor("node-0"), sensor("node-2")),
            -80.0,
        );
        let delta = twin.delta(&observed);
        let dj = serde_json::to_string(&delta).unwrap();
        let back_delta: TwinDelta = serde_json::from_str(&dj).unwrap();
        assert_eq!(delta, back_delta);
    }
}

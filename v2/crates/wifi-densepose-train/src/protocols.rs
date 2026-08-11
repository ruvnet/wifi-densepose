//! Standard public-benchmark split protocols (ADR-291 §2).
//!
//! The field's documented leakage failure is the window-level random split:
//! adjacent windows cut from one continuous recording are near-identical, so
//! splitting them across train/test inflates accuracy (one dataset's F1
//! collapsed from ~90% to ~22% under subject-disjoint splits — ADR-291
//! §Context). This module expresses the standard leaderboard evaluations as a
//! [`SplitProtocol`] whose assignment is a **pure function of sample metadata
//! plus a seed** — no RNG state, no iteration-order dependence, byte-identical
//! across runs and platforms.
//!
//! - [`SplitProtocol::CrossSubject`] — MM-Fi-style: held-out subjects.
//! - [`SplitProtocol::CrossEnvironment`] — held-out rooms/environments.
//! - [`SplitProtocol::CrossOrientation`] — Widar-style: held-out orientations.
//! - [`SplitProtocol::RandomBaseline`] — window-level random split, kept
//!   **only** as the explicitly leakage-prone comparison point; it makes no
//!   disjointness claim and will normally fail the
//!   [`leakage::LeakageAudit`].
//!
//! Structural verification of a produced split lives in [`leakage`].

pub mod leakage;

use serde::{Deserialize, Serialize};

use crate::error::ProtocolError;

// ---------------------------------------------------------------------------
// SampleMeta
// ---------------------------------------------------------------------------

/// Loader-agnostic per-window metadata consumed by split assignment and the
/// leakage audit. Produced by e.g.
/// [`WidarDataset::sample_meta`](crate::dataset::widar::WidarDataset::sample_meta).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleMeta {
    /// Subject/user id.
    pub subject_id: u32,
    /// Environment/room id (`0` when the dataset tree does not encode one).
    pub environment_id: u32,
    /// Orientation id (Widar face orientation; `0` when unknown).
    pub orientation_id: u32,
    /// Gesture/action id.
    pub gesture_id: u32,
    /// Identifier of the continuous recording this window was cut from.
    /// Windows sharing a `recording_id` are temporally correlated and must
    /// never straddle a train/test boundary.
    pub recording_id: u64,
    /// Window offset within the recording.
    pub window_index: u64,
}

// ---------------------------------------------------------------------------
// SplitProtocol
// ---------------------------------------------------------------------------

/// Which side of a train/test split a sample is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SplitSide {
    /// Training partition.
    Train,
    /// Held-out test partition.
    Test,
}

/// A standard evaluation protocol determining *what* is held out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SplitProtocol {
    /// Hold out whole subjects (MM-Fi cross-subject protocol).
    CrossSubject,
    /// Hold out whole environments/rooms (MM-Fi cross-environment protocol).
    CrossEnvironment,
    /// Hold out whole orientations (Widar3.0 cross-orientation protocol).
    CrossOrientation,
    /// Window-level random split. **Leakage-prone by construction** — kept
    /// only so leaderboard-style numbers can be contrasted against a leaky
    /// baseline; it claims no disjointness and normally fails the audit.
    RandomBaseline,
}

impl SplitProtocol {
    /// Stable lowercase tag for logs/reports.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            SplitProtocol::CrossSubject => "cross-subject",
            SplitProtocol::CrossEnvironment => "cross-environment",
            SplitProtocol::CrossOrientation => "cross-orientation",
            SplitProtocol::RandomBaseline => "random-baseline-leaky",
        }
    }

    /// Disjointness this protocol claims and the audit must verify.
    #[must_use]
    pub fn claims(self) -> leakage::LeakageClaims {
        match self {
            SplitProtocol::CrossSubject => leakage::LeakageClaims {
                subject_disjoint: true,
                environment_disjoint: false,
                orientation_disjoint: false,
            },
            SplitProtocol::CrossEnvironment => leakage::LeakageClaims {
                subject_disjoint: false,
                environment_disjoint: true,
                orientation_disjoint: false,
            },
            SplitProtocol::CrossOrientation => leakage::LeakageClaims {
                subject_disjoint: false,
                environment_disjoint: false,
                orientation_disjoint: true,
            },
            SplitProtocol::RandomBaseline => leakage::LeakageClaims {
                subject_disjoint: false,
                environment_disjoint: false,
                orientation_disjoint: false,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// SplitPlan
// ---------------------------------------------------------------------------

/// A concrete, seeded instantiation of a [`SplitProtocol`].
///
/// [`SplitPlan::assign`] is a pure function: the same `(protocol, seed,
/// test_fraction, meta)` always yields the same side, independent of call
/// order, thread, or platform.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplitPlan {
    /// The evaluation protocol.
    pub protocol: SplitProtocol,
    /// Seed mixed into every assignment hash.
    pub seed: u64,
    /// Target fraction of held-out *units* (subjects / environments /
    /// orientations / windows, per protocol), strictly inside `(0, 1)`.
    pub test_fraction: f64,
}

impl SplitPlan {
    /// Create a plan, validating `test_fraction`.
    ///
    /// # Errors
    ///
    /// [`ProtocolError::InvalidTestFraction`] when the fraction is not finite
    /// or not strictly inside `(0, 1)`.
    pub fn new(
        protocol: SplitProtocol,
        seed: u64,
        test_fraction: f64,
    ) -> Result<Self, ProtocolError> {
        if !test_fraction.is_finite() || test_fraction <= 0.0 || test_fraction >= 1.0 {
            return Err(ProtocolError::InvalidTestFraction {
                value: test_fraction,
            });
        }
        Ok(SplitPlan {
            protocol,
            seed,
            test_fraction,
        })
    }

    /// Assign one sample to a side — pure, deterministic, stateless.
    ///
    /// The protocol's held-out *unit* (subject, environment, orientation, or
    /// individual window) is hashed together with a protocol-specific domain
    /// tag and the seed; the unit lands in the test set when its hash falls
    /// below `test_fraction` of the hash space. All windows of one unit
    /// therefore always land on the same side (except under
    /// [`SplitProtocol::RandomBaseline`], which hashes per window — that is
    /// its documented leak).
    #[must_use]
    pub fn assign(&self, meta: &SampleMeta) -> SplitSide {
        // Distinct domain tags keep e.g. subject 3 and orientation 3 from
        // sharing a hash under the same seed.
        const DOMAIN_SUBJECT: u64 = 0x5355424a; // "SUBJ"
        const DOMAIN_ENVIRONMENT: u64 = 0x454e5652; // "ENVR"
        const DOMAIN_ORIENTATION: u64 = 0x4f524e54; // "ORNT"
        const DOMAIN_RANDOM: u64 = 0x524e444d; // "RNDM"

        let unit = match self.protocol {
            SplitProtocol::CrossSubject => {
                mix2(DOMAIN_SUBJECT, meta.subject_id as u64)
            }
            SplitProtocol::CrossEnvironment => {
                mix2(DOMAIN_ENVIRONMENT, meta.environment_id as u64)
            }
            SplitProtocol::CrossOrientation => {
                mix2(DOMAIN_ORIENTATION, meta.orientation_id as u64)
            }
            SplitProtocol::RandomBaseline => mix2(
                mix2(DOMAIN_RANDOM, meta.recording_id),
                meta.window_index,
            ),
        };
        let h = splitmix64(unit ^ splitmix64(self.seed));

        // Integer threshold comparison — no float accumulation, identical on
        // every platform.
        let threshold = (self.test_fraction * (1u128 << 64) as f64) as u128;
        if (h as u128) < threshold {
            SplitSide::Test
        } else {
            SplitSide::Train
        }
    }

    /// Partition metadata into `(train_indices, test_indices)` by
    /// [`Self::assign`], preserving input order within each side.
    #[must_use]
    pub fn partition(&self, metas: &[SampleMeta]) -> (Vec<usize>, Vec<usize>) {
        let mut train = Vec::new();
        let mut test = Vec::new();
        for (i, meta) in metas.iter().enumerate() {
            match self.assign(meta) {
                SplitSide::Train => train.push(i),
                SplitSide::Test => test.push(i),
            }
        }
        (train, test)
    }
}

/// SplitMix64 finalizer — a well-distributed 64-bit mixing function.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Order-sensitive combination of two words through SplitMix64.
fn mix2(a: u64, b: u64) -> u64 {
    splitmix64(splitmix64(a) ^ b.rotate_left(32))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 subjects × 2 environments × 4 orientations, 2 recordings each with
    /// 5 windows — a deterministic synthetic corpus.
    fn corpus() -> Vec<SampleMeta> {
        let mut metas = Vec::new();
        let mut recording = 0u64;
        for subject in 1..=4u32 {
            for environment in 1..=2u32 {
                for orientation in 1..=4u32 {
                    for _ in 0..2 {
                        for window in 0..5u64 {
                            metas.push(SampleMeta {
                                subject_id: subject,
                                environment_id: environment,
                                orientation_id: orientation,
                                gesture_id: 1 + (recording % 6) as u32,
                                recording_id: recording,
                                window_index: window,
                            });
                        }
                        recording += 1;
                    }
                }
            }
        }
        metas
    }

    #[test]
    fn plan_rejects_bad_fractions() {
        for bad in [0.0, 1.0, -0.2, 1.7, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                SplitPlan::new(SplitProtocol::CrossSubject, 1, bad),
                Err(ProtocolError::InvalidTestFraction { .. })
            ));
        }
        assert!(SplitPlan::new(SplitProtocol::CrossSubject, 1, 0.25).is_ok());
    }

    #[test]
    fn assignment_is_deterministic_across_calls_and_order() {
        let metas = corpus();
        let plan = SplitPlan::new(SplitProtocol::CrossSubject, 42, 0.3).unwrap();
        let (tr1, te1) = plan.partition(&metas);
        let (tr2, te2) = plan.partition(&metas);
        assert_eq!(tr1, tr2);
        assert_eq!(te1, te2);

        // Pure per-sample function: reversing iteration order changes nothing.
        let reversed: Vec<SampleMeta> = metas.iter().rev().copied().collect();
        for (meta, rev) in metas.iter().zip(reversed.iter().rev()) {
            assert_eq!(plan.assign(meta), plan.assign(rev));
        }
    }

    #[test]
    fn different_seeds_change_the_split() {
        let metas = corpus();
        let a = SplitPlan::new(SplitProtocol::CrossSubject, 1, 0.5).unwrap();
        let b = SplitPlan::new(SplitProtocol::CrossSubject, 2, 0.5).unwrap();
        // With 4 subjects at 50% some seed pair must differ; these two do —
        // and if the hash ever changes this test flags the compat break.
        let (_, te_a) = a.partition(&metas);
        let (_, te_b) = b.partition(&metas);
        assert_ne!(te_a, te_b, "seeds 1 and 2 should hold out different subjects");
    }

    #[test]
    fn cross_subject_keeps_subjects_whole() {
        let metas = corpus();
        let plan = SplitPlan::new(SplitProtocol::CrossSubject, 7, 0.4).unwrap();
        let mut side_by_subject = std::collections::BTreeMap::new();
        for meta in &metas {
            let side = plan.assign(meta);
            let prev = side_by_subject.insert(meta.subject_id, side);
            if let Some(prev) = prev {
                assert_eq!(prev, side, "subject {} split across sides", meta.subject_id);
            }
        }
    }

    #[test]
    fn cross_environment_keeps_environments_whole() {
        let metas = corpus();
        let plan = SplitPlan::new(SplitProtocol::CrossEnvironment, 11, 0.5).unwrap();
        let mut side_by_env = std::collections::BTreeMap::new();
        for meta in &metas {
            let side = plan.assign(meta);
            if let Some(prev) = side_by_env.insert(meta.environment_id, side) {
                assert_eq!(prev, side);
            }
        }
    }

    #[test]
    fn cross_orientation_keeps_orientations_whole() {
        let metas = corpus();
        let plan = SplitPlan::new(SplitProtocol::CrossOrientation, 13, 0.5).unwrap();
        let mut side_by_orient = std::collections::BTreeMap::new();
        for meta in &metas {
            let side = plan.assign(meta);
            if let Some(prev) = side_by_orient.insert(meta.orientation_id, side) {
                assert_eq!(prev, side);
            }
        }
    }

    #[test]
    fn random_baseline_splits_within_recordings() {
        // The leaky baseline must (for some recording) place windows of the
        // same recording on both sides — that is the leak it demonstrates.
        let metas = corpus();
        let plan = SplitPlan::new(SplitProtocol::RandomBaseline, 3, 0.5).unwrap();
        let mut crossing = false;
        let mut side_by_recording = std::collections::BTreeMap::new();
        for meta in &metas {
            let side = plan.assign(meta);
            if let Some(prev) = side_by_recording.insert(meta.recording_id, side) {
                if prev != side {
                    crossing = true;
                }
            }
        }
        assert!(crossing, "window-level split should cross recordings");
    }

    #[test]
    fn protocol_claims_match_semantics() {
        assert!(SplitProtocol::CrossSubject.claims().subject_disjoint);
        assert!(SplitProtocol::CrossEnvironment.claims().environment_disjoint);
        assert!(SplitProtocol::CrossOrientation.claims().orientation_disjoint);
        let random = SplitProtocol::RandomBaseline.claims();
        assert!(!random.subject_disjoint);
        assert!(!random.environment_disjoint);
        assert!(!random.orientation_disjoint);
    }

    #[test]
    fn tags_are_stable() {
        assert_eq!(SplitProtocol::CrossSubject.tag(), "cross-subject");
        assert_eq!(SplitProtocol::RandomBaseline.tag(), "random-baseline-leaky");
    }
}

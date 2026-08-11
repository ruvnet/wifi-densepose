//! The fusion engine (ADR-308 §2): many observations → one world state.
//!
//! [`FusionEngine::fuse`] groups the input observations by their canonical
//! container and, for each container, combines the usable presence estimates by
//! inverse-variance weighting into one [`ZoneState`]. It is a pure, deterministic
//! function of its inputs: no I/O, no clock, no randomness. The disagreement
//! between sources is measured against their stated uncertainty; when it exceeds
//! the configured threshold the zone resolves to [`UnknownReason::IrreconcilableConflict`]
//! rather than a confident average, and when too few sources cover a zone it
//! resolves to [`UnknownReason::InsufficientCoverage`].

use std::collections::BTreeMap;

use ruview_ontology::{Container, EvidenceLevel};

use crate::estimate::{combine, Estimate};
use crate::observation::PresenceObservation;
use crate::world::{Contribution, Presence, UnknownReason, WorldState, ZoneState};

/// Configuration for the fusion engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionConfig {
    /// Reduced chi-square disagreement threshold. When contributing sources
    /// disagree by more than this (relative to their stated uncertainty), the
    /// zone resolves to UNKNOWN (irreconcilable conflict) instead of a confident
    /// average. The default `9.0` corresponds to roughly a 3-sigma pairwise
    /// disagreement.
    pub conflict_reduced_chi_square: f64,
    /// Minimum number of usable (non-degraded, quantified) observations required
    /// to resolve a zone. Below this the zone is UNKNOWN (insufficient
    /// coverage). Values below `1` are treated as `1`.
    pub min_observations: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            conflict_reduced_chi_square: 9.0,
            min_observations: 1,
        }
    }
}

impl FusionConfig {
    /// The effective minimum observation count (never below `1`).
    fn effective_min(&self) -> usize {
        self.min_observations.max(1)
    }
}

/// A deterministic, uncertainty-aware multimodal fusion engine.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusionEngine {
    config: FusionConfig,
}

impl FusionEngine {
    /// Build an engine with the given configuration.
    #[must_use]
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// The engine's configuration.
    #[must_use]
    pub fn config(&self) -> FusionConfig {
        self.config
    }

    /// Fuse a set of observations into one probabilistic world state.
    ///
    /// Observations are grouped by their canonical container; each group becomes
    /// one [`ZoneState`]. Malformed input never panics: a degraded observation
    /// simply abstains. The result is independent of input order (observations
    /// are combined in a canonical order), so the fusion is deterministic.
    #[must_use]
    pub fn fuse(&self, observations: &[PresenceObservation]) -> WorldState {
        // Group observation indices by a stable container key so the output
        // order is deterministic and independent of input order.
        let mut groups: BTreeMap<(u8, String), Vec<usize>> = BTreeMap::new();
        for (i, obs) in observations.iter().enumerate() {
            let (kind, id) = container_key(&obs.hal.observation.located_in);
            groups.entry((kind, id.to_string())).or_default().push(i);
        }

        let at_unix_ms = observations
            .iter()
            .map(|o| o.hal.observation.at_unix_ms)
            .max()
            .unwrap_or(0);

        let zones = groups
            .into_values()
            .map(|idxs| self.fuse_zone(observations, &idxs))
            .collect();

        WorldState { at_unix_ms, zones }
    }

    /// Fuse the observations that share one container into a single zone state.
    fn fuse_zone(&self, observations: &[PresenceObservation], idxs: &[usize]) -> ZoneState {
        let container = observations[idxs[0]].hal.observation.located_in.clone();

        // Canonical order: sort by observation id so the fused value and the
        // provenance ordering do not depend on input order.
        let mut order = idxs.to_vec();
        order.sort_by(|&a, &b| {
            observations[a]
                .hal
                .observation
                .id
                .as_str()
                .cmp(observations[b].hal.observation.id.as_str())
        });

        // Split into contributing (usable estimate) and abstaining sources.
        let mut contributors: Vec<(usize, Estimate)> = Vec::new();
        for &i in &order {
            if let Some(est) = observations[i].usable_estimate() {
                contributors.push((i, est));
            }
        }

        let mut weight_by_idx: BTreeMap<usize, f64> = BTreeMap::new();
        let (presence, evidence_level) = if contributors.len() < self.config.effective_min() {
            // Not enough usable coverage to resolve this zone.
            (
                Presence::Unknown {
                    reason: UnknownReason::InsufficientCoverage,
                },
                EvidenceLevel::L0,
            )
        } else {
            let estimates: Vec<Estimate> = contributors.iter().map(|(_, e)| *e).collect();
            // Safe: contributors is non-empty here (>= effective_min >= 1).
            let combined = combine(&estimates).expect("non-empty contributor set");

            // Evidence never rises above the weakest contributing input.
            let evidence = contributors
                .iter()
                .map(|(i, _)| observations[*i].hal.evidence_level())
                .min()
                .unwrap_or(EvidenceLevel::L0);

            // Record normalized inverse-variance weights for auditability.
            let precision_sum: f64 = estimates.iter().map(Estimate::precision).sum();
            for (i, e) in &contributors {
                weight_by_idx.insert(*i, e.precision() / precision_sum);
            }

            let presence = if combined.reduced_chi_square > self.config.conflict_reduced_chi_square {
                // Sources disagree beyond their stated uncertainty: refuse to
                // emit a confident average of irreconcilable evidence.
                Presence::Unknown {
                    reason: UnknownReason::IrreconcilableConflict {
                        reduced_chi_square: combined.reduced_chi_square,
                    },
                }
            } else {
                Presence::Estimated {
                    probability: combined.probability,
                    variance: combined.variance,
                }
            };
            (presence, evidence)
        };

        // Per-observation provenance for every observation in the group.
        let contributions = order
            .iter()
            .map(|&i| {
                let obs = &observations[i];
                Contribution {
                    observation: obs.hal.observation.id.clone(),
                    sensor: obs.hal.sensor().clone(),
                    modality: obs.hal.modality.clone(),
                    evidence_level: obs.hal.evidence_level(),
                    estimate: obs.usable_estimate(),
                    weight: weight_by_idx.get(&i).copied().unwrap_or(0.0),
                }
            })
            .collect();

        ZoneState {
            container,
            presence,
            evidence_level,
            contributions,
        }
    }
}

/// A stable ordering/grouping key for a container: a kind discriminant plus its
/// id string. Two containers with the same key are the same container.
fn container_key(container: &Container) -> (u8, &str) {
    match container {
        Container::Space { id } => (0, id.as_str()),
        Container::Zone { id } => (1, id.as_str()),
    }
}

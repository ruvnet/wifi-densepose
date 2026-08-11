//! # `ruview-memory` — long-term spatial memory (ADR-309, ADR-297 phase 3)
//!
//! **SYNTHETIC / L0 — a simulation / model scaffold, not a measurement system.**
//!
//! This crate is a *research-forward primitive*: it learns the **normal physics
//! of a location** so anomalies surface as *deviations from a learned baseline
//! of normality* — without training a detector for every anomaly class. It is a
//! **model**, not a sensor. It predicts what is normal for a place and time and
//! flags a statistically significant delta; it never *measures* anything, and it
//! asserts **no** detection-accuracy, false-positive, or health/safety number
//! (ADR-282 bounded-claims discipline, ADR-309 evidence discipline, CLAUDE.md
//! honesty rule). A flagged deviation is a *candidate change to corroborate*,
//! never a confident detection and never a diagnosis.
//!
//! Following ADR-297 rule 1, *insufficient information* is a first-class value
//! ([`Assessment::Unknown`]), never an error and never a false positive: no
//! anomaly is ever flagged before a baseline exists.
//!
//! ## What "normal" is learned over (ADR-309 §1)
//!
//! Per ADR-303 [`ZoneId`], a [`ZoneBaseline`] accumulates:
//!
//! - **Occupancy periodicity** — a distribution of occupancy by UTC hour-of-day
//!   (the "bedroom usually occupied certain hours" case).
//! - **RF-propagation signature** — a per-link learned normal, *anchored on the
//!   twin's expected distributions* ([`SpatialMemory::seed_zone_propagation_from_twin`])
//!   and refined online (the "chair moved" / "propagation changed" / "new
//!   reflector" cases, which all surface as a per-link RSSI delta).
//! - **Coarse modality signatures** — per-channel learned normal (the "machine's
//!   vibration signature changed" case).
//!
//! Each baseline updates **online** with a documented forgetting factor
//! ([`crate::stat`]); slow legitimate drift is absorbed into the baseline while
//! an abrupt change deviates from it. Every baseline carries the **floor**
//! evidence level of the observations it was learned from and is never presented
//! above them (ADR-282 no-upgrade).
//!
//! ## Anomaly = deviation from learned normal (ADR-309 §3)
//!
//! [`SpatialMemory::observe`] scores a live [`ZoneObservation`] against the
//! applicable learned baseline (matched by zone and hour), returns an
//! [`ObserveOutcome`] of per-channel [`Assessment`]s, and emits an
//! [`AnomalyEvent`] for each channel whose deviation meets the significance
//! gate. Anomalies project to append-only [`ruview_evidence`] records with
//! provenance ([`AnomalyEvent::to_evidence_record`]).
//!
//! ## The four ADR-297 non-negotiable rules, as they bind this crate
//!
//! 1. **UNKNOWN is first-class, never an error.** A channel with fewer than
//!    `min_history` updates is [`Assessment::Unknown`]; `observe` is total and
//!    never panics on malformed input (it returns a typed [`MemoryError`]).
//! 2. **Certificates bind cryptographically.** Out of scope here; a baseline is
//!    keyed by an already-authenticated ADR-303 [`ZoneId`] and its evidence
//!    level is the floor of its source observations.
//! 3. **One canonical semantics.** The memory reuses the canonical
//!    [`ZoneId`]/[`EvidenceLevel`]/[`SemanticProvenance`] vocabulary, the twin's
//!    [`LinkId`]/[`ObservationSet`]/[`ExpectedDistribution`], and the
//!    [`ruview_evidence`] ledger, rather than reinventing per-crate shapes.
//! 4. **Honest evidence.** Every emitted record is **synthetic** (forced L0);
//!    the deviation is carried as `drift` and its significance as `uncertainty`.
//!    No accuracy rate is fabricated.
//!
//! ## Determinism
//!
//! Everything is deterministic: injected time (no wall-clock), no randomness
//! (synthetic scenes vary only by the twin's explicit seed), fixed iteration
//! order, and bounded allocation. The same observations in the same order always
//! produce the same state and the same anomalies.
//!
//! ```
//! use ruview_memory::*;
//! use ruview_ontology::{EvidenceLevel, ZoneId};
//! use ruview_twin::{synthetic_deployment, RfTwin, ObservationSet};
//!
//! let mut mem = SpatialMemory::new(MemoryConfig::default_synthetic()).unwrap();
//! let zone = ZoneId::new("bedroom").unwrap();
//! let twin = RfTwin::build(synthetic_deployment(7)).unwrap();
//!
//! // Anchor the propagation baseline on the twin's expected distributions.
//! mem.seed_zone_propagation_from_twin(zone.clone(), &twin).unwrap();
//!
//! // A snapshot that matches the twin's predictions is normal, not an anomaly.
//! let mut obs = ZoneObservation::new(zone, 0, 1.0, EvidenceLevel::L0);
//! obs.links = ObservationSet::from_twin_prediction(&twin);
//! let outcome = mem.observe(&obs).unwrap();
//! assert!(!outcome.has_anomaly());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod anomaly;
mod baseline;
mod error;
mod stat;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use ruview_evidence::{EvidenceError, EvidenceLedger};
use ruview_twin::{predict_link, ExpectedDistribution, RfTwin};

pub use anomaly::{AnomalyEvent, AnomalyKind, Assessment};
pub use baseline::{
    hour_of_day_utc, LinkStat, MemoryConfig, ZoneBaseline, ZoneObservation, HOURS_PER_DAY,
    MAX_CHANNEL_ID_LEN, MAX_LINKS_PER_ZONE, MAX_MODALITY_CHANNELS, MAX_ZONES,
};
pub use error::MemoryError;
pub use stat::RunningStat;

// Re-export the canonical vocabulary consumers speak (ADR-297 rule 3, ADR-303),
// and the twin's link type the propagation model is keyed by.
pub use ruview_ontology::{EvidenceLevel, SemanticProvenance, ZoneId};
pub use ruview_twin::LinkId;

/// The provenance stamped on every anomaly this scaffold emits.
const PROVENANCE_MODEL: &str = "ruview-memory@0 (SYNTHETIC/L0)";

/// The learned normal physics of every zone, and the operation that scores a
/// live snapshot against it (ADR-309).
///
/// **SYNTHETIC / L0.** A learned model of normality, never a measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialMemory {
    config: MemoryConfig,
    zones: BTreeMap<ZoneId, ZoneBaseline>,
}

impl SpatialMemory {
    /// Build an empty memory with the given configuration, validated at the
    /// boundary.
    ///
    /// # Errors
    /// [`MemoryError::InvalidConfig`] for an out-of-domain configuration.
    pub fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        config.validate()?;
        Ok(Self {
            config,
            zones: BTreeMap::new(),
        })
    }

    /// The active configuration.
    #[must_use]
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// The learned baseline for a zone, if one exists.
    #[must_use]
    pub fn baseline(&self, zone: &ZoneId) -> Option<&ZoneBaseline> {
        self.zones.get(zone)
    }

    /// The number of zones with a learned baseline.
    #[must_use]
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    /// Anchor a zone's propagation baseline on a twin's expected distributions
    /// (ADR-309 §1). Each link the twin can predict seeds a learned statistic
    /// with the twin's modelled mean/variance and a prior weight of
    /// `min_history`, so the propagation model is usable immediately as a prior
    /// and then refined online. The twin prior is SYNTHETIC/L0, so the zone's
    /// evidence floor is lowered to L0.
    ///
    /// Returns the number of links seeded.
    ///
    /// # Errors
    /// [`MemoryError::TooManyZones`] / [`MemoryError::TooManyLinks`] at the
    /// bounded-allocation limits.
    pub fn seed_zone_propagation_from_twin(
        &mut self,
        zone: ZoneId,
        twin: &RfTwin,
    ) -> Result<usize, MemoryError> {
        self.ensure_zone(&zone)?;
        let min_history = self.config.min_history;
        let baseline = self.zones.get_mut(&zone).expect("zone ensured above");
        let mut seeded = 0;
        for link in twin.links() {
            if let ExpectedDistribution::Known { mean, variance, .. } = predict_link(twin, &link) {
                baseline.seed_link(link, mean, variance, min_history)?;
                seeded += 1;
            }
        }
        baseline.record_prior(EvidenceLevel::L0);
        Ok(seeded)
    }

    /// Score a live snapshot against the zone's learned normal, then fold it into
    /// the baseline (ADR-309 §3). Scoring uses the baseline learned *before* this
    /// snapshot, so a flagged anomaly is a genuine deviation and the current
    /// value does not mask itself. Deterministic; never panics on malformed
    /// input.
    ///
    /// # Errors
    /// [`MemoryError`] for non-finite/negative input or a bounded-allocation
    /// limit; the memory is left unchanged when an error is returned.
    pub fn observe(&mut self, obs: &ZoneObservation) -> Result<ObserveOutcome, MemoryError> {
        obs.validate()?;
        self.ensure_zone(&obs.zone)?;

        let hour = hour_of_day_utc(obs.at_unix_ms);
        let cfg = self.config.clone();
        let baseline = self.zones.get_mut(&obs.zone).expect("zone ensured above");

        // Evidence level attributed to any anomaly: the floor of the source
        // observations that formed the baseline, never above the current source.
        let source_level = baseline.evidence_floor().unwrap_or(obs.evidence_level);

        // --- Read phase: copy stats out, score against the prior baseline. ---
        let occ_stat = *baseline.occupancy_hour(hour);
        let occupancy = assess(
            Some(occ_stat),
            obs.occupancy,
            cfg.occupancy_floor_std,
            cfg.min_history,
            cfg.significance_threshold,
        );

        let mut propagation: Vec<(LinkId, Assessment)> =
            Vec::with_capacity(obs.links.observations.len());
        for lo in &obs.links.observations {
            let stat = baseline.link_stat(&lo.link).copied();
            let a = assess(
                stat,
                lo.value,
                cfg.signature_floor_std,
                cfg.min_history,
                cfg.significance_threshold,
            );
            propagation.push((lo.link.clone(), a));
        }

        let mut modality: Vec<(String, Assessment)> = Vec::with_capacity(obs.modality.len());
        for (channel, value) in &obs.modality {
            let stat = baseline.modality_stat(channel).copied();
            let a = assess(
                stat,
                *value,
                cfg.signature_floor_std,
                cfg.min_history,
                cfg.significance_threshold,
            );
            modality.push((channel.clone(), a));
        }

        // --- Collect anomalies from the assessments made above. ---
        let mut anomalies = Vec::new();
        if let Assessment::Anomalous { significance } = occupancy {
            anomalies.push(make_event(
                obs.zone.clone(),
                AnomalyKind::UnusualOccupancyHour,
                obs.at_unix_ms,
                hour,
                format!("hour={hour}"),
                obs.occupancy,
                &occ_stat,
                significance,
                source_level,
            ));
        }
        for (lo, (link, a)) in obs.links.observations.iter().zip(propagation.iter()) {
            if let Assessment::Anomalous { significance } = a {
                let stat = baseline.link_stat(link).copied().unwrap_or_default();
                anomalies.push(make_event(
                    obs.zone.clone(),
                    AnomalyKind::PropagationChange,
                    obs.at_unix_ms,
                    hour,
                    format!("link={}~{}", link.a, link.b),
                    lo.value,
                    &stat,
                    *significance,
                    source_level,
                ));
            }
        }
        for ((channel, value), (_, a)) in obs.modality.iter().zip(modality.iter()) {
            if let Assessment::Anomalous { significance } = a {
                let stat = baseline.modality_stat(channel).copied().unwrap_or_default();
                anomalies.push(make_event(
                    obs.zone.clone(),
                    AnomalyKind::ModalityChange,
                    obs.at_unix_ms,
                    hour,
                    format!("channel={channel}"),
                    *value,
                    &stat,
                    *significance,
                    source_level,
                ));
            }
        }

        // --- Write phase: fold the snapshot into the baseline. ---
        baseline.update_occupancy(hour, obs.occupancy, cfg.forgetting_factor);
        for lo in &obs.links.observations {
            baseline
                .link_stat_mut(&lo.link)?
                .update(lo.value, cfg.forgetting_factor);
        }
        for (channel, value) in &obs.modality {
            baseline
                .modality_stat_mut(channel)?
                .update(*value, cfg.forgetting_factor);
        }
        baseline.record_source(obs.evidence_level);

        Ok(ObserveOutcome {
            zone: obs.zone.clone(),
            hour_of_day: hour,
            occupancy,
            propagation,
            modality,
            anomalies,
        })
    }

    /// Append each anomaly in an outcome to an [`EvidenceLedger`] as a synthetic
    /// record (ADR-309: "emit anomalies as evidence records with provenance").
    /// Returns the ledger sequence assigned to each record, in order.
    ///
    /// # Errors
    /// Propagates [`EvidenceError`] from record construction or a full ledger.
    pub fn record_anomalies(
        &self,
        outcome: &ObserveOutcome,
        ledger: &mut EvidenceLedger,
        timestamp_ns: u64,
    ) -> Result<Vec<u64>, EvidenceError> {
        let mut seqs = Vec::with_capacity(outcome.anomalies.len());
        for ev in &outcome.anomalies {
            let record = ev.to_evidence_record(&self.config.model_version, timestamp_ns)?;
            seqs.push(ledger.append(record)?);
        }
        Ok(seqs)
    }

    /// Ensure a zone has a baseline, respecting the bounded-allocation cap.
    fn ensure_zone(&mut self, zone: &ZoneId) -> Result<(), MemoryError> {
        if !self.zones.contains_key(zone) {
            if self.zones.len() >= MAX_ZONES {
                return Err(MemoryError::TooManyZones { max: MAX_ZONES });
            }
            self.zones.insert(zone.clone(), ZoneBaseline::new());
        }
        Ok(())
    }
}

/// Score one value against its (optional) learned statistic. UNKNOWN when there
/// is no statistic or its history is below `min_history` (ADR-297 rule 1).
fn assess(
    stat: Option<RunningStat>,
    x: f64,
    floor: f64,
    min_history: u32,
    threshold: f64,
) -> Assessment {
    match stat {
        Some(s) if s.count() >= min_history => {
            let significance = s.significance(x, floor);
            if significance >= threshold {
                Assessment::Anomalous { significance }
            } else {
                Assessment::Normal { significance }
            }
        }
        _ => Assessment::Unknown,
    }
}

/// Assemble an [`AnomalyEvent`] from a flagged channel and its baseline stat.
#[allow(clippy::too_many_arguments)]
fn make_event(
    zone: ZoneId,
    kind: AnomalyKind,
    at_unix_ms: i64,
    hour_of_day: u8,
    detail: String,
    observed: f64,
    stat: &RunningStat,
    significance: f64,
    evidence_level: EvidenceLevel,
) -> AnomalyEvent {
    AnomalyEvent {
        zone,
        kind,
        at_unix_ms,
        hour_of_day,
        detail,
        observed,
        baseline_mean: stat.mean(),
        significance,
        baseline_count: stat.count(),
        evidence_level,
        provenance: SemanticProvenance::declared(PROVENANCE_MODEL),
    }
}

/// The result of scoring one snapshot against a zone's learned normal.
///
/// **SYNTHETIC / L0.** Per-channel [`Assessment`]s and the derived
/// [`AnomalyEvent`]s are model-relative summaries, not detections.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObserveOutcome {
    /// The zone scored.
    pub zone: ZoneId,
    /// UTC hour-of-day the snapshot was attributed to.
    pub hour_of_day: u8,
    /// Occupancy assessment for that hour.
    pub occupancy: Assessment,
    /// Per-link propagation assessments, in observation order.
    pub propagation: Vec<(LinkId, Assessment)>,
    /// Per-channel modality assessments, in channel order.
    pub modality: Vec<(String, Assessment)>,
    /// Flagged deviations, derived from the anomalous assessments above.
    pub anomalies: Vec<AnomalyEvent>,
}

impl ObserveOutcome {
    /// True when any channel deviated beyond the significance gate.
    #[must_use]
    pub fn has_anomaly(&self) -> bool {
        !self.anomalies.is_empty()
    }

    /// The distinct anomaly kinds flagged, in first-seen order.
    #[must_use]
    pub fn anomaly_kinds(&self) -> Vec<AnomalyKind> {
        let mut out = Vec::new();
        for ev in &self.anomalies {
            if !out.contains(&ev.kind) {
                out.push(ev.kind);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_evidence::{EvidenceContext, EvidenceLevel as EvLevel, EvidenceLedger};
    use ruview_twin::{synthetic_deployment, LinkId, ObservationSet, RfTwin, SensorId};

    fn cfg() -> MemoryConfig {
        MemoryConfig::default_synthetic()
    }

    fn zone() -> ZoneId {
        ZoneId::new("bedroom").unwrap()
    }

    fn sensor(id: &str) -> SensorId {
        SensorId::new(id).unwrap()
    }

    /// UTC-ms for a given day index and hour, so occupancy buckets are addressable
    /// deterministically without a wall-clock.
    fn ts(day: i64, hour: i64) -> i64 {
        day * 86_400_000 + hour * 3_600_000
    }

    #[test]
    fn no_anomaly_within_normal_variation() {
        let mut mem = SpatialMemory::new(cfg()).unwrap();
        let twin = RfTwin::build(synthetic_deployment(7)).unwrap();
        mem.seed_zone_propagation_from_twin(zone(), &twin).unwrap();
        let predicted = ObservationSet::from_twin_prediction(&twin);

        let mut last = None;
        for day in 0..12 {
            let mut obs = ZoneObservation::new(zone(), ts(day, 14), 1.0, EvidenceLevel::L0);
            obs.links = predicted.clone();
            let outcome = mem.observe(&obs).unwrap();
            // A snapshot matching the twin prediction never flags an anomaly.
            assert!(!outcome.has_anomaly(), "day {day} should be within normal variation");
            // Propagation is assessable from the twin prior and stays Normal.
            assert!(outcome.propagation.iter().all(|(_, a)| !a.is_anomalous()));
            last = Some(outcome);
        }
        // After enough history the occupancy bucket is Normal (not Unknown).
        let last = last.unwrap();
        assert!(matches!(last.occupancy, Assessment::Normal { .. }));
    }

    #[test]
    fn flags_a_propagation_signature_delta() {
        let mut mem = SpatialMemory::new(cfg()).unwrap();
        let twin = RfTwin::build(synthetic_deployment(2)).unwrap();
        mem.seed_zone_propagation_from_twin(zone(), &twin).unwrap();

        // One link observed far from its twin-anchored normal (a "chair moved" /
        // "new reflector" style propagation change).
        let target = twin.links()[0].clone();
        let predicted_mean = mem
            .baseline(&zone())
            .unwrap()
            .link_stat(&target)
            .unwrap()
            .mean();
        let mut obs = ZoneObservation::new(zone(), ts(0, 12), 1.0, EvidenceLevel::L1);
        obs.links = ObservationSet::new().with(target.clone(), predicted_mean + 30.0);

        let outcome = mem.observe(&obs).unwrap();
        assert!(outcome.has_anomaly());
        assert!(outcome.anomaly_kinds().contains(&AnomalyKind::PropagationChange));
        let (_, a) = &outcome.propagation[0];
        assert!(a.is_anomalous());
        // The anomaly's evidence level is the SYNTHETIC/L0 twin-prior floor,
        // never above the source.
        let ev = &outcome.anomalies[0];
        assert_eq!(ev.kind, AnomalyKind::PropagationChange);
        assert_eq!(ev.evidence_level, EvidenceLevel::L0);
        assert!(ev.significance >= cfg().significance_threshold);
    }

    #[test]
    fn flags_off_hours_occupancy() {
        let mut mem = SpatialMemory::new(cfg()).unwrap();

        // Learn that hour 3 (night) is normally unoccupied, and hour 14 (day) is
        // normally occupied.
        for day in 0..12 {
            let night = ZoneObservation::new(zone(), ts(day, 3), 0.0, EvidenceLevel::L2);
            let day_obs = ZoneObservation::new(zone(), ts(day, 14), 1.0, EvidenceLevel::L2);
            assert!(!mem.observe(&night).unwrap().has_anomaly());
            assert!(!mem.observe(&day_obs).unwrap().has_anomaly());
        }

        // Occupied at 3am: an unusual-occupancy-hour deviation.
        let off = ZoneObservation::new(zone(), ts(99, 3), 1.0, EvidenceLevel::L2);
        let outcome = mem.observe(&off).unwrap();
        assert!(outcome.occupancy.is_anomalous());
        assert!(outcome.anomaly_kinds().contains(&AnomalyKind::UnusualOccupancyHour));

        // Occupied at 2pm is normal, not flagged.
        let normal = ZoneObservation::new(zone(), ts(100, 14), 1.0, EvidenceLevel::L2);
        let outcome = mem.observe(&normal).unwrap();
        assert!(matches!(outcome.occupancy, Assessment::Normal { .. }));
        assert!(!outcome.has_anomaly());
    }

    #[test]
    fn insufficient_history_is_unknown_not_false_positive() {
        let mut mem = SpatialMemory::new(cfg()).unwrap();

        // No baseline anywhere, and a wildly off snapshot: every channel is
        // UNKNOWN (first-class), and nothing is flagged.
        let obs = ZoneObservation::new(zone(), ts(0, 3), 999.0, EvidenceLevel::L2)
            .with_link(LinkId::new(sensor("a"), sensor("b")), -999.0)
            .with_modality("vibration_rms", 999.0);
        let outcome = mem.observe(&obs).unwrap();

        assert!(outcome.occupancy.is_unknown());
        assert!(outcome.propagation.iter().all(|(_, a)| a.is_unknown()));
        assert!(outcome.modality.iter().all(|(_, a)| a.is_unknown()));
        assert!(!outcome.has_anomaly(), "must not false-positive on thin history");
    }

    #[test]
    fn baseline_update_is_deterministic_and_serde_round_trips() {
        let build = || {
            let mut mem = SpatialMemory::new(cfg()).unwrap();
            let twin = RfTwin::build(synthetic_deployment(5)).unwrap();
            mem.seed_zone_propagation_from_twin(zone(), &twin).unwrap();
            let predicted = ObservationSet::from_twin_prediction(&twin);
            for day in 0..10 {
                let mut obs = ZoneObservation::new(zone(), ts(day, 9), 1.0, EvidenceLevel::L1);
                obs.links = predicted.clone();
                obs.modality.insert("vibration_rms".into(), 0.5);
                mem.observe(&obs).unwrap();
            }
            mem
        };

        // Same inputs in the same order ⇒ bitwise-identical learned state. (The
        // update recursion is a pure, deterministic function of the input stream;
        // see `stat::tests` for the per-statistic proof.)
        let a = build();
        let b = build();
        assert_eq!(a, b, "same observations in the same order ⇒ identical state");

        let json = serde_json::to_string(&a).unwrap();
        let back: SpatialMemory = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn flags_a_modality_signature_delta() {
        let mut mem = SpatialMemory::new(cfg()).unwrap();

        // Learn a normal vibration signature (constant baseline).
        for day in 0..12 {
            let obs = ZoneObservation::new(zone(), ts(day, 10), 1.0, EvidenceLevel::L2)
                .with_modality("vibration_rms", 0.5);
            assert!(!mem.observe(&obs).unwrap().has_anomaly());
        }

        // A machine whose vibration signature changed: a modality deviation.
        let obs = ZoneObservation::new(zone(), ts(99, 10), 1.0, EvidenceLevel::L2)
            .with_modality("vibration_rms", 5.0);
        let outcome = mem.observe(&obs).unwrap();
        assert!(outcome.anomaly_kinds().contains(&AnomalyKind::ModalityChange));
        assert!(outcome.modality[0].1.is_anomalous());
    }

    #[test]
    fn anomalies_emit_synthetic_evidence_records_with_provenance() {
        let mut mem = SpatialMemory::new(cfg()).unwrap();
        let twin = RfTwin::build(synthetic_deployment(3)).unwrap();
        mem.seed_zone_propagation_from_twin(zone(), &twin).unwrap();

        let target = twin.links()[0].clone();
        let predicted_mean = mem
            .baseline(&zone())
            .unwrap()
            .link_stat(&target)
            .unwrap()
            .mean();
        let mut obs = ZoneObservation::new(zone(), ts(0, 12), 1.0, EvidenceLevel::L1);
        obs.links = ObservationSet::new().with(target.clone(), predicted_mean - 40.0);
        let outcome = mem.observe(&obs).unwrap();
        assert!(outcome.has_anomaly());

        let mut ledger = EvidenceLedger::new();
        let seqs = mem.record_anomalies(&outcome, &mut ledger, 42).unwrap();
        assert_eq!(seqs.len(), outcome.anomalies.len());
        assert!(!ledger.is_empty());

        let ev = &outcome.anomalies[0];
        let ctx = EvidenceContext::new(
            ev.zone.as_str(),
            "spatial-memory-scaffold",
            format!("anomaly:{}", ev.kind.label()),
            &mem.config().model_version,
        )
        .unwrap();
        let slice = ledger.query(&ctx);
        assert_eq!(slice.len(), 1);
        let rec = slice.records()[0];
        // Emitted evidence is honest: synthetic, forced L0.
        assert_eq!(rec.level(), EvLevel::L0);
        // The deviation magnitude is carried as drift.
        assert!((rec.metrics().drift - ev.deviation().abs()).abs() < 1e-9);
        assert!((rec.metrics().uncertainty - ev.significance).abs() < 1e-9);
    }
}

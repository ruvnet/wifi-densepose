//! Deviation categories, per-channel assessment, and the anomaly event
//! (ADR-312 §3 — anomaly = deviation from learned normal).
//!
//! **SYNTHETIC / L0.** An [`AnomalyEvent`] is a *model-relative* statement: a
//! live value sits statistically far from the location's own learned normal. It
//! is a **candidate** change to corroborate, never a confident detection and
//! never a diagnosis (ADR-282 bounded-claims discipline, ADR-300). No accuracy,
//! detection-rate, or false-positive number is asserted anywhere. Consistent
//! with ADR-300 rule 1, [`Assessment::Unknown`] (insufficient history) is a
//! first-class value, never an error and never a false positive.

use serde::{Deserialize, Serialize};

use ruview_evidence::{AccuracyMetrics, EvidenceContext, EvidenceError, EvidenceRecord};
use ruview_ontology::{EvidenceLevel, SemanticProvenance, ZoneId};

/// The coarse category of a learned-normal deviation (ADR-312 §1). None of
/// these is a labelled anomaly *class* trained from examples — each is a
/// deviation from a baseline of normality, so a novel change still registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyKind {
    /// The space is occupied (or unoccupied) at an hour-of-day it normally is
    /// not — the "bedroom usually occupied certain hours" case.
    UnusualOccupancyHour,
    /// A link's propagation signature deviates from the learned normal — the
    /// "chair moved" / "RF propagation changed" / "new reflector appeared"
    /// cases, which all surface as a per-link RSSI delta.
    PropagationChange,
    /// A coarse per-modality signature channel deviates from normal — the
    /// "machine's vibration signature changed" case.
    ModalityChange,
}

impl AnomalyKind {
    /// A stable snake_case label used in evidence-record context keys.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            AnomalyKind::UnusualOccupancyHour => "unusual_occupancy_hour",
            AnomalyKind::PropagationChange => "propagation_change",
            AnomalyKind::ModalityChange => "modality_change",
        }
    }
}

/// The outcome of scoring one channel (occupancy bucket, link, or modality
/// channel) against its learned baseline.
///
/// **SYNTHETIC / L0.** `significance` is a modelled standard-deviation count
/// against the baseline's own learned variance, not a calibrated probability.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Assessment {
    /// Insufficient history to judge (fewer than `min_history` updates). A
    /// first-class value (ADR-300 rule 1): the channel is *not* flagged, so no
    /// anomaly is emitted before a baseline exists.
    Unknown,
    /// Evaluated and within normal variation (`significance < threshold`).
    Normal {
        /// Modelled deviation significance (standard deviations), `≥ 0`.
        significance: f64,
    },
    /// Evaluated and statistically far from normal (`significance ≥ threshold`).
    Anomalous {
        /// Modelled deviation significance (standard deviations), `≥ 0`.
        significance: f64,
    },
}

impl Assessment {
    /// True when the channel had too little history to judge.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Assessment::Unknown)
    }

    /// True when the channel deviated beyond the threshold.
    #[must_use]
    pub fn is_anomalous(&self) -> bool {
        matches!(self, Assessment::Anomalous { .. })
    }

    /// The modelled significance when evaluated, `None` when UNKNOWN.
    #[must_use]
    pub fn significance(&self) -> Option<f64> {
        match self {
            Assessment::Unknown => None,
            Assessment::Normal { significance } | Assessment::Anomalous { significance } => {
                Some(*significance)
            }
        }
    }
}

/// A flagged deviation from a zone's learned normal (ADR-312 §3).
///
/// **SYNTHETIC / L0.** Carries the baseline it deviated from, the deviation
/// magnitude and significance, and its evidence level — which is the floor of
/// the observations the baseline was learned from and is **never presented
/// above** them (ADR-282 no-upgrade rule).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnomalyEvent {
    /// The zone whose learned normal was deviated from.
    pub zone: ZoneId,
    /// The deviation category.
    pub kind: AnomalyKind,
    /// Producer-supplied observation time (Unix ms). Injected, never sampled.
    pub at_unix_ms: i64,
    /// UTC hour-of-day derived deterministically from `at_unix_ms`.
    pub hour_of_day: u8,
    /// Which channel deviated (link endpoints, modality channel, or hour).
    pub detail: String,
    /// The observed value.
    pub observed: f64,
    /// The learned baseline mean it deviated from.
    pub baseline_mean: f64,
    /// Modelled deviation significance (standard deviations), `≥ 0`.
    pub significance: f64,
    /// Number of observations backing the baseline.
    pub baseline_count: u32,
    /// Evidence level — the floor of the baseline's source observations, never
    /// above them.
    pub evidence_level: EvidenceLevel,
    /// Provenance travelling with the event (SYNTHETIC / L0 scaffold).
    pub provenance: SemanticProvenance,
}

impl AnomalyEvent {
    /// The signed deviation `observed − baseline_mean`.
    #[must_use]
    pub fn deviation(&self) -> f64 {
        self.observed - self.baseline_mean
    }

    /// Project this anomaly into an append-only [`EvidenceRecord`]
    /// (ADR-304/ADR-312: "emit anomalies as evidence records with provenance").
    ///
    /// The record is always **synthetic** (forced [`ruview_evidence::EvidenceLevel::L0`]),
    /// keyed by context `(room = zone, device = "spatial-memory-scaffold",
    /// subject_class = "anomaly:<kind>", model_version)`. The deviation
    /// magnitude is carried as the record's `drift` (fingerprint distance from
    /// baseline) and the significance as its `uncertainty`; `sample_count` is
    /// the baseline's backing history. No rate is fabricated — the accuracy
    /// rates are left at `0.0` because this scaffold asserts none.
    ///
    /// # Errors
    /// Propagates [`EvidenceError`] if a context field is empty/over-length.
    pub fn to_evidence_record(
        &self,
        model_version: &str,
        timestamp_ns: u64,
    ) -> Result<EvidenceRecord, EvidenceError> {
        let context = EvidenceContext::new(
            self.zone.as_str(),
            "spatial-memory-scaffold",
            format!("anomaly:{}", self.kind.label()),
            model_version,
        )?;
        let metrics = AccuracyMetrics {
            moving_recall: 0.0,
            stationary_recall: 0.0,
            false_positive_rate: 0.0,
            drift: self.deviation().abs(),
            uncertainty: self.significance,
            calibration_age_secs: 0,
            sample_count: u64::from(self.baseline_count.max(1)),
        };
        EvidenceRecord::synthetic(context, metrics, timestamp_ns)
    }
}

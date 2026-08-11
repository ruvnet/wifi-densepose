//! # `ruview-evidence` — the append-only accuracy ledger (ADR-304, ADR-300 §4)
//!
//! "MLflow for physical sensing." Where an experiment tracker overwrites
//! yesterday's number, this crate is an **append-only** record of how a model
//! actually performs, keyed per deployment context
//! `(room, device, subject-class, model-version)` and carrying, per record,
//! the ADR-304 metrics (moving/stationary recall, false-positive rate, drift,
//! predictive uncertainty, calibration age, sample count) plus exactly one
//! [`EvidenceLevel`] (L0–L5, mirroring ADR-282 semantics).
//!
//! ## Leaf, deterministic, honest
//!
//! - **Leaf**: this crate depends only on `serde`/`thiserror`. The
//!   [`EvidenceLevel`] ladder mirrors ADR-282 (`frame::EvidenceLevel`) but is
//!   defined locally so the ledger never pulls in the frame crate.
//! - **Deterministic**: no wall-clock and no randomness. Record time is
//!   injected by the caller; the ledger assigns a monotonic append sequence.
//! - **Honest by construction**:
//!   - A record's [`EvidenceLevel`] is fixed by its *provenance* at write time
//!     ([`EvidenceRecord::synthetic`] is `L0` and cannot be raised — there is
//!     no `set_level`). This is the ADR-282/288/290 "no upgrade" rule.
//!   - Records are **append-only**: [`EvidenceLedger::append`] consumes a
//!     record by value and nothing hands back a mutable reference. A correction
//!     is a *new* record, never an in-place edit (ADR-304 §1).
//!   - Aggregation **never pools across contexts** (ADR-304 §2/§Consequences):
//!     an [`EvidenceSlice`] is minted by [`EvidenceLedger::query`] for exactly
//!     one context and there is no API that averages two contexts into one
//!     number. A summary's evidence level is the **floor** (minimum) of the
//!     levels present in the slice — a slice can never report a level above the
//!     weakest record it contains.
//!   - An empty context returns [`SummaryEvidence::NoEvidence`], distinct from a
//!     present-but-zero-accuracy summary — downstream (ADR-318) must treat
//!     "no evidence" as "no capability", not as a `0.0` score.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Maximum byte length accepted for any context identifier string. Bounds
/// allocation at the untrusted-input boundary (CLAUDE.md).
pub const MAX_ID_LEN: usize = 256;

/// Default upper bound on records held by a single ledger. Bounds allocation;
/// [`EvidenceLedger::with_capacity`] can raise or lower it.
pub const DEFAULT_MAX_RECORDS: usize = 1_000_000;

/// The ADR-282 evidence ladder, L0–L5, mirrored locally to keep this crate a
/// leaf (no dependency on the frame crate). Exactly one level travels with each
/// [`EvidenceRecord`]. Ordering is meaningful and load-bearing: the summary
/// floor rule takes `min` over these, so `L0 < L1 < … < L5`.
///
/// Semantics mirror `frame::EvidenceLevel` (ADR-282 §4): L0 simulation-only,
/// rising to L5 production/witnessed evidence. See ADR-282 for the canonical
/// ladder; this enum is a faithful local copy, not an independent scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceLevel {
    /// L0 — simulation / synthetic only, no signal evidence (ADR-282).
    L0,
    /// L1 — captured replay / heuristic evidence.
    L1,
    /// L2 — controlled single-surface signal evidence.
    L2,
    /// L3 — corroborated / held-out room-and-subject validation.
    L3,
    /// L4 — calibrated multi-site field evidence.
    L4,
    /// L5 — production, witnessed / certified (ADR-319).
    L5,
}

/// Accuracy tag for a record (CLAUDE.md honesty rule). The class is fixed by
/// the constructor used and cannot alias: synthetic input can never be minted
/// as `Measured`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceClass {
    /// Produced by a simulator/generator — L0 by construction (ADR-276/301).
    Synthetic,
    /// Real inference but no ground-truth reference backs the accuracy.
    Claimed,
    /// Backed by an ADR-303 reference plus a reproducer handle.
    Measured,
}

/// The deployment context a record is keyed by: `(room, device, subject-class,
/// model-version)`. Identity is caller-supplied (ADR-306 space id, ADR-305
/// signed device id); this crate treats the fields as opaque bounded handles
/// and never invents them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EvidenceContext {
    /// Space / room id (ADR-306).
    pub room: String,
    /// Signed device id (ADR-305).
    pub device: String,
    /// Subject class where consented/available; empty means "no subject"
    /// (ADR-304 §1 — subject id only where consented).
    pub subject_class: String,
    /// Model version that produced the inferences (ADR-136).
    pub model_version: String,
}

impl EvidenceContext {
    /// Construct a context, validating every field at the boundary. `room`,
    /// `device`, and `model_version` must be non-empty; every field is bounded
    /// to [`MAX_ID_LEN`] bytes. `subject_class` may be empty (no consented
    /// subject) but is still length-bounded.
    ///
    /// # Errors
    /// Returns [`EvidenceError::EmptyField`] for a missing required field and
    /// [`EvidenceError::IdTooLong`] for any over-length field.
    pub fn new(
        room: impl Into<String>,
        device: impl Into<String>,
        subject_class: impl Into<String>,
        model_version: impl Into<String>,
    ) -> Result<Self, EvidenceError> {
        let room = room.into();
        let device = device.into();
        let subject_class = subject_class.into();
        let model_version = model_version.into();

        check_bound("room", &room)?;
        check_bound("device", &device)?;
        check_bound("subject_class", &subject_class)?;
        check_bound("model_version", &model_version)?;
        check_nonempty("room", &room)?;
        check_nonempty("device", &device)?;
        check_nonempty("model_version", &model_version)?;

        Ok(Self {
            room,
            device,
            subject_class,
            model_version,
        })
    }
}

fn check_bound(field: &'static str, value: &str) -> Result<(), EvidenceError> {
    if value.len() > MAX_ID_LEN {
        return Err(EvidenceError::IdTooLong {
            field,
            len: value.len(),
            max: MAX_ID_LEN,
        });
    }
    Ok(())
}

fn check_nonempty(field: &'static str, value: &str) -> Result<(), EvidenceError> {
    if value.is_empty() {
        return Err(EvidenceError::EmptyField { field });
    }
    Ok(())
}

/// The per-inference-window accuracy metrics accumulated into a record
/// (ADR-304 §1). Rates are fractions in `[0, 1]`; `drift` and `uncertainty`
/// are non-negative finite magnitudes; `sample_count` is the number of
/// inferences the record summarizes and must be at least one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Recall on moving subjects, `[0, 1]`.
    pub moving_recall: f64,
    /// Recall on stationary subjects, `[0, 1]`.
    pub stationary_recall: f64,
    /// False-positive rate, `[0, 1]`.
    pub false_positive_rate: f64,
    /// Drift magnitude — fingerprint distance from the calibration baseline
    /// (ADR-301); non-negative.
    pub drift: f64,
    /// Predictive uncertainty; non-negative.
    pub uncertainty: f64,
    /// Age of the calibration certificate in effect, seconds (ADR-301).
    pub calibration_age_secs: u64,
    /// Number of inferences this record summarizes; at least one.
    pub sample_count: u64,
}

impl AccuracyMetrics {
    /// Validate the metrics at the boundary. Rates must be finite and within
    /// `[0, 1]`; `drift`/`uncertainty` must be finite and non-negative;
    /// `sample_count` must be `>= 1` (a record represents at least one
    /// inference, which also guarantees non-zero aggregation weight).
    ///
    /// # Errors
    /// [`EvidenceError::RateOutOfRange`], [`EvidenceError::NegativeMagnitude`],
    /// or [`EvidenceError::ZeroSamples`].
    pub fn validate(&self) -> Result<(), EvidenceError> {
        check_rate("moving_recall", self.moving_recall)?;
        check_rate("stationary_recall", self.stationary_recall)?;
        check_rate("false_positive_rate", self.false_positive_rate)?;
        check_magnitude("drift", self.drift)?;
        check_magnitude("uncertainty", self.uncertainty)?;
        if self.sample_count == 0 {
            return Err(EvidenceError::ZeroSamples);
        }
        Ok(())
    }
}

fn check_rate(field: &'static str, v: f64) -> Result<(), EvidenceError> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(EvidenceError::RateOutOfRange { field, value: v });
    }
    Ok(())
}

fn check_magnitude(field: &'static str, v: f64) -> Result<(), EvidenceError> {
    if !v.is_finite() || v < 0.0 {
        return Err(EvidenceError::NegativeMagnitude { field, value: v });
    }
    Ok(())
}

/// One immutable, append-only accuracy record (ADR-304 §1). All fields are
/// private: there is no setter and no `&mut` accessor, so a level can never be
/// upgraded and a record can never be edited in place — a correction is a new
/// record. Construct via [`EvidenceRecord::synthetic`],
/// [`EvidenceRecord::claimed`], or [`EvidenceRecord::measured`]; the sequence
/// number is assigned by the ledger on [`EvidenceLedger::append`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    context: EvidenceContext,
    metrics: AccuracyMetrics,
    level: EvidenceLevel,
    class: ProvenanceClass,
    /// Reproducer handle for `Measured` records (ADR-303); empty otherwise.
    reproducer: String,
    /// Caller-injected record time, nanoseconds. Never read from a clock here.
    timestamp_ns: u64,
    /// Ledger-assigned monotonic append sequence; `None` until appended.
    seq: Option<u64>,
}

impl EvidenceRecord {
    /// Mint a **synthetic** record. Class is [`ProvenanceClass::Synthetic`] and
    /// the evidence level is forced to [`EvidenceLevel::L0`] — synthetic input
    /// is L0 by construction (ADR-304 §3) and there is no way to raise it.
    ///
    /// # Errors
    /// Propagates [`AccuracyMetrics::validate`] failures.
    pub fn synthetic(
        context: EvidenceContext,
        metrics: AccuracyMetrics,
        timestamp_ns: u64,
    ) -> Result<Self, EvidenceError> {
        metrics.validate()?;
        Ok(Self {
            context,
            metrics,
            level: EvidenceLevel::L0,
            class: ProvenanceClass::Synthetic,
            reproducer: String::new(),
            timestamp_ns,
            seq: None,
        })
    }

    /// Mint a **claimed** record: a real inference with no ADR-303 reference
    /// backing its accuracy. The level is set by the caller's provenance at
    /// write time and is never MEASURED. A claimed record may not be minted at
    /// `L0`, which is reserved for synthetic input.
    ///
    /// # Errors
    /// Propagates metric validation; [`EvidenceError::SyntheticOnlyL0`] if
    /// `level` is `L0`.
    pub fn claimed(
        context: EvidenceContext,
        metrics: AccuracyMetrics,
        level: EvidenceLevel,
        timestamp_ns: u64,
    ) -> Result<Self, EvidenceError> {
        metrics.validate()?;
        if level == EvidenceLevel::L0 {
            return Err(EvidenceError::SyntheticOnlyL0);
        }
        Ok(Self {
            context,
            metrics,
            level,
            class: ProvenanceClass::Claimed,
            reproducer: String::new(),
            timestamp_ns,
            seq: None,
        })
    }

    /// Mint a **measured** record: accuracy backed by an ADR-303 reference and
    /// a non-empty reproducer handle. The level is set by provenance and must
    /// not be `L0`.
    ///
    /// # Errors
    /// Propagates metric validation; [`EvidenceError::MissingReproducer`] if
    /// the reproducer handle is empty or over-length;
    /// [`EvidenceError::SyntheticOnlyL0`] if `level` is `L0`.
    pub fn measured(
        context: EvidenceContext,
        metrics: AccuracyMetrics,
        level: EvidenceLevel,
        reproducer: impl Into<String>,
        timestamp_ns: u64,
    ) -> Result<Self, EvidenceError> {
        metrics.validate()?;
        if level == EvidenceLevel::L0 {
            return Err(EvidenceError::SyntheticOnlyL0);
        }
        let reproducer = reproducer.into();
        check_bound("reproducer", &reproducer)?;
        if reproducer.is_empty() {
            return Err(EvidenceError::MissingReproducer);
        }
        Ok(Self {
            context,
            metrics,
            level,
            class: ProvenanceClass::Measured,
            reproducer,
            timestamp_ns,
            seq: None,
        })
    }

    /// The context this record is keyed by.
    #[must_use]
    pub fn context(&self) -> &EvidenceContext {
        &self.context
    }

    /// The record's metrics.
    #[must_use]
    pub fn metrics(&self) -> &AccuracyMetrics {
        &self.metrics
    }

    /// The record's evidence level, fixed at write time.
    #[must_use]
    pub fn level(&self) -> EvidenceLevel {
        self.level
    }

    /// The record's provenance class.
    #[must_use]
    pub fn class(&self) -> ProvenanceClass {
        self.class
    }

    /// The reproducer handle (empty unless [`ProvenanceClass::Measured`]).
    #[must_use]
    pub fn reproducer(&self) -> &str {
        &self.reproducer
    }

    /// Caller-injected record time in nanoseconds.
    #[must_use]
    pub fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    /// Ledger-assigned append sequence, or `None` before the record is
    /// appended.
    #[must_use]
    pub fn seq(&self) -> Option<u64> {
        self.seq
    }
}

/// The append-only evidence ledger (ADR-304). The record vector is private and
/// exposed only through read-only queries; nothing returns a mutable reference
/// to a stored record, so the append-only and no-upgrade invariants hold at the
/// type level.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvidenceLedger {
    records: Vec<EvidenceRecord>,
    next_seq: u64,
    max_records: usize,
}

impl EvidenceLedger {
    /// A new empty ledger bounded to [`DEFAULT_MAX_RECORDS`] records.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_RECORDS)
    }

    /// A new empty ledger bounded to `max_records`.
    #[must_use]
    pub fn with_capacity(max_records: usize) -> Self {
        Self {
            records: Vec::new(),
            next_seq: 0,
            max_records,
        }
    }

    /// Append a record. The ledger stamps it with the next monotonic sequence
    /// and stores it; the record is consumed by value, so the caller cannot
    /// retain a handle to mutate the stored copy. Returns the assigned
    /// sequence.
    ///
    /// # Errors
    /// [`EvidenceError::LedgerFull`] once the bounded capacity is reached, so
    /// a malformed or runaway producer cannot exhaust memory.
    pub fn append(&mut self, mut record: EvidenceRecord) -> Result<u64, EvidenceError> {
        if self.records.len() >= self.max_records {
            return Err(EvidenceError::LedgerFull {
                max: self.max_records,
            });
        }
        let seq = self.next_seq;
        record.seq = Some(seq);
        self.next_seq += 1;
        self.records.push(record);
        Ok(seq)
    }

    /// Total number of records in the ledger.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the ledger holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Every record, in append order (read-only).
    #[must_use]
    pub fn records(&self) -> &[EvidenceRecord] {
        &self.records
    }

    /// Query the records for exactly one context, in append order. The returned
    /// [`EvidenceSlice`] carries only records whose context equals `context`,
    /// so aggregation over it can never mix two contexts (ADR-304 §2 — no
    /// pooling).
    #[must_use]
    pub fn query<'a>(&'a self, context: &EvidenceContext) -> EvidenceSlice<'a> {
        let records: Vec<&'a EvidenceRecord> = self
            .records
            .iter()
            .filter(|r| &r.context == context)
            .collect();
        EvidenceSlice {
            context: context.clone(),
            records,
        }
    }

    /// The distinct contexts present in the ledger, in first-append order.
    #[must_use]
    pub fn contexts(&self) -> Vec<EvidenceContext> {
        let mut out: Vec<EvidenceContext> = Vec::new();
        for r in &self.records {
            if !out.contains(&r.context) {
                out.push(r.context.clone());
            }
        }
        out
    }

    /// Summarize **each** context independently and return one summary per
    /// context — never a single pooled number across contexts (ADR-304
    /// §Consequences: "never paper over a thin context with a global average").
    #[must_use]
    pub fn summarize(&self) -> Vec<ContextSummary> {
        self.contexts()
            .into_iter()
            .map(|ctx| self.query(&ctx).summarize())
            .collect()
    }
}

/// A read-only view of the records for exactly one context. It can only be
/// minted by [`EvidenceLedger::query`], so a slice is always single-context —
/// there is no constructor that merges two contexts, which is what makes
/// pooling impossible through the API.
#[derive(Clone, Debug)]
pub struct EvidenceSlice<'a> {
    context: EvidenceContext,
    records: Vec<&'a EvidenceRecord>,
}

impl<'a> EvidenceSlice<'a> {
    /// The single context this slice covers.
    #[must_use]
    pub fn context(&self) -> &EvidenceContext {
        &self.context
    }

    /// The records in the slice, in append order (read-only).
    #[must_use]
    pub fn records(&self) -> &[&'a EvidenceRecord] {
        &self.records
    }

    /// Number of records in the slice.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the slice has no records (the context has no evidence).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Aggregate the slice into a per-context summary. This is a **pure**
    /// function of the records (deterministic; no clock, no randomness):
    ///
    /// - An empty slice yields [`SummaryEvidence::NoEvidence`] — distinct from
    ///   a zero-accuracy summary (ADR-304 §3).
    /// - The summary's evidence level is the **floor** — the minimum level over
    ///   the records — so a slice can never report a level above its weakest
    ///   record (the "no upgrade" honesty rule). Synthetic (L0) records pin the
    ///   floor to L0.
    /// - Rates and uncertainty are sample-count-weighted means; `drift` and
    ///   `calibration_age` report the latest (by append sequence) value with
    ///   the running maximum; `sample_count` is the sum. All within this one
    ///   context — nothing is pooled across contexts.
    #[must_use]
    pub fn summarize(&self) -> ContextSummary {
        if self.records.is_empty() {
            return ContextSummary {
                context: self.context.clone(),
                evidence: SummaryEvidence::NoEvidence,
            };
        }

        // Floor over evidence levels — never an upgrade. Safe: non-empty.
        let level = self
            .records
            .iter()
            .map(|r| r.level)
            .min()
            .expect("slice is non-empty");

        // The class is Measured only if *every* record is Measured; any weaker
        // record downgrades the aggregate class (honesty, no upgrade).
        let aggregate_class = self.aggregate_class();

        let mut total_samples: u128 = 0;
        let mut w_moving: f64 = 0.0;
        let mut w_stationary: f64 = 0.0;
        let mut w_fpr: f64 = 0.0;
        let mut w_uncertainty: f64 = 0.0;
        let mut max_drift: f64 = 0.0;
        let mut max_calibration_age_secs: u64 = 0;

        // Latest by append sequence (deterministic, no clock). Records without
        // a seq (never appended) sort before any appended record.
        let latest = self
            .records
            .iter()
            .max_by_key(|r| r.seq.unwrap_or(0))
            .expect("slice is non-empty");

        for r in &self.records {
            let m = &r.metrics;
            let w = m.sample_count as f64;
            total_samples += u128::from(m.sample_count);
            w_moving += m.moving_recall * w;
            w_stationary += m.stationary_recall * w;
            w_fpr += m.false_positive_rate * w;
            w_uncertainty += m.uncertainty * w;
            if m.drift > max_drift {
                max_drift = m.drift;
            }
            if m.calibration_age_secs > max_calibration_age_secs {
                max_calibration_age_secs = m.calibration_age_secs;
            }
        }

        // Every record has sample_count >= 1, so the divisor is never zero.
        let denom = total_samples as f64;
        let agg = AggregateMetrics {
            record_count: self.records.len(),
            sample_count: total_samples,
            moving_recall: w_moving / denom,
            stationary_recall: w_stationary / denom,
            false_positive_rate: w_fpr / denom,
            uncertainty: w_uncertainty / denom,
            latest_drift: latest.metrics.drift,
            max_drift,
            latest_calibration_age_secs: latest.metrics.calibration_age_secs,
            max_calibration_age_secs,
        };

        ContextSummary {
            context: self.context.clone(),
            evidence: SummaryEvidence::Aggregated {
                level,
                class: aggregate_class,
                metrics: agg,
            },
        }
    }

    fn aggregate_class(&self) -> ProvenanceClass {
        let mut any_synthetic = false;
        let mut all_measured = true;
        for r in &self.records {
            match r.class {
                ProvenanceClass::Synthetic => any_synthetic = true,
                ProvenanceClass::Claimed => all_measured = false,
                ProvenanceClass::Measured => {}
            }
        }
        if any_synthetic {
            ProvenanceClass::Synthetic
        } else if all_measured {
            ProvenanceClass::Measured
        } else {
            ProvenanceClass::Claimed
        }
    }
}

/// A per-context summary. Always carries the context it belongs to, so a
/// summary can never be mistaken for a global rollup.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextSummary {
    /// The context this summary covers.
    pub context: EvidenceContext,
    /// Either "no evidence" or the aggregated metrics for this one context.
    pub evidence: SummaryEvidence,
}

impl ContextSummary {
    /// Whether this context has any evidence at all.
    #[must_use]
    pub fn has_evidence(&self) -> bool {
        matches!(self.evidence, SummaryEvidence::Aggregated { .. })
    }
}

/// The evidence outcome for a context: explicitly absent, or aggregated.
///
/// [`SummaryEvidence::NoEvidence`] is deliberately **not** a zero-accuracy
/// summary: an empty context has *no capability*, which downstream (ADR-318)
/// must not read as a `0.0` score.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SummaryEvidence {
    /// The context has no records — no evidence, not zero accuracy.
    NoEvidence,
    /// Aggregated metrics for the one context.
    Aggregated {
        /// Floor evidence level (min over the slice) — never upgraded.
        level: EvidenceLevel,
        /// Aggregate provenance class (Measured only if all records are).
        class: ProvenanceClass,
        /// The aggregated metrics for this context.
        metrics: AggregateMetrics,
    },
}

/// Aggregated metrics for a single context. Every field is derived purely from
/// that context's records; nothing here is pooled across contexts.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Number of records aggregated.
    pub record_count: usize,
    /// Sum of `sample_count` across records.
    pub sample_count: u128,
    /// Sample-weighted mean moving recall.
    pub moving_recall: f64,
    /// Sample-weighted mean stationary recall.
    pub stationary_recall: f64,
    /// Sample-weighted mean false-positive rate.
    pub false_positive_rate: f64,
    /// Sample-weighted mean predictive uncertainty.
    pub uncertainty: f64,
    /// Drift of the latest record (by append sequence) — trajectory endpoint.
    pub latest_drift: f64,
    /// Maximum drift observed in the context.
    pub max_drift: f64,
    /// Calibration age of the latest record, seconds.
    pub latest_calibration_age_secs: u64,
    /// Maximum calibration age observed, seconds.
    pub max_calibration_age_secs: u64,
}

/// Errors raised at the ledger's input boundaries. No variant panics; malformed
/// input is always a returned error (CLAUDE.md).
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EvidenceError {
    /// A required context field was empty.
    #[error("context field `{field}` must not be empty")]
    EmptyField {
        /// The offending field name.
        field: &'static str,
    },
    /// A context/reproducer identifier exceeded [`MAX_ID_LEN`].
    #[error("identifier `{field}` is {len} bytes, exceeds max {max}")]
    IdTooLong {
        /// The offending field name.
        field: &'static str,
        /// Actual byte length.
        len: usize,
        /// Allowed maximum.
        max: usize,
    },
    /// A rate metric was outside `[0, 1]` or non-finite.
    #[error("rate `{field}` = {value} is out of range [0, 1] or non-finite")]
    RateOutOfRange {
        /// The offending field name.
        field: &'static str,
        /// The rejected value.
        value: f64,
    },
    /// A magnitude metric was negative or non-finite.
    #[error("magnitude `{field}` = {value} must be finite and non-negative")]
    NegativeMagnitude {
        /// The offending field name.
        field: &'static str,
        /// The rejected value.
        value: f64,
    },
    /// A record claimed zero samples.
    #[error("sample_count must be at least 1")]
    ZeroSamples,
    /// A non-synthetic record was minted at L0, which is reserved for
    /// synthetic input.
    #[error("L0 is reserved for synthetic records")]
    SyntheticOnlyL0,
    /// A measured record was minted without a reproducer handle.
    #[error("a measured record requires a non-empty reproducer handle")]
    MissingReproducer,
    /// The bounded ledger is full.
    #[error("ledger is full ({max} records)")]
    LedgerFull {
        /// The capacity that was reached.
        max: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(room: &str, subject: &str) -> EvidenceContext {
        EvidenceContext::new(room, "dev-esp32-A", subject, "model-v1").expect("valid context")
    }

    fn metrics(sample_count: u64) -> AccuracyMetrics {
        AccuracyMetrics {
            moving_recall: 0.8,
            stationary_recall: 0.6,
            false_positive_rate: 0.05,
            drift: 0.1,
            uncertainty: 0.2,
            calibration_age_secs: 3600,
            sample_count,
        }
    }

    #[test]
    fn append_assigns_monotonic_seq_and_query_filters_by_context() {
        let mut ledger = EvidenceLedger::new();
        let kitchen = ctx("kitchen", "adult");
        let bedroom = ctx("bedroom", "adult");

        let s0 = ledger
            .append(EvidenceRecord::synthetic(kitchen.clone(), metrics(10), 1).unwrap())
            .unwrap();
        let s1 = ledger
            .append(EvidenceRecord::synthetic(bedroom.clone(), metrics(20), 2).unwrap())
            .unwrap();
        let s2 = ledger
            .append(EvidenceRecord::synthetic(kitchen.clone(), metrics(30), 3).unwrap())
            .unwrap();

        assert_eq!((s0, s1, s2), (0, 1, 2));
        assert_eq!(ledger.len(), 3);

        let k = ledger.query(&kitchen);
        assert_eq!(k.len(), 2);
        assert!(k.records().iter().all(|r| r.context() == &kitchen));

        let b = ledger.query(&bedroom);
        assert_eq!(b.len(), 1);
        assert_eq!(b.records()[0].metrics().sample_count, 20);
    }

    #[test]
    fn records_are_append_only_no_in_place_edit() {
        // The only mutation is `append`, which consumes by value and stamps a
        // seq. Corrections are new records; the original is unchanged.
        let mut ledger = EvidenceLedger::new();
        let c = ctx("lab", "adult");

        ledger
            .append(EvidenceRecord::measured(c.clone(), metrics(100), EvidenceLevel::L3, "repro-1", 1).unwrap())
            .unwrap();
        // A "correction" is appended, not edited in place.
        ledger
            .append(EvidenceRecord::measured(c.clone(), metrics(50), EvidenceLevel::L3, "repro-2", 2).unwrap())
            .unwrap();

        let slice = ledger.query(&c);
        assert_eq!(slice.len(), 2);
        // Original record still present and unmodified.
        assert_eq!(slice.records()[0].metrics().sample_count, 100);
        assert_eq!(slice.records()[0].reproducer(), "repro-1");
        assert_eq!(slice.records()[0].seq(), Some(0));
        // `records()` returns shared references — no path mutates a stored
        // record. (If a `&mut` accessor existed this test would need to change;
        // its absence is the invariant.)
    }

    #[test]
    fn no_pooling_across_contexts() {
        // The API only ever summarizes one context at a time. `summarize()`
        // returns one entry per context; there is no call that averages two
        // contexts into a single number.
        let mut ledger = EvidenceLedger::new();
        let kitchen = ctx("kitchen", "adult");
        let bedroom = ctx("bedroom", "adult");

        // Kitchen: perfect. Bedroom: poor. A pooled average would hide the poor
        // context; per-context summaries must not.
        let good = AccuracyMetrics { moving_recall: 1.0, ..metrics(100) };
        let bad = AccuracyMetrics { moving_recall: 0.0, ..metrics(100) };
        ledger.append(EvidenceRecord::measured(kitchen.clone(), good, EvidenceLevel::L3, "r", 1).unwrap()).unwrap();
        ledger.append(EvidenceRecord::measured(bedroom.clone(), bad, EvidenceLevel::L3, "r", 2).unwrap()).unwrap();

        let summaries = ledger.summarize();
        assert_eq!(summaries.len(), 2, "one summary per context, never pooled");

        let k = ledger.query(&kitchen).summarize();
        let b = ledger.query(&bedroom).summarize();
        match (k.evidence, b.evidence) {
            (
                SummaryEvidence::Aggregated { metrics: km, .. },
                SummaryEvidence::Aggregated { metrics: bm, .. },
            ) => {
                assert_eq!(km.moving_recall, 1.0);
                assert_eq!(bm.moving_recall, 0.0);
                // No global average exists; if it did it would be 0.5 and hide
                // the bad context. The API offers no such value.
            }
            _ => panic!("both contexts should have evidence"),
        }
    }

    #[test]
    fn evidence_level_floor_is_the_minimum_never_an_upgrade() {
        let mut ledger = EvidenceLedger::new();
        let c = ctx("lab", "adult");

        // A strong measured record...
        ledger.append(EvidenceRecord::measured(c.clone(), metrics(100), EvidenceLevel::L4, "repro", 1).unwrap()).unwrap();
        // ...alongside a synthetic (L0) record in the same context.
        ledger.append(EvidenceRecord::synthetic(c.clone(), metrics(100), 2).unwrap()).unwrap();

        let summary = ledger.query(&c).summarize();
        match summary.evidence {
            SummaryEvidence::Aggregated { level, class, .. } => {
                // Floor: the L0 synthetic record pins the level to L0 — the
                // slice cannot report the higher L4.
                assert_eq!(level, EvidenceLevel::L0);
                // And the class downgrades to Synthetic (no upgrade).
                assert_eq!(class, ProvenanceClass::Synthetic);
            }
            SummaryEvidence::NoEvidence => panic!("context has records"),
        }
    }

    #[test]
    fn synthetic_is_forced_l0_and_cannot_be_upgraded() {
        let c = ctx("sim", "adult");
        let rec = EvidenceRecord::synthetic(c, metrics(10), 1).unwrap();
        assert_eq!(rec.level(), EvidenceLevel::L0);
        assert_eq!(rec.class(), ProvenanceClass::Synthetic);
        // There is no setter to raise the level: the type has no `set_level`.

        // A non-synthetic record cannot occupy L0.
        let c2 = ctx("sim", "adult");
        assert_eq!(
            EvidenceRecord::claimed(c2, metrics(10), EvidenceLevel::L0, 1).unwrap_err(),
            EvidenceError::SyntheticOnlyL0
        );
    }

    #[test]
    fn empty_context_is_no_evidence_not_zero_accuracy() {
        let ledger = EvidenceLedger::new();
        let never_seen = ctx("attic", "adult");

        let slice = ledger.query(&never_seen);
        assert!(slice.is_empty());

        let summary = slice.summarize();
        assert!(!summary.has_evidence());
        assert_eq!(summary.evidence, SummaryEvidence::NoEvidence);
        // Explicitly NOT a zero-accuracy Aggregated summary.
        assert!(!matches!(summary.evidence, SummaryEvidence::Aggregated { .. }));
    }

    #[test]
    fn summarize_is_deterministic_and_serde_round_trips() {
        let build = || {
            let mut ledger = EvidenceLedger::new();
            let c = ctx("kitchen", "adult");
            ledger.append(EvidenceRecord::measured(c.clone(), metrics(100), EvidenceLevel::L3, "r1", 10).unwrap()).unwrap();
            ledger.append(EvidenceRecord::measured(c.clone(), metrics(300), EvidenceLevel::L4, "r2", 20).unwrap()).unwrap();
            ledger
        };

        let a = build().summarize();
        let b = build().summarize();
        assert_eq!(a, b, "aggregation is a pure function of the records");

        // Sample-weighted mean check: same metrics, weights 100 and 300 → 0.8.
        let c = ctx("kitchen", "adult");
        let s = build().query(&c).summarize();
        if let SummaryEvidence::Aggregated { level, metrics: m, .. } = &s.evidence {
            assert_eq!(*level, EvidenceLevel::L3); // floor of L3 and L4
            assert_eq!(m.sample_count, 400);
            assert!((m.moving_recall - 0.8).abs() < 1e-12);
            assert_eq!(m.latest_calibration_age_secs, 3600);
        } else {
            panic!("expected aggregated evidence");
        }

        // Serde round-trip of a summary is stable.
        let json = serde_json::to_string(&a).unwrap();
        let back: Vec<ContextSummary> = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn boundary_validation_rejects_malformed_input_without_panicking() {
        assert_eq!(
            EvidenceContext::new("", "d", "s", "m").unwrap_err(),
            EvidenceError::EmptyField { field: "room" }
        );
        let long = "x".repeat(MAX_ID_LEN + 1);
        assert!(matches!(
            EvidenceContext::new(long, "d", "s", "m").unwrap_err(),
            EvidenceError::IdTooLong { .. }
        ));

        let bad_rate = AccuracyMetrics { moving_recall: 1.5, ..metrics(1) };
        assert!(matches!(
            bad_rate.validate().unwrap_err(),
            EvidenceError::RateOutOfRange { .. }
        ));
        let nan = AccuracyMetrics { uncertainty: f64::NAN, ..metrics(1) };
        assert!(matches!(
            nan.validate().unwrap_err(),
            EvidenceError::NegativeMagnitude { .. }
        ));
        let zero = AccuracyMetrics { sample_count: 0, ..metrics(1) };
        assert_eq!(zero.validate().unwrap_err(), EvidenceError::ZeroSamples);

        let c = ctx("lab", "adult");
        assert_eq!(
            EvidenceRecord::measured(c, metrics(1), EvidenceLevel::L3, "", 1).unwrap_err(),
            EvidenceError::MissingReproducer
        );
    }

    #[test]
    fn ledger_capacity_is_bounded() {
        let mut ledger = EvidenceLedger::with_capacity(1);
        let c = ctx("lab", "adult");
        ledger.append(EvidenceRecord::synthetic(c.clone(), metrics(1), 1).unwrap()).unwrap();
        assert_eq!(
            ledger.append(EvidenceRecord::synthetic(c, metrics(1), 2).unwrap()).unwrap_err(),
            EvidenceError::LedgerFull { max: 1 }
        );
    }
}

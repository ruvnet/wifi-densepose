//! # `ruview-scorecard` — the multi-domain benchmark scorecard (ADR-317, ADR-300 §4)
//!
//! ADR-300 program rule 4 is non-negotiable: **pooled accuracy is never
//! sufficient for promotion**. A single headline number is exactly the surface
//! a domain-generalization regression hides behind — a model can raise mean
//! presence accuracy while quietly collapsing on unseen rooms, unseen devices,
//! or a stationary subject at range (the canonical WiFi failure case).
//!
//! This crate is the *data model* for that discipline (ADR-317 §1). It holds,
//! per capability, one cell **per operating domain** rather than one pooled
//! figure:
//!
//! - **Presence**: `room-known`, `room-unseen`, `device-unseen`,
//!   `stationary-10m`.
//! - **Pose**: `matched`, `subject-unseen`, `room-unseen`.
//! - **OOD rejection**: the rate at which genuinely out-of-distribution input
//!   is correctly returned as UNKNOWN (a capability, ADR-302).
//! - **Calibration drift**: the fingerprint-distance trajectory against the
//!   ADR-301 certificate (lower is better) plus the fraction of inferences in
//!   each ADR-302 [`DomainState`] under the `VALID → DEGRADED → UNKNOWN`
//!   staleness guard (ADR-300).
//!
//! ## Honesty by construction (CLAUDE.md, ADR-282, ADR-300)
//!
//! - Every scored [`Cell`] carries a point estimate, a **confidence interval**
//!   (a documented deterministic Wilson score interval — no RNG), and exactly
//!   one [`EvidenceLevel`]. A slice scored on synthetic input is `L0` by
//!   construction ([`Metric::synthetic`]); nothing raises it here.
//! - An empty domain is [`Cell::NoEvidence`], a first-class value distinct
//!   from a present-but-zero score. The promotion gate treats no evidence as
//!   **no coverage, never a pass** (ADR-317 §Provenance).
//! - [`Scorecard::worst_domain`] returns the promotion-relevant number: the
//!   minimum across a task's domain slices. [`promotion_gate`] fails if *any*
//!   worst-domain slice regresses beyond its budget, **even when the pooled
//!   average improved** — a regression cannot hide behind pooled accuracy.
//!
//! Deterministic and leaf-shaped: no wall clock, no randomness, bounded input
//! validation at every constructor.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use ruview_evidence::EvidenceLevel;

/// The two-sided z-multiplier for a 95% Wilson score interval (the 97.5th
/// percentile of the standard normal). Fixed and documented so intervals are
/// reproducible byte-for-byte.
pub const Z_95: f64 = 1.959_963_984_540_054;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Boundary-validation failures. No constructor panics on malformed input.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
pub enum ScorecardError {
    /// A rate/point estimate was not a finite value inside `[0, 1]`.
    #[error("point estimate {value} out of range (expected finite in [0, 1])")]
    PointOutOfRange {
        /// The offending value.
        value: f64,
    },
    /// A metric was minted with zero samples; a confidence interval needs at
    /// least one observation.
    #[error("sample_count must be >= 1")]
    ZeroSamples,
    /// The supplied confidence z-multiplier was not finite and positive.
    #[error("confidence z-multiplier {value} must be finite and > 0")]
    BadConfidence {
        /// The offending value.
        value: f64,
    },
    /// A [`StateFractions`] triple was out of range or did not sum to ~1.
    #[error("domain-state fractions invalid: {reason}")]
    BadStateFractions {
        /// Human-readable reason.
        reason: &'static str,
    },
}

// ---------------------------------------------------------------------------
// Wilson score interval (deterministic, no RNG)
// ---------------------------------------------------------------------------

/// Compute the two-sided **Wilson score interval** for a binomial proportion.
///
/// For an observed proportion `p` over `n` samples at z-multiplier `z`:
///
/// ```text
/// center = (p + z²/2n) / (1 + z²/n)
/// margin = (z / (1 + z²/n)) · sqrt( p(1-p)/n + z²/4n² )
/// [lo, hi] = clamp(center ∓ margin, 0, 1)
/// ```
///
/// The Wilson interval is preferred over the naive normal approximation
/// `p ± z·sqrt(p(1-p)/n)` because it stays inside `[0, 1]` and behaves well at
/// the `p → 0` / `p → 1` extremes and for small `n` — the regimes a thin
/// unseen-domain slice lives in. Caller guarantees `p ∈ [0,1]`, `n ≥ 1`, and a
/// finite `z > 0`; the result is clamped defensively regardless.
#[must_use]
pub fn wilson_interval(p: f64, n: u64, z: f64) -> (f64, f64) {
    let n = n as f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let radicand = (p * (1.0 - p) / n) + z2 / (4.0 * n * n);
    let margin = (z / denom) * radicand.max(0.0).sqrt();
    let lo = (center - margin).clamp(0.0, 1.0);
    let hi = (center + margin).clamp(0.0, 1.0);
    (lo, hi)
}

// ---------------------------------------------------------------------------
// Metric: a scored cell
// ---------------------------------------------------------------------------

/// A single scored measurement: a point estimate, its confidence interval, the
/// sample count it was computed from, and exactly one [`EvidenceLevel`]. The CI
/// is derived deterministically at construction; there is no setter that could
/// desynchronise the interval from its inputs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    point: f64,
    ci_low: f64,
    ci_high: f64,
    sample_count: u64,
    level: EvidenceLevel,
}

impl Metric {
    /// Construct a metric at 95% confidence ([`Z_95`]), computing the Wilson
    /// interval from `point` and `sample_count`.
    ///
    /// # Errors
    /// [`ScorecardError::PointOutOfRange`] if `point` is not finite in
    /// `[0, 1]`; [`ScorecardError::ZeroSamples`] if `sample_count == 0`.
    pub fn new(
        point: f64,
        sample_count: u64,
        level: EvidenceLevel,
    ) -> Result<Self, ScorecardError> {
        Self::with_confidence(point, sample_count, level, Z_95)
    }

    /// Construct a metric at a caller-chosen z-multiplier.
    ///
    /// # Errors
    /// As [`Metric::new`], plus [`ScorecardError::BadConfidence`] if `z` is not
    /// finite and positive.
    pub fn with_confidence(
        point: f64,
        sample_count: u64,
        level: EvidenceLevel,
        z: f64,
    ) -> Result<Self, ScorecardError> {
        if !point.is_finite() || !(0.0..=1.0).contains(&point) {
            return Err(ScorecardError::PointOutOfRange { value: point });
        }
        if sample_count == 0 {
            return Err(ScorecardError::ZeroSamples);
        }
        if !z.is_finite() || z <= 0.0 {
            return Err(ScorecardError::BadConfidence { value: z });
        }
        let (ci_low, ci_high) = wilson_interval(point, sample_count, z);
        Ok(Self {
            point,
            ci_low,
            ci_high,
            sample_count,
            level,
        })
    }

    /// Construct a **synthetic** metric: the evidence level is forced to `L0`
    /// (ADR-282/ADR-300 — synthetic input is `L0` by construction and cannot be
    /// raised here).
    ///
    /// # Errors
    /// As [`Metric::new`].
    pub fn synthetic(point: f64, sample_count: u64) -> Result<Self, ScorecardError> {
        Self::new(point, sample_count, EvidenceLevel::L0)
    }

    /// The point estimate.
    #[must_use]
    pub fn point(&self) -> f64 {
        self.point
    }

    /// The lower confidence bound.
    #[must_use]
    pub fn ci_low(&self) -> f64 {
        self.ci_low
    }

    /// The upper confidence bound.
    #[must_use]
    pub fn ci_high(&self) -> f64 {
        self.ci_high
    }

    /// The confidence interval as `(low, high)`.
    #[must_use]
    pub fn ci(&self) -> (f64, f64) {
        (self.ci_low, self.ci_high)
    }

    /// The number of samples the estimate was computed from.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// The evidence level travelling with this cell.
    #[must_use]
    pub fn level(&self) -> EvidenceLevel {
        self.level
    }
}

// ---------------------------------------------------------------------------
// Cell: scored or explicitly no-evidence
// ---------------------------------------------------------------------------

/// One scorecard cell. A domain with no coverage is [`Cell::NoEvidence`] — a
/// first-class value the promotion gate treats as no coverage, never a pass
/// (ADR-317). It is deliberately distinct from a scored cell whose point
/// happens to be `0.0`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Cell {
    /// No coverage for this domain slice.
    NoEvidence,
    /// A scored measurement.
    Scored(Metric),
}

impl Cell {
    /// The point estimate if scored, else `None`.
    #[must_use]
    pub fn point(&self) -> Option<f64> {
        match self {
            Cell::NoEvidence => None,
            Cell::Scored(m) => Some(m.point),
        }
    }

    /// The evidence level if scored, else `None`.
    #[must_use]
    pub fn level(&self) -> Option<EvidenceLevel> {
        match self {
            Cell::NoEvidence => None,
            Cell::Scored(m) => Some(m.level),
        }
    }

    /// The scored metric, if any.
    #[must_use]
    pub fn metric(&self) -> Option<Metric> {
        match self {
            Cell::NoEvidence => None,
            Cell::Scored(m) => Some(*m),
        }
    }

    /// Whether this cell carries evidence.
    #[must_use]
    pub fn has_evidence(&self) -> bool {
        matches!(self, Cell::Scored(_))
    }
}

impl From<Metric> for Cell {
    fn from(m: Metric) -> Self {
        Cell::Scored(m)
    }
}

// ---------------------------------------------------------------------------
// Domain-state staleness guard (ADR-302 / ADR-300)
// ---------------------------------------------------------------------------

/// The ADR-302 domain state under the ADR-300 staleness guard
/// `VALID → DEGRADED → UNKNOWN`. A certificate is conditional on a continuously
/// evaluated domain signature; crossing the OOD threshold degrades the state
/// rather than silently continuing. `Unknown` is a first-class output, not an
/// error (ADR-300 rule 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DomainState {
    /// In-distribution; the certificate holds.
    Valid,
    /// Drifting; the affected capability is degraded pending recalibration.
    Degraded,
    /// Out of distribution; the surface answers UNKNOWN.
    Unknown,
}

/// The fraction of scored inferences observed in each [`DomainState`] over the
/// scoring window (ADR-317 calibration-drift axis). Fractions are each in
/// `[0, 1]` and sum to ~1.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateFractions {
    /// Fraction of inferences in `VALID`.
    pub valid: f64,
    /// Fraction of inferences in `DEGRADED`.
    pub degraded: f64,
    /// Fraction of inferences in `UNKNOWN`.
    pub unknown: f64,
}

impl StateFractions {
    /// Tolerance on the fractions summing to one.
    pub const SUM_EPS: f64 = 1e-6;

    /// Validate and construct. Each fraction must be finite in `[0, 1]` and the
    /// three must sum to `1 ± [`Self::SUM_EPS`]`.
    ///
    /// # Errors
    /// [`ScorecardError::BadStateFractions`].
    pub fn new(valid: f64, degraded: f64, unknown: f64) -> Result<Self, ScorecardError> {
        for v in [valid, degraded, unknown] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(ScorecardError::BadStateFractions {
                    reason: "each fraction must be finite in [0, 1]",
                });
            }
        }
        if (valid + degraded + unknown - 1.0).abs() > Self::SUM_EPS {
            return Err(ScorecardError::BadStateFractions {
                reason: "fractions must sum to 1",
            });
        }
        Ok(Self {
            valid,
            degraded,
            unknown,
        })
    }

    /// The dominant [`DomainState`] under the staleness guard. Monotone,
    /// documented thresholds: `UNKNOWN` when a majority of inferences fell out
    /// of distribution (`unknown >= 0.5`); otherwise `DEGRADED` when a majority
    /// were not `VALID` (`valid < 0.5`); otherwise `VALID`. This mirrors the
    /// `VALID → DEGRADED → UNKNOWN` progression: rising OOD mass walks the
    /// state strictly downward, never silently back up.
    #[must_use]
    pub fn dominant(&self) -> DomainState {
        if self.unknown >= 0.5 {
            DomainState::Unknown
        } else if self.valid < 0.5 {
            DomainState::Degraded
        } else {
            DomainState::Valid
        }
    }
}

// ---------------------------------------------------------------------------
// Slice identity
// ---------------------------------------------------------------------------

/// A capability task grouping domain slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    /// Presence detection.
    Presence,
    /// Pose estimation.
    Pose,
}

/// The identity of a single scorecard cell across every capability and domain.
/// Enumerable ([`SliceId::ALL`]) so `worst_domain`, the gate, and `render` all
/// walk the same canonical order deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceId {
    /// Presence on a room seen during training.
    PresenceRoomKnown,
    /// Presence on an unseen room (a pooled score hides regressions here).
    PresenceRoomUnseen,
    /// Presence on unseen device hardware.
    PresenceDeviceUnseen,
    /// Presence on a stationary subject at ~10 m — the canonical WiFi failure.
    PresenceStationary10m,
    /// Pose on a matched (in-distribution) split.
    PoseMatched,
    /// Pose on an unseen subject.
    PoseSubjectUnseen,
    /// Pose on an unseen room.
    PoseRoomUnseen,
    /// Rate of correctly rejecting out-of-distribution input as UNKNOWN.
    OodRejection,
    /// Calibration-drift fingerprint distance against the ADR-301 certificate.
    CalibrationDrift,
}

impl SliceId {
    /// Every slice in canonical render/iteration order.
    pub const ALL: [SliceId; 9] = [
        SliceId::PresenceRoomKnown,
        SliceId::PresenceRoomUnseen,
        SliceId::PresenceDeviceUnseen,
        SliceId::PresenceStationary10m,
        SliceId::PoseMatched,
        SliceId::PoseSubjectUnseen,
        SliceId::PoseRoomUnseen,
        SliceId::OodRejection,
        SliceId::CalibrationDrift,
    ];

    /// Human-readable label used by [`Scorecard::render`].
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            SliceId::PresenceRoomKnown => "presence/room-known",
            SliceId::PresenceRoomUnseen => "presence/room-unseen",
            SliceId::PresenceDeviceUnseen => "presence/device-unseen",
            SliceId::PresenceStationary10m => "presence/stationary-10m",
            SliceId::PoseMatched => "pose/matched",
            SliceId::PoseSubjectUnseen => "pose/subject-unseen",
            SliceId::PoseRoomUnseen => "pose/room-unseen",
            SliceId::OodRejection => "ood-rejection",
            SliceId::CalibrationDrift => "calibration-drift",
        }
    }

    /// The task this slice belongs to, if it is a per-domain accuracy task.
    /// OOD rejection and calibration drift are single cells and return `None`.
    #[must_use]
    pub fn task(self) -> Option<Task> {
        match self {
            SliceId::PresenceRoomKnown
            | SliceId::PresenceRoomUnseen
            | SliceId::PresenceDeviceUnseen
            | SliceId::PresenceStationary10m => Some(Task::Presence),
            SliceId::PoseMatched | SliceId::PoseSubjectUnseen | SliceId::PoseRoomUnseen => {
                Some(Task::Pose)
            }
            SliceId::OodRejection | SliceId::CalibrationDrift => None,
        }
    }

    /// Whether a *higher* point estimate is better. True for every accuracy /
    /// rejection slice; false for calibration drift, where a larger
    /// fingerprint distance is a regression.
    #[must_use]
    pub fn higher_is_better(self) -> bool {
        !matches!(self, SliceId::CalibrationDrift)
    }

    /// Whether this is a strict-budget domain (unseen / stationary / OOD) —
    /// the ones a pooled score hides, per ADR-317 §2.
    #[must_use]
    pub fn is_strict(self) -> bool {
        matches!(
            self,
            SliceId::PresenceRoomUnseen
                | SliceId::PresenceDeviceUnseen
                | SliceId::PresenceStationary10m
                | SliceId::PoseSubjectUnseen
                | SliceId::PoseRoomUnseen
                | SliceId::OodRejection
        )
    }

    /// Whether this slice contributes to the pooled accuracy average (every
    /// higher-is-better slice; drift has different units and direction and is
    /// excluded).
    #[must_use]
    fn is_pooled(self) -> bool {
        self.higher_is_better()
    }
}

// ---------------------------------------------------------------------------
// Scorecard
// ---------------------------------------------------------------------------

/// The multi-domain scorecard (ADR-317 §1): one [`Cell`] per operating domain,
/// never pooled into a single figure. Fields are public for direct
/// construction from an evidence query; every cell defaults to
/// [`Cell::NoEvidence`] via [`Scorecard::empty`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scorecard {
    /// Presence on a known room.
    pub presence_room_known: Cell,
    /// Presence on an unseen room.
    pub presence_room_unseen: Cell,
    /// Presence on unseen device hardware.
    pub presence_device_unseen: Cell,
    /// Presence on a stationary subject at ~10 m.
    pub presence_stationary_10m: Cell,
    /// Pose on a matched split.
    pub pose_matched: Cell,
    /// Pose on an unseen subject.
    pub pose_subject_unseen: Cell,
    /// Pose on an unseen room.
    pub pose_room_unseen: Cell,
    /// OOD-rejection rate.
    pub ood_rejection: Cell,
    /// Calibration-drift fingerprint distance (lower is better).
    pub calibration_drift: Cell,
    /// Fraction of inferences per [`DomainState`] over the window, if tracked.
    pub state_fractions: Option<StateFractions>,
}

impl Default for Scorecard {
    fn default() -> Self {
        Self::empty()
    }
}

impl Scorecard {
    /// An all-`NoEvidence` scorecard. Fill the cells that have coverage; the
    /// rest stay honestly empty.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            presence_room_known: Cell::NoEvidence,
            presence_room_unseen: Cell::NoEvidence,
            presence_device_unseen: Cell::NoEvidence,
            presence_stationary_10m: Cell::NoEvidence,
            pose_matched: Cell::NoEvidence,
            pose_subject_unseen: Cell::NoEvidence,
            pose_room_unseen: Cell::NoEvidence,
            ood_rejection: Cell::NoEvidence,
            calibration_drift: Cell::NoEvidence,
            state_fractions: None,
        }
    }

    /// The cell for a given slice.
    #[must_use]
    pub fn cell(&self, slice: SliceId) -> Cell {
        match slice {
            SliceId::PresenceRoomKnown => self.presence_room_known,
            SliceId::PresenceRoomUnseen => self.presence_room_unseen,
            SliceId::PresenceDeviceUnseen => self.presence_device_unseen,
            SliceId::PresenceStationary10m => self.presence_stationary_10m,
            SliceId::PoseMatched => self.pose_matched,
            SliceId::PoseSubjectUnseen => self.pose_subject_unseen,
            SliceId::PoseRoomUnseen => self.pose_room_unseen,
            SliceId::OodRejection => self.ood_rejection,
            SliceId::CalibrationDrift => self.calibration_drift,
        }
    }

    /// The dominant [`DomainState`] under the staleness guard, if state
    /// fractions were tracked. Absent tracking is `None`, not `Valid` — the
    /// scorecard never invents a healthy state it did not observe.
    #[must_use]
    pub fn domain_state(&self) -> Option<DomainState> {
        self.state_fractions.map(|f| f.dominant())
    }

    /// The **worst domain** for a task: the promotion-relevant number
    /// (ADR-300 §4). Returns the slice with the lowest point estimate, with a
    /// [`Cell::NoEvidence`] slice ranking below any scored cell — an uncovered
    /// domain is the worst possible outcome, never silently skipped.
    ///
    /// Every task in this scorecard is higher-is-better, so "worst" is
    /// unambiguously the minimum. Returns the first slice in canonical order on
    /// a tie for determinism.
    #[must_use]
    pub fn worst_domain(&self, task: Task) -> WorstCell {
        let mut worst: Option<WorstCell> = None;
        for slice in SliceId::ALL {
            if slice.task() != Some(task) {
                continue;
            }
            let cell = self.cell(slice);
            let candidate = WorstCell { slice, cell };
            worst = Some(match worst {
                None => candidate,
                Some(cur) => {
                    if candidate.is_worse_than(&cur) {
                        candidate
                    } else {
                        cur
                    }
                }
            });
        }
        // Every task has at least one member slice, so this is always `Some`.
        worst.expect("task has at least one slice")
    }

    /// The pooled accuracy average across covered higher-is-better slices — the
    /// figure ADR-300 rule 4 forbids relying on alone. Provided precisely so
    /// [`promotion_gate`] can prove a regression was *hidden behind* a rising
    /// pool. `None` when no such slice is covered.
    #[must_use]
    pub fn pooled_accuracy(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0u64;
        for slice in SliceId::ALL {
            if !slice.is_pooled() {
                continue;
            }
            if let Some(p) = self.cell(slice).point() {
                sum += p;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    /// Render an ASCII scorecard approximating the ADR-317 layout: one row per
    /// domain slice with its point estimate, confidence interval, evidence
    /// level, and sample count; the per-task worst domain; the pooled figure
    /// (labelled as insufficient on its own); and the domain state. Empty
    /// slices print `no-evidence`, never a fabricated number.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("ADR-317 multi-domain scorecard\n");
        out.push_str(
            "  (per-domain; pooled accuracy is never sufficient for promotion, ADR-300 §4)\n",
        );
        out.push_str(
            "  slice                     point   ci_low  ci_high  level  samples\n",
        );
        out.push_str(
            "  ------------------------- ------- ------- -------- ------ -------\n",
        );
        for slice in SliceId::ALL {
            let cell = self.cell(slice);
            match cell {
                Cell::NoEvidence => {
                    out.push_str(&format!(
                        "  {:<25} {:>7} {:>7} {:>8} {:>6} {:>7}\n",
                        slice.label(),
                        "no-ev",
                        "-",
                        "-",
                        "-",
                        "0",
                    ));
                }
                Cell::Scored(m) => {
                    out.push_str(&format!(
                        "  {:<25} {:>7.4} {:>7.4} {:>8.4} {:>6} {:>7}\n",
                        slice.label(),
                        m.point(),
                        m.ci_low(),
                        m.ci_high(),
                        format!("{:?}", m.level()),
                        m.sample_count(),
                    ));
                }
            }
        }
        out.push('\n');
        for task in [Task::Presence, Task::Pose] {
            let worst = self.worst_domain(task);
            let shown = match worst.cell {
                Cell::NoEvidence => "no-evidence (no coverage)".to_string(),
                Cell::Scored(m) => format!("{:.4}", m.point()),
            };
            out.push_str(&format!(
                "  worst {:<9} -> {} = {}\n",
                format!("{:?}", task).to_lowercase(),
                worst.slice.label(),
                shown,
            ));
        }
        match self.pooled_accuracy() {
            Some(p) => out.push_str(&format!(
                "  pooled accuracy = {:.4}  (INSUFFICIENT ALONE — see worst-domain)\n",
                p
            )),
            None => out.push_str("  pooled accuracy = no-evidence\n"),
        }
        match self.domain_state() {
            Some(s) => out.push_str(&format!("  domain state = {:?}\n", s)),
            None => out.push_str("  domain state = untracked\n"),
        }
        out
    }
}

/// The result of [`Scorecard::worst_domain`]: which slice was worst and its
/// cell (which may be [`Cell::NoEvidence`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorstCell {
    /// The worst slice.
    pub slice: SliceId,
    /// Its cell.
    pub cell: Cell,
}

impl WorstCell {
    /// Order for "worseness": a [`Cell::NoEvidence`] cell is worse than any
    /// scored cell; among scored cells a lower point estimate is worse (every
    /// task is higher-is-better).
    fn is_worse_than(&self, other: &WorstCell) -> bool {
        match (self.cell.point(), other.cell.point()) {
            (None, None) => false,
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (Some(a), Some(b)) => a < b,
        }
    }

    /// The worst point estimate, if the worst cell was scored.
    #[must_use]
    pub fn point(&self) -> Option<f64> {
        self.cell.point()
    }
}

// ---------------------------------------------------------------------------
// Promotion gate
// ---------------------------------------------------------------------------

/// Per-capability regression budgets. Strict-budget domains
/// (unseen / stationary / OOD) carry the tightest tolerance because they are
/// the ones a pooled score hides (ADR-317 §2).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatePolicy {
    /// Budget for non-strict presence/pose slices (e.g. room-known, matched).
    pub base_tolerance: f64,
    /// Budget for strict domains (unseen / stationary / OOD).
    pub strict_tolerance: f64,
    /// Budget for a *rise* in calibration drift before it is a regression.
    pub drift_tolerance: f64,
}

impl Default for GatePolicy {
    /// Conservative defaults: a small base budget, a much tighter strict budget
    /// for the domains that hide behind a pool, and a small drift budget.
    fn default() -> Self {
        Self {
            base_tolerance: 0.02,
            strict_tolerance: 0.005,
            drift_tolerance: 0.01,
        }
    }
}

impl GatePolicy {
    /// The tolerance that applies to a slice.
    #[must_use]
    pub fn tolerance(&self, slice: SliceId) -> f64 {
        if slice == SliceId::CalibrationDrift {
            self.drift_tolerance
        } else if slice.is_strict() {
            self.strict_tolerance
        } else {
            self.base_tolerance
        }
    }
}

/// Why a slice regressed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RegressionKind {
    /// A higher-is-better point estimate dropped beyond tolerance.
    AccuracyDrop,
    /// Calibration drift rose beyond tolerance.
    DriftIncrease,
    /// A previously-covered domain lost all evidence — no coverage is never a
    /// pass (ADR-317 §Provenance).
    CoverageLoss,
}

/// A single per-domain regression found by [`promotion_gate`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Regression {
    /// The slice that regressed.
    pub slice: SliceId,
    /// The nature of the regression.
    pub kind: RegressionKind,
    /// The previous point estimate, if it was scored.
    pub prev: Option<f64>,
    /// The current point estimate, if it is scored.
    pub curr: Option<f64>,
    /// The tolerance that was exceeded.
    pub tolerance: f64,
}

/// The verdict of the promotion gate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateReport {
    /// True only when no domain regressed.
    pub passed: bool,
    /// Pooled accuracy of the previous scorecard, if computable.
    pub pooled_prev: Option<f64>,
    /// Pooled accuracy of the current scorecard, if computable.
    pub pooled_curr: Option<f64>,
    /// Whether the pooled average improved.
    pub pooled_improved: bool,
    /// Every per-domain regression found (empty iff `passed`).
    pub regressions: Vec<Regression>,
}

impl GateReport {
    /// True when the pooled average improved yet the gate still failed — the
    /// exact "regression hiding behind pooled accuracy" case ADR-300 rule 4
    /// exists to catch.
    #[must_use]
    pub fn hidden_behind_pooled(&self) -> bool {
        self.pooled_improved && !self.passed
    }
}

/// Compare a candidate scorecard against a baseline and decide promotion.
///
/// The gate **fails if any single domain regresses beyond its budget**, even
/// when the pooled average improved (ADR-317 §2, ADR-300 rule 4): improvement
/// on `room-known` cannot buy a regression on `room-unseen`. Because the gate
/// evaluates every domain independently, a regression can never hide behind a
/// flattering pool; [`GateReport::hidden_behind_pooled`] reports when exactly
/// that was attempted.
///
/// Rules per slice:
/// - baseline `NoEvidence`: nothing to regress from — skipped (a newly covered
///   or still-empty domain is not itself a regression).
/// - baseline scored, candidate `NoEvidence`: [`RegressionKind::CoverageLoss`]
///   — losing a covered domain is a failure, never a pass.
/// - both scored, higher-is-better: regression if
///   `curr < prev - tolerance(slice)`.
/// - both scored, calibration drift: regression if
///   `curr > prev + tolerance(slice)`.
#[must_use]
pub fn promotion_gate(prev: &Scorecard, curr: &Scorecard, policy: &GatePolicy) -> GateReport {
    let mut regressions = Vec::new();
    for slice in SliceId::ALL {
        let tol = policy.tolerance(slice);
        let prev_cell = prev.cell(slice);
        let curr_cell = curr.cell(slice);
        match (prev_cell.point(), curr_cell.point()) {
            (None, _) => {
                // No baseline for this domain: cannot regress below nothing.
            }
            (Some(_), None) => {
                regressions.push(Regression {
                    slice,
                    kind: RegressionKind::CoverageLoss,
                    prev: prev_cell.point(),
                    curr: None,
                    tolerance: tol,
                });
            }
            (Some(p), Some(c)) => {
                let regressed = if slice.higher_is_better() {
                    c < p - tol
                } else {
                    c > p + tol
                };
                if regressed {
                    regressions.push(Regression {
                        slice,
                        kind: if slice.higher_is_better() {
                            RegressionKind::AccuracyDrop
                        } else {
                            RegressionKind::DriftIncrease
                        },
                        prev: Some(p),
                        curr: Some(c),
                        tolerance: tol,
                    });
                }
            }
        }
    }

    let pooled_prev = prev.pooled_accuracy();
    let pooled_curr = curr.pooled_accuracy();
    let pooled_improved = matches!((pooled_prev, pooled_curr), (Some(a), Some(b)) if b > a);

    GateReport {
        passed: regressions.is_empty(),
        pooled_prev,
        pooled_curr,
        pooled_improved,
        regressions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(point: f64, n: u64) -> Cell {
        // Route the inputs through `black_box` so the Wilson math runs at
        // runtime (as it does in the ADR-149 flow, computed from live
        // evidence) rather than being const-folded — const-eval and the
        // runtime FPU can disagree at the last ULP, which is a compiler
        // artifact, not real non-determinism.
        let point = std::hint::black_box(point);
        let n = std::hint::black_box(n);
        Cell::Scored(Metric::synthetic(point, n).expect("valid metric"))
    }

    // ---- CI computation vs hand-computed fixtures -------------------------

    #[test]
    fn wilson_ci_matches_hand_computed_50_of_100() {
        // p=0.5, n=100, z=1.96 (approx): classic Wilson 95% CI ≈ [0.4038, 0.5962].
        let m = Metric::with_confidence(0.5, 100, EvidenceLevel::L0, 1.96).unwrap();
        assert!((m.ci_low() - 0.4038).abs() < 1e-3, "lo={}", m.ci_low());
        assert!((m.ci_high() - 0.5962).abs() < 1e-3, "hi={}", m.ci_high());
        // Interval is symmetric about the point at p=0.5.
        assert!(((m.ci_low() + m.ci_high()) / 2.0 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn wilson_ci_matches_hand_computed_10_of_10() {
        // p=1.0, n=10, z=1.96: Wilson lower bound ≈ 0.7225, upper clamps to 1.0.
        let m = Metric::with_confidence(1.0, 10, EvidenceLevel::L0, 1.96).unwrap();
        assert!((m.ci_low() - 0.7225).abs() < 1e-3, "lo={}", m.ci_low());
        assert!(m.ci_high() <= 1.0 && m.ci_high() > 0.999, "hi={}", m.ci_high());
    }

    #[test]
    fn wilson_ci_stays_in_unit_interval_at_extremes() {
        for &p in &[0.0, 1.0, 0.01, 0.99] {
            for &n in &[1u64, 5, 1000] {
                let (lo, hi) = wilson_interval(p, n, Z_95);
                assert!((0.0..=1.0).contains(&lo), "lo={lo} p={p} n={n}");
                assert!((0.0..=1.0).contains(&hi), "hi={hi} p={p} n={n}");
                assert!(lo <= hi);
            }
        }
    }

    #[test]
    fn ci_narrows_with_more_samples() {
        let few = Metric::synthetic(0.8, 10).unwrap();
        let many = Metric::synthetic(0.8, 10_000).unwrap();
        let w_few = few.ci_high() - few.ci_low();
        let w_many = many.ci_high() - many.ci_low();
        assert!(w_many < w_few, "expected tighter CI with more samples");
    }

    // ---- boundary validation ---------------------------------------------

    #[test]
    fn malformed_metric_input_is_an_error_not_a_panic() {
        assert_eq!(
            Metric::new(1.5, 10, EvidenceLevel::L0).unwrap_err(),
            ScorecardError::PointOutOfRange { value: 1.5 }
        );
        // NaN != NaN, so match the variant rather than compare the payload.
        assert!(matches!(
            Metric::new(f64::NAN, 10, EvidenceLevel::L0).unwrap_err(),
            ScorecardError::PointOutOfRange { .. }
        ));
        assert_eq!(
            Metric::new(0.5, 0, EvidenceLevel::L0).unwrap_err(),
            ScorecardError::ZeroSamples
        );
        assert!(matches!(
            Metric::with_confidence(0.5, 10, EvidenceLevel::L0, 0.0).unwrap_err(),
            ScorecardError::BadConfidence { .. }
        ));
    }

    #[test]
    fn synthetic_metric_is_l0_by_construction() {
        assert_eq!(Metric::synthetic(0.9, 100).unwrap().level(), EvidenceLevel::L0);
    }

    #[test]
    fn state_fractions_validate_and_pick_dominant() {
        assert_eq!(
            StateFractions::new(0.9, 0.08, 0.02).unwrap().dominant(),
            DomainState::Valid
        );
        assert_eq!(
            StateFractions::new(0.3, 0.6, 0.1).unwrap().dominant(),
            DomainState::Degraded
        );
        assert_eq!(
            StateFractions::new(0.2, 0.2, 0.6).unwrap().dominant(),
            DomainState::Unknown
        );
        assert!(StateFractions::new(0.5, 0.4, 0.4).is_err()); // sums to 1.3
        assert!(StateFractions::new(-0.1, 0.6, 0.5).is_err());
    }

    // ---- worst-domain selection ------------------------------------------

    #[test]
    fn worst_domain_is_the_minimum_slice() {
        let mut sc = Scorecard::empty();
        sc.presence_room_known = scored(0.95, 500);
        sc.presence_room_unseen = scored(0.70, 500);
        sc.presence_device_unseen = scored(0.82, 500);
        sc.presence_stationary_10m = scored(0.61, 500);
        let worst = sc.worst_domain(Task::Presence);
        assert_eq!(worst.slice, SliceId::PresenceStationary10m);
        assert_eq!(worst.point(), Some(0.61));
    }

    #[test]
    fn worst_domain_ranks_no_evidence_below_any_score() {
        let mut sc = Scorecard::empty();
        sc.presence_room_known = scored(0.95, 500);
        sc.presence_room_unseen = scored(0.10, 500);
        // device-unseen and stationary remain NoEvidence — no coverage is worst.
        let worst = sc.worst_domain(Task::Presence);
        assert!(matches!(worst.cell, Cell::NoEvidence));
        assert_eq!(worst.point(), None);
        // Canonical order breaks the NoEvidence tie deterministically.
        assert_eq!(worst.slice, SliceId::PresenceDeviceUnseen);
    }

    // ---- promotion gate ---------------------------------------------------

    #[test]
    fn gate_fails_on_hidden_worst_domain_regression_while_pooled_improves() {
        // Only two covered presence slices, so pooled == their mean.
        // prev pooled = (0.80 + 0.75)/2 = 0.775
        let mut prev = Scorecard::empty();
        prev.presence_room_known = scored(0.80, 1000);
        prev.presence_room_unseen = scored(0.75, 1000);

        // curr pooled = (0.95 + 0.65)/2 = 0.80  -> pooled IMPROVED
        // but room-unseen (strict budget) dropped 0.75 -> 0.65 -> regression.
        let mut curr = Scorecard::empty();
        curr.presence_room_known = scored(0.95, 1000);
        curr.presence_room_unseen = scored(0.65, 1000);

        let report = promotion_gate(&prev, &curr, &GatePolicy::default());
        assert!(report.pooled_improved, "pooled should have improved");
        assert!(!report.passed, "gate must fail on the hidden regression");
        assert!(report.hidden_behind_pooled());
        assert_eq!(report.regressions.len(), 1);
        assert_eq!(report.regressions[0].slice, SliceId::PresenceRoomUnseen);
        assert_eq!(report.regressions[0].kind, RegressionKind::AccuracyDrop);
    }

    #[test]
    fn gate_passes_on_across_the_board_improvement() {
        let mut prev = Scorecard::empty();
        prev.presence_room_known = scored(0.80, 1000);
        prev.presence_room_unseen = scored(0.70, 1000);
        prev.presence_device_unseen = scored(0.72, 1000);
        prev.presence_stationary_10m = scored(0.55, 1000);
        prev.pose_matched = scored(0.60, 1000);
        prev.ood_rejection = scored(0.90, 1000);
        prev.calibration_drift = scored(0.20, 1000);

        let mut curr = Scorecard::empty();
        curr.presence_room_known = scored(0.85, 1000);
        curr.presence_room_unseen = scored(0.74, 1000);
        curr.presence_device_unseen = scored(0.76, 1000);
        curr.presence_stationary_10m = scored(0.60, 1000);
        curr.pose_matched = scored(0.65, 1000);
        curr.ood_rejection = scored(0.93, 1000);
        curr.calibration_drift = scored(0.15, 1000); // drift down = better

        let report = promotion_gate(&prev, &curr, &GatePolicy::default());
        assert!(report.passed, "expected pass: {:?}", report.regressions);
        assert!(report.regressions.is_empty());
        assert!(!report.hidden_behind_pooled());
    }

    #[test]
    fn gate_fails_on_coverage_loss() {
        let mut prev = Scorecard::empty();
        prev.presence_room_unseen = scored(0.75, 1000);
        let curr = Scorecard::empty(); // lost the covered domain
        let report = promotion_gate(&prev, &curr, &GatePolicy::default());
        assert!(!report.passed);
        assert_eq!(report.regressions[0].kind, RegressionKind::CoverageLoss);
    }

    #[test]
    fn gate_fails_on_drift_increase() {
        let mut prev = Scorecard::empty();
        prev.calibration_drift = scored(0.10, 1000);
        let mut curr = Scorecard::empty();
        curr.calibration_drift = scored(0.30, 1000); // drift rose beyond budget
        let report = promotion_gate(&prev, &curr, &GatePolicy::default());
        assert!(!report.passed);
        assert_eq!(report.regressions[0].kind, RegressionKind::DriftIncrease);
    }

    #[test]
    fn small_move_within_tolerance_is_not_a_regression() {
        let mut prev = Scorecard::empty();
        prev.presence_room_known = scored(0.80, 1000); // base budget 0.02
        let mut curr = Scorecard::empty();
        curr.presence_room_known = scored(0.79, 1000); // within budget
        let report = promotion_gate(&prev, &curr, &GatePolicy::default());
        assert!(report.passed);
    }

    // ---- determinism ------------------------------------------------------

    fn sample_scorecard() -> Scorecard {
        let mut sc = Scorecard::empty();
        sc.presence_room_known = scored(0.91, 800);
        sc.presence_room_unseen = scored(0.68, 800);
        sc.presence_stationary_10m = scored(0.52, 400);
        sc.ood_rejection = scored(0.88, 300);
        sc.calibration_drift = scored(0.12, 800);
        sc.state_fractions = Some(StateFractions::new(0.7, 0.2, 0.1).unwrap());
        sc
    }

    #[test]
    fn render_and_gate_are_deterministic() {
        // Rendering and the gate are pure: the same inputs yield the same
        // output every time (no wall clock, no RNG).
        let a = sample_scorecard();
        let b = sample_scorecard();
        assert_eq!(a.render(), a.render());
        // Two independent builds render identically; 4-decimal formatting is
        // stable across any last-ULP difference the compiler may introduce by
        // const-folding one build differently from the other.
        assert_eq!(a.render(), b.render());
        assert_eq!(
            promotion_gate(&a, &b, &GatePolicy::default()),
            promotion_gate(&a, &b, &GatePolicy::default())
        );

        // Serialisation preserves the scorecard to reporting precision: a
        // JSON round-trip reproduces the same rendered scorecard. (Rendering
        // fixes precision at 4 decimals; the ADR-149 hash binding hashes the
        // reproducible reporting form, not raw f64 bits.)
        let back: Scorecard =
            serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(back.render(), a.render());
    }

    #[test]
    fn render_reports_no_evidence_not_a_number() {
        let sc = Scorecard::empty();
        let text = sc.render();
        assert!(text.contains("no-ev"));
        assert!(text.contains("no coverage"));
        assert!(text.contains("pooled accuracy = no-evidence"));
        assert!(text.contains("domain state = untracked"));
    }

    #[test]
    fn render_flags_pooled_as_insufficient() {
        let sc = sample_scorecard();
        let text = sc.render();
        assert!(text.contains("INSUFFICIENT ALONE"));
        assert!(text.contains("worst presence"));
        assert!(text.contains("domain state = Valid"));
    }
}

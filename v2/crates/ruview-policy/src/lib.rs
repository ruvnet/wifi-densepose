//! # `ruview-policy` — action authorization gate (ADR-318, ADR-297 phase 1)
//!
//! A capability certificate (ADR-315) is a statement of *knowledge*, not a
//! *grant of action*. The same certificate that is adequate to dim a light is
//! wholly inadequate to release a door lock. This crate is the authorization
//! layer that sits between governed spatial state and any actuator: given the
//! assurance an action demands and the live assurance actually available, it
//! returns [`Authorization::Allow`] or a **fail-closed**
//! [`Authorization::Deny`] that names the *specific* condition that failed.
//!
//! ## The four non-negotiable rules (ADR-297)
//!
//! - **UNKNOWN is a first-class value, never an error.** An UNKNOWN domain
//!   ([`DomainState::Unknown`]) does not raise — it *denies* high-assurance
//!   actions. It may still authorize a [`ActionClass::Convenience`] action if
//!   that class does not require a known domain, but the resulting
//!   [`Authorization::Allow`] *records* that it proceeded under UNKNOWN
//!   (`under_unknown_domain`).
//! - **Staleness guard `VALID → DEGRADED → UNKNOWN`.** A safety- or
//!   security-class action requires the live domain signature (ADR-299) to be
//!   `KNOWN`; a `DEGRADED` domain denies with [`FailedCondition::DomainDegraded`]
//!   and an `UNKNOWN` domain denies with [`FailedCondition::DomainNotKnown`].
//! - **Honesty / no silent optimism.** A missing or expired certificate, a
//!   certificate class below the floor, an over-ceiling uncertainty, an
//!   evidence level below the floor, or the *absence of any policy* all deny by
//!   default. Absence of a policy is not permission. No accuracy is claimed
//!   here; the crate ships the gating machinery only, and its test fixtures are
//!   SYNTHETIC / L0.
//!
//! The decision is a **pure function** of (action class, assurance inputs):
//! deterministic, clock-free (time is pre-reduced by the caller into a bool +
//! an age), free of randomness, bounded in allocation, and panic-free on
//! malformed input (a `NaN` uncertainty fails closed rather than aborting).
//!
//! ## Adapter note — real certificate + OOD domain state → [`AssuranceInputs`]
//!
//! To stay parallel-buildable this crate does **not** depend on the concrete
//! `ruview-certify` / `ruview-ood` types; it owns [`AssuranceInputs`]. A caller
//! that *does* hold those types maps them on as follows:
//!
//! - `certificate_valid` ← the certificate's **time + signature** validity
//!   only: `cert.verify(key) && now < content.valid_until_unix_s`. Note this is
//!   deliberately *not* `CapabilityCertificate::is_valid`, which also folds the
//!   live domain in — the domain gate is applied *separately* by this policy so
//!   that an out-of-domain deny is attributed to the domain condition
//!   ([`FailedCondition::DomainNotKnown`]) rather than being hidden inside a
//!   generic "certificate invalid".
//! - `certificate_age` ← `now - content.calibrated_date_unix_s`, clamped at 0.
//! - `certificate_class` ← the ADR-315 assurance tier the certificate was
//!   minted at (derived by the caller from the certificate's evidence floor and
//!   validated capability); see [`CertificateClass`].
//! - `domain_state` ← `ruview_ood::DomainState`: `Known → `[`DomainState::Known`],
//!   `Degraded(_) → `[`DomainState::Degraded`], `Unknown(_) → `[`DomainState::Unknown`].
//! - `uncertainty` ← the model head's live predictive uncertainty (ADR-299/301).
//! - `evidence_level` ← the certificate's [`EvidenceLevel`] (ADR-282/301).
//!
//! Every allow or deny is intended to be emitted as the terminal stage of the
//! witness chain (ADR-316); this crate returns the decision, the caller records
//! it.

#![forbid(unsafe_code)]

use ruview_evidence::EvidenceLevel;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Value types owned by this crate
// ---------------------------------------------------------------------------

/// The assurance tier a certificate was minted at (ADR-315). Ordering is
/// meaningful and load-bearing: an action declares a
/// [`AssuranceRequirements::min_certificate_class`] and a certificate at a
/// class strictly below that floor is rejected. `Basic < Standard < High`.
///
/// This is a policy-side ladder: the ADR-315 certificate binds a capability and
/// an evidence level, and the adapter (see crate docs) derives the class from
/// them. Keeping the ladder local lets the policy crate build in parallel with
/// the certificate crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CertificateClass {
    /// Convenience-grade attestation: adequate to gate low-stakes actions.
    Basic,
    /// Security-grade attestation: bounded uncertainty, held-out evidence.
    Standard,
    /// Safety-grade attestation: the strictest tier, for actuators whose
    /// failure is unsafe.
    High,
}

/// Local, simplified mirror of the ADR-299 domain signature. The concrete
/// `ruview_ood::DomainState` carries a `DomainCause`; this policy only needs
/// the three-way outcome, so the cause is dropped at the adapter boundary (see
/// crate docs). `Known` is the only state that satisfies a "requires known
/// domain" action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DomainState {
    /// The live situation is recognized: inside the calibrated domain (ADR-299).
    Known,
    /// Drift/quality has crossed the inner envelope — degraded but not lost.
    Degraded,
    /// The situation is not recognized (ADR-299). A first-class value, never an
    /// error; it *denies* high-assurance actions rather than guessing.
    Unknown,
}

impl DomainState {
    /// `true` only for [`DomainState::Known`].
    #[must_use]
    pub const fn is_known(self) -> bool {
        matches!(self, DomainState::Known)
    }

    /// `true` only for [`DomainState::Unknown`].
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        matches!(self, DomainState::Unknown)
    }
}

/// The live assurance actually available at the moment of the decision. Owned
/// by this crate so it does not depend on the concrete certificate / OOD types
/// (see the crate-level adapter note for the mapping).
///
/// Time is pre-reduced by the caller: `certificate_valid` is the injected
/// time+signature validity and `certificate_age_secs` the injected age. This
/// keeps [`authorize`] a pure, clock-free function.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssuranceInputs {
    /// The certificate's assurance tier (ADR-315), derived by the adapter.
    pub certificate_class: CertificateClass,
    /// Whether the certificate is currently signed and unexpired (time +
    /// signature validity **only** — the domain gate is applied separately).
    /// `false` covers both a *missing* and an *expired* certificate: absence is
    /// not permission.
    pub certificate_valid: bool,
    /// Age of the certificate's calibration, in seconds (`now - calibrated_date`).
    pub certificate_age_secs: u64,
    /// The live domain signature (ADR-299), reduced to three states.
    pub domain_state: DomainState,
    /// The model head's live predictive uncertainty, in `[0.0, 1.0]`. A `NaN`
    /// or out-of-range value is treated as over any ceiling (fail-closed).
    pub uncertainty: f64,
    /// The evidence floor backing this inference (ADR-282/301).
    pub evidence_level: EvidenceLevel,
}

/// The assurance an [`ActionClass`] demands (ADR-318 §1). Every field is a
/// gate; an input that fails any one denies.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssuranceRequirements {
    /// The certificate must be at least this class.
    pub min_certificate_class: CertificateClass,
    /// The certificate calibration must be no older than this (freshness).
    pub max_certificate_age_secs: u64,
    /// Inference uncertainty must not exceed this ceiling.
    pub max_uncertainty: f64,
    /// The evidence level must be at least this floor.
    pub min_evidence_level: EvidenceLevel,
    /// Whether the live domain must be [`DomainState::Known`]. When `true`, a
    /// `Degraded`/`Unknown` domain denies (the staleness guard). When `false`,
    /// an `Unknown` domain is allowed but recorded on the [`Authorization`].
    pub requires_domain_known: bool,
}

/// The class of action being authorized (ADR-318 §1). Each class declares the
/// assurance it demands via [`ActionClass::requirements`]. The classes are
/// reference defaults — illustrative and, in a fuller system, configurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActionClass {
    /// Lighting, scenes: tolerant — `Basic`+, higher uncertainty ok, does not
    /// require a known domain (but records an UNKNOWN proceed).
    Convenience,
    /// Alerts, arming: stricter — valid `Standard`+ cert, bounded uncertainty,
    /// requires a known domain.
    Security,
    /// Door lock, machine stop: strict — fresh `High` cert, low uncertainty,
    /// `L3`+ evidence, known domain only.
    SafetyCritical,
}

/// One day / one week / thirty days in seconds, for the reference freshness
/// ceilings below.
const ONE_DAY_SECS: u64 = 86_400;
const ONE_WEEK_SECS: u64 = 7 * ONE_DAY_SECS;
const THIRTY_DAYS_SECS: u64 = 30 * ONE_DAY_SECS;

impl ActionClass {
    /// The reference assurance requirements for this class (ADR-318 §1 table).
    #[must_use]
    pub const fn requirements(self) -> AssuranceRequirements {
        match self {
            ActionClass::Convenience => AssuranceRequirements {
                min_certificate_class: CertificateClass::Basic,
                max_certificate_age_secs: THIRTY_DAYS_SECS,
                max_uncertainty: 0.6,
                min_evidence_level: EvidenceLevel::L1,
                requires_domain_known: false,
            },
            ActionClass::Security => AssuranceRequirements {
                min_certificate_class: CertificateClass::Standard,
                max_certificate_age_secs: ONE_WEEK_SECS,
                max_uncertainty: 0.3,
                min_evidence_level: EvidenceLevel::L2,
                requires_domain_known: true,
            },
            ActionClass::SafetyCritical => AssuranceRequirements {
                min_certificate_class: CertificateClass::High,
                max_certificate_age_secs: ONE_DAY_SECS,
                max_uncertainty: 0.1,
                min_evidence_level: EvidenceLevel::L3,
                requires_domain_known: true,
            },
        }
    }
}

/// The specific condition that caused a [`Authorization::Deny`]. A denial always
/// names exactly one — the *first* unmet condition in the fixed evaluation
/// order — so "why was this actuator denied" is unambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailedCondition {
    /// No policy was supplied for the action — an unrecognized action class.
    /// Absence of a policy is not permission (ADR-318 §3).
    NoPolicy,
    /// The certificate is missing or expired (`certificate_valid == false`).
    CertificateInvalid,
    /// The certificate's class is below the action's floor.
    CertificateClassTooLow {
        /// The floor the action requires.
        required: CertificateClass,
        /// The class actually presented.
        actual: CertificateClass,
    },
    /// The certificate calibration is older than the freshness ceiling.
    CertificateStale {
        /// Actual age, seconds.
        age_secs: u64,
        /// Maximum permitted age, seconds.
        max_secs: u64,
    },
    /// The action requires a known domain and the live domain is `DEGRADED`.
    DomainDegraded,
    /// The action requires a known domain and the live domain is `UNKNOWN`
    /// (ADR-297 acceptance test: drift-invalidated capability denied at the
    /// actuator). This is the canonical `domain_not_known` failure.
    DomainNotKnown,
    /// Inference uncertainty exceeds the ceiling (a `NaN` lands here too).
    UncertaintyOverCeiling {
        /// The ceiling the action requires; the actual value is elided because
        /// `f64` is not `Eq`/`Hash`-friendly across the wire, but the ceiling
        /// names the boundary that was crossed.
        max_uncertainty: f64,
    },
    /// The evidence level is below the action's floor.
    EvidenceBelowFloor {
        /// The floor the action requires.
        required: EvidenceLevel,
        /// The level actually backing the inference.
        actual: EvidenceLevel,
    },
}

impl FailedCondition {
    /// A stable, lower-snake-case name for the condition. Useful for witness
    /// records and log lines; the acceptance test asserts the SafetyCritical
    /// drift case names `domain_not_known`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            FailedCondition::NoPolicy => "no_policy",
            FailedCondition::CertificateInvalid => "certificate_invalid",
            FailedCondition::CertificateClassTooLow { .. } => "certificate_class_too_low",
            FailedCondition::CertificateStale { .. } => "certificate_stale",
            FailedCondition::DomainDegraded => "domain_degraded",
            FailedCondition::DomainNotKnown => "domain_not_known",
            FailedCondition::UncertaintyOverCeiling { .. } => "uncertainty_over_ceiling",
            FailedCondition::EvidenceBelowFloor { .. } => "evidence_below_floor",
        }
    }
}

/// The authorization decision (ADR-318 §2). Fail-closed: anything that is not an
/// [`Authorization::Allow`] is a deny that names its condition.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Authorization {
    /// The action is authorized. `under_unknown_domain` is `true` only when a
    /// class that does *not* require a known domain (e.g.
    /// [`ActionClass::Convenience`]) was allowed while the domain was `UNKNOWN`
    /// — the allow is honest about having proceeded out-of-domain.
    Allow {
        /// Records that the allow proceeded while the domain was `UNKNOWN`.
        under_unknown_domain: bool,
    },
    /// The action is denied; `failed_condition` names the specific unmet gate.
    Deny {
        /// The first unmet condition in evaluation order.
        failed_condition: FailedCondition,
    },
}

impl Authorization {
    /// `true` only for [`Authorization::Allow`].
    #[must_use]
    pub const fn is_allowed(self) -> bool {
        matches!(self, Authorization::Allow { .. })
    }

    /// The failed condition, if this is a deny.
    #[must_use]
    pub const fn failed_condition(self) -> Option<FailedCondition> {
        match self {
            Authorization::Deny { failed_condition } => Some(failed_condition),
            Authorization::Allow { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// The decision
// ---------------------------------------------------------------------------

/// Authorize an action of `class` against the live `inputs` (ADR-318 §2).
///
/// A **pure**, fail-closed function of `(class, inputs)`: deterministic, no
/// clock, no randomness, no panics. It applies the class's reference
/// [`AssuranceRequirements`]; use [`authorize_with`] to supply custom
/// requirements or to model an unrecognized action (a `None` policy denies).
#[must_use]
pub fn authorize(class: ActionClass, inputs: &AssuranceInputs) -> Authorization {
    authorize_with(Some(&class.requirements()), inputs)
}

/// Authorize against an explicit, optional policy. `None` means *no policy was
/// found for this action* — an unrecognized action class — and denies with
/// [`FailedCondition::NoPolicy`] (absence of a policy is not permission,
/// ADR-318 §3).
///
/// Evaluation order (the first unmet condition is the one named):
/// 1. policy present,
/// 2. certificate valid (present + unexpired),
/// 3. certificate class ≥ floor,
/// 4. certificate age ≤ freshness ceiling,
/// 5. domain gate (when the class requires a known domain),
/// 6. uncertainty ≤ ceiling,
/// 7. evidence ≥ floor.
#[must_use]
pub fn authorize_with(
    requirements: Option<&AssuranceRequirements>,
    inputs: &AssuranceInputs,
) -> Authorization {
    let req = match requirements {
        Some(req) => req,
        None => {
            return Authorization::Deny {
                failed_condition: FailedCondition::NoPolicy,
            }
        }
    };

    // 2. A missing or expired certificate denies by default.
    if !inputs.certificate_valid {
        return Authorization::Deny {
            failed_condition: FailedCondition::CertificateInvalid,
        };
    }

    // 3. Certificate class must meet the floor.
    if inputs.certificate_class < req.min_certificate_class {
        return Authorization::Deny {
            failed_condition: FailedCondition::CertificateClassTooLow {
                required: req.min_certificate_class,
                actual: inputs.certificate_class,
            },
        };
    }

    // 4. Freshness / staleness ceiling on certificate age.
    if inputs.certificate_age_secs > req.max_certificate_age_secs {
        return Authorization::Deny {
            failed_condition: FailedCondition::CertificateStale {
                age_secs: inputs.certificate_age_secs,
                max_secs: req.max_certificate_age_secs,
            },
        };
    }

    // 5. Domain gate. A class that requires a known domain denies on
    //    DEGRADED/UNKNOWN, naming the specific state.
    if req.requires_domain_known {
        match inputs.domain_state {
            DomainState::Known => {}
            DomainState::Degraded => {
                return Authorization::Deny {
                    failed_condition: FailedCondition::DomainDegraded,
                }
            }
            DomainState::Unknown => {
                return Authorization::Deny {
                    failed_condition: FailedCondition::DomainNotKnown,
                }
            }
        }
    }

    // 6. Uncertainty ceiling. `!(<=)` catches NaN too, failing closed.
    if !(inputs.uncertainty <= req.max_uncertainty) {
        return Authorization::Deny {
            failed_condition: FailedCondition::UncertaintyOverCeiling {
                max_uncertainty: req.max_uncertainty,
            },
        };
    }

    // 7. Evidence floor.
    if inputs.evidence_level < req.min_evidence_level {
        return Authorization::Deny {
            failed_condition: FailedCondition::EvidenceBelowFloor {
                required: req.min_evidence_level,
                actual: inputs.evidence_level,
            },
        };
    }

    // All gates passed. Record if we proceeded under an UNKNOWN domain (only
    // reachable for a class that does not require a known domain).
    Authorization::Allow {
        under_unknown_domain: inputs.domain_state.is_unknown(),
    }
}

// ---------------------------------------------------------------------------
// Tests — all fixtures are SYNTHETIC / L0 (CLAUDE.md honesty rule).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A baseline SYNTHETIC input that *passes* every gate for `class`. Tests
    /// then mutate exactly one field to force a specific deny.
    fn passing(class: ActionClass) -> AssuranceInputs {
        let req = class.requirements();
        AssuranceInputs {
            certificate_class: req.min_certificate_class,
            certificate_valid: true,
            certificate_age_secs: 0,
            domain_state: DomainState::Known,
            uncertainty: req.max_uncertainty, // exactly at ceiling → allowed
            evidence_level: req.min_evidence_level, // exactly at floor → allowed
        }
    }

    #[test]
    fn baseline_passes_for_every_class() {
        for class in [
            ActionClass::Convenience,
            ActionClass::Security,
            ActionClass::SafetyCritical,
        ] {
            assert_eq!(
                authorize(class, &passing(class)),
                Authorization::Allow {
                    under_unknown_domain: false
                },
                "baseline should allow {class:?}",
            );
        }
    }

    #[test]
    fn absence_of_policy_denies() {
        let inputs = passing(ActionClass::SafetyCritical);
        assert_eq!(
            authorize_with(None, &inputs),
            Authorization::Deny {
                failed_condition: FailedCondition::NoPolicy
            },
        );
    }

    #[test]
    fn missing_or_expired_certificate_denies() {
        let mut inputs = passing(ActionClass::Convenience);
        inputs.certificate_valid = false;
        assert_eq!(
            authorize(ActionClass::Convenience, &inputs).failed_condition(),
            Some(FailedCondition::CertificateInvalid),
        );
    }

    #[test]
    fn certificate_class_too_low_denies() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.certificate_class = CertificateClass::Basic;
        assert_eq!(
            authorize(ActionClass::SafetyCritical, &inputs).failed_condition(),
            Some(FailedCondition::CertificateClassTooLow {
                required: CertificateClass::High,
                actual: CertificateClass::Basic,
            }),
        );
    }

    #[test]
    fn stale_certificate_denies() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.certificate_age_secs = ONE_DAY_SECS + 1;
        assert_eq!(
            authorize(ActionClass::SafetyCritical, &inputs).failed_condition(),
            Some(FailedCondition::CertificateStale {
                age_secs: ONE_DAY_SECS + 1,
                max_secs: ONE_DAY_SECS,
            }),
        );
    }

    #[test]
    fn uncertainty_over_ceiling_denies() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.uncertainty = 0.1 + 1e-6; // just above the 0.1 ceiling
        match authorize(ActionClass::SafetyCritical, &inputs).failed_condition() {
            Some(FailedCondition::UncertaintyOverCeiling { .. }) => {}
            other => panic!("expected uncertainty deny, got {other:?}"),
        }
    }

    #[test]
    fn nan_uncertainty_fails_closed() {
        let mut inputs = passing(ActionClass::Convenience);
        inputs.uncertainty = f64::NAN;
        match authorize(ActionClass::Convenience, &inputs).failed_condition() {
            Some(FailedCondition::UncertaintyOverCeiling { .. }) => {}
            other => panic!("NaN uncertainty must fail closed, got {other:?}"),
        }
    }

    #[test]
    fn evidence_below_floor_denies() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.evidence_level = EvidenceLevel::L2; // floor is L3
        assert_eq!(
            authorize(ActionClass::SafetyCritical, &inputs).failed_condition(),
            Some(FailedCondition::EvidenceBelowFloor {
                required: EvidenceLevel::L3,
                actual: EvidenceLevel::L2,
            }),
        );
    }

    #[test]
    fn unknown_domain_denies_security() {
        let mut inputs = passing(ActionClass::Security);
        inputs.domain_state = DomainState::Unknown;
        assert_eq!(
            authorize(ActionClass::Security, &inputs).failed_condition(),
            Some(FailedCondition::DomainNotKnown),
        );
    }

    #[test]
    fn unknown_domain_denies_safety_critical() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.domain_state = DomainState::Unknown;
        assert_eq!(
            authorize(ActionClass::SafetyCritical, &inputs).failed_condition(),
            Some(FailedCondition::DomainNotKnown),
        );
    }

    #[test]
    fn degraded_domain_denies_high_assurance_with_its_own_condition() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.domain_state = DomainState::Degraded;
        assert_eq!(
            authorize(ActionClass::SafetyCritical, &inputs).failed_condition(),
            Some(FailedCondition::DomainDegraded),
        );
    }

    #[test]
    fn convenience_may_proceed_under_unknown_but_records_it() {
        let mut inputs = passing(ActionClass::Convenience);
        inputs.domain_state = DomainState::Unknown;
        assert_eq!(
            authorize(ActionClass::Convenience, &inputs),
            Authorization::Allow {
                under_unknown_domain: true
            },
        );

        // Degraded convenience is allowed and is not "under unknown".
        inputs.domain_state = DomainState::Degraded;
        assert_eq!(
            authorize(ActionClass::Convenience, &inputs),
            Authorization::Allow {
                under_unknown_domain: false
            },
        );
    }

    /// ADR-297 / ADR-318 acceptance-test B: a post-drift UNKNOWN domain causes a
    /// `SafetyCritical` authorize() to Deny with `domain_not_known`, *before*
    /// the inference reaches the actuator. The certificate is otherwise valid
    /// (signed, unexpired, correct class, fresh) — the domain gate is what
    /// stops it.
    #[test]
    fn acceptance_test_b_post_drift_unknown_denies_safety_critical() {
        // Pre-drift: domain KNOWN → the safety-critical action is authorized.
        let mut inputs = passing(ActionClass::SafetyCritical);
        assert!(authorize(ActionClass::SafetyCritical, &inputs).is_allowed());

        // Drift drives the domain to UNKNOWN (ADR-299 VALID→DEGRADED→UNKNOWN).
        inputs.domain_state = DomainState::Unknown;
        let decision = authorize(ActionClass::SafetyCritical, &inputs);

        assert_eq!(
            decision,
            Authorization::Deny {
                failed_condition: FailedCondition::DomainNotKnown
            },
        );
        assert_eq!(
            decision.failed_condition().map(FailedCondition::name),
            Some("domain_not_known"),
        );
    }

    #[test]
    fn every_deny_names_a_condition() {
        // Force a deny in each class and assert the decision carries a named
        // condition (never a bare/empty deny).
        let cases = [
            (ActionClass::Convenience, {
                let mut i = passing(ActionClass::Convenience);
                i.certificate_valid = false;
                i
            }),
            (ActionClass::Security, {
                let mut i = passing(ActionClass::Security);
                i.domain_state = DomainState::Unknown;
                i
            }),
            (ActionClass::SafetyCritical, {
                let mut i = passing(ActionClass::SafetyCritical);
                i.evidence_level = EvidenceLevel::L0;
                i
            }),
        ];
        for (class, inputs) in cases {
            let decision = authorize(class, &inputs);
            let cond = decision
                .failed_condition()
                .expect("deny must name a condition");
            assert!(
                !cond.name().is_empty(),
                "{class:?} deny must have a non-empty condition name",
            );
        }
    }

    #[test]
    fn decision_is_deterministic() {
        let inputs = passing(ActionClass::SafetyCritical);
        let first = authorize(ActionClass::SafetyCritical, &inputs);
        for _ in 0..1_000 {
            assert_eq!(authorize(ActionClass::SafetyCritical, &inputs), first);
        }
    }

    /// The full authorization matrix:
    /// (cert valid / invalid) × (age fresh / stale) × (Known/Degraded/Unknown)
    /// × (uncertainty below / above ceiling) × (evidence above / below floor).
    /// Asserts the outcome and, for every deny, that a condition is named.
    #[test]
    fn full_matrix() {
        for class in [
            ActionClass::Convenience,
            ActionClass::Security,
            ActionClass::SafetyCritical,
        ] {
            let req = class.requirements();
            for cert_valid in [true, false] {
                for age in [0u64, req.max_certificate_age_secs + 1] {
                    for domain in [
                        DomainState::Known,
                        DomainState::Degraded,
                        DomainState::Unknown,
                    ] {
                        // "below ceiling" = ceiling itself (allowed, since <=);
                        // "above ceiling" = ceiling + a hair.
                        for &unc in &[req.max_uncertainty, req.max_uncertainty + 0.01] {
                            for evidence in [req.min_evidence_level, EvidenceLevel::L0] {
                                let inputs = AssuranceInputs {
                                    certificate_class: req.min_certificate_class,
                                    certificate_valid: cert_valid,
                                    certificate_age_secs: age,
                                    domain_state: domain,
                                    uncertainty: unc,
                                    evidence_level: evidence,
                                };
                                let decision = authorize(class, &inputs);

                                // Compute the expected outcome independently.
                                let unc_ok = unc <= req.max_uncertainty;
                                let evidence_ok = evidence >= req.min_evidence_level;
                                let age_ok = age <= req.max_certificate_age_secs;
                                let domain_ok = !req.requires_domain_known || domain.is_known();
                                let should_allow =
                                    cert_valid && age_ok && domain_ok && unc_ok && evidence_ok;

                                if should_allow {
                                    let under_unknown = !req.requires_domain_known
                                        && domain == DomainState::Unknown;
                                    assert_eq!(
                                        decision,
                                        Authorization::Allow {
                                            under_unknown_domain: under_unknown
                                        },
                                        "class {class:?} inputs {inputs:?}",
                                    );
                                } else {
                                    assert!(
                                        !decision.is_allowed(),
                                        "class {class:?} inputs {inputs:?} should deny",
                                    );
                                    assert!(
                                        decision.failed_condition().is_some(),
                                        "deny must name a condition for {inputs:?}",
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn serde_round_trips_the_decision() {
        let mut inputs = passing(ActionClass::SafetyCritical);
        inputs.domain_state = DomainState::Unknown;
        let decision = authorize(ActionClass::SafetyCritical, &inputs);
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: Authorization = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decision, back);
    }
}

//! ADR-141 — BFLD privacy **control plane**: named modes, enforced actions,
//! and a hash-chained runtime attestation.
//!
//! The existing [`PrivacyClass`](crate::PrivacyClass) (ADR-120, 4 byte-level
//! classes) describes *what a frame contains*. This module adds the *policy*
//! layer on top: a [`PrivacyMode`] (the operator-facing posture) maps to a
//! target [`PrivacyClass`] plus a set of enforced [`PrivacyAction`]s, and a
//! [`PrivacyModeRegistry`] makes the active mode the single source of truth that
//! the privacy gate and the ADR-139/140 layers consult. Every mode change emits
//! a [`PrivacyAttestationProof`] that is BLAKE3 hash-chained to the previous one
//! (ADR-010 witness-chain pattern), so an auditor can verify the privacy posture
//! was continuous and untampered.

use crate::PrivacyClass;

/// Operator-facing privacy posture (ADR-141 §2). Layered over the 4-class
/// [`PrivacyClass`]; selecting a mode pins the target class and enforced actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyMode {
    /// Local research: raw BFI retained, full fidelity. Maps to `Raw`.
    RawResearch,
    /// Home default: room-level occupancy, no identity. Maps to `Anonymous`.
    PrivateHome,
    /// Multi-tenant anonymous: aggregate only, multi-seed. Maps to `Anonymous`.
    EnterpriseAnonymous,
    /// Care deployment with explicit consent: identity-derived fields allowed
    /// (Soul Signature enabled). Maps to `Derived`.
    CareWithConsent,
    /// Regulated: no identity surface whatsoever. Maps to `Restricted`.
    StrictNoIdentity,
}

/// A concrete enforcement action a mode may require (ADR-141 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrivacyAction {
    /// No restriction beyond the class minimum.
    Allow = 0,
    /// Strip identity-derived fields (embedding, risk score, hash).
    SuppressIdentity = 1,
    /// Reduce angular/spatial resolution before emission.
    ReduceResolution = 2,
    /// Never retain or emit raw BFI.
    DropRaw = 3,
    /// Emit only aggregate counts, never per-entity records.
    AggregateOnly = 4,
}

impl PrivacyAction {
    /// All actions in canonical (bit) order — used to encode an action set.
    pub const ALL: [PrivacyAction; 5] = [
        PrivacyAction::Allow,
        PrivacyAction::SuppressIdentity,
        PrivacyAction::ReduceResolution,
        PrivacyAction::DropRaw,
        PrivacyAction::AggregateOnly,
    ];
}

impl PrivacyMode {
    /// The byte-level [`PrivacyClass`] this mode pins (ADR-141 §2).
    #[must_use]
    pub const fn target_class(self) -> PrivacyClass {
        match self {
            Self::RawResearch => PrivacyClass::Raw,
            Self::PrivateHome | Self::EnterpriseAnonymous => PrivacyClass::Anonymous,
            Self::CareWithConsent => PrivacyClass::Derived,
            Self::StrictNoIdentity => PrivacyClass::Restricted,
        }
    }

    /// Whether Soul-Signature (identity-derived) processing is permitted.
    #[must_use]
    pub const fn soul_signature_enabled(self) -> bool {
        matches!(self, Self::RawResearch | Self::CareWithConsent)
    }

    /// The actions this mode enforces, encoded as a bitset over
    /// [`PrivacyAction`] (bit `i` set ⇒ `PrivacyAction::ALL[i]` enforced).
    #[must_use]
    pub const fn action_bits(self) -> u8 {
        // Helper bit positions.
        const SUP: u8 = 1 << 1; // SuppressIdentity
        const RED: u8 = 1 << 2; // ReduceResolution
        const DROP: u8 = 1 << 3; // DropRaw
        const AGG: u8 = 1 << 4; // AggregateOnly
        match self {
            Self::RawResearch => 1, // Allow only
            Self::PrivateHome => SUP | DROP,
            Self::EnterpriseAnonymous => SUP | DROP | AGG,
            Self::CareWithConsent => 1, // Allow (consent granted)
            Self::StrictNoIdentity => SUP | RED | DROP | AGG,
        }
    }

    /// Whether `action` is enforced under this mode.
    #[must_use]
    pub fn enforces(self, action: PrivacyAction) -> bool {
        let bit = 1u8 << (action as u8);
        self.action_bits() & bit != 0
    }

    /// Stable mode byte for attestation hashing.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::RawResearch => 0,
            Self::PrivateHome => 1,
            Self::EnterpriseAnonymous => 2,
            Self::CareWithConsent => 3,
            Self::StrictNoIdentity => 4,
        }
    }
}

/// A hash-chained attestation that a given mode was active (ADR-141 §2 / ADR-010).
///
/// `hash = BLAKE3(prev_hash ‖ mode_byte ‖ action_bits ‖ class_byte)`. Chaining
/// `prev_hash` gives cryptographic continuity: an auditor replays the chain and
/// any gap or tamper breaks the hash linkage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrivacyAttestationProof {
    /// Active mode at attestation time.
    pub mode: PrivacyMode,
    /// Enforced-action bitset (mirrors [`PrivacyMode::action_bits`]).
    pub action_bits: u8,
    /// Target class byte.
    pub class: u8,
    /// Hash of the previous proof (`[0; 32]` for the genesis proof).
    pub prev_hash: [u8; 32],
    /// BLAKE3 of `(prev_hash ‖ mode ‖ action_bits ‖ class)`.
    pub hash: [u8; 32],
}

// `compute` is only reachable through `PrivacyModeRegistry` (the std-gated
// audit log); without `std` there is no caller, so gate it to match and avoid
// a dead-code error under `--no-default-features` + `-D warnings`.
#[cfg(feature = "std")]
impl PrivacyAttestationProof {
    fn compute(mode: PrivacyMode, prev_hash: [u8; 32]) -> Self {
        let action_bits = mode.action_bits();
        let class = mode.target_class().as_u8();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&prev_hash);
        hasher.update(&[mode.as_u8(), action_bits, class]);
        let hash = *hasher.finalize().as_bytes();
        Self { mode, action_bits, class, prev_hash, hash }
    }
}

/// The active-mode source of truth (ADR-141 §2). The privacy gate and the
/// ADR-139/140 layers consult this; every mode change appends a hash-chained
/// attestation to the audit log.
///
/// `std`-gated because the audit log is heap-allocated (`Vec`), matching the
/// crate convention (the ESP32-S3 no_std self-only path uses a fixed-mode
/// posture without a growable log; see `frame.rs`).
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct PrivacyModeRegistry {
    active: PrivacyMode,
    audit_log: Vec<PrivacyAttestationProof>,
}

#[cfg(feature = "std")]
impl PrivacyModeRegistry {
    /// Create a registry with an initial mode (emits the genesis attestation).
    #[must_use]
    pub fn new(initial: PrivacyMode) -> Self {
        let genesis = PrivacyAttestationProof::compute(initial, [0u8; 32]);
        Self { active: initial, audit_log: vec![genesis] }
    }

    /// The currently active mode.
    #[must_use]
    pub fn active_mode(&self) -> PrivacyMode {
        self.active
    }

    /// The class the active mode pins.
    #[must_use]
    pub fn active_class(&self) -> PrivacyClass {
        self.active.target_class()
    }

    /// Whether the active mode enforces `action`.
    #[must_use]
    pub fn is_action_enforced(&self, action: PrivacyAction) -> bool {
        self.active.enforces(action)
    }

    /// Switch the active mode, appending a hash-chained attestation.
    pub fn set_mode(&mut self, mode: PrivacyMode) -> &PrivacyAttestationProof {
        let prev = self.audit_log.last().map(|p| p.hash).unwrap_or([0u8; 32]);
        self.active = mode;
        self.audit_log.push(PrivacyAttestationProof::compute(mode, prev));
        self.audit_log.last().unwrap()
    }

    /// The latest attestation proof (for HA/Matter diagnostics).
    #[must_use]
    pub fn latest_proof(&self) -> &PrivacyAttestationProof {
        self.audit_log.last().expect("registry always has a genesis proof")
    }

    /// The full attestation chain.
    #[must_use]
    pub fn audit_log(&self) -> &[PrivacyAttestationProof] {
        &self.audit_log
    }

    /// Verify the hash chain is continuous and untampered: each proof's
    /// `prev_hash` must equal the prior proof's `hash`, and every proof must
    /// recompute to its stored `hash`.
    #[must_use]
    pub fn verify_chain(&self) -> bool {
        let mut expected_prev = [0u8; 32];
        for proof in &self.audit_log {
            if proof.prev_hash != expected_prev {
                return false;
            }
            let recomputed = PrivacyAttestationProof::compute(proof.mode, proof.prev_hash);
            if recomputed.hash != proof.hash {
                return false;
            }
            expected_prev = proof.hash;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_to_class_mapping() {
        assert_eq!(PrivacyMode::RawResearch.target_class(), PrivacyClass::Raw);
        assert_eq!(PrivacyMode::PrivateHome.target_class(), PrivacyClass::Anonymous);
        assert_eq!(PrivacyMode::EnterpriseAnonymous.target_class(), PrivacyClass::Anonymous);
        assert_eq!(PrivacyMode::CareWithConsent.target_class(), PrivacyClass::Derived);
        assert_eq!(PrivacyMode::StrictNoIdentity.target_class(), PrivacyClass::Restricted);
    }

    #[test]
    fn soul_signature_only_in_raw_and_care() {
        assert!(PrivacyMode::RawResearch.soul_signature_enabled());
        assert!(PrivacyMode::CareWithConsent.soul_signature_enabled());
        assert!(!PrivacyMode::PrivateHome.soul_signature_enabled());
        assert!(!PrivacyMode::StrictNoIdentity.soul_signature_enabled());
    }

    #[test]
    fn action_enforcement() {
        assert!(PrivacyMode::StrictNoIdentity.enforces(PrivacyAction::SuppressIdentity));
        assert!(PrivacyMode::StrictNoIdentity.enforces(PrivacyAction::AggregateOnly));
        assert!(PrivacyMode::StrictNoIdentity.enforces(PrivacyAction::ReduceResolution));
        assert!(!PrivacyMode::RawResearch.enforces(PrivacyAction::SuppressIdentity));
        assert!(PrivacyMode::PrivateHome.enforces(PrivacyAction::DropRaw));
        assert!(!PrivacyMode::PrivateHome.enforces(PrivacyAction::AggregateOnly));
    }

    #[cfg(feature = "std")]
    #[test]
    fn registry_tracks_active_and_actions() {
        let mut reg = PrivacyModeRegistry::new(PrivacyMode::PrivateHome);
        assert_eq!(reg.active_class(), PrivacyClass::Anonymous);
        assert!(reg.is_action_enforced(PrivacyAction::SuppressIdentity));
        reg.set_mode(PrivacyMode::StrictNoIdentity);
        assert_eq!(reg.active_class(), PrivacyClass::Restricted);
        assert!(reg.is_action_enforced(PrivacyAction::AggregateOnly));
    }

    #[cfg(feature = "std")]
    #[test]
    fn attestation_chain_is_continuous_and_verifiable() {
        let mut reg = PrivacyModeRegistry::new(PrivacyMode::RawResearch);
        let g = *reg.latest_proof();
        assert_eq!(g.prev_hash, [0u8; 32], "genesis prev is zero");

        let p1 = *reg.set_mode(PrivacyMode::PrivateHome);
        assert_eq!(p1.prev_hash, g.hash, "chain links to genesis");
        let p2 = *reg.set_mode(PrivacyMode::StrictNoIdentity);
        assert_eq!(p2.prev_hash, p1.hash, "chain links forward");

        assert_eq!(reg.audit_log().len(), 3);
        assert!(reg.verify_chain(), "untampered chain verifies");
    }

    #[cfg(feature = "std")]
    #[test]
    fn tampered_chain_fails_verification() {
        let mut reg = PrivacyModeRegistry::new(PrivacyMode::RawResearch);
        reg.set_mode(PrivacyMode::PrivateHome);
        reg.set_mode(PrivacyMode::StrictNoIdentity);
        // Tamper: forge the middle proof's recorded mode without rehashing.
        reg.audit_log[1].mode = PrivacyMode::CareWithConsent;
        assert!(!reg.verify_chain(), "tamper breaks the hash linkage");
    }
}

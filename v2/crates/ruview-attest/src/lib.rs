//! `ruview-attest` — authenticated sensor identity and RF chain of custody.
//!
//! This crate implements **ADR-302** (authenticated sensor identity), phase 1 of
//! the ADR-297 perception substrate. It models the chain of custody link
//! `device → signed measurement → sequence → timestamp → payload hash →
//! calibration`, verified at the ingest boundary.
//!
//! ## Relationship to sibling ADRs
//!
//! - **ADR-293** shipped step one — a loopback-default UDP bind and an optional
//!   source IP/CIDR allowlist — and explicitly deferred "per-device provisioned
//!   keys, MAC/AEAD, device identifiers, monotonic sequence numbers, freshness
//!   window, and replay rejection." **This crate is that step two.** An IP
//!   allowlist does not stop on-subnet spoofing; a cryptographic device
//!   identity bound into each measurement does.
//! - **ADR-316** (witness chain) consumes the [`VerifiedMeasurement`] lineage
//!   produced here and serializes it for offline re-verification.
//!
//! ## Signer / Verifier abstraction and the SYNTHETIC reference
//!
//! Signing is expressed through the [`Signer`] and [`Verifier`] traits so a
//! production **Ed25519** asymmetric signer is a drop-in: implement the two
//! traits over a real keypair and the envelope, sequence, freshness, and tamper
//! logic here are unchanged.
//!
//! The bundled reference is [`Blake3MacSigner`], a keyed-BLAKE3 MAC. It is a
//! symmetric MAC, **not** an asymmetric signature: the verifier holds the same
//! secret the signer does, so it demonstrates the end-to-end custody logic but
//! confers no non-repudiation and no public-key trust boundary. Every accuracy
//! or spoof-resistance guarantee obtained with this reference signer is
//! **SYNTHETIC-grade** (CLAUDE.md evidence rule): a passing test suite exercises
//! the logic, never a fielded device. A deployment-grade claim requires an
//! Ed25519 signer plus real-silicon evidence.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum accepted byte length of a [`DeviceId`]. Bounds allocation at the
/// untrusted ingest boundary.
pub const MAX_DEVICE_ID_LEN: usize = 128;

/// Maximum accepted byte length of a [`CalibrationRef`].
pub const MAX_CALIBRATION_REF_LEN: usize = 128;

/// Width, in bytes, of a payload hash and of the reference MAC tag.
pub const TAG_LEN: usize = 32;

/// Domain-separation prefix mixed into the canonical signing bytes so a tag
/// produced here can never be confused with a hash produced for another purpose.
const DOMAIN: &[u8] = b"ruview-attest/v1\x00signed-measurement\x00";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure while constructing a value from untrusted input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InputError {
    /// A device identifier was empty.
    #[error("device id must not be empty")]
    EmptyDeviceId,
    /// A device identifier exceeded [`MAX_DEVICE_ID_LEN`].
    #[error("device id length {0} exceeds maximum {max}", max = MAX_DEVICE_ID_LEN)]
    DeviceIdTooLong(usize),
    /// A calibration reference exceeded [`MAX_CALIBRATION_REF_LEN`].
    #[error("calibration ref length {0} exceeds maximum {max}", max = MAX_CALIBRATION_REF_LEN)]
    CalibrationRefTooLong(usize),
}

/// Reason a [`SignedMeasurement`] was rejected at the verification boundary.
///
/// Every variant is a hard `Err`: a rejected frame is dropped and counted,
/// never a warning that proceeds (mirroring ADR-293's source-drop behaviour).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifyError {
    /// The measurement's `DeviceId` is not enrolled.
    #[error("device is not enrolled")]
    UnknownDevice,
    /// The signature/MAC did not verify over the canonical bytes.
    #[error("signature verification failed")]
    BadSignature,
    /// The carried payload hash did not match the presented payload.
    #[error("payload hash does not match presented payload (tamper)")]
    Tampered,
    /// The sequence number did not strictly increase for this device.
    #[error("sequence {got} is not greater than last accepted {last} (replay)")]
    Replay {
        /// The last sequence number this device successfully advanced to.
        last: u64,
        /// The offending non-increasing sequence number.
        got: u64,
    },
    /// The timestamp is older than the freshness window allows.
    #[error("timestamp is stale by {by_nanos} ns beyond the freshness window")]
    Stale {
        /// How far past the allowed age the timestamp fell, in nanoseconds.
        by_nanos: i64,
    },
    /// The timestamp is further in the future than the clock-skew budget allows.
    #[error("timestamp is {by_nanos} ns further ahead than the skew budget")]
    FutureDated {
        /// How far past the allowed skew the timestamp fell, in nanoseconds.
        by_nanos: i64,
    },
}

// ---------------------------------------------------------------------------
// Core value types
// ---------------------------------------------------------------------------

/// Authenticated device identity. Constructed only through [`DeviceId::new`],
/// which validates length at the boundary.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DeviceId(String);

impl DeviceId {
    /// Validate and wrap a device identifier. Rejects empty or oversized ids.
    pub fn new(id: impl Into<String>) -> Result<Self, InputError> {
        let id = id.into();
        if id.is_empty() {
            return Err(InputError::EmptyDeviceId);
        }
        if id.len() > MAX_DEVICE_ID_LEN {
            return Err(InputError::DeviceIdTooLong(id.len()));
        }
        Ok(Self(id))
    }

    /// Borrow the identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Server-injected timestamp, nanoseconds since an agreed epoch. Time is always
/// injected (never read from a wall clock inside this crate) so verification is
/// deterministic and testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub i64);

/// BLAKE3 hash of a measurement payload (CSI/CIR bytes). The payload itself is
/// *not* embedded in the envelope; only this hash is signed, so tampering is
/// detectable without carrying the payload twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadHash(pub [u8; TAG_LEN]);

impl PayloadHash {
    /// Compute the hash of a payload.
    pub fn of(payload: &[u8]) -> Self {
        Self(*blake3::hash(payload).as_bytes())
    }
}

/// Optional reference to a calibration certificate (ADR-298) in effect for a
/// measurement. Validated length at the boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationRef(String);

impl CalibrationRef {
    /// Validate and wrap a calibration reference.
    pub fn new(reference: impl Into<String>) -> Result<Self, InputError> {
        let reference = reference.into();
        if reference.len() > MAX_CALIBRATION_REF_LEN {
            return Err(InputError::CalibrationRefTooLong(reference.len()));
        }
        Ok(Self(reference))
    }

    /// Borrow the reference string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A signature/MAC tag over the canonical measurement bytes. Fixed width so a
/// malformed wire value cannot force an unbounded allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(pub [u8; TAG_LEN]);

// ---------------------------------------------------------------------------
// The signed envelope
// ---------------------------------------------------------------------------

/// The unsigned content bound by a signature: everything a verifier must be able
/// to reconstruct byte-for-byte to check the tag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementContent {
    /// Authenticated origin device.
    pub device: DeviceId,
    /// Strictly monotonic per-device sequence number (replay defense).
    pub sequence: u64,
    /// Device-asserted capture timestamp, checked against the freshness window.
    pub timestamp: Timestamp,
    /// Hash of the measurement payload (tamper detection).
    pub payload_hash: PayloadHash,
    /// Optional calibration certificate reference in effect.
    pub calibration_ref: Option<CalibrationRef>,
}

impl MeasurementContent {
    /// Deterministic, length-prefixed canonical serialization used as the
    /// signing input. Length prefixes make the encoding unambiguous (no field
    /// can be confused with another) and independent of any serde format.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(DOMAIN.len() + 96 + self.device.0.len());
        out.extend_from_slice(DOMAIN);
        push_field(&mut out, self.device.0.as_bytes());
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.timestamp.0.to_le_bytes());
        push_field(&mut out, &self.payload_hash.0);
        match &self.calibration_ref {
            Some(c) => {
                out.push(1);
                push_field(&mut out, c.0.as_bytes());
            }
            None => out.push(0),
        }
        out
    }
}

/// A [`MeasurementContent`] together with its signature. This is the object on
/// the wire and the unit the witness chain (ADR-316) serializes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedMeasurement {
    /// The signed content.
    pub content: MeasurementContent,
    /// The tag over [`MeasurementContent::canonical_bytes`].
    pub signature: Signature,
}

impl SignedMeasurement {
    /// Build a signed measurement from its parts using `signer`.
    pub fn sign<S: Signer + ?Sized>(
        signer: &S,
        device: DeviceId,
        sequence: u64,
        timestamp: Timestamp,
        payload: &[u8],
        calibration_ref: Option<CalibrationRef>,
    ) -> Self {
        let content = MeasurementContent {
            device,
            sequence,
            timestamp,
            payload_hash: PayloadHash::of(payload),
            calibration_ref,
        };
        let signature = signer.sign(&content.canonical_bytes());
        Self { content, signature }
    }
}

/// The trusted result of verification: proof that a measurement's origin,
/// sequence, freshness, and payload integrity were all checked. Carries the
/// verified chain-of-custody fields forward to calibration, inference, and the
/// witness chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedMeasurement {
    /// Verified origin device.
    pub device: DeviceId,
    /// Verified sequence number (strictly greater than the previous accepted).
    pub sequence: u64,
    /// Verified timestamp (within the freshness window).
    pub timestamp: Timestamp,
    /// Verified payload hash (matched the presented payload).
    pub payload_hash: PayloadHash,
    /// Calibration reference in effect, if any.
    pub calibration_ref: Option<CalibrationRef>,
}

// ---------------------------------------------------------------------------
// Signer / Verifier abstraction
// ---------------------------------------------------------------------------

/// Produces a signature over canonical measurement bytes. A production Ed25519
/// signer implements this over its private key.
pub trait Signer {
    /// Sign `message`, returning a fixed-width tag.
    fn sign(&self, message: &[u8]) -> Signature;
}

/// Verifies a signature over canonical measurement bytes. A production Ed25519
/// verifier implements this over the enrolled public key.
pub trait Verifier {
    /// Return `true` iff `signature` is valid for `message` under this identity.
    fn verify(&self, message: &[u8], signature: &Signature) -> bool;
}

/// **SYNTHETIC-grade reference** signer/verifier: a keyed-BLAKE3 MAC.
///
/// This is a symmetric MAC — the same secret signs and verifies — so it proves
/// the chain-of-custody logic but provides no non-repudiation. Do not read a
/// spoof-resistance guarantee from tests that use it (CLAUDE.md evidence rule).
/// Swap in an Ed25519 [`Signer`]/[`Verifier`] for a real asymmetric identity.
#[derive(Clone)]
pub struct Blake3MacSigner {
    key: [u8; TAG_LEN],
}

impl Blake3MacSigner {
    /// Construct from a 32-byte secret key.
    pub fn new(key: [u8; TAG_LEN]) -> Self {
        Self { key }
    }

    fn tag(&self, message: &[u8]) -> Signature {
        Signature(*blake3::keyed_hash(&self.key, message).as_bytes())
    }
}

impl Signer for Blake3MacSigner {
    fn sign(&self, message: &[u8]) -> Signature {
        self.tag(message)
    }
}

impl Verifier for Blake3MacSigner {
    fn verify(&self, message: &[u8], signature: &Signature) -> bool {
        constant_time_eq(&self.tag(message).0, &signature.0)
    }
}

// ---------------------------------------------------------------------------
// Freshness policy
// ---------------------------------------------------------------------------

/// Bounds a measurement timestamp against the injected server clock. Rejects
/// frames older than `max_age_nanos` (stale) or more than `max_skew_ahead_nanos`
/// in the future (clock-skew budget). Reuses ADR-292's freshness notion rather
/// than inventing a parallel one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreshnessPolicy {
    /// Maximum accepted age (`now - timestamp`) in nanoseconds.
    pub max_age_nanos: i64,
    /// Maximum accepted lead (`timestamp - now`) in nanoseconds.
    pub max_skew_ahead_nanos: i64,
}

impl FreshnessPolicy {
    /// A policy with the given symmetric window.
    pub fn new(max_age_nanos: i64, max_skew_ahead_nanos: i64) -> Self {
        Self {
            max_age_nanos: max_age_nanos.max(0),
            max_skew_ahead_nanos: max_skew_ahead_nanos.max(0),
        }
    }

    fn check(&self, timestamp: Timestamp, now: Timestamp) -> Result<(), VerifyError> {
        let delta = now.0.saturating_sub(timestamp.0); // positive => in the past
        if delta > self.max_age_nanos {
            return Err(VerifyError::Stale {
                by_nanos: delta - self.max_age_nanos,
            });
        }
        let ahead = timestamp.0.saturating_sub(now.0); // positive => in the future
        if ahead > self.max_skew_ahead_nanos {
            return Err(VerifyError::FutureDated {
                by_nanos: ahead - self.max_skew_ahead_nanos,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The verifier: enrollment + per-device sequence state
// ---------------------------------------------------------------------------

struct Enrolled<V: Verifier> {
    verifier: V,
    last_sequence: Option<u64>,
}

/// The ingest-boundary verifier. Holds enrolled device identities (a device is
/// untrusted until an operator enrolls its verifier) and the last accepted
/// sequence per device, and applies signature + monotonic-sequence + freshness
/// + tamper checks.
pub struct AttestationVerifier<V: Verifier> {
    enrolled: BTreeMap<DeviceId, Enrolled<V>>,
    freshness: FreshnessPolicy,
}

impl<V: Verifier> AttestationVerifier<V> {
    /// Create an empty verifier with the given freshness policy.
    pub fn new(freshness: FreshnessPolicy) -> Self {
        Self {
            enrolled: BTreeMap::new(),
            freshness,
        }
    }

    /// Enroll (or re-enroll) a device with the verifier for its identity. This
    /// is the explicit, authorized enrollment step from ADR-302; re-enrolling
    /// resets the device's sequence state.
    pub fn enroll(&mut self, device: DeviceId, verifier: V) {
        self.enrolled.insert(
            device,
            Enrolled {
                verifier,
                last_sequence: None,
            },
        );
    }

    /// Whether a device is enrolled.
    pub fn is_enrolled(&self, device: &DeviceId) -> bool {
        self.enrolled.contains_key(device)
    }

    /// The last accepted sequence for a device, if any.
    pub fn last_sequence(&self, device: &DeviceId) -> Option<u64> {
        self.enrolled.get(device).and_then(|e| e.last_sequence)
    }

    /// Verify a signed measurement against the presented `payload` at injected
    /// time `now`.
    ///
    /// Checks, in order: device enrolled → signature → payload-hash (tamper) →
    /// strictly-monotonic sequence (replay) → freshness. Per-device sequence
    /// state advances **only** on full success, so a rejected frame never
    /// consumes a sequence number.
    pub fn verify(
        &mut self,
        measurement: &SignedMeasurement,
        payload: &[u8],
        now: Timestamp,
    ) -> Result<VerifiedMeasurement, VerifyError> {
        let content = &measurement.content;

        let entry = self
            .enrolled
            .get_mut(&content.device)
            .ok_or(VerifyError::UnknownDevice)?;

        // Authenticate the envelope: the tag covers the payload *hash*, so a
        // valid signature also authenticates the hash field itself.
        if !entry
            .verifier
            .verify(&content.canonical_bytes(), &measurement.signature)
        {
            return Err(VerifyError::BadSignature);
        }

        // Tamper detection: the presented payload must match the signed hash.
        if PayloadHash::of(payload) != content.payload_hash {
            return Err(VerifyError::Tampered);
        }

        // Replay defense: strictly increasing sequence per device.
        if let Some(last) = entry.last_sequence {
            if content.sequence <= last {
                return Err(VerifyError::Replay {
                    last,
                    got: content.sequence,
                });
            }
        }

        // Freshness window.
        self.freshness.check(content.timestamp, now)?;

        // All checks passed: advance the accepted sequence and emit the
        // verified custody record.
        entry.last_sequence = Some(content.sequence);
        Ok(VerifiedMeasurement {
            device: content.device.clone(),
            sequence: content.sequence,
            timestamp: content.timestamp,
            payload_hash: content.payload_hash,
            calibration_ref: content.calibration_ref.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Append a `u32` little-endian length prefix followed by the bytes.
fn push_field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Constant-time equality over equal-length byte arrays.
fn constant_time_eq(a: &[u8; TAG_LEN], b: &[u8; TAG_LEN]) -> bool {
    let mut diff = 0u8;
    for i in 0..TAG_LEN {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; TAG_LEN] = [7u8; TAG_LEN];

    fn signer() -> Blake3MacSigner {
        Blake3MacSigner::new(KEY)
    }

    fn device() -> DeviceId {
        DeviceId::new("esp32-node-01").unwrap()
    }

    fn fresh_policy() -> FreshnessPolicy {
        // 1 second age budget, 100 ms future skew budget.
        FreshnessPolicy::new(1_000_000_000, 100_000_000)
    }

    fn make_verifier() -> AttestationVerifier<Blake3MacSigner> {
        let mut v = AttestationVerifier::new(fresh_policy());
        v.enroll(device(), signer());
        v
    }

    #[test]
    fn valid_measurement_verifies() {
        let mut v = make_verifier();
        let m = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(1000), b"csi-frame", None);
        let out = v.verify(&m, b"csi-frame", Timestamp(1000)).unwrap();
        assert_eq!(out.device, device());
        assert_eq!(out.sequence, 1);
        assert_eq!(out.timestamp, Timestamp(1000));
        assert_eq!(v.last_sequence(&device()), Some(1));
    }

    #[test]
    fn valid_with_calibration_ref_verifies() {
        let mut v = make_verifier();
        let cal = CalibrationRef::new("cal-cert-abc").unwrap();
        let m = SignedMeasurement::sign(
            &signer(),
            device(),
            5,
            Timestamp(2000),
            b"payload",
            Some(cal.clone()),
        );
        let out = v.verify(&m, b"payload", Timestamp(2000)).unwrap();
        assert_eq!(out.calibration_ref, Some(cal));
    }

    #[test]
    fn replayed_or_old_sequence_rejected() {
        let mut v = make_verifier();
        let now = Timestamp(5000);
        let m3 = SignedMeasurement::sign(&signer(), device(), 3, now, b"p", None);
        v.verify(&m3, b"p", now).unwrap();

        // Exact replay of sequence 3.
        assert_eq!(v.verify(&m3, b"p", now), Err(VerifyError::Replay { last: 3, got: 3 }));

        // Older sequence 2.
        let m2 = SignedMeasurement::sign(&signer(), device(), 2, now, b"p", None);
        assert_eq!(v.verify(&m2, b"p", now), Err(VerifyError::Replay { last: 3, got: 2 }));

        // A strictly greater sequence still works, and the rejected frames did
        // not consume a sequence slot.
        let m4 = SignedMeasurement::sign(&signer(), device(), 4, now, b"p", None);
        assert!(v.verify(&m4, b"p", now).is_ok());
        assert_eq!(v.last_sequence(&device()), Some(4));
    }

    #[test]
    fn stale_timestamp_rejected() {
        let mut v = make_verifier();
        // Captured at t=0, verified at t=2s with a 1s age budget => 1s stale.
        let m = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(0), b"p", None);
        assert_eq!(
            v.verify(&m, b"p", Timestamp(2_000_000_000)),
            Err(VerifyError::Stale { by_nanos: 1_000_000_000 })
        );
        // Rejected frame did not advance sequence state.
        assert_eq!(v.last_sequence(&device()), None);
    }

    #[test]
    fn future_dated_timestamp_rejected() {
        let mut v = make_verifier();
        // Captured 500ms in the future with a 100ms skew budget => 400ms over.
        let m = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(500_000_000), b"p", None);
        assert_eq!(
            v.verify(&m, b"p", Timestamp(0)),
            Err(VerifyError::FutureDated { by_nanos: 400_000_000 })
        );
    }

    #[test]
    fn tampered_payload_rejected() {
        let mut v = make_verifier();
        let m = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(0), b"real-payload", None);
        // Same envelope, but a different payload is presented at ingest.
        assert_eq!(v.verify(&m, b"evil-payload", Timestamp(0)), Err(VerifyError::Tampered));
        assert_eq!(v.last_sequence(&device()), None);
    }

    #[test]
    fn tampered_envelope_field_fails_signature() {
        let mut v = make_verifier();
        let mut m = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(0), b"p", None);
        // Flip the sequence without re-signing.
        m.content.sequence = 999;
        assert_eq!(v.verify(&m, b"p", Timestamp(0)), Err(VerifyError::BadSignature));
    }

    #[test]
    fn wrong_key_fails_signature() {
        let mut v = make_verifier();
        let attacker = Blake3MacSigner::new([9u8; TAG_LEN]);
        let m = SignedMeasurement::sign(&attacker, device(), 1, Timestamp(0), b"p", None);
        assert_eq!(v.verify(&m, b"p", Timestamp(0)), Err(VerifyError::BadSignature));
    }

    #[test]
    fn unknown_device_rejected() {
        let mut v = make_verifier();
        let stranger = DeviceId::new("rogue-node").unwrap();
        let m = SignedMeasurement::sign(&signer(), stranger, 1, Timestamp(0), b"p", None);
        assert_eq!(v.verify(&m, b"p", Timestamp(0)), Err(VerifyError::UnknownDevice));
    }

    #[test]
    fn signing_is_deterministic() {
        let a = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(42), b"p", None);
        let b = SignedMeasurement::sign(&signer(), device(), 1, Timestamp(42), b"p", None);
        assert_eq!(a, b);
        assert_eq!(a.signature, b.signature);
    }

    #[test]
    fn canonical_bytes_are_field_unambiguous() {
        // "ab" + "" must not collide with "a" + "b": length prefixes prevent it.
        let mk = |d: &str, cal: Option<&str>| MeasurementContent {
            device: DeviceId::new(d).unwrap(),
            sequence: 1,
            timestamp: Timestamp(0),
            payload_hash: PayloadHash::of(b""),
            calibration_ref: cal.map(|c| CalibrationRef::new(c).unwrap()),
        };
        assert_ne!(
            mk("ab", None).canonical_bytes(),
            mk("a", Some("b")).canonical_bytes()
        );
    }

    #[test]
    fn envelope_round_trips_through_serde() {
        let m = SignedMeasurement::sign(
            &signer(),
            device(),
            7,
            Timestamp(123),
            b"payload",
            Some(CalibrationRef::new("cal").unwrap()),
        );
        let json = serde_json::to_string(&m).unwrap();
        let back: SignedMeasurement = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);

        // A deserialized envelope still verifies end-to-end.
        let mut v = make_verifier();
        assert!(v.verify(&back, b"payload", Timestamp(123)).is_ok());
    }

    #[test]
    fn device_id_boundary_validation() {
        assert_eq!(DeviceId::new(""), Err(InputError::EmptyDeviceId));
        let long = "x".repeat(MAX_DEVICE_ID_LEN + 1);
        assert_eq!(
            DeviceId::new(long),
            Err(InputError::DeviceIdTooLong(MAX_DEVICE_ID_LEN + 1))
        );
        assert!(DeviceId::new("x").is_ok());
    }

    #[test]
    fn per_device_sequence_is_independent() {
        let mut v = AttestationVerifier::new(fresh_policy());
        let d1 = DeviceId::new("node-1").unwrap();
        let d2 = DeviceId::new("node-2").unwrap();
        v.enroll(d1.clone(), signer());
        v.enroll(d2.clone(), signer());

        let now = Timestamp(100);
        let m1 = SignedMeasurement::sign(&signer(), d1.clone(), 10, now, b"p", None);
        let m2 = SignedMeasurement::sign(&signer(), d2.clone(), 1, now, b"p", None);
        // d1 at seq 10 does not block d2 at seq 1.
        assert!(v.verify(&m1, b"p", now).is_ok());
        assert!(v.verify(&m2, b"p", now).is_ok());
        assert_eq!(v.last_sequence(&d1), Some(10));
        assert_eq!(v.last_sequence(&d2), Some(1));
    }
}

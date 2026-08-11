//! Signed, versioned, invalidatable room-fingerprint certificate (ADR-301).
//!
//! ADR-301 is primitive 1 of the perception-substrate program (ADR-300) and the
//! first brick of its "certificate spine". This module implements the
//! **certificate portion**: a portable, signed, expiring artifact that says
//! *"this is the room, here is when it was measured, and here is the evidence
//! that it is still the same room."*
//!
//! Nothing here re-derives room state. The [`RoomFingerprint`] is a bounded,
//! fixed-length statistical summary *reused* from the existing calibration types
//! — the [`SpecialistBank`](crate::bank::SpecialistBank)'s empty-vs-occupied
//! presence separation (ADR-135 baseline / ADR-151) and its transceiver
//! [`GeometryEmbedding`](crate::geometry_embedding::GeometryEmbedding)
//! (ADR-152). The [`CalibrationCertificate`] binds that fingerprint to a space,
//! a signing sensor identity, a monotonic version, a capture time, an expiry,
//! an evidence level (ADR-282), and a content hash suitable for signing.
//!
//! ## Honesty discipline (ADR-301 §"Provenance and honesty")
//!
//! A certificate produced from generated CSI is [`EvidenceLevel::L0Synthetic`]
//! by construction; [`CalibrationCertificate::mint`] **rejects** labelling a
//! synthetic characterization as measured, and rejects an automatic
//! ([`CalibrationTier::Auto`]) characterization claiming more than L2.
//!
//! ## Invalidation is explicit, not a silent `STALE` flag
//!
//! [`CalibrationCertificate::status`] returns a typed [`CertificateStatus`]:
//! valid, past-expiry, tampered signature, or drifted beyond the
//! [`CompatibilityEnvelope`]. Small drift stays inside the envelope and is
//! absorbed; drift beyond it invalidates the certificate and forces
//! re-characterization. Compensation never rewrites a signed certificate in
//! place — [`CalibrationCertificate::renew`] mints a *new* version, preserving
//! an append-only history.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::bank::SpecialistBank;
use crate::error::{CalibrationError, Result};
use crate::geometry_embedding::GeometryEmbedding;

/// Schema version for the [`RoomFingerprint`] wire format. Bumped when the
/// fingerprint's field set changes (ADR-301 §1: "its schema is versioned").
pub const FINGERPRINT_SCHEMA_VERSION: u32 = 1;

/// Schema version for the [`CalibrationCertificate`] wire format.
pub const CERTIFICATE_SCHEMA_VERSION: u32 = 1;

// Fixed, data-independent normalization scales for the fingerprint distance.
// Data-independent so `distance` is strictly monotonic under a single-field
// perturbation (a data-dependent denominator would grow with the perturbation
// and could mask it) — see `distance_is_monotonic` in the tests.
const MEAN_SCALE: f32 = 1.0;
const VAR_SCALE: f32 = 10.0;
const GEOM_SCALE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Evidence level, tier, characterization source
// ---------------------------------------------------------------------------

/// Evidence ladder (ADR-282 L0–L5). An automatic characterization on real
/// captured CSI is at most L1/L2 and is labelled as such, never L3+ (ADR-301
/// §3). L0 is reserved for synthetic/generated input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// Generated / synthetic CSI — no measured evidence (ADR-279 invariant 6).
    L0Synthetic,
    /// Weakest measured evidence (automatic observe-only, unverified).
    L1,
    /// Automatic characterization with a passing quality gate.
    L2,
    /// Guided enrollment or better (not reachable from `autocal`).
    L3,
    /// Cross-validated against a held-out split.
    L4,
    /// Independently reproduced on real silicon.
    L5,
}

impl EvidenceLevel {
    /// `true` for any measured level (L1+); L0 is synthetic.
    pub fn is_measured(self) -> bool {
        self != EvidenceLevel::L0Synthetic
    }

    /// Stable tag for canonical hashing.
    fn tag(self) -> u8 {
        match self {
            EvidenceLevel::L0Synthetic => 0,
            EvidenceLevel::L1 => 1,
            EvidenceLevel::L2 => 2,
            EvidenceLevel::L3 => 3,
            EvidenceLevel::L4 => 4,
            EvidenceLevel::L5 => 5,
        }
    }
}

/// How the fingerprint was characterized (ADR-301 §"Provenance and honesty").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterizationSource {
    /// Generated / simulated CSI. Forces [`EvidenceLevel::L0Synthetic`].
    Synthetic,
    /// Real captured CSI from a sensor.
    MeasuredCsi,
}

impl CharacterizationSource {
    fn tag(self) -> u8 {
        match self {
            CharacterizationSource::Synthetic => 0,
            CharacterizationSource::MeasuredCsi => 1,
        }
    }
}

/// Calibration tier (ADR-301 §1/§3): the automatic observe-only path yields a
/// weaker evidence level than guided enrollment; the certificate states which
/// path produced it so consumers can weight it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationTier {
    /// Automatic observe-only characterization (`autocal`). Capped at L2.
    Auto,
    /// Guided human enrollment (existing anchor ritual).
    Guided,
}

impl CalibrationTier {
    fn tag(self) -> u8 {
        match self {
            CalibrationTier::Auto => 0,
            CalibrationTier::Guided => 1,
        }
    }

    /// The strongest evidence level this tier may honestly claim.
    fn max_measured_evidence(self) -> EvidenceLevel {
        match self {
            CalibrationTier::Auto => EvidenceLevel::L2,
            CalibrationTier::Guided => EvidenceLevel::L5,
        }
    }
}

// ---------------------------------------------------------------------------
// Room fingerprint (reused summary of the existing calibration state)
// ---------------------------------------------------------------------------

/// A bounded, fixed-length statistical summary of a room's CSI distribution —
/// the distance-comparable object ADR-302 measures against.
///
/// It is *derived* from the existing calibration state, not a new measurement:
/// the empty-vs-occupied separation comes from the bank's
/// [`PresenceSpecialist`](crate::specialist::PresenceSpecialist) (ADR-135
/// baseline / ADR-151), and the geometry conditioning comes from the bank's
/// [`GeometryEmbedding`](crate::geometry_embedding::GeometryEmbedding)
/// (ADR-152). Both the **empty** distribution and the **occupied** distribution
/// are stored so downstream OOD gating can distinguish "the empty room changed"
/// (furniture/geometry drift) from "occupancy statistics changed".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoomFingerprint {
    /// Schema version ([`FINGERPRINT_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Empty-room scalar mean (static multipath load), from the presence gate's
    /// `empty_mean` reference. `0.0` when the bank learned no presence gate.
    pub empty_mean: f32,
    /// Empty-room band energy (variance), reconstructed from the presence gate's
    /// finite decision boundary; `0.0` when unavailable.
    pub empty_variance: f32,
    /// Occupied-room band energy (mean occupied-anchor variance).
    pub occupied_variance: f32,
    /// Learned variance decision boundary. Finite-guarded: an "inert" (issue
    /// #1440, `+inf`) boundary is stored as `0.0` so distance math stays finite.
    pub presence_threshold: f32,
    /// Empty→occupied mean separation (occupancy signal strength).
    pub occupancy_mean_shift: f32,
    /// Transceiver-geometry conditioning (ADR-152); all-zero when no geometry
    /// was recorded.
    pub geometry: GeometryEmbedding,
}

impl RoomFingerprint {
    /// Derive the fingerprint from a trained [`SpecialistBank`] — a pure
    /// function of the bank's existing state (no new capture).
    pub fn from_bank(bank: &SpecialistBank) -> Self {
        let (empty_mean, empty_variance, occupied_variance, presence_threshold, mean_shift) =
            match bank.presence.as_ref() {
                Some(p) => {
                    let threshold = if p.threshold.is_finite() {
                        p.threshold
                    } else {
                        0.0
                    };
                    // threshold == 0.5 * (empty_var + occupied_var) when finite, so
                    // empty_var reconstructs as 2*threshold - occupied_var (>= 0).
                    let empty_var = if p.threshold.is_finite() {
                        (2.0 * p.threshold - p.occupied_var).max(0.0)
                    } else {
                        0.0
                    };
                    // presence threshold == 0.5 * mean_dist ⇒ mean_dist == 2*threshold.
                    let mean_shift = p.mean_dist_threshold.map(|t| 2.0 * t).unwrap_or(0.0);
                    (
                        p.empty_mean,
                        empty_var,
                        p.occupied_var,
                        threshold,
                        mean_shift,
                    )
                }
                None => (0.0, 0.0, 0.0, 0.0, 0.0),
            };

        Self {
            schema_version: FINGERPRINT_SCHEMA_VERSION,
            empty_mean,
            empty_variance,
            occupied_variance,
            presence_threshold,
            occupancy_mean_shift: mean_shift,
            geometry: bank.geometry_embedding(),
        }
    }

    /// Bounded fingerprint distance to another fingerprint — the primitive
    /// ADR-302 uses to gate KNOWN → DEGRADED → UNKNOWN.
    ///
    /// Splits drift into an **empty-room** component (static multipath + physical
    /// geometry) and an **occupancy** component (dynamics), so a consumer can
    /// tell furniture/geometry drift from a different subject. The `total` is
    /// squashed into `[0, 1)` and is monotonic in any single-field perturbation.
    pub fn distance(&self, other: &RoomFingerprint) -> FingerprintDistance {
        let dmean = (self.empty_mean - other.empty_mean) / MEAN_SCALE;
        let dempty_var = (self.empty_variance - other.empty_variance) / VAR_SCALE;
        let geom_sq = geometry_l2_sq(&self.geometry, &other.geometry) / (GEOM_SCALE * GEOM_SCALE);
        let baseline_raw = (dmean * dmean + dempty_var * dempty_var + geom_sq).sqrt();

        let docc_var = (self.occupied_variance - other.occupied_variance) / VAR_SCALE;
        let dshift = (self.occupancy_mean_shift - other.occupancy_mean_shift) / MEAN_SCALE;
        let dthr = (self.presence_threshold - other.presence_threshold) / VAR_SCALE;
        let occupancy_raw = (docc_var * docc_var + dshift * dshift + dthr * dthr).sqrt();

        let raw = baseline_raw + occupancy_raw;
        FingerprintDistance {
            baseline_drift: baseline_raw,
            occupancy_drift: occupancy_raw,
            total: raw / (1.0 + raw),
        }
    }
}

fn geometry_l2_sq(a: &GeometryEmbedding, b: &GeometryEmbedding) -> f32 {
    a.as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// A drift/distance summary between two [`RoomFingerprint`]s.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FingerprintDistance {
    /// Empty-room drift: static multipath + transceiver geometry ("did the room
    /// itself change").
    pub baseline_drift: f32,
    /// Occupancy drift: how differently occupants perturb the field.
    pub occupancy_drift: f32,
    /// Total drift, squashed into `[0, 1)`. Monotonic in the underlying raw
    /// distance, so it is directly comparable against a [`CompatibilityEnvelope`].
    pub total: f32,
}

impl FingerprintDistance {
    /// `true` when total drift stays within the envelope (small drift absorbed).
    pub fn within_envelope(&self, envelope: &CompatibilityEnvelope) -> bool {
        self.total <= envelope.max_total_drift
    }
}

/// The compatibility envelope for continuous drift compensation (ADR-301 §4).
/// Drift within the envelope is absorbed and logged; drift beyond it invalidates
/// the certificate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompatibilityEnvelope {
    /// Maximum tolerated total fingerprint drift in `[0, 1)` before the
    /// certificate is invalidated and re-characterization is forced.
    pub max_total_drift: f32,
}

impl Default for CompatibilityEnvelope {
    fn default() -> Self {
        // A conservative default: modest drift is absorbed, a clearly different
        // room is rejected. Consumers (ADR-302) may tighten this per space.
        Self {
            max_total_drift: 0.15,
        }
    }
}

impl CompatibilityEnvelope {
    /// Validated constructor. Rejects a non-finite or out-of-`[0, 1)` envelope
    /// (bounded-input discipline at the config boundary).
    pub fn new(max_total_drift: f32) -> Result<Self> {
        if !max_total_drift.is_finite() || !(0.0..1.0).contains(&max_total_drift) {
            return Err(CalibrationError::InvalidCertificate(format!(
                "compatibility envelope must be finite in [0, 1), got {max_total_drift}"
            )));
        }
        Ok(Self { max_total_drift })
    }
}

// ---------------------------------------------------------------------------
// Signing abstraction (kept behind a trait, consistent with the crate's style)
// ---------------------------------------------------------------------------

/// Signs a certificate content hash. Kept behind a trait so the RuField
/// provenance/signature backend (ADR-260/262/277/279) can be substituted
/// without changing the certificate types. A signature is mandatory: an
/// unsigned certificate is not a valid certificate (ADR-301 §3).
pub trait CertificateSigner {
    /// Identity of the signing key (bound into the certificate as the signer).
    fn key_id(&self) -> &str;
    /// Produce a detached signature over the 32-byte content hash.
    fn sign(&self, content_hash: &[u8; 32]) -> Vec<u8>;
}

/// Verifies a detached signature over a certificate content hash.
pub trait CertificateVerifier {
    /// `true` iff `signature` is a valid signature by `key_id` over `content_hash`.
    fn verify(&self, key_id: &str, content_hash: &[u8; 32], signature: &[u8]) -> bool;
}

/// A deterministic keyed-hash signer/verifier (`SHA-256(secret‖hash‖secret)`).
///
/// This is a self-contained, dependency-free stand-in for the RuField signature
/// backend so the certificate machinery is testable today. It is a *keyed MAC*,
/// not asymmetric provenance — it is honest about being a placeholder and is
/// never labelled as the production RuField signature. Determinism makes signing
/// reproducible in tests; secrecy of `secret` gives tamper detection.
#[derive(Debug, Clone)]
pub struct KeyedHashSigner {
    key_id: String,
    secret: Vec<u8>,
}

impl KeyedHashSigner {
    /// Construct from a key identity and secret bytes.
    pub fn new(key_id: impl Into<String>, secret: impl Into<Vec<u8>>) -> Self {
        Self {
            key_id: key_id.into(),
            secret: secret.into(),
        }
    }

    fn mac(&self, content_hash: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(&self.secret);
        h.update(content_hash);
        h.update(&self.secret);
        h.finalize().into()
    }
}

impl CertificateSigner for KeyedHashSigner {
    fn key_id(&self) -> &str {
        &self.key_id
    }
    fn sign(&self, content_hash: &[u8; 32]) -> Vec<u8> {
        self.mac(content_hash).to_vec()
    }
}

impl CertificateVerifier for KeyedHashSigner {
    fn verify(&self, key_id: &str, content_hash: &[u8; 32], signature: &[u8]) -> bool {
        // Constant-time-ish comparison is out of scope for a placeholder; the
        // production RuField verifier owns that. Bind the key identity too.
        key_id == self.key_id && signature == self.mac(content_hash).as_slice()
    }
}

/// A detached signature bound to a content hash and a signing-key identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateSignature {
    /// Identity of the signing key (ADR-305 sensor identity binding).
    pub key_id: String,
    /// Lowercase-hex SHA-256 of the certificate's canonical signable bytes.
    pub content_hash_hex: String,
    /// Lowercase-hex detached signature over the content hash.
    pub signature_hex: String,
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------

/// Parameters for [`CalibrationCertificate::mint`]. Keeping them in one struct
/// avoids a long positional argument list and documents each binding.
#[derive(Debug, Clone)]
pub struct MintParams {
    /// Canonical space identifier (ADR-306 ontology) — *which* space.
    pub space_id: String,
    /// Signing sensor identity (ADR-305) — *which signed device* produced it.
    /// Must equal the signer's `key_id`.
    pub sensor_id: String,
    /// Capture time (unix seconds). Injected, never read from the wall clock.
    pub captured_at_unix_s: i64,
    /// Validity window in seconds; `expires_at = captured_at + validity_secs`.
    pub validity_secs: i64,
    /// Monotonic version (start at 1; [`CalibrationCertificate::renew`] increments).
    pub version: u64,
    /// Which calibration path produced this (caps the evidence level).
    pub tier: CalibrationTier,
    /// Evidence level claimed (ADR-282). Validated against `tier`/`source`.
    pub evidence: EvidenceLevel,
    /// How the fingerprint was characterized (synthetic vs measured).
    pub source: CharacterizationSource,
    /// Drift envelope governing invalidation.
    pub envelope: CompatibilityEnvelope,
}

/// A signed, versioned, comparable, invalidatable room-fingerprint certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationCertificate {
    /// Certificate schema version ([`CERTIFICATE_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Canonical space identifier (ADR-306).
    pub space_id: String,
    /// Room scope carried through from the calibration state.
    pub room_id: String,
    /// ADR-135 baseline id the fingerprint was derived against.
    pub baseline_id: String,
    /// Signing sensor identity (ADR-305).
    pub sensor_id: String,
    /// Monotonic version (append-only history; renewal increments).
    pub version: u64,
    /// Capture time (unix seconds).
    pub captured_at_unix_s: i64,
    /// Expiry time (unix seconds); `captured_at + validity_secs`.
    pub expires_at_unix_s: i64,
    /// Calibration tier.
    pub tier: CalibrationTier,
    /// Evidence level (honesty-checked at mint).
    pub evidence: EvidenceLevel,
    /// Characterization source.
    pub source: CharacterizationSource,
    /// The room fingerprint this certificate attests.
    pub fingerprint: RoomFingerprint,
    /// Drift envelope governing invalidation.
    pub envelope: CompatibilityEnvelope,
    /// Mandatory signature over the canonical signable bytes.
    pub signature: CertificateSignature,
}

impl CalibrationCertificate {
    /// Mint a certificate from existing calibration state — a pure function over
    /// the [`SpecialistBank`] and [`MintParams`], plus a signer.
    ///
    /// Enforces the honesty discipline before signing:
    /// - a [`CharacterizationSource::Synthetic`] fingerprint must be labelled
    ///   [`EvidenceLevel::L0Synthetic`] (never measured);
    /// - a [`CharacterizationSource::MeasuredCsi`] fingerprint must be L1+;
    /// - the claimed evidence may not exceed what the `tier` can honestly bear
    ///   (an [`CalibrationTier::Auto`] characterization is capped at L2);
    /// - the `sensor_id` must match the signer's `key_id`;
    /// - `version` must be ≥ 1 and `validity_secs` ≥ 0.
    pub fn mint<S: CertificateSigner>(
        params: MintParams,
        bank: &SpecialistBank,
        signer: &S,
    ) -> Result<Self> {
        Self::validate_mint(&params, signer)?;

        let fingerprint = RoomFingerprint::from_bank(bank);
        let expires_at_unix_s = params.captured_at_unix_s.saturating_add(params.validity_secs);

        // Build the unsigned certificate, then sign its canonical bytes.
        let unsigned = UnsignedCertificate {
            schema_version: CERTIFICATE_SCHEMA_VERSION,
            space_id: &params.space_id,
            room_id: &bank.room_id,
            baseline_id: &bank.baseline_id,
            sensor_id: &params.sensor_id,
            version: params.version,
            captured_at_unix_s: params.captured_at_unix_s,
            expires_at_unix_s,
            tier: params.tier,
            evidence: params.evidence,
            source: params.source,
            fingerprint: &fingerprint,
            envelope: params.envelope,
        };
        let content_hash = unsigned.content_hash();
        let signature = CertificateSignature {
            key_id: signer.key_id().to_string(),
            content_hash_hex: hex_lower(&content_hash),
            signature_hex: hex_lower(&signer.sign(&content_hash)),
        };

        Ok(Self {
            schema_version: CERTIFICATE_SCHEMA_VERSION,
            space_id: params.space_id,
            room_id: bank.room_id.clone(),
            baseline_id: bank.baseline_id.clone(),
            sensor_id: params.sensor_id,
            version: params.version,
            captured_at_unix_s: params.captured_at_unix_s,
            expires_at_unix_s,
            tier: params.tier,
            evidence: params.evidence,
            source: params.source,
            fingerprint,
            envelope: params.envelope,
            signature,
        })
    }

    fn validate_mint<S: CertificateSigner>(params: &MintParams, signer: &S) -> Result<()> {
        if params.version == 0 {
            return Err(CalibrationError::InvalidCertificate(
                "certificate version must start at 1 (monotonic)".into(),
            ));
        }
        if params.validity_secs < 0 {
            return Err(CalibrationError::InvalidCertificate(
                "validity_secs must be non-negative".into(),
            ));
        }
        if params.sensor_id != signer.key_id() {
            return Err(CalibrationError::InvalidCertificate(format!(
                "sensor_id '{}' does not match signing key '{}'",
                params.sensor_id,
                signer.key_id()
            )));
        }
        match params.source {
            CharacterizationSource::Synthetic => {
                if params.evidence.is_measured() {
                    return Err(CalibrationError::SyntheticMislabel {
                        claimed: format!("{:?}", params.evidence),
                    });
                }
            }
            CharacterizationSource::MeasuredCsi => {
                if !params.evidence.is_measured() {
                    return Err(CalibrationError::InvalidCertificate(
                        "measured CSI cannot be labelled L0Synthetic".into(),
                    ));
                }
                if params.evidence > params.tier.max_measured_evidence() {
                    return Err(CalibrationError::InvalidCertificate(format!(
                        "{:?} tier may claim at most {:?}, got {:?}",
                        params.tier,
                        params.tier.max_measured_evidence(),
                        params.evidence
                    )));
                }
            }
        }
        Ok(())
    }

    /// Re-characterize into the **next** version, preserving the append-only
    /// history (ADR-301 §4). Same space/sensor/tier/evidence/source/envelope,
    /// `version + 1`, re-signed over the fresh fingerprint and capture time.
    ///
    /// `source`/`evidence` are inherited so a renewal cannot silently upgrade a
    /// synthetic or auto certificate past its honesty cap.
    pub fn renew<S: CertificateSigner>(
        &self,
        captured_at_unix_s: i64,
        validity_secs: i64,
        bank: &SpecialistBank,
        signer: &S,
    ) -> Result<Self> {
        let params = MintParams {
            space_id: self.space_id.clone(),
            sensor_id: self.sensor_id.clone(),
            captured_at_unix_s,
            validity_secs,
            version: self.version.saturating_add(1),
            tier: self.tier,
            evidence: self.evidence,
            source: self.source,
            envelope: self.envelope,
        };
        Self::mint(params, bank, signer)
    }

    /// The 32-byte content hash over this certificate's canonical signable bytes
    /// — the object the signature covers and a witness-chain anchor (ADR-319).
    pub fn content_hash(&self) -> [u8; 32] {
        self.as_unsigned().content_hash()
    }

    fn as_unsigned(&self) -> UnsignedCertificate<'_> {
        UnsignedCertificate {
            schema_version: self.schema_version,
            space_id: &self.space_id,
            room_id: &self.room_id,
            baseline_id: &self.baseline_id,
            sensor_id: &self.sensor_id,
            version: self.version,
            captured_at_unix_s: self.captured_at_unix_s,
            expires_at_unix_s: self.expires_at_unix_s,
            tier: self.tier,
            evidence: self.evidence,
            source: self.source,
            fingerprint: &self.fingerprint,
            envelope: self.envelope,
        }
    }

    /// `true` iff the signature verifies against `verifier` and the recorded
    /// content hash matches the recomputed one (tamper rejection).
    pub fn verify_signature<V: CertificateVerifier>(&self, verifier: &V) -> bool {
        let content_hash = self.content_hash();
        if self.signature.content_hash_hex != hex_lower(&content_hash) {
            return false;
        }
        let Some(sig) = hex_decode(&self.signature.signature_hex) else {
            return false;
        };
        verifier.verify(&self.signature.key_id, &content_hash, &sig)
    }

    /// Distance between this certificate's fingerprint and another's — two
    /// certificates for the same space are comparable (ADR-301 §3).
    pub fn distance(&self, other: &CalibrationCertificate) -> FingerprintDistance {
        self.fingerprint.distance(&other.fingerprint)
    }

    /// Evaluate validity against live room state and a signature verifier.
    ///
    /// Invalidation is an explicit, typed transition (ADR-301 §4), never a
    /// silent flag. Order of precedence: tampered signature → expired → drift
    /// beyond the envelope → valid. `now_unix_s` is injected (no wall clock).
    pub fn status<V: CertificateVerifier>(
        &self,
        current: &RoomFingerprint,
        now_unix_s: i64,
        verifier: &V,
    ) -> CertificateStatus {
        if !self.verify_signature(verifier) {
            return CertificateStatus::TamperedSignature;
        }
        if now_unix_s >= self.expires_at_unix_s {
            return CertificateStatus::Expired {
                now_unix_s,
                expires_at_unix_s: self.expires_at_unix_s,
            };
        }
        let distance = self.fingerprint.distance(current);
        if !distance.within_envelope(&self.envelope) {
            return CertificateStatus::Drifted {
                distance,
                envelope: self.envelope,
            };
        }
        CertificateStatus::Valid { distance }
    }

    /// Convenience: `true` iff [`Self::status`] is [`CertificateStatus::Valid`].
    pub fn is_valid<V: CertificateVerifier>(
        &self,
        current: &RoomFingerprint,
        now_unix_s: i64,
        verifier: &V,
    ) -> bool {
        matches!(
            self.status(current, now_unix_s, verifier),
            CertificateStatus::Valid { .. }
        )
    }

    /// Serialize to pretty JSON (matches [`SpecialistBank`]'s persistence style).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| CalibrationError::Serde(e.to_string()))
    }

    /// Deserialize from JSON, validating the schema version at the boundary.
    pub fn from_json(s: &str) -> Result<Self> {
        let cert: Self =
            serde_json::from_str(s).map_err(|e| CalibrationError::Serde(e.to_string()))?;
        if cert.schema_version != CERTIFICATE_SCHEMA_VERSION {
            return Err(CalibrationError::InvalidCertificate(format!(
                "unsupported certificate schema version {} (expected {})",
                cert.schema_version, CERTIFICATE_SCHEMA_VERSION
            )));
        }
        Ok(cert)
    }
}

/// The typed result of a certificate validity check (ADR-301 §4).
#[derive(Debug, Clone, PartialEq)]
pub enum CertificateStatus {
    /// Still valid; carries the (in-envelope) drift for logging/compensation.
    Valid {
        /// Measured drift vs the live fingerprint (within the envelope).
        distance: FingerprintDistance,
    },
    /// Past its expiry (`now_unix_s >= expires_at_unix_s`).
    Expired {
        /// The injected evaluation time.
        now_unix_s: i64,
        /// The certificate's recorded expiry.
        expires_at_unix_s: i64,
    },
    /// Drift beyond the compatibility envelope — re-characterization required.
    Drifted {
        /// The measured drift that breached the envelope.
        distance: FingerprintDistance,
        /// The envelope it breached.
        envelope: CompatibilityEnvelope,
    },
    /// Signature did not verify (content hash mismatch or bad signature).
    TamperedSignature,
}

impl CertificateStatus {
    /// `true` only for [`CertificateStatus::Valid`].
    pub fn is_valid(&self) -> bool {
        matches!(self, CertificateStatus::Valid { .. })
    }
}

// ---------------------------------------------------------------------------
// Canonical signable encoding
// ---------------------------------------------------------------------------

/// A borrowed view of the signable fields, in a fixed order, used to compute the
/// content hash. Excludes the signature itself (which covers this hash).
struct UnsignedCertificate<'a> {
    schema_version: u32,
    space_id: &'a str,
    room_id: &'a str,
    baseline_id: &'a str,
    sensor_id: &'a str,
    version: u64,
    captured_at_unix_s: i64,
    expires_at_unix_s: i64,
    tier: CalibrationTier,
    evidence: EvidenceLevel,
    source: CharacterizationSource,
    fingerprint: &'a RoomFingerprint,
    envelope: CompatibilityEnvelope,
}

impl UnsignedCertificate<'_> {
    /// SHA-256 over a deterministic, architecture-independent byte encoding.
    ///
    /// Fields are hashed in a fixed order: strings length-prefixed, integers and
    /// floats as little-endian, enums as a stable one-byte tag. No text
    /// formatting of floats (raw IEEE-754 LE), matching the ADR-136
    /// `CanonicalFrame` precedent, so the hash is stable across runs and
    /// architectures.
    fn content_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        // Domain separation so this hash can never collide with another artifact.
        h.update(b"ruview.adr298.calibration-certificate.v1");
        h.update(self.schema_version.to_le_bytes());
        hash_str(&mut h, self.space_id);
        hash_str(&mut h, self.room_id);
        hash_str(&mut h, self.baseline_id);
        hash_str(&mut h, self.sensor_id);
        h.update(self.version.to_le_bytes());
        h.update(self.captured_at_unix_s.to_le_bytes());
        h.update(self.expires_at_unix_s.to_le_bytes());
        h.update([self.tier.tag()]);
        h.update([self.evidence.tag()]);
        h.update([self.source.tag()]);
        hash_fingerprint(&mut h, self.fingerprint);
        h.update(self.envelope.max_total_drift.to_le_bytes());
        h.finalize().into()
    }
}

fn hash_str(h: &mut Sha256, s: &str) {
    h.update((s.len() as u64).to_le_bytes());
    h.update(s.as_bytes());
}

fn hash_fingerprint(h: &mut Sha256, fp: &RoomFingerprint) {
    h.update(fp.schema_version.to_le_bytes());
    h.update(fp.empty_mean.to_le_bytes());
    h.update(fp.empty_variance.to_le_bytes());
    h.update(fp.occupied_variance.to_le_bytes());
    h.update(fp.presence_threshold.to_le_bytes());
    h.update(fp.occupancy_mean_shift.to_le_bytes());
    h.update((GeometryEmbedding::DIM as u64).to_le_bytes());
    for v in fp.geometry.as_slice() {
        h.update(v.to_le_bytes());
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Decode lowercase/uppercase hex. Returns `None` on malformed input (odd length
/// or non-hex digit) — no panics at the deserialization boundary.
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_val(bytes[i])?;
        let lo = hex_val(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Some(out)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchor::AnchorLabel;
    use crate::extract::{AnchorFeature, Features};
    use crate::geometry::NodeGeometry;

    fn af(label: AnchorLabel, variance: f32, motion: f32) -> AnchorFeature {
        af_mean(label, 1.0, variance, motion)
    }

    fn af_mean(label: AnchorLabel, mean: f32, variance: f32, motion: f32) -> AnchorFeature {
        AnchorFeature {
            room_id: "living-room".into(),
            label,
            features: Features {
                mean,
                variance,
                motion,
                breathing_score: 0.0,
                breathing_hz: 0.0,
                heart_score: 0.0,
                heart_hz: 0.0,
            },
        }
    }

    fn anchors() -> Vec<AnchorFeature> {
        vec![
            af_mean(AnchorLabel::Empty, 1.0, 1.0, 0.1),
            af_mean(AnchorLabel::StandStill, 3.0, 10.0, 0.2),
            af(AnchorLabel::Sit, 6.0, 0.2),
            af(AnchorLabel::LieDown, 3.0, 0.2),
            af(AnchorLabel::SmallMove, 4.0, 1.2),
            af(AnchorLabel::SleepPosture, 3.0, 0.1),
        ]
    }

    fn bank_with_geometry() -> SpecialistBank {
        let geometry = vec![
            NodeGeometry::new(1, "tape-measure").with_position(0.0, 0.0, 1.0),
            NodeGeometry::new(2, "tape-measure").with_position(3.0, 0.0, 1.0),
        ];
        SpecialistBank::train("living-room", "base-1", &anchors(), 1000)
            .unwrap()
            .with_geometry(geometry)
    }

    fn signer() -> KeyedHashSigner {
        KeyedHashSigner::new("sensor-42", b"top-secret".to_vec())
    }

    fn measured_params(version: u64) -> MintParams {
        MintParams {
            space_id: "home/living-room".into(),
            sensor_id: "sensor-42".into(),
            captured_at_unix_s: 1_000_000,
            validity_secs: 3600,
            version,
            tier: CalibrationTier::Auto,
            evidence: EvidenceLevel::L2,
            source: CharacterizationSource::MeasuredCsi,
            envelope: CompatibilityEnvelope::default(),
        }
    }

    #[test]
    fn mint_is_deterministic() {
        let bank = bank_with_geometry();
        let s = signer();
        let a = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        let b = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        assert_eq!(a, b, "same inputs → identical certificate");
        assert_eq!(a.content_hash(), b.content_hash());
        assert_eq!(a.signature, b.signature);
    }

    #[test]
    fn signature_round_trips_and_rejects_tampering() {
        let bank = bank_with_geometry();
        let s = signer();
        let cert = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        assert!(cert.verify_signature(&s), "freshly minted cert verifies");

        // Tamper with a signable field: the recorded content hash no longer matches.
        let mut tampered = cert.clone();
        tampered.expires_at_unix_s += 10_000;
        assert!(!tampered.verify_signature(&s), "expiry tamper is rejected");

        // Tamper with the fingerprint payload.
        let mut tampered2 = cert.clone();
        tampered2.fingerprint.empty_mean += 5.0;
        assert!(
            !tampered2.verify_signature(&s),
            "fingerprint tamper is rejected"
        );

        // Wrong key does not verify.
        let other = KeyedHashSigner::new("sensor-42", b"different-secret".to_vec());
        assert!(!cert.verify_signature(&other), "wrong secret is rejected");
    }

    #[test]
    fn version_is_monotonic_across_renewals() {
        let bank = bank_with_geometry();
        let s = signer();
        let v1 = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        let v2 = v1.renew(2_000_000, 3600, &bank, &s).unwrap();
        let v3 = v2.renew(3_000_000, 3600, &bank, &s).unwrap();
        assert_eq!(v1.version, 1);
        assert_eq!(v2.version, 2);
        assert_eq!(v3.version, 3);
        assert!(v1.version < v2.version && v2.version < v3.version);
        // Renewal preserves identity but is a distinct, freshly signed artifact.
        assert_eq!(v2.space_id, v1.space_id);
        assert_eq!(v2.sensor_id, v1.sensor_id);
        assert_ne!(v2.content_hash(), v1.content_hash());
        assert!(v2.verify_signature(&s));
    }

    #[test]
    fn version_zero_is_rejected() {
        let bank = bank_with_geometry();
        let s = signer();
        assert!(CalibrationCertificate::mint(measured_params(0), &bank, &s).is_err());
    }

    #[test]
    fn compare_identical_vs_drifted() {
        let bank = bank_with_geometry();
        let s = signer();
        let cert = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();

        // Identical fingerprint → zero drift.
        let same = cert.fingerprint.clone();
        let d0 = cert.fingerprint.distance(&same);
        assert_eq!(d0.total, 0.0);
        assert_eq!(d0.baseline_drift, 0.0);
        assert_eq!(d0.occupancy_drift, 0.0);

        // A drifted room (empty-room mean moved) → positive, larger drift.
        let mut drifted = cert.fingerprint.clone();
        drifted.empty_mean += 4.0;
        let d1 = cert.fingerprint.distance(&drifted);
        assert!(d1.total > d0.total);
        assert!(d1.baseline_drift > 0.0);
        assert!(d1.total < 1.0, "total is bounded in [0, 1)");
    }

    #[test]
    fn distance_is_monotonic() {
        let base = RoomFingerprint {
            schema_version: FINGERPRINT_SCHEMA_VERSION,
            empty_mean: 1.0,
            empty_variance: 5.0,
            occupied_variance: 10.0,
            presence_threshold: 5.5,
            occupancy_mean_shift: 2.0,
            geometry: GeometryEmbedding::default(),
        };
        let mut last = -1.0;
        for step in 0..8 {
            let mut perturbed = base.clone();
            perturbed.empty_mean = base.empty_mean + step as f32;
            let d = base.distance(&perturbed).total;
            assert!(
                d > last,
                "distance must increase with perturbation (step {step}: {d} <= {last})"
            );
            last = d;
        }
    }

    #[test]
    fn expiry_invalidates() {
        let bank = bank_with_geometry();
        let s = signer();
        let cert = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        let current = cert.fingerprint.clone();

        // Before expiry, same room → valid.
        assert!(cert.is_valid(&current, 1_000_500, &s));
        assert!(matches!(
            cert.status(&current, 1_000_500, &s),
            CertificateStatus::Valid { .. }
        ));

        // At/after expiry → Expired.
        assert!(!cert.is_valid(&current, cert.expires_at_unix_s, &s));
        assert!(matches!(
            cert.status(&current, cert.expires_at_unix_s + 1, &s),
            CertificateStatus::Expired { .. }
        ));
    }

    #[test]
    fn drift_beyond_envelope_invalidates() {
        let bank = bank_with_geometry();
        let s = signer();
        let mut params = measured_params(1);
        params.envelope = CompatibilityEnvelope::new(0.05).unwrap();
        let cert = CalibrationCertificate::mint(params, &bank, &s).unwrap();

        // Small drift stays inside the envelope → valid.
        let mut small = cert.fingerprint.clone();
        small.empty_mean += 0.01;
        assert!(cert.is_valid(&small, 1_000_500, &s));

        // Large drift breaches the envelope → Drifted (explicit invalidation).
        let mut large = cert.fingerprint.clone();
        large.empty_mean += 10.0;
        match cert.status(&large, 1_000_500, &s) {
            CertificateStatus::Drifted { distance, envelope } => {
                assert!(distance.total > envelope.max_total_drift);
            }
            other => panic!("expected Drifted, got {other:?}"),
        }
    }

    #[test]
    fn tampered_signature_takes_precedence() {
        let bank = bank_with_geometry();
        let s = signer();
        let mut cert = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        cert.fingerprint.empty_mean += 1.0; // invalidate the signature
        let current = cert.fingerprint.clone();
        assert!(matches!(
            cert.status(&current, 1_000_500, &s),
            CertificateStatus::TamperedSignature
        ));
    }

    #[test]
    fn json_round_trip() {
        let bank = bank_with_geometry();
        let s = signer();
        let cert = CalibrationCertificate::mint(measured_params(1), &bank, &s).unwrap();
        let json = cert.to_json().unwrap();
        let back = CalibrationCertificate::from_json(&json).unwrap();
        assert_eq!(cert, back);
        // Signature still verifies after a serialization round-trip.
        assert!(back.verify_signature(&s));
    }

    #[test]
    fn synthetic_cannot_be_labelled_measured() {
        let bank = bank_with_geometry();
        let s = signer();
        let mut params = measured_params(1);
        params.source = CharacterizationSource::Synthetic;
        params.evidence = EvidenceLevel::L2; // synthetic claiming measured
        let err = CalibrationCertificate::mint(params, &bank, &s).unwrap_err();
        assert!(matches!(err, CalibrationError::SyntheticMislabel { .. }));

        // Correctly labelled synthetic (L0) is accepted.
        let mut ok = measured_params(1);
        ok.source = CharacterizationSource::Synthetic;
        ok.evidence = EvidenceLevel::L0Synthetic;
        assert!(CalibrationCertificate::mint(ok, &bank, &s).is_ok());
    }

    #[test]
    fn auto_tier_cannot_over_claim_evidence() {
        let bank = bank_with_geometry();
        let s = signer();
        let mut params = measured_params(1);
        params.tier = CalibrationTier::Auto;
        params.evidence = EvidenceLevel::L3; // Auto is capped at L2
        assert!(CalibrationCertificate::mint(params, &bank, &s).is_err());
    }

    #[test]
    fn sensor_identity_must_match_signer() {
        let bank = bank_with_geometry();
        let s = signer();
        let mut params = measured_params(1);
        params.sensor_id = "some-other-device".into();
        assert!(CalibrationCertificate::mint(params, &bank, &s).is_err());
    }

    #[test]
    fn fingerprint_from_bank_reuses_presence_separation() {
        let bank = bank_with_geometry();
        let fp = RoomFingerprint::from_bank(&bank);
        let presence = bank.presence.as_ref().unwrap();
        assert_eq!(fp.empty_mean, presence.empty_mean);
        assert_eq!(fp.occupied_variance, presence.occupied_var);
        assert_eq!(fp.geometry, bank.geometry_embedding());
        assert!(fp.geometry.as_slice().iter().any(|&x| x != 0.0));
    }

    #[test]
    fn invalid_envelope_is_rejected() {
        assert!(CompatibilityEnvelope::new(-0.1).is_err());
        assert!(CompatibilityEnvelope::new(1.0).is_err());
        assert!(CompatibilityEnvelope::new(f32::NAN).is_err());
        assert!(CompatibilityEnvelope::new(0.2).is_ok());
    }
}

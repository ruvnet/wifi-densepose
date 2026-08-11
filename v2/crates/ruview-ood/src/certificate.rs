//! Cross-ADR adapter: turn an ADR-301 [`CalibrationCertificate`] plus a live
//! fingerprint into the two OOD inputs it governs — the [`FingerprintDistance`]
//! and the [`CalibrationCompat`] (ADR-302 §1 inputs 1 and 3).
//!
//! This is the point where certificate *staleness* becomes a domain signal:
//! an expired, tampered, drifted, or identity-mismatched certificate maps to a
//! non-`Valid` compatibility, which the state machine drives straight to
//! UNKNOWN (ADR-300 staleness guard). Absence of a certificate is handled by
//! [`no_certificate`] and likewise defaults to UNKNOWN — absence of evidence is
//! absence of capability (ADR-302 §3).

use wifi_densepose_calibration::certificate::{
    CalibrationCertificate, CertificateStatus, CertificateVerifier, FingerprintDistance, RoomFingerprint,
};

use crate::domain::CalibrationCompat;

/// Identity the live inference expects the certificate to attest: which space
/// (ADR-306) and which signed device (ADR-305). Validated before the
/// certificate's own status, so a certificate for the wrong room/device can
/// never present as compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedIdentity<'a> {
    /// The canonical space id the inference is running in.
    pub space_id: &'a str,
    /// The signed device id producing the live traffic.
    pub device_id: &'a str,
}

/// Assess a present certificate against the live fingerprint and expected
/// identity, returning the domain distance and the calibration compatibility.
///
/// `now_unix_s` is **injected** — never read from the wall clock — so the
/// staleness decision is deterministic and testable. The distance is always the
/// certificate-fingerprint-vs-live distance, computed even for a stale/tampered
/// certificate so the drift is still reported.
///
/// Precedence mirrors ADR-301 `status()` but adds the identity checks first:
/// space mismatch → device mismatch → tampered → expired → drifted → valid.
pub fn assess_certificate<V: CertificateVerifier>(
    cert: &CalibrationCertificate,
    live: &RoomFingerprint,
    expected: ExpectedIdentity<'_>,
    now_unix_s: i64,
    verifier: &V,
) -> (FingerprintDistance, CalibrationCompat) {
    let distance = cert.fingerprint.distance(live);

    // Identity binding first (ADR-305/303): a certificate for the wrong
    // space/device is incompatible regardless of its own validity.
    if cert.space_id != expected.space_id {
        return (distance, CalibrationCompat::SpaceMismatch);
    }
    if cert.sensor_id != expected.device_id {
        return (distance, CalibrationCompat::DeviceMismatch);
    }

    let compat = match cert.status(live, now_unix_s, verifier) {
        CertificateStatus::Valid { .. } => CalibrationCompat::Valid,
        CertificateStatus::Expired { .. } => CalibrationCompat::Expired,
        CertificateStatus::Drifted { .. } => CalibrationCompat::DriftedBeyondEnvelope,
        CertificateStatus::TamperedSignature => CalibrationCompat::Tampered,
    };
    (distance, compat)
}

/// The compatibility for a space/device with **no** certificate present. Always
/// [`CalibrationCompat::Absent`], which the gate treats as UNKNOWN (ADR-302 §3:
/// the default state without a valid certificate is UNKNOWN, not KNOWN).
pub fn no_certificate() -> CalibrationCompat {
    CalibrationCompat::Absent
}

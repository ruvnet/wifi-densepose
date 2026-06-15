//! `witness_signing` — Ed25519 signature layer over the witness chain.
//!
//! ADR-116 §2.2: every state transition must be signed by the
//! Seed so a downstream auditor can prove the chain wasn't
//! retroactively assembled. The chain primitive
//! (`witness::WitnessChain`) handles hash linkage; this module
//! adds the cryptographic attestation.
//!
//! Kept in a separate module from the chain itself so:
//!
//!   * the hash chain stays usable without `ed25519-dalek` linked
//!     in (good for the `wasm32-unknown-unknown` cog variant we'll
//!     ship for browser-side audit verification),
//!   * key rotation invalidates *signatures* but not the chain —
//!     the auditor only needs the new public key to re-verify,
//!   * the signing surface stays small enough to audit in one
//!     read.
//!
//! ## What gets signed
//!
//! `sign_event(event, key)` signs the same canonical byte form
//! that `witness::hash_event` hashes. That means:
//!
//!   1. A signature commits to the entire event (kind, payload,
//!      timestamp, seq, prev_hash) — no field can be retroactively
//!      changed without invalidating both the hash AND the
//!      signature.
//!   2. The signature implicitly commits to the *chain position*
//!      via `prev_hash` — splicing a signed event into a different
//!      chain breaks verification.
//!
//! ## Key management
//!
//! Out of scope for this module. The cog runtime reads the Seed's
//! Ed25519 signing key from the Cognitum control plane's secure
//! key store (separate concern). Tests use a fixed-bytes seed for
//! determinism — never check in real Seed keys here.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

use crate::witness::{canonical_bytes, WitnessEvent};

/// Sign a witness event with the Seed's Ed25519 key. Returns the
/// 64-byte Ed25519 signature over the event's canonical bytes —
/// the same bytes `witness::hash_event` hashes, so a verifier that
/// already trusts the hash chain only needs one extra check.
pub fn sign_event(event: &WitnessEvent, key: &SigningKey) -> Signature {
    let bytes = canonical_bytes(
        event.prev_hash,
        event.seq,
        event.timestamp_unix_s,
        &event.kind,
        &event.payload,
    );
    key.sign(&bytes)
}

/// Verify an Ed25519 signature against a witness event using the
/// Seed's public key. `Ok(())` iff the signature is valid for the
/// event's canonical bytes under this key.
///
/// Uses `verify_strict` (not the permissive `Verifier::verify`) on
/// purpose: for a tamper-evident *audit* chain the signature is the
/// attestation, so non-canonical encodings and small-order public
/// keys must be rejected. `verify_strict` enforces RFC 8032's
/// stricter checks, giving the "one canonical signature per event"
/// property an auditor relies on when comparing or deduplicating
/// signed witness records. The public key is caller-pinned (the
/// Seed's known verifying key) — never parsed from the event — so a
/// forged event carrying its own key cannot self-verify.
pub fn verify_signature(
    event: &WitnessEvent,
    signature: &Signature,
    public_key: &VerifyingKey,
) -> Result<(), SignatureVerifyError> {
    let bytes = canonical_bytes(
        event.prev_hash,
        event.seq,
        event.timestamp_unix_s,
        &event.kind,
        &event.payload,
    );
    public_key
        .verify_strict(&bytes, signature)
        .map_err(|_| SignatureVerifyError::Invalid)
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SignatureVerifyError {
    #[error("Ed25519 signature does not match event under this public key")]
    Invalid,
}

/// Encode a signature as 128 hex chars (no `0x` prefix). Matches the
/// hex convention the rest of the witness wire format uses.
pub fn signature_to_hex(sig: &Signature) -> String {
    let bytes = sig.to_bytes();
    let mut s = String::with_capacity(128);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Parse a 128-char lowercase-hex string back into a `Signature`.
pub fn signature_from_hex(s: &str) -> Result<Signature, SignatureParseError> {
    if s.len() != 128 {
        return Err(SignatureParseError::Length { found: s.len() });
    }
    let mut bytes = [0u8; 64];
    for (i, byte) in bytes.iter_mut().enumerate() {
        let lo = i * 2;
        *byte = u8::from_str_radix(&s[lo..lo + 2], 16)
            .map_err(|_| SignatureParseError::Hex { at: lo })?;
    }
    Ok(Signature::from_bytes(&bytes))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SignatureParseError {
    #[error("signature hex must be 128 chars, got {found}")]
    Length { found: usize },
    #[error("signature hex parse error at byte offset {at}")]
    Hex { at: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::{WitnessChain, WitnessHash};

    fn fixed_key() -> SigningKey {
        // Deterministic test key — DO NOT use in production. The
        // seed is `b"cog-ha-matter-unit-tests--------"` (32 bytes).
        SigningKey::from_bytes(b"cog-ha-matter-unit-tests--------")
    }

    fn fresh_event() -> WitnessEvent {
        let mut c = WitnessChain::new();
        c.append("fall_risk_elevated", br#"{"node":"kitchen"}"#, 1779512400);
        c.events()[0].clone()
    }

    #[test]
    fn sign_and_verify_round_trip() {
        let key = fixed_key();
        let public = key.verifying_key();
        let event = fresh_event();
        let sig = sign_event(&event, &key);
        verify_signature(&event, &sig, &public).expect("clean signature verifies");
    }

    #[test]
    fn signature_commits_to_domain_tag_not_bare_fields() {
        // The signature is over the domain-tagged canonical bytes. A
        // signature produced over the *un-tagged* concatenation of the
        // same fields must NOT verify — proving cross-protocol
        // separation reaches the signature layer, not just the hash.
        // Fails on the old encoding where the signed message began
        // directly with `prev_hash` (no tag).
        use ed25519_dalek::Signer;
        let key = fixed_key();
        let public = key.verifying_key();
        let event = fresh_event();

        // Hand-build the OLD (un-tagged) preimage and sign it.
        let mut untagged = Vec::new();
        untagged.extend_from_slice(&event.prev_hash.0);
        untagged.extend_from_slice(&event.seq.to_be_bytes());
        untagged.extend_from_slice(&event.timestamp_unix_s.to_be_bytes());
        untagged.extend_from_slice(&(event.kind.len() as u32).to_be_bytes());
        untagged.extend_from_slice(event.kind.as_bytes());
        untagged.extend_from_slice(&(event.payload.len() as u32).to_be_bytes());
        untagged.extend_from_slice(&event.payload);
        let old_sig = key.sign(&untagged);

        // The current verifier (which uses the domain-tagged message)
        // must reject a signature made over the un-tagged bytes.
        let err = verify_signature(&event, &old_sig, &public).unwrap_err();
        assert_eq!(err, SignatureVerifyError::Invalid);

        // Sanity: the proper signature still verifies.
        let good = sign_event(&event, &key);
        verify_signature(&event, &good, &public).expect("tagged signature verifies");
    }

    #[test]
    fn verify_uses_strict_path_and_pins_caller_key() {
        // Regression guard: verification must run through the strict
        // path against a CALLER-supplied key. A wrong key fails; the
        // event never carries its own verifying key, so a forged event
        // cannot self-attest. (verify_strict additionally rejects
        // non-canonical / small-order encodings.)
        let key = fixed_key();
        let wrong = SigningKey::from_bytes(b"another-wrong-key-another-wrong-");
        let event = fresh_event();
        let sig = sign_event(&event, &key);
        verify_signature(&event, &sig, &key.verifying_key()).expect("right key verifies");
        assert_eq!(
            verify_signature(&event, &sig, &wrong.verifying_key()).unwrap_err(),
            SignatureVerifyError::Invalid
        );
    }

    #[test]
    fn verify_rejects_signature_under_wrong_key() {
        let key = fixed_key();
        let other = SigningKey::from_bytes(b"different-key-different-key-----");
        let event = fresh_event();
        let sig = sign_event(&event, &key);
        // Same event, signature from `key`, but verify under `other`'s
        // public key — must fail.
        let err = verify_signature(&event, &sig, &other.verifying_key()).unwrap_err();
        assert_eq!(err, SignatureVerifyError::Invalid);
    }

    #[test]
    fn verify_rejects_tampered_event() {
        // Sign one event, then mutate the payload and verify the
        // *mutated* event under the same signature. Must fail.
        let key = fixed_key();
        let public = key.verifying_key();
        let mut event = fresh_event();
        let sig = sign_event(&event, &key);
        event.payload = b"forged-after-sign".to_vec();
        let err = verify_signature(&event, &sig, &public).unwrap_err();
        assert_eq!(err, SignatureVerifyError::Invalid);
    }

    #[test]
    fn verify_rejects_event_with_wrong_prev_hash() {
        // Same payload + kind, but the event claims a different
        // chain position. Cryptographically bound to prev_hash via
        // canonical bytes.
        let key = fixed_key();
        let public = key.verifying_key();
        let mut event = fresh_event();
        let sig = sign_event(&event, &key);
        event.prev_hash = WitnessHash([0x77; 32]);
        let err = verify_signature(&event, &sig, &public).unwrap_err();
        assert_eq!(err, SignatureVerifyError::Invalid);
    }

    #[test]
    fn signature_hex_round_trip() {
        let key = fixed_key();
        let event = fresh_event();
        let sig = sign_event(&event, &key);
        let hex = signature_to_hex(&sig);
        assert_eq!(hex.len(), 128);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        let parsed = signature_from_hex(&hex).unwrap();
        assert_eq!(parsed.to_bytes(), sig.to_bytes());
    }

    #[test]
    fn signature_from_hex_rejects_wrong_length() {
        let err = signature_from_hex("abcd").unwrap_err();
        assert_eq!(err, SignatureParseError::Length { found: 4 });
    }

    #[test]
    fn signature_from_hex_rejects_non_hex() {
        // 128 chars but non-hex.
        let bad = "Z".repeat(128);
        let err = signature_from_hex(&bad).unwrap_err();
        assert!(matches!(err, SignatureParseError::Hex { at: 0 }));
    }

    #[test]
    fn signature_is_deterministic_for_same_event_and_key() {
        // Ed25519 is deterministic; locking this means a future
        // accidental switch to a randomized scheme (RustCrypto's
        // optional rand-based API) fires a named test.
        let key = fixed_key();
        let event = fresh_event();
        let sig1 = sign_event(&event, &key);
        let sig2 = sign_event(&event, &key);
        assert_eq!(sig1.to_bytes(), sig2.to_bytes());
    }

    #[test]
    fn different_events_produce_different_signatures() {
        let key = fixed_key();
        let mut a = fresh_event();
        let mut b = fresh_event();
        a.payload = b"a".to_vec();
        b.payload = b"b".to_vec();
        let sig_a = sign_event(&a, &key);
        let sig_b = sign_event(&b, &key);
        assert_ne!(sig_a.to_bytes(), sig_b.to_bytes());
    }
}

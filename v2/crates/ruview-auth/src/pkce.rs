//! OAuth 2.0 PKCE (RFC 7636) generation.
//!
//! Ported from `cognitum-one/meta-proxy` `src/oauth/pkce.rs`, itself ported from
//! `dashboard/apps/cli`. Kept byte-compatible on purpose: a verifier generated
//! here has to validate against the same `services/identity` code every other
//! Cognitum client already talks to.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::RngCore;
use sha2::{Digest, Sha256};

/// One login attempt's PKCE pair plus its CSRF `state`.
#[derive(Debug, Clone)]
pub struct PkceRequest {
    pub state: String,
    pub code_verifier: String,
    pub code_challenge: String,
}

fn random_url_safe_token(byte_len: usize) -> String {
    let mut bytes = vec![0u8; byte_len];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn challenge_from_verifier(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// Fresh `state` + verifier/challenge for one login attempt.
///
/// 32 random bytes each: base64url-encodes to 43 characters, comfortably inside
/// RFC 7636 §4.1's 43–128 range without padding.
pub fn generate() -> PkceRequest {
    let state = random_url_safe_token(32);
    let code_verifier = random_url_safe_token(32);
    let code_challenge = challenge_from_verifier(&code_verifier);
    PkceRequest {
        state,
        code_verifier,
        code_challenge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_rfc7636_appendix_b_worked_example() {
        // The spec's own vector. If this drifts, our S256 is not S256 and the
        // server will reject every exchange — worth pinning to the standard
        // rather than to our own output.
        assert_eq!(
            challenge_from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn verifier_length_is_within_rfc7636_bounds() {
        let r = generate();
        assert!(
            r.code_verifier.len() >= 43 && r.code_verifier.len() <= 128,
            "len {}",
            r.code_verifier.len()
        );
    }

    #[test]
    fn challenge_is_derived_from_the_verifier_it_ships_with() {
        let r = generate();
        assert_eq!(challenge_from_verifier(&r.code_verifier), r.code_challenge);
    }

    #[test]
    fn separate_attempts_share_nothing() {
        let (a, b) = (generate(), generate());
        assert_ne!(a.state, b.state);
        assert_ne!(a.code_verifier, b.code_verifier);
    }
}

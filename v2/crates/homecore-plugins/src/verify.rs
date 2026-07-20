//! Plugin signature & integrity verification (ADR-162, P4).
//!
//! ADR-161/B5 honestly relabelled the manifest's `wasm_module_hash` /
//! `wasm_module_sig` / `publisher_key` fields as "(P4 — not yet enforced)":
//! they were parsed and round-tripped but **never checked** before a plugin
//! ran. This module makes that claim TRUE — it is the real verification gate
//! the plugin load path runs before instantiating any `.wasm` module.
//!
//! ## What is verified, in order
//!
//! 1. **Module hash** — SHA-256 of the actual `.wasm` bytes must equal the
//!    manifest's `wasm_module_hash` (`sha256:<hex>`). A tampered module
//!    (one byte changed) fails here.
//! 2. **Ed25519 signature** — `wasm_module_sig` (`ed25519:<base64>`, 64-byte
//!    raw signature) must verify over the **32-byte SHA-256 digest** under
//!    the `publisher_key` (`ed25519:<base64>`, 32-byte raw verifying key).
//! 3. **Trust policy** — the `publisher_key` must be on the configured
//!    allowlist, unless [`PluginPolicy::AllowUnsigned`] is in force (a loud
//!    dev escape hatch).
//!
//! The crypto mirrors the in-repo Ed25519 pattern from
//! `cog-ha-matter::witness_signing` (same `ed25519-dalek` 2.x API, same
//! deterministic-test-key convention). SHA-256 matches the `sha256:` prefix
//! the manifest doc already declared for `wasm_module_hash`, and the
//! `cog-ha-matter` cog manifest's `binary_sha256` hex convention.
//!
//! ## Secure default
//!
//! [`PluginPolicy::trusted`] (the production constructor) **rejects**:
//!   * an unsigned module (no hash / sig / key),
//!   * a signature from a key not on the allowlist,
//!   * any hash or signature mismatch.
//!
//! Only [`PluginPolicy::AllowUnsigned`] loosens this, and every load it
//! waves through emits a `warn`-level log line so it cannot pass silently.

use base64::Engine as _;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::error::PluginError;
use crate::manifest::PluginManifest;

/// Trust policy governing which plugins may load.
///
/// The production path uses [`PluginPolicy::trusted`] with an explicit
/// allowlist of publisher verifying keys. [`PluginPolicy::AllowUnsigned`]
/// is the dev escape hatch — it loads anything (even unsigned modules) but
/// logs a loud warning per load.
#[derive(Debug, Clone)]
pub enum PluginPolicy {
    /// Secure default: a plugin loads only if its module hash matches, its
    /// Ed25519 signature verifies, AND its publisher key is in this
    /// allowlist. Each entry is the 32-byte raw Ed25519 verifying key.
    Trusted { allowlist: Vec<[u8; 32]> },
    /// Dev-only: skip signature/allowlist enforcement. Hash is still
    /// checked when a `wasm_module_hash` is present (cheap integrity), but
    /// unsigned / unknown-publisher modules are allowed. Every load logs a
    /// loud `warn`.
    AllowUnsigned,
}

impl PluginPolicy {
    /// Construct the secure (production) policy from a list of trusted
    /// publisher keys, each encoded as `ed25519:<base64>` (the same form
    /// the manifest `publisher_key` uses).
    pub fn trusted(publisher_keys: &[&str]) -> Result<Self, PluginError> {
        let mut allowlist = Vec::with_capacity(publisher_keys.len());
        for k in publisher_keys {
            allowlist.push(decode_verifying_key(k)?.to_bytes());
        }
        Ok(PluginPolicy::Trusted { allowlist })
    }

    /// Secure policy that trusts no publisher at all — every signed or
    /// unsigned module is rejected. Useful as a strict default.
    pub fn deny_all() -> Self {
        PluginPolicy::Trusted { allowlist: vec![] }
    }

    fn is_dev(&self) -> bool {
        matches!(self, PluginPolicy::AllowUnsigned)
    }

    fn allows(&self, key: &VerifyingKey) -> bool {
        match self {
            PluginPolicy::AllowUnsigned => true,
            PluginPolicy::Trusted { allowlist } => {
                allowlist.iter().any(|k| k == &key.to_bytes())
            }
        }
    }
}

/// Verify a `.wasm` module's integrity and signature against its manifest,
/// under the given trust `policy`. Returns `Ok(())` only if the module may
/// be instantiated.
///
/// On [`PluginPolicy::AllowUnsigned`] this still checks any present hash,
/// but waves through missing/untrusted signatures with a loud `warn`.
pub fn verify_module(
    manifest: &PluginManifest,
    wasm_bytes: &[u8],
    policy: &PluginPolicy,
) -> Result<(), PluginError> {
    let signed = manifest.wasm_module_hash.is_some()
        || manifest.wasm_module_sig.is_some()
        || manifest.publisher_key.is_some();

    if !signed {
        // No integrity material at all.
        if policy.is_dev() {
            eprintln!(
                "[PLUGIN WARN] loading UNSIGNED plugin `{}` — no wasm_module_hash/sig/publisher_key. \
                 AllowUnsigned dev policy is active; this is INSECURE and must not be used in production.",
                manifest.domain
            );
            return Ok(());
        }
        return Err(PluginError::SignatureRejected(format!(
            "plugin `{}` is unsigned (no wasm_module_hash/sig/publisher_key) and the trust policy \
             rejects unsigned modules; set PluginPolicy::AllowUnsigned to override in dev",
            manifest.domain
        )));
    }

    // (1) Hash check — always enforced when a hash is declared.
    let digest = sha256_digest(wasm_bytes);
    if let Some(declared) = &manifest.wasm_module_hash {
        let expected = parse_sha256(declared)?;
        if expected != digest {
            return Err(PluginError::SignatureRejected(format!(
                "plugin `{}` wasm hash mismatch: module does not match manifest wasm_module_hash \
                 (tampered or wrong binary)",
                manifest.domain
            )));
        }
    } else if !policy.is_dev() {
        return Err(PluginError::SignatureRejected(format!(
            "plugin `{}` carries a signature/publisher_key but no wasm_module_hash to bind it to",
            manifest.domain
        )));
    }

    // (2) Signature check + (3) allowlist.
    match (&manifest.wasm_module_sig, &manifest.publisher_key) {
        (Some(sig_str), Some(key_str)) => {
            let key = decode_verifying_key(key_str)?;
            let sig = decode_signature(sig_str)?;
            key.verify(&digest, &sig).map_err(|_| {
                PluginError::SignatureRejected(format!(
                    "plugin `{}` Ed25519 signature does not verify over the module hash under \
                     publisher_key",
                    manifest.domain
                ))
            })?;
            if !policy.allows(&key) {
                if policy.is_dev() {
                    eprintln!(
                        "[PLUGIN WARN] plugin `{}` is validly signed but its publisher_key is NOT on \
                         the trust allowlist; AllowUnsigned dev policy loads it anyway.",
                        manifest.domain
                    );
                    return Ok(());
                }
                return Err(PluginError::SignatureRejected(format!(
                    "plugin `{}` is validly signed but its publisher_key is not on the trust \
                     allowlist (untrusted publisher)",
                    manifest.domain
                )));
            }
            Ok(())
        }
        _ => {
            // Hash present but signature/key incomplete.
            if policy.is_dev() {
                eprintln!(
                    "[PLUGIN WARN] plugin `{}` has a hash but no complete Ed25519 signature; \
                     AllowUnsigned dev policy loads it anyway.",
                    manifest.domain
                );
                return Ok(());
            }
            Err(PluginError::SignatureRejected(format!(
                "plugin `{}` is missing a complete wasm_module_sig + publisher_key pair; the trust \
                 policy requires a valid signature",
                manifest.domain
            )))
        }
    }
}

/// SHA-256 of `bytes` as a 32-byte digest.
fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Parse a `sha256:<hex>` manifest hash into a 32-byte digest.
fn parse_sha256(s: &str) -> Result<[u8; 32], PluginError> {
    let hex_part = s.strip_prefix("sha256:").ok_or_else(|| {
        PluginError::InvalidManifest(format!(
            "wasm_module_hash must be `sha256:<hex>`, got {s:?}"
        ))
    })?;
    let raw = hex::decode(hex_part).map_err(|e| {
        PluginError::InvalidManifest(format!("wasm_module_hash hex decode: {e}"))
    })?;
    raw.try_into().map_err(|v: Vec<u8>| {
        PluginError::InvalidManifest(format!(
            "wasm_module_hash must decode to 32 bytes, got {}",
            v.len()
        ))
    })
}

/// Decode an `ed25519:<base64>` 32-byte verifying key.
fn decode_verifying_key(s: &str) -> Result<VerifyingKey, PluginError> {
    let b64 = s.strip_prefix("ed25519:").ok_or_else(|| {
        PluginError::InvalidManifest(format!(
            "publisher_key must be `ed25519:<base64>`, got {s:?}"
        ))
    })?;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| PluginError::InvalidManifest(format!("publisher_key base64: {e}")))?;
    let bytes: [u8; 32] = raw.try_into().map_err(|v: Vec<u8>| {
        PluginError::InvalidManifest(format!(
            "publisher_key must decode to 32 bytes, got {}",
            v.len()
        ))
    })?;
    VerifyingKey::from_bytes(&bytes)
        .map_err(|e| PluginError::InvalidManifest(format!("publisher_key not a valid Ed25519 point: {e}")))
}

/// Decode an `ed25519:<base64>` 64-byte signature.
fn decode_signature(s: &str) -> Result<Signature, PluginError> {
    let b64 = s.strip_prefix("ed25519:").ok_or_else(|| {
        PluginError::InvalidManifest(format!(
            "wasm_module_sig must be `ed25519:<base64>`, got {s:?}"
        ))
    })?;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| PluginError::InvalidManifest(format!("wasm_module_sig base64: {e}")))?;
    let bytes: [u8; 64] = raw.try_into().map_err(|v: Vec<u8>| {
        PluginError::InvalidManifest(format!(
            "wasm_module_sig must decode to 64 bytes, got {}",
            v.len()
        ))
    })?;
    Ok(Signature::from_bytes(&bytes))
}

/// Encode a SHA-256 digest as the manifest `sha256:<hex>` form. Exposed so
/// tooling (and tests) can produce a manifest hash for real `.wasm` bytes.
pub fn encode_sha256(wasm_bytes: &[u8]) -> String {
    format!("sha256:{}", hex::encode(sha256_digest(wasm_bytes)))
}

/// Encode an Ed25519 verifying key as the manifest `ed25519:<base64>` form.
pub fn encode_verifying_key(key: &VerifyingKey) -> String {
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(key.to_bytes())
    )
}

/// Encode an Ed25519 signature as the manifest `ed25519:<base64>` form.
pub fn encode_signature(sig: &Signature) -> String {
    format!(
        "ed25519:{}",
        base64::engine::general_purpose::STANDARD.encode(sig.to_bytes())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    /// Deterministic publisher key (mirrors witness_signing's fixed-bytes
    /// seed convention — DO NOT use in production).
    fn publisher() -> SigningKey {
        SigningKey::from_bytes(b"homecore-plugins-pub-test-seed--")
    }

    fn attacker() -> SigningKey {
        SigningKey::from_bytes(b"homecore-plugins-attacker-seed--")
    }

    /// Sign `wasm_bytes` with `key` and produce a manifest carrying the real
    /// hash + signature + publisher key.
    fn signed_manifest(wasm_bytes: &[u8], key: &SigningKey) -> PluginManifest {
        let digest = sha256_digest(wasm_bytes);
        let sig = key.sign(&digest);
        PluginManifest {
            domain: "demo".into(),
            name: "Demo".into(),
            version: "1.0.0".into(),
            documentation: None,
            iot_class: None,
            config_flow: false,
            integration_type: None,
            dependencies: vec![],
            requirements: vec![],
            wasm_module: Some("demo.wasm".into()),
            wasm_module_hash: Some(encode_sha256(wasm_bytes)),
            wasm_module_sig: Some(encode_signature(&sig)),
            publisher_key: Some(encode_verifying_key(&key.verifying_key())),
            min_homecore_version: None,
            host_imports_required: vec![],
            homecore_permissions: vec![],
            cog_id: None,
        }
    }

    #[test]
    fn valid_sig_from_trusted_key_passes() {
        let wasm = b"\0asm\x01\0\0\0fake module bytes";
        let key = publisher();
        let manifest = signed_manifest(wasm, &key);
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&key.verifying_key())]).unwrap();
        verify_module(&manifest, wasm, &policy).expect("trusted signed module should load");
    }

    #[test]
    fn tampered_module_is_rejected() {
        let wasm = b"\0asm\x01\0\0\0fake module bytes";
        let key = publisher();
        let manifest = signed_manifest(wasm, &key);
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&key.verifying_key())]).unwrap();
        // Flip a byte: hash no longer matches.
        let tampered = b"\0asm\x01\0\0\0FAKE module bytes";
        let err = verify_module(&manifest, tampered, &policy).unwrap_err();
        assert!(matches!(err, PluginError::SignatureRejected(_)), "got {err:?}");
    }

    #[test]
    fn valid_sig_from_untrusted_key_is_rejected() {
        let wasm = b"\0asm\x01\0\0\0fake module bytes";
        // Signed correctly by the attacker, but the attacker is not trusted.
        let manifest = signed_manifest(wasm, &attacker());
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&publisher().verifying_key())]).unwrap();
        let err = verify_module(&manifest, wasm, &policy).unwrap_err();
        assert!(matches!(err, PluginError::SignatureRejected(_)), "got {err:?}");
    }

    #[test]
    fn forged_signature_is_rejected() {
        // Manifest claims the trusted publisher_key but the signature was
        // produced by the attacker (a forged sig under a trusted identity).
        let wasm = b"\0asm\x01\0\0\0fake module bytes";
        let digest = sha256_digest(wasm);
        let forged = attacker().sign(&digest);
        let mut manifest = signed_manifest(wasm, &publisher());
        manifest.wasm_module_sig = Some(encode_signature(&forged));
        let policy =
            PluginPolicy::trusted(&[&encode_verifying_key(&publisher().verifying_key())]).unwrap();
        let err = verify_module(&manifest, wasm, &policy).unwrap_err();
        assert!(matches!(err, PluginError::SignatureRejected(_)), "got {err:?}");
    }

    #[test]
    fn unsigned_module_rejected_under_default_policy() {
        let wasm = b"\0asm\x01\0\0\0unsigned";
        let manifest = PluginManifest {
            domain: "u".into(),
            name: "U".into(),
            version: "1".into(),
            documentation: None,
            iot_class: None,
            config_flow: false,
            integration_type: None,
            dependencies: vec![],
            requirements: vec![],
            wasm_module: Some("u.wasm".into()),
            wasm_module_hash: None,
            wasm_module_sig: None,
            publisher_key: None,
            min_homecore_version: None,
            host_imports_required: vec![],
            homecore_permissions: vec![],
            cog_id: None,
        };
        let err = verify_module(&manifest, wasm, &PluginPolicy::deny_all()).unwrap_err();
        assert!(matches!(err, PluginError::SignatureRejected(_)), "got {err:?}");
        // ...but AllowUnsigned loads it (with a warn).
        verify_module(&manifest, wasm, &PluginPolicy::AllowUnsigned)
            .expect("AllowUnsigned should load an unsigned module");
    }
}

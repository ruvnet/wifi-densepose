//! Embedded signed cog manifest (ADR-100 §"manifest.json", ADR-159 §A4).
//!
//! The `cog-person-count manifest` subcommand emits the **real, signed**
//! manifest the release pipeline produced — byte-for-byte the artifact served
//! from GCS, with a real `binary_sha256`, `weights_sha256`, Ed25519
//! `binary_signature`, and honest `build_metadata` (e.g. `training_class1_accuracy
//! = 0.343`, not inflated). The previous implementation printed a hollow
//! skeleton with `binary_sha256: null`, which made the CLI look unsigned even
//! though the signed manifest existed on disk.
//!
//! The matching manifest for the build's target arch is selected via `cfg!`.

/// Real signed manifest for `x86_64-unknown-linux-gnu`.
pub const MANIFEST_X86_64: &str =
    include_str!("../cog/artifacts/manifests/x86_64/manifest.json");

/// Real signed manifest for `aarch64`/`arm` (the Seed appliance).
pub const MANIFEST_ARM: &str = include_str!("../cog/artifacts/manifests/arm/manifest.json");

/// The embedded signed manifest matching the build's target arch.
pub fn embedded_manifest_str() -> &'static str {
    if cfg!(any(target_arch = "aarch64", target_arch = "arm")) {
        MANIFEST_ARM
    } else {
        MANIFEST_X86_64
    }
}

/// Parse the embedded manifest into canonical JSON. Returns an error if the
/// embedded artifact is malformed (so the CLI fails loudly rather than printing
/// garbage).
pub fn embedded_manifest_value() -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str(embedded_manifest_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ADR-159 §A4 — the embedded manifest the CLI emits must carry a real
    /// `binary_sha256` (the field the old hollow `cmd_manifest` left null).
    #[test]
    fn embedded_manifest_has_non_null_binary_sha256() {
        let v = embedded_manifest_value().expect("embedded manifest parses");
        let sha = v.get("binary_sha256").and_then(|s| s.as_str());
        assert!(
            sha.is_some(),
            "embedded manifest must have a non-null binary_sha256 (got {:?})",
            v.get("binary_sha256")
        );
        let sha = sha.unwrap();
        assert_eq!(sha.len(), 64, "binary_sha256 must be a 32-byte hex digest");
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "binary_sha256 must be hex"
        );
    }

    #[test]
    fn embedded_manifest_is_signed() {
        let v = embedded_manifest_value().expect("parse");
        assert!(
            v.get("binary_signature").and_then(|s| s.as_str()).is_some(),
            "embedded manifest must carry an Ed25519 binary_signature"
        );
        assert_eq!(
            v.get("sig_algo").and_then(|s| s.as_str()),
            Some("Ed25519")
        );
    }

    #[test]
    fn embedded_manifest_id_matches_cog() {
        let v = embedded_manifest_value().expect("parse");
        assert_eq!(v.get("id").and_then(|s| s.as_str()), Some(crate::COG_ID));
    }
}

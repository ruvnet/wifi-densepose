//! Versioned format parser for HA storage schema version 13.
//!
//! Applies to (as of HA 2025.1):
//!   - `core.entity_registry` — `version=1, minor_version=13`
//!   - `core.device_registry` — `version=1, minor_version=13`
//!
//! Source: `homeassistant/helpers/entity_registry.py` `STORAGE_VERSION_MINOR`
//! and `homeassistant/helpers/device_registry.py` `STORAGE_VERSION_MINOR`.
//!
//! `core.config_entries` uses a different versioning scheme; see
//! `config_entries.rs` for details.

/// The major storage `version` this module handles.
pub const MAJOR_VERSION: u32 = 1;

/// The `minor_version` values this module handles.
/// Any value outside this set raises `MigrateError::UnsupportedSchemaVersion`.
pub const SUPPORTED_MINOR_VERSIONS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

/// Return `true` if the given envelope header is handled by this module.
pub fn handles(version: u32, minor_version: u32) -> bool {
    version == MAJOR_VERSION && SUPPORTED_MINOR_VERSIONS.contains(&minor_version)
}

/// Validate that `(version, minor_version)` is supported; return the error
/// with the given `file` path embedded if not.
///
/// Call this at the top of every parser that routes through v13 before
/// attempting any field access.
pub fn require_supported(
    file: &str,
    version: u32,
    minor_version: u32,
) -> Result<(), crate::MigrateError> {
    if !handles(version, minor_version) {
        return Err(crate::MigrateError::UnsupportedSchemaVersion {
            file: file.to_owned(),
            version,
            minor_version,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_all_supported_minor_versions() {
        for &mv in SUPPORTED_MINOR_VERSIONS {
            assert!(handles(1, mv), "minor_version {mv} should be supported");
        }
    }

    #[test]
    fn rejects_unknown_minor_version() {
        assert!(!handles(1, 99));
        assert!(!handles(2, 13));
    }

    #[test]
    fn require_supported_ok_for_v13() {
        assert!(require_supported("core.entity_registry", 1, 13).is_ok());
    }

    #[test]
    fn require_supported_err_carries_file_name() {
        let err = require_supported("core.entity_registry", 1, 99).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("core.entity_registry"),
            "error should contain file name: {msg}"
        );
        assert!(
            msg.contains("minor_version=99"),
            "error should contain minor_version: {msg}"
        );
    }
}

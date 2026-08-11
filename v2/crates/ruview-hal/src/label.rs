//! Bounded-string validation shared by the HAL's boundary types.
//!
//! Capability tags and `Modality::Custom` payloads arrive from potentially
//! untrusted hardware descriptors. They are validated at construction with the
//! same discipline the ontology applies to ids: non-empty, length-bounded, and
//! free of ASCII control characters (CLAUDE.md: validate untrusted input at
//! every boundary; bound allocation).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Maximum accepted label length, in bytes. Bounds allocation on untrusted
/// input.
pub const MAX_LABEL_LEN: usize = 128;

/// Reasons a raw label string is rejected at the boundary.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LabelError {
    /// The label was empty.
    #[error("label must not be empty")]
    Empty,
    /// The label exceeded [`MAX_LABEL_LEN`] bytes.
    #[error("label length {len} exceeds maximum {max}")]
    TooLong {
        /// Actual length in bytes.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
    /// The label contained an ASCII control character.
    #[error("label contains a control character at byte {pos}")]
    ControlChar {
        /// Byte offset of the offending control character.
        pos: usize,
    },
}

/// Validate a raw label: non-empty, bounded length, no control characters.
pub(crate) fn validate_label(raw: &str) -> Result<(), LabelError> {
    if raw.is_empty() {
        return Err(LabelError::Empty);
    }
    if raw.len() > MAX_LABEL_LEN {
        return Err(LabelError::TooLong {
            len: raw.len(),
            max: MAX_LABEL_LEN,
        });
    }
    if let Some(pos) = raw.bytes().position(|b| b.is_ascii_control()) {
        return Err(LabelError::ControlChar { pos });
    }
    Ok(())
}

/// A validated, bounded capability tag describing one phenomenon a sensor can
/// observe (e.g. `"amplitude"`, `"range"`, `"accel"`). Reuses the ontology's
/// id-style validation discipline rather than accepting a raw `String`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilityTag(String);

impl CapabilityTag {
    /// Construct a validated tag, rejecting empty, over-long, or
    /// control-character input at the boundary.
    pub fn new(raw: impl Into<String>) -> Result<Self, LabelError> {
        let s = raw.into();
        validate_label(&s)?;
        Ok(Self(s))
    }

    /// Borrow the underlying tag string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for CapabilityTag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

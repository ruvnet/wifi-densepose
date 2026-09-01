//! Deterministic, domain-separated SHA-256 digests.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// A deterministic SHA-256 digest over a canonical RuView payload.
///
/// Digests are bytes rather than platform-sized integers, so their meaning is
/// stable across CPU architectures. Higher-level types write lengths as
/// big-endian `u64`, floats as normalized IEEE-754 bits, and enum tags as
/// single bytes through the crate-private [`CanonicalWriter`].
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CanonicalDigest([u8; 32]);

impl CanonicalDigest {
    /// Hash an opaque byte payload with an explicit domain separator.
    #[must_use]
    pub fn of_bytes(domain: &[u8], payload: &[u8]) -> Self {
        let mut writer = CanonicalWriter::new(domain);
        writer.bytes(payload);
        writer.finish()
    }

    /// Construct from the exact digest bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the exact digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Return true only for the reserved placeholder digest.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|byte| *byte == 0)
    }

    /// Encode as fixed-width lowercase hexadecimal.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        output
    }
}

impl fmt::Debug for CanonicalDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CanonicalDigest")
            .field(&self.to_hex())
            .finish()
    }
}

impl fmt::Display for CanonicalDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

/// Canonical hash writer shared by validated core payloads.
pub(crate) struct CanonicalWriter(Sha256);

impl CanonicalWriter {
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"ruview-forecast-canonical-v1\0");
        hasher.update((domain.len() as u64).to_be_bytes());
        hasher.update(domain);
        Self(hasher)
    }

    pub(crate) fn tag(&mut self, value: u8) {
        self.0.update([value]);
    }

    pub(crate) fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.0.update(value.to_be_bytes());
    }

    pub(crate) fn f32(&mut self, value: f32) {
        let canonical = if value == 0.0 { 0.0 } else { value };
        self.0.update(canonical.to_bits().to_be_bytes());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        let canonical = if value == 0.0 { 0.0 } else { value };
        self.0.update(canonical.to_bits().to_be_bytes());
    }

    pub(crate) fn bool(&mut self, value: bool) {
        self.tag(u8::from(value));
    }

    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.u64(value.len() as u64);
        self.0.update(value);
    }

    pub(crate) fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    pub(crate) fn digest(&mut self, value: CanonicalDigest) {
        self.0.update(value.0);
    }

    pub(crate) fn finish(self) -> CanonicalDigest {
        CanonicalDigest(self.0.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_domain_separated_and_fixed_width() {
        let left = CanonicalDigest::of_bytes(b"left", b"payload");
        let right = CanonicalDigest::of_bytes(b"right", b"payload");
        assert_ne!(left, right);
        assert_eq!(left.to_hex().len(), 64);
        assert_eq!(left, CanonicalDigest::of_bytes(b"left", b"payload"));
    }

    #[test]
    fn writer_normalizes_negative_zero() {
        let mut positive = CanonicalWriter::new(b"float");
        positive.f32(0.0);
        positive.f64(0.0);
        let mut negative = CanonicalWriter::new(b"float");
        negative.f32(-0.0);
        negative.f64(-0.0);
        assert_eq!(positive.finish(), negative.finish());
    }
}

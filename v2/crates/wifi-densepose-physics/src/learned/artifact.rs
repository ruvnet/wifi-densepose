//! Signed learned-artifact identity, compatibility, and bounded output.

use super::{
    features::FEATURE_WIDTH,
    model::{HIDDEN_WIDTH, HISTORY_FRAMES},
};

pub const LEARNED_ARTIFACT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LearnedArtifactManifest {
    pub schema_version: u16,
    pub history_frames: usize,
    pub feature_width: usize,
    pub hidden_width: usize,
    pub model_id: String,
}

impl LearnedArtifactManifest {
    #[must_use]
    pub fn adr323(model_id: String) -> Self {
        Self {
            schema_version: LEARNED_ARTIFACT_SCHEMA_VERSION,
            history_frames: HISTORY_FRAMES,
            feature_width: FEATURE_WIDTH,
            hidden_width: HIDDEN_WIDTH,
            model_id,
        }
    }

    fn is_compatible(&self) -> bool {
        self.schema_version == LEARNED_ARTIFACT_SCHEMA_VERSION
            && self.history_frames == HISTORY_FRAMES
            && self.feature_width == FEATURE_WIDTH
            && self.hidden_width == HIDDEN_WIDTH
            && !self.model_id.is_empty()
            && self.model_id.len() <= 128
    }
}

/// Receipt from the existing signed RVF/Cog activation boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureVerification {
    pub key_id: String,
    pub envelope_hash: [u8; 32],
}

impl SignatureVerification {
    #[must_use]
    pub fn accepted(key_id: String, envelope_hash: [u8; 32]) -> Option<Self> {
        (!key_id.is_empty() && key_id.len() <= 128 && envelope_hash != [0; 32]).then_some(Self {
            key_id,
            envelope_hash,
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LearnedArtifactError {
    #[error("artifact content hash mismatch")]
    HashMismatch,
    #[error("artifact manifest is incompatible")]
    IncompatibleManifest,
    #[error("artifact signature receipt is invalid")]
    InvalidSignatureReceipt,
    #[error("artifact exceeds the configured byte limit")]
    ArtifactTooLarge,
}

/// Verified immutable model bytes and activation provenance.
#[derive(Clone, Debug)]
pub struct LearnedArtifact {
    pub content_hash: [u8; 32],
    pub manifest: LearnedArtifactManifest,
    pub signature: SignatureVerification,
    bytes: Vec<u8>,
}

impl LearnedArtifact {
    pub fn verified(
        bytes: Vec<u8>,
        expected: [u8; 32],
        manifest: LearnedArtifactManifest,
        signature: SignatureVerification,
        maximum_bytes: usize,
    ) -> Result<Self, LearnedArtifactError> {
        if bytes.len() > maximum_bytes {
            return Err(LearnedArtifactError::ArtifactTooLarge);
        }
        if !manifest.is_compatible() {
            return Err(LearnedArtifactError::IncompatibleManifest);
        }
        if signature.key_id.is_empty() || signature.envelope_hash == [0; 32] {
            return Err(LearnedArtifactError::InvalidSignatureReceipt);
        }
        let actual = *blake3::hash(&bytes).as_bytes();
        if actual != expected {
            return Err(LearnedArtifactError::HashMismatch);
        }
        Ok(Self {
            content_hash: actual,
            manifest,
            signature,
            bytes,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Learned heads after hard output validation.
#[derive(Clone, Copy, Debug)]
pub struct LearnedResidual {
    pub joints_m: [[f32; 3]; 17],
    pub log_variance: [[f32; 3]; 17],
    pub contact_probability: [f32; 2],
    pub abstention_probability: f32,
}

impl LearnedResidual {
    /// Reject non-finite output, then clamp residuals and probabilistic heads.
    #[must_use]
    pub fn bounded(mut self, cap_m: f32) -> Option<Self> {
        if !cap_m.is_finite()
            || cap_m <= 0.0
            || !self
                .joints_m
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            || !self
                .log_variance
                .iter()
                .flatten()
                .all(|value| value.is_finite())
            || !self
                .contact_probability
                .iter()
                .all(|value| value.is_finite())
            || !self.abstention_probability.is_finite()
        {
            return None;
        }
        for joint in &mut self.joints_m {
            for coordinate in joint {
                *coordinate = coordinate.clamp(-cap_m, cap_m);
            }
        }
        for joint in &mut self.log_variance {
            for coordinate in joint {
                *coordinate = coordinate.clamp(-20.0, 10.0);
            }
        }
        for probability in &mut self.contact_probability {
            *probability = probability.clamp(0.0, 1.0);
        }
        self.abstention_probability = self.abstention_probability.clamp(0.0, 1.0);
        Some(self)
    }
}

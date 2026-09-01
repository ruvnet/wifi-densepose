//! CPU activation of a previously verified forecast artifact.

use burn_core::tensor::Tensor;
use burn_ndarray::NdArray;
pub use burn_ndarray::NdArrayDevice as CpuDevice;

use crate::{
    ArtifactActivationPolicy, ForecastModelConfig, ForecastModelOutput, ModelError, ModelInput,
    RuForecastMixer, TrustedSignerSet, VerifiedModelArtifact,
};

/// Native CPU backend used by the portable `RuView` runtime.
pub type CpuBackend = NdArray<f32>;

/// Immutable CPU runtime bound to one artifact identity.
pub struct CpuForecastRuntime {
    model: RuForecastMixer<CpuBackend>,
    config: ForecastModelConfig,
    device: CpuDevice,
    artifact_digest: [u8; 32],
}

impl CpuForecastRuntime {
    /// Decode and verify an envelope before any Burn record is activated.
    pub fn activate(
        encoded: &[u8],
        trusted_signers: &TrustedSignerSet,
        policy: &ArtifactActivationPolicy,
    ) -> Result<Self, ModelError> {
        let artifact =
            VerifiedModelArtifact::decode_and_verify(encoded, trusted_signers)?.activate(policy)?;
        let config = artifact.manifest().config.clone();
        let device = CpuDevice::default();
        let model = RuForecastMixer::from_activated_artifact(&config, &artifact, &device)?;
        let runtime = Self {
            model,
            config,
            device,
            artifact_digest: artifact.envelope_digest(),
        };
        runtime.self_test()?;
        Ok(runtime)
    }

    /// Architecture activated by this runtime.
    #[must_use]
    pub const fn config(&self) -> &ForecastModelConfig {
        &self.config
    }

    /// Digest of the complete verified artifact envelope.
    #[must_use]
    pub const fn artifact_digest(&self) -> [u8; 32] {
        self.artifact_digest
    }

    /// Backend device used to construct input tensors.
    #[must_use]
    pub const fn device(&self) -> &CpuDevice {
        &self.device
    }

    /// Run one canonical batch. Shape mismatches fail before convolution.
    pub fn predict(
        &self,
        input: ModelInput<CpuBackend>,
    ) -> Result<ForecastModelOutput<CpuBackend>, ModelError> {
        self.model.forward(input)
    }

    fn self_test(&self) -> Result<(), ModelError> {
        let input = ModelInput::new(
            &self.config,
            Tensor::zeros([1, self.config.context_len, 1], &self.device),
            Tensor::ones([1, self.config.context_len, 1], &self.device),
            Tensor::zeros([1, self.config.context_len, 1], &self.device),
            Tensor::zeros(
                [1, self.config.context_len, self.config.time_width],
                &self.device,
            ),
            Tensor::zeros(
                [1, self.config.horizon, self.config.time_width],
                &self.device,
            ),
            Tensor::zeros([1, 1, self.config.descriptor_width], &self.device),
            Tensor::ones([1, 1], &self.device),
        )?;
        self.model.forward(input).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;

    use super::*;
    use crate::{ArtifactManifest, ModelArtifact, TINY_PARAMETER_COUNT};

    #[test]
    fn unsigned_candidate_cannot_activate() {
        let weights = b"not-even-a-burn-record".to_vec();
        let candidate = ModelArtifact::new(
            ArtifactManifest {
                schema_version: 1,
                architecture: "ruview-factorized-forecast-mixer-v1".to_owned(),
                config: ForecastModelConfig::tiny_ci(),
                parameter_count: TINY_PARAMETER_COUNT,
                feature_schema_digest: [1; 32],
                training_manifest_digest: [2; 32],
                weights_digest: *blake3::hash(&weights).as_bytes(),
                seed: 7,
                release_epoch: 1,
                minimum_runtime_version: 1,
                maximum_runtime_version: 1,
                expires_at_unix_ms: None,
                build_id: "unsigned-test".into(),
                teacher_outputs_used: false,
                independently_implemented: true,
            },
            weights,
        )
        .unwrap();
        let key = SigningKey::from_bytes(&[7; 32]).verifying_key().to_bytes();
        let trusted = TrustedSignerSet::new(vec![key]).unwrap();
        let policy = ArtifactActivationPolicy::new(1, 1, 1, [1; 32]).unwrap();
        assert!(matches!(
            CpuForecastRuntime::activate(&candidate.encode().unwrap(), &trusted, &policy),
            Err(ModelError::Malformed(_))
        ));
    }
}

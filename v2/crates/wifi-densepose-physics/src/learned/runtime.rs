//! CPU inference activation for verified Burn artifacts.

use burn_core::tensor::{Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};

use super::{
    features::FEATURE_WIDTH,
    model::{ResidualGru, HISTORY_FRAMES},
    LearnedArtifact, LearnedResidual,
};

type CpuBackend = NdArray<f32>;

/// Native CPU residual inference. Construction requires a verified artifact.
pub struct CpuResidualRuntime {
    model: ResidualGru<CpuBackend>,
    device: NdArrayDevice,
    artifact_hash: [u8; 32],
}

impl CpuResidualRuntime {
    /// Atomically deserialize a verified Burn record before activation.
    pub fn activate(artifact: &LearnedArtifact) -> Result<Self, burn_core::record::RecorderError> {
        let device = NdArrayDevice::default();
        let model = ResidualGru::from_verified_artifact(artifact, &device)?;
        Ok(Self {
            model,
            device,
            artifact_hash: artifact.content_hash,
        })
    }

    /// Active immutable artifact identity.
    #[must_use]
    pub const fn artifact_hash(&self) -> [u8; 32] {
        self.artifact_hash
    }

    /// Infer one complete history. Partial, non-finite, or malformed windows
    /// abstain before entering Burn.
    #[must_use]
    pub fn predict(
        &self,
        flattened_history: &[f32],
        correction_cap_m: f32,
    ) -> Option<LearnedResidual> {
        if flattened_history.len() != HISTORY_FRAMES * FEATURE_WIDTH
            || !flattened_history.iter().all(|value| value.is_finite())
        {
            return None;
        }
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                flattened_history.to_vec(),
                [1, HISTORY_FRAMES, FEATURE_WIDTH],
            ),
            &self.device,
        );
        let output = self.model.forward(input);
        let residual = output.joint_residual.into_data().to_vec::<f32>().ok()?;
        let log_variance = output
            .residual_log_variance
            .into_data()
            .to_vec::<f32>()
            .ok()?;
        let contact = output
            .contact_probability
            .into_data()
            .to_vec::<f32>()
            .ok()?;
        let abstention = output
            .abstention_probability
            .into_data()
            .to_vec::<f32>()
            .ok()?;
        let candidate = LearnedResidual {
            joints_m: core::array::from_fn(|joint| {
                core::array::from_fn(|axis| residual[joint * 3 + axis] * correction_cap_m)
            }),
            log_variance: core::array::from_fn(|joint| {
                core::array::from_fn(|axis| log_variance[joint * 3 + axis])
            }),
            contact_probability: [contact[0], contact[1]],
            abstention_probability: abstention[0],
        };
        candidate.bounded(correction_cap_m)
    }
}

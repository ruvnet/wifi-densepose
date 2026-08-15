//! Burn GRU residual architecture. Backend selection remains a Cargo feature.

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder, RecorderError},
    tensor::{backend::Backend, Tensor},
};
use burn_core as burn;
use burn_nn::{Gru, GruConfig, Linear, LinearConfig, Sigmoid, Tanh};

use super::features::FEATURE_WIDTH;

pub const HISTORY_FRAMES: usize = 20;
pub const HIDDEN_WIDTH: usize = 128;
pub const JOINT_RESIDUAL_WIDTH: usize = 17 * 3;

/// Multi-head outputs for bounded residuals, log variance, contact, and abstention.
pub struct ResidualModelOutput<B: Backend> {
    pub joint_residual: Tensor<B, 2>,
    pub residual_log_variance: Tensor<B, 2>,
    pub contact_probability: Tensor<B, 2>,
    pub abstention_probability: Tensor<B, 2>,
}

/// Two-layer unidirectional GRU followed by independent auditable heads.
#[derive(Module, Debug)]
pub struct ResidualGru<B: Backend> {
    gru1: Gru<B>,
    gru2: Gru<B>,
    residual_head: Linear<B>,
    variance_head: Linear<B>,
    contact_head: Linear<B>,
    abstention_head: Linear<B>,
    tanh: Tanh,
    sigmoid: Sigmoid,
}

impl<B: Backend> ResidualGru<B> {
    /// Initialize the ADR-323 architecture on an explicit backend device.
    #[must_use]
    pub fn init(device: &B::Device) -> Self {
        Self {
            gru1: GruConfig::new(FEATURE_WIDTH, HIDDEN_WIDTH, true)
                .with_clip(Some(10.0))
                .init(device),
            gru2: GruConfig::new(HIDDEN_WIDTH, HIDDEN_WIDTH, true)
                .with_clip(Some(10.0))
                .init(device),
            residual_head: LinearConfig::new(HIDDEN_WIDTH, JOINT_RESIDUAL_WIDTH).init(device),
            variance_head: LinearConfig::new(HIDDEN_WIDTH, JOINT_RESIDUAL_WIDTH).init(device),
            contact_head: LinearConfig::new(HIDDEN_WIDTH, 2).init(device),
            abstention_head: LinearConfig::new(HIDDEN_WIDTH, 1).init(device),
            tanh: Tanh::new(),
            sigmoid: Sigmoid::new(),
        }
    }

    /// Deserialize only bytes that already passed manifest, hash, and
    /// signature-receipt verification at the artifact boundary.
    pub fn from_verified_artifact(
        artifact: &super::LearnedArtifact,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(artifact.bytes().to_vec(), device)?;
        Ok(Self::init(device).load_record(record))
    }

    /// Serialize a trainer-produced record in the exact activation format.
    pub fn into_artifact_bytes(self) -> Result<Vec<u8>, RecorderError> {
        NamedMpkBytesRecorder::<FullPrecisionSettings>::default().record(self.into_record(), ())
    }

    /// Forward `[batch, 20, FEATURE_WIDTH]` to four bounded output heads.
    #[must_use]
    pub fn forward(&self, input: Tensor<B, 3>) -> ResidualModelOutput<B> {
        let hidden = self.gru2.forward(self.gru1.forward(input, None), None);
        let [batch, sequence, width] = hidden.shape().dims();
        debug_assert_eq!(sequence, HISTORY_FRAMES);
        debug_assert_eq!(width, HIDDEN_WIDTH);
        let last = hidden
            .slice([0..batch, (sequence - 1)..sequence, 0..width])
            .squeeze_dim::<2>(1);
        ResidualModelOutput {
            joint_residual: self.tanh.forward(self.residual_head.forward(last.clone())),
            residual_log_variance: self.variance_head.forward(last.clone()),
            contact_probability: self
                .sigmoid
                .forward(self.contact_head.forward(last.clone())),
            abstention_probability: self.sigmoid.forward(self.abstention_head.forward(last)),
        }
    }
}

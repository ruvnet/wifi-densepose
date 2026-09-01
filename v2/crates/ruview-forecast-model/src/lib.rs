//! Independent multivariate forecasting model for `RuView`.
//!
//! This crate contains no Google model code, weights, outputs, configuration,
//! or tests.  The architecture is a `RuView`-specific composition of generic ML
//! primitives: masked patch tokens, gated depthwise temporal mixing,
//! permutation-equivariant variate attention, and an ordered quantile head.
//!
//! Backend code is deliberately feature-gated.  With default features disabled
//! the crate exposes only configuration and bounded artifact validation, keeping
//! the normal `RuView` workspace build independent of Burn and `CubeCL`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod artifact;
mod config;

#[cfg(feature = "model")]
mod network;
#[cfg(feature = "cpu")]
mod runtime;
#[cfg(feature = "ruvector")]
mod ruvector_adapter;

pub use artifact::{
    ActivatedModelArtifact, ArtifactActivationPolicy, ArtifactManifest, ModelArtifact, ModelError,
    SignedModelArtifact, TrustedSignerSet, VerifiedModelArtifact, ARTIFACT_MAGIC,
    MAX_ARTIFACT_BYTES, RUNTIME_COMPATIBILITY_VERSION, SIGNED_ARTIFACT_MAGIC,
};
pub use config::{
    ConfigError, ForecastModelConfig, DESCRIPTOR_WIDTH, LARGE_FORWARD_MULTIPLY_ADDS,
    LARGE_PARAMETER_COUNT, MAX_CONFIG_ACTIVATION_CELLS, MAX_FORWARD_MULTIPLY_ADDS, QUANTILE_COUNT,
    TIME_WIDTH, TINY_PARAMETER_COUNT,
};

#[cfg(feature = "model")]
pub use network::{
    masked_pinball_loss, record_to_bytes, ForecastModelOutput, ModelInput, RuForecastMixer,
    TrainingBatch, MAX_FORWARD_ACTIVATION_CELLS, MAX_INPUT_CELLS, MAX_MODEL_BATCH,
};
#[cfg(feature = "cpu")]
pub use runtime::{CpuBackend, CpuDevice, CpuForecastRuntime};
/// Burn CUDA backend used by Linux/NVIDIA training jobs.
#[cfg(feature = "cuda")]
pub type CudaBackend = burn_cuda::Cuda<f32, i32>;
/// CUDA device selector used by [`CudaBackend`].
#[cfg(feature = "cuda")]
pub use burn_cuda::CudaDevice;
/// Burn WGPU backend for portable Vulkan, Metal, DX12, or WebGPU inference.
#[cfg(feature = "wgpu")]
pub type WgpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;
/// WGPU device selector used by [`WgpuBackend`].
#[cfg(feature = "wgpu")]
pub use burn_wgpu::WgpuDevice;
#[cfg(feature = "ruvector")]
pub use ruvector_adapter::RuVectorAnalogIndex;

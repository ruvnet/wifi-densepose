//! Architecture configuration and arithmetic validation.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Number of clock features supplied at each context and forecast step.
pub const TIME_WIDTH: usize = 8;
/// Width of the packed, permutation-carrying variate descriptor.
///
/// Layout: feature kind one-hot (32), modality one-hot (8), RF band one-hot
/// (8), and twelve numeric metadata values paired with twelve validity bits.
pub const DESCRIPTOR_WIDTH: usize = 72;
/// Fixed probabilistic output grid for the first implementation.
pub const QUANTILE_COUNT: usize = 7;
/// Exact parameter count of [`ForecastModelConfig::large_linux`].
pub const LARGE_PARAMETER_COUNT: usize = 20_285_108;
/// Exact parameter count of [`ForecastModelConfig::tiny_ci`].
pub const TINY_PARAMETER_COUNT: usize = 35_700;
/// Maximum learned scalars accepted by the activation boundary.
pub const MAX_PARAMETERS: usize = 64_000_000;
/// Maximum patch tokens per variate accepted by the activation boundary.
pub const MAX_PATCHES: usize = 4_096;
/// Maximum per-example intermediate tensor cells implied by a configuration.
pub const MAX_CONFIG_ACTIVATION_CELLS: usize = 64 * 1024 * 1024;
/// Conservative multiply-add estimate for one `large_linux` forward example.
pub const LARGE_FORWARD_MULTIPLY_ADDS: u64 = 164_103_634_944;
/// Maximum conservative multiply-add estimate accepted by one forward call.
///
/// This permits two `large_linux` examples, matching the training preflight.
pub const MAX_FORWARD_MULTIPLY_ADDS: u64 = 2 * LARGE_FORWARD_MULTIPLY_ADDS;

/// Invalid architecture configuration.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ConfigError {
    /// A required dimension is zero.
    #[error("{0} must be non-zero")]
    Zero(&'static str),
    /// Patch geometry does not cover the configured context exactly.
    #[error("context_len - patch_len must be divisible by patch_stride")]
    PatchCoverage,
    /// The temporal kernel must be odd so same-padding is unambiguous.
    #[error("temporal_kernel must be odd")]
    EvenKernel,
    /// Attention heads do not divide the model width.
    #[error("d_model must be divisible by variate_heads")]
    HeadWidth,
    /// Dropout is not a finite probability below one.
    #[error("dropout must be finite and in [0, 1)")]
    Dropout,
    /// Quantiles are not the fixed increasing grid with a median center.
    #[error("quantiles must be strictly increasing, lie in (0,1), and have 0.5 at the center")]
    Quantiles,
    /// A checked parameter-count operation overflowed.
    #[error("parameter-count arithmetic overflow")]
    ParameterOverflow,
    /// A dimension exceeds the bounded v1 activation envelope.
    #[error("{name}={value} exceeds the v1 limit {limit}")]
    Capacity {
        /// Dimension name.
        name: &'static str,
        /// Rejected value.
        value: usize,
        /// Inclusive limit.
        limit: usize,
    },
    /// Version-one descriptor/time widths are immutable.
    #[error("version one requires descriptor_width=72 and time_width=8")]
    VersionWidth,
    /// Version one accepts only a reviewed, named architecture profile.
    #[error("configuration is not an exact supported v1 model profile")]
    UnsupportedProfile,
    /// A checked forward-work operation overflowed.
    #[error("forward-work arithmetic overflow")]
    WorkOverflow,
}

/// Backend-independent architecture configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastModelConfig {
    /// Number of context samples per variate.
    pub context_len: usize,
    /// Number of future samples emitted in one forward pass.
    pub horizon: usize,
    /// Samples flattened into each patch token.
    pub patch_len: usize,
    /// Samples between adjacent patch starts.
    pub patch_stride: usize,
    /// Token width.
    pub d_model: usize,
    /// Number of factorized temporal/variate mixer blocks.
    pub layers: usize,
    /// Odd depthwise temporal kernel width.
    pub temporal_kernel: usize,
    /// Number of dense variate-attention heads.
    pub variate_heads: usize,
    /// Hidden width of the gated feed-forward network.
    pub ff_width: usize,
    /// Rank of the future-horizon basis.
    pub horizon_rank: usize,
    /// Maximum variates accepted by this artifact.
    pub max_variates: usize,
    /// Dropout probability used only in training mode.
    pub dropout: f64,
    /// Strictly increasing output quantiles.
    pub quantiles: [f32; QUANTILE_COUNT],
    /// Packed descriptor width. Version one requires [`DESCRIPTOR_WIDTH`].
    pub descriptor_width: usize,
    /// Deterministic time-feature width. Version one requires [`TIME_WIDTH`].
    pub time_width: usize,
}

impl ForecastModelConfig {
    /// Small deterministic preset used for shape and optimizer tests in CI.
    #[must_use]
    pub const fn tiny_ci() -> Self {
        Self {
            context_len: 64,
            horizon: 12,
            patch_len: 8,
            patch_stride: 4,
            d_model: 32,
            layers: 2,
            temporal_kernel: 3,
            variate_heads: 4,
            ff_width: 64,
            horizon_rank: 8,
            max_variates: 8,
            dropout: 0.0,
            quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            descriptor_width: DESCRIPTOR_WIDTH,
            time_width: TIME_WIDTH,
        }
    }

    /// Approximately twenty-million-parameter Linux/CUDA training preset.
    #[must_use]
    pub const fn large_linux() -> Self {
        Self {
            context_len: 1_024,
            horizon: 300,
            patch_len: 16,
            patch_stride: 8,
            d_model: 384,
            layers: 8,
            temporal_kernel: 5,
            variate_heads: 8,
            ff_width: 1_248,
            horizon_rank: 32,
            max_variates: 64,
            dropout: 0.1,
            quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            descriptor_width: DESCRIPTOR_WIDTH,
            time_width: TIME_WIDTH,
        }
    }

    /// Validate every shape and probabilistic invariant before allocation.
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("context_len", self.context_len),
            ("horizon", self.horizon),
            ("patch_len", self.patch_len),
            ("patch_stride", self.patch_stride),
            ("d_model", self.d_model),
            ("layers", self.layers),
            ("temporal_kernel", self.temporal_kernel),
            ("variate_heads", self.variate_heads),
            ("ff_width", self.ff_width),
            ("horizon_rank", self.horizon_rank),
            ("max_variates", self.max_variates),
            ("descriptor_width", self.descriptor_width),
            ("time_width", self.time_width),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero(name));
            }
        }
        if self.patch_len > self.context_len
            || !(self.context_len - self.patch_len).is_multiple_of(self.patch_stride)
        {
            return Err(ConfigError::PatchCoverage);
        }
        if self.temporal_kernel.is_multiple_of(2) {
            return Err(ConfigError::EvenKernel);
        }
        if !self.d_model.is_multiple_of(self.variate_heads) {
            return Err(ConfigError::HeadWidth);
        }
        if !(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout)) {
            return Err(ConfigError::Dropout);
        }
        let center = QUANTILE_COUNT / 2;
        if self.quantiles[center].to_bits() != 0.5_f32.to_bits()
            || self
                .quantiles
                .iter()
                .any(|q| !q.is_finite() || *q <= 0.0 || *q >= 1.0)
            || self.quantiles.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(ConfigError::Quantiles);
        }
        if self.descriptor_width != DESCRIPTOR_WIDTH || self.time_width != TIME_WIDTH {
            return Err(ConfigError::VersionWidth);
        }
        for (name, value, limit) in [
            ("context_len", self.context_len, 16_384),
            ("horizon", self.horizon, 4_096),
            ("patch_len", self.patch_len, 1_024),
            ("patch_stride", self.patch_stride, 16_384),
            ("d_model", self.d_model, 1_024),
            ("layers", self.layers, 32),
            ("temporal_kernel", self.temporal_kernel, 31),
            ("variate_heads", self.variate_heads, 64),
            ("ff_width", self.ff_width, 8_192),
            ("horizon_rank", self.horizon_rank, 256),
            ("max_variates", self.max_variates, 128),
        ] {
            if value > limit {
                return Err(ConfigError::Capacity { name, value, limit });
            }
        }
        let patches = self.patch_count()?;
        if patches > MAX_PATCHES {
            return Err(ConfigError::Capacity {
                name: "patch_count",
                value: patches,
                limit: MAX_PATCHES,
            });
        }
        let activation_cells = self.activation_cells()?;
        if activation_cells > MAX_CONFIG_ACTIVATION_CELLS {
            return Err(ConfigError::Capacity {
                name: "activation_cells",
                value: activation_cells,
                limit: MAX_CONFIG_ACTIVATION_CELLS,
            });
        }
        let parameters = self.parameter_count()?;
        if parameters > MAX_PARAMETERS {
            return Err(ConfigError::Capacity {
                name: "parameter_count",
                value: parameters,
                limit: MAX_PARAMETERS,
            });
        }
        // The public activation boundary intentionally admits only reviewed
        // presets. This prevents a low-parameter, high-attention-work custom
        // configuration from turning signed metadata into a compute DoS.
        if self != &Self::tiny_ci() && self != &Self::large_linux() {
            return Err(ConfigError::UnsupportedProfile);
        }
        Ok(())
    }

    /// Number of patch tokens per variate.
    pub fn patch_count(&self) -> Result<usize, ConfigError> {
        self.context_len
            .checked_sub(self.patch_len)
            .and_then(|value| value.checked_div(self.patch_stride))
            .and_then(|value| value.checked_add(1))
            .ok_or(ConfigError::ParameterOverflow)
    }

    /// Conservative per-example upper bound for the largest live block.
    ///
    /// This includes resident tokens plus temporal gates, attention Q/K/V and
    /// per-head score matrices, both live `SwiGLU` branches, and head outputs.
    pub fn activation_cells(&self) -> Result<usize, ConfigError> {
        let patches = self.patch_count()?;
        let token_cells = patches
            .checked_mul(self.max_variates)
            .and_then(|value| value.checked_mul(self.d_model))
            .ok_or(ConfigError::ParameterOverflow)?;
        let attention_scores = patches
            .checked_mul(self.variate_heads)
            .and_then(|value| value.checked_mul(self.max_variates))
            .and_then(|value| value.checked_mul(self.max_variates))
            .ok_or(ConfigError::ParameterOverflow)?;
        let ff_cells = patches
            .checked_mul(self.max_variates)
            .and_then(|value| value.checked_mul(self.ff_width))
            .ok_or(ConfigError::ParameterOverflow)?;
        let output_cells = self
            .max_variates
            .checked_mul(self.horizon)
            .and_then(|value| value.checked_mul(QUANTILE_COUNT))
            .ok_or(ConfigError::ParameterOverflow)?;
        let temporal_peak = token_cells
            .checked_mul(5)
            .ok_or(ConfigError::ParameterOverflow)?;
        let attention_peak = token_cells
            .checked_mul(6)
            .and_then(|value| value.checked_add(attention_scores))
            .ok_or(ConfigError::ParameterOverflow)?;
        let ff_peak = token_cells
            .checked_mul(3)
            .and_then(|value| value.checked_add(ff_cells.checked_mul(2)?))
            .ok_or(ConfigError::ParameterOverflow)?;
        let head_peak = token_cells
            .checked_add(output_cells)
            .ok_or(ConfigError::ParameterOverflow)?;
        Ok(temporal_peak
            .max(attention_peak)
            .max(ff_peak)
            .max(head_peak))
    }

    /// Exact number of learned scalars implied by this configuration.
    ///
    /// The count assumes biased linear/conv layers and affine `LayerNorm`.
    pub fn parameter_count(&self) -> Result<usize, ConfigError> {
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or(ConfigError::ParameterOverflow);
        let add = |a: usize, b: usize| a.checked_add(b).ok_or(ConfigError::ParameterOverflow);
        let linear = |input: usize, output: usize| add(mul(input, output)?, output);

        let d = self.d_model;
        let f = self.ff_width;
        let q = QUANTILE_COUNT;
        let r = self.horizon_rank;

        let mut total = 0usize;
        let patch_input = mul(3, self.patch_len)?;
        let two_d = mul(2, d)?;
        let three_d = mul(3, d)?;
        let qr = mul(q, r)?;
        total = add(total, linear(patch_input, d)?)?;
        total = add(total, linear(self.descriptor_width, d)?)?;
        total = add(total, linear(self.time_width, d)?)?;

        let mut block = 0usize;
        block = add(block, mul(6, d)?)?; // three affine LayerNorms
        block = add(block, add(mul(d, self.temporal_kernel)?, d)?)?;
        block = add(block, linear(d, two_d)?)?;
        block = add(block, linear(d, d)?)?;
        block = add(block, linear(d, three_d)?)?;
        block = add(block, linear(d, d)?)?;
        block = add(block, linear(d, f)?)?;
        block = add(block, linear(d, f)?)?;
        block = add(block, linear(f, d)?)?;
        total = add(total, mul(self.layers, block)?)?;

        total = add(total, linear(two_d, d)?)?;
        total = add(total, two_d)?; // final affine LayerNorm
        total = add(total, linear(d, qr)?)?;
        total = add(total, mul(self.horizon, r)?)?;
        total = add(total, linear(self.time_width, r)?)?;
        total = add(total, mul(self.horizon, q)?)?;
        Ok(total)
    }

    /// Conservative multiply-add estimate for one maximum-variate example.
    ///
    /// The bound includes patch/time/descriptor projections, depthwise
    /// temporal convolution, temporal gates, Q/K/V/output projections, both
    /// dense attention matrix products, both `SwiGLU` branches, and the
    /// low-rank forecast head. Elementwise operations are intentionally not
    /// used to make the estimate smaller.
    pub fn forward_multiply_adds(&self) -> Result<u64, ConfigError> {
        let product = |values: &[u64]| {
            values.iter().try_fold(1_u64, |total, value| {
                total.checked_mul(*value).ok_or(ConfigError::WorkOverflow)
            })
        };
        let sum = |values: &[u64]| {
            values.iter().try_fold(0_u64, |total, value| {
                total.checked_add(*value).ok_or(ConfigError::WorkOverflow)
            })
        };
        let value = |input: usize| u64::try_from(input).map_err(|_| ConfigError::WorkOverflow);

        let patches = value(self.patch_count()?)?;
        let variates = value(self.max_variates)?;
        let width = value(self.d_model)?;
        let patch_len = value(self.patch_len)?;
        let layers = value(self.layers)?;
        let kernel = value(self.temporal_kernel)?;
        let feed_forward = value(self.ff_width)?;
        let horizon_rank = value(self.horizon_rank)?;
        let horizon = value(self.horizon)?;
        let descriptor = value(self.descriptor_width)?;
        let time = value(self.time_width)?;
        let quantiles = value(QUANTILE_COUNT)?;

        let input = sum(&[
            product(&[patches, variates, 3, patch_len, width])?,
            product(&[patches, time, width])?,
            product(&[variates, descriptor, width])?,
        ])?;
        let block = sum(&[
            product(&[variates, patches, width, kernel])?,
            product(&[3, variates, patches, width, width])?,
            product(&[4, patches, variates, width, width])?,
            product(&[2, patches, variates, variates, width])?,
            product(&[3, patches, variates, width, feed_forward])?,
        ])?;
        let head = sum(&[
            product(&[2, variates, width, width])?,
            product(&[variates, width, quantiles, horizon_rank])?,
            product(&[horizon, time, horizon_rank])?,
            product(&[variates, quantiles, horizon_rank, horizon])?,
        ])?;
        sum(&[input, product(&[layers, block])?, head])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_parameter_counts_are_frozen() {
        let tiny = ForecastModelConfig::tiny_ci();
        let large = ForecastModelConfig::large_linux();
        assert_eq!(tiny.parameter_count().unwrap(), TINY_PARAMETER_COUNT);
        assert_eq!(large.parameter_count().unwrap(), LARGE_PARAMETER_COUNT);
        assert_eq!(tiny.patch_count().unwrap(), 15);
        assert_eq!(large.patch_count().unwrap(), 127);
        assert_eq!(tiny.forward_multiply_adds().unwrap(), 3_492_096);
        assert_eq!(
            large.forward_multiply_adds().unwrap(),
            LARGE_FORWARD_MULTIPLY_ADDS
        );
    }

    #[test]
    fn malformed_configs_fail_before_allocation() {
        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.temporal_kernel = 4;
        assert_eq!(cfg.validate(), Err(ConfigError::EvenKernel));

        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.quantiles[0] = 0.75;
        assert_eq!(cfg.validate(), Err(ConfigError::Quantiles));

        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.patch_stride = 5;
        assert_eq!(cfg.validate(), Err(ConfigError::PatchCoverage));

        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.patch_len = usize::MAX;
        assert!(cfg.patch_count().is_err());

        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.context_len = 16_384;
        cfg.patch_len = 1;
        cfg.patch_stride = 1;
        cfg.d_model = 1_024;
        cfg.max_variates = 128;
        assert!(matches!(cfg.validate(), Err(ConfigError::Capacity { .. })));

        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.context_len = 4_096;
        cfg.patch_len = 1;
        cfg.patch_stride = 1;
        cfg.d_model = 64;
        cfg.ff_width = 8_192;
        cfg.layers = 1;
        cfg.max_variates = 1;
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::Capacity {
                name: "activation_cells",
                ..
            })
        ));

        let mut cfg = ForecastModelConfig::tiny_ci();
        cfg.context_len = 4_096;
        cfg.patch_len = 1;
        cfg.patch_stride = 1;
        cfg.d_model = 8;
        cfg.layers = 32;
        cfg.variate_heads = 1;
        cfg.ff_width = 8;
        cfg.max_variates = 100;
        assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedProfile));
    }
}

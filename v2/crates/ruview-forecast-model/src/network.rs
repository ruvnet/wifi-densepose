//! Burn implementation of the `RuView` factorized forecast mixer.

use burn::{
    module::{Module, ModuleMapper, Param, ParamId},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        activation::{silu, softplus},
        backend::Backend,
        Bool, Distribution, Int, Tensor, TensorData,
    },
};
use burn_core as burn;
use burn_nn::{
    attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    conv::{Conv1d, Conv1dConfig},
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};

use crate::{ActivatedModelArtifact, ForecastModelConfig, ModelError, MAX_FORWARD_MULTIPLY_ADDS};

/// Maximum batch size accepted by the public tensor boundary.
pub const MAX_MODEL_BATCH: usize = 64;
/// Maximum total input cells accepted by one tensor batch.
pub const MAX_INPUT_CELLS: usize = 32 * 1024 * 1024;
/// Maximum conservative intermediate cells accepted by one forward pass.
pub const MAX_FORWARD_ACTIVATION_CELLS: usize = 256 * 1024 * 1024;

/// Canonical, already transformed and context-normalized tensor batch.
///
/// Missing payload values are multiplied by `observed_mask` inside the model,
/// so the numeric bytes in missing cells cannot influence a forecast.
#[derive(Debug, Clone)]
pub struct ModelInput<B: Backend> {
    /// Context values `[batch, context, variates]` in normalized feature space.
    normalized_values: Tensor<B, 3>,
    /// Binary float mask `[batch, context, variates]`.
    observed_mask: Tensor<B, 3>,
    /// Clipped age-since-last-observation channel `[batch, context, variates]`.
    age: Tensor<B, 3>,
    /// Periodic UTC context features `[batch, context, 8]`.
    context_time: Tensor<B, 3>,
    /// Known periodic UTC future features `[batch, horizon, 8]`.
    future_time: Tensor<B, 3>,
    /// Packed descriptors `[batch, variates, 72]` which move with a variate.
    descriptors: Tensor<B, 3>,
    /// Binary float validity mask `[batch, variates]`.
    series_valid: Tensor<B, 2>,
    config: ForecastModelConfig,
}

impl<B: Backend> ModelInput<B> {
    /// Validate and construct the only input value accepted by [`RuForecastMixer`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &ForecastModelConfig,
        normalized_values: Tensor<B, 3>,
        observed_mask: Tensor<B, 3>,
        age: Tensor<B, 3>,
        context_time: Tensor<B, 3>,
        future_time: Tensor<B, 3>,
        descriptors: Tensor<B, 3>,
        series_valid: Tensor<B, 2>,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        let input = Self {
            normalized_values,
            observed_mask,
            age,
            context_time,
            future_time,
            descriptors,
            series_valid,
            config: config.clone(),
        };
        input.validate(config)?;
        input.validate_values()?;
        Ok(input)
    }

    fn validate(&self, config: &ForecastModelConfig) -> Result<[usize; 2], ModelError> {
        if self.config != *config {
            return Err(ModelError::Shape(
                "input was validated for a different config".into(),
            ));
        }
        let [batch, context, variates] = self.normalized_values.dims();
        if batch == 0 || batch > MAX_MODEL_BATCH || variates == 0 || variates > config.max_variates
        {
            return Err(ModelError::Shape(
                "empty batch/variates or max_variates exceeded".into(),
            ));
        }
        if context != config.context_len
            || self.observed_mask.dims() != [batch, context, variates]
            || self.age.dims() != [batch, context, variates]
            || self.context_time.dims() != [batch, context, config.time_width]
            || self.future_time.dims() != [batch, config.horizon, config.time_width]
            || self.descriptors.dims() != [batch, variates, config.descriptor_width]
            || self.series_valid.dims() != [batch, variates]
        {
            return Err(ModelError::Shape(
                "tensor dimensions do not match artifact config".into(),
            ));
        }
        let context_cells = checked_cells(&[batch, context, variates])?;
        let context_time_cells = checked_cells(&[batch, context, config.time_width])?;
        let future_time_cells = checked_cells(&[batch, config.horizon, config.time_width])?;
        let descriptor_cells = checked_cells(&[batch, variates, config.descriptor_width])?;
        let total_input = context_cells
            .checked_mul(3)
            .and_then(|value| value.checked_add(context_time_cells))
            .and_then(|value| value.checked_add(future_time_cells))
            .and_then(|value| value.checked_add(descriptor_cells))
            .and_then(|value| value.checked_add(batch.checked_mul(variates)?))
            .ok_or_else(|| ModelError::Shape("input cell count overflow".into()))?;
        if total_input > MAX_INPUT_CELLS {
            return Err(ModelError::Shape("input cell limit exceeded".into()));
        }
        let activation_cells = config
            .activation_cells()?
            .checked_mul(batch)
            .ok_or_else(|| ModelError::Shape("activation cell count overflow".into()))?;
        if activation_cells > MAX_FORWARD_ACTIVATION_CELLS {
            return Err(ModelError::Shape(
                "forward activation cell limit exceeded".into(),
            ));
        }
        let forward_multiply_adds = config
            .forward_multiply_adds()?
            .checked_mul(
                u64::try_from(batch)
                    .map_err(|_| ModelError::Shape("forward multiply-add count overflow".into()))?,
            )
            .ok_or_else(|| ModelError::Shape("forward multiply-add count overflow".into()))?;
        if forward_multiply_adds > MAX_FORWARD_MULTIPLY_ADDS {
            return Err(ModelError::Shape(
                "forward multiply-add limit exceeded".into(),
            ));
        }
        Ok([batch, variates])
    }

    fn validate_values(&self) -> Result<(), ModelError> {
        require_finite("normalized_values", self.normalized_values.clone())?;
        require_binary("observed_mask", self.observed_mask.clone(), false)?;
        require_finite_range("age", self.age.clone(), 0.0, 1.0)?;
        require_finite("context_time", self.context_time.clone())?;
        require_finite("future_time", self.future_time.clone())?;
        require_finite("descriptors", self.descriptors.clone())?;
        require_binary("series_valid", self.series_valid.clone(), true)
    }

    /// Validated `[batch, variates]` shape.
    #[must_use]
    pub fn batch_variates(&self) -> [usize; 2] {
        let [batch, _, variates] = self.normalized_values.dims();
        [batch, variates]
    }
}

fn checked_cells(dimensions: &[usize]) -> Result<usize, ModelError> {
    dimensions.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| ModelError::Shape("tensor cell count overflow".into()))
    })
}

fn tensor_values<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, ModelError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|_| ModelError::Shape(format!("{name} could not be inspected")))
}

fn require_finite<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
) -> Result<(), ModelError> {
    if tensor_values(name, tensor)?
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(ModelError::Shape(format!(
            "{name} contains NaN or infinity"
        )));
    }
    Ok(())
}

fn require_finite_range<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
    minimum: f32,
    maximum: f32,
) -> Result<(), ModelError> {
    if tensor_values(name, tensor)?
        .iter()
        .any(|value| !value.is_finite() || *value < minimum || *value > maximum)
    {
        return Err(ModelError::Shape(format!(
            "{name} is outside its finite range"
        )));
    }
    Ok(())
}

fn require_binary<B: Backend, const D: usize>(
    name: &str,
    tensor: Tensor<B, D>,
    require_each_row: bool,
) -> Result<(), ModelError> {
    let dimensions = tensor.dims();
    let values = tensor_values(name, tensor)?;
    if values.iter().any(|value| {
        !matches!(value.classify(), std::num::FpCategory::Zero)
            && value.to_bits() != 1.0_f32.to_bits()
    }) {
        return Err(ModelError::Shape(format!(
            "{name} must contain only zero or one"
        )));
    }
    if require_each_row {
        let row = dimensions[D - 1];
        if row == 0
            || values
                .chunks_exact(row)
                .any(|values| !values.contains(&1.0))
        {
            return Err(ModelError::Shape(format!(
                "{name} has an empty-validity batch row"
            )));
        }
    }
    Ok(())
}

/// Model output in normalized feature space.
#[derive(Debug, Clone)]
pub struct ForecastModelOutput<B: Backend> {
    /// Strictly ordered quantiles for valid series `[batch, variates, horizon, 7]`.
    pub normalized_quantiles: Tensor<B, 4>,
    /// Per-variate forecast state `[batch, variates, d_model]` for `RuVector` indexing.
    pub state: Tensor<B, 3>,
    /// Validity mask copied from the canonical request `[batch, variates]`.
    pub series_valid: Tensor<B, 2>,
}

/// One optimizer batch.  The training crate owns loading and split isolation.
#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
    /// Model inputs.
    pub input: ModelInput<B>,
    /// Normalized targets `[batch, variates, horizon]`.
    pub targets: Tensor<B, 3>,
    /// Binary target mask `[batch, variates, horizon]`.
    pub target_mask: Tensor<B, 3>,
}

#[derive(Module, Debug)]
struct MixerBlock<B: Backend> {
    temporal_norm: LayerNorm<B>,
    temporal_conv: Conv1d<B>,
    temporal_gate: Linear<B>,
    temporal_out: Linear<B>,
    variate_norm: LayerNorm<B>,
    variate_attention: MultiHeadAttention<B>,
    ff_norm: LayerNorm<B>,
    ff_value: Linear<B>,
    ff_gate: Linear<B>,
    ff_out: Linear<B>,
    residual_dropout: Dropout,
    d_model: usize,
}

impl<B: Backend> MixerBlock<B> {
    fn init(config: &ForecastModelConfig, layer: usize, device: &B::Device) -> Self {
        let dilation = [1, 2, 4, 8][layer % 4];
        let padding = dilation * (config.temporal_kernel - 1) / 2;
        Self {
            temporal_norm: LayerNormConfig::new(config.d_model).init(device),
            temporal_conv: Conv1dConfig::new(
                config.d_model,
                config.d_model,
                config.temporal_kernel,
            )
            .with_groups(config.d_model)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(padding, padding))
            .init(device),
            temporal_gate: LinearConfig::new(config.d_model, 2 * config.d_model).init(device),
            temporal_out: LinearConfig::new(config.d_model, config.d_model).init(device),
            variate_norm: LayerNormConfig::new(config.d_model).init(device),
            variate_attention: MultiHeadAttentionConfig::new(config.d_model, config.variate_heads)
                .with_dropout(config.dropout)
                .init(device),
            ff_norm: LayerNormConfig::new(config.d_model).init(device),
            ff_value: LinearConfig::new(config.d_model, config.ff_width).init(device),
            ff_gate: LinearConfig::new(config.d_model, config.ff_width).init(device),
            ff_out: LinearConfig::new(config.ff_width, config.d_model).init(device),
            residual_dropout: DropoutConfig::new(config.dropout).init(),
            d_model: config.d_model,
        }
    }

    fn forward(&self, mut tokens: Tensor<B, 4>, series_valid: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch, variates, patches, width] = tokens.dims();
        debug_assert_eq!(width, self.d_model);

        let temporal = self.temporal_norm.forward(tokens.clone());
        let temporal = temporal
            .permute([0, 1, 3, 2])
            .reshape([batch * variates, width, patches]);
        let temporal = self.temporal_conv.forward(temporal);
        let temporal = temporal
            .reshape([batch, variates, width, patches])
            .permute([0, 1, 3, 2]);
        let gated = self.temporal_gate.forward(temporal);
        let value = gated
            .clone()
            .slice([0..batch, 0..variates, 0..patches, 0..width]);
        let gate = gated.slice([0..batch, 0..variates, 0..patches, width..2 * width]);
        let temporal = self.temporal_out.forward(value * silu(gate));
        tokens = tokens + self.residual_dropout.forward(temporal);

        let variate = self
            .variate_norm
            .forward(tokens.clone())
            .permute([0, 2, 1, 3])
            .reshape([batch * patches, variates, width]);
        let pad_mask = series_valid
            .clone()
            .lower_equal_elem(0.5)
            .reshape([batch, 1, variates])
            .repeat_dim(1, patches)
            .reshape([batch * patches, variates]);
        let variate = self
            .variate_attention
            .forward(MhaInput::self_attn(variate).mask_pad(pad_mask))
            .context
            .reshape([batch, patches, variates, width])
            .permute([0, 2, 1, 3]);
        tokens = tokens + self.residual_dropout.forward(variate);

        let ff = self.ff_norm.forward(tokens.clone());
        let ff = self.ff_value.forward(ff.clone()) * silu(self.ff_gate.forward(ff));
        let ff = self.ff_out.forward(ff);
        let valid = series_valid.reshape([batch, variates, 1, 1]);
        (tokens + self.residual_dropout.forward(ff)) * valid
    }
}

/// RuView-specific multivariate forecast network.
#[derive(Module, Debug)]
pub struct RuForecastMixer<B: Backend> {
    patch_projection: Linear<B>,
    descriptor_projection: Linear<B>,
    context_time_projection: Linear<B>,
    blocks: Vec<MixerBlock<B>>,
    pool_projection: Linear<B>,
    final_norm: LayerNorm<B>,
    state_to_coefficients: Linear<B>,
    horizon_basis: Param<Tensor<B, 2>>,
    future_time_projection: Linear<B>,
    horizon_bias: Param<Tensor<B, 2>>,
    context_len: usize,
    horizon: usize,
    patch_len: usize,
    patch_stride: usize,
    patch_count: usize,
    d_model: usize,
    horizon_rank: usize,
    max_variates: usize,
    descriptor_width: usize,
    time_width: usize,
    temporal_kernel: usize,
    variate_heads: usize,
    ff_width: usize,
    dropout: f64,
}

impl<B: Backend> RuForecastMixer<B> {
    /// Initialize a validated architecture on an explicit backend device.
    pub fn init(config: &ForecastModelConfig, device: &B::Device) -> Result<Self, ModelError> {
        config.validate()?;
        let q = config.quantiles.len();
        Ok(Self {
            patch_projection: LinearConfig::new(3 * config.patch_len, config.d_model).init(device),
            descriptor_projection: LinearConfig::new(config.descriptor_width, config.d_model)
                .init(device),
            context_time_projection: LinearConfig::new(config.time_width, config.d_model)
                .init(device),
            blocks: (0..config.layers)
                .map(|layer| MixerBlock::init(config, layer, device))
                .collect(),
            pool_projection: LinearConfig::new(2 * config.d_model, config.d_model).init(device),
            final_norm: LayerNormConfig::new(config.d_model).init(device),
            state_to_coefficients: LinearConfig::new(config.d_model, q * config.horizon_rank)
                .init(device),
            horizon_basis: Param::from_tensor(Tensor::random(
                [config.horizon, config.horizon_rank],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            future_time_projection: LinearConfig::new(config.time_width, config.horizon_rank)
                .init(device),
            horizon_bias: Param::from_tensor(Tensor::zeros([config.horizon, q], device)),
            context_len: config.context_len,
            horizon: config.horizon,
            patch_len: config.patch_len,
            patch_stride: config.patch_stride,
            patch_count: config.patch_count()?,
            d_model: config.d_model,
            horizon_rank: config.horizon_rank,
            max_variates: config.max_variates,
            descriptor_width: config.descriptor_width,
            time_width: config.time_width,
            temporal_kernel: config.temporal_kernel,
            variate_heads: config.variate_heads,
            ff_width: config.ff_width,
            dropout: config.dropout,
        })
    }

    /// Activate only a signature-verified, shape-compatible artifact.
    pub fn from_activated_artifact(
        config: &ForecastModelConfig,
        artifact: &ActivatedModelArtifact,
        device: &B::Device,
    ) -> Result<Self, ModelError> {
        if artifact.manifest().config != *config {
            return Err(ModelError::Malformed(
                "activated config differs from artifact",
            ));
        }
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load(artifact.weights().to_vec(), device)
            .map_err(|error| ModelError::Record(error.to_string()))?;
        let model = Self::init(config, device)?.load_record(record);
        if model.num_params() != artifact.manifest().parameter_count {
            return Err(ModelError::Record(
                "decoded record parameter count mismatch".into(),
            ));
        }
        Ok(model)
    }

    /// Forward one canonical batch.
    pub fn forward(&self, input: ModelInput<B>) -> Result<ForecastModelOutput<B>, ModelError> {
        let config = self.config_view();
        let [batch, variates] = input.validate(&config)?;
        let valid4 = input.series_valid.clone().reshape([batch, variates, 1, 1]);
        let descriptors = self.descriptor_projection.forward(input.descriptors);
        let values = input.normalized_values * input.observed_mask.clone();
        let mut patches = Vec::with_capacity(self.patch_count);

        for patch in 0..self.patch_count {
            let start = patch * self.patch_stride;
            let end = start + self.patch_len;
            let values_patch = values
                .clone()
                .slice([0..batch, start..end, 0..variates])
                .swap_dims(1, 2);
            let mask_patch = input
                .observed_mask
                .clone()
                .slice([0..batch, start..end, 0..variates])
                .swap_dims(1, 2);
            let age_patch = input
                .age
                .clone()
                .slice([0..batch, start..end, 0..variates])
                .swap_dims(1, 2);
            let packed = Tensor::cat(vec![values_patch, mask_patch, age_patch], 2);
            let endpoint_time = input
                .context_time
                .clone()
                .slice([0..batch, end - 1..end, 0..self.time_width])
                .squeeze_dim::<2>(1);
            let time = self
                .context_time_projection
                .forward(endpoint_time)
                .reshape([batch, 1, self.d_model])
                .repeat_dim(1, variates);
            patches.push(self.patch_projection.forward(packed) + descriptors.clone() + time);
        }

        let mut tokens = Tensor::stack(patches, 2) * valid4.clone();
        for block in &self.blocks {
            tokens = block.forward(tokens, input.series_valid.clone());
        }

        let mean = tokens.clone().mean_dim(2).squeeze_dim::<3>(2);
        let last = tokens
            .slice([
                0..batch,
                0..variates,
                self.patch_count - 1..self.patch_count,
                0..self.d_model,
            ])
            .squeeze_dim::<3>(2);
        let state = self.final_norm.forward(
            self.pool_projection
                .forward(Tensor::cat(vec![last, mean], 2)),
        ) * input.series_valid.clone().reshape([batch, variates, 1]);

        let q = crate::config::QUANTILE_COUNT;
        let coefficients = self.state_to_coefficients.forward(state.clone()).reshape([
            batch,
            variates * q,
            self.horizon_rank,
        ]);
        let horizon = self
            .horizon_basis
            .val()
            .reshape([1, self.horizon, self.horizon_rank])
            .repeat_dim(0, batch)
            + self.future_time_projection.forward(input.future_time);
        let raw = coefficients
            .matmul(horizon.swap_dims(1, 2))
            .reshape([batch, variates, q, self.horizon])
            .swap_dims(2, 3)
            + self.horizon_bias.val().reshape([1, 1, self.horizon, q]);
        let ordered = ordered_quantiles(&raw) * valid4;
        let output = ForecastModelOutput {
            normalized_quantiles: ordered,
            state,
            series_valid: input.series_valid,
        };
        output.validate()?;
        Ok(output)
    }

    fn config_view(&self) -> ForecastModelConfig {
        ForecastModelConfig {
            context_len: self.context_len,
            horizon: self.horizon,
            patch_len: self.patch_len,
            patch_stride: self.patch_stride,
            d_model: self.d_model,
            layers: self.blocks.len(),
            temporal_kernel: self.temporal_kernel,
            variate_heads: self.variate_heads,
            ff_width: self.ff_width,
            horizon_rank: self.horizon_rank,
            max_variates: self.max_variates,
            dropout: self.dropout,
            quantiles: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            descriptor_width: self.descriptor_width,
            time_width: self.time_width,
        }
    }
}

impl<B: Backend> ForecastModelOutput<B> {
    fn validate(&self) -> Result<(), ModelError> {
        require_finite("forecast_quantiles", self.normalized_quantiles.clone())?;
        require_finite("forecast_state", self.state.clone())?;
        let [_, _, horizon, quantiles] = self.normalized_quantiles.dims();
        let values = tensor_values("forecast_quantiles", self.normalized_quantiles.clone())?;
        let valid = tensor_values("series_valid", self.series_valid.clone())?;
        for (row_index, row) in values.chunks_exact(quantiles).enumerate() {
            let series_index = row_index / horizon;
            if valid.get(series_index) == Some(&1.0)
                && row.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(ModelError::Shape(
                    "forecast quantiles are not strictly ordered".into(),
                ));
            }
        }
        Ok(())
    }
}

fn ordered_quantiles<B: Backend>(raw: &Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, variates, horizon, _] = raw.dims();
    let part = |index: usize| {
        raw.clone()
            .slice([0..batch, 0..variates, 0..horizon, index..index + 1])
    };
    let median = part(3);
    let q2 = median.clone() - (softplus(part(2), 1.0) + 1.0e-4);
    let q1 = q2.clone() - (softplus(part(1), 1.0) + 1.0e-4);
    let q0 = q1.clone() - (softplus(part(0), 1.0) + 1.0e-4);
    let q4 = median.clone() + softplus(part(4), 1.0) + 1.0e-4;
    let q5 = q4.clone() + softplus(part(5), 1.0) + 1.0e-4;
    let q6 = q5.clone() + softplus(part(6), 1.0) + 1.0e-4;
    Tensor::cat(vec![q0, q1, q2, median, q4, q5, q6], 3)
}

/// Masked equal-weight pinball loss in normalized feature space.
pub fn masked_pinball_loss<B: Backend>(
    output: &ForecastModelOutput<B>,
    targets: Tensor<B, 3>,
    target_mask: Tensor<B, 3>,
    device: &B::Device,
) -> Result<Tensor<B, 1>, ModelError> {
    let [batch, variates, horizon, q] = output.normalized_quantiles.dims();
    if targets.dims() != [batch, variates, horizon]
        || target_mask.dims() != [batch, variates, horizon]
        || q != crate::config::QUANTILE_COUNT
    {
        return Err(ModelError::Shape(
            "loss target dimensions disagree with output".into(),
        ));
    }
    require_finite("targets", targets.clone())?;
    require_binary("target_mask", target_mask.clone(), false)?;
    let mask_values = tensor_values("target_mask", target_mask.clone())?;
    let valid_values = tensor_values("series_valid", output.series_valid.clone())?;
    let has_target = mask_values.iter().enumerate().any(|(index, observed)| {
        observed.to_bits() == 1.0_f32.to_bits()
            && valid_values[index / horizon].to_bits() == 1.0_f32.to_bits()
    });
    if !has_target {
        return Err(ModelError::Shape(
            "loss batch has no valid observed target".into(),
        ));
    }
    let error =
        targets.reshape([batch, variates, horizon, 1]) - output.normalized_quantiles.clone();
    let indicator = error.clone().lower_elem(0.0).float();
    let levels = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![0.05_f32, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            [1, 1, 1, q],
        ),
        device,
    );
    let mask = target_mask.reshape([batch, variates, horizon, 1])
        * output.series_valid.clone().reshape([batch, variates, 1, 1]);
    let denominator = mask.clone().sum().clamp_min(1.0).mul_scalar(q as f32);
    Ok(((levels - indicator) * error * mask).sum() / denominator)
}

/// Serialize a trainer-owned model record for a verified artifact envelope.
pub fn record_to_bytes<B: Backend>(model: RuForecastMixer<B>) -> Result<Vec<u8>, ModelError> {
    let mut canonical_ids = CanonicalParamIds { next: 1 };
    let model = model.map(&mut canonical_ids);
    NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .record(model.into_record(), ())
        .map_err(|error| ModelError::Record(error.to_string()))
}

struct CanonicalParamIds {
    next: u64,
}

impl<B: Backend> ModuleMapper<B> for CanonicalParamIds {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (_, tensor, mapper) = param.consume();
        let id = ParamId::from(self.next);
        self.next = self.next.saturating_add(1);
        Param::from_mapped_value(id, tensor, mapper)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        let (_, tensor, mapper) = param.consume();
        let id = ParamId::from(self.next);
        self.next = self.next.saturating_add(1);
        Param::from_mapped_value(id, tensor, mapper)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Bool>>,
    ) -> Param<Tensor<B, D, Bool>> {
        let (_, tensor, mapper) = param.consume();
        let id = ParamId::from(self.next);
        self.next = self.next.saturating_add(1);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use burn_core::module::Module;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type Cpu = NdArray<f32>;
    static BURN_RNG_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn input(config: &ForecastModelConfig, device: NdArrayDevice) -> ModelInput<Cpu> {
        let [batch, variates] = [2, 4];
        ModelInput::new(
            config,
            Tensor::zeros([batch, config.context_len, variates], &device),
            Tensor::ones([batch, config.context_len, variates], &device),
            Tensor::zeros([batch, config.context_len, variates], &device),
            Tensor::zeros([batch, config.context_len, config.time_width], &device),
            Tensor::zeros([batch, config.horizon, config.time_width], &device),
            Tensor::zeros([batch, variates, config.descriptor_width], &device),
            Tensor::ones([batch, variates], &device),
        )
        .unwrap()
    }

    #[test]
    fn tiny_forward_shape_and_quantile_order() {
        let _guard = BURN_RNG_TEST_LOCK.lock().unwrap();
        let device = NdArrayDevice::default();
        Cpu::seed(&device, 17);
        let config = ForecastModelConfig::tiny_ci();
        let model = RuForecastMixer::<Cpu>::init(&config, &device).unwrap();
        assert_eq!(model.num_params(), crate::TINY_PARAMETER_COUNT);
        let output = model.forward(input(&config, device)).unwrap();
        assert_eq!(output.normalized_quantiles.dims(), [2, 4, 12, 7]);
        let data = output
            .normalized_quantiles
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for row in data.chunks_exact(7) {
            assert!(row.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn same_seed_fresh_models_have_canonical_record_bytes() {
        let _guard = BURN_RNG_TEST_LOCK.lock().unwrap();
        let device = NdArrayDevice::default();
        Cpu::seed(&device, 29);
        let config = ForecastModelConfig::tiny_ci();
        let first =
            record_to_bytes(RuForecastMixer::<Cpu>::init(&config, &device).unwrap()).unwrap();
        Cpu::seed(&device, 29);
        let second =
            record_to_bytes(RuForecastMixer::<Cpu>::init(&config, &device).unwrap()).unwrap();
        assert!(
            first == second,
            "canonical record digests differ: {} != {}",
            blake3::hash(&first),
            blake3::hash(&second)
        );
    }

    #[test]
    fn malformed_shape_is_rejected_without_backend_panic() {
        let device = NdArrayDevice::default();
        let config = ForecastModelConfig::tiny_ci();
        let malformed = ModelInput::<Cpu>::new(
            &config,
            Tensor::zeros([2, config.context_len, 4], &device),
            Tensor::ones([2, config.context_len, 4], &device),
            Tensor::zeros([2, config.context_len - 1, 4], &device),
            Tensor::zeros([2, config.context_len, config.time_width], &device),
            Tensor::zeros([2, config.horizon, config.time_width], &device),
            Tensor::zeros([2, 4, config.descriptor_width], &device),
            Tensor::ones([2, 4], &device),
        );
        assert!(matches!(malformed, Err(ModelError::Shape(_))));
    }

    #[test]
    fn nonfinite_and_nonbinary_inputs_fail_closed() {
        let device = NdArrayDevice::default();
        let config = ForecastModelConfig::tiny_ci();
        let nonfinite = ModelInput::<Cpu>::new(
            &config,
            Tensor::full([1, config.context_len, 1], f32::NAN, &device),
            Tensor::ones([1, config.context_len, 1], &device),
            Tensor::zeros([1, config.context_len, 1], &device),
            Tensor::zeros([1, config.context_len, config.time_width], &device),
            Tensor::zeros([1, config.horizon, config.time_width], &device),
            Tensor::zeros([1, 1, config.descriptor_width], &device),
            Tensor::ones([1, 1], &device),
        );
        assert!(matches!(nonfinite, Err(ModelError::Shape(_))));

        let nonbinary = ModelInput::<Cpu>::new(
            &config,
            Tensor::zeros([1, config.context_len, 1], &device),
            Tensor::full([1, config.context_len, 1], 0.5, &device),
            Tensor::zeros([1, config.context_len, 1], &device),
            Tensor::zeros([1, config.context_len, config.time_width], &device),
            Tensor::zeros([1, config.horizon, config.time_width], &device),
            Tensor::zeros([1, 1, config.descriptor_width], &device),
            Tensor::ones([1, 1], &device),
        );
        assert!(matches!(nonbinary, Err(ModelError::Shape(_))));
    }
}

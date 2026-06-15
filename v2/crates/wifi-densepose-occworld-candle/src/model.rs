//! Weight loading utilities for the OccWorld SafeTensors checkpoint.
//!
//! Phase-5 retraining produces a `.safetensors` file whose tensor keys
//! follow PyTorch naming conventions (e.g. `encoder.conv_in.weight`).
//! The functions here map those keys to the Candle `VarBuilder` sub-path
//! convention used in this crate (e.g. `enc.conv_in.weight`).

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::error::OccWorldError;

/// Load all tensors from a SafeTensors file into a key→Tensor map.
///
/// Returns `Err(OccWorldError::CheckpointNotFound)` if the path does not
/// exist, so callers can gracefully fall back to the Python bridge.
pub fn load_safetensors(
    path: &Path,
    device: &Device,
) -> Result<HashMap<String, Tensor>, OccWorldError> {
    if !path.exists() {
        return Err(OccWorldError::CheckpointNotFound(
            path.display().to_string(),
        ));
    }

    // Read the raw bytes; safetensors requires the full file in memory.
    let bytes = std::fs::read(path)?;
    let named_tensors = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| OccWorldError::CheckpointParse(e.to_string()))?;

    let mut map = HashMap::new();
    for (name, view) in named_tensors.tensors() {
        let candle_key = map_pytorch_key(&name);
        let dtype = safetensor_dtype_to_candle(view.dtype())
            .ok_or_else(|| OccWorldError::CheckpointParse(
                format!("unsupported dtype for key '{name}'"),
            ))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let tensor = Tensor::from_raw_buffer(data, dtype, &shape, device)
            .map_err(OccWorldError::Candle)?;
        map.insert(candle_key, tensor);
    }
    Ok(map)
}

/// Map a PyTorch weight key to the Candle naming convention used here.
///
/// # Mapping rules
///
/// | PyTorch prefix         | Candle prefix          |
/// |------------------------|------------------------|
/// | `encoder.`             | `enc.`                 |
/// | `decoder.`             | `dec.`                 |
/// | `quantize.`            | `quantize.`            |
/// | `quant_conv.`          | `quant_conv.`          |
/// | `post_quant_conv.`     | `post_quant_conv.`     |
/// | `transformer.`         | `transformer.`         |
/// | `class_embedding.`     | `class_embed.`         |
///
/// All other keys are passed through unchanged.  Extend this function
/// whenever the checkpoint adds new top-level modules.
pub fn map_pytorch_key(key: &str) -> String {
    // Strip any leading "model." prefix that PyTorch Lightning adds
    let key = key.strip_prefix("model.").unwrap_or(key);

    if let Some(rest) = key.strip_prefix("encoder.") {
        return format!("enc.{rest}");
    }
    if let Some(rest) = key.strip_prefix("decoder.") {
        return format!("dec.{rest}");
    }
    if let Some(rest) = key.strip_prefix("class_embedding.") {
        return format!("class_embed.{rest}");
    }

    // No transformation needed for these prefixes
    key.to_owned()
}

/// Convert a `safetensors::Dtype` to a `candle_core::DType`.
///
/// Returns `None` for unsupported variants (e.g. BF16 on CPU without
/// the `bf16` feature).
fn safetensor_dtype_to_candle(dt: safetensors::Dtype) -> Option<candle_core::DType> {
    use candle_core::DType;
    use safetensors::Dtype;
    match dt {
        Dtype::F32 => Some(DType::F32),
        Dtype::F64 => Some(DType::F64),
        Dtype::F16 => Some(DType::F16),
        Dtype::BF16 => Some(DType::BF16),
        Dtype::I32 => Some(DType::I64), // widen for Candle compatibility
        Dtype::I64 => Some(DType::I64),
        Dtype::U8 => Some(DType::U8),
        Dtype::U32 => Some(DType::U32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_pytorch_key_encoder() {
        assert_eq!(
            map_pytorch_key("encoder.conv_in.weight"),
            "enc.conv_in.weight"
        );
    }

    #[test]
    fn test_map_pytorch_key_decoder() {
        assert_eq!(
            map_pytorch_key("decoder.conv_out.bias"),
            "dec.conv_out.bias"
        );
    }

    #[test]
    fn test_map_pytorch_key_class_embedding() {
        assert_eq!(
            map_pytorch_key("class_embedding.weight"),
            "class_embed.weight"
        );
    }

    #[test]
    fn test_map_pytorch_key_passthrough() {
        assert_eq!(
            map_pytorch_key("quantize.embedding.weight"),
            "quantize.embedding.weight"
        );
        assert_eq!(
            map_pytorch_key("quant_conv.weight"),
            "quant_conv.weight"
        );
        assert_eq!(
            map_pytorch_key("transformer.layer_0.ffn.fc1.weight"),
            "transformer.layer_0.ffn.fc1.weight"
        );
    }

    #[test]
    fn test_map_pytorch_key_lightning_prefix() {
        // PyTorch Lightning wraps everything under "model."
        assert_eq!(
            map_pytorch_key("model.encoder.conv_in.weight"),
            "enc.conv_in.weight"
        );
    }

    #[test]
    fn test_load_nonexistent_checkpoint() {
        let device = candle_core::Device::Cpu;
        let result = load_safetensors(Path::new("/nonexistent/checkpoint.safetensors"), &device);
        assert!(
            matches!(result, Err(OccWorldError::CheckpointNotFound(_))),
            "expected CheckpointNotFound, got {result:?}"
        );
    }
}

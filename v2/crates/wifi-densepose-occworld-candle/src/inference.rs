//! Top-level inference engine — `OccWorldCandle`.
//!
//! Provides the public-facing API:
//! - `OccWorldCandle::load` — load from a SafeTensors checkpoint
//! - `OccWorldCandle::dummy` — random weights for testing / benchmarking
//! - `OccWorldCandle::predict` — infer 15 future occupancy frames
//!
//! The `dummy` constructor allows end-to-end benchmarking (wall-clock timing,
//! shape verification, memory footprint) before the Phase-5 checkpoint exists.

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::OccWorldConfig;
use crate::error::OccWorldError;
use crate::transformer::OccWorldTransformer;
use crate::vqvae::{decode_to_logits, encode_occupancy, VQVAEComponents};

// ── Output types ─────────────────────────────────────────────────────────────

/// A predicted future trajectory waypoint in 3-D grid coordinates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrajectoryWaypoint {
    /// Frame index within the prediction horizon (0 = first predicted frame).
    pub frame: usize,
    /// Grid X position of the predicted agent centroid.
    pub grid_x: f32,
    /// Grid Y position of the predicted agent centroid.
    pub grid_y: f32,
    /// Grid Z position of the predicted agent centroid.
    pub grid_z: f32,
    /// Confidence score in `[0, 1]`.
    pub confidence: f32,
}

/// Outputs produced by one call to `OccWorldCandle::predict`.
pub struct InferenceOutput {
    /// Predicted semantic class for each voxel.
    ///
    /// Shape: `(1, 15, 200, 200, 16)`, dtype `u8`.
    /// Values are class indices in `[0, num_classes)`.
    pub sem_pred: Tensor,

    /// Trajectory priors extracted from the predicted occupancy.
    ///
    /// One waypoint per predicted frame, centred on the non-free voxel
    /// with the highest occupancy probability.  Empty when the model
    /// predicts all frames as free space.
    ///
    /// **Honesty note:** these priors are always computed by the *real*
    /// convolutional forward pass (encoder → VQ → transformer → decoder).
    /// When [`InferenceOutput::weights_trained`] is `false` they are a
    /// deterministic, input-dependent function of the input but come from an
    /// **untrained** network — do not treat them as trained-model accuracy.
    pub trajectory_priors: Vec<TrajectoryWaypoint>,

    /// Whether the weights driving this prediction came from a trained
    /// checkpoint.
    ///
    /// * `true`  — produced by [`OccWorldCandle::load`] from a real
    ///   SafeTensors checkpoint; priors reflect trained-model behaviour.
    /// * `false` — produced by [`OccWorldCandle::dummy`] with deterministic
    ///   but **untrained** weights. The forward pass is real and
    ///   input-dependent, but accuracy is *data-gated*: consumers MUST NOT
    ///   present these priors as trained predictions.
    ///
    /// This flag is the explicit, machine-readable disclosure that replaces
    /// the old silently-fake `randn` stubs.
    pub weights_trained: bool,

    /// Wall-clock time for the full `predict` call in milliseconds.
    pub inference_ms: f64,
}

// ── Main engine ───────────────────────────────────────────────────────────────

/// Native Rust OccWorld inference engine backed by Candle.
///
/// # Loading
///
/// ```no_run
/// # use wifi_densepose_occworld_candle::inference::OccWorldCandle;
/// # use wifi_densepose_occworld_candle::config::OccWorldConfig;
/// # use candle_core::Device;
/// # use std::path::Path;
/// let cfg = OccWorldConfig::default();
/// match OccWorldCandle::load(Path::new("/path/to/occworld.safetensors"), cfg) {
///     Ok(engine) => { /* use engine */ }
///     Err(_) => { /* fall back to Python bridge */ }
/// }
/// ```
pub struct OccWorldCandle {
    // Note: Device does not implement Debug; derive manually below.
    config: OccWorldConfig,
    vqvae: VQVAEComponents,
    transformer: OccWorldTransformer,
    device: Device,
    /// `true` when weights came from a real checkpoint via [`Self::load`];
    /// `false` for [`Self::dummy`] (deterministic but untrained).
    weights_trained: bool,
}

impl std::fmt::Debug for OccWorldCandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OccWorldCandle")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl OccWorldCandle {
    /// Load model weights from a SafeTensors checkpoint.
    ///
    /// Returns `Err` if the checkpoint does not exist, so callers can
    /// gracefully fall back to the Python bridge (`wifi-densepose-worldmodel`).
    pub fn load(
        checkpoint_path: &Path,
        config: OccWorldConfig,
    ) -> Result<Self, OccWorldError> {
        if !checkpoint_path.exists() {
            return Err(OccWorldError::CheckpointNotFound(
                checkpoint_path.display().to_string(),
            ));
        }

        let device = pick_device();

        // Load weights through the safe file-read path in `model::load_safetensors`.
        // This avoids the `unsafe` mmap block forbidden by our lint config, at the
        // cost of reading the full file into memory rather than memory-mapping it.
        // Switch to `VarBuilder::from_mmaped_safetensors` (in a crate that allows
        // unsafe) once the checkpoint is large enough that mmap matters.
        let tensors = crate::model::load_safetensors(checkpoint_path, &device)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let vqvae = VQVAEComponents::new(&config, vb.clone()).map_err(OccWorldError::Candle)?;
        let transformer =
            OccWorldTransformer::new(config.clone(), vb).map_err(OccWorldError::Candle)?;

        Ok(Self {
            config,
            vqvae,
            transformer,
            device,
            // A checkpoint was successfully loaded → weights are trained.
            weights_trained: true,
        })
    }

    /// Construct with deterministic *untrained* weights for testing and
    /// benchmarking.
    ///
    /// All shapes are correct and the forward pass is real and
    /// input-dependent; no checkpoint is required. Predictions are flagged
    /// `weights_trained: false` so consumers know accuracy is data-gated.
    pub fn dummy(config: OccWorldConfig, device: Device) -> Result<Self, OccWorldError> {
        let vqvae =
            VQVAEComponents::dummy(&config, &device).map_err(OccWorldError::Candle)?;
        let transformer =
            OccWorldTransformer::dummy(config.clone(), &device).map_err(OccWorldError::Candle)?;
        Ok(Self {
            config,
            vqvae,
            transformer,
            device,
            // Deterministic but untrained → honestly flagged as not trained.
            weights_trained: false,
        })
    }

    /// Whether this engine is backed by trained weights (`true`) or
    /// deterministic-but-untrained `dummy` weights (`false`).
    pub fn weights_trained(&self) -> bool {
        self.weights_trained
    }

    /// The Candle device this engine runs on (CPU, or CUDA when the `cuda`
    /// feature is enabled and a GPU is available).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Infer 15 future occupancy frames from 16 past frames.
    ///
    /// # Arguments
    /// * `past_occupancy` — `(1, 16, 200, 200, 16)` tensor of `u8` class indices.
    ///
    /// # Returns
    /// [`InferenceOutput`] containing:
    /// - `sem_pred`: `(1, 15, 200, 200, 16)` u8 predicted class indices
    /// - `trajectory_priors`: one waypoint per predicted frame
    /// - `inference_ms`: wall-clock latency
    pub fn predict(&self, past_occupancy: &Tensor) -> Result<InferenceOutput, OccWorldError> {
        let t0 = Instant::now();

        let cfg = &self.config;
        let (b, f_in, h, w, d) = past_occupancy.dims5().map_err(OccWorldError::Candle)?;

        if h != cfg.grid_h || w != cfg.grid_w || d != cfg.grid_d {
            return Err(OccWorldError::ShapeMismatch(format!(
                "expected past_occupancy (_, _, {}, {}, {}), got (_, _, {h}, {w}, {d})",
                cfg.grid_h, cfg.grid_w, cfg.grid_d
            )));
        }

        // ── Step 1: VQVAE encode each past frame ──────────────────────────
        // Flatten batch*frames: (B, F, H, W, D) → (B*F, H, W, D)
        let occ_flat = past_occupancy
            .reshape((b * f_in, h, w, d))
            .map_err(OccWorldError::Candle)?;

        // Cast to u32 for class embedding (input is u8)
        let occ_u32 = occ_flat
            .to_dtype(DType::U32)
            .map_err(OccWorldError::Candle)?;

        // Class embedding → (B*F, base_channels, H, W*D)
        let embedded = self
            .vqvae
            .class_embed
            .forward(&occ_u32, cfg.grid_d)
            .map_err(OccWorldError::Candle)?;

        // Real conv encoder → (B*F, z_channels, token_h, token_w).
        // Deterministic and input-dependent — no randn.
        let z = encode_occupancy(&self.vqvae.encoder, &embedded)
            .map_err(OccWorldError::Candle)?;

        // quant_conv → (B*F, embed_dim, token_h, token_w)
        let z_e = self
            .vqvae
            .quant_conv
            .forward(&z)
            .map_err(OccWorldError::Candle)?;

        // Vector quantisation → z_q (B*F, embed_dim, token_h, token_w), indices
        // Reshape to (B*F, H*W, embed_dim) for VQCodebook.encode
        let (bf, e_dim, th, tw) = z_e.dims4().map_err(OccWorldError::Candle)?;
        let z_e_flat = z_e
            .permute((0, 2, 3, 1)) // (B*F, th, tw, embed_dim)
            .map_err(OccWorldError::Candle)?
            .reshape((bf, th * tw, e_dim))
            .map_err(OccWorldError::Candle)?;

        let (z_q_flat, _indices) = self
            .vqvae
            .codebook
            .encode(&z_e_flat)
            .map_err(OccWorldError::Candle)?;

        // Back to (B*F, embed_dim, th, tw) → (B, F, embed_dim, th, tw)
        let z_q = z_q_flat
            .reshape((bf, th, tw, e_dim))
            .map_err(OccWorldError::Candle)?
            .permute((0, 3, 1, 2)) // (B*F, embed_dim, th, tw)
            .map_err(OccWorldError::Candle)?
            .reshape((b, f_in, e_dim, th, tw))
            .map_err(OccWorldError::Candle)?;

        // ── Step 2: Transformer predicts future token logits ──────────────
        // Output: (B, F_out, vocab, th, tw)
        let pred_logits = self.transformer.forward(&z_q)?;

        let f_out = pred_logits.dim(1).map_err(OccWorldError::Candle)?;

        // ── Step 3: Argmax over vocab dim → predicted token indices ───────
        let pred_indices = pred_logits
            .argmax(2) // (B, F_out, th, tw)  — over vocab dim
            .map_err(OccWorldError::Candle)?;

        // ── Step 4: Decode token indices → z_q values ────────────────────
        // Flatten to (B*F_out * th * tw,) for codebook lookup
        let idx_flat = pred_indices
            .flatten_all()
            .map_err(OccWorldError::Candle)?;
        let z_decoded = self
            .vqvae
            .codebook
            .decode(&idx_flat)
            .map_err(OccWorldError::Candle)?; // (B*F_out*th*tw, embed_dim)

        // Reshape to (B*F_out, embed_dim, th, tw) for post_quant_conv
        let z_dec_4d = z_decoded
            .reshape((b * f_out, e_dim, th, tw))
            .map_err(OccWorldError::Candle)?;

        let z_post = self
            .vqvae
            .post_quant_conv
            .forward(&z_dec_4d)
            .map_err(OccWorldError::Candle)?;

        // ── Step 5: Real conv decoder → class logits → class predictions ──
        let class_logits = decode_to_logits(&self.vqvae.decoder, &z_post)
            .map_err(OccWorldError::Candle)?;
        // class_logits: (B*F_out, num_classes, H, W, D)
        // Argmax over class dim → (B*F_out, H, W, D)
        let sem_flat = class_logits
            .argmax(1)
            .map_err(OccWorldError::Candle)?
            .to_dtype(DType::U8)
            .map_err(OccWorldError::Candle)?;

        let sem_pred = sem_flat
            .reshape((b, f_out, cfg.grid_h, cfg.grid_w, cfg.grid_d))
            .map_err(OccWorldError::Candle)?;

        // ── Step 6: Extract trajectory priors ─────────────────────────────
        let trajectory_priors = extract_trajectory_priors(&sem_pred, cfg, f_out)?;

        let inference_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(InferenceOutput {
            sem_pred,
            trajectory_priors,
            weights_trained: self.weights_trained,
            inference_ms,
        })
    }
}

// ── Trajectory prior extraction ───────────────────────────────────────────────

/// Extract one trajectory waypoint per predicted frame.
///
/// For each frame, finds the non-free voxel with the highest probability
/// (approximated by the centroid of all non-free voxels, weighted equally).
/// Returns an empty `Vec` when all frames are predicted as free space.
fn extract_trajectory_priors(
    sem_pred: &Tensor,
    cfg: &OccWorldConfig,
    f_out: usize,
) -> Result<Vec<TrajectoryWaypoint>, OccWorldError> {
    // sem_pred: (1, F_out, H, W, D) u8
    // Pull to CPU Vec for coordinate extraction — lightweight post-processing
    let data: Vec<u8> = sem_pred
        .flatten_all()
        .map_err(OccWorldError::Candle)?
        .to_vec1()
        .map_err(OccWorldError::Candle)?;

    let h = cfg.grid_h;
    let w = cfg.grid_w;
    let d = cfg.grid_d;
    let frame_stride = h * w * d;

    let mut waypoints = Vec::with_capacity(f_out);
    for fi in 0..f_out {
        let frame_slice = &data[fi * frame_stride..(fi + 1) * frame_stride];
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_z = 0.0f64;
        let mut count = 0usize;

        for (idx, &cls) in frame_slice.iter().enumerate() {
            if cls != cfg.free_class {
                let xi = idx / (w * d);
                let yi = (idx % (w * d)) / d;
                let zi = idx % d;
                sum_x += xi as f64;
                sum_y += yi as f64;
                sum_z += zi as f64;
                count += 1;
            }
        }

        if count > 0 {
            let n = count as f64;
            waypoints.push(TrajectoryWaypoint {
                frame: fi,
                grid_x: (sum_x / n) as f32,
                grid_y: (sum_y / n) as f32,
                grid_z: (sum_z / n) as f32,
                confidence: (count as f32) / (frame_stride as f32),
            });
        }
    }
    Ok(waypoints)
}

// ── Device selection ──────────────────────────────────────────────────────────

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::cuda_if_available(0) {
        return d;
    }
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OccWorldConfig;

    fn small_cfg() -> OccWorldConfig {
        OccWorldConfig {
            grid_h: 8,
            grid_w: 8,
            grid_d: 4,
            num_classes: 4,
            free_class: 3,
            base_channels: 8,
            z_channels: 8,
            codebook_size: 4,
            embed_dim: 8,
            num_frames: 2,
            token_h: 4,
            token_w: 4,
            num_heads: 2,
            num_layers: 1,
            ffn_hidden: 16,
        }
    }

    #[test]
    fn test_dummy_predict_shape() -> Result<(), OccWorldError> {
        let device = Device::Cpu;
        let cfg = small_cfg();
        let engine = OccWorldCandle::dummy(cfg.clone(), device.clone())?;

        // (1, 2, 8, 8, 4) — batch=1, 2 past frames (matches num_frames)
        let past = Tensor::zeros(
            (1, cfg.num_frames, cfg.grid_h, cfg.grid_w, cfg.grid_d),
            DType::U8,
            &device,
        )
        .map_err(OccWorldError::Candle)?;

        let out = engine.predict(&past)?;
        let dims = out.sem_pred.dims();
        assert_eq!(dims[0], 1, "batch dim");
        assert_eq!(dims[1], cfg.num_frames, "frame dim");
        assert_eq!(dims[2], cfg.grid_h, "H dim");
        assert_eq!(dims[3], cfg.grid_w, "W dim");
        assert_eq!(dims[4], cfg.grid_d, "D dim");

        Ok(())
    }

    // The centerpiece honesty/determinism tests (input-dependence, run-to-run
    // determinism, the `weights_trained` flag) live in
    // `tests/predict_honesty.rs` so they exercise only the public API and keep
    // this file under the 500-line limit.

    #[test]
    fn test_load_nonexistent_checkpoint() {
        let cfg = small_cfg();
        let result = OccWorldCandle::load(Path::new("/no/such/checkpoint.safetensors"), cfg);
        assert!(
            matches!(result, Err(OccWorldError::CheckpointNotFound(_))),
            "expected CheckpointNotFound, got {result:?}"
        );
    }
}

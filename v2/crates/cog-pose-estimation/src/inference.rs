//! Inference engine — loads `pose_v1.safetensors` (produced by the
//! Candle training run on `ruvultra`'s RTX 5080, see
//! `cog/artifacts/pose_v1.safetensors` + `docs/benchmarks/pose-estimation-cog.md`)
//! and runs the encoder + pose head on each CSI window.
//!
//! Architecture mirrors the training script exactly:
//!     Conv1d(56 -> 64,  k=3, dilation=1, padding=1)
//!     Conv1d(64 -> 128, k=3, dilation=2, padding=2)
//!     Conv1d(128 -> 128, k=3, dilation=4, padding=4)
//!     mean over time -> [128]
//!     Linear(128 -> 256) -> ReLU
//!     Linear(256 -> 34)  -> sigmoid -> reshape [17, 2]
//!
//! When the safetensors file is missing the engine falls back to a
//! centred-skeleton baseline with `confidence=0` so the cog still
//! satisfies the ADR-100 runtime contract and the dashboard surfaces
//! "no model yet" instead of dropping frames silently.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use std::path::Path;
use std::sync::Arc;

/// 56 subcarriers × 20 frames per CSI window — matches the format
/// produced by `scripts/align-ground-truth.js` after #641.
pub const INPUT_SUBCARRIERS: usize = 56;
pub const INPUT_TIMESTEPS: usize = 20;
pub const OUTPUT_KEYPOINTS: usize = 17;

#[derive(Debug, Clone)]
pub struct CsiWindow {
    pub data: Vec<f32>, // length INPUT_SUBCARRIERS * INPUT_TIMESTEPS
}

#[derive(Debug, Clone)]
pub struct PoseOutput {
    /// Flat `[OUTPUT_KEYPOINTS * 2]` keypoints in `[0, 1]` normalised
    /// image coords, ordered (x0, y0, x1, y1, …).
    pub keypoints: Vec<f32>,
    pub confidence: f32,
}

impl PoseOutput {
    pub fn is_finite(&self) -> bool {
        self.keypoints.iter().all(|v| v.is_finite()) && self.confidence.is_finite()
    }
}

/// Internal model — mirrors the training script's `PoseModel` exactly.
struct PoseNet {
    c1: Conv1d,
    c2: Conv1d,
    c3: Conv1d,
    fc1: Linear,
    fc2: Linear,
}

impl PoseNet {
    fn new(vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let enc = vb.pp("enc");
        let head = vb.pp("head");

        let c1 = candle_nn::conv1d(
            56,
            64,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            },
            enc.pp("c1"),
        )?;
        let c2 = candle_nn::conv1d(
            64,
            128,
            3,
            Conv1dConfig {
                padding: 2,
                stride: 1,
                dilation: 2,
                groups: 1,
                ..Default::default()
            },
            enc.pp("c2"),
        )?;
        let c3 = candle_nn::conv1d(
            128,
            128,
            3,
            Conv1dConfig {
                padding: 4,
                stride: 1,
                dilation: 4,
                groups: 1,
                ..Default::default()
            },
            enc.pp("c3"),
        )?;
        let fc1 = candle_nn::linear(128, 256, head.pp("fc1"))?;
        let fc2 = candle_nn::linear(256, 34, head.pp("fc2"))?;

        Ok(Self {
            c1,
            c2,
            c3,
            fc1,
            fc2,
        })
    }

    /// Forward pass: `[B, 56, 20]` -> `[B, 34]` in `[0, 1]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.c1.forward(x)?.relu()?;
        let h = self.c2.forward(&h)?.relu()?;
        let h = self.c3.forward(&h)?.relu()?;
        // Global average pool over time dim (last dim) -> [B, 128]
        let h = h.mean(2)?;
        let h = self.fc1.forward(&h)?.relu()?;
        let h = self.fc2.forward(&h)?;
        // sigmoid -> keep in [0, 1]
        candle_nn::ops::sigmoid(&h)
    }
}

pub struct InferenceEngine {
    inner: Option<Arc<LoadedModel>>,
    device: Device,
}

struct LoadedModel {
    net: PoseNet,
}

impl InferenceEngine {
    /// Create an engine. Tries to load weights from `cog/artifacts/pose_v1.safetensors`
    /// (relative to current dir or the cog install dir under
    /// `/var/lib/cognitum/apps/pose-estimation/`). Returns a usable
    /// engine either way — without weights, `infer` produces the
    /// stub output.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_weights(default_weights_path().as_deref())
    }

    /// Create an engine with a specific weights path (used by `--config`
    /// in `cog-pose-estimation run`). If `weights_path` is `None` or the
    /// file does not exist on disk, the engine falls back to the
    /// centred-skeleton stub and emits a `tracing::warn!` so the
    /// appliance log shows why no real keypoints are coming through.
    pub fn with_weights(weights_path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let device = pick_device();
        let inner = match weights_path {
            Some(p) if p.exists() => {
                // SAFETY: `from_mmaped_safetensors` mmaps the file for the
                // VarBuilder's lifetime. We don't modify the file while the
                // VarBuilder is alive, and the file is read-only on disk on
                // appliance installs.
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[p.to_path_buf()], DType::F32, &device)?
                };
                let net = PoseNet::new(vb)?;
                tracing::info!(
                    weights = %p.display(),
                    "loaded pose_v1.safetensors into candle backend"
                );
                Some(Arc::new(LoadedModel { net }))
            }
            Some(p) => {
                tracing::warn!(
                    weights = %p.display(),
                    "pose weights file not found; falling back to centred-skeleton stub (confidence=0)"
                );
                None
            }
            None => {
                tracing::warn!(
                    "no pose weights path configured and no default weights found on disk; \
                     falling back to centred-skeleton stub (confidence=0)"
                );
                None
            }
        };
        Ok(Self { inner, device })
    }

    /// Where the weights actually came from. Useful for the run.started event.
    pub fn backend(&self) -> &'static str {
        match (&self.inner, &self.device) {
            (Some(_), Device::Cuda(_)) => "candle-cuda",
            (Some(_), _) => "candle-cpu",
            (None, _) => "stub",
        }
    }

    pub fn infer(&self, window: &CsiWindow) -> Result<PoseOutput, Box<dyn std::error::Error>> {
        if window.data.len() != INPUT_SUBCARRIERS * INPUT_TIMESTEPS {
            return Err(format!(
                "expected {} input values, got {}",
                INPUT_SUBCARRIERS * INPUT_TIMESTEPS,
                window.data.len()
            )
            .into());
        }

        let Some(model) = &self.inner else {
            // Stub fallback — model not loaded.
            return Ok(PoseOutput {
                keypoints: vec![0.5f32; OUTPUT_KEYPOINTS * 2],
                confidence: 0.0,
            });
        };

        // Build [1, 56, 20] tensor from the flat row-major buffer.
        let t = Tensor::from_slice(
            &window.data,
            (1, INPUT_SUBCARRIERS, INPUT_TIMESTEPS),
            &self.device,
        )?;
        let out = model.net.forward(&t)?; // [1, 34]
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        // Confidence from pose_v1 is a published constant rather than per-frame —
        // the trained model didn't emit a confidence head. Use the validation-set
        // PCK@50 (18.5%) as the published self-reported confidence so downstream
        // consumers can gate display decisions on it.
        Ok(PoseOutput {
            keypoints: flat,
            confidence: 0.185,
        })
    }
}

/// Synthetic CSI window for the `health` subcommand. Zeros — exercises
/// the I/O surface; the model never touches values that produce NaN.
pub struct SyntheticInput;

impl Default for SyntheticInput {
    fn default() -> Self {
        Self
    }
}

impl SyntheticInput {
    pub fn as_window(&self) -> CsiWindow {
        CsiWindow {
            data: vec![0.0; INPUT_SUBCARRIERS * INPUT_TIMESTEPS],
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::cuda_if_available(0) {
        return d;
    }
    Device::Cpu
}

fn default_weights_path() -> Option<std::path::PathBuf> {
    // Search in the order an installed Cog would see it.
    let candidates = [
        std::path::PathBuf::from("/var/lib/cognitum/apps/pose-estimation/pose_v1.safetensors"),
        std::path::PathBuf::from("./pose_v1.safetensors"),
        std::path::PathBuf::from("./cog/artifacts/pose_v1.safetensors"),
        // From the repo root.
        std::path::PathBuf::from("v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors"),
        // From inside v2/.
        std::path::PathBuf::from("crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

// ---------------------------------------------------------------------------
// Unit tests — exercise the safetensors → forward-pass path. Integration-level
// assertions (CLI surface, manifest round-trip, etc.) live in `tests/smoke.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Locate `pose_v1.safetensors` from any of the cwds a `cargo test`
    /// invocation might land in (workspace root, `v2/`, or the crate dir).
    fn locate_weights() -> Option<std::path::PathBuf> {
        let candidates = [
            std::path::PathBuf::from("cog/artifacts/pose_v1.safetensors"),
            std::path::PathBuf::from("crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors"),
            std::path::PathBuf::from(
                "v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors",
            ),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    #[test]
    fn stub_fallback_when_weights_missing() {
        // `with_weights(None)` must never panic and must produce a
        // finite, well-shaped output so the runtime loop keeps making
        // progress while the operator notices the warn log.
        let engine = InferenceEngine::with_weights(None).expect("engine init");
        assert_eq!(engine.backend(), "stub");
        let out = engine.infer(&SyntheticInput.as_window()).expect("infer");
        assert!(out.is_finite());
        assert_eq!(out.keypoints.len(), OUTPUT_KEYPOINTS * 2);
        assert_eq!(out.confidence, 0.0);
    }

    #[test]
    fn weights_load_and_forward_produces_seventeen_keypoint_pairs() {
        let Some(weights) = locate_weights() else {
            eprintln!(
                "(skipping — pose_v1.safetensors not on disk; run from the cog crate or repo root)"
            );
            return;
        };
        let engine = InferenceEngine::with_weights(Some(&weights)).expect("load real weights");
        assert!(
            engine.backend().starts_with("candle-"),
            "expected candle backend, got {}",
            engine.backend()
        );

        // Synthetic [56, 20] zero-input window — the documented "no-op"
        // test signal. Anything finite and well-shaped proves the
        // safetensors weights flowed through the forward pass.
        let out = engine.infer(&SyntheticInput.as_window()).expect("infer");

        // Shape: 17 (x, y) pairs = 34 scalars, no NaN, no Inf, all in
        // sigmoid's [0, 1] range.
        assert_eq!(
            out.keypoints.len(),
            OUTPUT_KEYPOINTS * 2,
            "expected {} scalars for 17 keypoint pairs",
            OUTPUT_KEYPOINTS * 2
        );
        let pairs: Vec<[f32; 2]> = out
            .keypoints
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        assert_eq!(pairs.len(), OUTPUT_KEYPOINTS, "expected 17 (x, y) pairs");
        for (i, [x, y]) in pairs.iter().enumerate() {
            assert!(
                x.is_finite() && y.is_finite(),
                "keypoint {i} not finite: ({x}, {y})"
            );
            assert!(!x.is_nan() && !y.is_nan(), "keypoint {i} is NaN");
            assert!(
                (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y),
                "keypoint {i} out of [0,1]: ({x}, {y})"
            );
        }

        // Confidence is the published PCK@50 (constant for v0.0.1), so
        // anything > 0 proves we didn't silently fall through to the stub.
        assert!(out.confidence > 0.0);
        assert!(out.confidence.is_finite());
    }

    #[test]
    fn rejects_wrong_shape_input_before_any_forward_pass() {
        let engine = InferenceEngine::with_weights(None).expect("engine init");
        let bad = CsiWindow { data: vec![0.0; 7] };
        assert!(engine.infer(&bad).is_err());
    }
}

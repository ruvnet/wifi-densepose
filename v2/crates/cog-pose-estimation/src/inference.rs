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

/// The model's typical self-reported confidence. `pose_v1` has **no confidence
/// head** (the head emits 34 keypoint coordinates only), so per-frame confidence
/// is not available from the network. This is the validation-set PCK@50 (18.5%)
/// the training run reported, used as the published per-frame confidence floor.
///
/// Surfaced as a public constant so the runtime can warn when a configured
/// `min_confidence` threshold exceeds it — otherwise a default install would
/// silently emit zero `pose.frame` events while health reports healthy.
pub const MODEL_TYPICAL_CONFIDENCE: f32 = 0.185;

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

/// Per-room LoRA calibration adapter (ADR-150 §3.5–3.6). Low-rank deltas on the pose
/// head: `delta = (x · A) · B`, with `A:[in,r]`, `B:[r,out]` (scale baked into `B` at
/// save time). A handful of labeled in-room samples fit this ~few-KB adapter and recover
/// SOTA-level pose for an unseen room/person, on top of the frozen shared base.
/// Adapter safetensors keys: `fc1.a`, `fc1.b`, `fc2.a`, `fc2.b` (any subset).
#[derive(Clone)]
struct PoseLora {
    fc1: Option<(Tensor, Tensor)>,
    fc2: Option<(Tensor, Tensor)>,
}

impl PoseLora {
    /// Load from an adapter safetensors. Missing layer keys are simply skipped.
    fn load(path: &Path, device: &Device) -> candle_core::Result<Self> {
        let t = candle_core::safetensors::load(path, device)?;
        let pair = |a: &str, b: &str| match (t.get(a), t.get(b)) {
            (Some(x), Some(y)) => Some((x.clone(), y.clone())),
            _ => None,
        };
        Ok(Self {
            fc1: pair("fc1.a", "fc1.b"),
            fc2: pair("fc2.a", "fc2.b"),
        })
    }

    /// `y + (x · A) · B` when an adapter for this layer is present, else `y` unchanged.
    fn apply(slot: &Option<(Tensor, Tensor)>, x: &Tensor, y: Tensor) -> candle_core::Result<Tensor> {
        match slot {
            Some((a, b)) => y + x.matmul(a)?.matmul(b)?,
            None => Ok(y),
        }
    }
}

/// Internal model — mirrors the training script's `PoseModel` exactly.
struct PoseNet {
    c1: Conv1d,
    c2: Conv1d,
    c3: Conv1d,
    fc1: Linear,
    fc2: Linear,
    /// Optional per-room calibration adapter (none = shared base behaviour).
    adapter: Option<PoseLora>,
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
            adapter: None,
        })
    }

    /// Forward pass: `[B, 56, 20]` -> `[B, 34]` in `[0, 1]`. Applies the per-room
    /// LoRA calibration adapter on the head layers when one is attached.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.c1.forward(x)?.relu()?;
        let h = self.c2.forward(&h)?.relu()?;
        let h = self.c3.forward(&h)?.relu()?;
        // Global average pool over time dim (last dim) -> [B, 128]
        let pooled = h.mean(2)?;
        // fc1 (+ adapter delta) -> ReLU
        let mut h1 = self.fc1.forward(&pooled)?;
        if let Some(ad) = &self.adapter {
            h1 = PoseLora::apply(&ad.fc1, &pooled, h1)?;
        }
        let h1 = h1.relu()?;
        // fc2 (+ adapter delta)
        let mut h2 = self.fc2.forward(&h1)?;
        if let Some(ad) = &self.adapter {
            h2 = PoseLora::apply(&ad.fc2, &h1, h2)?;
        }
        // sigmoid -> keep in [0, 1]
        candle_nn::ops::sigmoid(&h2)
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

    /// Engine from the default base weights plus an optional per-room calibration
    /// adapter (ADR-150 §3.5). Used by `cog-pose-estimation run --adapter <path>`.
    pub fn with_adapter(adapter_path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_weights_and_adapter(default_weights_path().as_deref(), adapter_path)
    }

    /// Create an engine with a specific weights path (used by `--config`
    /// in `cog-pose-estimation run`). If `weights_path` is `None`, the
    /// stub fallback is used.
    pub fn with_weights(weights_path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_weights_and_adapter(weights_path, None)
    }

    /// Create an engine with a shared base **and an optional per-room calibration
    /// adapter** (ADR-150 §3.5). The adapter is a tiny LoRA **safetensors with keys
    /// `fc1.a`/`fc1.b`/`fc2.a`/`fc2.b`** — low-rank deltas for *this* engine's conv+MLP
    /// pose head, fitted from a short labeled in-room capture. (It applies the same LoRA
    /// calibration *mechanism* demonstrated by the reference tool in
    /// `aether-arena/calibration/`, but that reference targets the MM-Fi transformer model
    /// and emits a different key layout — adapters are model-specific and not interchangeable.)
    /// `None` = uncalibrated base.
    pub fn with_weights_and_adapter(
        weights_path: Option<&Path>,
        adapter_path: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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
                let mut net = PoseNet::new(vb)?;
                if let Some(ap) = adapter_path {
                    if ap.exists() {
                        net.adapter = Some(PoseLora::load(ap, &device)?);
                    }
                }
                Some(Arc::new(LoadedModel { net }))
            }
            _ => None,
        };
        Ok(Self { inner, device })
    }

    /// Whether a per-room calibration adapter is currently attached.
    pub fn is_calibrated(&self) -> bool {
        self.inner
            .as_ref()
            .map(|m| m.net.adapter.is_some())
            .unwrap_or(false)
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
        // the trained model has no confidence head (the head emits 34 keypoint
        // coordinates only), so a real per-frame value is genuinely unavailable.
        // We surface the validation-set PCK@50 (`MODEL_TYPICAL_CONFIDENCE`) as the
        // honest self-reported confidence. The runtime's `min_confidence` default
        // is pinned at or below this so a default install actually emits frames
        // (and warns if an operator raises the threshold above the model's reach).
        Ok(PoseOutput {
            keypoints: flat,
            confidence: MODEL_TYPICAL_CONFIDENCE,
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

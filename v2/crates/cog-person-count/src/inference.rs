//! Single-node count inference — Candle forward over a CSI window.
//!
//! Architecture (matches ADR-103 §"Architecture (v0.1.0)"):
//!     Conv1d(56 -> 64,   k=3, dilation=1, padding=1)
//!     Conv1d(64 -> 128,  k=3, dilation=2, padding=2)
//!     Conv1d(128 -> 128, k=3, dilation=4, padding=4)
//!     mean over time -> [128]                ← shared encoder
//!     ├── Linear(128 -> 64) -> ReLU -> Linear(64 -> 8)  → softmax over {0..7}
//!     └── Linear(128 -> 32) -> ReLU -> Linear(32 -> 1)  → sigmoid → confidence
//!
//! When the safetensors file is missing the engine falls back to a
//! "single-person, zero-confidence" stub so the cog still satisfies the
//! ADR-100 runtime contract and the dashboard surfaces "no model yet"
//! instead of dropping frames silently.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use std::path::Path;
use std::sync::Arc;

/// `[56 subcarriers × 20 frames]` window — same shape as cog-pose-estimation.
pub const INPUT_SUBCARRIERS: usize = 56;
pub const INPUT_TIMESTEPS: usize = 20;
/// Count classification over {0, 1, ..., 7} persons.
pub const COUNT_CLASSES: usize = 8;

#[derive(Debug, Clone)]
pub struct CsiWindow {
    pub data: Vec<f32>,
}

/// Per-node prediction emitted by the count head + confidence head.
#[derive(Debug, Clone)]
pub struct CountPrediction {
    /// Categorical distribution over {0..7} persons. Sums to 1 within float
    /// precision. Maximum-likelihood class is `argmax(probs)`.
    pub probs: [f32; COUNT_CLASSES],
    /// `[0, 1]` — confidence head output. Calibrated against (predicted == truth)
    /// during training so consumers can use it as a probability of being right.
    pub confidence: f32,
}

impl CountPrediction {
    pub fn is_finite(&self) -> bool {
        self.probs.iter().all(|v| v.is_finite()) && self.confidence.is_finite()
    }

    /// Maximum-likelihood class.
    pub fn argmax(&self) -> usize {
        let mut best_i = 0;
        let mut best_v = self.probs[0];
        for (i, &v) in self.probs.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i
    }

    /// `(low, high)` such that `Σ probs[low..=high] ≥ 0.95`. Used for the
    /// `count_p95_low` / `count_p95_high` fields surfaced to consumers.
    pub fn p95_range(&self) -> (usize, usize) {
        let mode = self.argmax();
        let mut lo = mode;
        let mut hi = mode;
        let mut acc = self.probs[mode];
        while acc < 0.95 && (lo > 0 || hi < COUNT_CLASSES - 1) {
            let left = if lo > 0 { self.probs[lo - 1] } else { -1.0 };
            let right = if hi < COUNT_CLASSES - 1 {
                self.probs[hi + 1]
            } else {
                -1.0
            };
            if left >= right && lo > 0 {
                lo -= 1;
                acc += self.probs[lo];
            } else if hi < COUNT_CLASSES - 1 {
                hi += 1;
                acc += self.probs[hi];
            } else if lo > 0 {
                lo -= 1;
                acc += self.probs[lo];
            } else {
                break;
            }
        }
        (lo, hi)
    }
}

struct CountNet {
    c1: Conv1d,
    c2: Conv1d,
    c3: Conv1d,
    count_fc1: Linear,
    count_fc2: Linear,
    conf_fc1: Linear,
    conf_fc2: Linear,
}

impl CountNet {
    fn new(vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let enc = vb.pp("enc");
        let count = vb.pp("count_head");
        let conf = vb.pp("conf_head");

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
        let count_fc1 = candle_nn::linear(128, 64, count.pp("fc1"))?;
        let count_fc2 = candle_nn::linear(64, COUNT_CLASSES, count.pp("fc2"))?;
        let conf_fc1 = candle_nn::linear(128, 32, conf.pp("fc1"))?;
        let conf_fc2 = candle_nn::linear(32, 1, conf.pp("fc2"))?;
        Ok(Self {
            c1,
            c2,
            c3,
            count_fc1,
            count_fc2,
            conf_fc1,
            conf_fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let h = self.c1.forward(x)?.relu()?;
        let h = self.c2.forward(&h)?.relu()?;
        let h = self.c3.forward(&h)?.relu()?;
        let h = h.mean(2)?; // [B, 128]

        // Count head — logits then softmax
        let c = self.count_fc1.forward(&h)?.relu()?;
        let c = self.count_fc2.forward(&c)?;
        let probs = candle_nn::ops::softmax(&c, candle_core::D::Minus1)?;

        // Confidence head — sigmoid
        let cf = self.conf_fc1.forward(&h)?.relu()?;
        let cf = self.conf_fc2.forward(&cf)?;
        let conf = candle_nn::ops::sigmoid(&cf)?;

        Ok((probs, conf))
    }
}

pub struct InferenceEngine {
    inner: Option<Arc<CountNet>>,
    device: Device,
}

impl InferenceEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_weights(default_weights_path().as_deref())
    }

    pub fn with_weights(weights_path: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let device = pick_device();
        let inner = match weights_path {
            Some(p) if p.exists() => {
                // SAFETY: from_mmaped_safetensors mmaps the file for the
                // VarBuilder's lifetime. Same pattern as cog-pose-estimation.
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&[p.to_path_buf()], DType::F32, &device)?
                };
                let net = CountNet::new(vb)?;
                Some(Arc::new(net))
            }
            _ => None,
        };
        Ok(Self { inner, device })
    }

    pub fn backend(&self) -> &'static str {
        match (&self.inner, &self.device) {
            (Some(_), Device::Cuda(_)) => "candle-cuda",
            (Some(_), _) => "candle-cpu",
            (None, _) => "stub",
        }
    }

    pub fn infer(&self, window: &CsiWindow) -> Result<CountPrediction, Box<dyn std::error::Error>> {
        if window.data.len() != INPUT_SUBCARRIERS * INPUT_TIMESTEPS {
            return Err(format!(
                "expected {} input values, got {}",
                INPUT_SUBCARRIERS * INPUT_TIMESTEPS,
                window.data.len()
            )
            .into());
        }

        let Some(net) = &self.inner else {
            // Stub fallback: single-person, zero confidence. Surfaces "no
            // model yet" honestly instead of pretending to know.
            let mut probs = [0.0f32; COUNT_CLASSES];
            probs[1] = 1.0; // mass on "1 person"
            return Ok(CountPrediction {
                probs,
                confidence: 0.0,
            });
        };

        let t = Tensor::from_slice(
            &window.data,
            (1, INPUT_SUBCARRIERS, INPUT_TIMESTEPS),
            &self.device,
        )?;
        let (probs_t, conf_t) = net.forward(&t)?;
        let flat: Vec<f32> = probs_t.flatten_all()?.to_vec1()?;
        if flat.len() != COUNT_CLASSES {
            return Err(format!(
                "count head produced {} probs, expected {}",
                flat.len(),
                COUNT_CLASSES
            )
            .into());
        }
        let mut probs = [0.0f32; COUNT_CLASSES];
        probs.copy_from_slice(&flat[..COUNT_CLASSES]);
        let conf = conf_t.flatten_all()?.to_vec1::<f32>()?[0];

        Ok(CountPrediction {
            probs,
            confidence: conf,
        })
    }
}

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

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::cuda_if_available(0) {
        return d;
    }
    Device::Cpu
}

fn default_weights_path() -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from("/var/lib/cognitum/apps/person-count/count_v1.safetensors"),
        std::path::PathBuf::from("./count_v1.safetensors"),
        std::path::PathBuf::from("./cog/artifacts/count_v1.safetensors"),
        std::path::PathBuf::from("v2/crates/cog-person-count/cog/artifacts/count_v1.safetensors"),
        std::path::PathBuf::from("crates/cog-person-count/cog/artifacts/count_v1.safetensors"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

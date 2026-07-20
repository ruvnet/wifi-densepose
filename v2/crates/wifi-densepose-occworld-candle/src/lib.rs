//! `wifi-densepose-occworld-candle` — OccWorld TransVQVAE inference in Candle.
//!
//! Ports the 72.4 M-parameter OccWorld world model (VQVAE tokeniser +
//! autoregressive transformer) from Python to native Rust using the
//! Hugging Face Candle framework.  The goal is to eliminate the
//! 208 ms Python/IPC overhead of the existing `wifi-densepose-worldmodel`
//! bridge and enable tight integration with the streaming engine.
//!
//! ## Module structure
//!
//! | Module          | Contents                                              |
//! |-----------------|-------------------------------------------------------|
//! | `config`        | `OccWorldConfig` — hyper-parameters                  |
//! | `error`         | `OccWorldError` — unified error enum                 |
//! | `cnn`           | Real conv `Encoder2D` / `Decoder2D` (deterministic)   |
//! | `vqvae`         | Class embedding, VQ codebook, quant convolutions      |
//! | `transformer`   | Autoregressive transformer (`PlanUAutoRegTransformer`) |
//! | `model`         | SafeTensors weight loading + key mapping              |
//! | `inference`     | `OccWorldCandle` end-to-end inference engine          |
//!
//! ## Implementation status
//!
//! The VQVAE encoder/decoder are a **real, deterministic, input-dependent**
//! convolutional forward pass (`crate::cnn`) — no `randn` anywhere in the
//! prediction path. All other components (class embedding, VQ codebook,
//! quant/post-quant convolutions, transformer, trajectory extraction) are
//! fully implemented. What remains **data-gated** is a *trained* checkpoint:
//! with `OccWorldCandle::dummy` the weights are deterministically initialised
//! but untrained, so the model is honest-but-unaccurate. This is surfaced via
//! [`InferenceOutput::weights_trained`] (`false` until `load` reads a real
//! checkpoint) — consumers must never treat untrained priors as trained.
//!
//! ## Usage
//!
//! ```no_run
//! use wifi_densepose_occworld_candle::inference::OccWorldCandle;
//! use wifi_densepose_occworld_candle::config::OccWorldConfig;
//! use candle_core::{Device, DType, Tensor};
//! use std::path::Path;
//!
//! let cfg = OccWorldConfig::default();
//! let engine = OccWorldCandle::dummy(cfg, Device::Cpu).expect("dummy init");
//! let past = Tensor::zeros((1, 15, 200, 200, 16), DType::U8, &Device::Cpu).unwrap();
//! let out = engine.predict(&past).expect("predict");
//! println!("predicted {} frames in {:.1} ms", out.sem_pred.dim(1).unwrap(), out.inference_ms);
//! ```

pub mod cnn;
pub mod config;
pub mod error;
pub mod inference;
pub mod model;
pub mod transformer;
pub mod vqvae;

pub use config::OccWorldConfig;
pub use error::OccWorldError;
pub use inference::{InferenceOutput, OccWorldCandle, TrajectoryWaypoint};

//! Input-validation boundary tests for `OccWorldCandle::predict`.
//!
//! Security review (Milestone #9, crate 4/4). `predict` takes an
//! externally-supplied occupancy tensor; per the project's "validate input at
//! system boundaries" rule it must reject degenerate / out-of-capacity shapes
//! with a clear domain error rather than surfacing a cryptic deep-pipeline
//! Candle error (over-capacity frame counts over-index the temporal positional
//! embedding) or processing a zero-element tensor.
//!
//! These exercise only the public API and live here (not inline in
//! `inference.rs`) to keep that module under the 500-line cap.

use candle_core::{DType, Device, Tensor};
use wifi_densepose_occworld_candle::config::OccWorldConfig;
use wifi_densepose_occworld_candle::inference::OccWorldCandle;
use wifi_densepose_occworld_candle::error::OccWorldError;

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

/// Zero frames is a degenerate input that would otherwise feed a zero-element
/// tensor into the reshape/conv pipeline. Must be rejected at the boundary.
#[test]
fn predict_rejects_zero_frames() {
    let device = Device::Cpu;
    let cfg = small_cfg();
    let engine = OccWorldCandle::dummy(cfg.clone(), device.clone()).unwrap();
    let past = Tensor::zeros(
        (1usize, 0usize, cfg.grid_h, cfg.grid_w, cfg.grid_d),
        DType::U8,
        &device,
    )
    .unwrap();
    let result = engine.predict(&past);
    assert!(
        matches!(result, Err(OccWorldError::ShapeMismatch(_))),
        "zero-frame input must be rejected with ShapeMismatch"
    );
}

/// Zero batch must also be rejected (same zero-element-tensor hazard).
#[test]
fn predict_rejects_zero_batch() {
    let device = Device::Cpu;
    let cfg = small_cfg();
    let engine = OccWorldCandle::dummy(cfg.clone(), device.clone()).unwrap();
    let past = Tensor::zeros(
        (0usize, cfg.num_frames, cfg.grid_h, cfg.grid_w, cfg.grid_d),
        DType::U8,
        &device,
    )
    .unwrap();
    let result = engine.predict(&past);
    assert!(
        matches!(result, Err(OccWorldError::ShapeMismatch(_))),
        "zero-batch input must be rejected with ShapeMismatch"
    );
}

/// More frames than the temporal embedding can index (`> num_frames*2`).
///
/// On the old code this over-indexed the temporal positional embedding deep in
/// the transformer and surfaced as a cryptic Candle "gather" `InvalidIndex`
/// error. The boundary guard now rejects it cleanly with `ShapeMismatch`.
#[test]
fn predict_rejects_too_many_frames() {
    let device = Device::Cpu;
    let cfg = small_cfg(); // num_frames = 2 → temporal capacity = 4
    let engine = OccWorldCandle::dummy(cfg.clone(), device.clone()).unwrap();
    let too_many = cfg.num_frames * 2 + 1;
    let past = Tensor::zeros(
        (1usize, too_many, cfg.grid_h, cfg.grid_w, cfg.grid_d),
        DType::U8,
        &device,
    )
    .unwrap();
    let result = engine.predict(&past);
    assert!(
        matches!(result, Err(OccWorldError::ShapeMismatch(_))),
        "over-capacity frame count must be rejected with ShapeMismatch"
    );
}

/// A frame count exactly at capacity (`num_frames*2`) must still succeed —
/// the guard rejects only *over*-capacity, not the boundary value.
#[test]
fn predict_accepts_frame_count_at_capacity() {
    let device = Device::Cpu;
    let cfg = small_cfg();
    let engine = OccWorldCandle::dummy(cfg.clone(), device.clone()).unwrap();
    let at_cap = cfg.num_frames * 2;
    let past = Tensor::zeros(
        (1usize, at_cap, cfg.grid_h, cfg.grid_w, cfg.grid_d),
        DType::U8,
        &device,
    )
    .unwrap();
    let out = engine
        .predict(&past)
        .expect("at-capacity frame count must predict");
    assert_eq!(out.sem_pred.dims()[1], at_cap, "frame dim preserved");
}

/// Wrong spatial geometry (H/W/D) is still rejected — pins the pre-existing
/// guard alongside the new frame/batch ones.
#[test]
fn predict_rejects_wrong_grid_dims() {
    let device = Device::Cpu;
    let cfg = small_cfg();
    let engine = OccWorldCandle::dummy(cfg.clone(), device.clone()).unwrap();
    let past = Tensor::zeros(
        (1usize, cfg.num_frames, cfg.grid_h + 1, cfg.grid_w, cfg.grid_d),
        DType::U8,
        &device,
    )
    .unwrap();
    let result = engine.predict(&past);
    assert!(
        matches!(result, Err(OccWorldError::ShapeMismatch(_))),
        "wrong grid dims must be rejected with ShapeMismatch"
    );
}

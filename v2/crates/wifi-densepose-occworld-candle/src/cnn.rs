//! Real convolutional encoder / decoder for the OccWorld VQVAE.
//!
//! This module replaces the former `Tensor::randn` stubs in [`crate::vqvae`]
//! with a genuine, **deterministic, input-dependent** forward pass:
//!
//! * [`Encoder2D`] — a 3-stage convolutional encoder (`Conv2d` + GELU) that
//!   maps the class-embedded occupancy grid
//!   `(B*F, base_channels, H, W*D)` to a latent feature map
//!   `(B*F, z_channels, token_h, token_w)`.  The final spatial resolution is
//!   pinned with `interpolate2d` (adaptive average pooling) so the encoder
//!   works for *any* grid/token geometry, not just power-of-two factors.
//! * [`Decoder2D`] — the mirror network (`upsample_nearest2d` + `Conv2d`)
//!   mapping latent codes `(B*F, z_channels, token_h, token_w)` back to
//!   per-voxel class logits `(B*F, num_classes, H, W, D)`.
//!
//! ## Honesty / determinism contract
//!
//! * **No randomness in the forward path.** Given identical weights and an
//!   identical input tensor, both networks produce bit-identical output.
//! * **Input-dependent.** Two different inputs produce different outputs
//!   (the convolutions are linear maps of the input plus a bias; only an
//!   all-zero weight tensor would break this — and we never zero the weights).
//! * **Deterministic initialisation.** The `dummy` / untrained constructors
//!   use a fixed-seed pseudo-random fill ([`det_fill`]) so test runs are
//!   reproducible across machines. Untrained weights are an honest,
//!   *data-gated* deliverable — see `weights_trained` in
//!   [`crate::inference::InferenceOutput`].
//!
//! When a real Phase-5 checkpoint exists, [`Encoder2D::from_weights`] /
//! [`Decoder2D::from_weights`] load the trained tensors via a
//! [`candle_nn::VarBuilder`]; nothing else in the forward path changes.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::config::OccWorldConfig;

/// Deterministic, seed-driven weight fill in `[-scale, scale)`.
///
/// A tiny xorshift64* PRNG generates the values, so the result is identical
/// on every platform for a given `(shape, seed)` — unlike `Tensor::randn`,
/// which draws from the global RNG and is therefore non-reproducible and
/// (crucially) decouples the output from the input. We *only* use this to
/// initialise weights, never inside `forward`.
///
/// Exposed `pub(crate)` so the VQVAE/transformer `dummy` constructors share the
/// same deterministic initialisation, making two independently-built untrained
/// engines bit-for-bit identical (and therefore reproducible in tests).
pub(crate) fn det_fill(shape: &[usize], seed: u64, scale: f32, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut state = seed | 1; // never zero
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // map high 24 bits → [0, 1) → [-scale, scale)
        let unit = ((r >> 40) as f32) / (1u32 << 24) as f32;
        data.push((unit * 2.0 - 1.0) * scale);
    }
    Tensor::from_vec(data, shape, device)
}

/// Build a `Conv2d` with deterministic weights (Kaiming-ish fan-in scaling).
fn det_conv2d(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    cfg: Conv2dConfig,
    seed: u64,
    device: &Device,
) -> Result<Conv2d> {
    let fan_in = (in_c * kernel * kernel) as f32;
    let scale = (1.0 / fan_in).sqrt();
    let w = det_fill(&[out_c, in_c, kernel, kernel], seed, scale, device)?;
    // Small non-zero deterministic bias so even all-zero inputs differ per channel.
    let b = det_fill(&[out_c], seed.wrapping_add(0x9E37_79B9_7F4A_7C15), scale, device)?;
    Ok(Conv2d::new(w, Some(b), cfg))
}

// ── Encoder ───────────────────────────────────────────────────────────────────

/// Real 2-D convolutional encoder: `(B*F, base_channels, H, W*D)` →
/// `(B*F, z_channels, token_h, token_w)`.
///
/// Three `Conv2d` stages (stride-2, stride-2, stride-1) with GELU
/// non-linearities progressively expand channels and contract resolution;
/// a final `interpolate2d` pins the output to the exact token grid so the
/// network is geometry-agnostic.
pub struct Encoder2D {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    token_h: usize,
    token_w: usize,
}

impl Encoder2D {
    fn channels(cfg: &OccWorldConfig) -> (usize, usize, usize) {
        let mid = cfg.z_channels.max(cfg.base_channels);
        (cfg.base_channels, mid, cfg.z_channels)
    }

    /// Deterministic untrained encoder (fixed-seed weights).
    pub fn dummy(cfg: &OccWorldConfig, device: &Device) -> Result<Self> {
        let (c_in, c_mid, c_out) = Self::channels(cfg);
        let down = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let keep = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        Ok(Self {
            conv1: det_conv2d(c_in, c_mid, 3, down, 0x0CCD_0001, device)?,
            conv2: det_conv2d(c_mid, c_mid, 3, down, 0x0CCD_0002, device)?,
            conv3: det_conv2d(c_mid, c_out, 3, keep, 0x0CCD_0003, device)?,
            token_h: cfg.token_h,
            token_w: cfg.token_w,
        })
    }

    /// Load trained encoder weights from a checkpoint.
    pub fn from_weights(cfg: &OccWorldConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let (c_in, c_mid, c_out) = Self::channels(cfg);
        let down = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let keep = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let vb = vb.pp("enc");
        Ok(Self {
            conv1: candle_nn::conv2d(c_in, c_mid, 3, down, vb.pp("conv1"))?,
            conv2: candle_nn::conv2d(c_mid, c_mid, 3, down, vb.pp("conv2"))?,
            conv3: candle_nn::conv2d(c_mid, c_out, 3, keep, vb.pp("conv3"))?,
            token_h: cfg.token_h,
            token_w: cfg.token_w,
        })
    }

    /// Forward: `(B*F, base_channels, H, W*D)` → `(B*F, z_channels, token_h, token_w)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = self.conv3.forward(&x)?.gelu()?;
        // Pin to the exact token grid (adaptive average pooling).
        x.interpolate2d(self.token_h, self.token_w)
    }
}

// ── Decoder ───────────────────────────────────────────────────────────────────

/// Real 2-D convolutional decoder: `(B*F, z_channels, token_h, token_w)` →
/// per-voxel class logits `(B*F, num_classes, grid_h, grid_w, grid_d)`.
///
/// The latent map is up-sampled to the folded `(grid_h, grid_w*grid_d)`
/// resolution, refined by two `Conv2d` layers, and projected to
/// `num_classes` channels by a 1×1 head before being unfolded back to 3-D.
pub struct Decoder2D {
    up1: Conv2d,
    up2: Conv2d,
    head: Conv2d,
    grid_h: usize,
    grid_w: usize,
    grid_d: usize,
    num_classes: usize,
}

impl Decoder2D {
    fn channels(cfg: &OccWorldConfig) -> (usize, usize) {
        let mid = cfg.z_channels.max(cfg.base_channels);
        (cfg.z_channels, mid)
    }

    /// Deterministic untrained decoder (fixed-seed weights).
    pub fn dummy(cfg: &OccWorldConfig, device: &Device) -> Result<Self> {
        let (c_in, c_mid) = Self::channels(cfg);
        let keep = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let head = Conv2dConfig::default(); // 1×1, padding 0
        Ok(Self {
            up1: det_conv2d(c_in, c_mid, 3, keep, 0x0DEC_0001, device)?,
            up2: det_conv2d(c_mid, c_mid, 3, keep, 0x0DEC_0002, device)?,
            head: det_conv2d(c_mid, cfg.num_classes, 1, head, 0x0DEC_0003, device)?,
            grid_h: cfg.grid_h,
            grid_w: cfg.grid_w,
            grid_d: cfg.grid_d,
            num_classes: cfg.num_classes,
        })
    }

    /// Load trained decoder weights from a checkpoint.
    pub fn from_weights(cfg: &OccWorldConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let (c_in, c_mid) = Self::channels(cfg);
        let keep = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let head = Conv2dConfig::default();
        let vb = vb.pp("dec");
        Ok(Self {
            up1: candle_nn::conv2d(c_in, c_mid, 3, keep, vb.pp("up1"))?,
            up2: candle_nn::conv2d(c_mid, c_mid, 3, keep, vb.pp("up2"))?,
            head: candle_nn::conv2d(c_mid, cfg.num_classes, 1, head, vb.pp("head"))?,
            grid_h: cfg.grid_h,
            grid_w: cfg.grid_w,
            grid_d: cfg.grid_d,
            num_classes: cfg.num_classes,
        })
    }

    /// Forward: `(B*F, z_channels, token_h, token_w)` →
    /// `(B*F, num_classes, grid_h, grid_w, grid_d)`.
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let bf = z.dim(0)?;
        // Up-sample latent map to the folded occupancy resolution (H, W*D).
        let target_w = self.grid_w * self.grid_d;
        let x = z.upsample_nearest2d(self.grid_h, target_w)?;
        let x = self.up1.forward(&x)?.gelu()?;
        let x = self.up2.forward(&x)?.gelu()?;
        // 1×1 head → (B*F, num_classes, H, W*D)
        let logits2d = self.head.forward(&x)?;
        // Unfold width back into (W, D): (B*F, num_classes, H, W, D)
        logits2d.reshape((bf, self.num_classes, self.grid_h, self.grid_w, self.grid_d))
    }
}

// ── Free-function wrappers (drop-in replacements for the old stubs) ─────────────

/// Real encoder forward, dispatched through an [`Encoder2D`].
///
/// Accepts the class-embedded grid `(B*F, base_channels, H, W*D)` and returns
/// `(B*F, z_channels, token_h, token_w)`. Deterministic and input-dependent.
pub fn encode_occupancy(encoder: &Encoder2D, x: &Tensor) -> Result<Tensor> {
    encoder.forward(x)
}

/// Real decoder forward, dispatched through a [`Decoder2D`].
pub fn decode_to_logits(decoder: &Decoder2D, z: &Tensor) -> Result<Tensor> {
    decoder.forward(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn cfg() -> OccWorldConfig {
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
    fn det_fill_is_reproducible() -> Result<()> {
        let dev = Device::Cpu;
        let a = det_fill(&[3, 4], 42, 1.0, &dev)?;
        let b = det_fill(&[3, 4], 42, 1.0, &dev)?;
        let diff = (a - b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "same seed must give identical fill");
        Ok(())
    }

    #[test]
    fn encoder_shape_and_determinism() -> Result<()> {
        let dev = Device::Cpu;
        let c = cfg();
        let enc = Encoder2D::dummy(&c, &dev)?;
        let x = Tensor::randn(
            0f32,
            1.0,
            (2, c.base_channels, c.grid_h, c.grid_w * c.grid_d),
            &dev,
        )?;
        let z1 = enc.forward(&x)?;
        let z2 = enc.forward(&x)?;
        assert_eq!(z1.dims(), &[2, c.z_channels, c.token_h, c.token_w]);
        // Same input → identical output (no randn in forward).
        let diff = (z1 - z2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "encoder forward must be deterministic");
        Ok(())
    }

    #[test]
    fn encoder_is_input_dependent() -> Result<()> {
        let dev = Device::Cpu;
        let c = cfg();
        let enc = Encoder2D::dummy(&c, &dev)?;
        let shape = (1, c.base_channels, c.grid_h, c.grid_w * c.grid_d);
        let x0 = Tensor::zeros(shape, DType::F32, &dev)?;
        let x1 = Tensor::ones(shape, DType::F32, &dev)?;
        let z0 = enc.forward(&x0)?;
        let z1 = enc.forward(&x1)?;
        let diff = (z0 - z1)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            diff > 1e-4,
            "different inputs must give different latents (got {diff})"
        );
        Ok(())
    }

    #[test]
    fn decoder_shape_and_determinism() -> Result<()> {
        let dev = Device::Cpu;
        let c = cfg();
        let dec = Decoder2D::dummy(&c, &dev)?;
        let z = Tensor::randn(0f32, 1.0, (2, c.z_channels, c.token_h, c.token_w), &dev)?;
        let l1 = dec.forward(&z)?;
        let l2 = dec.forward(&z)?;
        assert_eq!(l1.dims(), &[2, c.num_classes, c.grid_h, c.grid_w, c.grid_d]);
        let diff = (l1 - l2)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "decoder forward must be deterministic");
        Ok(())
    }
}

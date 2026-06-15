//! VQVAE components — class embedding, codebook, quant/post-quant convolutions.
//!
//! ## Implementation status
//!
//! | Component            | Status  | Notes                                          |
//! |----------------------|---------|------------------------------------------------|
//! | `ClassEmbedding`     | Full    | `Embedding(18, 64)` — matches Python exactly   |
//! | `VQCodebook`         | Full    | Nearest-neighbour lookup via squared-L2        |
//! | `QuantConv`          | Full    | `Conv2d(128 → 512, k=1)` — quant_conv          |
//! | `PostQuantConv`      | Full    | `Conv2d(512 → 128, k=1)` — post_quant_conv     |
//! | `fold_3d_to_2d`      | Full    | (B*F, C, H, W*D) reshape for 2D CNN            |
//! | `Encoder2D` (conv)   | Full    | Real deterministic conv encoder — see [`crate::cnn`]. |
//! | `Decoder2D` (conv)   | Full    | Real deterministic conv decoder — see [`crate::cnn`]. |
//!
//! The encoder/decoder are a genuine, input-dependent convolutional forward
//! pass (no `randn`). With the `dummy` constructor the weights are
//! deterministically initialised but **untrained** — accuracy is data-gated
//! on a Phase-5 checkpoint, disclosed via the `weights_trained` flag on
//! [`crate::inference::InferenceOutput`].

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, VarBuilder};

use crate::cnn::{Decoder2D, Encoder2D};
use crate::config::OccWorldConfig;

// ── Class embedding ───────────────────────────────────────────────────────────

/// Embeds integer class labels `[0, num_classes)` into `base_channels`-dim vectors.
///
/// Matches `nn.Embedding(18, 64)` in `vae_2d_resnet.py`.
pub struct ClassEmbedding {
    embed: Embedding,
}

impl ClassEmbedding {
    /// Build from a [`VarBuilder`] using the sub-path `"class_embed"`.
    pub fn new(num_classes: usize, embed_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let embed = candle_nn::embedding(num_classes, embed_dim, vb.pp("class_embed"))?;
        Ok(Self { embed })
    }

    /// Build with deterministic untrained initialisation (tests / benchmarks).
    pub fn dummy(num_classes: usize, embed_dim: usize, device: &Device) -> Result<Self> {
        let w = crate::cnn::det_fill(&[num_classes, embed_dim], 0x0CE0_0001, 1.0, device)?;
        let embed = Embedding::new(w, embed_dim);
        Ok(Self { embed })
    }

    /// Forward: `(B*F, H, W, D)` u32 indices → `(B*F, embed_dim, H, W*D)`.
    ///
    /// The 3-D grid is folded along the depth axis so a 2-D CNN can process it.
    pub fn forward(&self, x: &Tensor, grid_d: usize) -> Result<Tensor> {
        // x: (B*F, H, W, D) — integer class labels stored as u32
        let (bf, h, w, _d) = x.dims4()?;

        // Flatten spatial+depth → apply embedding → (B*F, H, W, D, embed_dim)
        let flat = x.flatten_all()?; // (B*F*H*W*D,)
        let embedded = self.embed.forward(&flat)?; // (B*F*H*W*D, embed_dim)
        let c = embedded.dim(1)?;

        // Reshape to (B*F, H, W, D, C) then transpose to (B*F, C, H, W*D)
        let vol = embedded.reshape((bf, h, w, grid_d, c))?;
        // (B*F, H, W, D, C) → (B*F, C, H, W, D) → (B*F, C, H, W*D)
        let transposed = vol.permute((0, 4, 1, 2, 3))?;
        let (bf2, c2, h2, w2, d2) = transposed.dims5()?;
        transposed.reshape((bf2, c2, h2, w2 * d2))
    }
}

// ── fold_3d_to_2d helper ─────────────────────────────────────────────────────

/// Reshape `(B*F, C, H, W, D)` into `(B*F, C, H, W*D)` for 2-D CNNs.
///
/// This is the "fold" operation described in `vae_2d_resnet.py`:
/// the depth axis is concatenated into the width so that standard
/// `Conv2d` layers can process the full 3-D occupancy volume.
pub fn fold_3d_to_2d(x: &Tensor) -> Result<Tensor> {
    let (bf, c, h, w, d) = x.dims5()?;
    x.reshape((bf, c, h, w * d))
}

/// Inverse of `fold_3d_to_2d`: `(B*F, C, H, W*D)` → `(B*F, C, H, W, D)`.
pub fn unfold_2d_to_3d(x: &Tensor, grid_w: usize, grid_d: usize) -> Result<Tensor> {
    let (bf, c, h, _wd) = x.dims4()?;
    x.reshape((bf, c, h, grid_w, grid_d))
}

// ── Vector-quantisation codebook ─────────────────────────────────────────────

/// VQ codebook: `num_codes × embed_dim` lookup table.
///
/// Nearest-neighbour assignment uses squared L2 distance:
/// ```text
/// d(z, e_k) = ||z − e_k||² = ||z||² − 2·z·e_kᵀ + ||e_k||²
/// ```
/// This is standard VQ-VAE (van den Oord et al., 2017).
pub struct VQCodebook {
    /// Shape: `(codebook_size, embed_dim)`.
    embeddings: Tensor,
    /// Number of discrete codes in the codebook.
    pub codebook_size: usize,
    /// Dimensionality of each codebook embedding vector.
    pub embed_dim: usize,
}

impl VQCodebook {
    /// Load from a [`VarBuilder`] using the sub-path `"quantize.embedding.weight"`.
    pub fn new(codebook_size: usize, embed_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let embeddings = vb
            .pp("quantize")
            .pp("embedding")
            .get((codebook_size, embed_dim), "weight")?;
        Ok(Self {
            embeddings,
            codebook_size,
            embed_dim,
        })
    }

    /// Deterministic untrained initialisation (for tests / benchmarks).
    pub fn dummy(codebook_size: usize, embed_dim: usize, device: &Device) -> Result<Self> {
        let embeddings =
            crate::cnn::det_fill(&[codebook_size, embed_dim], 0x0CE0_0002, 1.0, device)?;
        Ok(Self {
            embeddings,
            codebook_size,
            embed_dim,
        })
    }

    /// Quantise `z` (any shape `[..., embed_dim]`) → `(z_q, indices)`.
    ///
    /// `z_q` has the same shape as `z`; `indices` has shape `[..., 1]` squeezed
    /// to `[...]` (batch of scalar indices).
    pub fn encode(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let orig_shape = z.shape().clone();
        let orig_dims = orig_shape.dims().to_vec();
        let last = *orig_shape.dims().last().unwrap_or(&0);
        // Guard the divide below: a scalar (rank-0) or empty-last-dim tensor
        // would make `last == 0` and panic on the `elem_count() / last`
        // division. `encode` is a `pub fn` on a `pub struct`, so this is a
        // reachable public boundary — fail closed with a clear error instead.
        if last == 0 {
            return Err(candle_core::Error::Msg(format!(
                "VQCodebook::encode expects a tensor with a non-zero last dim of \
                 size embed_dim={}, got shape {orig_dims:?}",
                self.embed_dim
            )));
        }
        // Flatten to (N, embed_dim)
        let n = z.elem_count() / last;
        let z_flat = z.reshape((n, last))?; // (N, D)

        // Squared L2: ||z||² - 2*z*Eᵀ + ||E||²
        // z_sq: (N, 1)
        let z_sq = z_flat
            .sqr()?
            .sum(candle_core::D::Minus1)?
            .unsqueeze(1)?;
        // e_sq: (1, codebook_size)
        let e_sq = self
            .embeddings
            .sqr()?
            .sum(candle_core::D::Minus1)?
            .unsqueeze(0)?;
        // dot: (N, codebook_size)
        let dot = z_flat.matmul(&self.embeddings.t()?)?;
        // distances: (N, codebook_size)
        let distances = z_sq.broadcast_add(&e_sq)?.broadcast_sub(&dot.affine(2.0, 0.0)?)?;
        // indices: (N,)
        let indices = distances.argmin(candle_core::D::Minus1)?;

        // Look up quantised embeddings
        let z_q_flat = self.embeddings.index_select(&indices, 0)?; // (N, D)

        // Reshape back to original shape
        let z_q = z_q_flat.reshape(orig_dims.clone())?;
        let idx_shape: Vec<usize> = orig_dims[..orig_dims.len() - 1].to_vec();
        let indices_out = indices.reshape(idx_shape)?;

        Ok((z_q, indices_out))
    }

    /// Decode flat index tensor `(N,)` or `(B, ...)` → same shape `+ embed_dim`.
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let flat = indices.flatten_all()?;
        let z_flat = self.embeddings.index_select(&flat, 0)?; // (N, D)
        let mut out_shape: Vec<usize> = indices.dims().to_vec();
        out_shape.push(self.embed_dim);
        z_flat.reshape(out_shape)
    }
}

// ── Quant / post-quant convolutions ──────────────────────────────────────────

/// `Conv2d(z_channels → embed_dim, kernel=1)` — `quant_conv` in Python.
pub struct QuantConv {
    conv: Conv2d,
}

impl QuantConv {
    /// Load from weights.
    pub fn new(z_channels: usize, embed_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let conv = candle_nn::conv2d(
            z_channels,
            embed_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("quant_conv"),
        )?;
        Ok(Self { conv })
    }

    /// Deterministic untrained initialisation.
    pub fn dummy(z_channels: usize, embed_dim: usize, device: &Device) -> Result<Self> {
        let w = crate::cnn::det_fill(&[embed_dim, z_channels, 1, 1], 0x0CE0_0003, 1.0, device)?;
        let b = Tensor::zeros(embed_dim, DType::F32, device)?;
        let conv = Conv2d::new(w, Some(b), Conv2dConfig::default());
        Ok(Self { conv })
    }

    /// Forward: `(B*F, z_channels, H, W)` → `(B*F, embed_dim, H, W)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

/// `Conv2d(embed_dim → z_channels, kernel=1)` — `post_quant_conv` in Python.
pub struct PostQuantConv {
    conv: Conv2d,
}

impl PostQuantConv {
    /// Load from weights.
    pub fn new(embed_dim: usize, z_channels: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let conv = candle_nn::conv2d(
            embed_dim,
            z_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("post_quant_conv"),
        )?;
        Ok(Self { conv })
    }

    /// Deterministic untrained initialisation.
    pub fn dummy(embed_dim: usize, z_channels: usize, device: &Device) -> Result<Self> {
        let w = crate::cnn::det_fill(&[z_channels, embed_dim, 1, 1], 0x0CE0_0004, 1.0, device)?;
        let b = Tensor::zeros(z_channels, DType::F32, device)?;
        let conv = Conv2d::new(w, Some(b), Conv2dConfig::default());
        Ok(Self { conv })
    }

    /// Forward: `(B*F, embed_dim, H, W)` → `(B*F, z_channels, H, W)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)
    }
}

// ── Encoder / decoder entry points ────────────────────────────────────────────
//
// The former `Tensor::randn` stubs are gone. The real, deterministic,
// input-dependent convolutional encoder/decoder live in [`crate::cnn`]; the
// VQVAE bundle below owns a concrete [`Encoder2D`] / [`Decoder2D`] instance and
// the inference engine drives them directly. These thin re-exports keep the
// historical call sites working.
pub use crate::cnn::{decode_to_logits, encode_occupancy};

// ── VQVAE component bundle ────────────────────────────────────────────────────

/// All VQVAE components bundled together for use in `OccWorldCandle`.
pub struct VQVAEComponents {
    /// Class label → float embedding (`nn.Embedding(18, 64)` in Python).
    pub class_embed: ClassEmbedding,
    /// Real convolutional encoder: occupancy grid → latent feature map.
    pub encoder: Encoder2D,
    /// `Conv2d(z_channels → embed_dim, k=1)` before quantisation.
    pub quant_conv: QuantConv,
    /// VQ codebook for nearest-neighbour quantisation.
    pub codebook: VQCodebook,
    /// `Conv2d(embed_dim → z_channels, k=1)` after quantisation.
    pub post_quant_conv: PostQuantConv,
    /// Real convolutional decoder: latent codes → per-voxel class logits.
    pub decoder: Decoder2D,
}

impl VQVAEComponents {
    /// Build all components from a single [`VarBuilder`] (trained checkpoint).
    pub fn new(cfg: &OccWorldConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let class_embed = ClassEmbedding::new(cfg.num_classes, cfg.base_channels, vb.clone())?;
        let encoder = Encoder2D::from_weights(cfg, vb.clone())?;
        let quant_conv = QuantConv::new(cfg.z_channels, cfg.embed_dim, vb.clone())?;
        let codebook = VQCodebook::new(cfg.codebook_size, cfg.embed_dim, vb.clone())?;
        let post_quant_conv = PostQuantConv::new(cfg.embed_dim, cfg.z_channels, vb.clone())?;
        let decoder = Decoder2D::from_weights(cfg, vb)?;
        Ok(Self {
            class_embed,
            encoder,
            quant_conv,
            codebook,
            post_quant_conv,
            decoder,
        })
    }

    /// Build all components with deterministic *untrained* weights (tests /
    /// benchmarks). The forward pass is real and input-dependent; only the
    /// weight values are not from a trained checkpoint.
    pub fn dummy(cfg: &OccWorldConfig, device: &Device) -> Result<Self> {
        let class_embed = ClassEmbedding::dummy(cfg.num_classes, cfg.base_channels, device)?;
        let encoder = Encoder2D::dummy(cfg, device)?;
        let quant_conv = QuantConv::dummy(cfg.z_channels, cfg.embed_dim, device)?;
        let codebook = VQCodebook::dummy(cfg.codebook_size, cfg.embed_dim, device)?;
        let post_quant_conv = PostQuantConv::dummy(cfg.embed_dim, cfg.z_channels, device)?;
        let decoder = Decoder2D::dummy(cfg, device)?;
        Ok(Self {
            class_embed,
            encoder,
            quant_conv,
            codebook,
            post_quant_conv,
            decoder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vq_codebook_roundtrip() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let codebook = VQCodebook::dummy(512, 512, &device)?;

        // Random input of shape (4, 512) — simulate a batch of 4 latent vectors
        let z = Tensor::randn(0f32, 1.0, (4, 512), &device)?;

        let (z_q, indices) = codebook.encode(&z)?;
        // z_q must have same shape as z
        assert_eq!(z_q.dims(), z.dims());
        // indices must have shape (4,) — one per row
        assert_eq!(indices.dims(), &[4]);

        // Decode must recover the same codebook entries
        let z_decoded = codebook.decode(&indices)?;
        assert_eq!(z_decoded.dims(), &[4, 512]);

        Ok(())
    }

    #[test]
    fn encode_rejects_scalar_without_panicking() {
        // A rank-0 (scalar) tensor has an empty dims list → `last == 0`.
        // Before the guard this divided by zero and panicked; now it returns
        // a clean error. `encode` is public, so this is a reachable boundary.
        let device = Device::Cpu;
        let codebook = VQCodebook::dummy(4, 8, &device).unwrap();
        let scalar = Tensor::from_vec(vec![1.0f32], (), &device).unwrap();
        let result = codebook.encode(&scalar);
        assert!(
            result.is_err(),
            "scalar input must error, not panic; got {result:?}"
        );
    }

    #[test]
    fn test_fold_unfold_roundtrip() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 64, 10, 10, 8), &device)?;
        let folded = fold_3d_to_2d(&x)?;
        assert_eq!(folded.dims(), &[2, 64, 10, 80]);
        let unfolded = unfold_2d_to_3d(&folded, 10, 8)?;
        assert_eq!(unfolded.dims(), &[2, 64, 10, 10, 8]);
        Ok(())
    }
}

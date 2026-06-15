//! OccWorld autoregressive transformer — `PlanUAutoRegTransformer` port.
//!
//! Architecture summary (matches `PlanUtransformer.py`):
//!
//! 1. Input: quantised VQVAE tokens `z_q` of shape `(B, F, C, H, W)`.
//! 2. Spatial flatten: `(B*F, C, H*W)` so each frame is a sequence of spatial tokens.
//! 3. Temporal embedding: learned positional bias added to the C-dim channel.
//! 4. Per-layer: `TemporalCrossAttn` → `SpatialCrossAttn` → FFN.
//! 5. Output head: `Linear(C → vocab)` producing logits `(B, F_out, vocab, H, W)`.
//!
//! The two-level UNet attention (`num_layers = 2`) uses separate query/key/value
//! projections at each level so the encoder sees the full past context while
//! the decoder generates one future frame at a time.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, ops::softmax, Embedding, Linear, VarBuilder};

use crate::config::OccWorldConfig;
use crate::error::OccWorldError;

// ── Temporal positional embedding ─────────────────────────────────────────────

/// Maps frame indices `[0, num_frames*2)` to `embed_dim`-dimensional vectors.
///
/// The doubled range (`num_frames*2`) allows future frame positions to be
/// distinct from past frame positions (Python: `nn.Embedding(16 * 2, 512)`).
pub struct TemporalEmbedding {
    embed: Embedding,
}

impl TemporalEmbedding {
    /// Build from weights.
    pub fn new(num_frames: usize, embed_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let embed = candle_nn::embedding(num_frames * 2, embed_dim, vb.pp("temporal_embed"))?;
        Ok(Self { embed })
    }

    /// Deterministic untrained initialisation.
    pub fn dummy(num_frames: usize, embed_dim: usize, device: &Device) -> Result<Self> {
        let w = crate::cnn::det_fill(&[num_frames * 2, embed_dim], 0x07A0_0001, 1.0, device)?;
        let embed = Embedding::new(w, embed_dim);
        Ok(Self { embed })
    }

    /// Produce positional embedding for frame indices `[0, F)`.
    ///
    /// Returns `(F, embed_dim)` — broadcast over batch and spatial dimensions
    /// by the caller.
    pub fn forward(&self, num_frames: usize, device: &Device) -> Result<Tensor> {
        let indices = Tensor::arange(0u32, num_frames as u32, device)?;
        self.embed.forward(&indices) // (F, embed_dim)
    }
}

// ── Scaled-dot-product attention helpers ─────────────────────────────────────

/// Scaled dot-product attention: `softmax(Q·Kᵀ / √d) · V`.
///
/// All tensors are `(B, heads, seq_len, head_dim)`.
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let head_dim = q.dim(candle_core::D::Minus1)? as f64;
    let scale = (head_dim).sqrt();
    // (B, heads, q_len, k_len)
    let attn_weights = (q.matmul(&k.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?)?
        / scale)?;
    let attn_probs = softmax(&attn_weights, candle_core::D::Minus1)?;
    attn_probs.matmul(v)
}

// ── Spatial cross-attention ───────────────────────────────────────────────────

/// Multi-head self/cross-attention over the spatial token sequence.
///
/// Used to capture dependencies between different spatial locations within
/// the same frame (or across frames when keys/values come from a different
/// temporal index).
pub struct SpatialCrossAttn {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SpatialCrossAttn {
    /// Build from weights with sub-path `prefix`.
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    /// Deterministic untrained initialisation (distinct seed per projection).
    pub fn dummy(embed_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let mk_linear = |i: usize, o: usize, seed: u64| -> Result<Linear> {
            let w = crate::cnn::det_fill(&[o, i], seed, 0.02, device)?;
            let b = Tensor::zeros(o, DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj: mk_linear(embed_dim, embed_dim, 0x07A0_1001)?,
            k_proj: mk_linear(embed_dim, embed_dim, 0x07A0_1002)?,
            v_proj: mk_linear(embed_dim, embed_dim, 0x07A0_1003)?,
            out_proj: mk_linear(embed_dim, embed_dim, 0x07A0_1004)?,
            num_heads,
            head_dim,
        })
    }

    /// Forward attention.
    ///
    /// `queries`: `(B, q_len, C)`, `keys`/`values`: `(B, kv_len, C)`.
    /// Returns: `(B, q_len, C)`.
    pub fn forward(&self, queries: &Tensor, keys: &Tensor, values: &Tensor) -> Result<Tensor> {
        let (b, q_len, _c) = queries.dims3()?;

        let project = |proj: &Linear, x: &Tensor, seq: usize| -> Result<Tensor> {
            let out = proj.forward(x)?; // (B, seq, C)
            out.reshape((b, seq, self.num_heads, self.head_dim))?
                .permute((0, 2, 1, 3)) // (B, heads, seq, head_dim)
        };

        let kv_len = keys.dim(1)?;
        let q = project(&self.q_proj, queries, q_len)?.contiguous()?;
        let k = project(&self.k_proj, keys, kv_len)?.contiguous()?;
        let v = project(&self.v_proj, values, kv_len)?.contiguous()?;

        // (B, heads, q_len, head_dim)
        let attended = scaled_dot_product_attention(&q, &k, &v)?;
        // → (B, q_len, C)
        let merged = attended
            .permute((0, 2, 1, 3))?
            .reshape((b, q_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&merged)
    }
}

// ── Temporal cross-attention ──────────────────────────────────────────────────

/// Cross-attention between past-frame tokens (keys/values) and query tokens.
///
/// Identical in structure to `SpatialCrossAttn` — kept as a distinct type
/// for clarity and separate weight namespacing in the checkpoint.
pub struct TemporalCrossAttn {
    inner: SpatialCrossAttn,
}

impl TemporalCrossAttn {
    /// Build from weights.
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            inner: SpatialCrossAttn::new(embed_dim, num_heads, vb)?,
        })
    }

    /// Random initialisation.
    pub fn dummy(embed_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            inner: SpatialCrossAttn::dummy(embed_dim, num_heads, device)?,
        })
    }

    /// Forward: `queries (B, q_len, C)` attend to `keys/values (B, kv_len, C)`.
    pub fn forward(&self, queries: &Tensor, keys: &Tensor, values: &Tensor) -> Result<Tensor> {
        self.inner.forward(queries, keys, values)
    }
}

// ── Feed-forward network ──────────────────────────────────────────────────────

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(embed_dim: usize, ffn_hidden: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let fc1 = linear(embed_dim, ffn_hidden, vb.pp("fc1"))?;
        let fc2 = linear(ffn_hidden, embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn dummy(embed_dim: usize, ffn_hidden: usize, device: &Device) -> Result<Self> {
        let mk = |i: usize, o: usize, seed: u64| -> Result<Linear> {
            let w = crate::cnn::det_fill(&[o, i], seed, 0.02, device)?;
            let b = Tensor::zeros(o, DType::F32, device)?;
            Ok(Linear::new(w, Some(b)))
        };
        Ok(Self {
            fc1: mk(embed_dim, ffn_hidden, 0x07A0_2001)?,
            fc2: mk(ffn_hidden, embed_dim, 0x07A0_2002)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

// ── Single encoder layer ─────────────────────────────────────────────────────

/// One layer of the OccWorld UNet-style encoder:
/// `TemporalCrossAttn → SpatialCrossAttn → FFN` with residual connections.
pub struct OccWorldTransformerLayer {
    temporal_attn: TemporalCrossAttn,
    spatial_attn: SpatialCrossAttn,
    ffn: FeedForward,
    // Layer-norms for pre-norm formulation
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    norm3: candle_nn::LayerNorm,
}

impl OccWorldTransformerLayer {
    /// Build from weights.
    pub fn new(cfg: &OccWorldConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let temporal_attn =
            TemporalCrossAttn::new(cfg.embed_dim, cfg.num_heads, vb.pp("temporal_attn"))?;
        let spatial_attn =
            SpatialCrossAttn::new(cfg.embed_dim, cfg.num_heads, vb.pp("spatial_attn"))?;
        let ffn = FeedForward::new(cfg.embed_dim, cfg.ffn_hidden, vb.pp("ffn"))?;
        let norm_cfg = candle_nn::LayerNormConfig::default();
        let norm1 = candle_nn::layer_norm(cfg.embed_dim, norm_cfg, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(cfg.embed_dim, norm_cfg, vb.pp("norm2"))?;
        let norm3 = candle_nn::layer_norm(cfg.embed_dim, norm_cfg, vb.pp("norm3"))?;
        Ok(Self {
            temporal_attn,
            spatial_attn,
            ffn,
            norm1,
            norm2,
            norm3,
        })
    }

    /// Random initialisation.
    pub fn dummy(cfg: &OccWorldConfig, device: &Device) -> Result<Self> {
        let temporal_attn = TemporalCrossAttn::dummy(cfg.embed_dim, cfg.num_heads, device)?;
        let spatial_attn = SpatialCrossAttn::dummy(cfg.embed_dim, cfg.num_heads, device)?;
        let ffn = FeedForward::dummy(cfg.embed_dim, cfg.ffn_hidden, device)?;
        let norm_cfg = candle_nn::LayerNormConfig::default();
        // Dummy layer norms with ones/zeros
        let mk_norm = |d: usize| -> Result<candle_nn::LayerNorm> {
            let w = Tensor::ones(d, DType::F32, device)?;
            let b = Tensor::zeros(d, DType::F32, device)?;
            Ok(candle_nn::LayerNorm::new(w, b, norm_cfg.eps))
        };
        Ok(Self {
            temporal_attn,
            spatial_attn,
            ffn,
            norm1: mk_norm(cfg.embed_dim)?,
            norm2: mk_norm(cfg.embed_dim)?,
            norm3: mk_norm(cfg.embed_dim)?,
        })
    }

    /// Forward one layer.
    ///
    /// `x`: `(B, seq_len, C)` — queries (current frame tokens).
    /// `ctx`: `(B, ctx_len, C)` — past-frame context tokens for temporal attn.
    /// Returns `(B, seq_len, C)`.
    pub fn forward(&self, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        // Temporal cross-attention with residual
        let x = {
            let normed = self.norm1.forward(x)?;
            let attended = self.temporal_attn.forward(&normed, ctx, ctx)?;
            (x + attended)?
        };
        // Spatial self-attention with residual
        let x = {
            let normed = self.norm2.forward(&x)?;
            let attended = self.spatial_attn.forward(&normed, &normed, &normed)?;
            (x + attended)?
        };
        // FFN with residual
        let normed = self.norm3.forward(&x)?;
        let ff_out = self.ffn.forward(&normed)?;
        x + ff_out
    }
}

// ── Full transformer ──────────────────────────────────────────────────────────

/// OccWorld autoregressive transformer (`PlanUAutoRegTransformer`).
///
/// Takes quantised VQVAE tokens for past frames and predicts logits for
/// the next `F_out` frames.
pub struct OccWorldTransformer {
    temporal_embed: TemporalEmbedding,
    layers: Vec<OccWorldTransformerLayer>,
    output_head: Linear,
    cfg: OccWorldConfig,
}

impl OccWorldTransformer {
    /// Build from weights.
    pub fn new(cfg: OccWorldConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let temporal_embed =
            TemporalEmbedding::new(cfg.num_frames, cfg.embed_dim, vb.pp("transformer"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(OccWorldTransformerLayer::new(
                &cfg,
                vb.pp("transformer").pp(format!("layer_{i}")),
            )?);
        }
        let output_head = linear(
            cfg.embed_dim,
            cfg.codebook_size,
            vb.pp("transformer").pp("output_head"),
        )?;
        Ok(Self {
            temporal_embed,
            layers,
            output_head,
            cfg,
        })
    }

    /// Build with random weights (for tests / benchmarks).
    pub fn dummy(cfg: OccWorldConfig, device: &Device) -> Result<Self> {
        let temporal_embed = TemporalEmbedding::dummy(cfg.num_frames, cfg.embed_dim, device)?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(OccWorldTransformerLayer::dummy(&cfg, device)?);
        }
        let w = crate::cnn::det_fill(
            &[cfg.codebook_size, cfg.embed_dim],
            0x07A0_3001,
            0.02,
            device,
        )?;
        let b = Tensor::zeros(cfg.codebook_size, DType::F32, device)?;
        let output_head = Linear::new(w, Some(b));
        Ok(Self {
            temporal_embed,
            layers,
            output_head,
            cfg,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `z_q` — quantised tokens: `(B, F, C, H, W)` where `C = embed_dim`.
    ///
    /// # Returns
    /// Predicted logits: `(B, F_out, vocab, H, W)` where `F_out = F` and
    /// `vocab = codebook_size`.
    pub fn forward(
        &self,
        z_q: &Tensor,
    ) -> std::result::Result<Tensor, OccWorldError> {
        let (b, f, c, h, w) = z_q.dims5().map_err(OccWorldError::Candle)?;
        let device = z_q.device();

        // Flatten spatial: (B, F, C, H, W) → (B, F, H*W, C)
        // Then flatten batch*frames for parallel processing: (B*F, H*W, C)
        let z_flat = z_q
            .permute((0, 1, 3, 4, 2)) // (B, F, H, W, C)
            .map_err(OccWorldError::Candle)?
            .reshape((b * f, h * w, c))
            .map_err(OccWorldError::Candle)?;

        // Add temporal positional embedding — broadcast over spatial tokens
        let temp_pos = self
            .temporal_embed
            .forward(f, device)
            .map_err(OccWorldError::Candle)?; // (F, C)
        // Expand to (B*F, 1, C) for broadcast addition
        let temp_pos = temp_pos
            .reshape((f, 1, c))
            .map_err(OccWorldError::Candle)?
            .repeat(vec![b, 1, 1])
            .map_err(OccWorldError::Candle)?
            .reshape((b * f, 1, c))
            .map_err(OccWorldError::Candle)?;
        let mut x = z_flat
            .broadcast_add(&temp_pos)
            .map_err(OccWorldError::Candle)?; // (B*F, H*W, C)

        // Context for temporal attention: reshape back to (B, F*H*W, C) per batch
        // and use the full past sequence as keys/values
        let ctx = x
            .reshape((b, f * h * w, c))
            .map_err(OccWorldError::Candle)?
            .repeat(vec![f, 1, 1])
            .map_err(OccWorldError::Candle)?
            .reshape((b * f, f * h * w, c))
            .map_err(OccWorldError::Candle)?;

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(&x, &ctx).map_err(OccWorldError::Candle)?;
        }

        // Output head: (B*F, H*W, C) → (B*F, H*W, vocab)
        let logits = self
            .output_head
            .forward(&x)
            .map_err(OccWorldError::Candle)?;
        let vocab = self.cfg.codebook_size;

        // Reshape to (B, F, H*W, vocab) → (B, F, vocab, H, W)
        let logits_out = logits
            .reshape((b, f, h * w, vocab))
            .map_err(OccWorldError::Candle)?
            .permute((0, 1, 3, 2)) // (B, F, vocab, H*W)
            .map_err(OccWorldError::Candle)?
            .reshape((b, f, vocab, h, w))
            .map_err(OccWorldError::Candle)?;

        Ok(logits_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_forward_shape() -> std::result::Result<(), OccWorldError> {
        let device = Device::Cpu;
        let cfg = OccWorldConfig {
            num_frames: 4, // smaller for fast test
            embed_dim: 16,
            codebook_size: 8,
            token_h: 4,
            token_w: 4,
            num_heads: 2,
            num_layers: 1,
            ffn_hidden: 32,
            ..OccWorldConfig::default()
        };

        let transformer = OccWorldTransformer::dummy(cfg.clone(), &device)
            .map_err(OccWorldError::Candle)?;

        // (B=1, F=4, C=16, H=4, W=4)
        let z_q = Tensor::randn(
            0f32,
            1.0,
            (1, cfg.num_frames, cfg.embed_dim, cfg.token_h, cfg.token_w),
            &device,
        )
        .map_err(OccWorldError::Candle)?;

        let logits = transformer.forward(&z_q)?;
        // Expected: (1, 4, 8, 4, 4)
        assert_eq!(
            logits.dims(),
            &[1, cfg.num_frames, cfg.codebook_size, cfg.token_h, cfg.token_w]
        );

        Ok(())
    }
}

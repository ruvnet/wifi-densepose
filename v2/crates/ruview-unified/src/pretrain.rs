//! Masked-reconstruction pretraining for the RF foundation encoder
//! (ADR-274 §3.2), with an exact hand-derived backward pass.
//!
//! The correctness argument is not "the loss went down" alone: the analytic
//! gradients of *every* parameter group are verified against central finite
//! differences (`tests::gradients_match_finite_differences`), which pins the
//! backward pass to the forward pass to ~1e-8 relative error. The training
//! loop is then ordinary SGD.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::encoder::{ForwardCache, Linear, RfEncoder, WindowContext};
use crate::math::seeded_rng;
use crate::tokenizer::{position_encoding, RfToken, TokenizedWindow};

/// Gradient accumulator mirroring [`RfEncoder`]'s parameter groups.
pub struct EncoderGrads {
    /// Token embedding grads.
    pub w1: Linear,
    /// Context mixing 1 grads.
    pub w2: Linear,
    /// Context mixing 2 grads.
    pub w2b: Linear,
    /// Age gate weight grads.
    pub age_w: Vec<f64>,
    /// Age gate bias grads.
    pub age_b: Vec<f64>,
    /// Geometry encoder grads.
    pub wg: Linear,
    /// Reconstruction head grads.
    pub w3: Linear,
}

impl EncoderGrads {
    fn zeros(enc: &RfEncoder) -> Self {
        Self {
            w1: enc.w1.zeros_like(),
            w2: enc.w2.zeros_like(),
            w2b: enc.w2b.zeros_like(),
            age_w: vec![0.0; enc.age_w.len()],
            age_b: vec![0.0; enc.age_b.len()],
            wg: enc.wg.zeros_like(),
            w3: enc.w3.zeros_like(),
        }
    }
}

/// Mean-squared masked-reconstruction loss for one window under a fixed mask.
#[must_use]
pub fn masked_loss(enc: &RfEncoder, tokens: &[RfToken], masked: &[usize], ctx: WindowContext) -> f64 {
    let cache = enc.forward(tokens, masked, ctx);
    loss_from_cache(enc, &cache, tokens, masked)
}

fn loss_from_cache(
    enc: &RfEncoder,
    cache: &ForwardCache,
    tokens: &[RfToken],
    masked: &[usize],
) -> f64 {
    let d = enc.cfg.d_in;
    let mut loss = 0.0;
    for &j in masked {
        let xhat = enc.reconstruct(cache, j);
        for k in 0..d {
            loss += (xhat[k] - tokens[j].features[k]).powi(2);
        }
    }
    loss / (masked.len() as f64 * d as f64)
}

/// Loss and analytic gradients for one window under a fixed mask.
///
/// Derivation (matching the forward pass in [`RfEncoder::forward`]):
/// `∂L/∂x̂_j = 2(x̂_j − x_j)/(|M|·D)`; the reconstruction input is
/// `u_j = [z ; pos(j)]`, so `∂L/∂z = Σ_j W3[:, :H]ᵀ ∂L/∂x̂_j`; the fusion
/// `z = g⊙gate + Wg·geo + bg` splits the gradient into the tanh chain
/// (`g → m → c → h_i → W1`) and the gate/geometry paths.
#[must_use]
pub fn masked_loss_and_grads(
    enc: &RfEncoder,
    tokens: &[RfToken],
    masked: &[usize],
    ctx: WindowContext,
) -> (f64, EncoderGrads) {
    let h_dim = enc.cfg.d_model;
    let d = enc.cfg.d_in;
    let cache = enc.forward(tokens, masked, ctx);
    let mut grads = EncoderGrads::zeros(enc);

    let norm = 1.0 / (masked.len() as f64 * d as f64);
    let mut dz = vec![0.0; h_dim];
    let mut loss = 0.0;
    for &j in masked {
        let mut u = cache.z.clone();
        u.extend_from_slice(&position_encoding(j)[..enc.cfg.d_pos]);
        let xhat = enc.w3.forward(&u);
        let mut dxhat = vec![0.0; d];
        for k in 0..d {
            let err = xhat[k] - tokens[j].features[k];
            loss += err * err;
            dxhat[k] = 2.0 * err * norm;
        }
        enc.w3.accumulate_grad(&mut grads.w3, &dxhat, &u);
        let du = enc.w3.backward_input(&dxhat);
        for k in 0..h_dim {
            dz[k] += du[k];
        }
    }
    loss *= norm;

    // Fusion: z = g ⊙ gate + Wg·geo + bg. The gate input is the
    // log-scaled age feature, matching the forward pass.
    let age_feat = crate::encoder::age_feature(ctx.age_s);
    let mut dg = vec![0.0; h_dim];
    for k in 0..h_dim {
        let dgate = dz[k] * cache.g[k];
        dg[k] = dz[k] * cache.gate[k];
        let dsig = cache.gate[k] * (1.0 - cache.gate[k]);
        grads.age_w[k] += dgate * dsig * age_feat;
        grads.age_b[k] += dgate * dsig;
    }
    enc.wg.accumulate_grad(&mut grads.wg, &dz, &ctx.geometry);

    // g = tanh(W2b·m + b2b).
    let dg_pre: Vec<f64> = (0..h_dim).map(|k| dg[k] * (1.0 - cache.g[k] * cache.g[k])).collect();
    enc.w2b.accumulate_grad(&mut grads.w2b, &dg_pre, &cache.m);
    let dm = enc.w2b.backward_input(&dg_pre);

    // m = tanh(W2·c + b2).
    let dm_pre: Vec<f64> = (0..h_dim).map(|k| dm[k] * (1.0 - cache.m[k] * cache.m[k])).collect();
    enc.w2.accumulate_grad(&mut grads.w2, &dm_pre, &cache.c);
    let dc = enc.w2.backward_input(&dm_pre);

    // c = mean of h_i; h_i = tanh(W1·x_i + b1).
    let inv_n = 1.0 / cache.unmasked.len() as f64;
    for (slot, &i) in cache.unmasked.iter().enumerate() {
        let hi = &cache.h[slot];
        let dh_pre: Vec<f64> =
            (0..h_dim).map(|k| dc[k] * inv_n * (1.0 - hi[k] * hi[k])).collect();
        enc.w1.accumulate_grad(&mut grads.w1, &dh_pre, &tokens[i].features);
    }

    (loss, grads)
}

/// Applies one SGD step.
pub fn apply_grads(enc: &mut RfEncoder, grads: &EncoderGrads, lr: f64) {
    enc.w1.sgd(&grads.w1, lr);
    enc.w2.sgd(&grads.w2, lr);
    enc.w2b.sgd(&grads.w2b, lr);
    for (p, g) in enc.age_w.iter_mut().zip(&grads.age_w) {
        *p -= lr * g;
    }
    for (p, g) in enc.age_b.iter_mut().zip(&grads.age_b) {
        *p -= lr * g;
    }
    enc.wg.sgd(&grads.wg, lr);
    enc.w3.sgd(&grads.w3, lr);
}

/// Pretraining hyper-parameters.
#[derive(Debug, Clone, Copy)]
pub struct PretrainConfig {
    /// Fraction of tokens masked per window (≥1 token is always masked and
    /// ≥1 always kept).
    pub mask_fraction: f64,
    /// SGD learning rate.
    pub lr: f64,
    /// Epochs over the window set.
    pub epochs: usize,
    /// Mask-sampling seed (weight init is seeded separately at
    /// [`RfEncoder::new`]).
    pub seed: u64,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self { mask_fraction: 0.25, lr: 0.05, epochs: 30, seed: 0x5EED }
    }
}

/// What pretraining measured (reported honestly, not smoothed).
#[derive(Debug, Clone, Copy)]
pub struct PretrainReport {
    /// Mean masked loss over the corpus before any update (fixed eval mask).
    pub initial_loss: f64,
    /// Mean masked loss after the final epoch (same fixed eval mask).
    pub final_loss: f64,
    /// Epochs run.
    pub epochs: usize,
}

fn sample_mask(rng: &mut rand_chacha::ChaCha20Rng, n_tokens: usize, fraction: f64) -> Vec<usize> {
    // A window with fewer than 2 tokens has nothing left to reconstruct from
    // once one token is masked; return an empty mask rather than panicking
    // (`clamp(1, n_tokens - 1)` is invalid once `n_tokens - 1 < 1`).
    if n_tokens < 2 {
        return Vec::new();
    }
    let n_mask = ((n_tokens as f64 * fraction).round() as usize).clamp(1, n_tokens - 1);
    let mut idx: Vec<usize> = (0..n_tokens).collect();
    idx.shuffle(rng);
    idx.truncate(n_mask);
    idx.sort_unstable();
    idx
}

/// Runs masked-reconstruction SGD over `windows`, mutating `enc` in place.
pub fn pretrain(
    enc: &mut RfEncoder,
    windows: &[TokenizedWindow],
    cfg: &PretrainConfig,
) -> PretrainReport {
    assert!(!windows.is_empty(), "pretrain needs at least one window");
    let mut rng = seeded_rng(cfg.seed);

    // Fixed evaluation masks so initial/final losses are comparable.
    let eval_masks: Vec<Vec<usize>> = windows
        .iter()
        .map(|w| sample_mask(&mut rng, w.tokens.len(), cfg.mask_fraction))
        .collect();
    let eval = |e: &RfEncoder| {
        let mut total = 0.0;
        let mut n = 0usize;
        for (w, m) in windows.iter().zip(&eval_masks) {
            if m.is_empty() {
                continue;
            }
            total += masked_loss(e, &w.tokens, m, WindowContext::from(w));
            n += 1;
        }
        if n == 0 { 0.0 } else { total / n as f64 }
    };

    let initial_loss = eval(enc);
    let mut order: Vec<usize> = (0..windows.len()).collect();
    for _ in 0..cfg.epochs {
        order.shuffle(&mut rng);
        for &wi in &order {
            let w = &windows[wi];
            if w.tokens.len() < 2 {
                continue;
            }
            let mask = sample_mask(&mut rng, w.tokens.len(), cfg.mask_fraction);
            let (_, grads) =
                masked_loss_and_grads(enc, &w.tokens, &mask, WindowContext::from(w));
            apply_grads(enc, &grads, cfg.lr);
        }
    }
    let final_loss = eval(enc);
    PretrainReport { initial_loss, final_loss, epochs: cfg.epochs }
}

/// Deterministic pseudo-random corpus where tokens within a window share a
/// latent factor — masked tokens are predictable from context, so a correct
/// learner must beat the constant predictor. Used by tests and benches.
#[must_use]
pub fn correlated_toy_windows(n_windows: usize, tokens_per_window: usize, seed: u64) -> Vec<TokenizedWindow> {
    use crate::tokenizer::{RfToken, D_IN};
    let mut rng = seeded_rng(seed);
    (0..n_windows)
        .map(|_| {
            let latent: f64 = rng.gen_range(-1.0..1.0);
            let tokens = (0..tokens_per_window)
                .map(|k| {
                    let mut f = [0.0f64; D_IN];
                    for (d, v) in f.iter_mut().enumerate() {
                        // Smooth deterministic function of (latent, token, dim)
                        // plus small noise: reconstructable from context.
                        *v = 0.6 * (latent * (1.0 + d as f64 / 8.0) + k as f64 * 0.3).sin()
                            + rng.gen_range(-0.05..0.05);
                    }
                    RfToken { features: f, link: 0, group: k }
                })
                .collect();
            TokenizedWindow {
                tokens,
                age_s: rng.gen_range(0.0..0.5),
                geometry: [0.1, 0.0, 0.2, 0.4, 0.0, 0.2],
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::EncoderConfig;

    /// Central finite differences over EVERY parameter group. This is the
    /// crate's proof that the backward pass matches the forward pass.
    #[test]
    fn gradients_match_finite_differences() {
        let cfg = EncoderConfig { d_in: 24, d_pos: 6, d_model: 7 };
        let mut enc = RfEncoder::new(cfg, 11);
        let windows = correlated_toy_windows(1, 5, 21);
        let tokens = &windows[0].tokens;
        let ctx = WindowContext { age_s: 0.3, geometry: [0.1, -0.2, 0.3, 0.0, 0.2, -0.1] };
        let masked = vec![1, 3];

        let (_, grads) = masked_loss_and_grads(&enc, tokens, &masked, ctx);

        let eps = 1e-6;
        let mut checked = 0usize;
        let mut max_rel = 0.0f64;

        // Closure-free param walker: (getter, analytic grad) pairs by index.
        // Group 0: w1.w, 1: w1.b, 2: w2.w, 3: w2.b, 4: w2b.w, 5: w2b.b,
        // 6: age_w, 7: age_b, 8: wg.w, 9: wg.b, 10: w3.w, 11: w3.b.
        for group in 0..12 {
            let len = match group {
                0 => enc.w1.w.len(),
                1 => enc.w1.b.len(),
                2 => enc.w2.w.len(),
                3 => enc.w2.b.len(),
                4 => enc.w2b.w.len(),
                5 => enc.w2b.b.len(),
                6 => enc.age_w.len(),
                7 => enc.age_b.len(),
                8 => enc.wg.w.len(),
                9 => enc.wg.b.len(),
                10 => enc.w3.w.len(),
                _ => enc.w3.b.len(),
            };
            // Sample a spread of indices per group to keep the test fast
            // while touching every group.
            let stride = (len / 17).max(1);
            for idx in (0..len).step_by(stride) {
                fn param_at(e: &mut RfEncoder, group: usize, idx: usize) -> &mut f64 {
                    match group {
                        0 => &mut e.w1.w[idx],
                        1 => &mut e.w1.b[idx],
                        2 => &mut e.w2.w[idx],
                        3 => &mut e.w2.b[idx],
                        4 => &mut e.w2b.w[idx],
                        5 => &mut e.w2b.b[idx],
                        6 => &mut e.age_w[idx],
                        7 => &mut e.age_b[idx],
                        8 => &mut e.wg.w[idx],
                        9 => &mut e.wg.b[idx],
                        10 => &mut e.w3.w[idx],
                        _ => &mut e.w3.b[idx],
                    }
                }
                let analytic = match group {
                    0 => grads.w1.w[idx],
                    1 => grads.w1.b[idx],
                    2 => grads.w2.w[idx],
                    3 => grads.w2.b[idx],
                    4 => grads.w2b.w[idx],
                    5 => grads.w2b.b[idx],
                    6 => grads.age_w[idx],
                    7 => grads.age_b[idx],
                    8 => grads.wg.w[idx],
                    9 => grads.wg.b[idx],
                    10 => grads.w3.w[idx],
                    _ => grads.w3.b[idx],
                };

                let orig = *param_at(&mut enc, group, idx);
                *param_at(&mut enc, group, idx) = orig + eps;
                let lp = masked_loss(&enc, tokens, &masked, ctx);
                *param_at(&mut enc, group, idx) = orig - eps;
                let lm = masked_loss(&enc, tokens, &masked, ctx);
                *param_at(&mut enc, group, idx) = orig;

                let numeric = (lp - lm) / (2.0 * eps);
                let denom = analytic.abs().max(numeric.abs()).max(1e-8);
                let rel = (analytic - numeric).abs() / denom;
                // Accept either a tight relative match or an absolute
                // difference at the central-difference roundoff floor
                // (ε_machine·|L|/ε ≈ 5e-11) — tiny gradients hit the floor.
                assert!(
                    rel < 1e-5 || (analytic - numeric).abs() < 1e-9,
                    "group {group} idx {idx}: analytic {analytic:.3e} vs numeric {numeric:.3e} (rel {rel:.3e})"
                );
                max_rel = max_rel.max(rel);
                checked += 1;
            }
        }
        assert!(checked > 150, "gradient check must cover a real sample, got {checked}");
        println!("gradient check: {checked} params, max relative error {max_rel:.3e}");
    }

    #[test]
    fn pretraining_reduces_masked_loss_and_beats_mean_baseline() {
        let windows = correlated_toy_windows(40, 8, 99);
        let mut enc = RfEncoder::new(EncoderConfig { d_in: 24, d_pos: 8, d_model: 32 }, 7);
        let report = pretrain(
            &mut enc,
            &windows,
            &PretrainConfig { mask_fraction: 0.25, lr: 0.05, epochs: 40, seed: 123 },
        );
        assert!(
            report.final_loss < 0.5 * report.initial_loss,
            "loss must at least halve: {report:?}"
        );

        // Constant (global-mean) predictor baseline on the same corpus: the
        // per-dim variance of token features. The encoder must beat it —
        // otherwise it learned nothing about context.
        let mut all: Vec<[f64; 24]> = Vec::new();
        for w in &windows {
            for t in &w.tokens {
                all.push(t.features);
            }
        }
        let n = all.len() as f64;
        let mut mean = [0.0f64; 24];
        for f in &all {
            for (m, v) in mean.iter_mut().zip(f) {
                *m += v / n;
            }
        }
        let mut var = 0.0;
        for f in &all {
            for (m, v) in mean.iter().zip(f) {
                var += (v - m).powi(2);
            }
        }
        var /= n * 24.0;
        assert!(
            report.final_loss < 0.8 * var,
            "must beat constant predictor: final {} vs baseline variance {}",
            report.final_loss,
            var
        );
        println!(
            "pretrain: initial {:.4} → final {:.4} (baseline variance {:.4})",
            report.initial_loss, report.final_loss, var
        );
    }

    #[test]
    fn training_is_deterministic() {
        let windows = correlated_toy_windows(10, 6, 5);
        let cfg = PretrainConfig { mask_fraction: 0.3, lr: 0.05, epochs: 5, seed: 77 };
        let mut a = RfEncoder::new(EncoderConfig { d_in: 24, d_pos: 8, d_model: 16 }, 2);
        let mut b = RfEncoder::new(EncoderConfig { d_in: 24, d_pos: 8, d_model: 16 }, 2);
        let ra = pretrain(&mut a, &windows, &cfg);
        let rb = pretrain(&mut b, &windows, &cfg);
        assert_eq!(a.w1.w, b.w1.w);
        assert!((ra.final_loss - rb.final_loss).abs() < 1e-15);
    }
}

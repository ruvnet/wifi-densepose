//! RF tokenizer — canonical tensor in, encoder-ready tokens out
//! (ADR-274 §3.1).
//!
//! One token per `(link, subcarrier-group)`; each token carries amplitude,
//! delay-spectrum, Doppler-spectrum, phase-dynamics, freshness, geometry,
//! clock-quality, and uncertainty features. The exact 24-dimensional layout
//! is documented on [`RfToken`]; the encoder treats it as an opaque vector,
//! so new feature dims only require bumping [`D_IN`].

use num_complex::Complex64;

use crate::math::DftPlan;
use crate::tensor::{RfTensor, CANONICAL_SNAPSHOTS};

/// Subcarrier-group width: 56 bins / 8 = 7 tokens per link.
pub const GROUP_BINS: usize = 8;

/// Token feature dimension.
pub const D_IN: usize = 24;

/// Sinusoidal position-encoding dimension (used only by masked
/// reconstruction so the decoder knows *which* token it is predicting).
pub const D_POS: usize = 16;

/// One tokenized `(link, group)` cell.
///
/// All amplitude-derived features are computed on window-median-normalized,
/// CFO-aligned samples (see [`RfTokenizer::tokenize`]), so they are
/// invariant to front-end gain and common phase drift.
///
/// Feature layout (all values finite, roughly unit-scale):
///
/// | idx | feature |
/// |-----|---------|
/// | 0–7 | `ln(1+amp)` per bin, averaged over snapshots |
/// | 8–11 | delay-domain DFT magnitudes (bins 0–3) of the snapshot-mean group |
/// | 12–15 | `ln(1+100·mag)` Doppler DFT magnitudes (bins 1–4, DC skipped) across snapshots |
/// | 16 | `ln(1+20·std)` temporal amplitude deviation (motion energy) |
/// | 17 | mean inter-snapshot phase velocity (rad/snapshot) |
/// | 18 | sample age (seconds, clipped to 10) |
/// | 19 | link distance / 10 m |
/// | 20 | link midpoint height / 3 m |
/// | 21 | link azimuth / π |
/// | 22 | clock quality |
/// | 23 | uncertainty |
#[derive(Debug, Clone)]
pub struct RfToken {
    /// Feature vector, layout above.
    pub features: [f64; D_IN],
    /// Link index within the source tensor.
    pub link: usize,
    /// Subcarrier-group index within the link.
    pub group: usize,
}

/// A tokenized tensor window plus the window-level context the encoder's
/// age/geometry paths consume.
#[derive(Debug, Clone)]
pub struct TokenizedWindow {
    /// Tokens, link-major then group order.
    pub tokens: Vec<RfToken>,
    /// Window age in seconds (drives the multiplicative freshness gate).
    pub age_s: f64,
    /// Window-level geometry summary: mean TX xyz then mean RX xyz, in
    /// decametres (÷10) to keep unit scale.
    pub geometry: [f64; 6],
}

/// Tokenizer with precomputed DFT plans (delay + Doppler transforms are the
/// hot path; see `benches/unified_bench.rs` for the measured speedup over
/// planless DFTs).
pub struct RfTokenizer {
    delay_plan: DftPlan,
    doppler_plan: DftPlan,
}

impl Default for RfTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl RfTokenizer {
    /// Builds the tokenizer (allocates the two DFT twiddle tables once).
    #[must_use]
    pub fn new() -> Self {
        Self {
            delay_plan: DftPlan::new(GROUP_BINS, 4),
            doppler_plan: DftPlan::new(CANONICAL_SNAPSHOTS, 5),
        }
    }

    /// Tokenizes a canonical tensor. Panics never: the tensor's validated
    /// invariants (canonical dims after adapter normalization) are assumed;
    /// non-canonical bin counts simply produce fewer/more groups.
    ///
    /// Two hardware-invariance steps happen before feature extraction
    /// (ADR-274 §3.1 — without them, chipset gain and CFO drift dominate
    /// every downstream feature):
    ///
    /// 1. **Scale**: all samples are divided by the window's median
    ///    amplitude, so front-end gain and absolute path loss cancel.
    /// 2. **CFO alignment**: per link, each snapshot is de-rotated by the
    ///    common phase between it and snapshot 0
    ///    (`arg Σ_b H[b,s]·H̄[b,0]`) — carrier-frequency-offset drift is a
    ///    *common* rotation and cancels, while a moving scatterer's
    ///    frequency-selective perturbation survives.
    #[must_use]
    pub fn tokenize(&self, tensor: &RfTensor) -> TokenizedWindow {
        let (n_links, n_bins, n_snaps) = tensor.dims();
        let n_groups = n_bins / GROUP_BINS;
        let mut tokens = Vec::with_capacity(n_links * n_groups);

        // Window-level robust amplitude scale.
        let amps: Vec<f64> = tensor.data.iter().map(|z| z.norm()).collect();
        let scale = crate::math::median(&amps).max(1e-12);

        // Per-(link, snapshot) CFO-alignment rotations against snapshot 0.
        let mut align = vec![vec![Complex64::new(1.0, 0.0); n_snaps]; n_links];
        for l in 0..n_links {
            for s in 1..n_snaps {
                let mut acc = Complex64::new(0.0, 0.0);
                for b in 0..n_bins {
                    acc += tensor.data[[l, b, s]] * tensor.data[[l, b, 0]].conj();
                }
                if acc.norm() > 1e-18 {
                    align[l][s] = (acc / acc.norm()).conj();
                }
            }
        }
        let sample = |l: usize, b: usize, s: usize| tensor.data[[l, b, s]] * align[l][s] / scale;

        for l in 0..n_links {
            let geo = &tensor.links[l];
            let dx = geo.rx_pos[0] - geo.tx_pos[0];
            let dy = geo.rx_pos[1] - geo.tx_pos[1];
            let dist = geo.distance_m().max(1e-6);
            let mid_z = (geo.tx_pos[2] + geo.rx_pos[2]) / 2.0;
            let azimuth = dy.atan2(dx);

            for g in 0..n_groups {
                let b0 = g * GROUP_BINS;
                let mut f = [0.0f64; D_IN];

                // Snapshot-mean complex value per bin (delay features) and
                // group-mean complex value per snapshot (Doppler features).
                let mut bin_means = [Complex64::new(0.0, 0.0); GROUP_BINS];
                let mut snap_means = vec![Complex64::new(0.0, 0.0); n_snaps];
                let mut amp_sum = [0.0f64; GROUP_BINS];
                let mut amp_all = Vec::with_capacity(GROUP_BINS * n_snaps);
                for (bi, bin) in (b0..b0 + GROUP_BINS).enumerate() {
                    for (s, sm) in snap_means.iter_mut().enumerate() {
                        let z = sample(l, bin, s);
                        bin_means[bi] += z;
                        *sm += z;
                        amp_sum[bi] += z.norm();
                        amp_all.push(z.norm());
                    }
                }
                for b in &mut bin_means {
                    *b /= n_snaps as f64;
                }
                for s in &mut snap_means {
                    *s /= GROUP_BINS as f64;
                }

                // 0–7: log-amplitudes.
                for bi in 0..GROUP_BINS {
                    f[bi] = (1.0 + amp_sum[bi] / n_snaps as f64).ln();
                }
                // 8–11: delay spectrum of the group.
                for (k, m) in self.delay_plan.magnitudes(&bin_means).iter().enumerate() {
                    f[8 + k] = *m;
                }
                // 12–15: Doppler spectrum across snapshots (skip DC bin 0),
                // log-compressed to O(1): motion magnitudes live at 1e-2 of
                // the static field, and leaving them 20× smaller than the
                // amplitude dims stalls every downstream linear adapter
                // (feature design, not adapter parameters).
                let dop_scale = |m: f64| (1.0 + 100.0 * m).ln();
                if n_snaps == CANONICAL_SNAPSHOTS {
                    let dop = self.doppler_plan.magnitudes(&snap_means);
                    for k in 0..4 {
                        f[12 + k] = dop_scale(dop[1 + k]);
                    }
                } else {
                    for (k, m) in
                        crate::math::dft_magnitudes(&snap_means, 5).iter().skip(1).enumerate()
                    {
                        f[12 + k] = dop_scale(*m);
                    }
                }
                // 16: temporal amplitude std (motion energy), same treatment.
                let mean_amp = amp_all.iter().sum::<f64>() / amp_all.len() as f64;
                let var = amp_all.iter().map(|a| (a - mean_amp).powi(2)).sum::<f64>()
                    / amp_all.len() as f64;
                f[16] = (1.0 + 20.0 * var.sqrt()).ln();
                // 17: mean inter-snapshot phase velocity of the group mean.
                let mut dphi = 0.0;
                for s in 1..n_snaps {
                    dphi += (snap_means[s] * snap_means[s - 1].conj()).arg();
                }
                f[17] = dphi / (n_snaps.max(2) - 1) as f64;
                // 18–23: freshness, geometry, clock, uncertainty.
                f[18] = tensor.sample_age_s.min(10.0);
                f[19] = dist / 10.0;
                f[20] = mid_z / 3.0;
                f[21] = azimuth / std::f64::consts::PI;
                f[22] = tensor.clock_quality;
                f[23] = tensor.uncertainty;

                tokens.push(RfToken { features: f, link: l, group: g });
            }
        }

        let mut geometry = [0.0f64; 6];
        for geo in &tensor.links {
            for i in 0..3 {
                geometry[i] += geo.tx_pos[i];
                geometry[3 + i] += geo.rx_pos[i];
            }
        }
        for v in &mut geometry {
            *v /= 10.0 * n_links as f64;
        }

        TokenizedWindow { tokens, age_s: tensor.sample_age_s, geometry }
    }
}

/// Fixed sinusoidal position encoding for token index `idx` (masked
/// reconstruction target addressing; not a learned parameter).
#[must_use]
pub fn position_encoding(idx: usize) -> [f64; D_POS] {
    let mut p = [0.0f64; D_POS];
    for k in 0..D_POS / 2 {
        let freq = 1.0 / 10_000f64.powf(2.0 * k as f64 / D_POS as f64);
        p[2 * k] = (idx as f64 * freq).sin();
        p[2 * k + 1] = (idx as f64 * freq).cos();
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{CalibrationMeta, LinkGeometry, RfModality, CANONICAL_BINS};
    use ndarray::Array3;

    fn tensor_with(motion: bool) -> RfTensor {
        let data = Array3::from_shape_fn((2, CANONICAL_BINS, CANONICAL_SNAPSHOTS), |(l, b, s)| {
            let base = 1.0 + 0.1 * (b as f64 / 10.0).sin() + 0.05 * l as f64;
            let wobble = if motion {
                // Snapshot-varying, frequency-selective perturbation — a
                // moving scatterer (bin-dependent so CFO alignment, which
                // only removes *common* rotations, must not cancel it).
                0.3 * (2.0 * std::f64::consts::PI * 2.0 * s as f64 / 8.0).sin()
                    * (1.0 + b as f64 / 56.0)
            } else {
                0.0
            };
            Complex64::new(0.0, wobble).exp() * (base + wobble.abs())
        });
        RfTensor::new(
            RfModality::WifiCsi,
            2.437e9,
            20e6,
            data,
            vec![
                LinkGeometry { tx_pos: [0.0, 0.0, 2.0], rx_pos: [4.0, 0.0, 2.0] },
                LinkGeometry { tx_pos: [0.0, 0.0, 2.0], rx_pos: [4.0, 0.3, 2.0] },
            ],
            0.05,
            0,
            "tok-test".into(),
            0.8,
            0.1,
            CalibrationMeta::default(),
        )
        .expect("valid tensor")
    }

    #[test]
    fn produces_expected_token_grid() {
        let w = RfTokenizer::new().tokenize(&tensor_with(false));
        assert_eq!(w.tokens.len(), 2 * (CANONICAL_BINS / GROUP_BINS));
        for t in &w.tokens {
            assert!(t.features.iter().all(|v| v.is_finite()), "non-finite feature");
        }
        // Context passthrough.
        assert!((w.age_s - 0.05).abs() < 1e-12);
        assert!(w.tokens[0].features[22] > 0.79 && w.tokens[0].features[22] < 0.81);
    }

    #[test]
    fn motion_raises_doppler_and_variance_features() {
        let tok = RfTokenizer::new();
        let still = tok.tokenize(&tensor_with(false));
        let moving = tok.tokenize(&tensor_with(true));
        let dop = |w: &TokenizedWindow| {
            w.tokens.iter().map(|t| t.features[12..16].iter().sum::<f64>()).sum::<f64>()
        };
        let var = |w: &TokenizedWindow| w.tokens.iter().map(|t| t.features[16]).sum::<f64>();
        assert!(
            dop(&moving) > 10.0 * dop(&still) + 1e-9,
            "Doppler features must respond to motion: moving={} still={}",
            dop(&moving),
            dop(&still)
        );
        assert!(var(&moving) > var(&still));
    }

    #[test]
    fn position_encoding_is_unique_and_bounded() {
        let a = position_encoding(0);
        let b = position_encoding(7);
        assert_ne!(a, b);
        for v in a.iter().chain(b.iter()) {
            assert!(v.abs() <= 1.0);
        }
    }
}

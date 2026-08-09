//! Link-throughput model for the protected node.
//!
//! The claim under test is "throughput stays above 95% with the shield on".
//! The model is intentionally transparent and errs toward *charging* the
//! shield, not flattering it:
//!
//! - **Beamforming residual.** The legitimate receiver shares the session key
//!   and inverts the protector's rotation, so it does not pay the rotation
//!   itself — only the residual from quantizing the extra angles at
//!   `feedback_bits` resolution. Per-angle mean-square quantization error is
//!   `Δ²/12` for step `Δ = (π/2)/2^bits`; this fraction of beamforming gain is
//!   lost. At 7 bits it is ~1e-5 — negligible, which matches the DySPAN-2026
//!   finding that fine feedback resolution makes the privacy–utility tradeoff
//!   nearly free.
//! - **Sounding overhead.** Randomizing the NDP sounding cadence costs airtime
//!   directly; charged as a flat `sounding_overhead` fraction of throughput.
//!
//! Throughput ratio = `(1 − overhead) · C(SNR·(1−ρ)) / C(SNR)` where
//! `C(x) = log2(1 + x)` is the Shannon capacity of the data-carrying beam. The
//! comm block is never perturbed, so its geometry is intact; only the SNR is
//! nudged by the residual `ρ`.

use crate::protector::ShieldConfig;

/// A single-stream link model.
#[derive(Debug, Clone)]
pub struct LinkModel {
    /// Operating SNR of the data-carrying beam, in dB.
    pub snr_db: f64,
}

impl Default for LinkModel {
    fn default() -> Self {
        Self { snr_db: 20.0 }
    }
}

impl LinkModel {
    /// Linear SNR.
    #[must_use]
    pub fn snr_linear(&self) -> f64 {
        10f64.powf(self.snr_db / 10.0)
    }

    /// Baseline Shannon capacity (bits/s/Hz) with no shield.
    #[must_use]
    pub fn baseline_capacity(&self) -> f64 {
        (1.0 + self.snr_linear()).log2()
    }

    /// Uncompensated beamforming-gain residual from finite feedback resolution.
    #[must_use]
    pub fn beamforming_residual(shield: &ShieldConfig) -> f64 {
        if !shield.enabled {
            return 0.0;
        }
        let step = (core::f64::consts::FRAC_PI_2) / f64::from(1u32 << shield.feedback_bits);
        // Mean-square quantization error of a uniform quantizer, as a fraction
        // of unit gain. Clamp for safety at absurdly low resolutions.
        (step * step / 12.0).min(0.5)
    }

    /// Throughput ratio of the protected link versus the unshielded baseline,
    /// in `[0, 1]`.
    #[must_use]
    pub fn throughput_ratio(&self, shield: &ShieldConfig) -> f64 {
        if !shield.enabled {
            return 1.0;
        }
        let rho = Self::beamforming_residual(shield);
        let snr = self.snr_linear();
        let protected = (1.0 + snr * (1.0 - rho)).log2();
        let ratio = protected / self.baseline_capacity();
        ((1.0 - shield.sounding_overhead) * ratio).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_ratio_is_one() {
        let cfg = ShieldConfig {
            enabled: false,
            ..ShieldConfig::default()
        };
        assert!((LinkModel::default().throughput_ratio(&cfg) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fine_resolution_is_nearly_free() {
        let ratio = LinkModel::default().throughput_ratio(&ShieldConfig::default());
        assert!(ratio > 0.95, "ratio {ratio}");
        // Almost all of the (small) loss is the sounding overhead, not the
        // perturbation — consistent with the DySPAN-2026 fine-resolution result.
        assert!(ratio < 1.0);
    }

    #[test]
    fn coarse_resolution_costs_more() {
        // Lowering feedback resolution raises the residual and lowers throughput
        // — the tradeoff is real, just cheap at fine resolution.
        let fine = ShieldConfig {
            feedback_bits: 9,
            ..ShieldConfig::default()
        };
        let coarse = ShieldConfig {
            feedback_bits: 2,
            ..ShieldConfig::default()
        };
        let link = LinkModel::default();
        assert!(link.throughput_ratio(&fine) > link.throughput_ratio(&coarse));
    }
}

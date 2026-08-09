//! Link-throughput model for the protected node.
//!
//! The claim under test is "throughput stays above 95% with the shield on".
//! The model is intentionally transparent and errs toward *charging* the
//! shield, not flattering it. Three costs are charged:
//!
//! - **Beamforming residual.** The legitimate receiver shares the session key
//!   and inverts the protector's rotation, so it does not pay the rotation
//!   itself — only the residual from quantizing the extra angles at
//!   `feedback_bits` resolution. Per-angle mean-square quantization error is
//!   `Δ²/12` for step `Δ = (π/2)/2^bits`; this fraction of beamforming gain is
//!   lost. It shrinks fast with more bits.
//! - **Feedback airtime.** Reporting the angles at higher resolution costs more
//!   uplink airtime — charged as `feedback_overhead_per_bit · feedback_bits`.
//!   It grows with more bits.
//! - **Sounding overhead.** Randomizing the NDP sounding cadence costs airtime
//!   directly; a flat `sounding_overhead` fraction.
//!
//! The residual (falling) and the feedback airtime (rising) pull `feedback_bits`
//! in opposite directions, so throughput has a genuine **interior optimum** in
//! the number of feedback bits — the quantity [`crate::optimize`] searches for.
//! The optimum lands at coarse-to-moderate resolution because the receiver
//! compensates the keyed rotation, so extra bits mostly buy airtime, not gain —
//! echoing the DySPAN-2026 finding that ~3-bit feedback is near the sweet spot.
//!
//! Throughput ratio =
//! `(1 − sounding − feedback_airtime) · C(SNR·(1−ρ)) / C(SNR)` where
//! `C(x) = log2(1 + x)`. The comm block is never perturbed, so its geometry is
//! intact; only the SNR is nudged by the residual `ρ`.

use crate::protector::ShieldConfig;

/// A single-stream link model.
#[derive(Debug, Clone)]
pub struct LinkModel {
    /// Operating SNR of the data-carrying beam, in dB.
    pub snr_db: f64,
    /// Uplink airtime charged per feedback bit, as a fraction of throughput.
    /// Larger values push the throughput-optimal `feedback_bits` lower.
    pub feedback_overhead_per_bit: f64,
}

impl Default for LinkModel {
    fn default() -> Self {
        Self {
            snr_db: 20.0,
            feedback_overhead_per_bit: 0.0008,
        }
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

    /// Uplink airtime cost of reporting angles at `feedback_bits` resolution.
    #[must_use]
    pub fn feedback_airtime(&self, shield: &ShieldConfig) -> f64 {
        if !shield.enabled {
            return 0.0;
        }
        self.feedback_overhead_per_bit * f64::from(shield.feedback_bits)
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
        let capacity_ratio = (1.0 + snr * (1.0 - rho)).log2() / self.baseline_capacity();
        let airtime = shield.sounding_overhead + self.feedback_airtime(shield);
        ((1.0 - airtime) * capacity_ratio).clamp(0.0, 1.0)
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
    fn default_config_preserves_throughput() {
        let ratio = LinkModel::default().throughput_ratio(&ShieldConfig::default());
        assert!(ratio > 0.95, "ratio {ratio}");
        assert!(ratio < 1.0);
    }

    #[test]
    fn throughput_has_interior_optimum_in_bits() {
        // Very low resolution pays the residual; very high resolution pays
        // airtime. The optimum is strictly interior — neither extreme wins.
        let link = LinkModel::default();
        let at = |bits: u32| {
            link.throughput_ratio(&ShieldConfig {
                feedback_bits: bits,
                ..ShieldConfig::default()
            })
        };
        let lo = at(1);
        let hi = at(12);
        let best_bits = (1..=12)
            .max_by(|&a, &b| at(a).partial_cmp(&at(b)).unwrap())
            .unwrap();
        assert!(
            best_bits > 1 && best_bits < 12,
            "optimum at edge: {best_bits}"
        );
        assert!(at(best_bits) > lo && at(best_bits) > hi);
    }
}

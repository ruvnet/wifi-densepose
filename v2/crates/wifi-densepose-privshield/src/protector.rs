//! The VEIL protector: compliant waveform controls that hide identity.
//!
//! # What it does (and does not do)
//!
//! The protector shapes the node's **own** beamforming feedback before it goes
//! on air. It applies a per-session, key-derived **orthogonal rotation** to the
//! fine block of the report, composed from extra Givens rotations — the same
//! angle primitive the report already carries. Because the rotation is:
//!
//! - **orthogonal** → it preserves the report's energy exactly (no added
//!   transmit power, no out-of-mask emission → **not jamming**, see
//!   [`crate::compliance`]);
//! - **keyed per session** → the legitimate AP/STA, which shares the session
//!   key, inverts it and recovers the true precoder (throughput preserved,
//!   see [`crate::throughput`]);
//! - **fresh each session** → an external sniffer sees a different rotation of
//!   the identity signature every session and cannot average them back to the
//!   signature, so cross-session re-identification collapses toward chance.
//!
//! This is the shared-secret precoding idea (cf. MIMOCrypt, NSDI-adjacent work)
//! specialized to the identity-bearing fine subspace.
//!
//! # Scope limit (stated honestly)
//!
//! VEIL defends against a **third-party passive sniffer**. It does *not* hide
//! identity from the AP the node is associated with (that party holds the key
//! by construction). Protecting against a malicious AP is a different problem
//! handled by the BFLD detection layer and privacy-class policy (ADR-118/141),
//! not by this shield. VEIL never jams and never touches another station's
//! frames.

use crate::identity::BfiSample;
use crate::linalg::apply_givens;
use crate::prng::Rng;

/// Configuration of the protector.
#[derive(Debug, Clone)]
pub struct ShieldConfig {
    /// Master switch. When `false`, [`Protector::protect`] is the identity map
    /// (used to model the "shield off" baseline).
    pub enabled: bool,
    /// Number of keyed Givens rotations composed per session. Enough passes
    /// (≈ `2 × fine_dims`) approximate a Haar-random rotation of the fine
    /// block, which is what drives the attacker to chance.
    pub givens_passes: usize,
    /// Bits used to quantize each reported angle (802.11 uses 5–9). Higher
    /// resolution ⇒ smaller uncompensated residual at the legitimate receiver
    /// ⇒ smaller throughput cost. See [`crate::throughput`].
    pub feedback_bits: u32,
    /// Fractional airtime overhead from sounding-cadence randomization
    /// (jittering NDP intervals so an eavesdropper under-samples motion).
    pub sounding_overhead: f64,
}

impl Default for ShieldConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            givens_passes: 112, // 2 × 56 fine dims at the default scene
            feedback_bits: 7,
            sounding_overhead: 0.02,
        }
    }
}

/// Applies compliant waveform controls to outgoing beamforming feedback.
#[derive(Debug, Clone)]
pub struct Protector {
    cfg: ShieldConfig,
}

impl Protector {
    /// Build a protector.
    #[must_use]
    pub fn new(cfg: ShieldConfig) -> Self {
        Self { cfg }
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &ShieldConfig {
        &self.cfg
    }

    /// Build the list of `(i, j, theta)` Givens rotations for a session. The
    /// legitimate receiver derives the identical list from the shared session
    /// key and applies the inverse (negated angles, reversed order).
    fn session_rotation(&self, fine_dims: usize, session_key: u64) -> Vec<(usize, usize, f32)> {
        let mut rng = Rng::new(session_key);
        let mut ops = Vec::with_capacity(self.cfg.givens_passes);
        for _ in 0..self.cfg.givens_passes {
            // Draw a distinct coordinate pair in the fine block.
            let i = (rng.next_u64() as usize) % fine_dims;
            let mut j = (rng.next_u64() as usize) % fine_dims;
            if j == i {
                j = (j + 1) % fine_dims;
            }
            let theta = rng.next_range(0.0, core::f32::consts::TAU);
            ops.push((i, j, theta));
        }
        ops
    }

    /// Protect an outgoing report for the given session. When the shield is
    /// disabled this clones the input unchanged.
    #[must_use]
    pub fn protect(&self, sample: &BfiSample, session_key: u64) -> BfiSample {
        let mut out = sample.clone();
        if !self.cfg.enabled {
            return out;
        }
        let fine_dims = out.fine().len();
        let ops = self.session_rotation(fine_dims, session_key);
        let fine = out.fine_mut();
        for (i, j, theta) in ops {
            apply_givens(fine, i, j, theta);
        }
        out
    }

    /// Recover the true report at the legitimate receiver, which shares the
    /// session key. Applies the inverse rotation. Used to demonstrate that the
    /// transform is reversible for the authorized party (the basis of the
    /// throughput claim), not part of the attacker's world.
    #[must_use]
    pub fn recover(&self, sample: &BfiSample, session_key: u64) -> BfiSample {
        let mut out = sample.clone();
        if !self.cfg.enabled {
            return out;
        }
        let fine_dims = out.fine().len();
        let ops = self.session_rotation(fine_dims, session_key);
        let fine = out.fine_mut();
        for (i, j, theta) in ops.into_iter().rev() {
            apply_givens(fine, i, j, -theta);
        }
        out
    }
}

/// A minimal detector for unsolicited sensing activity. In a deployment this
/// watches the rate of NDP/sensing-sounding solicitations; here it exposes the
/// decision rule so the control plane (ADR-280) can engage the shield only when
/// sensing is actually observed, rather than perturbing continuously.
#[derive(Debug, Clone)]
pub struct SensingDetector {
    /// Solicitations per second above which the shield engages.
    pub threshold_hz: f32,
}

impl Default for SensingDetector {
    fn default() -> Self {
        Self { threshold_hz: 5.0 }
    }
}

impl SensingDetector {
    /// Should the shield engage given the observed solicitation rate?
    #[must_use]
    pub fn should_engage(&self, observed_hz: f32) -> bool {
        observed_hz >= self.threshold_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Channel, SceneConfig};
    use crate::linalg::{dist_sq, norm};

    #[test]
    fn protection_preserves_energy() {
        let ch = Channel::new(SceneConfig::default());
        let s = ch.observe(0, b"enroll", 1);
        let p = Protector::new(ShieldConfig::default());
        let out = p.protect(&s, 12345);
        assert!((norm(&s.values) - norm(&out.values)).abs() < 1e-3);
    }

    #[test]
    fn protection_leaves_comm_block_untouched() {
        let ch = Channel::new(SceneConfig::default());
        let s = ch.observe(0, b"enroll", 1);
        let p = Protector::new(ShieldConfig::default());
        let out = p.protect(&s, 999);
        assert_eq!(s.comm(), out.comm());
    }

    #[test]
    fn protection_scrambles_fine_block() {
        let ch = Channel::new(SceneConfig::default());
        let s = ch.observe(0, b"enroll", 1);
        let p = Protector::new(ShieldConfig::default());
        let out = p.protect(&s, 42);
        assert!(dist_sq(s.fine(), out.fine()).sqrt() > 0.5);
    }

    #[test]
    fn legitimate_receiver_recovers() {
        let ch = Channel::new(SceneConfig::default());
        let s = ch.observe(0, b"enroll", 1);
        let p = Protector::new(ShieldConfig::default());
        let out = p.protect(&s, 7);
        let back = p.recover(&out, 7);
        assert!(dist_sq(s.fine(), back.fine()).sqrt() < 1e-2);
    }

    #[test]
    fn disabled_shield_is_identity() {
        let ch = Channel::new(SceneConfig::default());
        let s = ch.observe(0, b"enroll", 1);
        let cfg = ShieldConfig {
            enabled: false,
            ..ShieldConfig::default()
        };
        let p = Protector::new(cfg);
        assert_eq!(s, p.protect(&s, 7));
    }

    #[test]
    fn detector_engages_above_threshold() {
        let d = SensingDetector::default();
        assert!(d.should_engage(10.0));
        assert!(!d.should_engage(1.0));
    }
}

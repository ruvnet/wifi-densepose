//! Synthetic beamforming-feedback model. **SYNTHETIC data only.**
//!
//! Nothing here is captured from a real radio. The model is a deliberately
//! simple, physically-motivated abstraction of a flattened 802.11 compressed
//! beamforming report, chosen so the attacker/protector dynamics are
//! transparent and the experiment is byte-reproducible. It is *not* a channel
//! simulator and its accuracy numbers describe this model, not real hardware
//! (per CLAUDE.md: results are `SYNTHETIC`, reproduced by `cargo test`).
//!
//! # The two-subspace abstraction
//!
//! A beamforming report is split into two orthogonal blocks:
//!
//! - **Comm block** (`comm_dims` leading coordinates) — the dominant beam
//!   direction the AP actually uses to steer data. It varies per session with
//!   position/traffic and carries **no** identity. Link throughput rides here.
//! - **Fine block** (the remainder) — the fine cross-subcarrier phase
//!   structure. This is where a re-identification attacker's signal lives: the
//!   literature (BFId, CCS 2025) shows the *stable* fine structure re-IDs
//!   people. Communication barely uses it.
//!
//! Each identity owns a fixed, near-orthogonal signature vector in the fine
//! block. A session observation is `signature + environmental nuisance`; the
//! comm block is fresh per session. This is the honest crux of the whole
//! design: **identity leakage and data throughput live in (mostly) separable
//! subspaces**, so a transform can wreck the former while sparing the latter.

use crate::linalg::set_norm_inplace;
use crate::prng::{derive_key, Rng};

/// A flattened compressed-beamforming-report vector, split into a comm block
/// and a fine block.
#[derive(Debug, Clone, PartialEq)]
pub struct BfiSample {
    /// The full report: `comm_dims` comm coordinates followed by fine ones.
    pub values: Vec<f32>,
    /// Number of leading coordinates that form the comm (data-carrying) block.
    pub comm_dims: usize,
}

impl BfiSample {
    /// Comm (data-carrying) block.
    #[must_use]
    pub fn comm(&self) -> &[f32] {
        &self.values[..self.comm_dims]
    }

    /// Fine (identity-bearing) block.
    #[must_use]
    pub fn fine(&self) -> &[f32] {
        &self.values[self.comm_dims..]
    }

    /// Mutable fine block — the only part the protector is allowed to rotate.
    pub fn fine_mut(&mut self) -> &mut [f32] {
        &mut self.values[self.comm_dims..]
    }
}

/// Configuration of the synthetic scene.
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// Total report dimension.
    pub dim: usize,
    /// Leading coordinates forming the comm block.
    pub comm_dims: usize,
    /// Number of distinct identities (candidates). Chance level is `1/identities`.
    pub identities: usize,
    /// L2 norm of each identity's fine-block signature.
    pub signature_norm: f32,
    /// Std-dev of per-session environmental nuisance added to the fine block.
    pub env_sigma: f32,
    /// L2 norm of the fresh per-session comm-block beam.
    pub beam_amplitude: f32,
    /// Master seed. All keys derive from this; nothing touches OS entropy.
    pub seed: u64,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            comm_dims: 8,
            identities: 16,
            signature_norm: 1.0,
            env_sigma: 0.15,
            beam_amplitude: 0.30,
            seed: 0x5EED_1BF1,
        }
    }
}

impl SceneConfig {
    /// Ideal chance-level accuracy, `1 / identities`.
    #[must_use]
    pub fn chance_level(&self) -> f32 {
        1.0 / self.identities as f32
    }

    /// Length of the fine block.
    #[must_use]
    pub fn fine_dims(&self) -> usize {
        self.dim - self.comm_dims
    }
}

/// Synthetic channel: turns `(identity, session)` into a [`BfiSample`].
#[derive(Debug, Clone)]
pub struct Channel {
    cfg: SceneConfig,
    /// Precomputed per-identity fine-block signatures.
    signatures: Vec<Vec<f32>>,
}

impl Channel {
    /// Build the channel, drawing each identity's stable signature.
    #[must_use]
    pub fn new(cfg: SceneConfig) -> Self {
        let fine = cfg.fine_dims();
        let mut signatures = Vec::with_capacity(cfg.identities);
        for id in 0..cfg.identities {
            let mut rng = Rng::new(derive_key(cfg.seed, b"signature", id as u64, 0));
            let mut s: Vec<f32> = (0..fine).map(|_| rng.next_gaussian()).collect();
            set_norm_inplace(&mut s, cfg.signature_norm);
            signatures.push(s);
        }
        Self { cfg, signatures }
    }

    /// The scene configuration.
    #[must_use]
    pub fn config(&self) -> &SceneConfig {
        &self.cfg
    }

    /// The stable fine-block signature of `identity` (the thing an attacker
    /// wants and the thing the shield must hide).
    #[must_use]
    pub fn signature(&self, identity: usize) -> &[f32] {
        &self.signatures[identity]
    }

    /// Observe the unprotected report for `identity` in the given session under
    /// `phase` (an experiment stage label, e.g. `b"enroll"` / `b"test"`, so the
    /// same session index draws independent nuisance across stages).
    #[must_use]
    pub fn observe(&self, identity: usize, phase: &[u8], session: u64) -> BfiSample {
        let cfg = &self.cfg;
        let mut values = vec![0.0f32; cfg.dim];

        // Comm block: fresh per session, identity-independent. This is the
        // data-carrying dominant beam — it holds no re-ID information.
        let mut brng = Rng::new(derive_key(cfg.seed, b"beam", session, phase[0] as u64));
        for v in values[..cfg.comm_dims].iter_mut() {
            *v = brng.next_gaussian();
        }
        set_norm_inplace(&mut values[..cfg.comm_dims], cfg.beam_amplitude);

        // Fine block: stable identity signature + per-session nuisance.
        let mut nrng = Rng::new(derive_key(cfg.seed, phase, identity as u64, session));
        let sig = &self.signatures[identity];
        for (v, s) in values[cfg.comm_dims..].iter_mut().zip(sig) {
            *v = s + cfg.env_sigma * nrng.next_gaussian();
        }

        BfiSample {
            values,
            comm_dims: cfg.comm_dims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dist_sq, norm};

    #[test]
    fn signatures_are_well_separated() {
        let ch = Channel::new(SceneConfig::default());
        // Distinct identities' signatures are near-orthogonal in high-dim,
        // so pairwise distance is large relative to env noise.
        let d = dist_sq(ch.signature(0), ch.signature(1)).sqrt();
        assert!(d > 1.0, "signatures too close: {d}");
    }

    #[test]
    fn signature_norm_matches_config() {
        let ch = Channel::new(SceneConfig::default());
        assert!((norm(ch.signature(3)) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn observation_is_deterministic() {
        let ch = Channel::new(SceneConfig::default());
        assert_eq!(ch.observe(2, b"enroll", 5), ch.observe(2, b"enroll", 5));
    }

    #[test]
    fn same_session_different_phase_differs() {
        let ch = Channel::new(SceneConfig::default());
        assert_ne!(ch.observe(2, b"enroll", 5), ch.observe(2, b"test", 5));
    }
}

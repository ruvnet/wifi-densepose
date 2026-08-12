//! Deterministic, WASM-safe pseudo-random generator.
//!
//! VEIL never draws from OS entropy: every stochastic quantity in the
//! experiment (identity signatures, environmental nuisance, per-session
//! precoder rotations) seeds from an explicit `u64`. Same seed in → same
//! bytes out, on any platform including `wasm32-unknown-unknown`. This is
//! what makes [`crate::proof`] a byte-stable witness rather than a flaky
//! statistical assertion.
//!
//! The core is SplitMix64 (Steele, Lea & Flood 2014) — a well-mixed
//! finalizer that is more than adequate for synthetic-data generation and
//! keyed subspace rotation. It is **not** a cryptographic RNG and must not
//! be used to derive real key material; in a deployment the per-session
//! rotation key comes from the negotiated link secret, not from this PRNG.

/// A deterministic SplitMix64 stream.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the stream. Distinct seeds yield independent streams.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Next raw 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)` using the top 24 mantissa bits.
    pub fn next_f32(&mut self) -> f32 {
        // 24 bits of precision keeps the value exactly representable.
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Standard-normal `f32` via the Box–Muller transform.
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}

/// FNV-1a 64-bit hash — a dependency-free, deterministic byte folder used to
/// derive per-session keys from `(scene_seed, phase, index)` tuples and to
/// build the [`crate::proof`] witness. Not cryptographic.
#[must_use]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Fold a label and two indices into a stable `u64` key.
#[must_use]
pub fn derive_key(scene_seed: u64, label: &[u8], a: u64, b: u64) -> u64 {
    let mut buf = Vec::with_capacity(label.len() + 24);
    buf.extend_from_slice(&scene_seed.to_le_bytes());
    buf.extend_from_slice(label);
    buf.extend_from_slice(&a.to_le_bytes());
    buf.extend_from_slice(&b.to_le_bytes());
    fnv1a_64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn gaussian_mean_near_zero() {
        let mut r = Rng::new(9);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| f64::from(r.next_gaussian())).sum::<f64>() / f64::from(n);
        assert!(mean.abs() < 0.02, "mean {mean} not near 0");
    }
}

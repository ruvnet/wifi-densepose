//! Deterministic top-K **coverage** harness for the RaBitQ sketch
//! (ADR-084 acceptance bar / ADR-156 §8 Pass-2 measurement).
//!
//! Single source of truth for the coverage number quoted in ADR-084 and
//! ADR-156: both the in-crate regression test (`pass2_coverage_not_worse_…`)
//! and the criterion bench (`benches/sketch_bench.rs`) call into here, so they
//! can never silently measure different things.
//!
//! **Coverage** is defined exactly as in ADR-084:
//!
//! > the Top-K candidate set chosen by the sketch must contain **≥ 90%** of the
//! > candidates the full-float pass would have picked.
//!
//! i.e. `coverage = |sketch_topK ∩ float_topK| / K`, averaged over a set of
//! queries. The float top-K (squared-euclidean — AETHER's actual metric) is the
//! ground truth; the sketch top-K is a *candidate* set, so in practice a system
//! over-fetches `C ≥ K` sketch candidates and refines. We measure at
//! `candidate_k == K` (the strict bar) by default; the bench also reports an
//! over-fetch curve.
//!
//! # The synthetic distribution — and why it is *anisotropic*
//!
//! Pure 1-bit sign quantization (Pass 1) is near-optimal on **isotropic,
//! zero-centred** embeddings — on such data a rotation barely moves the number,
//! so testing rotation there proves nothing. ADR-084's "Open questions" and
//! ADR-156 §8 both flag the *anisotropic / correlated* case (skewed CSI
//! spectrogram embeddings) as exactly where the rotation is supposed to earn
//! its keep. So [`make_anisotropic_embedding`] deliberately builds **correlated,
//! axis-aligned, non-isotropic** vectors: a few dominant low-frequency factors
//! shared across many coordinates (heavy coordinate correlation) plus a small
//! per-dim offset that biases signs — the structure that defeats raw
//! sign-quantization and that a randomized rotation is designed to fix. Every
//! value derives from a seed via SplitMix64, so the whole harness is
//! reproducible bit-for-bit.

use crate::{Rotation, SketchBank};

/// SplitMix64 step — reproducible PRNG for fixture generation (dependency-free).
#[inline]
fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform `f32` in `[0, 1)` from the PRNG state.
#[inline]
fn unif01(state: &mut u64) -> f32 {
    let r = split_mix64(state);
    // top 24 bits → [0,1)
    ((r >> 40) as f32) / ((1u64 << 24) as f32)
}

/// A standard-normal-ish `f32` via Box–Muller from two uniforms. Deterministic.
#[inline]
fn gauss(state: &mut u64) -> f32 {
    let u1 = unif01(state).max(1e-7); // avoid log(0)
    let u2 = unif01(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Fixed **anisotropic axis scale** for coordinate `i` of `dim`.
///
/// A learned embedding space is not isotropic: a handful of axes carry most of
/// the variance and the rest are near-flat. We model that with a smoothly
/// decaying per-axis scale (≈10× spread between the most- and least-energetic
/// axes). This axis-aligned imbalance is exactly what a 1-bit sign sketch
/// handles poorly (the low-variance axes' sign bits are noise) and exactly what
/// a randomized rotation re-balances (it spreads the variance across all axes so
/// every sign bit carries comparable information). The scale depends only on the
/// coordinate index, so it is the *same fixed geometry* for every vector.
#[inline]
fn axis_scale(i: usize, dim: usize) -> f32 {
    let t = i as f32 / dim.max(1) as f32;
    // exp decay from ~3.0 down to ~0.3 → ~10× anisotropy.
    3.0 * (-2.3 * t).exp() + 0.3
}

/// Build the **planted-cluster** fixture: `n_clusters` random centres in the
/// anisotropic space. Returned as raw centres (pre-scale); callers add scale +
/// intra-cluster noise. Deterministic from `seed`.
fn cluster_centres(dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n_clusters)
        .map(|c| {
            let mut s = seed ^ 0xC0FFEE_u64.wrapping_mul(c as u64 + 1);
            (0..dim).map(|_| gauss(&mut s)).collect()
        })
        .collect()
}

/// One embedding = its cluster centre + small intra-cluster noise, then the
/// fixed anisotropic axis scale, then a small off-centre bias. This makes the
/// **cosine top-K meaningful** (same-cluster members are genuine near-neighbours,
/// not random-noise ties), while keeping the space anisotropic so the rotation
/// has something real to fix.
fn realize(centre: &[f32], dim: usize, noise: f32, vec_seed: u64) -> Vec<f32> {
    let mut s = vec_seed ^ 0x5151_5151_5151_5151;
    (0..dim)
        .map(|i| {
            let jitter = gauss(&mut s) * noise;
            let bias = ((i % 11) as f32 - 5.0) * 0.05;
            axis_scale(i, dim) * (centre[i] + jitter) + bias
        })
        .collect()
}

/// Cosine distance `1 - cos(a,b)` — the metric a sign sketch approximates
/// (hamming over sign bits is a monotone estimate of the angle between vectors).
/// This is the correct full-float ground truth for top-K *coverage*: the sketch
/// is an angular sensor, so we grade it against the angular full-float ranking,
/// per ADR-084's `float_cosine` baseline.
#[inline]
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < f32::EPSILON {
        1.0
    } else {
        1.0 - dot / denom
    }
}

/// Full-float cosine top-K ids (ground truth), ascending by cosine distance.
fn float_topk(bank: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = bank
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

/// Parameters for a coverage measurement, documented in the report.
#[derive(Debug, Clone, Copy)]
pub struct CoverageParams {
    /// Embedding dimension.
    pub dim: usize,
    /// Number of stored vectors in the bank (N).
    pub n: usize,
    /// Number of distinct query vectors averaged over.
    pub n_queries: usize,
    /// True top-K size (the bar's K).
    pub k: usize,
    /// Sketch candidate-set size to compare against the float top-K. Equal to
    /// `k` for the strict ADR-084 bar; `> k` models over-fetch + refine.
    pub candidate_k: usize,
    /// Number of planted clusters. Same-cluster vectors are genuine near
    /// neighbours, so the cosine top-K is *meaningful* (not random-noise ties).
    pub n_clusters: usize,
    /// Intra-cluster Gaussian jitter (relative to unit-variance centres). Small
    /// jitter → tight, easily-recovered clusters; larger → harder top-K.
    pub noise: f32,
    /// Master seed (the whole fixture derives from this).
    pub seed: u64,
}

impl CoverageParams {
    /// The canonical AETHER-shape fixture used for the ADR-quoted numbers:
    /// 128-d, planted clusters, modest intra-cluster jitter. Override fields
    /// with struct-update syntax (`CoverageParams { candidate_k: 32, ..base }`).
    pub fn aether_default(seed: u64) -> Self {
        Self {
            dim: 128,
            n: 2048,
            n_queries: 128,
            k: 8,
            candidate_k: 8,
            n_clusters: 64,
            noise: 0.35,
            seed,
        }
    }
}

/// Result of a coverage measurement.
#[derive(Debug, Clone, Copy)]
pub struct CoverageResult {
    /// Mean coverage in `[0, 1]` (fraction of float top-K found in the sketch
    /// candidate set), averaged over queries.
    pub coverage: f64,
}

/// Measure mean top-K coverage of the **Pass-1** (no rotation) sketch against
/// the full-float top-K, on the anisotropic synthetic distribution.
pub fn measure_pass1(p: CoverageParams) -> CoverageResult {
    measure_inner(p, None)
}

/// Measure mean top-K coverage of the **Pass-2** (rotated) sketch against the
/// full-float top-K, on the anisotropic synthetic distribution. `rotation_seed`
/// fixes the rotation (index and query share it — that is the contract).
pub fn measure_pass2(p: CoverageParams, rotation_seed: u64) -> CoverageResult {
    let rot = Rotation::new(rotation_seed, p.dim);
    measure_inner(p, Some(rot))
}

/// Measure mean top-K coverage of a **multi-bit (Pass-3)** rotated sketch:
/// `bits` bits per dimension instead of 1, ranked by L1 distance over the
/// per-dim codes (the natural multi-bit generalization of hamming). This is the
/// "Multi-bit / Extended RaBitQ" half of ADR-156 §8 — measured here as an
/// experiment to decide whether a full `MultiBitSketch` type is worth building.
///
/// Quantization: rotate (Pass-2 frame), then map each rotated coordinate through
/// a uniform mid-rise scalar quantizer with `2^bits` levels over a fixed
/// symmetric range `[-RANGE, RANGE]` (RANGE chosen from the rotated-coord scale).
/// `bits == 1` reduces to sign-quantization (sanity: should match Pass-2 within
/// quantizer-boundary noise). Memory cost is `bits×` the 1-bit sketch.
///
/// Returns the measured coverage; the caller reports the bit/coverage tradeoff.
pub fn measure_multibit(p: CoverageParams, rotation_seed: u64, bits: u32) -> CoverageResult {
    assert!((1..=8).contains(&bits), "bits must be in 1..=8");
    let rot = Rotation::new(rotation_seed, p.dim);
    let levels = 1u32 << bits; // 2^bits codes per dim
    // Rotated AETHER-shape coords after the normalized FHT sit roughly in
    // [-RANGE, RANGE]; clamp out-of-range to the end codes. RANGE picked to
    // cover ~99% of the rotated-coord magnitude on this fixture (empirically
    // ~3.0 after the 1/√m normalization).
    const RANGE: f32 = 3.0;
    let quantize = move |v: &[f32]| -> Vec<u16> {
        rot.apply(v)
            .iter()
            .map(|&x| {
                let t = ((x + RANGE) / (2.0 * RANGE)).clamp(0.0, 1.0); // → [0,1]
                let code = (t * (levels - 1) as f32).round() as u32;
                code.min(levels - 1) as u16
            })
            .collect()
    };
    // L1 distance over per-dim codes.
    let l1 = |a: &[u16], b: &[u16]| -> u32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs())
            .sum()
    };

    let float_bank = make_fixture(p);
    let centres = cluster_centres(p.dim, p.n_clusters.max(1), p.seed);
    let coded_bank: Vec<Vec<u16>> = float_bank.iter().map(|v| quantize(v)).collect();

    let mut total = 0.0f64;
    for q in 0..p.n_queries {
        let c = q % p.n_clusters.max(1);
        let qv = realize(
            &centres[c],
            p.dim,
            p.noise,
            p.seed ^ 0xDEAD_0000_0000 ^ (q as u64).wrapping_mul(0x2545_F491),
        );
        let truth = float_topk(&float_bank, &qv, p.k);
        let qc = quantize(&qv);
        // top candidate_k by L1 over codes.
        let mut scored: Vec<(u32, u32)> = coded_bank
            .iter()
            .enumerate()
            .map(|(i, code)| (i as u32, l1(&qc, code)))
            .collect();
        scored.sort_by_key(|&(_, d)| d);
        scored.truncate(p.candidate_k);
        let cand_ids: std::collections::HashSet<u32> =
            scored.into_iter().map(|(id, _)| id).collect();
        let hit = truth.iter().filter(|id| cand_ids.contains(id)).count();
        total += hit as f64 / p.k as f64;
    }
    CoverageResult {
        coverage: total / p.n_queries as f64,
    }
}

/// Build the deterministic float bank for `p`: `p.n` vectors, each assigned to
/// one of `p.n_clusters` planted clusters (round-robin), realized as
/// `centre + jitter` under the fixed anisotropic axis scale. Returned with the
/// cluster id of each vector so queries can be drawn from the same clusters.
pub fn make_fixture(p: CoverageParams) -> Vec<Vec<f32>> {
    let centres = cluster_centres(p.dim, p.n_clusters.max(1), p.seed);
    (0..p.n)
        .map(|i| {
            let c = i % p.n_clusters.max(1);
            realize(&centres[c], p.dim, p.noise, p.seed ^ (i as u64).wrapping_mul(0x9E37))
        })
        .collect()
}

fn measure_inner(p: CoverageParams, rotation: Option<Rotation>) -> CoverageResult {
    const SV: u16 = 1;
    // Float bank (ground truth) + sketch bank from the SAME vectors, so the
    // only variable is float-vs-sketch (and Pass-1-vs-Pass-2).
    let float_bank = make_fixture(p);
    let centres = cluster_centres(p.dim, p.n_clusters.max(1), p.seed);

    let mut bank = match &rotation {
        Some(r) => SketchBank::with_rotation(r.clone()),
        None => SketchBank::new(),
    };
    for (i, v) in float_bank.iter().enumerate() {
        // Use the bank's rotation policy for both Pass-1 and Pass-2 uniformly.
        bank.insert_embedding(i as u32, v, SV)
            .expect("schema-locked insert");
    }

    let mut total = 0.0f64;
    for q in 0..p.n_queries {
        // Each query is a fresh draw from a planted cluster (disjoint seed
        // range from the bank), so it HAS genuine same-cluster neighbours in
        // the bank — a meaningful top-K, not random-noise ties.
        let c = q % p.n_clusters.max(1);
        let qv = realize(
            &centres[c],
            p.dim,
            p.noise,
            p.seed ^ 0xDEAD_0000_0000 ^ (q as u64).wrapping_mul(0x2545_F491),
        );
        let truth = float_topk(&float_bank, &qv, p.k);
        let cand = bank
            .topk_embedding(&qv, SV, p.candidate_k)
            .expect("schema match");
        let cand_ids: std::collections::HashSet<u32> = cand.into_iter().map(|(id, _)| id).collect();
        let hit = truth.iter().filter(|id| cand_ids.contains(id)).count();
        total += hit as f64 / p.k as f64;
    }
    CoverageResult {
        coverage: total / p.n_queries as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_clusters_give_high_coverage_with_overfetch() {
        // Sanity / regression: on tight clusters with enough over-fetch the
        // sketch MUST recover essentially all of the float cosine top-K — this
        // both proves the harness is correct (a broken topk gives ~random here)
        // and pins the cluster structure as meaningful. Catches the heap
        // inversion bug found during this work (which made this ~6%).
        let p = CoverageParams {
            n: 1024,
            n_queries: 64,
            n_clusters: 64,
            noise: 0.1,
            candidate_k: 64,
            ..CoverageParams::aether_default(0x1111)
        };
        let cov = measure_pass1(p).coverage;
        assert!(
            cov > 0.95,
            "tight clusters + 8× over-fetch should recover >95% of top-K, got {:.3}",
            cov
        );
    }

    #[test]
    fn multibit_tradeoff_report() {
        // ADR-156 §8 "Multi-bit / Extended RaBitQ" measurement: bit/coverage
        // tradeoff at the STRICT bar (candidate_k == K). Reports b=1..4 bits
        // per dim alongside Pass-1 / Pass-2 (1-bit) baselines. Run with
        // --nocapture to see the table.
        let base = CoverageParams::aether_default(0xAD00_0084);
        let rot_seed = 0x5EED_C0DE_1234_5678u64;
        let p1 = measure_pass1(base).coverage;
        let p2 = measure_pass2(base, rot_seed).coverage;
        println!("\n=== ADR-156 §8 multi-bit tradeoff (strict candidate_k=K={}) ===", base.k);
        println!("dim={} N={} clusters={} noise={}  bar=90%", base.dim, base.n, base.n_clusters, base.noise);
        println!("  Pass1 (no rot, 1-bit)      : {:6.2}%", p1 * 100.0);
        println!("  Pass2 (rot, 1-bit)         : {:6.2}%", p2 * 100.0);
        for bits in 1..=4u32 {
            let cov = measure_multibit(base, rot_seed, bits).coverage;
            let bytes_per_vec = base.dim * bits as usize / 8;
            println!(
                "  Pass3 (rot, {bits}-bit, {bytes_per_vec:>3} B/vec): {:6.2}%  {}",
                cov * 100.0,
                if cov >= 0.90 { "≥90%" } else { "" }
            );
        }
        println!("=================================================================\n");
        assert!((0.0..=1.0).contains(&p1));
    }

    #[test]
    fn multibit_1bit_matches_pass2_approx() {
        // Sanity: 1-bit multi-bit quantization is essentially sign-quantization,
        // so its coverage should track Pass-2 (rotated 1-bit) closely. (Not
        // exact: the mid-rise quantizer's 0/1 boundary is at the RANGE midpoint,
        // which equals the sign boundary, so they should match very closely.)
        let p = CoverageParams {
            n: 256,
            n_queries: 16,
            n_clusters: 16,
            ..CoverageParams::aether_default(0x55)
        };
        let rot_seed = 0xABCDu64;
        let p2 = measure_pass2(p, rot_seed).coverage;
        let mb1 = measure_multibit(p, rot_seed, 1).coverage;
        assert!(
            (p2 - mb1).abs() < 0.05,
            "1-bit multibit {mb1:.3} should track Pass-2 {p2:.3}"
        );
    }

    #[test]
    fn fixture_is_deterministic() {
        let p = CoverageParams::aether_default(12345);
        let a = make_fixture(p);
        let b = make_fixture(p);
        assert_eq!(a, b);
        assert_eq!(a.len(), p.n);
        assert_eq!(a[0].len(), p.dim);
        let c = make_fixture(CoverageParams::aether_default(12346));
        assert_ne!(a[0], c[0]);
    }

    #[test]
    fn coverage_harness_runs_and_is_in_range() {
        // Small fixed fixture — fast, deterministic, in [0,1].
        let p = CoverageParams {
            n: 256,
            n_queries: 16,
            n_clusters: 16,
            ..CoverageParams::aether_default(0xABCD)
        };
        let c1 = measure_pass1(p);
        let c2 = measure_pass2(p, 0x1234_5678);
        assert!((0.0..=1.0).contains(&c1.coverage));
        assert!((0.0..=1.0).contains(&c2.coverage));
        // Determinism: same params → same number.
        assert_eq!(measure_pass1(p).coverage, c1.coverage);
        assert_eq!(measure_pass2(p, 0x1234_5678).coverage, c2.coverage);
    }
}

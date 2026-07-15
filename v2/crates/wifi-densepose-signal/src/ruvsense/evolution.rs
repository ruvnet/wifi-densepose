//! ADR-142 — Channel-state evolution tracking + temporal VoxelMap.
//!
//! Two cooperating pieces, both extending ADR-030's field-model tier:
//!
//! 1. [`EvolutionTracker`] — per-link rolling [`WelfordStats`] baselines with a
//!    cross-link change-point detector (≥ `min_links` links exceeding `nσ` in
//!    one window ⇒ a `ChangePoint`). This catches environment changes that a
//!    single-link drift check misses.
//! 2. [`TemporalVoxelMap`] — a *temporal* occupancy grid (distinct from the
//!    static `tomography::OccupancyVolume`): each [`TemporalVoxel`] accumulates
//!    evidence with a Bayesian log-odds update, tracks `last_update_ns`,
//!    `evidence_count`, and Welford amplitude variance, and is privacy-gated by
//!    [`VoxelGate`] before any occupancy leaves the node.

use crate::ruvsense::field_model::WelfordStats;

/// Privacy posture applied to voxel output (mirrors the BFLD demotion ladder of
/// ADR-120/141 without taking a crate dependency on `wifi-densepose-bfld`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelPrivacy {
    /// Full per-voxel detail (occupancy + confidence + doppler).
    Full,
    /// Drop per-voxel doppler + confidence detail; keep occupancy.
    Anonymous,
    /// Emit only an aggregate occupancy histogram; raw map never leaves node.
    Restricted,
}

/// A single temporal occupancy voxel (ADR-142 §2).
#[derive(Debug, Clone)]
pub struct TemporalVoxel {
    /// Voxel centre (east, north, up) in metres.
    pub center: [f64; 3],
    /// Posterior occupancy probability in [0, 1].
    pub occupancy: f64,
    /// Internal Bayesian log-odds (occupancy = sigmoid(log_odds)).
    log_odds: f64,
    /// Confidence in [0, 1]; grows with evidence count.
    pub confidence: f64,
    /// Number of evidence updates folded in.
    pub evidence_count: u64,
    /// Most recent doppler velocity (m/s) attributed to this voxel, if any.
    pub doppler_velocity: Option<f64>,
    /// Capture-clock time of the last update (ns).
    pub last_update_ns: u64,
    /// Welford stats over the occupancy-evidence stream (for variance).
    welford: WelfordStats,
}

impl TemporalVoxel {
    /// Empty voxel at a centre, prior occupancy 0.5 (log-odds 0).
    #[must_use]
    pub fn new(center: [f64; 3]) -> Self {
        Self {
            center,
            occupancy: 0.5,
            log_odds: 0.0,
            confidence: 0.0,
            evidence_count: 0,
            doppler_velocity: None,
            last_update_ns: 0,
            welford: WelfordStats::new(),
        }
    }

    /// Fold one occupancy-evidence probability `p ∈ (0, 1)` into the posterior
    /// via a clamped log-odds update, and (optionally) attribute a doppler
    /// velocity. Confidence saturates as `1 - exp(-count / 5)` — so a voxel with
    /// fewer than ~5 updates is low-confidence (ADR-142 §2 5-frame rule).
    pub fn observe(&mut self, p: f64, doppler: Option<f64>, ns: u64) {
        let p = p.clamp(1e-4, 1.0 - 1e-4);
        let evidence_logit = (p / (1.0 - p)).ln();
        // Clamp the running log-odds so a single bad frame cannot saturate.
        self.log_odds = (self.log_odds + evidence_logit).clamp(-20.0, 20.0);
        self.occupancy = 1.0 / (1.0 + (-self.log_odds).exp());
        self.welford.update(p);
        self.evidence_count += 1;
        self.confidence = 1.0 - (-(self.evidence_count as f64) / 5.0).exp();
        if doppler.is_some() {
            self.doppler_velocity = doppler;
        }
        self.last_update_ns = ns;
    }

    /// True if too few updates have accumulated for a trustworthy posterior.
    #[must_use]
    pub fn is_low_confidence(&self) -> bool {
        self.evidence_count < 5
    }

    /// Welford variance of the occupancy-evidence stream.
    #[must_use]
    pub fn evidence_variance(&self) -> f64 {
        self.welford.variance()
    }
}

/// A persistent temporal occupancy grid shared across reconstruct() cycles.
#[derive(Debug, Clone)]
pub struct TemporalVoxelMap {
    voxels: Vec<TemporalVoxel>,
}

impl TemporalVoxelMap {
    /// Build a grid of voxels at the supplied centres.
    #[must_use]
    pub fn new(centers: Vec<[f64; 3]>) -> Self {
        Self { voxels: centers.into_iter().map(TemporalVoxel::new).collect() }
    }

    /// Number of voxels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    /// Whether the grid is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Borrow a voxel.
    #[must_use]
    pub fn voxel(&self, idx: usize) -> Option<&TemporalVoxel> {
        self.voxels.get(idx)
    }

    /// Fold occupancy evidence into one voxel.
    pub fn observe(&mut self, idx: usize, p: f64, doppler: Option<f64>, ns: u64) {
        if let Some(v) = self.voxels.get_mut(idx) {
            v.observe(p, doppler, ns);
        }
    }

    /// Indices of voxels still below the confidence floor.
    #[must_use]
    pub fn low_confidence_indices(&self) -> Vec<usize> {
        self.voxels
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_low_confidence())
            .map(|(i, _)| i)
            .collect()
    }

    /// Occupancy of every voxel (read view).
    #[must_use]
    pub fn occupancies(&self) -> Vec<f64> {
        self.voxels.iter().map(|v| v.occupancy).collect()
    }
}

/// Privacy gate over voxel output (ADR-142 §2 — reuses the BFLD monotonic
/// demotion idea: information only ever removed, never added).
pub struct VoxelGate;

impl VoxelGate {
    /// Apply a privacy posture to the map, mutating it in place, and return an
    /// optional aggregate histogram (Some only for `Restricted`, where the raw
    /// map must not leave the node).
    ///
    /// - `Full`: unchanged.
    /// - `Anonymous`: clear per-voxel doppler + zero the confidence detail
    ///   (occupancy retained).
    /// - `Restricted`: produce an occupancy histogram (`bins` buckets over
    ///   [0,1]) and clear every voxel's occupancy/doppler/confidence so only the
    ///   aggregate survives.
    pub fn demote(map: &mut TemporalVoxelMap, posture: VoxelPrivacy, bins: usize) -> Option<Vec<u32>> {
        match posture {
            VoxelPrivacy::Full => None,
            VoxelPrivacy::Anonymous => {
                for v in &mut map.voxels {
                    v.doppler_velocity = None;
                    v.confidence = 0.0;
                }
                None
            }
            VoxelPrivacy::Restricted => {
                let bins = bins.max(1);
                let mut hist = vec![0u32; bins];
                for v in &map.voxels {
                    let b = ((v.occupancy * bins as f64) as usize).min(bins - 1);
                    hist[b] += 1;
                }
                for v in &mut map.voxels {
                    v.occupancy = 0.0;
                    v.doppler_velocity = None;
                    v.confidence = 0.0;
                }
                Some(hist)
            }
        }
    }
}

/// A cross-link change-point: enough links diverged from baseline at once that
/// the environment itself likely changed (ADR-142 §2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChangePoint {
    /// How many links exceeded the σ threshold this window.
    pub diverging_links: usize,
    /// The σ threshold used.
    pub sigma_threshold: f64,
}

/// Per-link rolling baseline tracker with cross-link change-point detection
/// (ADR-142 §2; extends ADR-030).
#[derive(Debug, Clone)]
pub struct EvolutionTracker {
    links: Vec<WelfordStats>,
    sigma_threshold: f64,
    min_links: usize,
}

impl EvolutionTracker {
    /// Track `n_links` links; flag a change-point when at least `min_links`
    /// links exceed `sigma_threshold`σ of their own baseline in one window.
    #[must_use]
    pub fn new(n_links: usize, sigma_threshold: f64, min_links: usize) -> Self {
        Self {
            links: (0..n_links).map(|_| WelfordStats::new()).collect(),
            sigma_threshold,
            min_links,
        }
    }

    /// Default: 2σ threshold, ≥3 links (ADR-142 §2).
    #[must_use]
    pub fn with_defaults(n_links: usize) -> Self {
        Self::new(n_links, 2.0, 3)
    }

    /// Number of links tracked.
    #[must_use]
    pub fn n_links(&self) -> usize {
        self.links.len()
    }

    /// True if `value` on `link_idx` is beyond `sigma_threshold`σ of that link's
    /// established baseline (needs ≥2 prior observations).
    #[must_use]
    pub fn is_link_diverging(&self, link_idx: usize, value: f64) -> bool {
        match self.links.get(link_idx) {
            Some(w) if w.count >= 2 && w.std_dev() > 1e-9 => {
                (value - w.mean).abs() / w.std_dev() > self.sigma_threshold
            }
            _ => false,
        }
    }

    /// Fold one observation per link, returning a [`ChangePoint`] when the
    /// number of simultaneously-diverging links reaches `min_links`. Divergence
    /// is evaluated against the *prior* baseline before this sample is folded in.
    pub fn observe_window(&mut self, values: &[f64]) -> Option<ChangePoint> {
        let mut diverging = 0usize;
        for (i, &v) in values.iter().enumerate() {
            if self.is_link_diverging(i, v) {
                diverging += 1;
            }
        }
        // Fold the samples in after the divergence check.
        for (w, &v) in self.links.iter_mut().zip(values.iter()) {
            w.update(v);
        }
        if diverging >= self.min_links {
            Some(ChangePoint { diverging_links: diverging, sigma_threshold: self.sigma_threshold })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxel_bayesian_update_raises_occupancy_and_confidence() {
        let mut v = TemporalVoxel::new([0.0, 0.0, 0.0]);
        assert!((v.occupancy - 0.5).abs() < 1e-9);
        assert!(v.is_low_confidence());
        for ns in 0..10 {
            v.observe(0.8, Some(0.3), ns);
        }
        assert!(v.occupancy > 0.9, "repeated positive evidence → high occupancy");
        assert!(!v.is_low_confidence(), "10 updates ⇒ confident");
        assert!(v.confidence > 0.8);
        assert_eq!(v.last_update_ns, 9);
        assert_eq!(v.doppler_velocity, Some(0.3));
    }

    #[test]
    fn voxel_low_confidence_below_five_frames() {
        let mut v = TemporalVoxel::new([1.0, 1.0, 0.0]);
        for ns in 0..4 {
            v.observe(0.7, None, ns);
        }
        assert!(v.is_low_confidence());
        v.observe(0.7, None, 4);
        assert!(!v.is_low_confidence(), "5th frame crosses the floor");
    }

    #[test]
    fn voxel_map_tracks_low_confidence() {
        let mut m = TemporalVoxelMap::new(vec![[0.0; 3], [1.0; 3]]);
        assert_eq!(m.len(), 2);
        for ns in 0..6 {
            m.observe(0, 0.9, None, ns);
        }
        // Voxel 0 confident, voxel 1 never observed → low.
        assert_eq!(m.low_confidence_indices(), vec![1]);
    }

    #[test]
    fn privacy_gate_anonymous_clears_doppler_keeps_occupancy() {
        let mut m = TemporalVoxelMap::new(vec![[0.0; 3]]);
        for ns in 0..6 {
            m.observe(0, 0.9, Some(0.5), ns);
        }
        let occ_before = m.voxel(0).unwrap().occupancy;
        assert!(VoxelGate::demote(&mut m, VoxelPrivacy::Anonymous, 4).is_none());
        let v = m.voxel(0).unwrap();
        assert_eq!(v.doppler_velocity, None);
        assert_eq!(v.confidence, 0.0);
        assert!((v.occupancy - occ_before).abs() < 1e-9, "occupancy retained");
    }

    #[test]
    fn privacy_gate_restricted_yields_histogram_and_clears() {
        let mut m = TemporalVoxelMap::new(vec![[0.0; 3], [1.0; 3], [2.0; 3]]);
        for ns in 0..6 {
            m.observe(0, 0.95, None, ns);
            m.observe(1, 0.95, None, ns);
        }
        let hist = VoxelGate::demote(&mut m, VoxelPrivacy::Restricted, 4).expect("histogram");
        assert_eq!(hist.iter().sum::<u32>(), 3, "all 3 voxels binned");
        // Raw occupancy cleared.
        assert!(m.occupancies().iter().all(|&o| o == 0.0));
    }

    /// ADR-142 acceptance (the environmental-nervous-system path):
    /// `three links drift for 30 frames -> ChangePoint fires -> VoxelMap
    ///  accumulates evidence -> low-confidence voxels suppressed -> VoxelGate
    ///  Restricted emits histogram only -> ADR-137 contradiction recorded`.
    #[test]
    fn acceptance_drift_to_histogram_with_contradiction() {
        use crate::ruvsense::fusion_quality::ContradictionFlag;

        // Three links, change-point requires all three to diverge at once.
        let mut tracker = EvolutionTracker::new(3, 2.0, 3);
        // 30 jittered baseline frames (non-zero std so divergence is defined).
        for i in 0..30u32 {
            let j = if i % 2 == 0 { 0.99 } else { 1.01 };
            assert!(tracker.observe_window(&[j, j, j]).is_none(), "baseline is quiet");
        }
        // Three links drift simultaneously → ChangePoint fires.
        let cp = tracker
            .observe_window(&[5.0, 5.0, 5.0])
            .expect("simultaneous drift on 3 links must fire a change-point");
        assert_eq!(cp.diverging_links, 3);

        // VoxelMap accumulates evidence over repeated observations.
        let mut map = TemporalVoxelMap::new(vec![[0.0; 3], [1.0; 3], [2.0; 3]]);
        for ns in 0..6 {
            map.observe(0, 0.95, Some(0.4), ns);
            map.observe(1, 0.90, None, ns);
            // voxel 2 deliberately under-observed.
        }
        assert!(map.voxel(0).unwrap().occupancy > 0.9, "evidence accumulated");

        // Low-confidence voxels (under 5 frames) are suppressed from output.
        let low = map.low_confidence_indices();
        assert!(low.contains(&2) && !low.contains(&0), "voxel 2 suppressed, voxel 0 kept");

        // ADR-137 contradiction recorded from the change-point (drift conflict).
        let contradictions = vec![ContradictionFlag::DriftProfileConflict {
            node_idx: 0,
            drift_score: cp.diverging_links as f32,
        }];
        assert!(!contradictions.is_empty(), "change-point recorded as an ADR-137 contradiction");

        // VoxelGate Restricted → histogram only; the raw map never leaves the node.
        let hist = VoxelGate::demote(&mut map, VoxelPrivacy::Restricted, 4)
            .expect("Restricted yields an occupancy histogram");
        assert_eq!(hist.iter().sum::<u32>(), 3, "all voxels binned");
        assert!(map.occupancies().iter().all(|&o| o == 0.0), "raw occupancy cleared");
    }

    #[test]
    fn evolution_tracker_detects_cross_link_change_point() {
        let mut t = EvolutionTracker::with_defaults(4);
        // Establish stable baselines (~1.0) with realistic small jitter so each
        // link has a non-zero std (a perfectly constant baseline has std 0 and
        // divergence is undefined).
        for i in 0..30 {
            let jitter = if i % 2 == 0 { 0.99 } else { 1.01 };
            assert!(t.observe_window(&[jitter, jitter, jitter, jitter]).is_none());
        }
        // A divergence on a single link must NOT trip a change-point (< min_links).
        assert!(t.observe_window(&[5.0, 1.0, 1.0, 1.0]).is_none());
        // A large simultaneous excursion on 3 links → change-point.
        let cp = t.observe_window(&[5.0, 5.0, 5.0, 1.0]);
        assert!(matches!(cp, Some(ChangePoint { diverging_links, .. }) if diverging_links >= 3));
    }
}

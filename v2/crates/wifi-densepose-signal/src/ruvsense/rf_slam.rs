//! ADR-143 — RF-SLAM: persistent reflector discovery and static-anchor learning.
//!
//! Ships **v1 fixed-map first** (known sensor positions + a small set of static
//! reflectors, `discovery_enabled = false`). v2 discovery — inferring persistent
//! reflector positions from ADR-134 CIR tap separation + temporal coherence,
//! clustering them into furniture/wall anchors, and detecting topology changes —
//! is gated behind `discovery_enabled` until a multi-day validation dataset is
//! collected (ADR-143 §2.5).
//!
//! Reflector positions, once discovered, are intended to land as ADR-139
//! `WorldNode::ObjectAnchor` nodes; this module owns the inference, the
//! WorldGraph owns the persistence.

use crate::ruvsense::field_model::WelfordStats;

/// Nanoseconds per day, for migration-rate (m/day) conversion (ADR-154 §7.4 —
/// de-magicked from the inline `86_400_000_000_000.0` literal). 24·60·60·1e9.
const NS_PER_DAY: f64 = 86_400_000_000_000.0;

/// Minimum observed span (in days) below which migration rate is reported as
/// 0.0 — guards `cumulative_drift_m / span_days` against a near-zero span.
const MIGRATION_MIN_SPAN_DAYS: f64 = 1e-9;

// ADR-154 §7.4: the v1 fixed-map defaults below were bare literals in
// `fixed_map()`. They are EMPIRICAL DEFAULTS (ADR-143), unchanged.

/// Default association radius (m): a sighting within this of a reflector's
/// running mean is folded into it; otherwise it seeds a new reflector.
const FIXED_MAP_ASSOC_RADIUS_M: f64 = 0.5;

/// Default minimum sightings before a reflector counts as "persistent".
const FIXED_MAP_MIN_SIGHTINGS: u64 = 20;

/// Default minimum tap coherence for a sighting to be admitted.
const FIXED_MAP_MIN_COHERENCE: f32 = 0.6;

/// Classification of a discovered persistent reflector (mirrors ADR-139
/// `AnchorKind`; kept local to avoid a crate dependency on the WorldGraph).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectorClass {
    /// A near-static reflector consistent with a wall (very low migration).
    Wall,
    /// A slowly-moving reflector consistent with furniture.
    Furniture,
    /// Moves too fast to be a static anchor (rejected from the anchor set).
    Mobile,
}

/// A single CIR-tap-derived reflector sighting at a point in time (ADR-134 CIR).
#[derive(Debug, Clone, Copy)]
pub struct ReflectorObservation {
    /// Inferred reflector position (east, north, up) in metres.
    pub position: [f64; 3],
    /// CIR dominant-tap delay (ns) that produced this sighting.
    pub delay_ns: f64,
    /// Temporal coherence of the tap in [0, 1] (gate quality).
    pub coherence: f32,
    /// Capture-clock time (ns).
    pub at_ns: u64,
}

/// A reflector accumulated over many sightings (ADR-143 §2).
#[derive(Debug, Clone)]
pub struct PersistentReflector {
    /// Per-axis position statistics (Welford).
    pos: [WelfordStats; 3],
    /// Number of sightings folded in.
    pub sightings: u64,
    /// First and last sighting times (ns).
    pub first_ns: u64,
    /// Last sighting time (ns).
    pub last_ns: u64,
    /// Total displacement of the running mean since the first sighting (m).
    cumulative_drift_m: f64,
    /// Last mean position, for incremental drift accumulation.
    last_mean: [f64; 3],
}

impl PersistentReflector {
    fn from_first(obs: &ReflectorObservation) -> Self {
        let mut pos = [WelfordStats::new(), WelfordStats::new(), WelfordStats::new()];
        for a in 0..3 {
            pos[a].update(obs.position[a]);
        }
        Self {
            pos,
            sightings: 1,
            first_ns: obs.at_ns,
            last_ns: obs.at_ns,
            cumulative_drift_m: 0.0,
            last_mean: obs.position,
        }
    }

    fn fold(&mut self, obs: &ReflectorObservation) {
        for a in 0..3 {
            self.pos[a].update(obs.position[a]);
        }
        let new_mean = self.mean_position();
        let d: f64 = (0..3).map(|a| (new_mean[a] - self.last_mean[a]).powi(2)).sum::<f64>().sqrt();
        self.cumulative_drift_m += d;
        self.last_mean = new_mean;
        self.last_ns = obs.at_ns;
        self.sightings += 1;
    }

    /// Mean reflector position.
    #[must_use]
    pub fn mean_position(&self) -> [f64; 3] {
        [self.pos[0].mean, self.pos[1].mean, self.pos[2].mean]
    }

    /// Positional spread (max per-axis std, m) — low ⇒ a stable reflector.
    #[must_use]
    pub fn position_std(&self) -> f64 {
        (0..3).map(|a| self.pos[a].std_dev()).fold(0.0, f64::max)
    }

    /// Mean-position migration rate in metres/day over the observed span.
    #[must_use]
    pub fn migration_m_per_day(&self) -> f64 {
        let span_ns = self.last_ns.saturating_sub(self.first_ns);
        if span_ns == 0 {
            return 0.0;
        }
        let span_days = span_ns as f64 / NS_PER_DAY; // ns → days
        if span_days < MIGRATION_MIN_SPAN_DAYS {
            return 0.0;
        }
        self.cumulative_drift_m / span_days
    }

    /// Classify by migration rate (ADR-143 §2): walls barely move, furniture
    /// migrates slowly, anything faster than `mobile_floor` m/day is rejected.
    #[must_use]
    pub fn classify(&self, wall_ceiling: f64, mobile_floor: f64) -> ReflectorClass {
        let m = self.migration_m_per_day();
        if m <= wall_ceiling {
            ReflectorClass::Wall
        } else if m < mobile_floor {
            ReflectorClass::Furniture
        } else {
            ReflectorClass::Mobile
        }
    }
}

/// RF-SLAM reflector discovery engine (ADR-143).
#[derive(Debug, Clone)]
pub struct RfSlam {
    reflectors: Vec<PersistentReflector>,
    /// Association radius (m): a sighting within this of a reflector's mean is
    /// folded in; otherwise it seeds a new reflector.
    assoc_radius_m: f64,
    /// Minimum sightings before a reflector counts as "persistent".
    min_sightings: u64,
    /// Minimum tap coherence for a sighting to be admitted.
    min_coherence: f32,
    /// v2 discovery gate — false ⇒ fixed-map v1 (no new reflectors learned).
    discovery_enabled: bool,
}

impl RfSlam {
    /// v1 fixed-map mode: discovery disabled.
    #[must_use]
    pub fn fixed_map() -> Self {
        Self {
            reflectors: Vec::new(),
            assoc_radius_m: FIXED_MAP_ASSOC_RADIUS_M,
            min_sightings: FIXED_MAP_MIN_SIGHTINGS,
            min_coherence: FIXED_MAP_MIN_COHERENCE,
            discovery_enabled: false,
        }
    }

    /// v2 discovery mode: learn persistent reflectors from sightings.
    #[must_use]
    pub fn with_discovery(assoc_radius_m: f64, min_sightings: u64, min_coherence: f32) -> Self {
        Self {
            reflectors: Vec::new(),
            assoc_radius_m,
            min_sightings,
            min_coherence,
            discovery_enabled: true,
        }
    }

    /// Whether v2 discovery is active.
    #[must_use]
    pub fn discovery_enabled(&self) -> bool {
        self.discovery_enabled
    }

    /// Ingest one CIR-derived sighting. In fixed-map mode this is a no-op
    /// (returns false). In discovery mode it associates to the nearest reflector
    /// within `assoc_radius_m` or seeds a new one; returns true if accepted.
    pub fn observe(&mut self, obs: &ReflectorObservation) -> bool {
        if !self.discovery_enabled || obs.coherence < self.min_coherence {
            return false;
        }
        // Nearest-reflector association.
        let mut best: Option<(usize, f64)> = None;
        for (i, r) in self.reflectors.iter().enumerate() {
            let m = r.mean_position();
            let d: f64 = (0..3).map(|a| (m[a] - obs.position[a]).powi(2)).sum::<f64>().sqrt();
            if d <= self.assoc_radius_m && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        match best {
            Some((i, _)) => self.reflectors[i].fold(obs),
            None => self.reflectors.push(PersistentReflector::from_first(obs)),
        }
        true
    }

    /// Indices/refs of reflectors that have crossed the persistence threshold.
    #[must_use]
    pub fn persistent(&self) -> Vec<&PersistentReflector> {
        self.reflectors.iter().filter(|r| r.sightings >= self.min_sightings).collect()
    }

    /// Static-anchor set: persistent reflectors classified Wall or Furniture
    /// (mobile reflectors rejected) — the candidate ADR-139 `ObjectAnchor`s.
    #[must_use]
    pub fn static_anchors(&self, wall_ceiling: f64, mobile_floor: f64) -> Vec<([f64; 3], ReflectorClass)> {
        self.persistent()
            .into_iter()
            .map(|r| (r.mean_position(), r.classify(wall_ceiling, mobile_floor)))
            .filter(|(_, c)| *c != ReflectorClass::Mobile)
            .collect()
    }

    /// Topology-change signal: the count of persistent reflectors. A caller
    /// compares this across time; an increase/decrease beyond a threshold marks
    /// a furniture-moved / room-changed event (ADR-143 §2 topology detection).
    #[must_use]
    pub fn persistent_count(&self) -> usize {
        self.reflectors.iter().filter(|r| r.sightings >= self.min_sightings).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(pos: [f64; 3], at_ns: u64) -> ReflectorObservation {
        ReflectorObservation { position: pos, delay_ns: 10.0, coherence: 0.9, at_ns }
    }

    #[test]
    fn fixed_map_does_not_discover() {
        let mut slam = RfSlam::fixed_map();
        assert!(!slam.discovery_enabled());
        assert!(!slam.observe(&obs([1.0, 1.0, 0.0], 0)));
        assert_eq!(slam.persistent_count(), 0);
    }

    #[test]
    fn discovery_learns_persistent_reflector() {
        let mut slam = RfSlam::with_discovery(0.5, 20, 0.6);
        // 25 sightings clustered tightly around (2,3,0).
        for i in 0..25u64 {
            let jitter = if i % 2 == 0 { 0.01 } else { -0.01 };
            assert!(slam.observe(&obs([2.0 + jitter, 3.0, 0.0], i * 1_000_000)));
        }
        assert_eq!(slam.persistent_count(), 1);
        let r = slam.persistent()[0];
        assert!((r.mean_position()[0] - 2.0).abs() < 0.05);
        assert!(r.position_std() < 0.1);
    }

    #[test]
    fn low_coherence_sightings_rejected() {
        let mut slam = RfSlam::with_discovery(0.5, 5, 0.6);
        let mut o = obs([1.0, 1.0, 0.0], 0);
        o.coherence = 0.3; // below min
        assert!(!slam.observe(&o));
        assert_eq!(slam.persistent_count(), 0);
    }

    #[test]
    fn separate_clusters_form_distinct_reflectors() {
        let mut slam = RfSlam::with_discovery(0.5, 3, 0.6);
        for i in 0..5u64 {
            slam.observe(&obs([0.0, 0.0, 0.0], i));
            slam.observe(&obs([5.0, 5.0, 0.0], i)); // > assoc_radius apart
        }
        assert_eq!(slam.persistent_count(), 2);
    }

    #[test]
    fn mobile_reflector_excluded_from_anchors() {
        // A reflector whose mean marches ~10 m/day is Mobile, not an anchor.
        let mut slam = RfSlam::with_discovery(50.0, 5, 0.6);
        let day_ns = 86_400_000_000_000u64;
        for i in 0..10u64 {
            // Position advances 1 m each tenth-of-a-day → ~10 m/day.
            let t = i * (day_ns / 10);
            slam.observe(&obs([i as f64, 0.0, 0.0], t));
        }
        let anchors = slam.static_anchors(0.05, 1.0);
        assert!(anchors.is_empty(), "fast-migrating reflector must not be an anchor");
        // But it is still a persistent reflector (tracked, just not anchored).
        assert_eq!(slam.persistent_count(), 1);
        assert_eq!(slam.persistent()[0].classify(0.05, 1.0), ReflectorClass::Mobile);
    }

    #[test]
    fn static_reflector_classified_wall() {
        let mut slam = RfSlam::with_discovery(0.5, 5, 0.6);
        let day_ns = 86_400_000_000_000u64;
        for i in 0..10u64 {
            // Tight cluster, spanning ~1 day → ~0 migration.
            let jitter = if i % 2 == 0 { 0.005 } else { -0.005 };
            slam.observe(&obs([3.0 + jitter, 0.0, 0.0], i * (day_ns / 10)));
        }
        let anchors = slam.static_anchors(0.05, 1.0);
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].1, ReflectorClass::Wall);
    }

    // -- ADR-154 §7.4: de-magic-constant + boundary characterization tests.

    /// De-magicked constants must equal the prior inline literals.
    #[test]
    fn migration_consts_unchanged_from_literals() {
        assert_eq!(NS_PER_DAY, 86_400_000_000_000.0);
        assert_eq!(NS_PER_DAY, 24.0 * 60.0 * 60.0 * 1e9);
        assert_eq!(MIGRATION_MIN_SPAN_DAYS, 1e-9);
        assert_eq!(FIXED_MAP_ASSOC_RADIUS_M, 0.5);
        assert_eq!(FIXED_MAP_MIN_SIGHTINGS, 20);
        assert_eq!(FIXED_MAP_MIN_COHERENCE, 0.6_f32);
    }

    /// A single sighting has first_ns == last_ns ⇒ zero span ⇒ migration rate
    /// 0.0 (pins the `span_ns == 0` / `span_days < MIGRATION_MIN_SPAN_DAYS`
    /// guard, and that such a reflector classifies as a Wall).
    #[test]
    fn migration_zero_span_is_zero_rate() {
        let mut slam = RfSlam::with_discovery(0.5, 1, 0.6);
        slam.observe(&obs([1.0, 2.0, 0.0], 12_345));
        let r = slam.persistent()[0];
        assert_eq!(r.migration_m_per_day(), 0.0);
        assert_eq!(r.classify(0.05, 1.0), ReflectorClass::Wall);
    }
}

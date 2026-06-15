//! Selectable self-learning strategies for swarm MARL.
//!
//! - Mappo: centralized-critic, decentralized-execution (CTDE). Best cooperative
//!   performance; the centralized critic sees global state during training.
//! - Ippo: independent PPO — each agent learns alone, no shared critic. Robust to
//!   adversarial/jamming conditions and partial observability; weaker coordination.
//! - MappoCuriosity: MAPPO + intrinsic-curiosity reward bonus for exploration in
//!   sparse-reward regimes (count-based novelty over visited regions).
//! - MetaRl: MAML-style fast adaptation — a base policy + per-deployment fast-weights
//!   that adapt in a few in-flight steps to wind/sensor drift.
//!
//! Pure Rust — always compiled (no Candle needed). This is the *strategy* layer;
//! the gradient backend lives in `candle_ppo.rs` behind the `train` feature.

/// Which self-learning strategy the swarm trains under. Selectable at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LearningPattern {
    /// Centralized critic, decentralized execution (CTDE).
    #[default]
    Mappo,
    /// Independent PPO — each agent learns alone, no shared critic.
    Ippo,
    /// MAPPO plus count-based intrinsic-curiosity reward bonus.
    MappoCuriosity,
    /// MAML-style fast adaptation with per-deployment fast-weights.
    MetaRl,
}

impl LearningPattern {
    /// Parse from a short identifier. Unknown strings fall back to the default
    /// (Mappo). Accepts both canonical names and friendly aliases.
    // Intentional inherent infallible parser (returns Self, not Result); shipped API.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "mappo" => LearningPattern::Mappo,
            "ippo" => LearningPattern::Ippo,
            "curiosity" | "mappocuriosity" | "mappo_curiosity" => {
                LearningPattern::MappoCuriosity
            }
            "meta" | "metarl" | "meta_rl" => LearningPattern::MetaRl,
            _ => LearningPattern::default(),
        }
    }

    /// Canonical short name. `from_str(p.name()) == p` for every variant.
    pub fn name(&self) -> &'static str {
        match self {
            LearningPattern::Mappo => "mappo",
            LearningPattern::Ippo => "ippo",
            LearningPattern::MappoCuriosity => "curiosity",
            LearningPattern::MetaRl => "meta",
        }
    }

    /// Whether this strategy uses a centralized critic (CTDE) vs independent.
    pub fn centralized_critic(&self) -> bool {
        matches!(
            self,
            LearningPattern::Mappo
                | LearningPattern::MappoCuriosity
                | LearningPattern::MetaRl
        )
    }

    /// Whether an intrinsic-curiosity bonus is added to the reward.
    pub fn uses_curiosity(&self) -> bool {
        matches!(self, LearningPattern::MappoCuriosity)
    }
}

// ---------------------------------------------------------------------------
// Curiosity: count-based intrinsic motivation
// ---------------------------------------------------------------------------

/// Count-based intrinsic-motivation module.
///
/// Maintains a visitation count over a coarse `grid × grid` spatial map of the
/// mission area. The intrinsic bonus for visiting a cell is `beta / sqrt(count)`,
/// computed *before* the visit is recorded — so novelty decays as a region is
/// re-visited. This rewards exploration in sparse-reward regimes.
pub struct CuriosityModule {
    counts: Vec<u32>,
    grid: u32,
    cell_w: f64,
    cell_h: f64,
    beta: f32,
}

impl CuriosityModule {
    /// Build a curiosity grid covering an `area_w × area_h` metre region split
    /// into `grid × grid` cells. `beta` scales the intrinsic bonus magnitude.
    pub fn new(area_w: f64, area_h: f64, grid: u32, beta: f32) -> Self {
        let g = grid.max(1);
        let cells = (g as usize) * (g as usize);
        let cell_w = if area_w > 0.0 { area_w / g as f64 } else { 1.0 };
        let cell_h = if area_h > 0.0 { area_h / g as f64 } else { 1.0 };
        Self {
            counts: vec![0; cells],
            grid: g,
            cell_w,
            cell_h,
            beta,
        }
    }

    /// Map a world-coordinate to a flat cell index, clamped to the grid.
    fn cell_index(&self, x: f64, y: f64) -> usize {
        let gx = ((x / self.cell_w).floor() as i64).clamp(0, self.grid as i64 - 1) as usize;
        let gy = ((y / self.cell_h).floor() as i64).clamp(0, self.grid as i64 - 1) as usize;
        gy * self.grid as usize + gx
    }

    /// Record a visit and return the intrinsic reward bonus for novelty.
    ///
    /// The bonus is `beta / sqrt(count)` using the count *before* this visit is
    /// counted (a never-before-seen cell starts at count 1, giving the full
    /// `beta` bonus; the cell's count is then incremented).
    pub fn visit_bonus(&mut self, x: f64, y: f64) -> f32 {
        let idx = self.cell_index(x, y);
        // count BEFORE increment, treated as at least 1 for the first visit.
        let prior = self.counts[idx] + 1;
        let bonus = self.beta / (prior as f32).sqrt();
        self.counts[idx] = self.counts[idx].saturating_add(1);
        bonus
    }

    /// Total recorded visits across the whole grid.
    pub fn total_visits(&self) -> u64 {
        self.counts.iter().map(|&c| c as u64).sum()
    }
}

// ---------------------------------------------------------------------------
// Meta-RL: MAML-style fast-weight adapter
// ---------------------------------------------------------------------------

/// MAML-style fast-weight adapter for few-shot in-flight adaptation.
///
/// Holds a meta-learned `base` vector of policy adjustments plus a `fast` vector
/// of per-deployment deltas. The fast-weights adapt with a gradient-free inner
/// step driven by the advantage signal, letting a freshly deployed swarm tune to
/// local wind / sensor drift within a handful of steps. `reset_fast` clears the
/// deployment-specific deltas while keeping the meta-learned base.
pub struct MetaAdapter {
    base: Vec<f32>,
    fast: Vec<f32>,
    inner_lr: f32,
}

impl MetaAdapter {
    /// New adapter with a zeroed `dim`-length base and fast-weight vector.
    pub fn new(dim: usize, inner_lr: f32) -> Self {
        Self {
            base: vec![0.0; dim],
            fast: vec![0.0; dim],
            inner_lr,
        }
    }

    /// One inner-loop adaptation step from an advantage signal (few-shot).
    ///
    /// Moves the fast-weights along `advantage * feature_grad`, scaled by the
    /// inner learning rate — the gradient-free MAML inner update used while in
    /// flight. `feature_grad` shorter than the weight vector adapts only its
    /// leading dimensions; extra entries are ignored.
    pub fn adapt(&mut self, advantage: f32, feature_grad: &[f32]) {
        let n = self.fast.len().min(feature_grad.len());
        for (f, &g) in self.fast.iter_mut().zip(feature_grad.iter()).take(n) {
            *f += self.inner_lr * advantage * g;
        }
    }

    /// Current effective weights (base + fast).
    pub fn effective(&self) -> Vec<f32> {
        self.base
            .iter()
            .zip(self.fast.iter())
            .map(|(b, f)| b + f)
            .collect()
    }

    /// Reset fast-weights for a new deployment (keeps the meta-learned base).
    pub fn reset_fast(&mut self) {
        for f in self.fast.iter_mut() {
            *f = 0.0;
        }
    }

    /// Fold the current fast-weights into the meta-learned base (outer-loop
    /// consolidation) and clear the fast deltas.
    pub fn consolidate(&mut self) {
        for (b, f) in self.base.iter_mut().zip(self.fast.iter()) {
            *b += *f;
        }
        self.reset_fast();
    }
}

// ---------------------------------------------------------------------------
// Reward shaping helper
// ---------------------------------------------------------------------------

/// Shape a base reward according to the selected learning pattern.
///
/// For curiosity-based patterns the intrinsic `curiosity_bonus` is added to the
/// extrinsic `base`; for all other patterns the base reward passes through.
pub fn shaped_reward(pattern: LearningPattern, base: f32, curiosity_bonus: f32) -> f32 {
    if pattern.uses_curiosity() {
        base + curiosity_bonus
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [LearningPattern; 4] = [
        LearningPattern::Mappo,
        LearningPattern::Ippo,
        LearningPattern::MappoCuriosity,
        LearningPattern::MetaRl,
    ];

    #[test]
    fn test_pattern_from_str_roundtrip() {
        for p in ALL {
            assert_eq!(
                LearningPattern::from_str(p.name()),
                p,
                "round-trip failed for {}",
                p.name()
            );
        }
    }

    #[test]
    fn test_centralized_vs_independent() {
        // Mappo IS centralized (CTDE); Ippo is NOT (independent learners).
        assert!(LearningPattern::Mappo.centralized_critic());
        assert!(!LearningPattern::Ippo.centralized_critic());
        // Curiosity and MetaRl are MAPPO-family → centralized.
        assert!(LearningPattern::MappoCuriosity.centralized_critic());
        assert!(LearningPattern::MetaRl.centralized_critic());
    }

    #[test]
    fn test_curiosity_bonus_decreases() {
        let mut cm = CuriosityModule::new(100.0, 100.0, 10, 1.0);
        let first = cm.visit_bonus(50.0, 50.0);
        let second = cm.visit_bonus(50.0, 50.0); // same cell again
        assert!(
            second < first,
            "novelty should decay: first={first}, second={second}"
        );
    }

    #[test]
    fn test_curiosity_bonus_in_bounds() {
        let mut cm = CuriosityModule::new(100.0, 100.0, 8, 0.5);
        // In-bounds, out-of-bounds, and negative coords all clamp safely.
        for &(x, y) in &[(0.0, 0.0), (50.0, 50.0), (999.0, -999.0), (-5.0, 1000.0)] {
            let b = cm.visit_bonus(x, y);
            assert!(b.is_finite(), "bonus must be finite, got {b}");
            assert!(b >= 0.0, "bonus must be >= 0, got {b}");
        }
    }

    #[test]
    fn test_meta_adapter_changes_weights() {
        let mut ma = MetaAdapter::new(4, 0.1);
        let base = ma.effective();
        ma.adapt(2.0, &[1.0, -1.0, 0.5, 0.0]);
        let adapted = ma.effective();
        assert_ne!(base, adapted, "adapt() must change effective weights");
        ma.reset_fast();
        assert_eq!(
            base,
            ma.effective(),
            "reset_fast() must restore the meta-learned base"
        );
    }

    #[test]
    fn test_shaped_reward_curiosity_only() {
        let base = 10.0;
        let bonus = 3.0;
        // MappoCuriosity adds the bonus.
        assert_eq!(
            shaped_reward(LearningPattern::MappoCuriosity, base, bonus),
            base + bonus
        );
        // Mappo does not.
        assert_eq!(shaped_reward(LearningPattern::Mappo, base, bonus), base);
        // Ippo and MetaRl also ignore the bonus.
        assert_eq!(shaped_reward(LearningPattern::Ippo, base, bonus), base);
        assert_eq!(shaped_reward(LearningPattern::MetaRl, base, bonus), base);
    }
}

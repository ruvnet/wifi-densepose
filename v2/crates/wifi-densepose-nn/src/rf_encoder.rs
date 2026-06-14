//! ADR-146 — RF encoder multi-task heads + uncertainty quantification.
//!
//! Extends ADR-024 (AETHER contrastive embedding) with seven task-specific head
//! branches over a shared RF embedding, per-head uncertainty, a
//! calibration-robustness loss tying invariance to the ADR-135 `calibration_id`,
//! and a `ContrastiveBatcher` sampling contract. The tensor ABI is **pure-Rust
//! `f32`** (no backend-specific tensor type at this boundary) so inference is
//! deterministic and witnessable (ADR-136 §2.5) and a head can be toggled by the
//! ADR-145 ablation matrix.

/// Shared RF embedding dimension (ADR-146 / ADR-024 AETHER).
pub const EMBEDDING_DIM: usize = 256;

/// A 256-d shared RF embedding (pure-Rust f32 ABI).
#[derive(Debug, Clone, PartialEq)]
pub struct RfEmbedding(pub Vec<f32>);

impl RfEmbedding {
    /// Wrap a vector, asserting it is [`EMBEDDING_DIM`] long.
    #[must_use]
    pub fn new(v: Vec<f32>) -> Self {
        debug_assert_eq!(v.len(), EMBEDDING_DIM, "embedding must be {EMBEDDING_DIM}-d");
        Self(v)
    }

    /// Squared L2 distance to another embedding.
    #[must_use]
    pub fn sq_dist(&self, other: &RfEmbedding) -> f32 {
        self.0.iter().zip(&other.0).map(|(a, b)| (a - b).powi(2)).sum()
    }
}

/// The seven task heads over the shared encoder (ADR-146 §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    /// 17-keypoint pose.
    Pose,
    /// Binary presence.
    Presence,
    /// Person count.
    Count,
    /// Activity class.
    Activity,
    /// Vital signs (HR/BR).
    Vitals,
    /// Gait signature.
    Gait,
    /// Identity embedding (AETHER re-ID).
    IdentityEmbedding,
}

impl TaskKind {
    /// All seven heads.
    pub const ALL: [TaskKind; 7] = [
        TaskKind::Pose,
        TaskKind::Presence,
        TaskKind::Count,
        TaskKind::Activity,
        TaskKind::Vitals,
        TaskKind::Gait,
        TaskKind::IdentityEmbedding,
    ];
}

/// One head's output: task values plus a scalar predictive uncertainty
/// (ADR-146 §2.2). `uncertainty` mirrors the spirit of the ADR-136
/// `QualityScored` trait — lower is more confident.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadOutput {
    /// Which head produced this.
    pub task: TaskKind,
    /// Raw output activations.
    pub values: Vec<f32>,
    /// Predictive uncertainty in [0, ∞); softplus of a learned log-variance.
    pub uncertainty: f32,
}

impl HeadOutput {
    /// Confidence in [0, 1] derived from uncertainty (`1 / (1 + uncertainty)`),
    /// matching the ADR-136 `QualityScored::quality_score` contract shape.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        1.0 / (1.0 + self.uncertainty)
    }
}

/// A linear task head: `out = W·emb + b`, plus a separate scalar log-variance
/// projection `lv = wᵥ·emb + bᵥ` whose softplus is the predictive uncertainty.
#[derive(Debug, Clone)]
pub struct LinearHead {
    task: TaskKind,
    /// Row-major `[out_dim × EMBEDDING_DIM]` weights.
    w: Vec<f32>,
    b: Vec<f32>,
    out_dim: usize,
    /// Uncertainty (log-variance) projection over the embedding.
    var_w: Vec<f32>,
    var_b: f32,
}

/// A shape mismatch when building a [`LinearHead`] from supplied weights.
///
/// Returned by [`LinearHead::try_new`] so a caller loading weights from an
/// **untrusted / deserialized** source can validate the tensor shapes without
/// the panic that [`LinearHead::new`] raises on a programmer-supplied mismatch
/// (ADR-155 M2 §3: a pure-Rust input guard ahead of the construction contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RfHeadError {
    /// `w.len()` was not `out_dim * EMBEDDING_DIM`.
    WeightShape {
        /// Expected length (`out_dim * EMBEDDING_DIM`).
        expected: usize,
        /// Actual `w.len()`.
        got: usize,
    },
    /// `b.len()` was not `out_dim`.
    BiasShape {
        /// Expected length (`out_dim`).
        expected: usize,
        /// Actual `b.len()`.
        got: usize,
    },
    /// `var_w.len()` was not `EMBEDDING_DIM`.
    VarWeightShape {
        /// Expected length (`EMBEDDING_DIM`).
        expected: usize,
        /// Actual `var_w.len()`.
        got: usize,
    },
}

impl std::fmt::Display for RfHeadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightShape { expected, got } => {
                write!(f, "weight shape mismatch: expected {expected}, got {got}")
            }
            Self::BiasShape { expected, got } => {
                write!(f, "bias shape mismatch: expected {expected}, got {got}")
            }
            Self::VarWeightShape { expected, got } => {
                write!(f, "var weight shape mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for RfHeadError {}

impl LinearHead {
    /// Build a head with given weights. `w.len()` must be `out_dim * EMBEDDING_DIM`.
    ///
    /// # Panics
    ///
    /// Panics on a shape mismatch (`w`/`b`/`var_w`). This is a construction-time
    /// API contract on *programmer-supplied* vectors. For weights from an
    /// untrusted / deserialized source, prefer [`LinearHead::try_new`], which
    /// returns a typed [`RfHeadError`] instead of panicking.
    #[must_use]
    pub fn new(task: TaskKind, out_dim: usize, w: Vec<f32>, b: Vec<f32>, var_w: Vec<f32>, var_b: f32) -> Self {
        assert_eq!(w.len(), out_dim * EMBEDDING_DIM, "weight shape mismatch");
        assert_eq!(b.len(), out_dim, "bias shape mismatch");
        assert_eq!(var_w.len(), EMBEDDING_DIM, "var weight shape mismatch");
        Self { task, w, b, out_dim, var_w, var_b }
    }

    /// Fallible constructor: validate the weight shapes and return a typed
    /// [`RfHeadError`] on mismatch instead of panicking (ADR-155 M2 §3).
    ///
    /// Use this when `w` / `b` / `var_w` originate from a checkpoint or any
    /// untrusted source. On success the produced head is byte-for-byte identical
    /// to [`LinearHead::new`] with the same arguments.
    ///
    /// # Errors
    ///
    /// Returns [`RfHeadError`] when any of:
    /// - `w.len() != out_dim * EMBEDDING_DIM`
    /// - `b.len() != out_dim`
    /// - `var_w.len() != EMBEDDING_DIM`
    pub fn try_new(
        task: TaskKind,
        out_dim: usize,
        w: Vec<f32>,
        b: Vec<f32>,
        var_w: Vec<f32>,
        var_b: f32,
    ) -> Result<Self, RfHeadError> {
        let expected_w = out_dim * EMBEDDING_DIM;
        if w.len() != expected_w {
            return Err(RfHeadError::WeightShape { expected: expected_w, got: w.len() });
        }
        if b.len() != out_dim {
            return Err(RfHeadError::BiasShape { expected: out_dim, got: b.len() });
        }
        if var_w.len() != EMBEDDING_DIM {
            return Err(RfHeadError::VarWeightShape { expected: EMBEDDING_DIM, got: var_w.len() });
        }
        Ok(Self { task, w, b, out_dim, var_w, var_b })
    }

    /// A zero-initialised head (uncertainty = softplus(0) ≈ 0.693).
    #[must_use]
    pub fn zeros(task: TaskKind, out_dim: usize) -> Self {
        Self::new(
            task,
            out_dim,
            vec![0.0; out_dim * EMBEDDING_DIM],
            vec![0.0; out_dim],
            vec![0.0; EMBEDDING_DIM],
            0.0,
        )
    }

    /// Forward pass over a shared embedding.
    #[must_use]
    pub fn forward(&self, emb: &RfEmbedding) -> HeadOutput {
        let mut values = vec![0.0f32; self.out_dim];
        for o in 0..self.out_dim {
            let row = &self.w[o * EMBEDDING_DIM..(o + 1) * EMBEDDING_DIM];
            let dot: f32 = row.iter().zip(&emb.0).map(|(wi, xi)| wi * xi).sum();
            values[o] = dot + self.b[o];
        }
        let log_var: f32 = self.var_w.iter().zip(&emb.0).map(|(wi, xi)| wi * xi).sum::<f32>() + self.var_b;
        let uncertainty = softplus(log_var);
        HeadOutput { task: self.task, values, uncertainty }
    }
}

/// Input magnitude above which `softplus(x) ≈ x` to f32 precision, so the
/// `exp` is skipped to avoid overflow (ADR-155 M2 §8: de-magicked from a bare
/// `20.0`; value unchanged). At x = 20, `ln(1+e^20) − 20 ≈ 2e-9`, below f32 eps.
const SOFTPLUS_LINEAR_THRESHOLD: f32 = 20.0;

fn softplus(x: f32) -> f32 {
    // Numerically stable softplus.
    if x > SOFTPLUS_LINEAR_THRESHOLD {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Multi-task encoder: a shared embedding feeding a set of [`LinearHead`]s
/// (ADR-146 §2.1). Heads can be subset for ADR-145 ablation.
#[derive(Debug, Clone, Default)]
pub struct MultiTaskHeads {
    heads: Vec<LinearHead>,
}

impl MultiTaskHeads {
    /// Empty head set.
    #[must_use]
    pub fn new() -> Self {
        Self { heads: Vec::new() }
    }

    /// Add a head.
    pub fn push(&mut self, head: LinearHead) {
        self.heads.push(head);
    }

    /// Number of active heads.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Whether no heads are configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Run every head on the shared embedding.
    #[must_use]
    pub fn forward(&self, emb: &RfEmbedding) -> Vec<HeadOutput> {
        self.heads.iter().map(|h| h.forward(emb)).collect()
    }

    /// Run only the heads in `enabled` (ADR-145 ablation toggle).
    #[must_use]
    pub fn forward_subset(&self, emb: &RfEmbedding, enabled: &[TaskKind]) -> Vec<HeadOutput> {
        self.heads
            .iter()
            .filter(|h| enabled.contains(&h.task))
            .map(|h| h.forward(emb))
            .collect()
    }
}

/// Calibration-robustness loss (ADR-146 §2.3): the encoder should produce the
/// same embedding for the same physical input under two different ADR-135
/// calibration baselines. Returns the mean squared embedding difference — a
/// penalty that is 0 under perfect calibration invariance.
#[must_use]
pub fn calibration_robustness_loss(under_cal_a: &RfEmbedding, under_cal_b: &RfEmbedding) -> f32 {
    under_cal_a.sq_dist(under_cal_b) / EMBEDDING_DIM as f32
}

/// Triplet contrastive loss (ADR-024 / ADR-146 §2.4): pull `anchor` toward
/// `positive` (same physical state), push from `negative` (different), with a
/// margin. `max(0, d(a,p) - d(a,n) + margin)`.
#[must_use]
pub fn triplet_loss(anchor: &RfEmbedding, positive: &RfEmbedding, negative: &RfEmbedding, margin: f32) -> f32 {
    (anchor.sq_dist(positive) - anchor.sq_dist(negative) + margin).max(0.0)
}

/// A contrastive training triplet over the shared embedding space.
#[derive(Debug, Clone)]
pub struct Triplet {
    /// Anchor sample index.
    pub anchor: usize,
    /// Positive (same state, different environment) index.
    pub positive: usize,
    /// Negative (different state) index.
    pub negative: usize,
}

/// Formalised contrastive pair/triplet sampler (ADR-146 §2.4): positives are the
/// *same physical state across different environments* (cross-room invariance,
/// ADR-027 MERIDIAN); negatives are *different states*.
#[derive(Debug, Clone)]
pub struct ContrastiveBatcher {
    /// `state_of[i]` = the physical-state label of sample `i`.
    state_of: Vec<u32>,
    /// `env_of[i]` = the environment/room label of sample `i`.
    env_of: Vec<u32>,
}

impl ContrastiveBatcher {
    /// Build from per-sample (state, environment) labels.
    #[must_use]
    pub fn new(state_of: Vec<u32>, env_of: Vec<u32>) -> Self {
        assert_eq!(state_of.len(), env_of.len(), "label vectors must align");
        Self { state_of, env_of }
    }

    /// Deterministically enumerate triplets: for each anchor, the first sample
    /// with the *same state but a different environment* is the positive, and
    /// the first sample with a *different state* is the negative. Anchors with
    /// no valid positive or negative are skipped. Determinism (lowest-index
    /// choice) keeps the batch witnessable (ADR-136 §2.5).
    #[must_use]
    pub fn triplets(&self) -> Vec<Triplet> {
        let n = self.state_of.len();
        let mut out = Vec::new();
        for a in 0..n {
            let positive = (0..n).find(|&p| {
                p != a && self.state_of[p] == self.state_of[a] && self.env_of[p] != self.env_of[a]
            });
            let negative = (0..n).find(|&q| self.state_of[q] != self.state_of[a]);
            if let (Some(positive), Some(negative)) = (positive, negative) {
                out.push(Triplet { anchor: a, positive, negative });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(fill: f32) -> RfEmbedding {
        RfEmbedding::new(vec![fill; EMBEDDING_DIM])
    }

    /// ADR-155 M2 §8: the de-magicked softplus linear-threshold must equal the
    /// prior inline `20.0` literal exactly (operating-value guard).
    #[test]
    fn softplus_threshold_unchanged_from_literal() {
        assert_eq!(SOFTPLUS_LINEAR_THRESHOLD, 20.0_f32);
    }

    /// ADR-155 M2 §3: `try_new` accepts correctly-shaped weights and produces a
    /// head byte-identical to `new`, but returns a typed error on a mismatched
    /// (e.g. corrupt-checkpoint) shape instead of panicking.
    #[test]
    fn try_new_accepts_valid_and_rejects_each_bad_shape() {
        let out_dim = 2;
        let w = vec![0.0; out_dim * EMBEDDING_DIM];
        let b = vec![0.0; out_dim];
        let var_w = vec![0.0; EMBEDDING_DIM];

        // Valid: try_new == new (forward identical on a probe embedding).
        let head = LinearHead::try_new(TaskKind::Presence, out_dim, w.clone(), b.clone(), var_w.clone(), 0.0)
            .expect("valid shapes must construct");
        let reference = LinearHead::new(TaskKind::Presence, out_dim, w.clone(), b.clone(), var_w.clone(), 0.0);
        assert_eq!(head.forward(&emb(0.5)).values, reference.forward(&emb(0.5)).values);

        // Bad weight length.
        assert_eq!(
            LinearHead::try_new(TaskKind::Presence, out_dim, vec![0.0; 3], b.clone(), var_w.clone(), 0.0)
                .unwrap_err(),
            RfHeadError::WeightShape { expected: out_dim * EMBEDDING_DIM, got: 3 }
        );
        // Bad bias length.
        assert_eq!(
            LinearHead::try_new(TaskKind::Presence, out_dim, w.clone(), vec![0.0; 1], var_w.clone(), 0.0)
                .unwrap_err(),
            RfHeadError::BiasShape { expected: out_dim, got: 1 }
        );
        // Bad var-weight length.
        assert_eq!(
            LinearHead::try_new(TaskKind::Presence, out_dim, w, b, vec![0.0; 5], 0.0).unwrap_err(),
            RfHeadError::VarWeightShape { expected: EMBEDDING_DIM, got: 5 }
        );
    }

    #[test]
    fn head_forward_produces_values_and_finite_uncertainty() {
        let head = LinearHead::zeros(TaskKind::Presence, 2);
        let out = head.forward(&emb(1.0));
        assert_eq!(out.values, vec![0.0, 0.0]); // zero weights
        assert!(out.uncertainty.is_finite() && out.uncertainty > 0.0);
        assert!((out.confidence() - 1.0 / (1.0 + out.uncertainty)).abs() < 1e-6);
    }

    #[test]
    fn uncertainty_responds_to_log_variance_weights() {
        // var_w all 1 → log_var = sum(emb) = 256 → softplus ≈ 256 (clamped path).
        let head = LinearHead::new(
            TaskKind::Vitals,
            1,
            vec![0.0; EMBEDDING_DIM],
            vec![0.0],
            vec![1.0; EMBEDDING_DIM],
            0.0,
        );
        let out = head.forward(&emb(1.0));
        assert!(out.uncertainty > 100.0, "high log-var → high uncertainty");
        assert!(out.confidence() < 0.02);
    }

    #[test]
    fn calibration_robustness_loss_zero_for_identical() {
        assert_eq!(calibration_robustness_loss(&emb(0.5), &emb(0.5)), 0.0);
        assert!(calibration_robustness_loss(&emb(0.0), &emb(1.0)) > 0.0);
    }

    #[test]
    fn triplet_loss_properties() {
        let a = emb(0.0);
        let p = emb(0.1); // close
        let n = emb(5.0); // far
        // d(a,p) << d(a,n) → loss should be 0 with a modest margin.
        assert_eq!(triplet_loss(&a, &p, &n, 0.5), 0.0);
        // Swap: positive far, negative close → positive loss.
        assert!(triplet_loss(&a, &n, &p, 0.5) > 0.0);
    }

    #[test]
    fn multitask_subset_ablation() {
        let mut heads = MultiTaskHeads::new();
        heads.push(LinearHead::zeros(TaskKind::Presence, 1));
        heads.push(LinearHead::zeros(TaskKind::Pose, 51));
        heads.push(LinearHead::zeros(TaskKind::Vitals, 2));
        assert_eq!(heads.forward(&emb(1.0)).len(), 3);
        // Ablate to just presence + vitals.
        let sub = heads.forward_subset(&emb(1.0), &[TaskKind::Presence, TaskKind::Vitals]);
        assert_eq!(sub.len(), 2);
        assert!(sub.iter().all(|o| o.task != TaskKind::Pose));
    }

    #[test]
    fn contrastive_batcher_samples_cross_env_positives() {
        // samples: 0=(stateA,room0) 1=(stateA,room1) 2=(stateB,room0)
        let b = ContrastiveBatcher::new(vec![0, 0, 1], vec![0, 1, 0]);
        let trips = b.triplets();
        // Anchor 0: positive=1 (same state, diff room), negative=2 (diff state).
        let t0 = trips.iter().find(|t| t.anchor == 0).unwrap();
        assert_eq!(t0.positive, 1);
        assert_eq!(t0.negative, 2);
        // Anchor 2 (stateB) has no same-state-diff-env positive → skipped.
        assert!(trips.iter().all(|t| t.anchor != 2));
        // Deterministic.
        assert_eq!(b.triplets().len(), trips.len());
    }

    #[test]
    fn seven_task_heads() {
        assert_eq!(TaskKind::ALL.len(), 7);
    }
}

//! Few-shot rapid adaptation (MERIDIAN Phase 5).
//!
//! Test-time training with contrastive learning and entropy minimization on
//! unlabeled CSI frames. Produces LoRA weight deltas for new environments.
//!
//! # Honesty note (ADR-155 §Tier-1.3)
//!
//! Earlier this module's `contrastive_step` / `entropy_step` wrote a *fake*
//! gradient (`grad += v * 0.01`) that did **not** descend the stated triplet /
//! entropy objective — so any "TTA improves the metric" claim was unsupported
//! by the code. That placeholder is gone. The two `*_loss` functions are now
//! pure evaluators of the real objective, and [`RapidAdaptation::adapt`]
//! descends them with a **finite-difference gradient** of that exact loss.
//! Finite differences genuinely minimize the stated objective (central
//! differences are accurate to O(ε²) truncation; see [`RapidAdaptation::adapt`]),
//! so "the adaptation loss decreases" is now a real, reproducible
//! measurement rather than an artefact of a hand-tuned fake step.
//!
//! **Scope caveat (still honest):** this minimizes a *self-supervised proxy*
//! (temporal-contrastive + prediction entropy) over a tiny LoRA bottleneck on
//! raw CSI frames. It is NOT yet wired to the pose model, and there is no
//! measured end-to-end PCK gain on WiFi pose from this path. ADR-155 records
//! TTA-on-pose as a future, not-yet-measured capability — do not cite a PCK
//! improvement from this module.

/// Loss function(s) for test-time adaptation.
#[derive(Debug, Clone)]
pub enum AdaptationLoss {
    /// Contrastive TTT: positive = temporally adjacent, negative = random.
    ContrastiveTTT {
        /// Gradient-descent epochs.
        epochs: usize,
        /// Learning rate.
        lr: f32,
    },
    /// Minimize entropy of confidence outputs for sharper predictions.
    EntropyMin {
        /// Gradient-descent epochs.
        epochs: usize,
        /// Learning rate.
        lr: f32,
    },
    /// Both contrastive and entropy losses combined.
    Combined {
        /// Gradient-descent epochs.
        epochs: usize,
        /// Learning rate.
        lr: f32,
        /// Weight for entropy term.
        lambda_ent: f32,
    },
}

impl AdaptationLoss {
    /// Number of epochs for this variant.
    pub fn epochs(&self) -> usize {
        match self {
            Self::ContrastiveTTT { epochs, .. }
            | Self::EntropyMin { epochs, .. }
            | Self::Combined { epochs, .. } => *epochs,
        }
    }
    /// Learning rate for this variant.
    pub fn lr(&self) -> f32 {
        match self {
            Self::ContrastiveTTT { lr, .. }
            | Self::EntropyMin { lr, .. }
            | Self::Combined { lr, .. } => *lr,
        }
    }
}

/// Result of [`RapidAdaptation::adapt`].
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// LoRA weight deltas.
    pub lora_weights: Vec<f32>,
    /// Final epoch loss.
    pub final_loss: f32,
    /// Calibration frames consumed.
    pub frames_used: usize,
    /// Epochs executed.
    pub adaptation_epochs: usize,
}

/// Error type for rapid adaptation.
#[derive(Debug, Clone)]
pub enum AdaptError {
    /// Not enough calibration frames.
    InsufficientFrames {
        /// Frames currently buffered.
        have: usize,
        /// Minimum required.
        need: usize,
    },
    /// LoRA rank must be at least 1.
    InvalidRank,
}

impl std::fmt::Display for AdaptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientFrames { have, need } => write!(
                f,
                "insufficient calibration frames: have {have}, need at least {need}"
            ),
            Self::InvalidRank => write!(f, "lora_rank must be >= 1"),
        }
    }
}

impl std::error::Error for AdaptError {}

/// Few-shot rapid adaptation engine.
///
/// Accumulates unlabeled CSI calibration frames and runs test-time training
/// to produce LoRA weight deltas. Buffer is capped at `max_buffer_frames`
/// (default 10 000) to prevent unbounded memory growth.
///
/// ```rust
/// use wifi_densepose_train::rapid_adapt::{RapidAdaptation, AdaptationLoss};
/// let loss = AdaptationLoss::Combined { epochs: 5, lr: 0.001, lambda_ent: 0.5 };
/// let mut ra = RapidAdaptation::new(10, 4, loss);
/// for i in 0..10 { ra.push_frame(&vec![i as f32; 8]); }
/// assert!(ra.is_ready());
/// let r = ra.adapt().unwrap();
/// assert_eq!(r.frames_used, 10);
/// ```
pub struct RapidAdaptation {
    /// Minimum frames before adaptation (default 200 = 10 s @ 20 Hz).
    pub min_calibration_frames: usize,
    /// LoRA factorization rank (must be >= 1).
    pub lora_rank: usize,
    /// Loss variant for test-time training.
    pub adaptation_loss: AdaptationLoss,
    /// Maximum buffer size (ring-buffer eviction beyond this cap).
    pub max_buffer_frames: usize,
    calibration_buffer: Vec<Vec<f32>>,
}

/// Default maximum calibration buffer size.
const DEFAULT_MAX_BUFFER: usize = 10_000;

impl RapidAdaptation {
    /// Create a new adaptation engine.
    pub fn new(
        min_calibration_frames: usize,
        lora_rank: usize,
        adaptation_loss: AdaptationLoss,
    ) -> Self {
        Self {
            min_calibration_frames,
            lora_rank,
            adaptation_loss,
            max_buffer_frames: DEFAULT_MAX_BUFFER,
            calibration_buffer: Vec::new(),
        }
    }
    /// Push a single unlabeled CSI frame. Evicts oldest frame when buffer is full.
    pub fn push_frame(&mut self, frame: &[f32]) {
        if self.calibration_buffer.len() >= self.max_buffer_frames {
            self.calibration_buffer.remove(0);
        }
        self.calibration_buffer.push(frame.to_vec());
    }
    /// True when buffer >= min_calibration_frames.
    pub fn is_ready(&self) -> bool {
        self.calibration_buffer.len() >= self.min_calibration_frames
    }
    /// Number of buffered frames.
    pub fn buffer_len(&self) -> usize {
        self.calibration_buffer.len()
    }

    /// Run test-time adaptation producing LoRA weight deltas.
    ///
    /// Returns an error if the calibration buffer is empty or lora_rank is 0.
    pub fn adapt(&self) -> Result<AdaptationResult, AdaptError> {
        if self.calibration_buffer.is_empty() {
            return Err(AdaptError::InsufficientFrames { have: 0, need: 1 });
        }
        if self.lora_rank == 0 {
            return Err(AdaptError::InvalidRank);
        }
        let (n, fdim) = (
            self.calibration_buffer.len(),
            self.calibration_buffer[0].len(),
        );
        let lora_sz = 2 * fdim * self.lora_rank;
        let mut w = vec![0.01_f32; lora_sz];
        let (epochs, lr) = (self.adaptation_loss.epochs(), self.adaptation_loss.lr());
        let mut final_loss = self.total_loss(&w, fdim);
        for _ in 0..epochs {
            // Real gradient of the *actual* objective via central finite
            // differences (ADR-155 §Tier-1.3). No hand-tuned fake step.
            let grad = self.finite_diff_grad(&w, fdim);
            for (wi, gi) in w.iter_mut().zip(grad.iter()) {
                *wi -= lr * gi;
            }
            final_loss = self.total_loss(&w, fdim);
        }
        Ok(AdaptationResult {
            lora_weights: w,
            final_loss,
            frames_used: n,
            adaptation_epochs: epochs,
        })
    }

    /// The scalar objective being minimized, for the active loss variant.
    fn total_loss(&self, w: &[f32], fdim: usize) -> f32 {
        match &self.adaptation_loss {
            AdaptationLoss::ContrastiveTTT { .. } => self.contrastive_loss(w, fdim),
            AdaptationLoss::EntropyMin { .. } => self.entropy_loss(w, fdim),
            AdaptationLoss::Combined { lambda_ent, .. } => {
                self.contrastive_loss(w, fdim) + lambda_ent * self.entropy_loss(w, fdim)
            }
        }
    }

    /// Central finite-difference gradient of [`Self::total_loss`] w.r.t. `w`.
    ///
    /// `∂L/∂wᵢ ≈ (L(w + ε eᵢ) − L(w − ε eᵢ)) / (2ε)`. This is the true gradient
    /// of the stated objective up to O(ε²) truncation — descending it genuinely
    /// reduces the loss (validated by the `*_loss_decreases` tests), unlike the
    /// removed `grad += v*0.01` placeholder which was unrelated to the loss.
    fn finite_diff_grad(&self, w: &[f32], fdim: usize) -> Vec<f32> {
        const EPS: f32 = 1e-3;
        let mut grad = vec![0.0_f32; w.len()];
        let mut wp = w.to_vec();
        for i in 0..w.len() {
            let orig = wp[i];
            wp[i] = orig + EPS;
            let lp = self.total_loss(&wp, fdim);
            wp[i] = orig - EPS;
            let lm = self.total_loss(&wp, fdim);
            wp[i] = orig;
            grad[i] = (lp - lm) / (2.0 * EPS);
        }
        grad
    }

    /// Temporal-contrastive triplet loss (pure evaluator — no gradient writes).
    ///
    /// Positive = temporally adjacent frame, negative = a half-buffer-away
    /// frame; margin-1 triplet hinge over the LoRA-projected features.
    fn contrastive_loss(&self, w: &[f32], fdim: usize) -> f32 {
        let n = self.calibration_buffer.len();
        if n < 2 {
            return 0.0;
        }
        let (margin, pairs) = (1.0_f32, n - 1);
        let mut total = 0.0_f32;
        for i in 0..pairs {
            let (anc, pos) = (&self.calibration_buffer[i], &self.calibration_buffer[i + 1]);
            let neg = &self.calibration_buffer[(i + n / 2) % n];
            let (pa, pp, pn) = (
                self.project(anc, w, fdim),
                self.project(pos, w, fdim),
                self.project(neg, w, fdim),
            );
            total += (l2_dist(&pa, &pp) - l2_dist(&pa, &pn) + margin).max(0.0);
        }
        total / pairs as f32
    }

    /// Prediction-entropy loss (pure evaluator — no gradient writes).
    fn entropy_loss(&self, w: &[f32], fdim: usize) -> f32 {
        let n = self.calibration_buffer.len();
        if n == 0 {
            return 0.0;
        }
        let nc = self.lora_rank.max(2);
        let mut total = 0.0_f32;
        for frame in &self.calibration_buffer {
            let proj = self.project(frame, w, fdim);
            let mut logits = vec![0.0_f32; nc];
            for (i, &v) in proj.iter().enumerate() {
                logits[i % nc] += v;
            }
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            total += exps
                .iter()
                .map(|&e| {
                    let p = e / s;
                    if p > 1e-10 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f32>();
        }
        total / n as f32
    }

    fn project(&self, frame: &[f32], w: &[f32], fdim: usize) -> Vec<f32> {
        let rank = self.lora_rank;
        let mut hidden = vec![0.0_f32; rank];
        for (r, hr) in hidden.iter_mut().enumerate() {
            #[allow(clippy::needless_range_loop)]
            for d in 0..fdim.min(frame.len()) {
                let idx = d * rank + r;
                if idx < w.len() {
                    *hr += w[idx] * frame[d];
                }
            }
        }
        let boff = fdim * rank;
        (0..fdim)
            .map(|d| {
                let lora: f32 = (0..rank)
                    .map(|r| {
                        let idx = boff + r * fdim + d;
                        if idx < w.len() {
                            w[idx] * hidden[r]
                        } else {
                            0.0
                        }
                    })
                    .sum();
                frame.get(d).copied().unwrap_or(0.0) + lora
            })
            .collect()
    }
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_frame_accumulates() {
        let mut a = RapidAdaptation::new(
            5,
            4,
            AdaptationLoss::ContrastiveTTT {
                epochs: 1,
                lr: 0.01,
            },
        );
        assert_eq!(a.buffer_len(), 0);
        a.push_frame(&[1.0, 2.0]);
        assert_eq!(a.buffer_len(), 1);
        a.push_frame(&[3.0, 4.0]);
        assert_eq!(a.buffer_len(), 2);
    }

    #[test]
    fn is_ready_threshold() {
        let mut a = RapidAdaptation::new(
            5,
            4,
            AdaptationLoss::EntropyMin {
                epochs: 3,
                lr: 0.001,
            },
        );
        for i in 0..4 {
            a.push_frame(&[i as f32; 8]);
            assert!(!a.is_ready());
        }
        a.push_frame(&[99.0; 8]);
        assert!(a.is_ready());
        a.push_frame(&[100.0; 8]);
        assert!(a.is_ready());
    }

    #[test]
    fn adapt_lora_weight_dimension() {
        let (fdim, rank) = (16, 4);
        let mut a = RapidAdaptation::new(
            10,
            rank,
            AdaptationLoss::ContrastiveTTT {
                epochs: 3,
                lr: 0.01,
            },
        );
        for i in 0..10 {
            a.push_frame(&vec![i as f32 * 0.1; fdim]);
        }
        let r = a.adapt().unwrap();
        assert_eq!(r.lora_weights.len(), 2 * fdim * rank);
        assert_eq!(r.frames_used, 10);
        assert_eq!(r.adaptation_epochs, 3);
    }

    #[test]
    fn contrastive_loss_decreases() {
        // ADR-155 §Tier-1.3: with REAL finite-difference gradients of the actual
        // triplet objective, more optimisation must not increase the loss.
        let (fdim, rank) = (32, 4);
        let mk = |ep| {
            let mut a = RapidAdaptation::new(
                20,
                rank,
                AdaptationLoss::ContrastiveTTT {
                    epochs: ep,
                    lr: 0.05,
                },
            );
            for i in 0..20 {
                let v = i as f32 * 0.1;
                a.push_frame(&(0..fdim).map(|d| v + d as f32 * 0.01).collect::<Vec<_>>());
            }
            a.adapt().unwrap().final_loss
        };
        let l0 = mk(0); // no optimisation: loss at the initial weights
        let l20 = mk(20); // 20 real gradient steps
        assert!(
            l20 <= l0 + 1e-6,
            "20 gradient steps must not increase the contrastive loss: l0={l0}, l20={l20}"
        );
    }

    #[test]
    fn entropy_loss_decreases() {
        // ADR-155 §Tier-1.3: entropy minimisation must actually reduce entropy.
        let (fdim, rank) = (16, 4);
        let mk = |ep| {
            let mut a = RapidAdaptation::new(
                10,
                rank,
                AdaptationLoss::EntropyMin {
                    epochs: ep,
                    lr: 0.05,
                },
            );
            for i in 0..10 {
                a.push_frame(
                    &(0..fdim)
                        .map(|d| ((i * fdim + d) as f32).sin())
                        .collect::<Vec<_>>(),
                );
            }
            a.adapt().unwrap().final_loss
        };
        let l0 = mk(0);
        let l30 = mk(30);
        assert!(
            l30 <= l0 + 1e-6,
            "entropy minimisation must not increase entropy: l0={l0}, l30={l30}"
        );
    }

    #[test]
    fn reported_loss_is_the_real_objective_not_a_placeholder() {
        // The returned final_loss must equal an independent recomputation of the
        // contrastive objective at the produced LoRA weights — i.e. it is the
        // real loss, not a fabricated number (ADR-155 §Tier-1.3).
        let (fdim, rank) = (16, 4);
        let mut a = RapidAdaptation::new(
            8,
            rank,
            AdaptationLoss::ContrastiveTTT {
                epochs: 3,
                lr: 0.02,
            },
        );
        for i in 0..8 {
            a.push_frame(&(0..fdim).map(|d| (i + d) as f32 * 0.05).collect::<Vec<_>>());
        }
        let r = a.adapt().unwrap();
        let recomputed = a.contrastive_loss(&r.lora_weights, fdim);
        assert!(
            (r.final_loss - recomputed).abs() < 1e-5,
            "final_loss {} must match the real objective {} at the output weights",
            r.final_loss,
            recomputed
        );
    }

    #[test]
    fn combined_loss_adaptation() {
        let (fdim, rank) = (16, 4);
        let mut a = RapidAdaptation::new(
            10,
            rank,
            AdaptationLoss::Combined {
                epochs: 5,
                lr: 0.001,
                lambda_ent: 0.5,
            },
        );
        for i in 0..10 {
            a.push_frame(
                &(0..fdim)
                    .map(|d| ((i * fdim + d) as f32).sin())
                    .collect::<Vec<_>>(),
            );
        }
        let r = a.adapt().unwrap();
        assert_eq!(r.frames_used, 10);
        assert_eq!(r.adaptation_epochs, 5);
        assert!(r.final_loss.is_finite());
        assert_eq!(r.lora_weights.len(), 2 * fdim * rank);
        assert!(r.lora_weights.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn adapt_empty_buffer_returns_error() {
        let a = RapidAdaptation::new(
            10,
            4,
            AdaptationLoss::ContrastiveTTT {
                epochs: 1,
                lr: 0.01,
            },
        );
        assert!(a.adapt().is_err());
    }

    #[test]
    fn adapt_zero_rank_returns_error() {
        let mut a = RapidAdaptation::new(
            1,
            0,
            AdaptationLoss::ContrastiveTTT {
                epochs: 1,
                lr: 0.01,
            },
        );
        a.push_frame(&[1.0, 2.0]);
        assert!(a.adapt().is_err());
    }

    #[test]
    fn buffer_cap_evicts_oldest() {
        let mut a = RapidAdaptation::new(
            2,
            4,
            AdaptationLoss::ContrastiveTTT {
                epochs: 1,
                lr: 0.01,
            },
        );
        a.max_buffer_frames = 3;
        for i in 0..5 {
            a.push_frame(&[i as f32]);
        }
        assert_eq!(a.buffer_len(), 3);
    }

    #[test]
    fn l2_distance_tests() {
        assert!(l2_dist(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).abs() < 1e-10);
        assert!((l2_dist(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn loss_accessors() {
        let c = AdaptationLoss::ContrastiveTTT {
            epochs: 7,
            lr: 0.02,
        };
        assert_eq!(c.epochs(), 7);
        assert!((c.lr() - 0.02).abs() < 1e-7);
        let e = AdaptationLoss::EntropyMin { epochs: 3, lr: 0.1 };
        assert_eq!(e.epochs(), 3);
        assert!((e.lr() - 0.1).abs() < 1e-7);
        let cb = AdaptationLoss::Combined {
            epochs: 5,
            lr: 0.001,
            lambda_ent: 0.3,
        };
        assert_eq!(cb.epochs(), 5);
        assert!((cb.lr() - 0.001).abs() < 1e-7);
    }
}

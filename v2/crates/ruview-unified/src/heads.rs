//! Task adapters ("heads") on the frozen encoder representation `z`
//! (ADR-274 §4).
//!
//! The ADR-273 acceptance test allows adapters **smaller than 1 % of the
//! backbone parameters** — enforced here by [`within_adapter_budget`] and by
//! a unit test over every head at the deployment config. Multi-class heads
//! use a low-rank (LoRA-style) factorization to stay inside the budget.

use crate::math::{sigmoid, softmax};

/// True iff the adapter fits the ADR-273 budget: `params(adapter) <
/// 1 % · params(backbone)`.
#[must_use]
pub fn within_adapter_budget(backbone_params: usize, adapter_params: usize) -> bool {
    adapter_params * 100 < backbone_params
}

/// Binary presence head: logistic regression on `z`.
#[derive(Debug, Clone)]
pub struct PresenceHead {
    /// Weights, length `d_model`.
    pub w: Vec<f64>,
    /// Bias.
    pub b: f64,
}

impl PresenceHead {
    /// Zero-initialized head (logistic regression is convex; zero init is
    /// the standard, deterministic choice).
    #[must_use]
    pub fn new(d_model: usize) -> Self {
        Self { w: vec![0.0; d_model], b: 0.0 }
    }

    /// Parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.w.len() + 1
    }

    /// P(present | z).
    #[must_use]
    pub fn predict_prob(&self, z: &[f64]) -> f64 {
        sigmoid(self.w.iter().zip(z).map(|(w, x)| w * x).sum::<f64>() + self.b)
    }

    /// Full-batch gradient-descent training (convex ⇒ deterministic
    /// convergence). Returns the final mean cross-entropy.
    pub fn train(&mut self, zs: &[Vec<f64>], labels: &[bool], lr: f64, epochs: usize) -> f64 {
        assert_eq!(zs.len(), labels.len());
        let n = zs.len() as f64;
        let mut final_ce = f64::INFINITY;
        for _ in 0..epochs {
            let mut gw = vec![0.0; self.w.len()];
            let mut gb = 0.0;
            let mut ce = 0.0;
            for (z, &y) in zs.iter().zip(labels) {
                let p = self.predict_prob(z);
                let t = if y { 1.0 } else { 0.0 };
                ce -= t * p.max(1e-12).ln() + (1.0 - t) * (1.0 - p).max(1e-12).ln();
                let err = (p - t) / n;
                for (g, x) in gw.iter_mut().zip(z) {
                    *g += err * x;
                }
                gb += err;
            }
            for (w, g) in self.w.iter_mut().zip(&gw) {
                *w -= lr * g;
            }
            self.b -= lr * gb;
            final_ce = ce / n;
        }
        final_ce
    }
}

/// Low-rank multi-class head: `logits = U·(V·z) + b` with rank `r`
/// (LoRA-style factorization keeps `C`-class heads inside the 1 % budget).
#[derive(Debug, Clone)]
pub struct ActivityHead {
    /// Classes.
    pub classes: usize,
    /// Rank.
    pub rank: usize,
    /// `classes × rank`, row-major.
    pub u: Vec<f64>,
    /// `rank × d_model`, row-major.
    pub v: Vec<f64>,
    /// Per-class bias.
    pub b: Vec<f64>,
    d_model: usize,
}

impl ActivityHead {
    /// Deterministic small-value init (breaks the U/V symmetry without RNG).
    #[must_use]
    pub fn new(d_model: usize, classes: usize, rank: usize) -> Self {
        let u = (0..classes * rank)
            .map(|i| 0.01 * ((i % 7) as f64 - 3.0))
            .collect();
        let v = (0..rank * d_model)
            .map(|i| 0.01 * ((i % 5) as f64 - 2.0))
            .collect();
        Self { classes, rank, u, v, b: vec![0.0; classes], d_model }
    }

    /// Parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.u.len() + self.v.len() + self.b.len()
    }

    /// Class probabilities.
    #[must_use]
    pub fn predict(&self, z: &[f64]) -> Vec<f64> {
        let mut probs = self.logits(z);
        softmax(&mut probs);
        probs
    }

    fn logits(&self, z: &[f64]) -> Vec<f64> {
        let mut vz = vec![0.0; self.rank];
        for r in 0..self.rank {
            let row = &self.v[r * self.d_model..(r + 1) * self.d_model];
            vz[r] = row.iter().zip(z).map(|(a, b)| a * b).sum();
        }
        let mut logits = self.b.clone();
        for c in 0..self.classes {
            let row = &self.u[c * self.rank..(c + 1) * self.rank];
            logits[c] += row.iter().zip(&vz).map(|(a, b)| a * b).sum::<f64>();
        }
        logits
    }

    /// Full-batch softmax cross-entropy training. Returns final mean CE.
    pub fn train(&mut self, zs: &[Vec<f64>], labels: &[usize], lr: f64, epochs: usize) -> f64 {
        assert_eq!(zs.len(), labels.len());
        let n = zs.len() as f64;
        let mut final_ce = f64::INFINITY;
        for _ in 0..epochs {
            let mut gu = vec![0.0; self.u.len()];
            let mut gv = vec![0.0; self.v.len()];
            let mut gb = vec![0.0; self.b.len()];
            let mut ce = 0.0;
            for (z, &y) in zs.iter().zip(labels) {
                let mut vz = vec![0.0; self.rank];
                for r in 0..self.rank {
                    let row = &self.v[r * self.d_model..(r + 1) * self.d_model];
                    vz[r] = row.iter().zip(z).map(|(a, b)| a * b).sum();
                }
                let mut p = self.b.clone();
                for c in 0..self.classes {
                    let row = &self.u[c * self.rank..(c + 1) * self.rank];
                    p[c] += row.iter().zip(&vz).map(|(a, b)| a * b).sum::<f64>();
                }
                softmax(&mut p);
                ce -= p[y].max(1e-12).ln();
                // dlogits = p − one_hot(y), scaled by 1/n.
                let mut dvz = vec![0.0; self.rank];
                for c in 0..self.classes {
                    let dl = (p[c] - if c == y { 1.0 } else { 0.0 }) / n;
                    gb[c] += dl;
                    for r in 0..self.rank {
                        gu[c * self.rank + r] += dl * vz[r];
                        dvz[r] += dl * self.u[c * self.rank + r];
                    }
                }
                for r in 0..self.rank {
                    for (d, zv) in z.iter().enumerate() {
                        gv[r * self.d_model + d] += dvz[r] * zv;
                    }
                }
            }
            for (w, g) in self.u.iter_mut().zip(&gu) {
                *w -= lr * g;
            }
            for (w, g) in self.v.iter_mut().zip(&gv) {
                *w -= lr * g;
            }
            for (w, g) in self.b.iter_mut().zip(&gb) {
                *w -= lr * g;
            }
            final_ce = ce / n;
        }
        final_ce
    }
}

/// Linear localization head: `xyz = W·z + b` (metres).
#[derive(Debug, Clone)]
pub struct LocalizationHead {
    /// `3 × d_model` weights.
    pub w: Vec<f64>,
    /// Bias.
    pub b: [f64; 3],
    d_model: usize,
}

impl LocalizationHead {
    /// Zero-initialized (linear least squares; convex).
    #[must_use]
    pub fn new(d_model: usize) -> Self {
        Self { w: vec![0.0; 3 * d_model], b: [0.0; 3], d_model }
    }

    /// Parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.w.len() + 3
    }

    /// Predicted position.
    #[must_use]
    pub fn predict(&self, z: &[f64]) -> [f64; 3] {
        let mut out = self.b;
        for (axis, o) in out.iter_mut().enumerate() {
            let row = &self.w[axis * self.d_model..(axis + 1) * self.d_model];
            *o += row.iter().zip(z).map(|(a, b)| a * b).sum::<f64>();
        }
        out
    }

    /// Full-batch MSE training. Returns final mean squared error (m²).
    pub fn train(&mut self, zs: &[Vec<f64>], targets: &[[f64; 3]], lr: f64, epochs: usize) -> f64 {
        assert_eq!(zs.len(), targets.len());
        let n = zs.len() as f64;
        let mut final_mse = f64::INFINITY;
        for _ in 0..epochs {
            let mut gw = vec![0.0; self.w.len()];
            let mut gb = [0.0; 3];
            let mut mse = 0.0;
            for (z, t) in zs.iter().zip(targets) {
                let pred = self.predict(z);
                for axis in 0..3 {
                    let err = pred[axis] - t[axis];
                    mse += err * err;
                    let scale = 2.0 * err / (3.0 * n);
                    gb[axis] += scale;
                    for (d, zv) in z.iter().enumerate() {
                        gw[axis * self.d_model + d] += scale * zv;
                    }
                }
            }
            for (w, g) in self.w.iter_mut().zip(&gw) {
                *w -= lr * g;
            }
            for (axis, g) in gb.iter().enumerate() {
                self.b[axis] -= lr * g;
            }
            final_mse = mse / (3.0 * n);
        }
        final_mse
    }
}

/// Anomaly head: z-scores the encoder's masked-reconstruction error against
/// a calibration distribution of *normal* windows. No learned weights —
/// two calibration statistics.
#[derive(Debug, Clone)]
pub struct AnomalyHead {
    /// Calibration mean of reconstruction error.
    pub mean: f64,
    /// Calibration standard deviation.
    pub std: f64,
}

impl AnomalyHead {
    /// Calibrates from reconstruction errors of known-normal windows.
    ///
    /// # Panics
    /// If fewer than 2 calibration samples are provided.
    #[must_use]
    pub fn calibrate(normal_errors: &[f64]) -> Self {
        assert!(normal_errors.len() >= 2, "need >= 2 calibration errors");
        let n = normal_errors.len() as f64;
        let mean = normal_errors.iter().sum::<f64>() / n;
        let var = normal_errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Self { mean, std: var.sqrt().max(1e-12) }
    }

    /// Parameter count (two statistics).
    #[must_use]
    pub fn param_count(&self) -> usize {
        2
    }

    /// Anomaly z-score for a window's reconstruction error.
    #[must_use]
    pub fn score(&self, recon_error: f64) -> f64 {
        (recon_error - self.mean) / self.std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{EncoderConfig, RfEncoder};

    #[test]
    fn every_head_fits_the_one_percent_budget_at_deployment_config() {
        let enc = RfEncoder::new(EncoderConfig::default(), 1);
        let backbone = enc.param_count();
        let d = EncoderConfig::default().d_model;

        let presence = PresenceHead::new(d).param_count();
        let activity = ActivityHead::new(d, 4, 2).param_count();
        let localization = LocalizationHead::new(d).param_count();
        let anomaly = AnomalyHead { mean: 0.0, std: 1.0 }.param_count();

        for (name, p) in [
            ("presence", presence),
            ("activity", activity),
            ("localization", localization),
            ("anomaly", anomaly),
        ] {
            assert!(
                within_adapter_budget(backbone, p),
                "{name} head has {p} params, budget is <{}",
                backbone / 100
            );
        }
        println!(
            "backbone {backbone} params; heads: presence {presence}, activity {activity}, \
             localization {localization}, anomaly {anomaly} (budget < {})",
            backbone / 100
        );
    }

    fn separable_data(n: usize, d: usize) -> (Vec<Vec<f64>>, Vec<bool>) {
        let mut zs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..n {
            let y = i % 2 == 0;
            let offset = if y { 1.0 } else { -1.0 };
            let z: Vec<f64> =
                (0..d).map(|k| offset * (0.5 + (k as f64 / d as f64)) + 0.1 * ((i * k) % 3) as f64).collect();
            zs.push(z);
            ys.push(y);
        }
        (zs, ys)
    }

    #[test]
    fn presence_head_learns_separable_data() {
        let (zs, ys) = separable_data(60, 16);
        let mut head = PresenceHead::new(16);
        let ce = head.train(&zs, &ys, 0.5, 200);
        assert!(ce < 0.1, "cross-entropy should collapse on separable data, got {ce}");
        let correct = zs
            .iter()
            .zip(&ys)
            .filter(|(z, &y)| (head.predict_prob(z) > 0.5) == y)
            .count();
        assert_eq!(correct, zs.len());
    }

    #[test]
    fn activity_head_learns_multiclass_toy() {
        let d = 16;
        let mut zs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..90 {
            let c = i % 3;
            let z: Vec<f64> = (0..d)
                .map(|k| if k % 3 == c { 1.0 } else { 0.0 } + 0.05 * ((i + k) % 5) as f64)
                .collect();
            zs.push(z);
            ys.push(c);
        }
        let mut head = ActivityHead::new(d, 3, 2);
        let ce = head.train(&zs, &ys, 0.5, 400);
        assert!(ce < 0.3, "multiclass CE should drop, got {ce}");
        let acc = zs
            .iter()
            .zip(&ys)
            .filter(|(z, &y)| {
                let p = head.predict(z);
                p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == y
            })
            .count() as f64
            / zs.len() as f64;
        assert!(acc > 0.95, "accuracy {acc}");
    }

    #[test]
    fn anomaly_head_zscores_against_calibration() {
        let normal: Vec<f64> = (0..50).map(|i| 0.10 + 0.001 * (i % 7) as f64).collect();
        let head = AnomalyHead::calibrate(&normal);
        assert!(head.score(0.103).abs() < 3.0, "in-distribution error must not alarm");
        assert!(head.score(0.5) > 10.0, "gross error must alarm loudly");
    }
}

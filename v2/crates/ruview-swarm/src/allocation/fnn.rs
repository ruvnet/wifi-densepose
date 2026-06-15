//! Lightweight 3-layer FNN bid scorer — pure Rust, no ONNX required.

/// 3-layer FNN: 5 inputs → 16 hidden (ReLU) → 8 hidden (ReLU) → 1 output (sigmoid).
pub struct FnnScorer {
    pub w1: [[f32; 5]; 16],
    pub b1: [f32; 16],
    pub w2: [[f32; 16]; 8],
    pub b2: [f32; 8],
    pub w3: [f32; 8],
    pub b3: f32,
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl FnnScorer {
    /// Score a feature vector. Returns sigmoid(output) ∈ [0, 1].
    /// Features: [dist_norm, battery_norm, link_quality, csi_confidence, workload_norm]
    pub fn score(&self, features: [f32; 5]) -> f32 {
        // Layer 1: 5 → 16 (ReLU)
        let mut h1 = [0.0f32; 16];
        for (i, row) in self.w1.iter().enumerate() {
            let z: f32 = row.iter().zip(features.iter()).map(|(w, x)| w * x).sum();
            h1[i] = relu(z + self.b1[i]);
        }

        // Layer 2: 16 → 8 (ReLU)
        let mut h2 = [0.0f32; 8];
        for (i, row) in self.w2.iter().enumerate() {
            let z: f32 = row.iter().zip(h1.iter()).map(|(w, x)| w * x).sum();
            h2[i] = relu(z + self.b2[i]);
        }

        // Layer 3: 8 → 1 (sigmoid)
        let z3: f32 = self.w3.iter().zip(h2.iter()).map(|(w, x)| w * x).sum::<f32>() + self.b3;
        sigmoid(z3)
    }

    /// Default weights initialised to a simple identity-like setup.
    pub fn default_weights() -> Self {
        // Simple: w1 diagonalish, others small constant
        // Index needed: diagonal/strided init uses i for both row and column.
        let mut w1 = [[0.0f32; 5]; 16];
        #[allow(clippy::needless_range_loop)]
        for i in 0..5 {
            w1[i][i] = 1.0;
        }
        for row in w1.iter_mut().take(16).skip(5) {
            row[0] = 0.1;
        }
        let mut w2 = [[0.0f32; 16]; 8];
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            w2[i][i * 2] = 1.0;
        }
        let w3 = [0.125f32; 8];
        Self {
            w1,
            b1: [0.0; 16],
            w2,
            b2: [0.0; 8],
            w3,
            b3: 0.0,
        }
    }
}

impl Default for FnnScorer {
    fn default() -> Self {
        Self::default_weights()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_in_unit_interval() {
        let scorer = FnnScorer::default_weights();
        let features = [0.3f32, 0.8, 0.9, 0.75, 0.2];
        let s = scorer.score(features);
        assert!(s >= 0.0 && s <= 1.0, "score {s} out of [0,1]");
    }

    #[test]
    fn test_score_deterministic() {
        let scorer = FnnScorer::default_weights();
        let f = [0.5f32; 5];
        assert_eq!(scorer.score(f), scorer.score(f));
    }
}

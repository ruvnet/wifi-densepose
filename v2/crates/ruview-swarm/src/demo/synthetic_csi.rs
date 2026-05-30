//! Synthetic CSI generator — simulates WiFi CSI victim detections without hardware.
//!
//! Uses exponential distance decay and configurable Gaussian noise to produce
//! realistic CsiDetection events for scenario testing and demo mode.

use rand::Rng;
use crate::types::{CsiDetection, NodeId, Position3D};

/// Generates synthetic CSI detection events for a set of victim positions.
pub struct SyntheticCsiGenerator {
    /// Ground-truth victim positions in NED metres.
    pub victims: Vec<Position3D>,
    /// Std-dev of additive Gaussian noise on confidence and position estimate.
    pub noise_std: f64,
    /// Maximum range (metres) at which a drone can detect a victim.
    pub detection_range_m: f64,
}

impl SyntheticCsiGenerator {
    pub fn new(victims: Vec<Position3D>, noise_std: f64, detection_range_m: f64) -> Self {
        Self { victims, noise_std, detection_range_m }
    }

    /// Attempt to detect a victim from the given drone position.
    ///
    /// Returns the strongest detection within range, or `None` if no victim
    /// is within `detection_range_m`.  Confidence is modelled as
    /// `exp(-dist / range)` plus zero-mean Gaussian noise.
    pub fn detect(
        &self,
        drone_id: NodeId,
        drone_pos: &Position3D,
        timestamp_ms: u64,
    ) -> Option<CsiDetection> {
        let mut rng = rand::thread_rng();
        let mut best: Option<CsiDetection> = None;

        for victim in &self.victims {
            let dist = drone_pos.distance_to(victim);
            if dist >= self.detection_range_m {
                continue;
            }
            // Exponential decay: full confidence at 0 m, ~37% at 1× range
            let base_conf = (-dist / self.detection_range_m).exp();
            let noise: f64 = rng.gen_range(-self.noise_std..self.noise_std);
            let confidence = (base_conf + noise).clamp(0.0, 1.0) as f32;

            if confidence <= 0.4 {
                continue;
            }

            // Add positional noise proportional to noise_std
            let pos_jitter = self.noise_std * 10.0;
            let est_pos = Position3D {
                x: victim.x + rng.gen_range(-pos_jitter..pos_jitter),
                y: victim.y + rng.gen_range(-pos_jitter..pos_jitter),
                z: victim.z,
            };

            let det = CsiDetection {
                drone_id,
                confidence,
                victim_position: Some(est_pos),
                timestamp_ms,
            };

            // Keep the highest-confidence detection
            match &best {
                None => best = Some(det),
                Some(b) if det.confidence > b.confidence => best = Some(det),
                _ => {}
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_close_victim() {
        // A victim right on the drone should nearly always return a detection.
        // Run 20 trials; at least 15 should detect (0.4 threshold at distance 0).
        let gen = SyntheticCsiGenerator::new(
            vec![Position3D { x: 0.0, y: 0.0, z: 0.0 }],
            0.01,
            28.0,
        );
        let mut hits = 0u32;
        for i in 0..20 {
            if gen.detect(NodeId(0), &Position3D::zero(), i as u64).is_some() {
                hits += 1;
            }
        }
        assert!(hits >= 15, "expected ≥15/20 detections at zero range, got {hits}");
    }

    #[test]
    fn test_detect_beyond_range_returns_none() {
        let gen = SyntheticCsiGenerator::new(
            vec![Position3D { x: 0.0, y: 0.0, z: 0.0 }],
            0.01,
            28.0,
        );
        let far_pos = Position3D { x: 1000.0, y: 1000.0, z: 0.0 };
        // All 10 attempts should return None since drone is 1414 m away.
        for i in 0..10 {
            assert!(
                gen.detect(NodeId(0), &far_pos, i).is_none(),
                "expected no detection at 1414 m"
            );
        }
    }

    #[test]
    fn test_best_of_two_victims_returned() {
        // Two victims: one very close (high conf), one just at boundary (low conf).
        let gen = SyntheticCsiGenerator::new(
            vec![
                Position3D { x: 1.0,  y: 0.0, z: 0.0 }, // close
                Position3D { x: 27.0, y: 0.0, z: 0.0 }, // near boundary
            ],
            0.01,
            28.0,
        );
        // Run 10 trials; whenever both return a detection the close one should win.
        for i in 0..10 {
            if let Some(det) = gen.detect(NodeId(0), &Position3D::zero(), i) {
                assert!(
                    det.confidence >= 0.4,
                    "returned confidence {:.3} is below threshold",
                    det.confidence
                );
            }
        }
    }
}

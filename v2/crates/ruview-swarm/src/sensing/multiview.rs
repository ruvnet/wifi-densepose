use crate::types::{NodeId, Position3D, CsiDetection};

/// A fused detection result from multiple drone viewpoints.
#[derive(Debug, Clone)]
pub struct FusedDetection {
    pub confidence: f32,
    pub estimated_position: Position3D,
    pub contributing_drones: Vec<NodeId>,
    /// Localization uncertainty ellipse (std dev in metres).
    pub uncertainty_m: f64,
}

/// Geometric diversity metric (Cramer-Rao bound proxy).
/// More diverse viewpoints -> lower bound -> better localization.
fn geometric_diversity_index(positions: &[Position3D]) -> f64 {
    if positions.len() < 2 {
        return 0.0;
    }
    // Compute average pairwise angular separation
    let n = positions.len();
    let centroid = Position3D {
        x: positions.iter().map(|p| p.x).sum::<f64>() / n as f64,
        y: positions.iter().map(|p| p.y).sum::<f64>() / n as f64,
        z: positions.iter().map(|p| p.z).sum::<f64>() / n as f64,
    };

    let mut total_angle = 0.0_f64;
    let mut pairs = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = (positions[i].x - centroid.x, positions[i].y - centroid.y);
            let b = (positions[j].x - centroid.x, positions[j].y - centroid.y);
            let dot = a.0 * b.0 + a.1 * b.1;
            let mag_a = (a.0 * a.0 + a.1 * a.1).sqrt().max(1e-9);
            let mag_b = (b.0 * b.0 + b.1 * b.1).sqrt().max(1e-9);
            let cos_angle = (dot / (mag_a * mag_b)).clamp(-1.0, 1.0);
            total_angle += cos_angle.acos();
            pairs += 1;
        }
    }

    if pairs > 0 { total_angle / pairs as f64 } else { 0.0 }
}

/// Multi-drone CSI fusion via confidence-weighted position averaging with geometric bias.
pub struct MultiViewFusion {
    /// Minimum number of independent viewpoints required to produce a fused result.
    pub min_viewpoints: usize,
    /// Minimum confidence of individual detections to include in fusion.
    pub min_confidence: f32,
}

impl Default for MultiViewFusion {
    fn default() -> Self {
        Self { min_viewpoints: 2, min_confidence: 0.5 }
    }
}

impl MultiViewFusion {
    /// Fuse multiple CSI detections from different drone viewpoints.
    /// Returns None if fewer than min_viewpoints pass the confidence threshold.
    pub fn fuse(
        &self,
        detections: &[CsiDetection],
        drone_positions: &[(NodeId, Position3D)],
    ) -> Option<FusedDetection> {
        // Filter by confidence and require estimated position
        let valid: Vec<(&CsiDetection, &Position3D)> = detections
            .iter()
            .filter(|d| d.confidence >= self.min_confidence && d.victim_position.is_some())
            .filter_map(|d| {
                let drone_pos = drone_positions
                    .iter()
                    .find(|(id, _)| *id == d.drone_id)
                    .map(|(_, p)| p)?;
                Some((d, drone_pos))
            })
            .collect();

        if valid.len() < self.min_viewpoints {
            return None;
        }

        // Compute geometric diversity index for uncertainty estimate
        let drone_pos_list: Vec<Position3D> = valid.iter().map(|(_, p)| **p).collect();
        let gdi = geometric_diversity_index(&drone_pos_list);

        // Weighted average of victim position estimates
        let total_weight: f32 = valid.iter().map(|(d, _)| d.confidence).sum();
        let mut fused_x = 0.0_f64;
        let mut fused_y = 0.0_f64;
        let mut fused_z = 0.0_f64;
        let mut fused_conf = 0.0_f32;

        for (det, _) in &valid {
            let w = det.confidence / total_weight;
            let vp = det.victim_position.unwrap();
            fused_x += w as f64 * vp.x;
            fused_y += w as f64 * vp.y;
            fused_z += w as f64 * vp.z;
            fused_conf += w * det.confidence;
        }

        // Uncertainty shrinks with geometric diversity and number of viewpoints:
        // baseline 5 m (single drone) -> scales down by sqrt(n) and gdi factor
        let base_uncertainty_m = 5.0;
        let n = valid.len() as f64;
        let gdi_factor = (1.0 + gdi / std::f64::consts::PI).clamp(1.0, 2.0);
        let uncertainty_m = base_uncertainty_m / (n.sqrt() * gdi_factor);

        Some(FusedDetection {
            confidence: fused_conf,
            estimated_position: Position3D { x: fused_x, y: fused_y, z: fused_z },
            contributing_drones: valid.iter().map(|(d, _)| d.drone_id).collect(),
            uncertainty_m,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_single_view_insufficient() {
        let fusion = MultiViewFusion { min_viewpoints: 2, min_confidence: 0.5 };
        let det = CsiDetection {
            drone_id: NodeId(0),
            confidence: 0.9,
            victim_position: Some(Position3D { x: 10.0, y: 5.0, z: 0.0 }),
            timestamp_ms: 0,
        };
        let result = fusion.fuse(&[det], &[(NodeId(0), Position3D::zero())]);
        assert!(result.is_none(), "single viewpoint should not produce fusion");
    }

    #[test]
    fn test_fusion_three_views() {
        let fusion = MultiViewFusion::default();
        let victim = Position3D { x: 50.0, y: 50.0, z: 0.0 };
        let detections = vec![
            CsiDetection {
                drone_id: NodeId(0),
                confidence: 0.85,
                victim_position: Some(Position3D { x: 51.0, y: 49.0, z: 0.0 }),
                timestamp_ms: 0,
            },
            CsiDetection {
                drone_id: NodeId(1),
                confidence: 0.78,
                victim_position: Some(Position3D { x: 49.0, y: 51.0, z: 0.0 }),
                timestamp_ms: 0,
            },
            CsiDetection {
                drone_id: NodeId(2),
                confidence: 0.92,
                victim_position: Some(Position3D { x: 50.0, y: 50.0, z: 0.0 }),
                timestamp_ms: 0,
            },
        ];
        let positions = vec![
            (NodeId(0), Position3D { x: 0.0, y: 0.0, z: -30.0 }),
            (NodeId(1), Position3D { x: 100.0, y: 0.0, z: -30.0 }),
            (NodeId(2), Position3D { x: 50.0, y: 86.6, z: -30.0 }), // equilateral triangle
        ];

        let result = fusion.fuse(&detections, &positions).unwrap();
        let err = result.estimated_position.distance_to(&victim);
        assert!(
            err < 3.0,
            "fusion error {} m should be < 3 m for 3 equilateral viewpoints",
            err
        );
        assert!(
            result.uncertainty_m < 5.0,
            "uncertainty {} should be < 5 m single-drone baseline",
            result.uncertainty_m
        );
    }
}

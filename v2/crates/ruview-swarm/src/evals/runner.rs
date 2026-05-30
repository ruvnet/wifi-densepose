//! Stage-1 kinematic rollout + seed × episode matrix (ADR-149).
//!
//! A single `run_episode` deterministically drives `drones` drones across a
//! mission area under a chosen [`FlightPattern`], marks coverage on a grid,
//! simulates CSI victim detection perturbed by `(sigma, kappa)` amplitude /
//! von-Mises-phase noise, and computes the GDOP of the contributing-drone
//! constellation at first detection. It is self-contained and seeded — no
//! Candle / training backend required — so it runs in CI by default.

use crate::config::SwarmConfig;
use crate::evals::gdop::gdop;
use crate::evals::metrics::EpisodeMetrics;
use crate::planning::patterns::{FlightPattern, PatternContext};
use crate::types::{NodeId, Position3D};

/// CSI-noise level: amplitude std `sigma` and von-Mises phase concentration `kappa`.
/// Higher `sigma` = noisier amplitude; *lower* `kappa` = noisier phase (more diffuse).
#[derive(Debug, Clone, Copy)]
pub struct NoiseLevel {
    pub sigma: f64,
    pub kappa: f64,
}

/// One evaluation configuration: a flight pattern + swarm/mission parameters.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub flight: FlightPattern,
    pub config: SwarmConfig,
    pub drones: usize,
    pub steps: usize,
    pub seeds: usize,             // ≥10 per ADR-149
    pub episodes_per_seed: usize, // e.g. 50
    pub victims: Vec<Position3D>,
    pub noise: NoiseLevel,
}

impl EvalConfig {
    /// A small SAR default suitable for fast CI runs.
    pub fn sar_small(flight: FlightPattern) -> Self {
        EvalConfig {
            flight,
            config: SwarmConfig::sar_default(),
            drones: 4,
            steps: 120,
            seeds: 10,
            episodes_per_seed: 10,
            victims: vec![
                Position3D { x: 120.0, y: 90.0, z: 0.0 },
                Position3D { x: 320.0, y: 280.0, z: 0.0 },
            ],
            noise: NoiseLevel { sigma: 0.05, kappa: 8.0 },
        }
    }
}

/// Minimal reproducible LCG → f64 in [0, 1). Self-contained for determinism.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0xD1B5_4A32_D192_ED03)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard-normal sample via Box–Muller (deterministic).
    #[inline]
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Run one kinematic episode deterministically from `seed`.
///
/// Drives drones step-by-step by the flight pattern, marks a coarse coverage
/// grid, and on the first step a drone comes within scan range of any victim
/// records a fused localization estimate (weighted centroid of contributing
/// drones' per-drone victim estimates, each perturbed by `(sigma, kappa)`
/// noise) and the GDOP of those contributing drones.
pub fn run_episode(cfg: &EvalConfig, seed: u64) -> EpisodeMetrics {
    let mut rng = Lcg::new(seed);

    let area_w = cfg.config.mission.area_width_m;
    let area_h = cfg.config.mission.area_height_m;
    let altitude_z = -cfg.config.planning.flight_altitude_m;
    let scan_width = cfg.config.planning.csi_scan_width_m.max(1.0);
    let min_sep = cfg.config.formation.min_separation_m.max(0.1);
    let n = cfg.drones.max(1);

    // Coverage grid sized so each cell ~= scan_width.
    let gx = ((area_w / scan_width).ceil() as usize).max(1);
    let gy = ((area_h / scan_width).ceil() as usize).max(1);
    let cell_w = area_w / gx as f64;
    let cell_h = area_h / gy as f64;
    let mut cover_count = vec![0u32; gx * gy];

    // Spread drones along the bottom edge with a small seeded jitter.
    let mut positions: Vec<Position3D> = (0..n)
        .map(|i| {
            let frac = (i as f64 + 0.5) / n as f64;
            Position3D {
                x: (frac * area_w + (rng.unit() - 0.5) * scan_width).clamp(0.0, area_w),
                y: (rng.unit() * scan_width).clamp(0.0, area_h),
                z: altitude_z,
            }
        })
        .collect();

    // Recent-visit ring buffer for pheromone / potential-field patterns.
    let mut visited: Vec<Position3D> = Vec::new();
    let max_visited = 32usize;

    let scan_range = scan_width; // detect a victim within one scan footprint
    let mut collisions = 0u32;
    let mut detected = false;
    let mut loc_error: Option<f64> = None;
    let mut gdop_val: Option<f64> = None;
    let mut t_detect: Option<f64> = None;

    let dt = step_seconds(cfg);

    for step in 0..cfg.steps {
        // Advance each drone one waypoint under the pattern.
        let snapshot = positions.clone();
        for (i, pos) in positions.iter_mut().enumerate() {
            let peers: Vec<Position3D> = snapshot
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| *p)
                .collect();
            let ctx = PatternContext {
                drone_id: NodeId(i as u32),
                swarm_size: n,
                current: *pos,
                area_w,
                area_h,
                altitude_z,
                scan_width_m: scan_width,
                step: step as u64,
                visited: &visited,
                peers: &peers,
            };
            *pos = cfg.flight.next_target(&ctx);
        }

        // Mark coverage + record visits.
        for pos in &positions {
            let cx = ((pos.x / cell_w).floor() as i64).clamp(0, gx as i64 - 1) as usize;
            let cy = ((pos.y / cell_h).floor() as i64).clamp(0, gy as i64 - 1) as usize;
            cover_count[cy * gx + cx] = cover_count[cy * gx + cx].saturating_add(1);
            visited.push(*pos);
        }
        if visited.len() > max_visited {
            let drop = visited.len() - max_visited;
            visited.drain(0..drop);
        }

        // Proximity / collision check (kinematic proxy).
        for a in 0..positions.len() {
            for b in (a + 1)..positions.len() {
                let d = positions[a].distance_to(&positions[b]);
                if d < min_sep {
                    collisions = collisions.saturating_add(1);
                }
            }
        }

        // Detection: first step any victim falls within scan range of ≥1 drone,
        // fuse a localization estimate from the contributing drones. A single
        // contributor still yields a (noisier) estimate; GDOP is only defined
        // for the multistatic ≥2-drone case and is `None` otherwise.
        if !detected {
            for victim in &cfg.victims {
                let contributors: Vec<Position3D> = positions
                    .iter()
                    .filter(|p| horiz_dist(p, victim) <= scan_range)
                    .copied()
                    .collect();
                if !contributors.is_empty() {
                    let (est, g) = fuse_estimate(&contributors, victim, cfg.noise, &mut rng);
                    loc_error = Some(horiz_dist(&est, victim));
                    gdop_val = g; // None for a single contributor
                    t_detect = Some((step as f64 + 1.0) * dt);
                    detected = true;
                    break;
                }
            }
        }
    }

    // Coverage + overlap.
    let total_cells = (gx * gy) as f64;
    let scanned = cover_count.iter().filter(|&&c| c > 0).count() as f64;
    let overlapped = cover_count.iter().filter(|&&c| c > 1).count() as f64;
    let coverage_pct = if total_cells > 0.0 { scanned / total_cells } else { 0.0 };
    let overlap_ratio = if scanned > 0.0 { overlapped / scanned } else { 0.0 };

    // Episodic return: reward coverage + detection, penalize overlap + collisions.
    let detect_bonus = if detected { 1.0 } else { 0.0 };
    let loc_term = match loc_error {
        Some(e) => (1.0 / (1.0 + e)).max(0.0),
        None => 0.0,
    };
    let episodic_return = 100.0 * coverage_pct + 30.0 * detect_bonus + 20.0 * loc_term
        - 10.0 * overlap_ratio
        - 5.0 * collisions as f64;

    EpisodeMetrics {
        coverage_pct,
        localization_error_m: loc_error,
        gdop_at_detection: gdop_val,
        time_to_first_detection_s: t_detect,
        detected,
        collisions,
        overlap_ratio,
        episodic_return,
    }
}

/// Per-step wall-clock seconds, derived from scan width and drone speed.
fn step_seconds(cfg: &EvalConfig) -> f64 {
    let speed = cfg.config.planning.max_speed_ms.max(0.1);
    (cfg.config.planning.csi_scan_width_m.max(1.0) / speed).max(0.1)
}

/// Horizontal (x, y) distance, ignoring altitude.
fn horiz_dist(a: &Position3D, b: &Position3D) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// Fuse contributing drones' per-drone victim estimates into a weighted
/// centroid, perturbed by `(sigma, kappa)` CSI noise, and compute the GDOP of
/// the contributing constellation.
fn fuse_estimate(
    contributors: &[Position3D],
    victim: &Position3D,
    noise: NoiseLevel,
    rng: &mut Lcg,
) -> (Position3D, Option<f64>) {
    // Phase noise std from von Mises concentration: sigma_phase ≈ 1/sqrt(kappa).
    let phase_std = 1.0 / noise.kappa.max(1e-3).sqrt();
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut wsum = 0.0;
    for c in contributors {
        let range = horiz_dist(c, victim).max(1e-6);
        // Each drone's estimate = true victim + range-scaled amplitude noise +
        // bearing error from phase noise (perpendicular to LOS).
        let amp = noise.sigma * range;
        let nx = rng.normal() * amp;
        let ny = rng.normal() * amp;
        // Bearing wobble: rotate LOS unit vector by a small phase-noise angle.
        let bearing = (victim.y - c.y).atan2(victim.x - c.x);
        let dtheta = rng.normal() * phase_std;
        let bx = range * (bearing + dtheta).cos();
        let by = range * (bearing + dtheta).sin();
        let est_x = c.x + bx + nx;
        let est_y = c.y + by + ny;
        // Inverse-range weighting: closer drones trusted more.
        let w = 1.0 / range;
        sx += est_x * w;
        sy += est_y * w;
        wsum += w;
    }
    let w = wsum.max(1e-9);
    let est = Position3D { x: sx / w, y: sy / w, z: 0.0 };
    let g = gdop(contributors, victim);
    (est, g)
}

/// Run the full seed × episode matrix → per-seed strata of [`EpisodeMetrics`].
pub fn run_matrix(cfg: &EvalConfig) -> Vec<Vec<EpisodeMetrics>> {
    (0..cfg.seeds)
        .map(|s| {
            (0..cfg.episodes_per_seed)
                .map(|e| {
                    // Distinct deterministic seed per (seed, episode) cell.
                    let cell_seed = (s as u64)
                        .wrapping_mul(0x100_0000)
                        .wrapping_add(e as u64)
                        .wrapping_add(0xABCD);
                    run_episode(cfg, cell_seed)
                })
                .collect()
        })
        .collect()
}

/// Standard ADR-149 noise sweep grid: cartesian product of σ × κ levels.
pub fn default_noise_sweep() -> Vec<NoiseLevel> {
    let sigmas = [0.02, 0.05, 0.10];
    let kappas = [16.0, 8.0, 4.0];
    let mut out = Vec::with_capacity(sigmas.len() * kappas.len());
    for &sigma in &sigmas {
        for &kappa in &kappas {
            out.push(NoiseLevel { sigma, kappa });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_episode_deterministic() {
        let cfg = EvalConfig::sar_small(FlightPattern::PartitionedLawnmower);
        let a = run_episode(&cfg, 12345);
        let b = run_episode(&cfg, 12345);
        assert_eq!(a.coverage_pct, b.coverage_pct);
        assert_eq!(a.detected, b.detected);
        assert_eq!(a.localization_error_m, b.localization_error_m);
        assert_eq!(a.collisions, b.collisions);
        assert_eq!(a.episodic_return, b.episodic_return);
    }

    #[test]
    fn test_partitioned_beats_levy_coverage() {
        let mut part = EvalConfig::sar_small(FlightPattern::PartitionedLawnmower);
        part.seeds = 3;
        part.episodes_per_seed = 5;
        let mut levy = part.clone();
        levy.flight = FlightPattern::LevyFlight;

        let part_m = run_matrix(&part);
        let levy_m = run_matrix(&levy);
        let part_agg = crate::evals::metrics::AggregateMetrics::from_strata(&part_m, 1);
        let levy_agg = crate::evals::metrics::AggregateMetrics::from_strata(&levy_m, 1);
        assert!(
            part_agg.coverage_iqm.point > levy_agg.coverage_iqm.point,
            "partitioned coverage {} should beat levy {}",
            part_agg.coverage_iqm.point,
            levy_agg.coverage_iqm.point
        );
    }

    #[test]
    fn test_matrix_shape() {
        let mut cfg = EvalConfig::sar_small(FlightPattern::Spiral);
        cfg.seeds = 4;
        cfg.episodes_per_seed = 6;
        let m = run_matrix(&cfg);
        assert_eq!(m.len(), 4);
        assert!(m.iter().all(|s| s.len() == 6));
    }

    #[test]
    fn test_noise_sweep_grid() {
        let sweep = default_noise_sweep();
        assert_eq!(sweep.len(), 9);
    }
}

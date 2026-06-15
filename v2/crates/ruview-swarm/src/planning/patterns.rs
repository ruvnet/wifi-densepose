//! Flight / coverage-optimization patterns for swarm area search.
//!
//! Different strategies trade off coverage completeness, time, and robustness:
//! - Boustrophedon: systematic lawnmower; complete but drones overlap if unpartitioned
//! - PartitionedLawnmower: area split into per-drone strips → no overlap, ~Nx faster coverage
//! - Spiral: outward spiral from a seed; good for centred search (last-known-position SAR)
//! - Pheromone: stigmergic — steer away from recently-visited cells; robust to dropout
//! - PotentialField: repelled by visited cells + peers, attracted to unscanned frontier
//! - LevyFlight: heavy-tailed random walk; good exploration when target location unknown

use crate::types::{NodeId, Position3D};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlightPattern {
    Boustrophedon,
    #[default]
    PartitionedLawnmower,
    Spiral,
    Pheromone,
    PotentialField,
    LevyFlight,
}

impl FlightPattern {
    // Intentional inherent infallible parser (returns Self, not Result); shipped API.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "boustrophedon" | "lawnmower" => FlightPattern::Boustrophedon,
            "partitioned" | "partitioned_lawnmower" => FlightPattern::PartitionedLawnmower,
            "spiral" => FlightPattern::Spiral,
            "pheromone" | "stigmergic" => FlightPattern::Pheromone,
            "potential" | "potential_field" => FlightPattern::PotentialField,
            "levy" | "levyflight" | "levy_flight" => FlightPattern::LevyFlight,
            _ => FlightPattern::default(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            FlightPattern::Boustrophedon => "boustrophedon",
            FlightPattern::PartitionedLawnmower => "partitioned_lawnmower",
            FlightPattern::Spiral => "spiral",
            FlightPattern::Pheromone => "pheromone",
            FlightPattern::PotentialField => "potential_field",
            FlightPattern::LevyFlight => "levy_flight",
        }
    }

    /// All pattern variants, for enumeration / UI selection.
    pub fn all() -> [FlightPattern; 6] {
        [
            FlightPattern::Boustrophedon,
            FlightPattern::PartitionedLawnmower,
            FlightPattern::Spiral,
            FlightPattern::Pheromone,
            FlightPattern::PotentialField,
            FlightPattern::LevyFlight,
        ]
    }
}

/// Inputs for computing the next waypoint under a pattern.
pub struct PatternContext<'a> {
    pub drone_id: NodeId,
    pub swarm_size: usize,
    pub current: Position3D,
    pub area_w: f64,
    pub area_h: f64,
    pub altitude_z: f64,      // flight z (negative NED)
    pub scan_width_m: f64,    // strip spacing
    pub step: u64,            // tick counter (for deterministic pseudo-random patterns)
    pub visited: &'a [Position3D], // recently visited cell centres (for pheromone/potential)
    pub peers: &'a [Position3D],   // peer positions (for potential-field repulsion)
}

impl FlightPattern {
    /// Compute the next target position for a drone under this pattern.
    pub fn next_target(&self, ctx: &PatternContext) -> Position3D {
        match self {
            FlightPattern::Boustrophedon => boustrophedon(ctx),
            FlightPattern::PartitionedLawnmower => partitioned_lawnmower(ctx),
            FlightPattern::Spiral => spiral(ctx),
            FlightPattern::Pheromone => pheromone(ctx),
            FlightPattern::PotentialField => potential_field(ctx),
            FlightPattern::LevyFlight => levy_flight(ctx),
        }
    }
}

/// Clamp a candidate (x, y) to the area bounds and lift it to the flight altitude.
fn clamp_to_area(x: f64, y: f64, ctx: &PatternContext) -> Position3D {
    Position3D {
        x: x.clamp(0.0, ctx.area_w),
        y: y.clamp(0.0, ctx.area_h),
        z: ctx.altitude_z,
    }
}

/// Serpentine waypoint within a rectangular sub-region.
///
/// Walks rows of height `scan_width_m`; on each row sweeps left→right or
/// right→left depending on the row parity, advancing one `scan_width_m`
/// segment per `step`.
fn serpentine_in_region(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    scan_width_m: f64,
    step: u64,
) -> (f64, f64) {
    let strip_w = (x1 - x0).max(scan_width_m);
    let height = (y1 - y0).max(scan_width_m);

    // Number of horizontal segments per row before stepping to the next row.
    let cols = ((strip_w / scan_width_m).ceil() as u64).max(1);
    // Number of rows in this region.
    let rows = ((height / scan_width_m).ceil() as u64).max(1);
    let total = cols * rows;
    let s = step % total;

    let row = s / cols;
    let col = s % cols;

    // Centre of the current row band.
    let y = y0 + (row as f64 + 0.5) * scan_width_m;
    let y = y.min(y1);

    // Serpentine: even rows L→R, odd rows R→L.
    let along = if row.is_multiple_of(2) { col } else { cols - 1 - col };
    let x = x0 + (along as f64 + 0.5) * scan_width_m;
    let x = x.min(x1);

    (x, y)
}

/// Classic full-area serpentine lawnmower (drones may overlap — baseline).
fn boustrophedon(ctx: &PatternContext) -> Position3D {
    let (x, y) = serpentine_in_region(
        0.0,
        ctx.area_w,
        0.0,
        ctx.area_h,
        ctx.scan_width_m,
        ctx.step,
    );
    clamp_to_area(x, y, ctx)
}

/// Partitioned lawnmower: split `area_w` into `swarm_size` vertical strips;
/// drone `i` lawnmowers ONLY within strip `[i*w/n, (i+1)*w/n]`.
///
/// This is the clustering fix: each drone covers a disjoint band, so total
/// coverage scales ~linearly with swarm size instead of all drones tracing
/// the same path.
fn partitioned_lawnmower(ctx: &PatternContext) -> Position3D {
    let n = ctx.swarm_size.max(1);
    let i = (ctx.drone_id.0 as usize) % n;
    let strip_w = ctx.area_w / n as f64;
    let x0 = i as f64 * strip_w;
    let x1 = x0 + strip_w;

    let (x, y) =
        serpentine_in_region(x0, x1, 0.0, ctx.area_h, ctx.scan_width_m, ctx.step);
    clamp_to_area(x, y, ctx)
}

/// Outward Archimedean spiral from the area centre; radius grows with step.
fn spiral(ctx: &PatternContext) -> Position3D {
    let cx = ctx.area_w / 2.0;
    let cy = ctx.area_h / 2.0;

    // Angular step keeps successive waypoints roughly `scan_width_m` apart.
    let theta = ctx.step as f64 * 0.6;
    // Archimedean spiral r = b * theta; b chosen so each turn adds scan_width_m.
    let b = ctx.scan_width_m / (2.0 * std::f64::consts::PI);
    let r = b * theta;

    let x = cx + r * theta.cos();
    let y = cy + r * theta.sin();
    clamp_to_area(x, y, ctx)
}

/// Stigmergic: sample candidate headings, step toward the least-visited one.
fn pheromone(ctx: &PatternContext) -> Position3D {
    let step_len = ctx.scan_width_m.max(1.0);
    // Deterministic base heading offset per drone so they diverge.
    let base = ctx.drone_id.0 as f64 * (std::f64::consts::PI / 3.0);

    let n_candidates = 8;
    let mut best: Option<(f64, f64, f64)> = None; // (score, x, y); lower score = less visited
    for k in 0..n_candidates {
        let theta = base + (k as f64) * (2.0 * std::f64::consts::PI / n_candidates as f64);
        let cx = ctx.current.x + step_len * theta.cos();
        let cy = ctx.current.y + step_len * theta.sin();
        let cx = cx.clamp(0.0, ctx.area_w);
        let cy = cy.clamp(0.0, ctx.area_h);

        // Penalty = sum of inverse-distance to recently-visited cell centres.
        let mut visit_pressure = 0.0;
        for v in ctx.visited {
            let d = (cx - v.x).hypot(cy - v.y);
            visit_pressure += 1.0 / (1.0 + d);
        }
        if best.as_ref().is_none_or(|(bs, _, _)| visit_pressure < *bs) {
            best = Some((visit_pressure, cx, cy));
        }
    }

    let (_, x, y) = best.unwrap_or((0.0, ctx.current.x, ctx.current.y));
    clamp_to_area(x, y, ctx)
}

/// Potential field: repelled by visited cells + peers, attracted to the
/// nearest unscanned frontier; step in the resultant direction.
fn potential_field(ctx: &PatternContext) -> Position3D {
    let mut fx = 0.0;
    let mut fy = 0.0;

    // Repulsion from recently-visited cells.
    for v in ctx.visited {
        let dx = ctx.current.x - v.x;
        let dy = ctx.current.y - v.y;
        let d2 = dx * dx + dy * dy + 1.0;
        let mag = 1.0 / d2;
        fx += dx / d2.sqrt() * mag;
        fy += dy / d2.sqrt() * mag;
    }

    // Repulsion from peers (collision / overlap avoidance).
    for p in ctx.peers {
        let dx = ctx.current.x - p.x;
        let dy = ctx.current.y - p.y;
        let d2 = dx * dx + dy * dy + 1.0;
        let mag = 2.0 / d2; // peers repel more strongly than stale trail
        fx += dx / d2.sqrt() * mag;
        fy += dy / d2.sqrt() * mag;
    }

    // Attraction toward the nearest unscanned frontier point. Sample a grid of
    // candidate area points; pick the one with greatest distance to any visited
    // cell (i.e. the least-explored region) and pull toward it.
    let mut frontier: Option<(f64, f64, f64)> = None; // (openness, x, y)
    let samples = 5;
    for ix in 0..=samples {
        for iy in 0..=samples {
            let px = ctx.area_w * ix as f64 / samples as f64;
            let py = ctx.area_h * iy as f64 / samples as f64;
            let mut nearest = f64::INFINITY;
            for v in ctx.visited {
                let d = (px - v.x).hypot(py - v.y);
                if d < nearest {
                    nearest = d;
                }
            }
            if !nearest.is_finite() {
                nearest = (px - ctx.current.x).hypot(py - ctx.current.y);
            }
            if frontier.as_ref().is_none_or(|(o, _, _)| nearest > *o) {
                frontier = Some((nearest, px, py));
            }
        }
    }
    if let Some((_, gx, gy)) = frontier {
        let dx = gx - ctx.current.x;
        let dy = gy - ctx.current.y;
        let d = (dx * dx + dy * dy).sqrt().max(1e-6);
        fx += dx / d * 1.5; // attraction gain
        fy += dy / d * 1.5;
    }

    let fmag = (fx * fx + fy * fy).sqrt();
    let step_len = ctx.scan_width_m.max(1.0);
    let (x, y) = if fmag > 1e-9 {
        (
            ctx.current.x + fx / fmag * step_len,
            ctx.current.y + fy / fmag * step_len,
        )
    } else {
        (ctx.current.x, ctx.current.y)
    };
    clamp_to_area(x, y, ctx)
}

/// Deterministic pseudo-random heavy-tailed step (Lévy flight). Most steps are
/// short; occasional long jumps. Seeded from drone_id + step via an LCG so the
/// trajectory is reproducible.
fn levy_flight(ctx: &PatternContext) -> Position3D {
    // Linear congruential generator (Numerical Recipes constants).
    let seed = (ctx.drone_id.0 as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(ctx.step.wrapping_mul(0x2545_F491_4F6C_DD1D));
    let r1 = lcg(seed);
    let r2 = lcg(r1);

    let u_angle = (r1 >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
    let u_len = ((r2 >> 11) as f64 / (1u64 << 53) as f64).max(1e-6); // (0,1]

    let theta = u_angle * 2.0 * std::f64::consts::PI;
    // Heavy-tailed step length: inverse power-law (Pareto-like), exponent ~1.5.
    let step_len = ctx.scan_width_m.max(1.0) * u_len.powf(-1.0 / 1.5);
    // Cap to the area diagonal so a single jump can't shoot arbitrarily far.
    let max_jump = (ctx.area_w * ctx.area_w + ctx.area_h * ctx.area_h).sqrt();
    let step_len = step_len.min(max_jump);

    let x = ctx.current.x + step_len * theta.cos();
    let y = ctx.current.y + step_len * theta.sin();
    clamp_to_area(x, y, ctx)
}

#[inline]
fn lcg(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx<'a>(
        drone_id: u32,
        swarm_size: usize,
        step: u64,
        current: Position3D,
        visited: &'a [Position3D],
        peers: &'a [Position3D],
    ) -> PatternContext<'a> {
        PatternContext {
            drone_id: NodeId(drone_id),
            swarm_size,
            current,
            area_w: 100.0,
            area_h: 80.0,
            altitude_z: -20.0,
            scan_width_m: 5.0,
            step,
            visited,
            peers,
        }
    }

    #[test]
    fn test_partitioned_strips_disjoint() {
        let empty: [Position3D; 0] = [];
        // Two drones, swarm of 2: drone 0 owns left half, drone 1 the right half.
        let mut d0_xs = Vec::new();
        let mut d1_xs = Vec::new();
        for s in 0..40u64 {
            let c0 = ctx(0, 2, s, Position3D::zero(), &empty, &empty);
            let c1 = ctx(1, 2, s, Position3D::zero(), &empty, &empty);
            d0_xs.push(FlightPattern::PartitionedLawnmower.next_target(&c0).x);
            d1_xs.push(FlightPattern::PartitionedLawnmower.next_target(&c1).x);
        }
        let mid = 100.0 / 2.0;
        // Drone 0 stays strictly in the left half, drone 1 strictly in the right.
        assert!(d0_xs.iter().all(|&x| x <= mid), "drone 0 left of midline");
        assert!(d1_xs.iter().all(|&x| x >= mid), "drone 1 right of midline");
        // And they never share an x position (disjoint strips → no overlap).
        for &a in &d0_xs {
            for &b in &d1_xs {
                assert!(a < b || (a <= mid && b >= mid), "strips overlap: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_all_patterns_in_bounds() {
        let visited = [
            Position3D { x: 10.0, y: 10.0, z: -20.0 },
            Position3D { x: 50.0, y: 40.0, z: -20.0 },
        ];
        let peers = [Position3D { x: 30.0, y: 20.0, z: -20.0 }];
        for pat in FlightPattern::all() {
            let mut current = Position3D { x: 25.0, y: 25.0, z: -20.0 };
            for s in 0..20u64 {
                let c = ctx(1, 4, s, current, &visited, &peers);
                let t = pat.next_target(&c);
                assert!(
                    t.x >= 0.0 && t.x <= 100.0,
                    "{} x out of bounds at step {s}: {}",
                    pat.name(),
                    t.x
                );
                assert!(
                    t.y >= 0.0 && t.y <= 80.0,
                    "{} y out of bounds at step {s}: {}",
                    pat.name(),
                    t.y
                );
                assert_eq!(t.z, -20.0, "{} altitude wrong", pat.name());
                current = t;
            }
        }
    }

    #[test]
    fn test_pattern_from_str_roundtrip() {
        for pat in FlightPattern::all() {
            assert_eq!(
                FlightPattern::from_str(pat.name()),
                pat,
                "roundtrip failed for {}",
                pat.name()
            );
        }
    }

    #[test]
    fn test_spiral_radius_grows() {
        let empty: [Position3D; 0] = [];
        let centre_x = 100.0 / 2.0;
        let centre_y = 80.0 / 2.0;
        let dist = |s: u64| {
            let c = ctx(0, 1, s, Position3D::zero(), &empty, &empty);
            let t = FlightPattern::Spiral.next_target(&c);
            ((t.x - centre_x).powi(2) + (t.y - centre_y).powi(2)).sqrt()
        };
        let near = dist(1);
        let far = dist(50);
        assert!(
            far > near,
            "spiral radius should grow: step1={near}, step50={far}"
        );
    }
}

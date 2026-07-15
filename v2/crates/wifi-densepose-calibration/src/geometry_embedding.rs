//! Geometry embedding — deterministic featurization of transceiver layout
//! (ADR-152 §2.1.2, the second half of the PerceptAlign fix).
//!
//! §2.1.1 ([`geometry`](crate::geometry)) *records* the layout; this module
//! turns that record into a fixed-length conditioning vector. PerceptAlign
//! fuses transceiver-position embeddings with CSI features so pose heads stop
//! memorising the deployment layout; transplanted to our per-room banks, the
//! ADR-151 P6 LoRA heads will concatenate this vector with the backbone
//! embedding. Statistical specialists (current) ignore it. The crate is pure
//! Rust and edge-deployable (no torch/candle), so the "embedding" is **not a
//! trained network** — it is a deterministic, well-conditioned featurization;
//! the learned part (if any) lives in the head that consumes it.
//!
//! Properties, by construction: **fixed dimension** ([`GeometryEmbedding::DIM`]
//! = 32) for any node count (designed for 1..=8; more nodes still aggregate,
//! only the per-node flag slots truncate); **permutation-invariant** (nodes
//! sorted by `node_id`; aggregates are order-free); and **total** — missing
//! data degrades gracefully: an all-unknown layout (or empty slice) yields a
//! well-defined vector, never `NaN`/`inf`; adversarial inputs (non-finite
//! coordinates, absurd magnitudes) are treated as unmeasured.
//!
//! ## Slot layout (v1)
//!
//! Positions/distances are raw meters (room-scale values are already
//! O(1)–O(10)); angles in radians; fractions in `[0, 1]`. Unmeasurable
//! slots are `0.0`.
//!
//! | Slot  | Content | Units / range |
//! |-------|---------|----------------|
//! | 0     | node count / 8 | `[0, 2]` (clamped; 8 nodes → 1.0) |
//! | 1     | fraction of nodes with a position | `[0, 1]` |
//! | 2     | fraction of nodes with an orientation | `[0, 1]` |
//! | 3     | fraction of nodes with ≥1 measured inter-node distance | `[0, 1]` |
//! | 4–6   | position centroid (x, y, z) | m, clamped ±[`MAX_COORD_M`] |
//! | 7–9   | position std-dev per axis (x, y, z) | m, `[0,` [`MAX_COORD_M`]`]` |
//! | 10–12 | pairwise position distance min / mean / max | m |
//! | 13–15 | inter-node distance min / mean / max — measured `distances_m`, falling back to position-derived distance per pair | m |
//! | 16    | measured-distance pair coverage (measured pairs / possible pairs) | `[0, 1]` |
//! | 17–18 | azimuth circular mean resultant vector (cos, sin components) | `[-1, 1]` |
//! | 19    | azimuth concentration (mean resultant length `R`; 1 = all boresights parallel) | `[0, 1]` |
//! | 20    | mean elevation | rad, `[-π/2, π/2]` |
//! | 21–22 | geometric diversity: eigenvalue ratios `λ2/λ1`, `λ3/λ1` of the position covariance — 0 = collinear/degenerate, →1 = isotropic spread (chosen over polygon area: defined for any node count, no 2-D planarity assumption) | `[0, 1]` |
//! | 23    | dominant spread scale `sqrt(λ1)` | m |
//! | 24–31 | per-node measurement flags, nodes sorted by `node_id`, rank `i` → slot `24+i` (first 8 nodes): `0` = no node at this rank, else `0.25` (node exists) `+0.25` (position) `+0.25` (orientation) `+0.25` (≥1 measured distance) | `{0}` ∪ `[0.25, 1]` |

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::geometry::NodeGeometry;

/// Coordinates / distances beyond this magnitude (meters) are treated as
/// unmeasured — rooms are not kilometer-scale, and the guard keeps
/// adversarial values from overflowing the covariance into `inf`.
pub const MAX_COORD_M: f32 = 1_000.0;

/// Number of per-node flag slots (slots 24..32); designed node count 1..=8.
const NODE_SLOTS: usize = 8;

fn schema_v1() -> u32 {
    GeometryEmbedding::SCHEMA_VERSION
}

/// Fixed-length featurization of a room's transceiver layout (ADR-152 §2.1.2).
///
/// Computed deterministically from the [`NodeGeometry`] snapshot via
/// [`GeometryEmbedding::from_nodes`]; the conditioning input the ADR-151 P6
/// LoRA heads concatenate with the backbone embedding. Not stored in the bank
/// — derive it via [`SpecialistBank::geometry_embedding`](crate::SpecialistBank::geometry_embedding)
/// — but schema-versioned and serde-serializable (the `NodeGeometry` compat
/// pattern) for callers that snapshot it alongside trained head weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeometryEmbedding {
    /// Slot-layout version; bump when the slot table changes meaning.
    #[serde(default = "schema_v1")]
    pub schema_version: u32,
    /// The embedding vector — see the module docs for the slot table.
    /// Invariant: every value is finite (never `NaN`/`inf`).
    pub values: [f32; GeometryEmbedding::DIM],
}

impl Default for GeometryEmbedding {
    /// All slots zero — the embedding of an empty layout.
    fn default() -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            values: [0.0; Self::DIM],
        }
    }
}

impl GeometryEmbedding {
    /// Output dimension. Fixed regardless of node count.
    pub const DIM: usize = 32;

    /// Current slot-layout version.
    pub const SCHEMA_VERSION: u32 = 1;

    /// The embedding as a slice (always [`Self::DIM`] long).
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Compute the embedding from a geometry snapshot. Permutation-invariant
    /// (nodes are sorted by `node_id` internally) and total: any input —
    /// empty, all-unknown, non-finite — produces a fully finite vector.
    pub fn from_nodes(nodes: &[NodeGeometry]) -> Self {
        let mut v = [0.0f32; Self::DIM];

        // Permutation invariance: order by node_id before per-node slots.
        let mut sorted: Vec<&NodeGeometry> = nodes.iter().collect();
        sorted.sort_by_key(|g| g.node_id);
        let n = sorted.len();
        if n == 0 {
            return Self::default();
        }

        // Sanitized views: a measurement with non-finite or absurd components
        // counts as not taken at all.
        let positions: Vec<Option<[f32; 3]>> = sorted.iter().map(|g| valid_position(g)).collect();
        let orientations: Vec<Option<(f32, f32)>> =
            sorted.iter().map(|g| valid_orientation(g)).collect();
        let measured = measured_pairs(&sorted);
        let node_has_dist = |id: u8| measured.keys().any(|&(a, b)| a == id || b == id);
        let has_dist: Vec<bool> = sorted.iter().map(|g| node_has_dist(g.node_id)).collect();

        // Slots 0–3: node count + measurement-presence fractions.
        let nf = n as f32;
        v[0] = (nf / NODE_SLOTS as f32).min(2.0);
        v[1] = positions.iter().flatten().count() as f32 / nf;
        v[2] = orientations.iter().flatten().count() as f32 / nf;
        v[3] = has_dist.iter().filter(|&&d| d).count() as f32 / nf;

        // Slots 4–9: centroid + per-axis std of the known positions.
        let known: Vec<[f32; 3]> = positions.iter().flatten().copied().collect();
        if !known.is_empty() {
            let kf = known.len() as f32;
            let mut centroid = [0.0f32; 3];
            for p in &known {
                for (c, x) in centroid.iter_mut().zip(p) {
                    *c += x / kf;
                }
            }
            for axis in 0..3 {
                v[4 + axis] = clamp_m(centroid[axis]);
                let mut var = 0.0;
                for p in &known {
                    var += (p[axis] - centroid[axis]).powi(2) / kf;
                }
                v[7 + axis] = clamp_m(var.max(0.0).sqrt());
            }

            // Slots 10–12: pairwise position distance stats.
            let mut dists = Vec::new();
            for i in 0..known.len() {
                for j in (i + 1)..known.len() {
                    dists.push(euclidean(&known[i], &known[j]));
                }
            }
            write_min_mean_max(&mut v, 10, &dists);

            // Slots 21–23: geometric diversity from the position covariance
            // eigenstructure (see module docs for why over polygon area).
            let (l1, l2, l3) = covariance_eigenvalues(&known, &centroid);
            if l1 > f32::EPSILON {
                v[21] = (l2 / l1).clamp(0.0, 1.0);
                v[22] = (l3 / l1).clamp(0.0, 1.0);
            }
            v[23] = clamp_m(l1.max(0.0).sqrt());
        }

        // Slots 13–16: inter-node distances — measured first, position fallback.
        let mut inter = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let key = pair_key(sorted[i].node_id, sorted[j].node_id);
                if let Some(&d) = measured.get(&key) {
                    inter.push(d);
                } else if let (Some(a), Some(b)) = (&positions[i], &positions[j]) {
                    inter.push(euclidean(a, b));
                }
            }
        }
        write_min_mean_max(&mut v, 13, &inter);
        let possible_pairs = n * n.saturating_sub(1) / 2;
        if possible_pairs > 0 {
            v[16] = (measured.len() as f32 / possible_pairs as f32).clamp(0.0, 1.0);
        }

        // Slots 17–20: orientation statistics (circular mean of azimuth).
        let known_orient: Vec<(f32, f32)> = orientations.iter().flatten().copied().collect();
        if !known_orient.is_empty() {
            let of = known_orient.len() as f32;
            let c = known_orient.iter().map(|(az, _)| az.cos()).sum::<f32>() / of;
            let s = known_orient.iter().map(|(az, _)| az.sin()).sum::<f32>() / of;
            v[17] = c.clamp(-1.0, 1.0);
            v[18] = s.clamp(-1.0, 1.0);
            v[19] = (c * c + s * s).sqrt().clamp(0.0, 1.0);
            let el = known_orient.iter().map(|(_, e)| e).sum::<f32>() / of;
            v[20] = el.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        }

        // Slots 24–31: per-node measurement flags (first NODE_SLOTS by id).
        for i in 0..n.min(NODE_SLOTS) {
            v[24 + i] = 0.25
                + 0.25 * f32::from(positions[i].is_some() as u8)
                + 0.25 * f32::from(orientations[i].is_some() as u8)
                + 0.25 * f32::from(has_dist[i] as u8);
        }

        // The finite invariant must hold whatever happened above.
        for x in &mut v {
            if !x.is_finite() {
                *x = 0.0;
            }
        }

        Self {
            schema_version: Self::SCHEMA_VERSION,
            values: v,
        }
    }
}

/// A position whose components are all finite and room-scale, else `None`.
fn valid_position(g: &NodeGeometry) -> Option<[f32; 3]> {
    let p = g.position?;
    let ok = |c: f32| c.is_finite() && c.abs() <= MAX_COORD_M;
    (ok(p.x_m) && ok(p.y_m) && ok(p.z_m)).then_some([p.x_m, p.y_m, p.z_m])
}

/// An orientation whose angles are both finite, else `None`.
fn valid_orientation(g: &NodeGeometry) -> Option<(f32, f32)> {
    let o = g.orientation?;
    let ok = o.azimuth_rad.is_finite() && o.elevation_rad.is_finite();
    ok.then_some((o.azimuth_rad, o.elevation_rad))
}

/// Canonical unordered pair key.
fn pair_key(a: u8, b: u8) -> (u8, u8) {
    (a.min(b), a.max(b))
}

/// Valid measured distances between *enrolled* nodes, deduplicated to
/// unordered pairs (both directions recorded → averaged); distances to
/// non-enrolled node ids are ignored.
fn measured_pairs(sorted: &[&NodeGeometry]) -> BTreeMap<(u8, u8), f32> {
    let ids: Vec<u8> = sorted.iter().map(|g| g.node_id).collect();
    let mut sums: BTreeMap<(u8, u8), (f32, u32)> = BTreeMap::new();
    for g in sorted {
        for (&other, &d) in &g.distances_m {
            let pair_ok = other != g.node_id && ids.contains(&other);
            if pair_ok && d.is_finite() && d > 0.0 && d <= MAX_COORD_M {
                let e = sums.entry(pair_key(g.node_id, other)).or_insert((0.0, 0));
                e.0 += d;
                e.1 += 1;
            }
        }
    }
    sums.into_iter()
        .map(|(k, (sum, n))| (k, sum / n as f32))
        .collect()
}

fn euclidean(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let mut d2 = 0.0;
    for k in 0..3 {
        d2 += (a[k] - b[k]).powi(2);
    }
    d2.sqrt()
}

/// Write min/mean/max of a sample into slots `base..base+3` (left at zero
/// when the sample is empty), clamped to the meters range.
fn write_min_mean_max(v: &mut [f32; GeometryEmbedding::DIM], base: usize, xs: &[f32]) {
    if xs.is_empty() {
        return;
    }
    let (mut min, mut max, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0);
    for &x in xs {
        min = min.min(x);
        max = max.max(x);
        sum += x;
    }
    v[base] = clamp_m(min);
    v[base + 1] = clamp_m(sum / xs.len() as f32);
    v[base + 2] = clamp_m(max);
}

/// Clamp a meters-valued slot into ±[`MAX_COORD_M`], mapping non-finite to 0.
fn clamp_m(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(-MAX_COORD_M, MAX_COORD_M)
    } else {
        0.0
    }
}

/// Eigenvalues `λ1 ≥ λ2 ≥ λ3 ≥ 0` of the 3×3 position covariance, via the
/// closed-form trigonometric solution for symmetric matrices (no linear-
/// algebra dependency; f64 internally for conditioning).
fn covariance_eigenvalues(points: &[[f32; 3]], centroid: &[f32; 3]) -> (f32, f32, f32) {
    let nf = points.len() as f64;
    // Upper triangle of the symmetric covariance: (xx, yy, zz, xy, xz, yz).
    const IJ: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let mut m = [0.0f64; 6];
    for p in points {
        let d: [f64; 3] = std::array::from_fn(|i| (p[i] - centroid[i]) as f64);
        for (k, &(i, j)) in IJ.iter().enumerate() {
            m[k] += d[i] * d[j] / nf;
        }
    }
    let (a, b, c, d, e, f) = (m[0], m[1], m[2], m[3], m[4], m[5]);
    let p1 = d * d + e * e + f * f;
    let q = (a + b + c) / 3.0;
    let p2 = (a - q).powi(2) + (b - q).powi(2) + (c - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let (l1, l2, l3) = if p < 1e-12 {
        (q, q, q) // (Near-)isotropic: all eigenvalues equal — diagonal incl.
    } else {
        // r = det((M - qI)/p) / 2, clamped into acos' domain.
        let (ba, bb, bc) = ((a - q) / p, (b - q) / p, (c - q) / p);
        let (bd, be, bf) = (d / p, e / p, f / p);
        let det = ba * (bb * bc - bf * bf) - bd * (bd * bc - bf * be) + be * (bd * bf - bb * be);
        let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
        let e1 = q + 2.0 * p * phi.cos();
        let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
        (e1, 3.0 * q - e1 - e3, e3)
    };
    // PSD matrix: tiny negatives are numerical noise — clamp.
    (l1.max(0.0) as f32, l2.max(0.0) as f32, l3.max(0.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fully-measured node at `(x, y, 1)` with boresight toward +Y.
    fn node(id: u8, x: f32, y: f32) -> NodeGeometry {
        NodeGeometry::new(id, "tape-measure")
            .with_position(x, y, 1.0)
            .with_orientation(std::f32::consts::FRAC_PI_2, 0.1)
    }

    /// 3 nodes on a 3-4-5 triangle; the (1,2) edge also measured by tape.
    fn full_layout() -> Vec<NodeGeometry> {
        vec![
            node(1, 0.0, 0.0).with_distance(2, 3.0),
            node(2, 3.0, 0.0).with_distance(1, 3.0),
            node(3, 0.0, 4.0),
        ]
    }

    fn assert_all_finite(e: &GeometryEmbedding) {
        for (i, x) in e.values.iter().enumerate() {
            assert!(x.is_finite(), "slot {i} is not finite: {x}");
        }
    }

    #[test]
    fn dimension_stable_and_empty_input_is_all_zero() {
        assert_eq!(GeometryEmbedding::DIM, 32);
        let full = GeometryEmbedding::from_nodes(&full_layout());
        assert_eq!(full.as_slice().len(), GeometryEmbedding::DIM);
        let empty = GeometryEmbedding::from_nodes(&[]);
        assert_eq!(empty, GeometryEmbedding::default(), "all-zero");
    }

    #[test]
    fn all_unknown_layout_degrades_gracefully() {
        let nodes = vec![NodeGeometry::unknown(1), NodeGeometry::unknown(2)];
        let e = GeometryEmbedding::from_nodes(&nodes);
        assert_all_finite(&e);
        assert!((e.values[0] - 2.0 / 8.0).abs() < 1e-6, "node count slot");
        // No measurements: presence fractions and all stats at zero …
        for slot in 1..24 {
            assert_eq!(e.values[slot], 0.0, "slot {slot} should be 0");
        }
        // … but the per-node existence flags still say two nodes were there.
        assert_eq!(&e.values[24..27], &[0.25, 0.25, 0.0]);
    }

    #[test]
    fn single_node_has_no_pairwise_stats() {
        let n = NodeGeometry::new(5, "t")
            .with_position(1.0, 2.0, 1.5)
            .with_orientation(0.0, 0.0);
        let e = GeometryEmbedding::from_nodes(&[n]);
        assert_all_finite(&e);
        assert_eq!(&e.values[4..7], &[1.0, 2.0, 1.5], "centroid = the node");
        assert_eq!(&e.values[7..10], &[0.0, 0.0, 0.0], "no spread");
        assert_eq!(&e.values[10..17], &[0.0; 7], "no pairs");
        assert_eq!(e.values[17], 1.0, "cos(0)");
        assert_eq!(e.values[19], 1.0, "single boresight is fully concentrated");
        assert_eq!(e.values[24], 0.75, "position + orientation, no distances");
    }

    /// Full-measurement layout: every slot family lands where the geometry
    /// says it should, and shuffling node order changes nothing.
    #[test]
    fn full_layout_statistics_and_permutation_invariance() {
        let nodes = full_layout();
        let e = GeometryEmbedding::from_nodes(&nodes);
        assert!((e.values[1] - 1.0).abs() < 1e-6, "all positioned");
        assert!((e.values[2] - 1.0).abs() < 1e-6, "all oriented");
        // 3-4-5 triangle: position-pair distances {3, 4, 5}.
        assert!((e.values[10] - 3.0).abs() < 1e-5, "min dist");
        assert!((e.values[11] - 4.0).abs() < 1e-5, "mean dist");
        assert!((e.values[12] - 5.0).abs() < 1e-5, "max dist");
        // Inter-node stats: pair (1,2) measured, (1,3)/(2,3) from positions.
        assert!((e.values[14] - 4.0).abs() < 1e-5, "mean inter-node dist");
        assert!((e.values[16] - 1.0 / 3.0).abs() < 1e-6, "1 of 3 measured");
        // Parallel boresights: fully concentrated, pointing +Y.
        assert!(e.values[17].abs() < 1e-6, "cos(π/2)");
        assert!((e.values[18] - 1.0).abs() < 1e-5, "sin(π/2)");
        assert!((e.values[19] - 1.0).abs() < 1e-5, "concentration");
        assert!((e.values[20] - 0.1).abs() < 1e-5, "mean elevation");
        // Coplanar triangle: λ1 ≈ 4.32, λ2 ≈ 1.23 (3-4-5 covariance), λ3 = 0.
        assert!((e.values[21] - 0.286).abs() < 0.01, "λ2/λ1 planar");
        assert!(e.values[22] < 1e-5, "λ3/λ1 ≈ 0 — coplanar nodes");
        assert!(e.values[23] > 0.5, "dominant spread is meter-scale");
        // Node 3 (rank 2) recorded no distances; nodes 1, 2 did.
        assert_eq!(&e.values[24..27], &[1.0, 1.0, 0.75]);

        let mut shuffled = nodes;
        shuffled.rotate_left(1);
        shuffled.swap(0, 1);
        assert_eq!(e, GeometryEmbedding::from_nodes(&shuffled));
    }

    #[test]
    fn measured_distance_overrides_position_distance() {
        // Positions say 3 m apart, the tape measure said 2.5 m: measured wins.
        let nodes = vec![
            NodeGeometry::new(1, "t")
                .with_position(0.0, 0.0, 1.0)
                .with_distance(2, 2.5),
            NodeGeometry::new(2, "t").with_position(3.0, 0.0, 1.0),
        ];
        let e = GeometryEmbedding::from_nodes(&nodes);
        assert!((e.values[10] - 3.0).abs() < 1e-5, "position pair stat raw");
        assert!((e.values[14] - 2.5).abs() < 1e-5, "measured wins");
        assert!((e.values[16] - 1.0).abs() < 1e-6, "full pair coverage");
    }

    #[test]
    fn adversarial_inputs_never_produce_nan() {
        let nodes = vec![
            NodeGeometry::new(1, "garbage")
                .with_position(f32::NAN, f32::INFINITY, -0.0)
                .with_orientation(f32::NAN, f32::NEG_INFINITY)
                .with_distance(2, f32::NAN)
                .with_distance(3, -5.0)
                .with_distance(1, 1.0), // self-distance: ignored
            NodeGeometry::new(2, "garbage")
                .with_position(1e30, 1e30, 1e30)
                .with_distance(99, 4.0), // unknown node: ignored
            NodeGeometry::new(3, "garbage").with_position(2.0, 0.0, 1.0),
        ];
        let e = GeometryEmbedding::from_nodes(&nodes);
        assert_all_finite(&e);
        // Only node 3's position survived sanitization.
        assert!((e.values[1] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(e.values[2], 0.0, "no valid orientations");
        assert_eq!(e.values[16], 0.0, "no valid measured pairs");
        assert!(e.values.iter().all(|x| x.abs() <= MAX_COORD_M), "bounded");
    }

    #[test]
    fn more_than_eight_nodes_still_aggregates() {
        let nodes: Vec<NodeGeometry> = (0..12)
            .map(|i| NodeGeometry::new(i, "plan").with_position(i as f32, 0.0, 1.0))
            .collect();
        let e = GeometryEmbedding::from_nodes(&nodes);
        assert!((e.values[0] - 12.0 / 8.0).abs() < 1e-6);
        // All 8 flag slots filled (positions known, ranks 0..8 by node_id).
        assert!(e.values[24..32].iter().all(|&f| f == 0.5));
        // Collinear nodes: zero planar/volume diversity, meter-scale spread.
        assert!(e.values[21] < 1e-5);
        assert!(e.values[22] < 1e-5);
        assert!(e.values[23] > 1.0);
    }

    #[test]
    fn serde_roundtrip_and_schema_default() {
        let e = GeometryEmbedding::from_nodes(&full_layout());
        let json = serde_json::to_string(&e).unwrap();
        let back: GeometryEmbedding = serde_json::from_str(&json).unwrap();
        assert_eq!(back, e);
        assert_eq!(back.schema_version, GeometryEmbedding::SCHEMA_VERSION);
        // JSON written by a pre-versioning producer (no version field)
        // defaults to the current schema — the NodeGeometry pattern.
        let vals = serde_json::to_string(&e.values).unwrap();
        let bare = format!("{{\"values\":{vals}}}");
        let from_bare: GeometryEmbedding = serde_json::from_str(&bare).unwrap();
        assert_eq!(from_bare.schema_version, 1);
        assert_eq!(from_bare.values, e.values);
    }
}

//! Antenna positions, synthetic-aperture trajectories, and point geometry.

use serde::{Deserialize, Serialize};

/// A point in 3D space, meters, in an arbitrary right-handed scene frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    /// X coordinate, meters.
    pub x: f64,
    /// Y coordinate, meters.
    pub y: f64,
    /// Z coordinate, meters.
    pub z: f64,
}

impl Point3 {
    /// Construct a point.
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another point, meters.
    pub fn distance(&self, other: &Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Unit vector pointing from `self` toward `other`. Returns `None` if
    /// the two points coincide (distance below `f64::EPSILON`).
    pub fn direction_to(&self, other: &Point3) -> Option<Point3> {
        let d = self.distance(other);
        if d < f64::EPSILON {
            return None;
        }
        Some(Point3::new(
            (other.x - self.x) / d,
            (other.y - self.y) / d,
            (other.z - self.z) / d,
        ))
    }

    /// Translate this point by `dist` meters along a unit vector `dir`.
    pub fn translated(&self, dir: Point3, dist: f64) -> Point3 {
        Point3::new(
            self.x + dir.x * dist,
            self.y + dir.y * dist,
            self.z + dir.z * dist,
        )
    }
}

/// A single antenna position along a synthetic-aperture trajectory.
///
/// Only position is modeled (an isotropic-antenna approximation, ADR-283
/// §4) -- no antenna gain pattern / boresight direction is applied to the
/// forward measurement model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AntennaPose {
    /// Antenna phase-center position, meters.
    pub position: Point3,
}

impl AntennaPose {
    /// Construct a pose at `position`.
    pub fn new(position: Point3) -> Self {
        Self { position }
    }
}

/// Generate `n` antenna poses evenly spaced along a straight line segment
/// from `start` to `end` (inclusive), the canonical "handheld linear sweep"
/// synthetic aperture. `n` must be >= 2 for a non-degenerate aperture.
pub fn linear_aperture(start: Point3, end: Point3, n: usize) -> Vec<AntennaPose> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![AntennaPose::new(start)];
    }
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            AntennaPose::new(Point3::new(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
                start.z + (end.z - start.z) * t,
            ))
        })
        .collect()
}

/// The physical length of a synthetic aperture: the distance between its
/// first and last pose. Used by [`crate::resolution::cross_range_resolution`].
pub fn aperture_length(poses: &[AntennaPose]) -> f64 {
    match (poses.first(), poses.last()) {
        (Some(a), Some(b)) => a.position.distance(&b.position),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_aperture_spans_endpoints() {
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(1.0, 0.0, 0.0);
        let poses = linear_aperture(start, end, 5);
        assert_eq!(poses.len(), 5);
        assert_eq!(poses[0].position, start);
        assert_eq!(poses[4].position, end);
        // Evenly spaced: 0, 0.25, 0.5, 0.75, 1.0 along x.
        assert!((poses[2].position.x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn aperture_length_matches_endpoint_distance() {
        let poses = linear_aperture(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 4.0, 0.0), 10);
        assert!((aperture_length(&poses) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn direction_to_is_unit_length() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(2.0, 0.0, 0.0);
        let dir = a.direction_to(&b).unwrap();
        assert!((dir.x - 1.0).abs() < 1e-12);
        let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
        assert!((len - 1.0).abs() < 1e-12);
    }

    #[test]
    fn direction_to_coincident_points_is_none() {
        let a = Point3::new(1.0, 1.0, 1.0);
        assert!(a.direction_to(&a).is_none());
    }
}

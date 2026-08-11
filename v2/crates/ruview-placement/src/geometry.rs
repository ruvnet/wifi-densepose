//! Coarse 2D floor-plan geometry the optimizer plans over (ADR-305 §1).
//!
//! **SYNTHETIC / L0.** This is a deliberately coarse stand-in for the ADR-303
//! scene: axis-aligned rectangular [`Rect`] bounds for a [`Space`](ruview_ontology::Space)
//! and its [`Zone`](ruview_ontology::Zone)s, plus attenuating [`Wall`] segments
//! (reused from the twin). It is a *model* of a room, never a surveyed floor
//! plan, and it makes no measurement or accuracy claim. Geometry references the
//! canonical ontology vocabulary ([`SpaceId`], [`ZoneId`], [`Container`]); it
//! does not invent a second identity scheme (ADR-297 rule 3).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use ruview_ontology::{Container, SpaceId, ZoneId};
use ruview_twin::Wall;

/// Upper bound on wall/reflector segments accepted in one floor plan. Bounds
/// allocation on untrusted input.
pub const MAX_PLAN_WALLS: usize = 4096;

/// Upper bound on zones accepted in one floor plan.
pub const MAX_ZONES: usize = 1024;

/// An axis-aligned rectangle in the deployment's local metric frame (metres).
/// A coarse abstraction of a room/zone footprint, not a surveyed boundary.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    /// Minimum x (east), metres.
    pub min_x: f64,
    /// Minimum y (north), metres.
    pub min_y: f64,
    /// Maximum x (east), metres.
    pub max_x: f64,
    /// Maximum y (north), metres.
    pub max_y: f64,
}

impl Rect {
    /// Construct a rectangle. Validity is checked separately with [`Self::is_valid`].
    #[must_use]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    /// True when every coordinate is finite and the rectangle is non-degenerate
    /// (`min < max` on both axes). Rejects `NaN`/`inf`/inverted rectangles at the
    /// boundary.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min_x.is_finite()
            && self.min_y.is_finite()
            && self.max_x.is_finite()
            && self.max_y.is_finite()
            && self.max_x > self.min_x
            && self.max_y > self.min_y
    }

    /// Width (x extent), metres.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height (y extent), metres.
    #[must_use]
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Centre point `(x, y)`.
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
    }
}

/// A zone footprint within a floor plan's space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoneGeometry {
    /// Ontology zone id (ADR-303).
    pub id: ZoneId,
    /// The zone's rectangular footprint.
    pub bounds: Rect,
}

/// A coarse floor plan: one space footprint, its zones, and attenuating walls.
///
/// **SYNTHETIC / L0.** A model of the physical scene the optimizer plans over;
/// not a measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FloorPlan {
    /// Ontology space this plan describes (geometry reference, not a copy).
    pub space: SpaceId,
    /// The space's rectangular footprint.
    pub bounds: Rect,
    /// Attenuating wall/reflector segments (reused twin geometry).
    pub walls: Vec<Wall>,
    /// Zone footprints within the space.
    pub zones: Vec<ZoneGeometry>,
}

impl FloorPlan {
    /// Validate the plan at the boundary. Never panics; returns a typed error for
    /// non-finite/inverted rectangles, non-finite walls, or over-limit counts.
    pub fn validate(&self) -> Result<(), PlacementError> {
        if !self.bounds.is_valid() {
            return Err(PlacementError::InvalidRect { what: "space bounds" });
        }
        if self.walls.len() > MAX_PLAN_WALLS {
            return Err(PlacementError::TooManyWalls {
                len: self.walls.len(),
                max: MAX_PLAN_WALLS,
            });
        }
        if self.zones.len() > MAX_ZONES {
            return Err(PlacementError::TooManyZones {
                len: self.zones.len(),
                max: MAX_ZONES,
            });
        }
        for wall in &self.walls {
            if !wall.is_finite() {
                return Err(PlacementError::NonFinite { what: "wall coordinate" });
            }
        }
        for zone in &self.zones {
            if !zone.bounds.is_valid() {
                return Err(PlacementError::InvalidRect { what: "zone bounds" });
            }
        }
        Ok(())
    }

    /// Resolve the rectangular region a [`Container`] targets, if present in this
    /// plan. A first-class `None` (never an error) when the target is not in the
    /// plan (ADR-297 rule 1 — the caller surfaces it as UNKNOWN).
    #[must_use]
    pub fn region_for(&self, target: &Container) -> Option<Rect> {
        match target {
            Container::Space { id } if id == &self.space => Some(self.bounds),
            Container::Space { .. } => None,
            Container::Zone { id } => self
                .zones
                .iter()
                .find(|z| &z.id == id)
                .map(|z| z.bounds),
        }
    }
}

/// Deterministic grid of sample points inside `rect`, at `step` metres, capped at
/// `max_points`. Points are cell centres; a rectangle smaller than one step still
/// yields its centre. No randomness; identical inputs give identical points.
#[must_use]
pub fn sample_points(rect: &Rect, step: f64, max_points: usize) -> Vec<(f64, f64)> {
    if !rect.is_valid() || !(step.is_finite() && step > 0.0) || max_points == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut y = rect.min_y + step / 2.0;
    while y < rect.max_y {
        let mut x = rect.min_x + step / 2.0;
        while x < rect.max_x {
            if out.len() >= max_points {
                return out;
            }
            out.push((x, y));
            x += step;
        }
        y += step;
    }
    if out.is_empty() {
        // Rectangle narrower than one step on an axis: fall back to its centre.
        out.push(rect.center());
    }
    out
}

/// Boundary errors from validating placement inputs. Malformed input yields one of
/// these; it never panics.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum PlacementError {
    /// A coordinate or parameter was non-finite (`NaN`/`inf`).
    #[error("non-finite value: {what}")]
    NonFinite {
        /// What was non-finite.
        what: &'static str,
    },
    /// A rectangle was degenerate or inverted (`min >= max`).
    #[error("invalid rectangle: {what}")]
    InvalidRect {
        /// Which rectangle.
        what: &'static str,
    },
    /// More walls than [`MAX_PLAN_WALLS`].
    #[error("too many walls: {len} exceeds maximum {max}")]
    TooManyWalls {
        /// Actual count.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
    /// More zones than [`MAX_ZONES`].
    #[error("too many zones: {len} exceeds maximum {max}")]
    TooManyZones {
        /// Actual count.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
    /// More radios than the inventory limit.
    #[error("too many radios: {len} exceeds maximum {max}")]
    TooManyRadios {
        /// Actual count.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
    /// A model parameter was out of its valid domain.
    #[error("invalid parameter: {what}")]
    InvalidParameter {
        /// Human-readable reason.
        what: &'static str,
    },
}

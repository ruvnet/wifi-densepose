//! RF-input mapping for the ADR-324 demo tiers.
//!
//! This module turns RuView's *existing* RF surfaces into a bounded,
//! honestly-labeled eye position for the off-axis camera:
//!
//! - [`field_peak`] extracts the strongest cell of a `/ws/sensing`
//!   `signal_field` grid using the **same** grid→world mapping as the
//!   sensing server's `field_localize.rs` (constants mirrored below, with
//!   the same honesty caveat: a single-link field peak is a representation
//!   of where field energy concentrates, **not** calibrated metric
//!   localization, and never a head position).
//! - [`CoarseParallax`] converts field-peak motion into a **Tier B** eye
//!   offset: deadbanded, gain-limited, hard-clamped, one-euro filtered.
//!   Its output is coarse body parallax by construction — the clamps make
//!   over-claiming impossible at the API level.
//! - [`ScreenCalibration`] holds the physical screen measurements and maps
//!   a normalized head position (Tier A, from any fine tracker the host
//!   provides) into metric eye coordinates in screen space.
//!
//! Evidence discipline: nothing in this module asserts an accuracy number.
//! All numeric defaults are interaction-design choices (`CLAIMED`), not
//! measured performance.

use crate::filter::{OneEuro, OneEuroConfig};
use crate::math::Vec3;

/// Grid-cell → world X scale (metres per cell), mirroring
/// `wifi-densepose-sensing-server/src/field_localize.rs::X_SCALE`.
pub const FIELD_X_SCALE: f64 = 0.6;
/// Grid-cell → world Z scale (metres per cell), mirroring
/// `field_localize.rs::Z_SCALE`.
pub const FIELD_Z_SCALE: f64 = 0.5;
/// Minimum normalized field value for a cell to count as a real peak,
/// mirroring `field_localize.rs::PEAK_THRESHOLD`.
pub const FIELD_PEAK_THRESHOLD: f64 = 0.35;

/// The strongest cell of a signal-field grid, in the demo's world mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldPeak {
    /// World X (metres): `(ix - nx/2) * FIELD_X_SCALE`.
    pub x: f64,
    /// World Z (metres): `(iz - nz/2) * FIELD_Z_SCALE`.
    pub z: f64,
    /// The peak's normalized field value in `[0, 1]`.
    pub value: f64,
    /// Grid column of the peak.
    pub ix: usize,
    /// Grid row of the peak.
    pub iz: usize,
}

/// Find the strongest field cell at or above [`FIELD_PEAK_THRESHOLD`].
///
/// `values` is row-major with the sensing server's layout
/// (`idx = iz * nx + ix`). Returns `None` when the slice length doesn't
/// match `nx * nz`, when the grid is empty, or when no cell reaches the
/// threshold (an honest "no localizable hotspot" outcome, mirroring the
/// server's gating).
pub fn field_peak(values: &[f32], nx: usize, nz: usize) -> Option<FieldPeak> {
    field_peak_with_threshold(values, nx, nz, FIELD_PEAK_THRESHOLD)
}

/// [`field_peak`] with a caller-supplied threshold (used by tests and by
/// demos that want to visualize sub-threshold energy without moving the
/// camera).
pub fn field_peak_with_threshold(
    values: &[f32],
    nx: usize,
    nz: usize,
    threshold: f64,
) -> Option<FieldPeak> {
    if nx == 0 || nz == 0 || values.len() != nx * nz {
        return None;
    }
    // Branch-light argmax: a NaN never satisfies `v > best_v`, so non-finite
    // cells are skipped without an explicit is_finite() in the hot loop
    // (+inf is excluded by the finite check on the winner below).
    let mut best_idx = usize::MAX;
    let mut best_v = f32::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_idx = idx;
        }
    }
    if best_idx == usize::MAX || !best_v.is_finite() {
        return None;
    }
    let (idx, v) = (best_idx, best_v);
    // Compare in f32: grid values arrive as f32 (Float32Array), and casting
    // 0.35_f32 up to f64 lands a hair below a 0.35_f64 threshold.
    if v < threshold as f32 {
        return None;
    }
    let ix = idx % nx;
    let iz = idx / nx;
    Some(FieldPeak {
        x: (ix as f64 - nx as f64 / 2.0) * FIELD_X_SCALE,
        z: (iz as f64 - nz as f64 / 2.0) * FIELD_Z_SCALE,
        value: v as f64,
        ix,
        iz,
    })
}

/// Tier B parameters. Every default is an interaction-design choice
/// (`CLAIMED`), deliberately conservative so the mode reads as what it is:
/// slow body-scale parallax, not head tracking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoarseParallaxConfig {
    /// Eye metres produced per body metre of peak movement (≤ 1 keeps the
    /// effect visibly sub-physical).
    pub gain: f64,
    /// Peak movement below this (metres, from the session origin) is
    /// ignored entirely — RF field peaks jitter, and the deadband keeps a
    /// still room visually still.
    pub deadband_m: f64,
    /// Hard clamp on the |x| eye offset (metres). The API cannot emit a
    /// larger excursion regardless of input.
    pub max_offset_m: f64,
    /// Nominal viewing distance (metres) used as the eye's Z when no depth
    /// modulation applies.
    pub base_distance_m: f64,
    /// One-euro parameters for the offset filter. Tier B wants heavy
    /// smoothing: default `min_cutoff` is well below the Tier A default.
    pub filter: OneEuroConfig,
}

impl Default for CoarseParallaxConfig {
    fn default() -> Self {
        Self {
            gain: 0.5,
            deadband_m: 0.15,
            max_offset_m: 0.35,
            base_distance_m: 0.65,
            filter: OneEuroConfig {
                min_cutoff: 0.4,
                beta: 0.2,
                d_cutoff: 1.0,
            },
        }
    }
}

/// Tier B: field-peak motion → bounded, smoothed eye position.
///
/// The first accepted peak establishes a session origin; subsequent peaks
/// move the eye relative to it. Peaks below threshold (i.e. `None` from
/// [`field_peak`]) hold the last eye position — the camera never snaps.
#[derive(Clone, Copy, Debug)]
pub struct CoarseParallax {
    cfg: CoarseParallaxConfig,
    fx: OneEuro,
    fz: OneEuro,
    origin: Option<(f64, f64)>,
    eye: Vec3,
}

impl CoarseParallax {
    /// Create with the given configuration.
    pub fn new(cfg: CoarseParallaxConfig) -> Self {
        Self {
            cfg,
            fx: OneEuro::new(cfg.filter),
            fz: OneEuro::new(cfg.filter),
            origin: None,
            eye: Vec3::new(0.0, 0.0, cfg.base_distance_m),
        }
    }

    /// The current (last computed) eye position in screen space (metres).
    pub fn eye(&self) -> Vec3 {
        self.eye
    }

    /// Forget the session origin and filter state; the eye returns to the
    /// centered rest position.
    pub fn reset(&mut self) {
        self.origin = None;
        self.fx.reset();
        self.fz.reset();
        self.eye = Vec3::new(0.0, 0.0, self.cfg.base_distance_m);
    }

    /// Ingest a field peak observed at `t_s` seconds; returns the updated
    /// eye position. Call with the output of [`field_peak`]; pass `None`
    /// (below-threshold field) to hold the current position.
    pub fn update(&mut self, peak: Option<FieldPeak>, t_s: f64) -> Vec3 {
        let Some(p) = peak else { return self.eye };
        let (ox, oz) = *self.origin.get_or_insert((p.x, p.z));
        let dx = deadband(p.x - ox, self.cfg.deadband_m);
        let dz = deadband(p.z - oz, self.cfg.deadband_m);
        let raw_x = (dx * self.cfg.gain).clamp(-self.cfg.max_offset_m, self.cfg.max_offset_m);
        // Peak Z (room depth) modulates viewing distance, same bound.
        let raw_z = (dz * self.cfg.gain).clamp(-self.cfg.max_offset_m, self.cfg.max_offset_m);
        let x = self.fx.filter(raw_x, t_s);
        let z_off = self.fz.filter(raw_z, t_s);
        // Keep the eye strictly in front of the screen: distance floor at
        // half the nominal distance.
        let z = (self.cfg.base_distance_m + z_off).max(self.cfg.base_distance_m * 0.5);
        self.eye = Vec3::new(x, 0.0, z);
        self.eye
    }
}

/// Zero inside `±band`, shifted toward zero outside it (continuous at the
/// band edge, so motion doesn't jump when leaving the deadband).
#[inline]
fn deadband(v: f64, band: f64) -> f64 {
    if v > band {
        v - band
    } else if v < -band {
        v + band
    } else {
        0.0
    }
}

/// Physical screen calibration (Tier A): metric screen size plus the
/// viewer's nominal distance. Mirrors the concept of a measured-screen
/// calibration wizard; values persist wherever the host keeps them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenCalibration {
    /// Physical screen width in metres.
    pub width_m: f64,
    /// Physical screen height in metres.
    pub height_m: f64,
    /// Nominal eye-to-screen distance in metres.
    pub base_distance_m: f64,
}

impl ScreenCalibration {
    /// Construct from centimetre measurements (how humans measure screens).
    pub fn from_cm(width_cm: f64, height_cm: f64, distance_cm: f64) -> Self {
        Self {
            width_m: width_cm / 100.0,
            height_m: height_cm / 100.0,
            base_distance_m: distance_cm / 100.0,
        }
    }

    /// The screen this calibration describes, centered at the origin
    /// (see [`crate::Screen::centered`]).
    pub fn screen(&self) -> Result<crate::Screen, crate::OffAxisError> {
        crate::Screen::centered(self.width_m, self.height_m)
    }

    /// Map a normalized head position from a fine tracker into metric eye
    /// coordinates in screen space.
    ///
    /// - `nx`, `ny`: head position in normalized image coordinates
    ///   (`[0, 1]`, origin top-left, x rightward — the usual camera-image
    ///   convention). The image is assumed mirrored (selfie view), so a
    ///   viewer moving to *their* left moves the eye left in screen space.
    /// - `depth_scale`: multiplier on the nominal distance (1.0 = at the
    ///   calibrated distance; a fine tracker derives it from e.g. apparent
    ///   inter-feature distance). Clamped to `[0.25, 4.0]`.
    /// - `lateral_range_m`: metres of eye travel represented by the full
    ///   image width/height. Deployment-specific; the screen width is a
    ///   reasonable default.
    pub fn eye_from_normalized(
        &self,
        nx: f64,
        ny: f64,
        depth_scale: f64,
        lateral_range_m: f64,
    ) -> Vec3 {
        let cx = (nx.clamp(0.0, 1.0) - 0.5) * lateral_range_m;
        let cy = (0.5 - ny.clamp(0.0, 1.0)) * lateral_range_m * (self.height_m / self.width_m);
        let z = self.base_distance_m * depth_scale.clamp(0.25, 4.0);
        Vec3::new(cx, cy, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, nz: usize, hot: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut v = vec![0.05_f32; nx * nz];
        for &(ix, iz, val) in hot {
            v[iz * nx + ix] = val;
        }
        v
    }

    #[test]
    fn peak_mapping_matches_field_localize_layout() {
        // 20×20 grid, hot cell at (ix=15, iz=4).
        let g = grid(20, 20, &[(15, 4, 0.9)]);
        let p = field_peak(&g, 20, 20).unwrap();
        assert_eq!(p.ix, 15);
        assert_eq!(p.iz, 4);
        // world_x = (15 - 10) * 0.6, world_z = (4 - 10) * 0.5 — the exact
        // transform documented in field_localize.rs / the Observatory.
        assert!((p.x - 3.0).abs() < 1e-12);
        assert!((p.z - -3.0).abs() < 1e-12);
        assert!((p.value - 0.9).abs() < 1e-6);
    }

    #[test]
    fn below_threshold_and_malformed_grids_yield_none() {
        let g = grid(20, 20, &[(3, 3, 0.34)]);
        assert_eq!(field_peak(&g, 20, 20), None, "0.34 < threshold 0.35");
        assert!(field_peak(&g, 19, 20).is_none(), "length mismatch");
        assert!(field_peak(&[], 0, 0).is_none(), "empty grid");
        // NaN cells are skipped, not propagated.
        let mut g = grid(4, 4, &[(1, 1, 0.8)]);
        g[0] = f32::NAN;
        let p = field_peak(&g, 4, 4).unwrap();
        assert_eq!((p.ix, p.iz), (1, 1));
    }

    #[test]
    fn exact_threshold_is_accepted() {
        let g = grid(20, 20, &[(2, 2, 0.35)]);
        assert!(field_peak(&g, 20, 20).is_some(), "threshold is inclusive");
    }

    #[test]
    fn coarse_parallax_is_deadbanded_gained_and_clamped() {
        let cfg = CoarseParallaxConfig::default();
        let mut cp = CoarseParallax::new(cfg);

        // First peak sets the origin: eye stays at rest.
        let origin = FieldPeak {
            x: 1.2,
            z: -0.5,
            value: 0.8,
            ix: 12,
            iz: 9,
        };
        let eye0 = cp.update(Some(origin), 0.0);
        assert_eq!(eye0, Vec3::new(0.0, 0.0, cfg.base_distance_m));

        // Movement inside the deadband: still at rest.
        let small = FieldPeak {
            x: 1.2 + 0.1,
            ..origin
        };
        let eye1 = cp.update(Some(small), 0.5);
        assert_eq!(eye1.x, 0.0, "0.1 m < 0.15 m deadband");

        // A huge excursion is clamped to max_offset regardless of gain.
        let huge = FieldPeak {
            x: 1.2 + 50.0,
            ..origin
        };
        let mut eye = Vec3::default();
        for i in 0..600 {
            eye = cp.update(Some(huge), 1.0 + i as f64 * 0.05);
        }
        assert!(eye.x <= cfg.max_offset_m + 1e-9, "clamped: {}", eye.x);
        assert!(
            eye.x > cfg.max_offset_m * 0.9,
            "filter converged near clamp"
        );

        // Below-threshold (None) holds position; never snaps back.
        let held = cp.update(None, 100.0);
        assert_eq!(held, eye);
    }

    #[test]
    fn coarse_parallax_depth_never_crosses_the_screen() {
        let cfg = CoarseParallaxConfig::default();
        let mut cp = CoarseParallax::new(cfg);
        let origin = FieldPeak {
            x: 0.0,
            z: 0.0,
            value: 0.9,
            ix: 10,
            iz: 10,
        };
        cp.update(Some(origin), 0.0);
        // Walk far toward the screen (negative z offset).
        let close = FieldPeak { z: -50.0, ..origin };
        let mut eye = Vec3::default();
        for i in 0..600 {
            eye = cp.update(Some(close), 0.1 + i as f64 * 0.05);
        }
        assert!(
            eye.z >= cfg.base_distance_m * 0.5 - 1e-9,
            "distance floor: {}",
            eye.z
        );
    }

    #[test]
    fn reset_returns_to_rest() {
        let mut cp = CoarseParallax::new(CoarseParallaxConfig::default());
        cp.update(
            Some(FieldPeak {
                x: 0.0,
                z: 0.0,
                value: 0.9,
                ix: 0,
                iz: 0,
            }),
            0.0,
        );
        cp.update(
            Some(FieldPeak {
                x: 9.0,
                z: 0.0,
                value: 0.9,
                ix: 0,
                iz: 0,
            }),
            1.0,
        );
        cp.reset();
        assert_eq!(
            cp.eye(),
            Vec3::new(0.0, 0.0, CoarseParallaxConfig::default().base_distance_m)
        );
    }

    #[test]
    fn calibration_maps_normalized_head_to_metric_eye() {
        let cal = ScreenCalibration::from_cm(60.0, 34.0, 65.0);
        assert!((cal.width_m - 0.6).abs() < 1e-12);

        // Centered head at nominal depth = centered eye at base distance.
        let c = cal.eye_from_normalized(0.5, 0.5, 1.0, cal.width_m);
        assert_eq!(c, Vec3::new(0.0, 0.0, 0.65));

        // Head at image right edge → eye half the range to the right;
        // image y grows downward → ny=0 (top) is +y in screen space.
        let e = cal.eye_from_normalized(1.0, 0.0, 1.0, 0.6);
        assert!((e.x - 0.3).abs() < 1e-12);
        assert!(e.y > 0.0);

        // Depth scale is clamped to a sane band.
        assert_eq!(cal.eye_from_normalized(0.5, 0.5, 100.0, 0.6).z, 0.65 * 4.0);
        assert_eq!(cal.eye_from_normalized(0.5, 0.5, 0.0, 0.6).z, 0.65 * 0.25);

        // Out-of-range normalized coords are clamped, not extrapolated.
        let clamped = cal.eye_from_normalized(7.0, -3.0, 1.0, 0.6);
        assert!((clamped.x - 0.3).abs() < 1e-12);
    }
}

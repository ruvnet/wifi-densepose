//! Field-peak localization for the Observatory 3D view (issue #1050).
//!
//! ## What this is (and is not)
//!
//! The `/ws/sensing` `sensing_update` frame already carries a real `signal_field`
//! — a 20×20 grid built by `generate_signal_field()` from **measured subcarrier
//! variances** weighted by the **measured motion-band power**. The grid's hot
//! cells are the strongest scatterers in that field representation; as the CSI
//! changes (a person moving through the link), the peak cell moves with it.
//!
//! This module reads the **strongest peak(s)** out of that real field and maps
//! the peak cell to the Observatory room's world coordinates. That gives the
//! 3D figure a position + motion magnitude that are **derived from real signal
//! data**, so the figure now tracks where the field energy concentrates.
//!
//! ### Honesty caveat (do not over-claim)
//!
//! The field's subcarrier→angle mapping in `generate_signal_field()` is a
//! *representation*, not calibrated multistatic triangulation in metric room
//! coordinates. A single ESP32 link cannot resolve a true (x, z) room position.
//! So the emitted `position` is **"strongest field peak in the room model"**,
//! not survey-grade localization. It is real (a function of live CSI), it moves
//! with real motion, and it is honest about its source — but it is NOT a
//! calibrated person fix. Per-person skeletal `pose` keypoints in room
//! coordinates remain gated on the pose model + paired ground-truth data
//! (ADR-079), so `pose` here is only ever set from a real aggregate posture
//! estimate when one exists, and is `None` otherwise (never fabricated).
//!
//! ## Coordinate mapping
//!
//! The Observatory builds its field point cloud (see `ui/observatory/js/main.js`
//! `_buildSignalField`) as, for grid cell `(ix, iz)` of a `20×20` grid:
//!
//! ```text
//! world_x = (ix - gridSize/2) * 0.6
//! world_z = (iz - gridSize/2) * 0.5
//! world_y = 0  (floor)
//! ```
//!
//! and indexes the field as `idx = iz * gridSize + ix` — identical to the
//! server's `generate_signal_field()` layout (`values[z * grid + x]`). We map
//! the peak cell with the **same** transform so the figure lands exactly on the
//! field hotspot it is standing on.

/// World-space scale factor for the X (width) axis, matching the Observatory's
/// `_buildSignalField`: `world_x = (ix - nx/2) * X_SCALE`.
pub const X_SCALE: f64 = 0.6;
/// World-space scale factor for the Z (depth) axis, matching the Observatory's
/// `_buildSignalField`: `world_z = (iz - nz/2) * Z_SCALE`.
pub const Z_SCALE: f64 = 0.5;

/// Minimum normalized field value (`signal_field.values` are normalized to
/// `[0, 1]`) for a cell to be considered a real peak rather than background
/// attenuation. Below this we treat the field as having no localizable hotspot.
pub const PEAK_THRESHOLD: f64 = 0.35;

/// A localized field peak in Observatory world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldPeak {
    /// World position `[x, y, z]` in Observatory scene units (meters). `y` is
    /// always `0.0` — the field is a floor-plane grid with no height info.
    pub position: [f64; 3],
    /// Normalized field intensity at the peak cell, in `[0, 1]`.
    pub intensity: f64,
    /// Source grid cell `(ix, iz)` the peak was read from (for tests/debug).
    pub cell: (usize, usize),
}

/// Map a grid cell `(ix, iz)` of an `nx × nz` field to Observatory world
/// coordinates, matching `ui/observatory/js/main.js::_buildSignalField`.
#[must_use]
pub fn cell_to_world(ix: usize, iz: usize, nx: usize, nz: usize) -> [f64; 3] {
    let wx = (ix as f64 - nx as f64 / 2.0) * X_SCALE;
    let wz = (iz as f64 - nz as f64 / 2.0) * Z_SCALE;
    [wx, 0.0, wz]
}

/// Extract up to `max_peaks` strongest, spatially-separated peaks from a
/// `signal_field` grid.
///
/// * `values` — row-major field grid, `values[iz * nx + ix]`, normalized to
///   `[0, 1]` (as produced by `generate_signal_field`).
/// * `nx`, `nz` — grid dimensions (the field's `grid_size` is `[nx, 1, nz]`).
/// * `max_peaks` — how many person positions to extract (≥ 1).
///
/// Returns peaks sorted strongest-first. Each successive peak is forced to be
/// at least `min_separation_cells` away from all previously selected peaks so
/// two persons don't collapse onto the same hotspot. Returns an **empty**
/// vector when no cell exceeds [`PEAK_THRESHOLD`] — an empty / no-presence
/// field yields no phantom person.
#[must_use]
pub fn extract_peaks(
    values: &[f64],
    nx: usize,
    nz: usize,
    max_peaks: usize,
    min_separation_cells: f64,
) -> Vec<FieldPeak> {
    if nx == 0 || nz == 0 || values.len() < nx * nz || max_peaks == 0 {
        return Vec::new();
    }

    // Collect all cells above threshold, strongest first.
    let mut candidates: Vec<(usize, usize, f64)> = Vec::new();
    for iz in 0..nz {
        for ix in 0..nx {
            let v = values[iz * nx + ix];
            if v >= PEAK_THRESHOLD {
                candidates.push((ix, iz, v));
            }
        }
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

    let mut peaks: Vec<FieldPeak> = Vec::new();
    for (ix, iz, v) in candidates {
        if peaks.len() >= max_peaks {
            break;
        }
        // Enforce spatial separation from already-chosen peaks (in cell units).
        let too_close = peaks.iter().any(|p| {
            let dx = p.cell.0 as f64 - ix as f64;
            let dz = p.cell.1 as f64 - iz as f64;
            (dx * dx + dz * dz).sqrt() < min_separation_cells
        });
        if too_close {
            continue;
        }
        peaks.push(FieldPeak {
            position: cell_to_world(ix, iz, nx, nz),
            intensity: v,
            cell: (ix, iz),
        });
    }
    peaks
}

/// Convert measured `motion_band_power` to the `motion_score` scale the
/// Observatory UI expects.
///
/// The UI compares `motion_score > 50` to switch between calm and energetic
/// emission (see `_updateDotMatrixMist` / `_updateParticleTrail`). The raw
/// `motion_band_power` is already in roughly that band for live ESP32 data
/// (the issue reports `motion_band_power: 63.3` while moving), so we pass it
/// through directly, clamped to a sane `[0, 100]` display range. This keeps the
/// emitted value a **direct, real** function of measured motion energy rather
/// than a re-scaled invention.
#[must_use]
pub fn motion_score_from_power(motion_band_power: f64) -> f64 {
    motion_band_power.clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_to_world_matches_observatory_layout() {
        // Center cell of a 20×20 grid maps near origin.
        let c = cell_to_world(10, 10, 20, 20);
        assert!((c[0] - 0.0).abs() < 1e-9);
        assert_eq!(c[1], 0.0);
        assert!((c[2] - 0.0).abs() < 1e-9);

        // Corner cell (0,0) maps to the room's near-left corner.
        let corner = cell_to_world(0, 0, 20, 20);
        assert!((corner[0] - (-6.0)).abs() < 1e-9); // (0-10)*0.6
        assert!((corner[2] - (-5.0)).abs() < 1e-9); // (0-10)*0.5
    }

    #[test]
    fn extract_peaks_finds_known_hotspot() {
        // 20×20 field, all background, single strong peak at cell (15, 4).
        let nx = 20;
        let nz = 20;
        let mut values = vec![0.05; nx * nz];
        let peak_ix = 15;
        let peak_iz = 4;
        values[peak_iz * nx + peak_ix] = 1.0;

        let peaks = extract_peaks(&values, nx, nz, 1, 3.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].cell, (peak_ix, peak_iz));

        // Position must match the Observatory cell→world transform within tol.
        let expected = cell_to_world(peak_ix, peak_iz, nx, nz);
        assert!((peaks[0].position[0] - expected[0]).abs() < 1e-9);
        assert!((peaks[0].position[2] - expected[2]).abs() < 1e-9);
        // Sanity: (15-10)*0.6 = 3.0, (4-10)*0.5 = -3.0
        assert!((peaks[0].position[0] - 3.0).abs() < 1e-9);
        assert!((peaks[0].position[2] - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn empty_field_yields_no_peaks() {
        let nx = 20;
        let nz = 20;
        // All cells below PEAK_THRESHOLD — no presence.
        let values = vec![0.10; nx * nz];
        let peaks = extract_peaks(&values, nx, nz, 3, 3.0);
        assert!(
            peaks.is_empty(),
            "below-threshold field must not produce a phantom peak"
        );
    }

    #[test]
    fn two_separated_peaks_do_not_collapse() {
        let nx = 20;
        let nz = 20;
        let mut values = vec![0.05; nx * nz];
        values[2 * nx + 3] = 0.95; // peak A at (3, 2)
        values[15 * nx + 17] = 0.90; // peak B at (17, 15)

        let peaks = extract_peaks(&values, nx, nz, 2, 3.0);
        assert_eq!(peaks.len(), 2);
        // Strongest first.
        assert_eq!(peaks[0].cell, (3, 2));
        assert_eq!(peaks[1].cell, (17, 15));
    }

    #[test]
    fn nearby_secondary_peak_is_suppressed() {
        let nx = 20;
        let nz = 20;
        let mut values = vec![0.05; nx * nz];
        values[10 * nx + 10] = 1.00; // primary
        values[10 * nx + 11] = 0.99; // adjacent — should be suppressed (sep 3.0)

        let peaks = extract_peaks(&values, nx, nz, 2, 3.0);
        assert_eq!(peaks.len(), 1, "adjacent cell must not become a 2nd person");
        assert_eq!(peaks[0].cell, (10, 10));
    }

    #[test]
    fn motion_score_passthrough_and_clamp() {
        assert!((motion_score_from_power(63.3) - 63.3).abs() < 1e-9);
        assert_eq!(motion_score_from_power(-5.0), 0.0);
        assert_eq!(motion_score_from_power(250.0), 100.0);
    }
}

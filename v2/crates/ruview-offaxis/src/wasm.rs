//! wasm-bindgen surface (compiled only for `wasm32`).
//!
//! Exposes two small classes to JavaScript:
//!
//! - [`OffAxisCamera`] — Tier A/B shared core: screen calibration + one-euro
//!   filtering + Kooima projection. Feed it an eye position each frame, read
//!   back column-major matrices ready for `THREE.Matrix4.fromArray`.
//! - [`RfParallax`] — Tier B input stage: `/ws/sensing` `signal_field`
//!   grids in, bounded coarse-parallax eye position out. Its output is
//!   coarse body parallax by construction (deadband + gain + clamp), so a
//!   demo cannot accidentally present it as head tracking.
//!
//! All methods that can fail return `Result<_, JsError>` (thrown as JS
//! exceptions); per-frame update methods instead hold the last good state so
//! a render loop never has to try/catch.

use crate::filter::{OneEuro3, OneEuroConfig};
use crate::math::Vec3;
use crate::projection::{off_axis, OffAxis, Screen};
use crate::rf::{field_peak, CoarseParallax, CoarseParallaxConfig, ScreenCalibration};
use wasm_bindgen::prelude::*;

/// Head-coupled off-axis camera for a physically calibrated screen.
#[wasm_bindgen]
pub struct OffAxisCamera {
    screen: Screen,
    cal: ScreenCalibration,
    near: f64,
    far: f64,
    filter: OneEuro3,
    current: OffAxis,
    eye: Vec3,
}

#[wasm_bindgen]
impl OffAxisCamera {
    /// Create from physical screen measurements in **centimetres** (the
    /// units a person measures with) plus clip planes in metres.
    #[wasm_bindgen(constructor)]
    pub fn new(
        screen_width_cm: f64,
        screen_height_cm: f64,
        viewing_distance_cm: f64,
        near_m: f64,
        far_m: f64,
    ) -> Result<OffAxisCamera, JsError> {
        let cal =
            ScreenCalibration::from_cm(screen_width_cm, screen_height_cm, viewing_distance_cm);
        let screen = cal.screen().map_err(|e| JsError::new(&e.to_string()))?;
        let eye = Vec3::new(0.0, 0.0, cal.base_distance_m);
        let current =
            off_axis(&screen, eye, near_m, far_m).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            screen,
            cal,
            near: near_m,
            far: far_m,
            filter: OneEuro3::new(OneEuroConfig {
                min_cutoff: 1.0,
                beta: 0.3,
                d_cutoff: 1.0,
            }),
            current,
            eye,
        })
    }

    /// Tune the one-euro filter (`min_cutoff` Hz, `beta`). Lower
    /// `min_cutoff` = smoother at rest; higher `beta` = less lag in motion.
    pub fn set_filter(&mut self, min_cutoff: f64, beta: f64) {
        self.filter.set_config(OneEuroConfig {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
        });
    }

    /// Reset filter state (e.g. after tracking was lost).
    pub fn reset_filter(&mut self) {
        self.filter.reset();
    }

    /// Update from a **metric eye position** in screen space (metres,
    /// origin = screen center, +x right, +y up, +z toward the viewer) at
    /// time `t_s` seconds. Returns `true` when the matrices were updated;
    /// `false` when the sample was rejected (eye behind screen / non-finite)
    /// and the previous matrices were held.
    pub fn update_eye(&mut self, x_m: f64, y_m: f64, z_m: f64, t_s: f64) -> bool {
        let eye = self.filter.filter(Vec3::new(x_m, y_m, z_m), t_s);
        match off_axis(&self.screen, eye, self.near, self.far) {
            Ok(oa) => {
                self.current = oa;
                self.eye = eye;
                true
            }
            Err(_) => false,
        }
    }

    /// Update from a **normalized head position** (`nx`, `ny` in `[0, 1]`,
    /// camera-image convention, mirrored/selfie view) with a depth scale
    /// (1.0 = at the calibrated distance) — the Tier A path fed by any fine
    /// tracker the host runs. `lateral_range_m` is how many metres of eye
    /// travel the full image spans (screen width is a reasonable start).
    pub fn update_normalized(
        &mut self,
        nx: f64,
        ny: f64,
        depth_scale: f64,
        lateral_range_m: f64,
        t_s: f64,
    ) -> bool {
        let eye = self
            .cal
            .eye_from_normalized(nx, ny, depth_scale, lateral_range_m);
        self.update_eye(eye.x, eye.y, eye.z, t_s)
    }

    /// The current asymmetric projection matrix, column-major, 16 elements
    /// (pass straight to `THREE.Matrix4.fromArray`).
    pub fn projection(&self) -> Vec<f64> {
        self.current.projection.to_vec()
    }

    /// The current screen-aligned view matrix, column-major, 16 elements.
    pub fn view(&self) -> Vec<f64> {
        self.current.view.to_vec()
    }

    /// `projection * view` as one matrix, column-major, 16 elements.
    pub fn view_projection(&self) -> Vec<f64> {
        self.current.view_projection().to_vec()
    }

    /// The current (filtered) eye position `[x, y, z]` in metres.
    pub fn eye(&self) -> Vec<f64> {
        vec![self.eye.x, self.eye.y, self.eye.z]
    }
}

/// Tier B input stage: signal-field grids → bounded coarse-parallax eye.
#[wasm_bindgen]
pub struct RfParallax {
    parallax: CoarseParallax,
    last_value: f64,
    has_peak: bool,
}

#[wasm_bindgen]
impl RfParallax {
    /// Create with the default conservative Tier B tuning and the given
    /// nominal viewing distance (metres).
    #[wasm_bindgen(constructor)]
    pub fn new(base_distance_m: f64) -> RfParallax {
        let cfg = CoarseParallaxConfig {
            base_distance_m,
            ..CoarseParallaxConfig::default()
        };
        RfParallax {
            parallax: CoarseParallax::new(cfg),
            last_value: 0.0,
            has_peak: false,
        }
    }

    /// Ingest one `signal_field` grid (`values.length == nx * nz`,
    /// `idx = iz * nx + ix` — the `/ws/sensing` layout) observed at `t_s`
    /// seconds. Returns `true` when a peak at/above the server's threshold
    /// was found (eye moved), `false` when the field had no localizable
    /// hotspot (eye held).
    pub fn update(&mut self, values: &[f32], nx: usize, nz: usize, t_s: f64) -> bool {
        let peak = field_peak(values, nx, nz);
        self.has_peak = peak.is_some();
        self.last_value = peak.map_or(self.last_value, |p| p.value);
        self.parallax.update(peak, t_s);
        self.has_peak
    }

    /// The current eye position `[x, y, z]` in metres (screen space) —
    /// feed to `OffAxisCamera.update_eye`.
    pub fn eye(&self) -> Vec<f64> {
        let e = self.parallax.eye();
        vec![e.x, e.y, e.z]
    }

    /// Whether the last grid had an at/above-threshold peak.
    pub fn has_peak(&self) -> bool {
        self.has_peak
    }

    /// The last accepted peak's normalized field value (HUD display).
    pub fn peak_value(&self) -> f64 {
        self.last_value
    }

    /// Forget the session origin (person left / demo reset).
    pub fn reset(&mut self) {
        self.parallax.reset();
        self.has_peak = false;
        self.last_value = 0.0;
    }
}

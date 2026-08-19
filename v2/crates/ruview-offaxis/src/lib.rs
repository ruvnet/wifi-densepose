//! # `ruview-offaxis` — clean-room off-axis (head-coupled) perspective (ADR-324)
//!
//! Generalized perspective projection for "window into the screen"
//! (fish-tank VR / head-coupled perspective) demos, implemented from the
//! published math — Robert Kooima, *Generalized Perspective Projection*
//! (2008) — plus the supporting input stages the ADR-324 demo needs:
//!
//! - [`Screen`] / [`off_axis`]: three physical screen corners + a tracked
//!   eye → asymmetric frustum and screen-aligned view matrix (column-major
//!   `f64`, the three.js `Matrix4.elements` layout).
//! - [`OneEuro`] / [`OneEuro3`]: the one-euro filter (Casiez et al., CHI
//!   2012) for tracking-noise smoothing with injected timestamps.
//! - [`rf`]: RuView-specific input mapping — `/ws/sensing` signal-field
//!   peak extraction (constants mirroring the sensing server's
//!   `field_localize.rs`) and the bounded **Tier B** coarse-parallax stage.
//! - A wasm-bindgen surface (wasm32 only) exposing [`wasm::OffAxisCamera`]
//!   and [`wasm::RfParallax`] to the browser demos.
//!
//! ## Clean-room statement
//!
//! ADR-324 records that the prior-art repository (`icurtis1/off-axis-sneaker`)
//! is unlicensed: no code, assets, or derived text from it appear here. This
//! crate is written solely from the published Kooima and Casiez papers and
//! RuView's own source.
//!
//! ## Honesty contract (repo rule)
//!
//! Nothing in this crate asserts sensing accuracy. The Tier B RF path is
//! **coarse body parallax by construction** — deadbanded, gain-limited,
//! hard-clamped — because a single-link CSI field peak is a representation
//! of field energy, not metric localization (see
//! `wifi-densepose-sensing-server/src/field_localize.rs`). Numeric defaults
//! are interaction-design choices (`CLAIMED`); benchmark numbers live in the
//! README tagged `MEASURED` with their reproducer.
//!
//! ## Native quick start
//!
//! ```
//! use ruview_offaxis::{off_axis, Screen, Vec3};
//!
//! // A 60 cm × 34 cm screen centered at the origin, eye 65 cm away and
//! // 10 cm to the right.
//! let screen = Screen::centered(0.60, 0.34)?;
//! let oa = off_axis(&screen, Vec3::new(0.10, 0.0, 0.65), 0.05, 100.0)?;
//! // Column-major, ready for three.js Matrix4.fromArray / any GL pipeline.
//! let _m: [f64; 16] = oa.view_projection();
//! # Ok::<(), ruview_offaxis::OffAxisError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod filter;
pub mod math;
pub mod projection;
pub mod rf;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use filter::{OneEuro, OneEuro3, OneEuroConfig};
pub use math::{Mat4, Vec3};
pub use projection::{off_axis, OffAxis, OffAxisError, Screen, MIN_EYE_DISTANCE};
pub use rf::{
    field_peak, CoarseParallax, CoarseParallaxConfig, FieldPeak, ScreenCalibration,
    FIELD_PEAK_THRESHOLD, FIELD_X_SCALE, FIELD_Z_SCALE,
};

//! ADR-117 — PyO3 bindings for the WiFi-DensePose Rust core.
//!
//! This crate is the compiled half of the `wifi-densepose` v2.x PyPI
//! wheel. The Python-facing facade lives in `python/wifi_densepose/`
//! and re-exports symbols from this module under their stable names.
//!
//! ## Phase status (per ADR-117 §6)
//!
//! - **P1 (scaffold) — this commit**: module loads, version constant
//!   exposed, smoke test passes via maturin develop.
//! - **P2**: bind `CsiFrame`, `Keypoint`, `PoseEstimate` (next).
//! - **P3**: bind 4-stage vitals + signal DSP.
//! - **P4**: pure-Python `wifi_densepose.client` (WS/MQTT) — no Rust
//!   surface needed; lives outside this crate.
//! - **P5**: cibuildwheel + PyPI publish.

use pyo3::prelude::*;

mod bindings {
    pub mod bfld;
    pub mod keypoint;
    pub mod pose;
    pub mod privacy_gate;
    pub mod vitals;
}

/// Version of the bound Rust core. Surfaced to Python as
/// `wifi_densepose.__rust_version__` so users can correlate wheel
/// behaviour with the exact `v2/crates/` HEAD it was built from.
const RUST_CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compile-time identifier for the Rust commit that produced this
/// wheel. Surfaced for diagnostics. Set via `CARGO_PKG_VERSION` for
/// now; P5 wires in the git SHA via `vergen`.
const RUST_BUILD_TAG: &str = env!("CARGO_PKG_VERSION");

/// One-line description of which feature flags were enabled at build
/// time. Helps users debug "is my wheel the slim one or the full one?".
fn build_features() -> Vec<&'static str> {
    let mut feats: Vec<&'static str> = Vec::new();
    feats.push("p1-scaffold");
    feats.push("p2-keypoint-bindings"); // Keypoint + KeypointType
    feats.push("p2-pose-bindings"); // BoundingBox + PersonPose + PoseEstimate
    feats.push("p3-vitals-bindings"); // BreathingExtractor + HeartRateExtractor + VitalEstimate
    feats.push("p3.5-bfld-bindings"); // BfldFrame + BfldReport + BfldKind (stub Rust)
    feats
}

/// Quick smoke test exposed to Python. Returns "ok" — used by the
/// integration tests in `python/tests/test_smoke.py` to assert the
/// PyO3 module is importable and callable.
#[pyfunction]
fn hello() -> PyResult<&'static str> {
    Ok("ok")
}

/// The `_native` module — re-exported in pure-Python as
/// `wifi_densepose._native`. End users should import the parent
/// package (`import wifi_densepose`) and never reach into `_native`
/// directly; the leading underscore is a Python convention marking
/// it as private.
///
/// The function name MUST match the `module-name` in pyproject.toml's
/// `[tool.maturin]` block — i.e. it must be `_native` because the
/// pyproject says `module-name = "wifi_densepose._native"`. PyO3
/// generates the `PyInit__native` symbol from this function name.
#[pymodule]
#[pyo3(name = "_native")]
fn wifi_densepose_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__rust_version__", RUST_CORE_VERSION)?;
    m.add("__rust_build_tag__", RUST_BUILD_TAG)?;
    m.add("__build_features__", build_features())?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;

    // P2 — Keypoint + KeypointType bindings.
    bindings::keypoint::register(m)?;
    // P2 — BoundingBox + PersonPose + PoseEstimate bindings.
    bindings::pose::register(m)?;
    // P3 — Vital sign extraction bindings.
    bindings::vitals::register(m)?;
    // P3.5 — BFLD bindings (stub Rust; future wifi-densepose-bfld crate
    // will replace the stub without changing the Python API).
    bindings::bfld::register(m)?;
    // ADR-118 PrivacyClass + HAP/Matter eligibility gates (SOTA — backed by
    // the published `wifi-densepose-bfld 0.3.0` crate, not the Python port).
    // Closes ADR-125 §2.1.d at the binding boundary.
    bindings::privacy_gate::register(m)?;
    Ok(())
}

//! Criterion benches for the heaviest `process_frame` hot paths in the edge
//! skill library (ADR-163, closing the ADR-160 §"Deferred Backlog" item
//! "Criterion benches for process_frame budget claims").
//!
//! ## HONEST SCOPE — read this before citing any number here
//!
//! These benches measure **HOST** wall-clock latency on a development laptop.
//! The per-module doc budgets (e.g. `exo_time_crystal` "H (heavy, <10ms) on
//! ESP32-S3 WASM3") are **for a different target**: an Xtensa ESP32-S3 running
//! the WASM3 interpreter. A native x86_64 host with `-O` is an **upper-bound
//! proxy for the ALGORITHM cost only**; it is NOT the ESP32 number and does NOT
//! reproduce the ESP32 budget. WASM3 interpretation on a ~240 MHz Xtensa core is
//! typically 1-2 orders of magnitude slower than native host code, so a host
//! median well under the budget does NOT prove the ESP32 meets it — it only
//! bounds the work. The ESP32 figure remains UNMEASURED (needs hardware).
//!
//! What these benches DO prove (MEASURED-on-host):
//!   * the hot paths run, on a fixed synthetic CSI frame, with a real median;
//!   * a regression guard exists so a future change that 10×'s the host cost
//!     is caught in CI/dev even before anyone reflashes an ESP32.
//!
//! Run (the crate is EXCLUDED from the v2 workspace — bench from the crate dir):
//!   cd v2/crates/wifi-densepose-wasm-edge
//!   cargo bench --features std
//!   # quick smoke:
//!   cargo bench --features std -- --warm-up-time 1 --measurement-time 2
//!
//! `med_seizure_detect` is gated behind `medical-experimental`; its bench is
//! `#[cfg(feature = "medical-experimental")]` and only runs when that feature is
//! also enabled:
//!   cargo bench --features std,medical-experimental

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;

use wifi_densepose_wasm_edge::exo_ghost_hunter::GhostHunterDetector;
use wifi_densepose_wasm_edge::exo_time_crystal::TimeCrystalDetector;
use wifi_densepose_wasm_edge::sec_weapon_detect::WeaponDetector;

// ── Fixed synthetic CSI fixtures (deterministic LCG, seed-stable) ────────────

/// Deterministic pseudo-random in [lo, hi) from a 32-bit LCG, matching the
/// generator style used by `tests/budget_compliance.rs`.
fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (*seed >> 16) as f32 / 32768.0
}

fn synthetic_phases(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * 6.2832 - 3.1416).collect()
}

fn synthetic_amplitudes(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * 10.0 + 0.1).collect()
}

fn synthetic_variance(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| lcg(&mut s) * 2.0 + 0.05).collect()
}

const N_SC: usize = 32; // per-subcarrier width (matches both modules' MAX_SC)

// ── exo_time_crystal: compute_autocorrelation 256×128 hot path ───────────────
//
// `compute_autocorrelation` is private, so we drive it through the public
// `process_frame`. To hit the full 256-point × 128-lag autocorrelation the
// circular buffer must be FULL (≥256 samples) and the signal must be
// non-constant (the module early-outs on `buf_var < 1e-8`). We pre-fill once
// with a periodic-plus-noise motion-energy stream, then bench a single
// `process_frame` (each call recomputes the full 256×128 autocorrelation =
// ~32K multiply-accumulates, the M6-audit-named hot path).

fn prefilled_time_crystal() -> TimeCrystalDetector {
    let mut d = TimeCrystalDetector::new();
    let mut s = 0xC0FFEEu32;
    // 300 frames (> BUF_LEN=256) so the buffer is full and statistics are warm.
    for i in 0..300 {
        // period-10 square wave + small noise → guarantees buf_var > 0 and a
        // genuine autocorrelation structure (the expensive path runs).
        let base = if (i % 10) < 5 { 1.0 } else { 0.0 };
        let me = base + lcg(&mut s) * 0.05;
        black_box(d.process_frame(black_box(me)));
    }
    d
}

fn bench_exo_time_crystal(c: &mut Criterion) {
    c.bench_function("exo_time_crystal::process_frame[autocorr_256x128]", |b| {
        let mut s = 0x1357_9BDFu32;
        b.iter_batched(
            prefilled_time_crystal,
            |mut d| {
                // One frame = one full 256×128 autocorrelation pass.
                let me = if (d.frame_count() % 10) < 5 { 1.0 } else { 0.0 } + lcg(&mut s) * 0.05;
                black_box(d.process_frame(black_box(me)));
            },
            BatchSize::SmallInput,
        );
    });
}

// ── exo_ghost_hunter: periodicity + hidden-breathing hot path ────────────────
//
// Heaviest path runs only when the room is reported EMPTY (presence == 0):
// per-group anomaly accumulation + aggregate-phase autocorrelation for hidden
// periodic (breathing) signatures. We warm the noise floor + phase buffer first,
// then bench one empty-room frame.

fn prefilled_ghost_hunter() -> GhostHunterDetector {
    let mut d = GhostHunterDetector::new();
    let mut s = 0xBADC0DEu32;
    // Warm the per-group EWMA noise floors + fill the phase buffer (PHASE_BUF_LEN=64)
    // with a periodic phase signal so the periodicity autocorrelation has structure.
    for i in 0..120u32 {
        let phases: Vec<f32> = (0..N_SC)
            .map(|k| libm::sinf(i as f32 * 0.4 + k as f32 * 0.1) * 0.3 + lcg(&mut s) * 0.02)
            .collect();
        let amps = synthetic_amplitudes(N_SC, 4000 + i);
        let var = synthetic_variance(N_SC, 4500 + i);
        black_box(d.process_frame(&phases, &amps, &var, 0, 0.05));
    }
    d
}

fn bench_exo_ghost_hunter(c: &mut Criterion) {
    let amps = synthetic_amplitudes(N_SC, 9000);
    let var = synthetic_variance(N_SC, 9500);
    c.bench_function("exo_ghost_hunter::process_frame[empty_room_periodicity]", |b| {
        let mut s = 0x2468_ACE0u32;
        b.iter_batched(
            prefilled_ghost_hunter,
            |mut d| {
                let i = d.frame_count();
                let phases: Vec<f32> = (0..N_SC)
                    .map(|k| libm::sinf(i as f32 * 0.4 + k as f32 * 0.1) * 0.3 + lcg(&mut s) * 0.02)
                    .collect();
                black_box(d.process_frame(
                    black_box(&phases),
                    black_box(&amps),
                    black_box(&var),
                    black_box(0),
                    black_box(0.05),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

// ── sec_weapon_detect: per-subcarrier Welford hot path ───────────────────────
//
// After calibration the detector runs a per-subcarrier online Welford update
// over MAX_SC=32 subcarriers each frame (the M6-audit-named hot path). We
// calibrate first (the early frames just accumulate baseline stats), then bench
// one steady-state frame.

fn calibrated_weapon_detector() -> WeaponDetector {
    let mut d = WeaponDetector::new();
    // Drive enough empty-room frames to complete calibration + warm the running
    // Welford state. Calibration window is internal; 200 frames is comfortably
    // past it for MAX_SC=32.
    for i in 0..200u32 {
        let phases = synthetic_phases(N_SC, 6000 + i);
        let amps = synthetic_amplitudes(N_SC, 6500 + i);
        let var = synthetic_variance(N_SC, 7000 + i);
        black_box(d.process_frame(&phases, &amps, &var, 0.05, 0));
    }
    d
}

fn bench_sec_weapon_detect(c: &mut Criterion) {
    c.bench_function("sec_weapon_detect::process_frame[per_sc_welford]", |b| {
        let mut seed = 8000u32;
        b.iter_batched(
            calibrated_weapon_detector,
            |mut d| {
                seed = seed.wrapping_add(1);
                let phases = synthetic_phases(N_SC, seed);
                let amps = synthetic_amplitudes(N_SC, seed.wrapping_add(500));
                let var = synthetic_variance(N_SC, seed.wrapping_add(1000));
                black_box(d.process_frame(
                    black_box(&phases),
                    black_box(&amps),
                    black_box(&var),
                    black_box(0.3),
                    black_box(1),
                ));
            },
            BatchSize::SmallInput,
        );
    });
}

// ── med_seizure_detect: detect_rhythm / clonic autocorrelation hot path ──────
//
// Gated behind `medical-experimental` (ADR-160 §A1). The clonic-phase rhythm
// detection autocorrelates the amplitude ring buffer (PHASE_WINDOW=100); we warm
// the buffers with a high-energy rhythmic signal, then bench one frame.
#[cfg(feature = "medical-experimental")]
mod med {
    use super::*;
    use wifi_densepose_wasm_edge::med_seizure_detect::SeizureDetector;

    fn warmed_seizure_detector() -> SeizureDetector {
        let mut d = SeizureDetector::new();
        let mut s = 0x5EE_D00Du32;
        // High-energy ~4 Hz rhythmic (period ~5 frames at 20 Hz) → exercises the
        // clonic-phase rhythm/autocorrelation path, with presence asserted.
        for i in 0..150u32 {
            let me = 2.5 + libm::sinf(i as f32 * 1.25) * 1.5;
            let amp = 1.0 + lcg(&mut s) * 0.2;
            black_box(d.process_frame(0.0, amp, me, 1));
        }
        d
    }

    pub fn bench_med_seizure_detect(c: &mut Criterion) {
        c.bench_function("med_seizure_detect::process_frame[clonic_rhythm]", |b| {
            let mut s = 0x9A_BCDE_F0u32;
            b.iter_batched(
                warmed_seizure_detector,
                |mut d| {
                    let i = d.frame_count();
                    let me = 2.5 + libm::sinf(i as f32 * 1.25) * 1.5;
                    let amp = 1.0 + lcg(&mut s) * 0.2;
                    black_box(d.process_frame(
                        black_box(0.0),
                        black_box(amp),
                        black_box(me),
                        black_box(1),
                    ));
                },
                BatchSize::SmallInput,
            );
        });
    }
}

#[cfg(feature = "medical-experimental")]
criterion_group!(
    benches,
    bench_exo_time_crystal,
    bench_exo_ghost_hunter,
    bench_sec_weapon_detect,
    med::bench_med_seizure_detect,
);

#[cfg(not(feature = "medical-experimental"))]
criterion_group!(
    benches,
    bench_exo_time_crystal,
    bench_exo_ghost_hunter,
    bench_sec_weapon_detect,
);

criterion_main!(benches);

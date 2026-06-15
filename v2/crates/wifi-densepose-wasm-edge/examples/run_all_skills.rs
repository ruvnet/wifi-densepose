//! Runnable demo of the unified [`EdgePipeline`]: constructs every registered
//! skill, feeds a short deterministic synthetic CSI frame sequence, and prints
//! the per-skill events plus a registration summary.
//!
//! ```bash
//! cd v2/crates/wifi-densepose-wasm-edge
//! cargo run --example run_all_skills --features std
//! cargo run --example run_all_skills --features std,medical-experimental
//! ```
//!
//! [`EdgePipeline`]: wifi_densepose_wasm_edge::pipeline_all::EdgePipeline

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("run_all_skills requires --features std");
}

#[cfg(feature = "std")]
fn main() {
    use std::collections::BTreeMap;
    use wifi_densepose_wasm_edge::pipeline_all::{CsiFrameView, EdgePipeline};

    const N_SC: usize = 32;
    let mut pipeline = EdgePipeline::new();

    println!("=== EdgePipeline registration ===");
    println!("registered skills: {}", pipeline.skill_count());
    let med = pipeline
        .skills()
        .iter()
        .filter(|s| s.medical_experimental)
        .count();
    println!(
        "  default tier: {}   medical-experimental tier: {}",
        pipeline.skill_count() - med,
        med
    );
    println!();

    let mut phases = [0.0f32; N_SC];
    let mut amps = [0.0f32; N_SC];
    let mut vars = [0.0f32; N_SC];
    let mut prev = [0.0f32; N_SC];

    // Per-skill event counters over the run.
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for s in pipeline.skills() {
        counts.insert(s.name, 0);
    }

    let frames = 300usize;
    for t in 0..frames {
        let tf = t as f32;
        let breath = (tf * 2.0 * std::f32::consts::PI * 0.3 / 20.0).sin();
        let heart = (tf * 2.0 * std::f32::consts::PI * 1.2 / 20.0).sin();
        let mut vmean = 0.0f32;
        for i in 0..N_SC {
            let sc = i as f32;
            phases[i] = (sc * 0.21 + tf * 0.05).sin() + 0.15 * breath;
            amps[i] = 1.0 + 0.3 * (sc * 0.11 + tf * 0.03).cos() + 0.1 * heart;
            vars[i] = 0.02 + 0.01 * (sc * 0.3).sin().abs()
                + if (t / 40) % 2 == 0 { 0.05 } else { 0.0 };
            vmean += vars[i];
        }
        vmean /= N_SC as f32;

        let v = CsiFrameView {
            phases: &phases,
            amplitudes: &amps,
            variances: &vars,
            prev_phases: &prev,
            presence: if (t / 30) % 3 == 0 { 0 } else { 1 },
            n_persons: ((t / 50) % 3) as i32,
            motion_energy: 0.3 + 0.2 * (tf * 0.07).sin().abs(),
            breathing_bpm: 18.0 + 2.0 * (tf * 0.01).sin(),
            heartrate_bpm: 72.0 + 5.0 * (tf * 0.02).sin(),
            coherence: 0.5 + 0.4 * (tf * 0.03).cos(),
            variance_mean: vmean,
        };

        for e in pipeline.on_frame(&v) {
            *counts.entry(e.skill).or_insert(0) += 1;
            // Print the first few events from the last frame to show liveness.
            if t == frames - 1 {
                println!(
                    "  frame {} | {:<26} event {:>3} = {:.4}",
                    t, e.skill, e.event_id, e.value
                );
            }
        }
        prev.copy_from_slice(&phases);
    }

    println!();
    println!("=== per-skill event totals over {} synthetic frames ===", frames);
    let total: usize = counts.values().sum();
    let active = counts.values().filter(|&&c| c > 0).count();
    for (name, c) in &counts {
        println!("  {:<28} {}", name, c);
    }
    println!();
    println!(
        "TOTAL events: {}   skills that emitted at least once: {}/{}",
        total,
        active,
        pipeline.skill_count()
    );
}

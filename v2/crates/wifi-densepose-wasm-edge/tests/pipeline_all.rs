//! Integration test for the unified [`EdgePipeline`] (ADR-160 deliverable 1).
//!
//! Proves that EVERY registered skill executes over a deterministic synthetic
//! CSI frame sequence without panicking, that the aggregated event stream is
//! well-formed (each event tagged with a known skill name + a declared event
//! id), and pins the registered-skill count (default vs +medical-experimental).
//!
//! Run:
//!   cargo test --features std --test pipeline_all
//!   cargo test --features std,medical-experimental --test pipeline_all
//!
//! [`EdgePipeline`]: wifi_densepose_wasm_edge::pipeline_all::EdgePipeline

#![cfg(feature = "std")]

use wifi_densepose_wasm_edge::pipeline_all::{CsiFrameView, EdgePipeline};

const N_SC: usize = 32;

/// Deterministic synthetic frame: a moving breathing/heartbeat target plus
/// structured per-subcarrier phase/amplitude. No randomness — fully reproducible.
fn synth_frame(t: usize, phases: &mut [f32], amps: &mut [f32], vars: &mut [f32]) {
    let tf = t as f32;
    // 0.3 Hz breathing modulation @ 20 Hz frame rate -> period ~66 frames.
    let breath = (tf * 2.0 * core::f32::consts::PI * 0.3 / 20.0).sin();
    // 1.2 Hz heartbeat.
    let heart = (tf * 2.0 * core::f32::consts::PI * 1.2 / 20.0).sin();
    for i in 0..phases.len() {
        let sc = i as f32;
        phases[i] = (sc * 0.21 + tf * 0.05).sin() + 0.15 * breath;
        amps[i] = 1.0 + 0.3 * (sc * 0.11 + tf * 0.03).cos() + 0.1 * heart;
        // motion-correlated variance, with one occasionally-hot zone.
        vars[i] = 0.02 + 0.01 * (sc * 0.3).sin().abs() + if (t / 40) % 2 == 0 { 0.05 } else { 0.0 };
    }
}

/// Build a view over the supplied buffers for frame `t`.
fn view<'a>(
    t: usize,
    phases: &'a [f32],
    amps: &'a [f32],
    vars: &'a [f32],
    prev_phases: &'a [f32],
) -> CsiFrameView<'a> {
    let tf = t as f32;
    let motion = 0.3 + 0.2 * (tf * 0.07).sin().abs();
    let mut vmean = 0.0f32;
    for &v in vars {
        vmean += v;
    }
    vmean /= vars.len().max(1) as f32;
    CsiFrameView {
        phases,
        amplitudes: amps,
        variances: vars,
        prev_phases,
        presence: if (t / 30) % 3 == 0 { 0 } else { 1 },
        n_persons: ((t / 50) % 3) as i32,
        motion_energy: motion,
        breathing_bpm: 18.0 + 2.0 * (tf * 0.01).sin(),
        heartrate_bpm: 72.0 + 5.0 * (tf * 0.02).sin(),
        coherence: 0.5 + 0.4 * (tf * 0.03).cos(),
        variance_mean: vmean,
    }
}

#[test]
fn all_skills_execute_without_panic_over_synthetic_stream() {
    let mut pipeline = EdgePipeline::new();
    let n_skills = pipeline.skill_count();
    assert!(n_skills > 0, "pipeline must register skills");

    let mut phases = [0.0f32; N_SC];
    let mut amps = [0.0f32; N_SC];
    let mut vars = [0.0f32; N_SC];
    let mut prev_phases = [0.0f32; N_SC];

    let known: std::collections::HashSet<&'static str> =
        pipeline.skills().iter().map(|s| s.name).collect();

    // Feed 300 frames (15 s @ 20 Hz) — enough for calibration windows, DTW
    // enrollment, periodicity buffers, and timer cadences to fire.
    let mut total_events = 0usize;
    for t in 0..300 {
        synth_frame(t, &mut phases, &mut amps, &mut vars);
        let v = view(t, &phases, &amps, &vars, &prev_phases);
        let events = pipeline.on_frame(&v);
        for e in &events {
            // Every event must be tagged with a registered skill name.
            assert!(known.contains(e.skill), "unknown skill tag: {}", e.skill);
            // Value must be finite (no NaN/Inf leaking from the DSP).
            assert!(e.value.is_finite(), "non-finite value from {}", e.skill);
        }
        total_events += events.len();
        prev_phases.copy_from_slice(&phases);
    }

    assert_eq!(pipeline.frame_count(), 300);
    // A real run over 300 frames must emit *some* events across 59+ skills.
    assert!(
        total_events > 0,
        "expected the skill library to emit events over 300 frames, got 0"
    );
    println!(
        "pipeline: {} skills, {} aggregated events over 300 synthetic frames",
        n_skills, total_events
    );
}

#[test]
fn every_emitted_event_id_is_declared_by_its_skill() {
    // Stronger well-formedness: each event's id must be one the producing skill
    // declared in its `event_ids()` introspection list.
    let mut pipeline = EdgePipeline::new();

    // skill name -> its declared event id set
    let mut declared: std::collections::HashMap<&'static str, std::collections::HashSet<i32>> =
        std::collections::HashMap::new();
    for s in pipeline.skills() {
        declared.insert(s.name, s.event_ids.iter().copied().collect());
    }

    let mut phases = [0.0f32; N_SC];
    let mut amps = [0.0f32; N_SC];
    let mut vars = [0.0f32; N_SC];
    let mut prev_phases = [0.0f32; N_SC];

    for t in 0..300 {
        synth_frame(t, &mut phases, &mut amps, &mut vars);
        let v = view(t, &phases, &amps, &vars, &prev_phases);
        for e in &pipeline.on_frame(&v) {
            let set = declared.get(e.skill).expect("skill declared");
            assert!(
                set.contains(&e.event_id),
                "{} emitted undeclared event id {}",
                e.skill,
                e.event_id
            );
        }
        prev_phases.copy_from_slice(&phases);
    }
}

#[test]
fn introspection_lists_every_skill_with_event_ids() {
    let pipeline = EdgePipeline::new();
    let infos = pipeline.skills();
    assert_eq!(infos.len(), pipeline.skill_count());
    for info in &infos {
        assert!(!info.name.is_empty());
        assert!(
            !info.event_ids.is_empty(),
            "skill {} declares no event ids",
            info.name
        );
    }
    // No duplicate skill names.
    let names: std::collections::HashSet<_> = infos.iter().map(|i| i.name).collect();
    assert_eq!(names.len(), infos.len(), "duplicate skill registration");
}

#[cfg(not(feature = "medical-experimental"))]
#[test]
fn default_tier_count_excludes_medical() {
    let pipeline = EdgePipeline::new();
    assert_eq!(
        pipeline.skill_count(),
        59,
        "default (non-medical) tier must register exactly 59 skills"
    );
    // The ADR-160 safety gate: no med_* skill is present in the default build.
    for info in pipeline.skills() {
        assert!(
            !info.medical_experimental,
            "medical skill {} leaked into default tier",
            info.name
        );
        assert!(
            !info.name.starts_with("med_"),
            "med_* skill {} present without the medical-experimental feature",
            info.name
        );
    }
}

#[cfg(feature = "medical-experimental")]
#[test]
fn medical_tier_adds_five_skills() {
    let pipeline = EdgePipeline::new();
    assert_eq!(
        pipeline.skill_count(),
        64,
        "default 59 + 5 medical = 64 skills"
    );
    let med: Vec<_> = pipeline
        .skills()
        .into_iter()
        .filter(|s| s.medical_experimental)
        .collect();
    assert_eq!(med.len(), 5, "exactly 5 medical-experimental skills");
    for m in &med {
        assert!(
            m.name.starts_with("med_"),
            "medical-flagged skill has non-med_ name: {}",
            m.name
        );
    }
}

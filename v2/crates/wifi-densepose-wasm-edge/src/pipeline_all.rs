//! Unified edge pipeline — registers **every** runtime skill module in the crate
//! behind one uniform [`EdgeSkill`] trait and runs them all per CSI frame.
//!
//! # Why this module exists
//!
//! Each skill in `src/*.rs` is an independently-loadable DSP module with its own
//! bespoke `process_frame` / `on_timer` signature (some take `&[f32]` phases,
//! some scalars like `motion_energy`, some `breathing_bpm`/`heartrate_bpm`, etc.).
//! On the wasm target only the flagship `gesture + coherence + adversarial`
//! pipeline (in `lib.rs`) is on the default `on_frame` path. This module wires
//! **all** of them into a single [`EdgePipeline`] so a host can run the whole
//! skill library over one CSI frame stream and collect every emitted event,
//! tagged by its source skill.
//!
//! # Design
//!
//! - [`CsiFrameView`] — a borrowed, host-supplied view of one CSI frame carrying
//!   every input any skill needs (phase/amplitude/variance slices + the scalar
//!   features the host derives: presence, n_persons, motion_energy, breathing &
//!   heart rate, coherence, plus the previous frame's phases for delta skills).
//! - [`EdgeSkill`] — the uniform adapter trait. Each skill gets a small adapter
//!   (see `skill_registry`) that pulls the fields it needs out of the view, calls
//!   the underlying detector **unchanged**, and returns an aggregated
//!   `&[(i32, f32)]` event buffer. **No skill DSP is modified.**
//! - [`EdgePipeline`] — owns one boxed adapter per skill, dispatches `on_frame`
//!   to all of them, and aggregates `(skill_name, event_id, value)` triples.
//!
//! # Feature gating (preserves the ADR-160 safety gate)
//!
//! The five `med_*` skills are registered **only** under
//! `--features medical-experimental`. They are NOT pulled into the default
//! pipeline, so they cannot be silently built into a shipping artifact. The
//! medical tier is opt-in; see `EdgePipeline::new` and `skills()`.
//!
//! Requires `std` (uses `Box`/`Vec`); the wasm `no_std` build keeps the small
//! flagship `lib.rs` pipeline instead.

#![cfg(feature = "std")]

extern crate std;
use std::boxed::Box;
use std::vec::Vec;

/// Borrowed view of one CSI frame: every input any registered skill can consume.
///
/// The host derives these from the Tier-2 DSP output. Slices are
/// per-subcarrier; scalars are frame-level aggregates. A skill adapter reads
/// only the fields it needs and ignores the rest — heterogeneity is absorbed
/// here, not in the skills.
#[derive(Clone, Copy)]
pub struct CsiFrameView<'a> {
    /// Per-subcarrier unwrapped phase (radians).
    pub phases: &'a [f32],
    /// Per-subcarrier amplitude (linear).
    pub amplitudes: &'a [f32],
    /// Per-subcarrier short-window variance.
    pub variances: &'a [f32],
    /// Previous frame's phases (for delta/velocity skills like the spiking tracker).
    pub prev_phases: &'a [f32],
    /// Presence flag from host (0 = empty, 1 = occupied).
    pub presence: i32,
    /// Estimated person count from host.
    pub n_persons: i32,
    /// Frame-level motion energy.
    pub motion_energy: f32,
    /// Breathing rate estimate (breaths/min); 0 if unavailable.
    pub breathing_bpm: f32,
    /// Heart rate estimate (beats/min); 0 if unavailable.
    pub heartrate_bpm: f32,
    /// Coherence score [0,1] from the coherence monitor (for gate-style skills).
    pub coherence: f32,
    /// Mean variance across `variances` (convenience scalar for skills wanting one).
    pub variance_mean: f32,
}

impl<'a> CsiFrameView<'a> {
    /// Mean amplitude across the frame (convenience for scalar-input skills).
    #[inline]
    pub fn amplitude_mean(&self) -> f32 {
        if self.amplitudes.is_empty() {
            return 0.0;
        }
        let mut s = 0.0f32;
        for &a in self.amplitudes {
            s += a;
        }
        s / self.amplitudes.len() as f32
    }

    /// Mean phase across the frame.
    #[inline]
    pub fn phase_mean(&self) -> f32 {
        if self.phases.is_empty() {
            return 0.0;
        }
        let mut s = 0.0f32;
        for &p in self.phases {
            s += p;
        }
        s / self.phases.len() as f32
    }
}

/// One emitted event, tagged by its source skill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkillEvent {
    /// Stable name of the skill that produced this event (e.g. `"occupancy"`).
    pub skill: &'static str,
    /// Event type id (the registry id from `event_types`).
    pub event_id: i32,
    /// Event payload value.
    pub value: f32,
}

/// Uniform adapter trait over a heterogeneous skill detector.
///
/// Implementors live in `skill_registry`; each wraps exactly one underlying
/// detector and forwards `on_frame` to its real `process_frame`/`on_timer`
/// without changing the DSP. `event_ids()` is introspection only.
pub trait EdgeSkill {
    /// Stable skill name (matches the `src/<name>.rs` module).
    fn name(&self) -> &'static str;
    /// The event ids this skill can emit (for introspection / docs).
    fn event_ids(&self) -> &'static [i32];
    /// Run this skill over one frame, returning its emitted `(event_id, value)`
    /// pairs. Returns an empty slice if the skill emitted nothing this frame.
    fn on_frame(&mut self, frame: &CsiFrameView) -> &[(i32, f32)];
}

/// Introspection record for one registered skill.
#[derive(Clone, Copy, Debug)]
pub struct SkillInfo {
    /// Skill name.
    pub name: &'static str,
    /// Event ids the skill can emit.
    pub event_ids: &'static [i32],
    /// Whether the skill is part of the gated `medical-experimental` tier.
    pub medical_experimental: bool,
}

/// The unified pipeline: holds one adapter per registered skill and runs them
/// all per frame.
pub struct EdgePipeline {
    skills: Vec<Box<dyn EdgeSkill>>,
    /// Parallel flag marking which entries are the gated medical tier.
    medical_flags: Vec<bool>,
    frame_count: u64,
}

impl EdgePipeline {
    /// Construct the pipeline with **every** registered skill.
    ///
    /// The five `med_*` skills are included **only** when the crate is built
    /// with `--features medical-experimental`; otherwise the default
    /// (non-medical) tier is registered. This preserves the ADR-160 safety gate.
    pub fn new() -> Self {
        let mut skills: Vec<Box<dyn EdgeSkill>> = Vec::new();
        let mut medical_flags: Vec<bool> = Vec::new();

        crate::skill_registry::register_default(&mut skills, &mut medical_flags);
        #[cfg(feature = "medical-experimental")]
        crate::skill_registry::register_medical(&mut skills, &mut medical_flags);

        Self {
            skills,
            medical_flags,
            frame_count: 0,
        }
    }

    /// Number of registered skills (default tier, or +medical if that feature is on).
    pub fn skill_count(&self) -> usize {
        self.skills.len()
    }

    /// Run every registered skill over one frame, aggregating all emitted events
    /// tagged by source skill. Order matches registration order.
    pub fn on_frame(&mut self, frame: &CsiFrameView) -> Vec<SkillEvent> {
        self.frame_count += 1;
        let mut out: Vec<SkillEvent> = Vec::new();
        for skill in self.skills.iter_mut() {
            let name = skill.name();
            for &(event_id, value) in skill.on_frame(frame) {
                out.push(SkillEvent {
                    skill: name,
                    event_id,
                    value,
                });
            }
        }
        out
    }

    /// Total frames processed so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Introspection: list every registered skill with its event ids and tier.
    pub fn skills(&self) -> Vec<SkillInfo> {
        let mut out = Vec::with_capacity(self.skills.len());
        for (i, skill) in self.skills.iter().enumerate() {
            out.push(SkillInfo {
                name: skill.name(),
                event_ids: skill.event_ids(),
                medical_experimental: self.medical_flags.get(i).copied().unwrap_or(false),
            });
        }
        out
    }
}

impl Default for EdgePipeline {
    fn default() -> Self {
        Self::new()
    }
}

//! Adapters wiring every runtime skill detector to the uniform [`EdgeSkill`]
//! trait, plus the registration functions consumed by [`EdgePipeline::new`].
//!
//! [`EdgePipeline::new`]: crate::pipeline_all::EdgePipeline::new
//! [`EdgeSkill`]: crate::pipeline_all::EdgeSkill
//!
//! # How adapters work
//!
//! Each underlying detector keeps its own bespoke `process_frame`/`on_timer`
//! signature and its owned `events: [(i32,f32); N]` buffer (the ADR-160 M6
//! soundness fix). An adapter holds the detector, implements [`EdgeSkill`], and
//! in `on_frame` simply pulls the needed fields out of [`CsiFrameView`] and
//! forwards the call **unchanged**. The detector returns `&self.events[..n]`;
//! the adapter forwards that borrow directly, so no extra buffer or copy is
//! needed for the common case.
//!
//! Three families need a small owned scratch buffer in the adapter instead of a
//! direct forward, because the underlying entry point does not itself return a
//! `&[(i32,f32)]`:
//! - `gesture` (`-> Option<u8>`), `coherence` (`-> f32`), `adversarial`
//!   (`-> bool`): the adapter synthesizes a single tagged event.
//! - `sig_sparse_recovery` (`process_frame(&mut [f32])`): the adapter copies the
//!   frame amplitudes into an owned scratch slice so the in-place ISTA recovery
//!   never mutates the shared frame, then forwards the borrow.
//! - timer-driven skills (`vital_trend`, `lrn_meta_adapt`, `sig_temporal_compress`,
//!   `tmp_goap_autonomy`, `tmp_pattern_sequence`): their `on_timer()` is driven
//!   once per frame here (a frame *is* the tick at the edge), forwarding the
//!   borrow. `tmp_pattern_sequence` additionally calls its `on_frame(...)`
//!   accumulator first.
//!
//! **No skill's DSP is changed.** Only the call wiring lives here.

#![cfg(feature = "std")]

extern crate std;
use std::boxed::Box;
use std::vec::Vec;

use crate::pipeline_all::{CsiFrameView, EdgeSkill};

// ── Direct-forward adapter macro ─────────────────────────────────────────────
//
// Generates an adapter whose `on_frame` forwards directly to a detector method
// that already returns `&[(i32, f32)]`. `$call` is an expression over `self.0`
// (the detector) and `f` (the `&CsiFrameView`).
macro_rules! fwd_skill {
    ($adapter:ident, $detector:path, $name:literal, $ids:expr, |$d:ident, $f:ident| $call:expr) => {
        pub struct $adapter($detector);
        impl $adapter {
            pub fn new() -> Self {
                Self(<$detector>::new())
            }
        }
        impl EdgeSkill for $adapter {
            fn name(&self) -> &'static str {
                $name
            }
            fn event_ids(&self) -> &'static [i32] {
                &$ids
            }
            fn on_frame(&mut self, $f: &CsiFrameView) -> &[(i32, f32)] {
                let $d = &mut self.0;
                $call
            }
        }
    };
}

// ── Synthesized-event adapter macro ──────────────────────────────────────────
//
// For detectors whose entry point does NOT return `&[(i32, f32)]`. The adapter
// owns a tiny scratch buffer; `$body` (over `self`, `f`, and `self.buf`/`self.n`)
// fills it and the trait returns the filled prefix.
macro_rules! synth_skill {
    ($adapter:ident, $detector:path, $name:literal, $ids:expr, $buf:literal,
     |$s:ident, $f:ident| $body:block) => {
        pub struct $adapter {
            det: $detector,
            buf: [(i32, f32); $buf],
            n: usize,
        }
        impl $adapter {
            pub fn new() -> Self {
                Self {
                    det: <$detector>::new(),
                    buf: [(0, 0.0); $buf],
                    n: 0,
                }
            }
        }
        impl EdgeSkill for $adapter {
            fn name(&self) -> &'static str {
                $name
            }
            fn event_ids(&self) -> &'static [i32] {
                &$ids
            }
            fn on_frame(&mut self, $f: &CsiFrameView) -> &[(i32, f32)] {
                let $s = self;
                $s.n = 0;
                $body
                &$s.buf[..$s.n]
            }
        }
    };
}

use crate::event_types as ev;

// ── Flagship (synthesized) ───────────────────────────────────────────────────

synth_skill!(GestureAdapter, crate::gesture::GestureDetector, "gesture",
    [ev::GESTURE_DETECTED], 1, |s, f| {
        if let Some(id) = s.det.process_frame(f.phases) {
            s.buf[0] = (ev::GESTURE_DETECTED, id as f32);
            s.n = 1;
        }
    });

synth_skill!(CoherenceAdapter, crate::coherence::CoherenceMonitor, "coherence",
    [ev::COHERENCE_SCORE], 1, |s, f| {
        let score = s.det.process_frame(f.phases);
        s.buf[0] = (ev::COHERENCE_SCORE, score);
        s.n = 1;
    });

synth_skill!(AdversarialAdapter, crate::adversarial::AnomalyDetector, "adversarial",
    [ev::ANOMALY_DETECTED], 1, |s, f| {
        if s.det.process_frame(f.phases, f.amplitudes) {
            s.buf[0] = (ev::ANOMALY_DETECTED, 1.0);
            s.n = 1;
        }
    });

// ── sig_sparse_recovery (needs owned mutable amplitude scratch) ───────────────

const SPARSE_SC: usize = 64;
pub struct SparseRecoveryAdapter {
    det: crate::sig_sparse_recovery::SparseRecovery,
    scratch: [f32; SPARSE_SC],
}
impl SparseRecoveryAdapter {
    pub fn new() -> Self {
        Self {
            det: crate::sig_sparse_recovery::SparseRecovery::new(),
            scratch: [0.0; SPARSE_SC],
        }
    }
}
impl EdgeSkill for SparseRecoveryAdapter {
    fn name(&self) -> &'static str {
        "sig_sparse_recovery"
    }
    fn event_ids(&self) -> &'static [i32] {
        &[ev::RECOVERY_COMPLETE, ev::RECOVERY_ERROR, ev::DROPOUT_RATE]
    }
    fn on_frame(&mut self, f: &CsiFrameView) -> &[(i32, f32)] {
        let n = f.amplitudes.len().min(SPARSE_SC);
        self.scratch[..n].copy_from_slice(&f.amplitudes[..n]);
        self.det.process_frame(&mut self.scratch[..n])
    }
}

// ── Standard direct-forward skills (return &[(i32,f32)]) ─────────────────────

fwd_skill!(AisBehavioralAdapter, crate::ais_behavioral_profiler::BehavioralProfiler,
    "ais_behavioral_profiler",
    [ev::BEHAVIOR_ANOMALY, ev::PROFILE_DEVIATION, ev::NOVEL_PATTERN, ev::PROFILE_MATURITY],
    |d, f| d.process_frame(f.presence != 0, f.motion_energy, f.n_persons.max(0) as u8));

fwd_skill!(AisPromptShieldAdapter, crate::ais_prompt_shield::PromptShield,
    "ais_prompt_shield",
    [ev::REPLAY_ATTACK, ev::INJECTION_DETECTED, ev::JAMMING_DETECTED, ev::SIGNAL_INTEGRITY],
    |d, f| d.process_frame(f.phases, f.amplitudes));

fwd_skill!(AutPsychoAdapter, crate::aut_psycho_symbolic::PsychoSymbolicEngine,
    "aut_psycho_symbolic",
    [ev::INFERENCE_RESULT, ev::INFERENCE_CONFIDENCE, ev::RULE_FIRED, ev::CONTRADICTION],
    |d, f| d.process_frame(f.presence as f32, f.motion_energy, f.breathing_bpm,
        f.heartrate_bpm, f.n_persons as f32, 0.0));

fwd_skill!(AutMeshAdapter, crate::aut_self_healing_mesh::SelfHealingMesh,
    "aut_self_healing_mesh",
    [ev::NODE_DEGRADED, ev::MESH_RECONFIGURE, ev::COVERAGE_SCORE, ev::HEALING_COMPLETE],
    |d, f| d.process_frame(f.variances));

fwd_skill!(BldElevatorAdapter, crate::bld_elevator_count::ElevatorCounter,
    "bld_elevator_count",
    [ev::ELEVATOR_COUNT, ev::DOOR_OPEN, ev::DOOR_CLOSE, ev::OVERLOAD_WARNING],
    |d, f| d.process_frame(f.amplitudes, f.phases, f.motion_energy, f.n_persons));

fwd_skill!(BldEnergyAdapter, crate::bld_energy_audit::EnergyAuditor,
    "bld_energy_audit",
    [ev::SCHEDULE_SUMMARY, ev::AFTER_HOURS_ALERT, ev::UTILIZATION_RATE],
    |d, f| d.process_frame(f.presence, f.n_persons));

fwd_skill!(BldHvacAdapter, crate::bld_hvac_presence::HvacPresenceDetector,
    "bld_hvac_presence",
    [ev::HVAC_OCCUPIED, ev::ACTIVITY_LEVEL, ev::DEPARTURE_COUNTDOWN],
    |d, f| d.process_frame(f.presence as f32, f.motion_energy));

fwd_skill!(BldLightingAdapter, crate::bld_lighting_zones::LightingZoneController,
    "bld_lighting_zones",
    [ev::LIGHT_ON, ev::LIGHT_DIM, ev::LIGHT_OFF],
    |d, f| d.process_frame(f.amplitudes, f.motion_energy));

fwd_skill!(BldMeetingAdapter, crate::bld_meeting_room::MeetingRoomTracker,
    "bld_meeting_room",
    [ev::MEETING_START, ev::MEETING_END, ev::PEAK_HEADCOUNT, ev::ROOM_AVAILABLE],
    |d, f| d.process_frame(f.presence, f.n_persons, f.motion_energy));

fwd_skill!(ExoBreathingSyncAdapter, crate::exo_breathing_sync::BreathingSyncDetector,
    "exo_breathing_sync",
    [ev::SYNC_DETECTED, ev::SYNC_PAIR_COUNT, ev::GROUP_COHERENCE, ev::SYNC_LOST],
    |d, f| d.process_frame(f.phases, f.variances, f.breathing_bpm, f.n_persons));

fwd_skill!(ExoEmotionAdapter, crate::exo_emotion_detect::EmotionDetector,
    "exo_emotion_detect",
    [ev::AROUSAL_LEVEL, ev::STRESS_INDEX, ev::CALM_DETECTED, ev::AGITATION_DETECTED],
    |d, f| d.process_frame(f.breathing_bpm, f.heartrate_bpm, f.motion_energy,
        f.phase_mean(), f.variance_mean));

fwd_skill!(ExoDreamAdapter, crate::exo_dream_stage::DreamStageDetector,
    "exo_dream_stage",
    [ev::SLEEP_STAGE, ev::SLEEP_QUALITY, ev::REM_EPISODE, ev::DEEP_SLEEP_RATIO],
    |d, f| d.process_frame(f.breathing_bpm, f.heartrate_bpm, f.motion_energy,
        f.phase_mean(), f.variance_mean, f.presence));

fwd_skill!(ExoGestureLangAdapter, crate::exo_gesture_language::GestureLanguageDetector,
    "exo_gesture_language",
    [ev::LETTER_RECOGNIZED, ev::LETTER_CONFIDENCE, ev::WORD_BOUNDARY, ev::GESTURE_REJECTED],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variance_mean, f.motion_energy, f.presence));

fwd_skill!(ExoGhostAdapter, crate::exo_ghost_hunter::GhostHunterDetector,
    "exo_ghost_hunter",
    [ev::EXO_ANOMALY_DETECTED, ev::EXO_ANOMALY_CLASS, ev::HIDDEN_PRESENCE, ev::ENVIRONMENTAL_DRIFT],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variances, f.presence, f.motion_energy));

fwd_skill!(ExoHappinessAdapter, crate::exo_happiness_score::HappinessScoreDetector,
    "exo_happiness_score",
    [ev::HAPPINESS_SCORE, ev::GAIT_ENERGY, ev::AFFECT_VALENCE, ev::SOCIAL_ENERGY, ev::TRANSIT_DIRECTION],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variances, f.presence,
        f.motion_energy, f.breathing_bpm, f.heartrate_bpm));

fwd_skill!(ExoHyperbolicAdapter, crate::exo_hyperbolic_space::HyperbolicEmbedder,
    "exo_hyperbolic_space",
    [ev::HIERARCHY_LEVEL, ev::HYPERBOLIC_RADIUS, ev::LOCATION_LABEL],
    |d, f| d.process_frame(f.amplitudes));

fwd_skill!(ExoMusicAdapter, crate::exo_music_conductor::MusicConductorDetector,
    "exo_music_conductor",
    [ev::CONDUCTOR_BPM, ev::BEAT_POSITION, ev::DYNAMIC_LEVEL, ev::GESTURE_CUTOFF, ev::GESTURE_FERMATA],
    |d, f| d.process_frame(f.phase_mean(), f.amplitude_mean(), f.motion_energy, f.variance_mean));

fwd_skill!(ExoPlantAdapter, crate::exo_plant_growth::PlantGrowthDetector,
    "exo_plant_growth",
    [ev::GROWTH_RATE, ev::CIRCADIAN_PHASE, ev::WILT_DETECTED, ev::WATERING_EVENT],
    |d, f| d.process_frame(f.amplitudes, f.phases, f.variances, f.presence));

fwd_skill!(ExoRainAdapter, crate::exo_rain_detect::RainDetector,
    "exo_rain_detect",
    [ev::RAIN_ONSET, ev::RAIN_INTENSITY, ev::RAIN_CESSATION],
    |d, f| d.process_frame(f.phases, f.variances, f.amplitudes, f.presence));

fwd_skill!(ExoTimeCrystalAdapter, crate::exo_time_crystal::TimeCrystalDetector,
    "exo_time_crystal",
    [ev::CRYSTAL_DETECTED, ev::CRYSTAL_STABILITY, ev::COORDINATION_INDEX],
    |d, f| d.process_frame(f.motion_energy));

fwd_skill!(IndCleanRoomAdapter, crate::ind_clean_room::CleanRoomMonitor,
    "ind_clean_room",
    [ev::OCCUPANCY_COUNT, ev::OCCUPANCY_VIOLATION, ev::TURBULENT_MOTION, ev::COMPLIANCE_REPORT],
    |d, f| d.process_frame(f.n_persons, f.presence, f.motion_energy));

fwd_skill!(IndConfinedAdapter, crate::ind_confined_space::ConfinedSpaceMonitor,
    "ind_confined_space",
    [ev::WORKER_ENTRY, ev::WORKER_EXIT, ev::BREATHING_OK, ev::EXTRACTION_ALERT, ev::IMMOBILE_ALERT],
    |d, f| d.process_frame(f.presence, f.breathing_bpm, f.motion_energy, f.variance_mean));

fwd_skill!(IndForkliftAdapter, crate::ind_forklift_proximity::ForkliftProximityDetector,
    "ind_forklift_proximity",
    [ev::PROXIMITY_WARNING, ev::VEHICLE_DETECTED, ev::HUMAN_NEAR_VEHICLE],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variances, f.motion_energy, f.presence, f.n_persons));

fwd_skill!(IndLivestockAdapter, crate::ind_livestock_monitor::LivestockMonitor,
    "ind_livestock_monitor",
    [ev::ANIMAL_PRESENT, ev::ABNORMAL_STILLNESS, ev::LABORED_BREATHING, ev::ESCAPE_ALERT],
    |d, f| d.process_frame(f.presence, f.breathing_bpm, f.motion_energy, f.variance_mean));

fwd_skill!(IndVibrationAdapter, crate::ind_structural_vibration::StructuralVibrationMonitor,
    "ind_structural_vibration",
    [ev::SEISMIC_DETECTED, ev::MECHANICAL_RESONANCE, ev::STRUCTURAL_DRIFT, ev::VIBRATION_SPECTRUM],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variances, f.presence));

fwd_skill!(IntrusionAdapter, crate::intrusion::IntrusionDetector,
    "intrusion",
    [ev::INTRUSION_ALERT, ev::INTRUSION_ZONE, 202],
    |d, f| d.process_frame(f.phases, f.amplitudes));

fwd_skill!(LrnAttractorAdapter, crate::lrn_anomaly_attractor::AttractorDetector,
    "lrn_anomaly_attractor",
    [ev::ATTRACTOR_TYPE, ev::LYAPUNOV_EXPONENT, ev::BASIN_DEPARTURE, ev::LEARNING_COMPLETE],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.motion_energy));

fwd_skill!(LrnDtwAdapter, crate::lrn_dtw_gesture_learn::GestureLearner,
    "lrn_dtw_gesture_learn",
    [ev::GESTURE_LEARNED, ev::GESTURE_MATCHED, ev::LRN_MATCH_DISTANCE, ev::TEMPLATE_COUNT],
    |d, f| d.process_frame(f.phases, f.motion_energy));

fwd_skill!(LrnEwcAdapter, crate::lrn_ewc_lifelong::EwcLifelong,
    "lrn_ewc_lifelong",
    [ev::KNOWLEDGE_RETAINED, ev::NEW_TASK_LEARNED, ev::FISHER_UPDATE, ev::FORGETTING_RISK],
    |d, f| d.process_frame(f.variances, f.presence));

fwd_skill!(OccupancyAdapter, crate::occupancy::OccupancyDetector,
    "occupancy",
    [ev::ZONE_OCCUPIED, ev::ZONE_COUNT, ev::ZONE_TRANSITION],
    |d, f| d.process_frame(f.phases, f.amplitudes));

fwd_skill!(QntInterferenceAdapter, crate::qnt_interference_search::InterferenceSearch,
    "qnt_interference_search",
    [ev::HYPOTHESIS_WINNER, ev::HYPOTHESIS_AMPLITUDE, ev::SEARCH_ITERATIONS],
    |d, f| d.process_frame(f.presence, f.motion_energy, f.n_persons));

fwd_skill!(QntCoherenceAdapter, crate::qnt_quantum_coherence::QuantumCoherenceMonitor,
    "qnt_quantum_coherence",
    [ev::ENTANGLEMENT_ENTROPY, ev::DECOHERENCE_EVENT, ev::BLOCH_DRIFT],
    |d, f| d.process_frame(f.phases));

fwd_skill!(RetFlowAdapter, crate::ret_customer_flow::CustomerFlowTracker,
    "ret_customer_flow",
    [ev::INGRESS, ev::EGRESS, ev::NET_OCCUPANCY, ev::HOURLY_TRAFFIC],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variance_mean, f.motion_energy));

fwd_skill!(RetDwellAdapter, crate::ret_dwell_heatmap::DwellHeatmapTracker,
    "ret_dwell_heatmap",
    [ev::DWELL_ZONE_UPDATE, ev::HOT_ZONE, ev::COLD_ZONE, ev::SESSION_SUMMARY],
    |d, f| d.process_frame(f.presence, f.variances, f.motion_energy, f.n_persons));

fwd_skill!(RetQueueAdapter, crate::ret_queue_length::QueueLengthEstimator,
    "ret_queue_length",
    [ev::QUEUE_LENGTH, ev::WAIT_TIME_ESTIMATE, ev::SERVICE_RATE, ev::QUEUE_ALERT],
    |d, f| d.process_frame(f.presence, f.n_persons, f.variance_mean, f.motion_energy));

fwd_skill!(RetShelfAdapter, crate::ret_shelf_engagement::ShelfEngagementDetector,
    "ret_shelf_engagement",
    [ev::SHELF_BROWSE, ev::SHELF_CONSIDER, ev::SHELF_ENGAGE, ev::REACH_DETECTED],
    |d, f| d.process_frame(f.presence, f.motion_energy, f.variance_mean, f.phases));

fwd_skill!(RetTableAdapter, crate::ret_table_turnover::TableTurnoverTracker,
    "ret_table_turnover",
    [ev::TABLE_SEATED, ev::TABLE_VACATED, ev::TABLE_AVAILABLE, ev::TURNOVER_RATE],
    |d, f| d.process_frame(f.presence, f.motion_energy, f.n_persons));

fwd_skill!(SecLoiteringAdapter, crate::sec_loitering::LoiteringDetector,
    "sec_loitering",
    [ev::LOITERING_START, ev::LOITERING_ONGOING, ev::LOITERING_END],
    |d, f| d.process_frame(f.presence, f.motion_energy));

fwd_skill!(SecPanicAdapter, crate::sec_panic_motion::PanicMotionDetector,
    "sec_panic_motion",
    [ev::PANIC_DETECTED, ev::STRUGGLE_PATTERN, ev::FLEEING_DETECTED],
    |d, f| d.process_frame(f.motion_energy, f.variance_mean, f.phase_mean(), f.presence));

fwd_skill!(SecPerimeterAdapter, crate::sec_perimeter_breach::PerimeterBreachDetector,
    "sec_perimeter_breach",
    [ev::PERIMETER_BREACH, ev::APPROACH_DETECTED, ev::DEPARTURE_DETECTED, ev::SEC_ZONE_TRANSITION],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variances, f.motion_energy));

fwd_skill!(SecTailgateAdapter, crate::sec_tailgating::TailgateDetector,
    "sec_tailgating",
    [ev::TAILGATE_DETECTED, ev::SINGLE_PASSAGE, ev::MULTI_PASSAGE],
    |d, f| d.process_frame(f.motion_energy, f.presence, f.n_persons, f.variance_mean));

fwd_skill!(SecWeaponAdapter, crate::sec_weapon_detect::WeaponDetector,
    "sec_weapon_detect",
    [ev::METAL_ANOMALY, ev::HIGH_METAL_REFLECTIVITY, ev::CALIBRATION_NEEDED],
    |d, f| d.process_frame(f.phases, f.amplitudes, f.variances, f.motion_energy, f.presence));

fwd_skill!(SigCoherenceGateAdapter, crate::sig_coherence_gate::CoherenceGate,
    "sig_coherence_gate",
    [ev::GATE_DECISION, ev::SIG_COHERENCE_SCORE, ev::RECALIBRATE_NEEDED],
    |d, f| d.process_frame(f.phases));

fwd_skill!(SigFlashAttnAdapter, crate::sig_flash_attention::FlashAttention,
    "sig_flash_attention",
    [ev::ATTENTION_PEAK_SC, ev::ATTENTION_SPREAD, ev::SPATIAL_FOCUS_ZONE],
    |d, f| d.process_frame(f.phases, f.amplitudes));

fwd_skill!(SigMincutAdapter, crate::sig_mincut_person_match::PersonMatcher,
    "sig_mincut_person_match",
    [ev::PERSON_ID_ASSIGNED, ev::PERSON_ID_SWAP, ev::MATCH_CONFIDENCE],
    |d, f| d.process_frame(f.amplitudes, f.variances, f.n_persons.max(0) as usize));

fwd_skill!(SigTransportAdapter, crate::sig_optimal_transport::OptimalTransportDetector,
    "sig_optimal_transport",
    [ev::WASSERSTEIN_DISTANCE, ev::DISTRIBUTION_SHIFT, ev::SUBTLE_MOTION],
    |d, f| d.process_frame(f.amplitudes));

fwd_skill!(SptHnswAdapter, crate::spt_micro_hnsw::MicroHnsw,
    "spt_micro_hnsw",
    [ev::NEAREST_MATCH_ID, ev::HNSW_MATCH_DISTANCE, ev::CLASSIFICATION, ev::LIBRARY_SIZE],
    |d, f| d.process_frame(f.variances));

fwd_skill!(SptPagerankAdapter, crate::spt_pagerank_influence::PageRankInfluence,
    "spt_pagerank_influence",
    [ev::DOMINANT_PERSON, ev::INFLUENCE_SCORE, ev::INFLUENCE_CHANGE],
    |d, f| d.process_frame(f.phases, f.n_persons.max(0) as usize));

fwd_skill!(SptSpikingAdapter, crate::spt_spiking_tracker::SpikingTracker,
    "spt_spiking_tracker",
    [ev::TRACK_UPDATE, ev::TRACK_VELOCITY, ev::SPIKE_RATE, ev::TRACK_LOST],
    |d, f| d.process_frame(f.phases, f.prev_phases));

fwd_skill!(TmpLogicGuardAdapter, crate::tmp_temporal_logic_guard::TemporalLogicGuard,
    "tmp_temporal_logic_guard",
    [ev::LTL_VIOLATION, ev::LTL_SATISFACTION, ev::COUNTEREXAMPLE],
    |d, f| {
        let input = crate::tmp_temporal_logic_guard::FrameInput {
            presence: f.presence,
            n_persons: f.n_persons,
            motion_energy: f.motion_energy,
            coherence: f.coherence,
            breathing_bpm: f.breathing_bpm,
            heartrate_bpm: f.heartrate_bpm,
            fall_alert: false,
            intrusion_alert: false,
            person_id_active: f.n_persons > 0,
            vital_signs_active: f.breathing_bpm > 0.0,
            seizure_detected: false,
            normal_gait: true,
        };
        d.on_frame(&input)
    });

// ── Timer-driven skills (driven once per frame) ──────────────────────────────

fwd_skill!(VitalTrendAdapter, crate::vital_trend::VitalTrendAnalyzer,
    "vital_trend",
    // 101-105 = brady/tachypnea, brady/tachycardia, apnea; 110/111 = breathing/heartrate
    // moving averages (module-local EVENT_BREATHING_AVG / EVENT_HEARTRATE_AVG).
    [ev::BRADYPNEA, ev::TACHYPNEA, ev::BRADYCARDIA, ev::TACHYCARDIA, ev::APNEA, 110, 111],
    |d, f| d.on_timer(f.breathing_bpm, f.heartrate_bpm));

fwd_skill!(LrnMetaAdapter, crate::lrn_meta_adapt::MetaAdapter,
    "lrn_meta_adapt",
    [ev::PARAM_ADJUSTED, ev::ADAPTATION_SCORE, ev::ROLLBACK_TRIGGERED, ev::META_LEVEL],
    |d, _f| d.on_timer());

fwd_skill!(SigTemporalCompressAdapter, crate::sig_temporal_compress::TemporalCompressor,
    "sig_temporal_compress",
    [ev::COMPRESSION_RATIO, ev::TIER_TRANSITION, ev::HISTORY_DEPTH_HOURS],
    |d, _f| d.on_timer());

fwd_skill!(TmpGoapAdapter, crate::tmp_goap_autonomy::GoapPlanner,
    "tmp_goap_autonomy",
    [ev::GOAL_SELECTED, ev::MODULE_ACTIVATED, ev::MODULE_DEACTIVATED, ev::PLAN_COST],
    |d, _f| d.on_timer());

// tmp_pattern_sequence: accumulate via on_frame, then drive on_timer per frame.
pub struct TmpPatternAdapter(crate::tmp_pattern_sequence::PatternSequenceAnalyzer);
impl TmpPatternAdapter {
    pub fn new() -> Self {
        Self(crate::tmp_pattern_sequence::PatternSequenceAnalyzer::new())
    }
}
impl EdgeSkill for TmpPatternAdapter {
    fn name(&self) -> &'static str {
        "tmp_pattern_sequence"
    }
    fn event_ids(&self) -> &'static [i32] {
        &[ev::PATTERN_DETECTED, ev::PATTERN_CONFIDENCE, ev::ROUTINE_DEVIATION, ev::PREDICTION_NEXT]
    }
    fn on_frame(&mut self, f: &CsiFrameView) -> &[(i32, f32)] {
        self.0.on_frame(f.presence, f.motion_energy, f.n_persons);
        self.0.on_timer()
    }
}

// ── Medical tier (gated) ─────────────────────────────────────────────────────

#[cfg(feature = "medical-experimental")]
mod medical {
    use super::*;

    // Medical event ids verified against each module's local consts (100-199 block).
    fwd_skill!(MedCardiacAdapter, crate::med_cardiac_arrhythmia::CardiacArrhythmiaDetector,
        "med_cardiac_arrhythmia",
        [110, 111, 112, 113],
        |d, f| d.process_frame(f.heartrate_bpm, f.phase_mean()));

    fwd_skill!(MedGaitAdapter, crate::med_gait_analysis::GaitAnalyzer,
        "med_gait_analysis",
        [130, 131, 132, 133, 134],
        |d, f| d.process_frame(f.phase_mean(), f.amplitude_mean(), f.variance_mean, f.motion_energy));

    fwd_skill!(MedRespiratoryAdapter, crate::med_respiratory_distress::RespiratoryDistressDetector,
        "med_respiratory_distress",
        [120, 121, 122, 123],
        |d, f| d.process_frame(f.breathing_bpm, f.phase_mean(), f.variance_mean));

    fwd_skill!(MedSeizureAdapter, crate::med_seizure_detect::SeizureDetector,
        "med_seizure_detect",
        [140, 141, 142, 143],
        |d, f| d.process_frame(f.phase_mean(), f.amplitude_mean(), f.motion_energy, f.presence));

    fwd_skill!(MedApneaAdapter, crate::med_sleep_apnea::SleepApneaDetector,
        "med_sleep_apnea",
        [100, 101, 102],
        |d, f| d.process_frame(f.breathing_bpm, f.presence, f.variance_mean));

    pub fn register(skills: &mut Vec<Box<dyn EdgeSkill>>, med: &mut Vec<bool>) {
        macro_rules! push {
            ($a:ty) => {{
                skills.push(Box::new(<$a>::new()));
                med.push(true);
            }};
        }
        push!(MedSeizureAdapter);
        push!(MedCardiacAdapter);
        push!(MedRespiratoryAdapter);
        push!(MedApneaAdapter);
        push!(MedGaitAdapter);
    }
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register every default-tier (non-medical) skill.
pub fn register_default(skills: &mut Vec<Box<dyn EdgeSkill>>, med: &mut Vec<bool>) {
    macro_rules! push {
        ($a:ty) => {{
            skills.push(Box::new(<$a>::new()));
            med.push(false);
        }};
    }

    // Flagship + synthesized
    push!(GestureAdapter);
    push!(CoherenceAdapter);
    push!(AdversarialAdapter);
    push!(OccupancyAdapter);
    push!(IntrusionAdapter);
    push!(VitalTrendAdapter);

    // Security
    push!(SecPerimeterAdapter);
    push!(SecWeaponAdapter);
    push!(SecTailgateAdapter);
    push!(SecLoiteringAdapter);
    push!(SecPanicAdapter);

    // Smart building
    push!(BldHvacAdapter);
    push!(BldLightingAdapter);
    push!(BldElevatorAdapter);
    push!(BldMeetingAdapter);
    push!(BldEnergyAdapter);

    // Retail
    push!(RetQueueAdapter);
    push!(RetDwellAdapter);
    push!(RetFlowAdapter);
    push!(RetTableAdapter);
    push!(RetShelfAdapter);

    // Industrial
    push!(IndForkliftAdapter);
    push!(IndConfinedAdapter);
    push!(IndCleanRoomAdapter);
    push!(IndLivestockAdapter);
    push!(IndVibrationAdapter);

    // Exotic / research
    push!(ExoTimeCrystalAdapter);
    push!(ExoHyperbolicAdapter);
    push!(ExoDreamAdapter);
    push!(ExoEmotionAdapter);
    push!(ExoGestureLangAdapter);
    push!(ExoMusicAdapter);
    push!(ExoPlantAdapter);
    push!(ExoGhostAdapter);
    push!(ExoRainAdapter);
    push!(ExoBreathingSyncAdapter);
    push!(ExoHappinessAdapter);

    // Signal intelligence
    push!(SigCoherenceGateAdapter);
    push!(SigFlashAttnAdapter);
    push!(SigTemporalCompressAdapter);
    push!(SparseRecoveryAdapter);
    push!(SigMincutAdapter);
    push!(SigTransportAdapter);

    // Adaptive learning
    push!(LrnDtwAdapter);
    push!(LrnAttractorAdapter);
    push!(LrnMetaAdapter);
    push!(LrnEwcAdapter);

    // Spatial reasoning
    push!(SptPagerankAdapter);
    push!(SptHnswAdapter);
    push!(SptSpikingAdapter);

    // Temporal analysis
    push!(TmpPatternAdapter);
    push!(TmpLogicGuardAdapter);
    push!(TmpGoapAdapter);

    // AI security
    push!(AisPromptShieldAdapter);
    push!(AisBehavioralAdapter);

    // Quantum-inspired
    push!(QntCoherenceAdapter);
    push!(QntInterferenceAdapter);

    // Autonomous systems
    push!(AutPsychoAdapter);
    push!(AutMeshAdapter);

    let _ = (skills.len(), med.len());
}

/// Register the gated `medical-experimental` tier (5 `med_*` skills).
#[cfg(feature = "medical-experimental")]
pub fn register_medical(skills: &mut Vec<Box<dyn EdgeSkill>>, med: &mut Vec<bool>) {
    medical::register(skills, med);
}

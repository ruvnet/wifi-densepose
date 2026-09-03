//! Bridge between sensing-server frame data and signal crate FieldModel
//! for eigenvalue-based person counting.
//!
//! The FieldModel decomposes CSI observations into environmental drift and
//! body perturbation via SVD eigenmodes. When calibrated, perturbation energy
//! provides a physics-grounded occupancy estimate that supplements the
//! score-based heuristic in `score_to_person_count`.

use std::collections::VecDeque;
use std::sync::LazyLock;
use wifi_densepose_signal::hardware_norm::HardwareNormalizer;
use wifi_densepose_signal::ruvsense::field_model::{
    CalibrationStatus, FieldModel, FieldModelConfig,
};

use super::score_to_person_count;

/// Length-only canonicalizer for calibration frames (issue #1170 pattern,
/// shared with `multistatic_bridge`). Raw ESP32 amplitudes arrive at the
/// hardware's native width (HT20 ≈ 64, HT40 ≈ 128/192); the FieldModel is
/// configured for the canonical 56-tone grid, and `feed_calibration` rejects
/// any other width with `DimensionMismatch`. Resampling here (default 56)
/// lets real HT40 nodes actually calibrate instead of silently feeding nothing.
static CALIB_NORMALIZER: LazyLock<HardwareNormalizer> = LazyLock::new(HardwareNormalizer::new);

/// Number of recent frames to feed into perturbation extraction.
const OCCUPANCY_WINDOW: usize = 50;

/// Perturbation energy threshold for detecting a second person.
const ENERGY_THRESH_2: f64 = 12.0;
/// Perturbation energy threshold for detecting a third person.
const ENERGY_THRESH_3: f64 = 25.0;

/// Maximum occupancy a single ESP32 link can plausibly resolve (#894).
/// The score heuristic (`score_to_person_count`) and the perturbation-energy
/// fallback below both cap here; the eigenvalue path is bounded to match,
/// rather than leaking its internal `min(10)` ceiling on noisy / under-
/// calibrated CSI (the "10 persons reported when 1 present" symptom).
/// Resolving more than this from one link's subcarrier covariance is not
/// reliable — genuine higher counts come from the multistatic fusion path.
const MAX_SINGLE_LINK_OCCUPANCY: usize = 3;

/// Provenance for a count produced by the calibrated field model.  This is
/// intentionally narrower than [`occupancy_or_fallback`]: callers that attach
/// calibration evidence to a wire frame must never silently substitute the
/// score heuristic when the calibrated path cannot evaluate the observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibratedOccupancyMethod {
    Eigenvalue,
    PerturbationEnergy,
    NullNormalized,
}

impl CalibratedOccupancyMethod {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Eigenvalue => "field_model_eigenvalue_v1",
            Self::PerturbationEnergy => "field_model_perturbation_energy_v1",
            Self::NullNormalized => "field_model_null_normalized_v2",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedOccupancy {
    pub person_count: usize,
    pub method: CalibratedOccupancyMethod,
    pub residual_evidence: Option<ResidualEnergyEvidence>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinkResidualEnergyEvidence {
    pub energy: f64,
    pub null_median: f64,
    pub null_p95: f64,
    pub null_p99: f64,
    pub normalized_energy: f64,
    pub decision_threshold: f64,
    pub above_threshold: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidualEnergyEvidence {
    pub links: Vec<LinkResidualEnergyEvidence>,
    pub aggregate_mean_energy: f64,
    pub aggregate_mean_normalized_energy: f64,
    pub nodes_above_threshold: usize,
    pub node_quorum: usize,
    pub hysteresis_present: bool,
    pub hysteresis_candidate_frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresenceDetectorConfig {
    pub node_quorum: usize,
    pub enter_frames: usize,
    pub exit_frames: usize,
}

impl Default for PresenceDetectorConfig {
    fn default() -> Self {
        Self {
            node_quorum: 2,
            enter_frames: 3,
            exit_frames: 10,
        }
    }
}

impl PresenceDetectorConfig {
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            node_quorum: bounded_usize_env(
                "WDP_CALIBRATED_PRESENCE_NODE_QUORUM",
                defaults.node_quorum,
                16,
            ),
            enter_frames: bounded_usize_env(
                "WDP_CALIBRATED_PRESENCE_ENTER_FRAMES",
                defaults.enter_frames,
                10_000,
            ),
            exit_frames: bounded_usize_env(
                "WDP_CALIBRATED_PRESENCE_EXIT_FRAMES",
                defaults.exit_frames,
                10_000,
            ),
        }
    }
}

fn bounded_usize_env(name: &str, default: usize, maximum: usize) -> usize {
    bounded_usize_value(std::env::var(name).ok().as_deref(), default, maximum)
}

fn bounded_usize_value(value: Option<&str>, default: usize, maximum: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=maximum).contains(value))
        .unwrap_or(default)
}

#[derive(Debug, Clone)]
pub struct CalibratedPresenceDetector {
    config: PresenceDetectorConfig,
    present: bool,
    candidate_frames: usize,
}

impl CalibratedPresenceDetector {
    pub fn new(mut config: PresenceDetectorConfig) -> Self {
        config.node_quorum = config.node_quorum.max(1);
        config.enter_frames = config.enter_frames.max(1);
        config.exit_frames = config.exit_frames.max(1);
        Self {
            config,
            present: false,
            candidate_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.present = false;
        self.candidate_frames = 0;
    }
}

/// Create a FieldModelConfig for single-link mode (one ESP32 node = one link).
/// This avoids the DimensionMismatch error when feeding single-frame observations.
pub fn single_link_config() -> FieldModelConfig {
    FieldModelConfig {
        n_links: 1,
        ..FieldModelConfig::default()
    }
}

/// Create a field model whose stable link ordering is the bound calibration
/// node ordering. A bound multi-node room must never be represented by a
/// single-link model fed sequential samples from unrelated radios.
pub fn multi_link_config(n_links: usize) -> FieldModelConfig {
    FieldModelConfig {
        n_links,
        ..FieldModelConfig::default()
    }
}

/// Feed one complete, already-canonicalized multi-link room observation.
/// Returns true only when the field model accepted the cohort.
pub fn maybe_feed_calibration_observation(
    field: &mut FieldModel,
    observations: &[Vec<f64>],
) -> bool {
    if !matches!(
        field.status(),
        CalibrationStatus::Uncalibrated | CalibrationStatus::Collecting
    ) {
        return false;
    }
    match field.feed_calibration(observations) {
        Ok(()) => true,
        Err(error) => {
            tracing::debug!("FieldModel calibration cohort feed: {error}");
            false
        }
    }
}

/// Estimate occupancy using the FieldModel when calibrated, falling back
/// to the score-based heuristic otherwise.
///
/// Prefers `estimate_occupancy()` (eigenvalue-based) when the model is
/// calibrated and enough frames are available. Falls back to perturbation
/// energy thresholds, then to the score heuristic.
pub fn occupancy_or_fallback(
    field: &FieldModel,
    frame_history: &VecDeque<Vec<f64>>,
    smoothed_score: f64,
    prev_count: usize,
) -> usize {
    match field.status() {
        CalibrationStatus::Fresh | CalibrationStatus::Stale => {
            let frames: Vec<Vec<f64>> = frame_history
                .iter()
                .rev()
                .take(OCCUPANCY_WINDOW)
                .cloned()
                .collect();

            if frames.is_empty() {
                return score_to_person_count(smoothed_score, prev_count);
            }

            // Try eigenvalue-based occupancy first (best accuracy). Bound it to
            // the same single-link maximum the sibling estimators use — the
            // perturbation fallback below and score_to_person_count both cap at
            // MAX_SINGLE_LINK_OCCUPANCY. Without this, estimate_occupancy's
            // internal min(10) ceiling leaks up to 10 persons on noisy / under-
            // calibrated CSI (#894), while every other path on the same data
            // would report ≤3.
            if let Ok(count) = field.estimate_occupancy(&frames) {
                return count.min(MAX_SINGLE_LINK_OCCUPANCY);
            } // else fall through to perturbation energy

            // Fallback: perturbation energy thresholds.
            // FieldModel expects [n_links][n_subcarriers] — we use n_links=1.
            let observation = vec![frames[0].clone()];
            match field.extract_perturbation(&observation) {
                Ok(perturbation) => {
                    if perturbation.total_energy > ENERGY_THRESH_3 {
                        3
                    } else if perturbation.total_energy > ENERGY_THRESH_2 {
                        2
                    } else if perturbation.total_energy > 1.0 {
                        1
                    } else {
                        0
                    }
                }
                Err(_) => score_to_person_count(smoothed_score, prev_count),
            }
        }
        _ => score_to_person_count(smoothed_score, prev_count),
    }
}

/// Produce a fail-closed occupancy result from a *fresh* calibrated model.
///
/// Unlike the legacy display helper above, this function returns `None` when
/// the model is unavailable/stale, the observation is empty, or both
/// calibrated estimators fail.  Raw hardware histories are normalized onto
/// the same canonical 56-tone grid used during calibration before inference.
pub fn calibrated_occupancy(
    field: &FieldModel,
    frame_history: &VecDeque<Vec<f64>>,
    observed_at_us: u64,
) -> Option<CalibratedOccupancy> {
    if field.check_freshness(observed_at_us) != CalibrationStatus::Fresh {
        return None;
    }

    let frames: Vec<Vec<f64>> = frame_history
        .iter()
        .rev()
        .take(OCCUPANCY_WINDOW)
        .map(|frame| CALIB_NORMALIZER.resample_to_canonical(frame))
        .collect();
    if frames.is_empty() {
        return None;
    }

    if let Ok(person_count) = field.estimate_occupancy(&frames) {
        return Some(CalibratedOccupancy {
            person_count: person_count.min(MAX_SINGLE_LINK_OCCUPANCY),
            method: CalibratedOccupancyMethod::Eigenvalue,
            residual_evidence: None,
        });
    }

    let observation = vec![frames[0].clone()];
    let perturbation = field.extract_perturbation(&observation).ok()?;
    let person_count = if perturbation.total_energy > ENERGY_THRESH_3 {
        3
    } else if perturbation.total_energy > ENERGY_THRESH_2 {
        2
    } else if perturbation.total_energy > 1.0 {
        1
    } else {
        0
    };
    Some(CalibratedOccupancy {
        person_count,
        method: CalibratedOccupancyMethod::PerturbationEnergy,
        residual_evidence: None,
    })
}

/// Produce binary calibrated presence from a coherent multi-link observation.
/// Person counting from a temporal single-link eigenvalue window is not valid
/// for a spatial cohort, so this path deliberately reports only 0/1 from mean
/// null-space residual energy.
pub fn calibrated_multilink_occupancy(
    field: &FieldModel,
    observations: &[Vec<f64>],
    observed_at_us: u64,
    detector: &mut CalibratedPresenceDetector,
    advance_hysteresis: bool,
) -> Option<CalibratedOccupancy> {
    if field.check_freshness(observed_at_us) != CalibrationStatus::Fresh || observations.is_empty()
    {
        return None;
    }
    let perturbation = field.extract_perturbation(observations).ok()?;
    let learned = &field.modes()?.null_residual_energy;
    if learned.len() != perturbation.energies.len()
        || learned.iter().any(|item| item.sample_count == 0)
    {
        return None;
    }
    let quorum = detector.config.node_quorum.min(learned.len()).max(1);
    let mut nodes_above_threshold = 0;
    let links = perturbation
        .energies
        .iter()
        .zip(learned.iter())
        .map(|(&energy, null)| {
            let threshold = if detector.present { null.p95 } else { null.p99 };
            let threshold = threshold.max(1e-9);
            let above_threshold = energy > threshold;
            nodes_above_threshold += usize::from(above_threshold);
            LinkResidualEnergyEvidence {
                energy,
                null_median: null.median,
                null_p95: null.p95,
                null_p99: null.p99,
                normalized_energy: energy / null.p99.max(1e-9),
                decision_threshold: threshold,
                above_threshold,
            }
        })
        .collect::<Vec<_>>();
    let candidate_present = nodes_above_threshold >= quorum;
    if advance_hysteresis {
        if candidate_present == detector.present {
            detector.candidate_frames = 0;
        } else {
            detector.candidate_frames += 1;
            let required = if candidate_present {
                detector.config.enter_frames
            } else {
                detector.config.exit_frames
            };
            if detector.candidate_frames >= required {
                detector.present = candidate_present;
                detector.candidate_frames = 0;
            }
        }
    }
    let mean_energy = perturbation.total_energy / observations.len() as f64;
    let mean_normalized =
        links.iter().map(|link| link.normalized_energy).sum::<f64>() / links.len() as f64;
    Some(CalibratedOccupancy {
        person_count: usize::from(detector.present),
        method: CalibratedOccupancyMethod::NullNormalized,
        residual_evidence: Some(ResidualEnergyEvidence {
            links,
            aggregate_mean_energy: mean_energy,
            aggregate_mean_normalized_energy: mean_normalized,
            nodes_above_threshold,
            node_quorum: quorum,
            hysteresis_present: detector.present,
            hysteresis_candidate_frames: detector.candidate_frames,
        }),
    })
}

/// Feed the latest frame to the FieldModel during calibration collection.
///
/// Acts while the model is `Uncalibrated` or `Collecting`. The first fed frame
/// flips a freshly-started (`Uncalibrated`) model to `Collecting` inside
/// `feed_calibration`; without accepting the `Uncalibrated` state here the two
/// gates deadlock and the frame count never leaves 0 (calibration/start yields
/// an `Uncalibrated` model that nothing would ever advance). Wraps the latest
/// frame as a single-link observation (n_links=1) and feeds it.
pub fn maybe_feed_calibration(field: &mut FieldModel, frame_history: &VecDeque<Vec<f64>>) {
    if !matches!(
        field.status(),
        CalibrationStatus::Uncalibrated | CalibrationStatus::Collecting
    ) {
        return;
    }
    if let Some(latest) = frame_history.back() {
        // Resample the raw amplitude vector onto the FieldModel's canonical
        // 56-tone grid before feeding. Real HT40 nodes stream 128-wide frames;
        // feeding those raw made every `feed_calibration` fail DimensionMismatch
        // (swallowed at debug level), pinning frame_count at 0 even after the
        // status-gate deadlock was fixed. Single-link observation: [1][56].
        let canonical = CALIB_NORMALIZER.resample_to_canonical(latest);
        let observations = vec![canonical];
        if let Err(e) = field.feed_calibration(&observations) {
            tracing::debug!("FieldModel calibration feed: {e}");
        }
    }
}

/// Parse node positions from a semicolon-delimited string.
///
/// Format: `"x,y,z;x,y,z;..."` where each coordinate is an `f32`.
/// Malformed entries are skipped with a warning log.
pub fn parse_node_positions(input: &str) -> Vec<[f32; 3]> {
    if input.is_empty() {
        return Vec::new();
    }
    input
        .split(';')
        .enumerate()
        .filter_map(|(idx, triplet)| {
            let parts: Vec<&str> = triplet.split(',').collect();
            if parts.len() != 3 {
                tracing::warn!(
                    "Skipping malformed node position entry {idx}: '{triplet}' (expected x,y,z)"
                );
                return None;
            }
            match (
                parts[0].parse::<f32>(),
                parts[1].parse::<f32>(),
                parts[2].parse::<f32>(),
            ) {
                (Ok(x), Ok(y), Ok(z)) => Some([x, y, z]),
                _ => {
                    tracing::warn!("Skipping unparseable node position entry {idx}: '{triplet}'");
                    None
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_test_model() -> FieldModel {
        let config = FieldModelConfig {
            n_links: 1,
            n_subcarriers: 56,
            n_modes: 3,
            min_calibration_frames: 10,
            baseline_expiry_s: 2.0,
        };
        let mut field = FieldModel::new(config).expect("field model");
        for index in 0..20 {
            let phase = index as f64 * 0.1;
            let observation = vec![(0..56)
                .map(|subcarrier| {
                    1.0 + subcarrier as f64 * 0.01 + (phase + subcarrier as f64 * 0.03).sin() * 0.01
                })
                .collect()];
            field
                .feed_calibration(&observation)
                .expect("calibration frame");
        }
        field
            .finalize_calibration(1_000_000, 7)
            .expect("finalize calibration");
        field
    }

    #[test]
    fn test_parse_node_positions() {
        let positions = parse_node_positions("0,0,1.5;3,0,1.5;1.5,3,1.5");
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], [0.0, 0.0, 1.5]);
        assert_eq!(positions[1], [3.0, 0.0, 1.5]);
        assert_eq!(positions[2], [1.5, 3.0, 1.5]);
    }

    #[test]
    fn test_parse_node_positions_empty() {
        let positions = parse_node_positions("");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_parse_node_positions_invalid() {
        let positions = parse_node_positions("abc;1,2,3");
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_node_positions_partial_triplet() {
        let positions = parse_node_positions("1,2;3,4,5");
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], [3.0, 4.0, 5.0]);
    }

    #[test]
    fn detector_configuration_rejects_zero_invalid_and_excessive_values() {
        assert_eq!(bounded_usize_value(Some("3"), 2, 16), 3);
        assert_eq!(bounded_usize_value(Some("0"), 2, 16), 2);
        assert_eq!(bounded_usize_value(Some("17"), 2, 16), 2);
        assert_eq!(bounded_usize_value(Some("nope"), 2, 16), 2);
        assert_eq!(bounded_usize_value(None, 2, 16), 2);
    }

    /// Regression: a freshly-started (`Uncalibrated`) field model must begin
    /// collecting once frames arrive. Before the fix, `maybe_feed_calibration`
    /// only fed while already `Collecting`, but only `feed_calibration` sets
    /// `Collecting` — so the first frame was never fed and the count stayed 0.
    #[test]
    fn maybe_feed_calibration_advances_uncalibrated_to_collecting() {
        let mut field = FieldModel::new(single_link_config()).expect("field model");
        assert_eq!(field.status(), CalibrationStatus::Uncalibrated);
        assert_eq!(field.calibration_frame_count(), 0);

        // n_subcarriers defaults to 56; one single-link frame of that width.
        let frame = vec![0.5_f64; 56];
        let mut history: VecDeque<Vec<f64>> = VecDeque::new();
        history.push_back(frame);

        maybe_feed_calibration(&mut field, &history);

        assert_eq!(
            field.status(),
            CalibrationStatus::Collecting,
            "first frame must flip Uncalibrated -> Collecting"
        );
        assert_eq!(
            field.calibration_frame_count(),
            1,
            "frame count must advance past 0"
        );

        // Subsequent frames keep accumulating while Collecting.
        maybe_feed_calibration(&mut field, &history);
        assert_eq!(field.calibration_frame_count(), 2);
    }

    /// Regression (#1170 pattern): a real HT40 node streams 128-wide amplitude
    /// frames, but the FieldModel is a 56-tone grid. Before canonicalization,
    /// `feed_calibration` rejected every frame with DimensionMismatch (swallowed
    /// at debug), so frame_count stayed 0 even with the deadlock fixed. The feed
    /// must resample 128 → 56 and actually accumulate.
    #[test]
    fn maybe_feed_calibration_resamples_wide_frames_and_accumulates() {
        let mut field = FieldModel::new(single_link_config()).expect("field model");

        // 128-wide frame (HT40), NOT the model's 56 — would DimensionMismatch raw.
        let wide = vec![0.5_f64; 128];
        let mut history: VecDeque<Vec<f64>> = VecDeque::new();
        history.push_back(wide);

        maybe_feed_calibration(&mut field, &history);

        assert_eq!(
            field.status(),
            CalibrationStatus::Collecting,
            "128-wide frame must resample to 56 and be accepted"
        );
        assert_eq!(
            field.calibration_frame_count(),
            1,
            "wide frame must accumulate, not be silently dropped"
        );
    }

    #[test]
    fn calibrated_occupancy_is_fresh_model_only_and_normalizes_runtime_frames() {
        let field = fresh_test_model();
        let mut history = VecDeque::new();
        // Deliberately use a real-world wide frame. The evidence path must use
        // the same canonicalizer as calibration instead of falling back.
        history.push_back(vec![1.02; 128]);

        let result =
            calibrated_occupancy(&field, &history, 1_500_000).expect("fresh calibrated inference");
        assert!(result.person_count <= MAX_SINGLE_LINK_OCCUPANCY);
        assert!(matches!(
            result.method,
            CalibratedOccupancyMethod::Eigenvalue | CalibratedOccupancyMethod::PerturbationEnergy
        ));

        assert_eq!(calibrated_occupancy(&field, &history, 4_000_000), None);
        assert_eq!(
            calibrated_occupancy(&field, &VecDeque::new(), 1_500_000),
            None
        );
    }

    #[test]
    fn multilink_calibration_consumes_one_spatial_cohort_per_frame() {
        let mut field = FieldModel::new(multi_link_config(3)).expect("field model");
        let cohort = vec![vec![0.5; 56], vec![0.6; 56], vec![0.7; 56]];

        assert!(maybe_feed_calibration_observation(&mut field, &cohort));
        assert_eq!(field.calibration_frame_count(), 1);
        assert!(
            !maybe_feed_calibration_observation(&mut field, &cohort[..1]),
            "a partial node cohort must be rejected instead of contaminating the baseline"
        );
        assert_eq!(field.calibration_frame_count(), 1);
    }

    fn calibrated_multilink_test_model() -> (FieldModel, Vec<Vec<f64>>) {
        let config = FieldModelConfig {
            n_links: 3,
            n_subcarriers: 8,
            n_modes: 3,
            min_calibration_frames: 32,
            baseline_expiry_s: 10.0,
        };
        let mut field = FieldModel::new(config).unwrap();
        let mut last = Vec::new();
        for frame in 0..96 {
            last = (0..3)
                .map(|link| {
                    (0..8)
                        .map(|tone| {
                            let phase = (frame * (link + 1) + tone * 7) as f64 * 0.071;
                            2.0 + link as f64 * 0.4 + tone as f64 * 0.02 + phase.sin() * 0.08
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            field.feed_calibration(&last).unwrap();
        }
        field.finalize_calibration(1_000_000, 7).unwrap();
        (field, last)
    }

    #[test]
    fn null_normalized_detector_requires_quorum_and_hysteresis() {
        let (field, null_like) = calibrated_multilink_test_model();
        let mut detector = CalibratedPresenceDetector::new(PresenceDetectorConfig {
            node_quorum: 2,
            enter_frames: 2,
            exit_frames: 3,
        });

        let empty =
            calibrated_multilink_occupancy(&field, &null_like, 1_500_000, &mut detector, true)
                .unwrap();
        assert_eq!(empty.person_count, 0);
        assert_eq!(empty.residual_evidence.as_ref().unwrap().links.len(), 3);
        assert!(empty.residual_evidence.as_ref().unwrap().links[0]
            .decision_threshold
            .is_finite());

        let mut one_noisy_link = null_like.clone();
        for (tone, value) in one_noisy_link[0].iter_mut().enumerate() {
            *value += if tone % 2 == 0 { 5.0 } else { -5.0 };
        }
        for _ in 0..3 {
            let result = calibrated_multilink_occupancy(
                &field,
                &one_noisy_link,
                1_500_000,
                &mut detector,
                true,
            )
            .unwrap();
            assert_eq!(
                result.person_count, 0,
                "one noisy node must not satisfy quorum"
            );
        }

        let mut two_changed_links = one_noisy_link;
        for (tone, value) in two_changed_links[1].iter_mut().enumerate() {
            *value += if tone % 2 == 0 { 5.0 } else { -5.0 };
        }
        let entering = calibrated_multilink_occupancy(
            &field,
            &two_changed_links,
            1_500_000,
            &mut detector,
            true,
        )
        .unwrap();
        assert_eq!(entering.person_count, 0);
        let present = calibrated_multilink_occupancy(
            &field,
            &two_changed_links,
            1_500_000,
            &mut detector,
            true,
        )
        .unwrap();
        assert_eq!(present.person_count, 1);
        assert_eq!(present.method.wire_name(), "field_model_null_normalized_v2");

        for expected in [1, 1, 0] {
            let result =
                calibrated_multilink_occupancy(&field, &null_like, 1_500_000, &mut detector, true)
                    .unwrap();
            assert_eq!(result.person_count, expected);
        }
    }
}

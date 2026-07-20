//! Ensemble classifier that combines breathing, heartbeat, and movement signals
//! into a unified survivor detection confidence score.
//!
//! The ensemble uses weighted voting across the three detector signals:
//! - Breathing presence is the strongest indicator of a living survivor
//! - Heartbeat (when enabled) provides high-confidence confirmation
//! - Movement type distinguishes active vs trapped survivors
//!
//! The classifier produces a single confidence score and a recommended
//! triage status based on the combined signals.

use crate::domain::{
    triage::TriageCalculator, MovementType, TriageStatus, VitalSignsReading,
};

/// Configuration for the ensemble classifier
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Weight for breathing signal (0.0-1.0)
    pub breathing_weight: f64,
    /// Weight for heartbeat signal (0.0-1.0)
    pub heartbeat_weight: f64,
    /// Weight for movement signal (0.0-1.0)
    pub movement_weight: f64,
    /// Minimum combined confidence to report a detection
    pub min_ensemble_confidence: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            breathing_weight: 0.50,
            heartbeat_weight: 0.30,
            movement_weight: 0.20,
            min_ensemble_confidence: 0.3,
        }
    }
}

/// Result of ensemble classification
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Combined confidence score (0.0-1.0)
    pub confidence: f64,
    /// Recommended triage status based on signal analysis
    pub recommended_triage: TriageStatus,
    /// Whether breathing was detected
    pub breathing_detected: bool,
    /// Whether heartbeat was detected
    pub heartbeat_detected: bool,
    /// Whether meaningful movement was detected
    pub movement_detected: bool,
    /// Individual signal confidences
    pub signal_confidences: SignalConfidences,
}

/// Individual confidence scores for each signal type
#[derive(Debug, Clone)]
pub struct SignalConfidences {
    /// Breathing detection confidence
    pub breathing: f64,
    /// Heartbeat detection confidence
    pub heartbeat: f64,
    /// Movement detection confidence
    pub movement: f64,
}

/// Ensemble classifier combining breathing, heartbeat, and movement detectors
pub struct EnsembleClassifier {
    config: EnsembleConfig,
}

impl EnsembleClassifier {
    /// Create a new ensemble classifier
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Classify a vital signs reading using weighted ensemble voting.
    ///
    /// The ensemble combines individual detector outputs with configured weights
    /// to produce a single confidence score and triage recommendation.
    pub fn classify(&self, reading: &VitalSignsReading) -> EnsembleResult {
        // Extract individual signal confidences (using method calls)
        let breathing_conf = reading
            .breathing
            .as_ref()
            .map(|b| b.confidence())
            .unwrap_or(0.0);

        let heartbeat_conf = reading
            .heartbeat
            .as_ref()
            .map(|h| h.confidence())
            .unwrap_or(0.0);

        let movement_conf = if reading.movement.movement_type != MovementType::None {
            reading.movement.confidence()
        } else {
            0.0
        };

        // Weighted ensemble confidence
        let total_weight = self.config.breathing_weight
            + self.config.heartbeat_weight
            + self.config.movement_weight;

        let ensemble_confidence = if total_weight > 0.0 {
            (breathing_conf * self.config.breathing_weight
                + heartbeat_conf * self.config.heartbeat_weight
                + movement_conf * self.config.movement_weight)
                / total_weight
        } else {
            0.0
        };

        let breathing_detected = reading.breathing.is_some();
        let heartbeat_detected = reading.heartbeat.is_some();
        let movement_detected = reading.movement.movement_type != MovementType::None;

        // Determine triage status from signal combination
        let recommended_triage = self.determine_triage(reading, ensemble_confidence);

        EnsembleResult {
            confidence: ensemble_confidence,
            recommended_triage,
            breathing_detected,
            heartbeat_detected,
            movement_detected,
            signal_confidences: SignalConfidences {
                breathing: breathing_conf,
                heartbeat: heartbeat_conf,
                movement: movement_conf,
            },
        }
    }

    /// Determine triage status for a reading.
    ///
    /// CANONICAL TRIAGE: this delegates to [`TriageCalculator::calculate`], the
    /// single source of truth used by both the ensemble gate (here) and the
    /// `Survivor` record (`Survivor::new` / `update_vitals`). Previously this
    /// method implemented a *second*, divergent START-protocol approximation
    /// (different rate bands, different movement handling). The pipeline gated
    /// on the ensemble's triage then discarded it and recomputed via
    /// `TriageCalculator` in `Survivor::new`, so a survivor could be gated as
    /// one priority and recorded as another (e.g. 28 bpm + Tremor: old ensemble
    /// said Delayed, the survivor record said Immediate). In a mass-casualty
    /// tool that divergence is a life-safety defect. The two are now unified.
    ///
    /// The only ensemble-specific behaviour retained is the confidence gate:
    /// when the combined ensemble confidence is below the configured minimum,
    /// the reading is reported [`TriageStatus::Unknown`] (insufficient signal to
    /// classify) UNLESS the canonical calculator flags it [`TriageStatus::Immediate`].
    /// Distress is never suppressed by low confidence — a false negative
    /// (missing a survivor in distress) is far more costly than a false positive.
    fn determine_triage(&self, reading: &VitalSignsReading, confidence: f64) -> TriageStatus {
        let canonical = TriageCalculator::calculate(reading);

        // Distress (Immediate) is always surfaced regardless of confidence.
        if canonical == TriageStatus::Immediate {
            return TriageStatus::Immediate;
        }

        // Below the ensemble confidence threshold: not enough signal to trust a
        // non-distress classification. Report Unknown rather than guessing.
        if confidence < self.config.min_ensemble_confidence {
            return TriageStatus::Unknown;
        }

        canonical
    }

    /// Get configuration
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        BreathingPattern, BreathingType, ConfidenceScore, HeartbeatSignature, MovementProfile,
        SignalStrength,
    };

    fn make_reading(
        breathing: Option<(f32, BreathingType)>,
        heartbeat: Option<f32>,
        movement: MovementType,
    ) -> VitalSignsReading {
        let bp = breathing.map(|(rate, pattern_type)| BreathingPattern {
            rate_bpm: rate,
            pattern_type,
            amplitude: 0.9,
            regularity: 0.9,
        });

        let hb = heartbeat.map(|rate| HeartbeatSignature {
            rate_bpm: rate,
            variability: 0.1,
            strength: SignalStrength::Moderate,
        });

        let is_moving = movement != MovementType::None;
        let mv = MovementProfile {
            movement_type: movement,
            intensity: if is_moving { 0.5 } else { 0.0 },
            frequency: 0.0,
            is_voluntary: is_moving,
        };

        VitalSignsReading::new(bp, hb, mv)
    }

    #[test]
    fn test_normal_breathing_with_periodic_movement_is_canonical() {
        // UNIFICATION: Periodic movement maps to MinimalMovement in the canonical
        // calculator (it is likely breathing-correlated, not purposeful), so
        // Normal breathing + Periodic → Delayed. The old ensemble engine treated
        // ANY non-None movement as "active" and returned Minor — diverging from
        // the survivor record. Gate and survivor must now agree.
        let classifier = EnsembleClassifier::new(EnsembleConfig::default());
        let reading = make_reading(
            Some((16.0, BreathingType::Normal)),
            None,
            MovementType::Periodic,
        );

        let result = classifier.classify(&reading);
        assert!(result.confidence > 0.0);
        assert!(result.breathing_detected);
        let survivor = crate::domain::triage::TriageCalculator::calculate(&reading);
        assert_eq!(result.recommended_triage, survivor);
        assert_eq!(result.recommended_triage, TriageStatus::Delayed);
    }

    #[test]
    fn test_normal_breathing_purposeful_movement_is_minor() {
        // Gross + voluntary = Responsive (following commands / walking wounded).
        // make_reading sets is_voluntary=true for any non-None movement, so Gross
        // here is voluntary → Responsive → Minor. Confirms the canonical "walking
        // wounded" path still resolves to Minor and gate==survivor.
        let classifier = EnsembleClassifier::new(EnsembleConfig::default());
        let reading = make_reading(
            Some((16.0, BreathingType::Normal)),
            None,
            MovementType::Gross,
        );

        let result = classifier.classify(&reading);
        let survivor = crate::domain::triage::TriageCalculator::calculate(&reading);
        assert_eq!(result.recommended_triage, survivor);
        assert_eq!(result.recommended_triage, TriageStatus::Minor);
    }

    #[test]
    fn test_agonal_breathing_is_immediate() {
        let classifier = EnsembleClassifier::new(EnsembleConfig::default());
        let reading = make_reading(Some((8.0, BreathingType::Agonal)), None, MovementType::None);

        let result = classifier.classify(&reading);
        assert_eq!(result.recommended_triage, TriageStatus::Immediate);
    }

    #[test]
    fn test_normal_breathing_no_movement_is_immediate_canonical() {
        // UNIFICATION: Normal breathing but ZERO detectable movement means the
        // survivor is unresponsive (not following commands) — START classifies
        // breathing-but-unresponsive as Immediate. The old ensemble engine
        // returned Delayed here, diverging from the survivor record. Gate and
        // survivor must agree.
        let classifier = EnsembleClassifier::new(EnsembleConfig {
            min_ensemble_confidence: 0.0,
            ..EnsembleConfig::default()
        });
        let reading = make_reading(
            Some((16.0, BreathingType::Normal)),
            None,
            MovementType::None,
        );

        let result = classifier.classify(&reading);
        let survivor = crate::domain::triage::TriageCalculator::calculate(&reading);
        assert_eq!(result.recommended_triage, survivor);
        assert_eq!(result.recommended_triage, TriageStatus::Immediate);
    }

    #[test]
    fn test_no_vitals_is_unknown_canonical() {
        // UNIFICATION: with the canonical TriageCalculator now driving the gate,
        // a reading with NO sensed vitals at all is Unknown (a remote sensor that
        // sees nothing cannot confirm death — it may be a signal/occlusion issue),
        // matching what `Survivor::new` records. The old ensemble engine returned
        // Deceased here, diverging from the survivor record; that is the bug this
        // task fixes.
        let mv = MovementProfile::default();
        let mut reading = VitalSignsReading::new(None, None, mv);
        reading.confidence = ConfidenceScore::new(0.5);

        let config = EnsembleConfig {
            min_ensemble_confidence: 0.0,
            ..EnsembleConfig::default()
        };
        let classifier = EnsembleClassifier::new(config);

        let result = classifier.classify(&reading);
        assert_eq!(result.recommended_triage, TriageStatus::Unknown);
        // And it must agree with the canonical calculator directly.
        assert_eq!(
            result.recommended_triage,
            crate::domain::triage::TriageCalculator::calculate(&reading)
        );
    }

    /// CRITICAL unification regression (fails on the old divergent engines).
    ///
    /// A 28 bpm Normal-rate breather with only an involuntary Tremor is a
    /// classic divergent boundary case:
    ///  - OLD ensemble engine: 28 ∈ [10,30] and ∈ [12,24] is false, but it had
    ///    movement → Delayed.
    ///  - OLD `TriageCalculator` (used by `Survivor::new`): 28 ∈ [10,30] = Normal
    ///    breathing, Tremor → InvoluntaryOnly (not following commands) → Immediate.
    /// The gate would have admitted it as Delayed while the survivor record said
    /// Immediate. After unification BOTH must return the SAME triage.
    #[test]
    fn test_divergent_boundary_28bpm_tremor_gate_equals_survivor() {
        let reading = make_reading(
            Some((28.0, BreathingType::Normal)),
            None,
            MovementType::Tremor,
        );

        let classifier = EnsembleClassifier::new(EnsembleConfig {
            min_ensemble_confidence: 0.0,
            ..EnsembleConfig::default()
        });

        // Gate triage (ensemble) and survivor-record triage (Survivor::new path,
        // i.e. TriageCalculator::calculate) must be identical.
        let gate = classifier.classify(&reading).recommended_triage;
        let survivor = crate::domain::triage::TriageCalculator::calculate(&reading);

        assert_eq!(
            gate, survivor,
            "gate triage {gate:?} must equal survivor-record triage {survivor:?}"
        );
        // And the canonical answer for this distress case is Immediate.
        assert_eq!(gate, TriageStatus::Immediate);
    }

    /// SAFETY regression: heartbeat present but no sensed breathing/movement is
    /// respiratory arrest — Immediate, never Deceased. Only the *total* absence
    /// of breathing, movement AND heartbeat (the test above) is Deceased.
    #[test]
    fn test_heartbeat_with_no_breathing_or_movement_is_immediate() {
        // breathing: None, heartbeat: Some(72 bpm), movement: None
        let reading = make_reading(None, Some(72.0), MovementType::None);

        let classifier = EnsembleClassifier::new(EnsembleConfig {
            min_ensemble_confidence: 0.0,
            ..EnsembleConfig::default()
        });

        let result = classifier.classify(&reading);
        assert_eq!(
            result.recommended_triage,
            TriageStatus::Immediate,
            "a survivor with a pulse must never be triaged Deceased"
        );
    }

    #[test]
    fn test_ensemble_confidence_weighting() {
        let classifier = EnsembleClassifier::new(EnsembleConfig {
            breathing_weight: 0.6,
            heartbeat_weight: 0.3,
            movement_weight: 0.1,
            min_ensemble_confidence: 0.0,
        });

        let reading = make_reading(
            Some((16.0, BreathingType::Normal)),
            Some(72.0),
            MovementType::Periodic,
        );

        let result = classifier.classify(&reading);
        assert!(result.confidence > 0.0);
        assert!(result.breathing_detected);
        assert!(result.heartbeat_detected);
        assert!(result.movement_detected);
    }
}

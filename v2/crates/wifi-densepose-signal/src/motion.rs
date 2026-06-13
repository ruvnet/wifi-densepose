//! Motion Detection Module
//!
//! This module provides motion detection and human presence detection
//! capabilities based on CSI features.

use crate::features::{AmplitudeFeatures, CorrelationFeatures, CsiFeatures, PhaseFeatures};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Tuning constants (ADR-154 §7.4 #18 — de-magicked; EMPIRICAL DEFAULTS).
//
// These were previously bare literals inside the scoring functions. They are
// lifted to named, documented consts so the implicit weighting becomes
// explicit and a future retune is a visible, tested change. The values are
// **unchanged** from the original literals — boundary/characterization tests
// pin the current behaviour. None of these is calibrated against labelled
// occupancy data; they are heuristic fusion weights.
// ---------------------------------------------------------------------------

/// Motion-score fusion weights when a Doppler component is present.
/// `(variance, correlation, phase, doppler)` — sums to 1.0.
const MOTION_WEIGHTS_WITH_DOPPLER: (f64, f64, f64, f64) = (0.3, 0.2, 0.2, 0.3);

/// Motion-score fusion weights with no Doppler component.
/// `(variance, correlation, phase)` — sums to 1.0.
const MOTION_WEIGHTS_NO_DOPPLER: (f64, f64, f64) = (0.4, 0.3, 0.3);

/// Doppler magnitude (Hz-ish, arbitrary units) that maps to a full-scale
/// (1.0) Doppler motion component. Larger magnitudes saturate at 1.0.
const DOPPLER_FULL_SCALE_MAGNITUDE: f64 = 100.0;

/// Reference variance that maps to a full-scale (1.0) heuristic motion score
/// when no calibrated baseline is available. Empirical default.
const VARIANCE_HEURISTIC_FULL_SCALE: f64 = 0.5;

/// Reference phase variance that maps to a full-scale (1.0) phase motion
/// component. Empirical default.
const PHASE_VARIANCE_FULL_SCALE: f64 = 0.5;

/// Blend weight between phase-variance and phase-coherence in the phase score.
const PHASE_SCORE_VARIANCE_WEIGHT: f64 = 0.5;

/// Reference dynamic range that maps to a full-scale (1.0) amplitude-quality
/// confidence indicator. Empirical default.
const AMP_QUALITY_FULL_SCALE_RANGE: f64 = 2.0;

/// Confidence-indicator blend weights (`amplitude`, `phase`, `correlation`,
/// `doppler`) — each is the fraction of total confidence that indicator
/// contributes when present.
const CONF_WEIGHT_AMPLITUDE: f64 = 0.3;
const CONF_WEIGHT_PHASE: f64 = 0.3;
const CONF_WEIGHT_CORRELATION: f64 = 0.2;
const CONF_WEIGHT_DOPPLER: f64 = 0.2;

/// Minimum baseline floor added before dividing by the calibration baseline
/// variance, preventing a divide-by-zero on an all-constant calibration.
const BASELINE_VARIANCE_FLOOR: f64 = 1e-10;

/// Lower / upper clamp for the adaptive human-detection threshold
/// (`mean + 1σ` of recent motion scores). Keeps the adaptive threshold inside
/// a sane operating band. Empirical default.
const ADAPTIVE_THRESHOLD_MIN: f64 = 0.3;
const ADAPTIVE_THRESHOLD_MAX: f64 = 0.95;

/// Minimum history length before the adaptive threshold engages; below this
/// the configured fixed threshold is used.
const ADAPTIVE_THRESHOLD_MIN_HISTORY: usize = 10;

/// Motion score with component breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionScore {
    /// Overall motion score (0.0 to 1.0)
    pub total: f64,

    /// Variance-based motion component
    pub variance_component: f64,

    /// Correlation-based motion component
    pub correlation_component: f64,

    /// Phase-based motion component
    pub phase_component: f64,

    /// Doppler-based motion component (if available)
    pub doppler_component: Option<f64>,
}

impl MotionScore {
    /// Create a new motion score
    pub fn new(
        variance_component: f64,
        correlation_component: f64,
        phase_component: f64,
        doppler_component: Option<f64>,
    ) -> Self {
        // Calculate weighted total
        let total = if let Some(doppler) = doppler_component {
            let (wv, wc, wp, wd) = MOTION_WEIGHTS_WITH_DOPPLER;
            wv * variance_component + wc * correlation_component + wp * phase_component + wd * doppler
        } else {
            let (wv, wc, wp) = MOTION_WEIGHTS_NO_DOPPLER;
            wv * variance_component + wc * correlation_component + wp * phase_component
        };

        Self {
            total: total.clamp(0.0, 1.0),
            variance_component,
            correlation_component,
            phase_component,
            doppler_component,
        }
    }

    /// Check if motion is detected above threshold
    pub fn is_motion_detected(&self, threshold: f64) -> bool {
        self.total >= threshold
    }
}

/// Motion analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionAnalysis {
    /// Motion score
    pub score: MotionScore,

    /// Temporal variance of motion
    pub temporal_variance: f64,

    /// Spatial variance of motion
    pub spatial_variance: f64,

    /// Estimated motion velocity (arbitrary units)
    pub estimated_velocity: f64,

    /// Motion direction estimate (radians, if available)
    pub motion_direction: Option<f64>,

    /// Confidence in the analysis
    pub confidence: f64,
}

/// Human detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanDetectionResult {
    /// Whether a human was detected
    pub human_detected: bool,

    /// Detection confidence (0.0 to 1.0)
    pub confidence: f64,

    /// Motion score
    pub motion_score: f64,

    /// Raw (unsmoothed) confidence
    pub raw_confidence: f64,

    /// Timestamp of detection
    pub timestamp: DateTime<Utc>,

    /// Detection threshold used
    pub threshold: f64,

    /// Detailed motion analysis
    pub motion_analysis: MotionAnalysis,

    /// Additional metadata
    #[serde(default)]
    pub metadata: DetectionMetadata,
}

/// Metadata for detection results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectionMetadata {
    /// Number of features used
    pub features_used: usize,

    /// Processing time in milliseconds
    pub processing_time_ms: Option<f64>,

    /// Whether Doppler was available
    pub doppler_available: bool,

    /// History length used
    pub history_length: usize,
}

/// Configuration for motion detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionDetectorConfig {
    /// Human detection threshold (0.0 to 1.0)
    pub human_detection_threshold: f64,

    /// Motion detection threshold (0.0 to 1.0)
    pub motion_threshold: f64,

    /// Temporal smoothing factor (0.0 to 1.0)
    /// Higher values give more weight to previous detections
    pub smoothing_factor: f64,

    /// Minimum amplitude indicator threshold
    pub amplitude_threshold: f64,

    /// Minimum phase indicator threshold
    pub phase_threshold: f64,

    /// History size for temporal analysis
    pub history_size: usize,

    /// Enable adaptive thresholding
    pub adaptive_threshold: bool,

    /// Weight for amplitude indicator
    pub amplitude_weight: f64,

    /// Weight for phase indicator
    pub phase_weight: f64,

    /// Weight for motion indicator
    pub motion_weight: f64,
}

impl Default for MotionDetectorConfig {
    fn default() -> Self {
        Self {
            human_detection_threshold: 0.8,
            motion_threshold: 0.3,
            smoothing_factor: 0.9,
            amplitude_threshold: 0.1,
            phase_threshold: 0.05,
            history_size: 100,
            adaptive_threshold: false,
            amplitude_weight: 0.4,
            phase_weight: 0.3,
            motion_weight: 0.3,
        }
    }
}

impl MotionDetectorConfig {
    /// Create a new builder
    pub fn builder() -> MotionDetectorConfigBuilder {
        MotionDetectorConfigBuilder::new()
    }
}

/// Builder for MotionDetectorConfig
#[derive(Debug, Default)]
pub struct MotionDetectorConfigBuilder {
    config: MotionDetectorConfig,
}

impl MotionDetectorConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: MotionDetectorConfig::default(),
        }
    }

    /// Set human detection threshold
    pub fn human_detection_threshold(mut self, threshold: f64) -> Self {
        self.config.human_detection_threshold = threshold;
        self
    }

    /// Set motion threshold
    pub fn motion_threshold(mut self, threshold: f64) -> Self {
        self.config.motion_threshold = threshold;
        self
    }

    /// Set smoothing factor
    pub fn smoothing_factor(mut self, factor: f64) -> Self {
        self.config.smoothing_factor = factor;
        self
    }

    /// Set amplitude threshold
    pub fn amplitude_threshold(mut self, threshold: f64) -> Self {
        self.config.amplitude_threshold = threshold;
        self
    }

    /// Set phase threshold
    pub fn phase_threshold(mut self, threshold: f64) -> Self {
        self.config.phase_threshold = threshold;
        self
    }

    /// Set history size
    pub fn history_size(mut self, size: usize) -> Self {
        self.config.history_size = size;
        self
    }

    /// Enable adaptive thresholding
    pub fn adaptive_threshold(mut self, enable: bool) -> Self {
        self.config.adaptive_threshold = enable;
        self
    }

    /// Set indicator weights
    pub fn weights(mut self, amplitude: f64, phase: f64, motion: f64) -> Self {
        self.config.amplitude_weight = amplitude;
        self.config.phase_weight = phase;
        self.config.motion_weight = motion;
        self
    }

    /// Build configuration
    pub fn build(self) -> MotionDetectorConfig {
        self.config
    }
}

/// Motion detector for human presence detection
#[derive(Debug)]
pub struct MotionDetector {
    config: MotionDetectorConfig,
    previous_confidence: f64,
    motion_history: VecDeque<MotionScore>,
    detection_count: usize,
    total_detections: usize,
    baseline_variance: Option<f64>,
}

impl MotionDetector {
    /// Create a new motion detector
    pub fn new(config: MotionDetectorConfig) -> Self {
        Self {
            motion_history: VecDeque::with_capacity(config.history_size),
            config,
            previous_confidence: 0.0,
            detection_count: 0,
            total_detections: 0,
            baseline_variance: None,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MotionDetectorConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &MotionDetectorConfig {
        &self.config
    }

    /// Analyze motion patterns from CSI features
    pub fn analyze_motion(&self, features: &CsiFeatures) -> MotionAnalysis {
        // Calculate variance-based motion score
        let variance_score = self.calculate_variance_score(&features.amplitude);

        // Calculate correlation-based motion score
        let correlation_score = self.calculate_correlation_score(&features.correlation);

        // Calculate phase-based motion score
        let phase_score = self.calculate_phase_score(&features.phase);

        // Calculate Doppler-based score if available
        let doppler_score = features.doppler.as_ref().map(|d| {
            // Normalize Doppler magnitude to 0-1 range
            (d.mean_magnitude / DOPPLER_FULL_SCALE_MAGNITUDE).clamp(0.0, 1.0)
        });

        let motion_score = MotionScore::new(
            variance_score,
            correlation_score,
            phase_score,
            doppler_score,
        );

        // Calculate temporal and spatial variance
        let temporal_variance = self.calculate_temporal_variance();
        let spatial_variance = features.amplitude.variance.iter().sum::<f64>()
            / features.amplitude.variance.len() as f64;

        // Estimate velocity from Doppler if available
        let estimated_velocity = features
            .doppler
            .as_ref()
            .map(|d| d.mean_magnitude)
            .unwrap_or(0.0);

        // Motion direction from phase gradient
        let motion_direction = if !features.phase.gradient.is_empty() {
            let mean_grad: f64 =
                features.phase.gradient.iter().sum::<f64>() / features.phase.gradient.len() as f64;
            Some(mean_grad.atan())
        } else {
            None
        };

        // Calculate confidence based on signal quality indicators
        let confidence = self.calculate_motion_confidence(features);

        MotionAnalysis {
            score: motion_score,
            temporal_variance,
            spatial_variance,
            estimated_velocity,
            motion_direction,
            confidence,
        }
    }

    /// Calculate variance-based motion score
    fn calculate_variance_score(&self, amplitude: &AmplitudeFeatures) -> f64 {
        let mean_variance =
            amplitude.variance.iter().sum::<f64>() / amplitude.variance.len() as f64;

        // Normalize using baseline if available
        if let Some(baseline) = self.baseline_variance {
            let ratio = mean_variance / (baseline + BASELINE_VARIANCE_FLOOR);
            (ratio - 1.0).max(0.0).tanh()
        } else {
            // Use heuristic normalization
            (mean_variance / VARIANCE_HEURISTIC_FULL_SCALE).clamp(0.0, 1.0)
        }
    }

    /// Calculate correlation-based motion score
    fn calculate_correlation_score(&self, correlation: &CorrelationFeatures) -> f64 {
        let n = correlation.matrix.dim().0;
        if n < 2 {
            return 0.0;
        }

        // Calculate mean deviation from identity matrix
        let mut deviation_sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                deviation_sum += (correlation.matrix[[i, j]] - expected).abs();
                count += 1;
            }
        }

        let mean_deviation = deviation_sum / count as f64;
        mean_deviation.clamp(0.0, 1.0)
    }

    /// Calculate phase-based motion score
    fn calculate_phase_score(&self, phase: &PhaseFeatures) -> f64 {
        // Use phase variance and coherence
        let mean_variance = phase.variance.iter().sum::<f64>() / phase.variance.len() as f64;
        let coherence_factor = 1.0 - phase.coherence.abs();

        // Combine factors
        let w = PHASE_SCORE_VARIANCE_WEIGHT;
        let score = w * (mean_variance / PHASE_VARIANCE_FULL_SCALE).clamp(0.0, 1.0)
            + (1.0 - w) * coherence_factor;
        score.clamp(0.0, 1.0)
    }

    /// Calculate temporal variance from motion history
    fn calculate_temporal_variance(&self) -> f64 {
        if self.motion_history.len() < 2 {
            return 0.0;
        }

        let scores: Vec<f64> = self.motion_history.iter().map(|m| m.total).collect();
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance: f64 =
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        variance.sqrt()
    }

    /// Calculate confidence in motion detection
    fn calculate_motion_confidence(&self, features: &CsiFeatures) -> f64 {
        let mut confidence = 0.0;
        let mut weight_sum = 0.0;

        // Amplitude quality indicator
        let amp_quality =
            (features.amplitude.dynamic_range / AMP_QUALITY_FULL_SCALE_RANGE).clamp(0.0, 1.0);
        confidence += amp_quality * CONF_WEIGHT_AMPLITUDE;
        weight_sum += CONF_WEIGHT_AMPLITUDE;

        // Phase coherence indicator
        let phase_quality = features.phase.coherence.abs();
        confidence += phase_quality * CONF_WEIGHT_PHASE;
        weight_sum += CONF_WEIGHT_PHASE;

        // Correlation consistency indicator
        let corr_quality = (1.0 - features.correlation.correlation_spread).clamp(0.0, 1.0);
        confidence += corr_quality * CONF_WEIGHT_CORRELATION;
        weight_sum += CONF_WEIGHT_CORRELATION;

        // Doppler quality if available
        if let Some(ref doppler) = features.doppler {
            let doppler_quality =
                (doppler.spread / doppler.mean_magnitude.max(1.0)).clamp(0.0, 1.0);
            confidence += (1.0 - doppler_quality) * CONF_WEIGHT_DOPPLER;
            weight_sum += CONF_WEIGHT_DOPPLER;
        }

        if weight_sum > 0.0 {
            confidence / weight_sum
        } else {
            0.0
        }
    }

    /// Calculate detection confidence from features and motion score
    fn calculate_detection_confidence(&self, features: &CsiFeatures, motion_score: f64) -> f64 {
        // Amplitude indicator
        let amplitude_mean =
            features.amplitude.mean.iter().sum::<f64>() / features.amplitude.mean.len() as f64;
        let amplitude_indicator = if amplitude_mean > self.config.amplitude_threshold {
            1.0
        } else {
            0.0
        };

        // Phase indicator
        let phase_std = features.phase.variance.iter().sum::<f64>().sqrt()
            / features.phase.variance.len() as f64;
        let phase_indicator = if phase_std > self.config.phase_threshold {
            1.0
        } else {
            0.0
        };

        // Motion indicator
        let motion_indicator = if motion_score > self.config.motion_threshold {
            1.0
        } else {
            0.0
        };

        // Weighted combination
        let confidence = self.config.amplitude_weight * amplitude_indicator
            + self.config.phase_weight * phase_indicator
            + self.config.motion_weight * motion_indicator;

        confidence.clamp(0.0, 1.0)
    }

    /// Apply temporal smoothing (exponential moving average)
    fn apply_temporal_smoothing(&mut self, raw_confidence: f64) -> f64 {
        let smoothed = self.config.smoothing_factor * self.previous_confidence
            + (1.0 - self.config.smoothing_factor) * raw_confidence;
        self.previous_confidence = smoothed;
        smoothed
    }

    /// Detect human presence from CSI features
    pub fn detect_human(&mut self, features: &CsiFeatures) -> HumanDetectionResult {
        // Analyze motion
        let motion_analysis = self.analyze_motion(features);

        // Add to history
        if self.motion_history.len() >= self.config.history_size {
            self.motion_history.pop_front();
        }
        self.motion_history.push_back(motion_analysis.score.clone());

        // Calculate detection confidence
        let raw_confidence =
            self.calculate_detection_confidence(features, motion_analysis.score.total);

        // Apply temporal smoothing
        let smoothed_confidence = self.apply_temporal_smoothing(raw_confidence);

        // Get effective threshold (adaptive if enabled)
        let threshold = if self.config.adaptive_threshold {
            self.calculate_adaptive_threshold()
        } else {
            self.config.human_detection_threshold
        };

        // Determine detection
        let human_detected = smoothed_confidence >= threshold;

        self.total_detections += 1;
        if human_detected {
            self.detection_count += 1;
        }

        let metadata = DetectionMetadata {
            features_used: 4, // amplitude, phase, correlation, psd
            processing_time_ms: None,
            doppler_available: features.doppler.is_some(),
            history_length: self.motion_history.len(),
        };

        HumanDetectionResult {
            human_detected,
            confidence: smoothed_confidence,
            motion_score: motion_analysis.score.total,
            raw_confidence,
            timestamp: Utc::now(),
            threshold,
            motion_analysis,
            metadata,
        }
    }

    /// Calculate adaptive threshold based on recent history
    fn calculate_adaptive_threshold(&self) -> f64 {
        if self.motion_history.len() < ADAPTIVE_THRESHOLD_MIN_HISTORY {
            return self.config.human_detection_threshold;
        }

        let scores: Vec<f64> = self.motion_history.iter().map(|m| m.total).collect();
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let std: f64 = {
            let var: f64 =
                scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            var.sqrt()
        };

        // Threshold is mean + 1 std deviation, clamped to reasonable range
        (mean + std).clamp(ADAPTIVE_THRESHOLD_MIN, ADAPTIVE_THRESHOLD_MAX)
    }

    /// Update baseline variance (for calibration)
    pub fn calibrate(&mut self, features: &CsiFeatures) {
        let mean_variance = features.amplitude.variance.iter().sum::<f64>()
            / features.amplitude.variance.len() as f64;
        self.baseline_variance = Some(mean_variance);
    }

    /// Clear calibration
    pub fn clear_calibration(&mut self) {
        self.baseline_variance = None;
    }

    /// Get detection statistics
    pub fn get_statistics(&self) -> DetectionStatistics {
        DetectionStatistics {
            total_detections: self.total_detections,
            positive_detections: self.detection_count,
            detection_rate: if self.total_detections > 0 {
                self.detection_count as f64 / self.total_detections as f64
            } else {
                0.0
            },
            history_size: self.motion_history.len(),
            is_calibrated: self.baseline_variance.is_some(),
        }
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.previous_confidence = 0.0;
        self.motion_history.clear();
        self.detection_count = 0;
        self.total_detections = 0;
    }

    /// Get previous confidence value
    pub fn previous_confidence(&self) -> f64 {
        self.previous_confidence
    }
}

/// Detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionStatistics {
    /// Total number of detection attempts
    pub total_detections: usize,

    /// Number of positive detections
    pub positive_detections: usize,

    /// Detection rate (0.0 to 1.0)
    pub detection_rate: f64,

    /// Current history size
    pub history_size: usize,

    /// Whether detector is calibrated
    pub is_calibrated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csi_processor::CsiData;
    use crate::features::FeatureExtractor;
    use ndarray::Array2;

    fn create_test_csi_data(motion_level: f64) -> CsiData {
        let amplitude = Array2::from_shape_fn((4, 64), |(i, j)| {
            1.0 + motion_level * 0.5 * ((i + j) as f64 * 0.1).sin()
        });
        let phase = Array2::from_shape_fn((4, 64), |(i, j)| {
            motion_level * 0.3 * ((i + j) as f64 * 0.15).sin()
        });

        CsiData::builder()
            .amplitude(amplitude)
            .phase(phase)
            .frequency(5.0e9)
            .bandwidth(20.0e6)
            .snr(25.0)
            .build()
            .unwrap()
    }

    fn create_test_features(motion_level: f64) -> CsiFeatures {
        let csi_data = create_test_csi_data(motion_level);
        let extractor = FeatureExtractor::default_config();
        extractor.extract(&csi_data)
    }

    #[test]
    fn test_motion_score() {
        let score = MotionScore::new(0.5, 0.6, 0.4, None);
        assert!(score.total > 0.0 && score.total <= 1.0);
        assert_eq!(score.variance_component, 0.5);
        assert_eq!(score.correlation_component, 0.6);
        assert_eq!(score.phase_component, 0.4);
    }

    #[test]
    fn test_motion_score_with_doppler() {
        let score = MotionScore::new(0.5, 0.6, 0.4, Some(0.7));
        assert!(score.total > 0.0 && score.total <= 1.0);
        assert_eq!(score.doppler_component, Some(0.7));
    }

    #[test]
    fn test_motion_detector_creation() {
        let config = MotionDetectorConfig::default();
        let detector = MotionDetector::new(config);
        assert_eq!(detector.previous_confidence(), 0.0);
    }

    #[test]
    fn test_motion_analysis() {
        let detector = MotionDetector::default_config();
        let features = create_test_features(0.5);

        let analysis = detector.analyze_motion(&features);
        assert!(analysis.score.total >= 0.0 && analysis.score.total <= 1.0);
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
    }

    #[test]
    fn test_human_detection() {
        let config = MotionDetectorConfig::builder()
            .human_detection_threshold(0.5)
            .smoothing_factor(0.5)
            .build();
        let mut detector = MotionDetector::new(config);

        let features = create_test_features(0.8);
        let result = detector.detect_human(&features);

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.motion_score >= 0.0 && result.motion_score <= 1.0);
    }

    #[test]
    fn test_temporal_smoothing() {
        let config = MotionDetectorConfig::builder()
            .smoothing_factor(0.9)
            .build();
        let mut detector = MotionDetector::new(config);

        // First detection with low confidence
        let features_low = create_test_features(0.1);
        let result1 = detector.detect_human(&features_low);

        // Second detection with high confidence should be smoothed
        let features_high = create_test_features(0.9);
        let result2 = detector.detect_human(&features_high);

        // Due to smoothing, result2.confidence should be between result1 and raw
        assert!(result2.confidence >= result1.confidence);
    }

    #[test]
    fn test_calibration() {
        let mut detector = MotionDetector::default_config();
        let features = create_test_features(0.5);

        assert!(!detector.get_statistics().is_calibrated);
        detector.calibrate(&features);
        assert!(detector.get_statistics().is_calibrated);

        detector.clear_calibration();
        assert!(!detector.get_statistics().is_calibrated);
    }

    #[test]
    fn test_detection_statistics() {
        let mut detector = MotionDetector::default_config();

        for i in 0..5 {
            let features = create_test_features((i as f64) / 5.0);
            let _ = detector.detect_human(&features);
        }

        let stats = detector.get_statistics();
        assert_eq!(stats.total_detections, 5);
        assert!(stats.detection_rate >= 0.0 && stats.detection_rate <= 1.0);
    }

    #[test]
    fn test_reset() {
        let mut detector = MotionDetector::default_config();
        let features = create_test_features(0.5);

        for _ in 0..5 {
            let _ = detector.detect_human(&features);
        }

        detector.reset();

        let stats = detector.get_statistics();
        assert_eq!(stats.total_detections, 0);
        assert_eq!(stats.history_size, 0);
        assert_eq!(detector.previous_confidence(), 0.0);
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = MotionDetectorConfig::builder()
            .adaptive_threshold(true)
            .history_size(20)
            .build();
        let mut detector = MotionDetector::new(config);

        // Build up history
        for i in 0..15 {
            let features = create_test_features((i as f64 % 5.0) / 5.0);
            let _ = detector.detect_human(&features);
        }

        // The adaptive threshold should now be calculated
        let features = create_test_features(0.5);
        let result = detector.detect_human(&features);

        // Threshold should be different from default
        // (this is a weak assertion, mainly checking it runs)
        assert!(result.threshold > 0.0);
    }

    #[test]
    fn test_config_builder() {
        let config = MotionDetectorConfig::builder()
            .human_detection_threshold(0.7)
            .motion_threshold(0.4)
            .smoothing_factor(0.85)
            .amplitude_threshold(0.15)
            .phase_threshold(0.08)
            .history_size(200)
            .adaptive_threshold(true)
            .weights(0.35, 0.35, 0.30)
            .build();

        assert_eq!(config.human_detection_threshold, 0.7);
        assert_eq!(config.motion_threshold, 0.4);
        assert_eq!(config.smoothing_factor, 0.85);
        assert_eq!(config.amplitude_threshold, 0.15);
        assert_eq!(config.phase_threshold, 0.08);
        assert_eq!(config.history_size, 200);
        assert!(config.adaptive_threshold);
        assert_eq!(config.amplitude_weight, 0.35);
        assert_eq!(config.phase_weight, 0.35);
        assert_eq!(config.motion_weight, 0.30);
    }

    #[test]
    fn test_low_motion_no_detection() {
        let config = MotionDetectorConfig::builder()
            .human_detection_threshold(0.8)
            .smoothing_factor(0.0) // No smoothing for clear test
            .build();
        let mut detector = MotionDetector::new(config);

        // Very low motion should not trigger detection
        let features = create_test_features(0.01);
        let result = detector.detect_human(&features);

        // With very low motion, detection should likely be false
        // (depends on thresholds, but confidence should be low)
        assert!(result.motion_score < 0.5);
    }

    #[test]
    fn test_motion_history() {
        let config = MotionDetectorConfig::builder().history_size(10).build();
        let mut detector = MotionDetector::new(config);

        for i in 0..15 {
            let features = create_test_features((i as f64) / 15.0);
            let _ = detector.detect_human(&features);
        }

        let stats = detector.get_statistics();
        assert_eq!(stats.history_size, 10); // Should not exceed max
    }

    // -- ADR-154 §7.4 #18: de-magic-constant + boundary characterization tests.
    // These pin CURRENT behaviour so a future retune is a visible, tested change.

    /// The de-magicked tuning consts MUST equal the prior bare literals exactly
    /// (this milestone is cleanup — operating values are unchanged).
    #[test]
    fn motion_tuning_consts_unchanged_from_literals() {
        assert_eq!(MOTION_WEIGHTS_WITH_DOPPLER, (0.3, 0.2, 0.2, 0.3));
        assert_eq!(MOTION_WEIGHTS_NO_DOPPLER, (0.4, 0.3, 0.3));
        assert_eq!(DOPPLER_FULL_SCALE_MAGNITUDE, 100.0);
        assert_eq!(VARIANCE_HEURISTIC_FULL_SCALE, 0.5);
        assert_eq!(PHASE_VARIANCE_FULL_SCALE, 0.5);
        assert_eq!(PHASE_SCORE_VARIANCE_WEIGHT, 0.5);
        assert_eq!(AMP_QUALITY_FULL_SCALE_RANGE, 2.0);
        assert_eq!(CONF_WEIGHT_AMPLITUDE, 0.3);
        assert_eq!(CONF_WEIGHT_PHASE, 0.3);
        assert_eq!(CONF_WEIGHT_CORRELATION, 0.2);
        assert_eq!(CONF_WEIGHT_DOPPLER, 0.2);
        assert_eq!(BASELINE_VARIANCE_FLOOR, 1e-10);
        assert_eq!(ADAPTIVE_THRESHOLD_MIN, 0.3);
        assert_eq!(ADAPTIVE_THRESHOLD_MAX, 0.95);
        assert_eq!(ADAPTIVE_THRESHOLD_MIN_HISTORY, 10);
        // Fusion weights are a convex combination (sum to 1.0).
        let (wv, wc, wp, wd) = MOTION_WEIGHTS_WITH_DOPPLER;
        assert!((wv + wc + wp + wd - 1.0).abs() < 1e-12);
        let (wv, wc, wp) = MOTION_WEIGHTS_NO_DOPPLER;
        assert!((wv + wc + wp - 1.0).abs() < 1e-12);
    }

    /// Doppler component saturates at full scale (`/100.0` then clamp(0,1)).
    /// Pins behaviour at/just-below/just-above the full-scale magnitude.
    #[test]
    fn doppler_component_saturates_at_full_scale() {
        use crate::features::DopplerFeatures;
        use ndarray::Array1;
        let make = |mag: f64| DopplerFeatures {
            shifts: Array1::zeros(1),
            peak_frequency: 0.0,
            mean_magnitude: mag,
            spread: 0.0,
        };
        let detector = MotionDetector::default_config();
        // just below full scale -> < 1.0
        let mut features = create_test_features(0.5);
        features.doppler = Some(make(DOPPLER_FULL_SCALE_MAGNITUDE - 1.0));
        let below = detector.analyze_motion(&features).score.doppler_component.unwrap();
        assert!(below < 1.0 && below > 0.98);
        // exactly full scale -> 1.0
        features.doppler = Some(make(DOPPLER_FULL_SCALE_MAGNITUDE));
        let at = detector.analyze_motion(&features).score.doppler_component.unwrap();
        assert_eq!(at, 1.0);
        // above full scale -> clamped to 1.0
        features.doppler = Some(make(DOPPLER_FULL_SCALE_MAGNITUDE * 10.0));
        let above = detector.analyze_motion(&features).score.doppler_component.unwrap();
        assert_eq!(above, 1.0);
    }

    /// `calculate_correlation_score` returns 0.0 for n<2 (the small-matrix
    /// guard) and a finite, clamped value for n>=2. Pins the n=1 boundary.
    #[test]
    fn correlation_score_zero_below_n2_boundary() {
        use crate::features::CorrelationFeatures;
        use ndarray::Array2;
        let detector = MotionDetector::default_config();
        let one = CorrelationFeatures {
            matrix: Array2::from_elem((1, 1), 1.0),
            mean_correlation: 0.0,
            max_correlation: 0.0,
            correlation_spread: 0.0,
        };
        assert_eq!(detector.calculate_correlation_score(&one), 0.0);
        let two = CorrelationFeatures {
            matrix: Array2::from_shape_fn((2, 2), |(i, j)| if i == j { 1.0 } else { 0.0 }),
            mean_correlation: 0.0,
            max_correlation: 0.0,
            correlation_spread: 0.0,
        };
        let s = detector.calculate_correlation_score(&two);
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }

    /// `calculate_temporal_variance` returns 0.0 with fewer than 2 history
    /// entries, finite otherwise. Pins the len<2 boundary.
    #[test]
    fn temporal_variance_zero_below_two_history() {
        let mut detector = MotionDetector::default_config();
        assert_eq!(detector.calculate_temporal_variance(), 0.0); // 0 entries
        detector
            .motion_history
            .push_back(MotionScore::new(0.5, 0.5, 0.5, None));
        assert_eq!(detector.calculate_temporal_variance(), 0.0); // 1 entry
        detector
            .motion_history
            .push_back(MotionScore::new(0.1, 0.1, 0.1, None));
        assert!(detector.calculate_temporal_variance() > 0.0); // 2 entries
    }

    /// The adaptive threshold engages only at/after `ADAPTIVE_THRESHOLD_MIN_HISTORY`
    /// history entries; below it falls back to the configured fixed threshold.
    /// Pins the history=9 (fixed) vs history=10 (adaptive) boundary.
    #[test]
    fn adaptive_threshold_engages_at_history_boundary() {
        let config = MotionDetectorConfig::builder()
            .adaptive_threshold(true)
            .human_detection_threshold(0.8)
            .history_size(50)
            .build();
        let mut detector = MotionDetector::new(config);
        // Push exactly 9 entries: still uses the fixed configured threshold.
        for _ in 0..(ADAPTIVE_THRESHOLD_MIN_HISTORY - 1) {
            detector
                .motion_history
                .push_back(MotionScore::new(0.5, 0.5, 0.5, None));
        }
        assert_eq!(detector.calculate_adaptive_threshold(), 0.8);
        // 10th entry: adaptive band kicks in, clamped to [MIN, MAX].
        detector
            .motion_history
            .push_back(MotionScore::new(0.5, 0.5, 0.5, None));
        let t = detector.calculate_adaptive_threshold();
        assert!((ADAPTIVE_THRESHOLD_MIN..=ADAPTIVE_THRESHOLD_MAX).contains(&t));
    }
}

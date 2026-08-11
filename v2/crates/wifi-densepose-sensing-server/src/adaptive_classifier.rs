//! Adaptive CSI Activity Classifier
//!
//! Learns environment-specific classification thresholds from labeled JSONL
//! recordings.  Uses a lightweight approach:
//!
//! 1. **Feature statistics**: per-class mean/stddev for each server-published CSI feature
//! 2. **Mahalanobis-like distance**: weighted distance to each class centroid
//! 3. **Logistic regression weights**: learned via gradient descent on the
//!    labeled data for fine-grained boundary tuning
//!
//! The trained model is serialised as JSON and hot-loaded at runtime so that
//! the classification thresholds adapt to the specific room and ESP32 placement.
//!
//! Classes are discovered dynamically from training data filenames instead of
//! being hardcoded, so new activity classes can be added just by recording data
//! with the appropriate filename convention.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── Feature vector ───────────────────────────────────────────────────────────

/// Instantaneous feature vector published identically to the recorder and runtime.
/// Raw amplitudes are intentionally excluded: recorders may omit/cap them while the
/// runtime sees full internal vectors, which previously caused train/serve skew.
const N_BASE_FEATURES: usize = 7;
const TEMPORAL_FEATURE_INDICES: [usize; N_BASE_FEATURES] = [0, 1, 2, 3, 4, 5, 6];
const N_TEMPORAL_FEATURES: usize = TEMPORAL_FEATURE_INDICES.len();
/// Five-second patch: current features + velocity + standard deviation.
const N_FEATURES: usize = N_BASE_FEATURES + N_TEMPORAL_FEATURES * 2;
const ADAPTIVE_FEATURE_SCHEMA_VERSION: u32 = 4;
const MIN_RUNTIME_BALANCED_ACCURACY: f64 = 0.90;
const MIN_RUNTIME_CLASS_RECALL: f64 = 0.90;
const TEMPORAL_BUCKET_SECONDS: f64 = 1.0;
const TEMPORAL_WINDOW_BUCKETS: i64 = 5;
const TEMPORAL_MIN_CONTEXT_BUCKETS: usize = 3;

/// Default class names for backward compatibility with old saved models.
const DEFAULT_CLASSES: &[&str] = &["absent", "present"];

/// Extract the server-published feature vector from a JSONL frame.
pub fn features_from_frame(frame: &serde_json::Value) -> [f64; N_BASE_FEATURES] {
    // spatial-intelligence stores the original RuView envelope under
    // `payload`; RuView's own recorder writes the envelope at the root. Keep
    // one feature contract so offline evaluation cannot silently turn every
    // wrapped feature into zero.
    let frame = frame.get("payload").unwrap_or(frame);
    let feat = frame
        .get("features")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    features_from_runtime(&feat, &[])
}

/// Runtime equivalent of [`features_from_frame`]. The amplitude argument remains for
/// source compatibility but is excluded from schema v4 to guarantee train/serve parity.
pub fn features_from_runtime(
    feat: &serde_json::Value,
    _amplitudes: &[f64],
) -> [f64; N_BASE_FEATURES] {
    let variance = feat.get("variance").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let mbp = feat
        .get("motion_band_power")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let bbp = feat
        .get("breathing_band_power")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let sp = feat
        .get("spectral_power")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let df = feat
        .get("dominant_freq_hz")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let cp = feat
        .get("change_points")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let rssi = feat
        .get("mean_rssi")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    [variance, mbp, bbp, sp, df, cp, rssi]
}

/// Rate-invariant short temporal patch shared by offline fitting and runtime.
///
/// RuView can process ESP32 frames around 30 Hz while the product recorder
/// polls `/latest` at 1 Hz. Keeping one latest observation per wall-clock
/// second prevents sample rate from becoming an accidental class label.
#[derive(Debug, Default)]
pub struct AdaptiveFeatureExtractor {
    history: VecDeque<(i64, [f64; N_BASE_FEATURES])>,
    runtime_origin: Option<Instant>,
}

impl AdaptiveFeatureExtractor {
    /// Add one observation at a monotonic/epoch timestamp in seconds.
    pub fn observe_at(
        &mut self,
        base: [f64; N_BASE_FEATURES],
        timestamp_seconds: f64,
    ) -> Option<[f64; N_FEATURES]> {
        if !timestamp_seconds.is_finite() || base.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let bucket = (timestamp_seconds / TEMPORAL_BUCKET_SECONDS).floor() as i64;
        match self.history.back_mut() {
            Some((last_bucket, last_base)) if *last_bucket == bucket => {
                *last_base = base;
            }
            Some((last_bucket, _)) if bucket < *last_bucket => {
                self.history.clear();
                self.history.push_back((bucket, base));
            }
            Some((last_bucket, _)) => {
                if bucket - *last_bucket >= TEMPORAL_WINDOW_BUCKETS {
                    self.history.clear();
                }
                self.history.push_back((bucket, base));
            }
            None => self.history.push_back((bucket, base)),
        }

        while self
            .history
            .front()
            .is_some_and(|(oldest, _)| bucket - *oldest >= TEMPORAL_WINDOW_BUCKETS)
        {
            self.history.pop_front();
        }

        if self.history.len() < TEMPORAL_MIN_CONTEXT_BUCKETS {
            return None;
        }
        let span = self.history.back()?.0 - self.history.front()?.0;
        if span < (TEMPORAL_MIN_CONTEXT_BUCKETS - 1) as i64 {
            return None;
        }

        let mut output = [0.0; N_FEATURES];
        output[..N_BASE_FEATURES].copy_from_slice(&base);
        for (temporal_index, feature_index) in TEMPORAL_FEATURE_INDICES.iter().copied().enumerate()
        {
            let mean = self
                .history
                .iter()
                .map(|(_, values)| values[feature_index])
                .sum::<f64>()
                / self.history.len() as f64;
            let variance = self
                .history
                .iter()
                .map(|(_, values)| (values[feature_index] - mean).powi(2))
                .sum::<f64>()
                / self.history.len() as f64;
            let mean_abs_velocity = self
                .history
                .iter()
                .zip(self.history.iter().skip(1))
                .map(|((before_bucket, before), (after_bucket, after))| {
                    (after[feature_index] - before[feature_index]).abs()
                        / (*after_bucket - *before_bucket) as f64
                })
                .sum::<f64>()
                / (self.history.len() - 1) as f64;
            output[N_BASE_FEATURES + temporal_index] = mean_abs_velocity;
            output[N_BASE_FEATURES + N_TEMPORAL_FEATURES + temporal_index] = variance.sqrt();
        }
        Some(output)
    }

    /// Add a live observation using a process-local monotonic clock.
    pub fn observe_runtime(
        &mut self,
        feature_json: &serde_json::Value,
        amplitudes: &[f64],
    ) -> Option<[f64; N_FEATURES]> {
        let origin = *self.runtime_origin.get_or_insert_with(Instant::now);
        self.observe_at(
            features_from_runtime(feature_json, amplitudes),
            origin.elapsed().as_secs_f64(),
        )
    }

    /// A newly loaded model must never inherit temporal context from the old one.
    pub fn reset(&mut self) {
        self.history.clear();
        self.runtime_origin = None;
    }
}

/// Merge the learned presence stage with the high-rate heuristic motion stage.
pub fn reconcile_presence_prediction(
    learned_label: &str,
    heuristic_motion: &str,
) -> (String, bool) {
    match learned_label {
        "absent" => ("absent".to_string(), false),
        "present" => {
            let motion = if heuristic_motion == "absent" {
                "present_still"
            } else {
                heuristic_motion
            };
            (motion.to_string(), true)
        }
        // Preserve compatibility with explicit activity classes in externally
        // produced models while the built-in trainer remains hierarchical.
        other => (other.to_string(), other != "absent"),
    }
}

/// Debounces learned presence probabilities before they can change room state.
#[derive(Debug, Default)]
pub struct AdaptivePresenceSmoother {
    present_score: Option<f64>,
    present: Option<bool>,
}

impl AdaptivePresenceSmoother {
    pub fn update(&mut self, learned_label: &str, confidence: f64) -> bool {
        let confidence = if confidence.is_finite() {
            confidence.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let observation = if learned_label == "absent" {
            1.0 - confidence
        } else {
            confidence
        };
        let score = self
            .present_score
            .map(|previous| previous * 0.9 + observation * 0.1)
            .unwrap_or(observation);
        self.present_score = Some(score);
        let present = match self.present {
            Some(true) => score >= 0.35,
            Some(false) => score > 0.65,
            None => score >= 0.5,
        };
        self.present = Some(present);
        present
    }

    pub fn is_present(&self) -> Option<bool> {
        self.present
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ── Per-class statistics ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassStats {
    pub label: String,
    pub count: usize,
    pub mean: [f64; N_FEATURES],
    pub stddev: [f64; N_FEATURES],
}

// ── Trained model ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveModel {
    /// Per-class feature statistics (centroid + spread).
    pub class_stats: Vec<ClassStats>,
    /// Logistic regression weights: [n_classes x (N_FEATURES + 1)] (last = bias).
    /// Dynamic: the outer Vec length equals the number of discovered classes.
    pub weights: Vec<Vec<f64>>,
    /// Global feature normalisation: mean and stddev across all training data.
    pub global_mean: [f64; N_FEATURES],
    pub global_std: [f64; N_FEATURES],
    /// Training metadata.
    pub trained_frames: usize,
    pub training_accuracy: f64,
    pub version: u32,
    /// Dynamically discovered class names (in index order).
    #[serde(default = "default_class_names")]
    pub class_names: Vec<String>,
    /// Sensor topology this local specialist was fitted for.
    #[serde(default)]
    pub expected_node_count: Option<usize>,
    /// Session-held-out evaluation. Models saved before schema v2 do not have
    /// this field and are deliberately ineligible for automatic activation.
    #[serde(default)]
    pub validation: Option<ValidationMetrics>,
}

/// Leakage-resistant evaluation metadata for the runtime activation gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetrics {
    /// Recording sessions excluded from fitting and used only for evaluation.
    pub held_out_recordings: usize,
    /// Frames contained in the held-out sessions.
    pub held_out_frames: usize,
    /// Macro-average recall, so a dominant class cannot hide a failed class.
    pub balanced_accuracy: f64,
    /// Recall for each class name.
    pub per_class_recall: HashMap<String, f64>,
}

/// Backward-compatible fallback for models saved without class_names.
fn default_class_names() -> Vec<String> {
    DEFAULT_CLASSES.iter().map(|s| s.to_string()).collect()
}

impl Default for AdaptiveModel {
    fn default() -> Self {
        let n_classes = DEFAULT_CLASSES.len();
        Self {
            class_stats: Vec::new(),
            weights: vec![vec![0.0; N_FEATURES + 1]; n_classes],
            global_mean: [0.0; N_FEATURES],
            global_std: [1.0; N_FEATURES],
            trained_frames: 0,
            training_accuracy: 0.0,
            version: ADAPTIVE_FEATURE_SCHEMA_VERSION,
            class_names: default_class_names(),
            expected_node_count: None,
            validation: None,
        }
    }
}

impl AdaptiveModel {
    /// Whether this model has enough independent evidence to override the
    /// conservative threshold classifier in production.
    pub fn runtime_eligibility(&self) -> Result<(), String> {
        if self.version != ADAPTIVE_FEATURE_SCHEMA_VERSION {
            return Err(format!(
                "model feature schema v{} is stale; runtime requires v{}",
                self.version, ADAPTIVE_FEATURE_SCHEMA_VERSION
            ));
        }
        let metrics = self.validation.as_ref().ok_or_else(|| {
            "model has no session-held-out validation; automatic activation is unsafe".to_string()
        })?;
        let n_classes = self.class_names.len();
        if n_classes < 2 {
            return Err("model needs at least two classes".into());
        }
        if self.trained_frames == 0 {
            return Err("model contains no fitted frames".into());
        }
        if self.expected_node_count == Some(0) {
            return Err("model node topology cannot be zero".into());
        }
        if self.weights.len() != n_classes
            || self
                .weights
                .iter()
                .any(|row| row.len() != N_FEATURES + 1 || row.iter().any(|v| !v.is_finite()))
        {
            return Err("model weight shape or values do not match its classes".into());
        }
        if self.class_stats.len() != n_classes
            || self.global_mean.iter().any(|v| !v.is_finite())
            || self.global_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("model normalization or class statistics are invalid".into());
        }
        if metrics.held_out_recordings < n_classes * 2 {
            return Err(format!(
                "held-out validation needs at least two independent recordings per class (have {}, need {})",
                metrics.held_out_recordings,
                n_classes * 2
            ));
        }
        if metrics.held_out_frames == 0 {
            return Err("held-out validation contains no frames".into());
        }

        if !metrics.balanced_accuracy.is_finite()
            || metrics.balanced_accuracy < MIN_RUNTIME_BALANCED_ACCURACY
        {
            return Err(format!(
                "held-out balanced accuracy {:.3} is below activation floor {:.3}",
                metrics.balanced_accuracy, MIN_RUNTIME_BALANCED_ACCURACY
            ));
        }

        for class_name in &self.class_names {
            let recall = metrics
                .per_class_recall
                .get(class_name)
                .copied()
                .ok_or_else(|| format!("held-out recall missing for class '{class_name}'"))?;
            if !recall.is_finite() || recall < MIN_RUNTIME_CLASS_RECALL {
                return Err(format!(
                    "held-out recall for class '{class_name}' is {recall:.3}, below {MIN_RUNTIME_CLASS_RECALL:.3}"
                ));
            }
        }
        Ok(())
    }

    /// Classify a raw feature vector.  Returns (class_label, confidence).
    pub fn classify(&self, raw_features: &[f64; N_FEATURES]) -> (String, f64) {
        let n_classes = self.weights.len();
        if n_classes == 0 || self.class_stats.is_empty() {
            return ("present_still".to_string(), 0.5);
        }

        // Normalise features.
        let mut x = [0.0f64; N_FEATURES];
        for i in 0..N_FEATURES {
            x[i] = (raw_features[i] - self.global_mean[i]) / (self.global_std[i] + 1e-9);
        }

        // Compute logits: w·x + b for each class.
        let logits: Vec<f64> = (0..n_classes)
            .map(|c| {
                let w = &self.weights[c];
                w[N_FEATURES]
                    + w[..N_FEATURES]
                        .iter()
                        .zip(x.iter())
                        .map(|(&wi, &xi)| wi * xi)
                        .sum::<f64>()
            })
            .collect();

        // Softmax.
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|z| (z - max_logit).exp()).sum();
        let mut probs: Vec<f64> = vec![0.0; n_classes];
        for c in 0..n_classes {
            probs[c] = ((logits[c] - max_logit).exp()) / exp_sum;
        }

        // Pick argmax. Same NaN-panic class as #611: if any raw_feature is NaN
        // it propagates through normalize → logits → softmax, then partial_cmp
        // returns None and unwrap() panics the sensing server on every frame.
        let (best_c, best_p) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let label = if best_c < self.class_names.len() {
            self.class_names[best_c].clone()
        } else {
            "present_still".to_string()
        };
        (label, *best_p)
    }

    /// Save model to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load model from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }

    /// Load a persisted model only when its independent-session evidence is
    /// strong enough for automatic runtime activation.
    pub fn load_runtime(path: &Path) -> Result<Self, String> {
        let model = Self::load(path).map_err(|error| error.to_string())?;
        model.runtime_eligibility()?;
        Ok(model)
    }
}

// ── Training ─────────────────────────────────────────────────────────────────

/// A labeled training sample.
#[derive(Clone)]
struct Sample {
    features: [f64; N_FEATURES],
    class_idx: usize,
}

/// Frames captured during one continuous physical session.
struct Recording {
    name: String,
    class_idx: usize,
    samples: Vec<Sample>,
}

/// Load JSONL recording frames and assign a class label based on filename.
fn load_recording(path: &Path, class_idx: usize) -> (Vec<Sample>, Option<usize>) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return (Vec::new(), None),
    };
    let mut temporal = AdaptiveFeatureExtractor::default();
    let mut samples = Vec::new();
    let mut topology_counts: HashMap<usize, usize> = HashMap::new();
    for (index, line) in content.lines().enumerate() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let envelope = v.get("payload").unwrap_or(&v);
        if let Some(node_count) = envelope
            .get("nodes")
            .and_then(|nodes| nodes.as_array())
            .map(Vec::len)
            .filter(|count| *count > 0)
        {
            *topology_counts.entry(node_count).or_default() += 1;
        }
        let timestamp_seconds = envelope
            .get("timestamp")
            .and_then(|value| value.as_f64())
            .or_else(|| {
                v.get("captured_at")
                    .and_then(|value| value.as_str())
                    .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
                    .map(|value| value.timestamp_millis() as f64 / 1000.0)
            })
            // Old recordings without timestamps remain usable, but their
            // ordering is made explicit as one sample per second.
            .unwrap_or(index as f64);
        if let Some(features) = temporal.observe_at(features_from_frame(&v), timestamp_seconds) {
            samples.push(Sample {
                features,
                class_idx,
            });
        }
    }
    let node_count = topology_counts
        .into_iter()
        .max_by_key(|(node_count, observations)| (*observations, *node_count))
        .map(|(node_count, _)| node_count);
    (samples, node_count)
}

/// Map a recording filename to a class name (String).
/// Returns the discovered class name for the file, or None if it cannot be determined.
fn classify_recording_name(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    // Strip "train_" prefix and ".jsonl" suffix, then extract the class label.
    // Convention: train_<class>_<description>.jsonl
    // The class is the first segment after "train_" that matches a known pattern,
    // or the entire middle portion if no pattern matches.

    // Check common patterns first for backward compat
    if lower.contains("empty") || lower.contains("absent") {
        return Some("absent".into());
    }
    if lower.contains("still")
        || lower.contains("sitting")
        || lower.contains("standing")
        || lower.contains("walking")
        || lower.contains("moving")
        || lower.contains("active")
        || lower.contains("exercise")
        || lower.contains("running")
    {
        // Presence is the safety-critical first stage. Motion granularity stays
        // with the high-rate heuristic until each activity has enough truly
        // independent sessions to support its own held-out classifier.
        return Some("present".into());
    }

    // Fallback: extract class from filename structure train_<class>_*.jsonl
    let stem = lower
        .trim_start_matches("train_")
        .trim_end_matches(".jsonl");
    let class_name = stem.split('_').next().unwrap_or(stem);
    if !class_name.is_empty() {
        Some(class_name.to_string())
    } else {
        None
    }
}

/// Train a model from labeled JSONL recordings in a directory.
///
/// Recordings are matched to classes by filename pattern. Classes are discovered
/// dynamically from the training data filenames:
/// - `*empty*` / `*absent*`   → absent
/// - occupied still/walking/active patterns → present
/// - Any other `train_<class>_*.jsonl` → <class>
pub fn train_from_recordings(recordings_dir: &Path) -> Result<AdaptiveModel, String> {
    train_from_recordings_for_node_count(recordings_dir, None)
}

/// Train only from sessions matching the active sensor topology when known.
pub fn train_from_recordings_for_node_count(
    recordings_dir: &Path,
    expected_node_count: Option<usize>,
) -> Result<AdaptiveModel, String> {
    // First pass: scan filenames to discover all unique class names.
    let entries: Vec<_> = std::fs::read_dir(recordings_dir)
        .map_err(|e| format!("Cannot read {}: {}", recordings_dir.display(), e))?
        .flatten()
        .collect();

    // Collect (entry, class_name) pairs for files that match.
    let mut file_classes: Vec<(PathBuf, String, String)> = Vec::new(); // (path, fname, class_name)
    for entry in &entries {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.starts_with("train_") || !fname.ends_with(".jsonl") {
            continue;
        }
        if let Some(class_name) = classify_recording_name(&fname) {
            file_classes.push((entry.path(), fname, class_name));
        }
    }

    if file_classes.is_empty() {
        return Err("No training samples found. Record data with train_* prefix.".into());
    }

    // Stable ordering makes class indices, folds, and fitting reproducible
    // across filesystems whose read_dir order differs.
    file_classes.sort_by(|a, b| a.1.cmp(&b.1));
    let mut class_names: Vec<String> = file_classes
        .iter()
        .map(|(_, _, class_name)| class_name.clone())
        .collect();
    class_names.sort();
    class_names.dedup();
    let class_map: HashMap<String, usize> = class_names
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, name)| (name, index))
        .collect();
    let n_classes = class_names.len();

    let mut recordings = Vec::with_capacity(file_classes.len());
    let mut recordings_per_class: HashMap<String, usize> = HashMap::new();
    for (path, fname, class_name) in file_classes {
        let class_idx = class_map[&class_name];
        let (samples, observed_node_count) = load_recording(&path, class_idx);
        if samples.is_empty() {
            return Err(format!(
                "Training recording '{fname}' contains no valid frames"
            ));
        }
        if expected_node_count.is_some() && observed_node_count != expected_node_count {
            eprintln!(
                "  Skipped {}: recorded topology {:?}, active topology {:?}",
                fname, observed_node_count, expected_node_count
            );
            continue;
        }
        eprintln!(
            "  Loaded {}: {} frames → class '{}'",
            fname,
            samples.len(),
            class_name
        );
        *recordings_per_class.entry(class_name.clone()).or_default() += 1;
        recordings.push(Recording {
            name: fname,
            class_idx,
            samples,
        });
    }

    // Adjacent CSI frames are highly autocorrelated. Leave-one-session-out
    // (LOSO) ensures every reported prediction comes from a model that never
    // saw that physical session, instead of validating on a favorable tail.
    let can_validate_sessions = class_names
        .iter()
        .all(|name| recordings_per_class.get(name).copied().unwrap_or(0) >= 2);

    let validation = if can_validate_sessions {
        let mut class_correct = vec![0usize; n_classes];
        let mut class_total = vec![0usize; n_classes];
        for held_out_index in 0..recordings.len() {
            let training_recordings: Vec<&Recording> = recordings
                .iter()
                .enumerate()
                .filter_map(|(index, recording)| (index != held_out_index).then_some(recording))
                .collect();
            let fold_model = fit_model(
                &training_recordings,
                &class_names,
                expected_node_count,
                false,
            )?;
            let held_out = &recordings[held_out_index];
            eprintln!("  LOSO held out '{}'", held_out.name);
            for sample in &held_out.samples {
                class_total[sample.class_idx] += 1;
                let (predicted, _) = fold_model.classify(&sample.features);
                if predicted == class_names[sample.class_idx] {
                    class_correct[sample.class_idx] += 1;
                }
            }
        }

        let per_class_recall: HashMap<String, f64> = class_names
            .iter()
            .enumerate()
            .map(|(class_idx, class_name)| {
                let recall = class_correct[class_idx] as f64 / class_total[class_idx] as f64;
                (class_name.clone(), recall)
            })
            .collect();
        let balanced_accuracy = per_class_recall.values().sum::<f64>() / n_classes as f64;
        let held_out_frames = class_total.iter().sum();
        eprintln!(
            "LOSO balanced accuracy: {:.1}% across {} recording(s) / {} frames",
            balanced_accuracy * 100.0,
            recordings.len(),
            held_out_frames
        );
        Some(ValidationMetrics {
            held_out_recordings: recordings.len(),
            held_out_frames,
            balanced_accuracy,
            per_class_recall,
        })
    } else {
        eprintln!(
            "Session-held-out validation unavailable: record at least two independent sessions per class"
        );
        None
    };

    // Validation models are disposable. The deployed model is refit on every
    // accepted session so scarce local calibration data is not wasted.
    let all_recordings: Vec<&Recording> = recordings.iter().collect();
    let mut model = fit_model(&all_recordings, &class_names, expected_node_count, true)?;
    model.validation = validation;
    Ok(model)
}

/// Fit one room-specific classifier from complete recording sessions.
fn fit_model(
    recordings: &[&Recording],
    class_names: &[String],
    expected_node_count: Option<usize>,
    print_progress: bool,
) -> Result<AdaptiveModel, String> {
    let n_classes = class_names.len();
    let n: usize = recordings
        .iter()
        .map(|recording| recording.samples.len())
        .sum();
    if n == 0 {
        return Err("No training samples found. Record data with train_* prefix.".into());
    }

    let mut sessions_per_class = vec![0usize; n_classes];
    for recording in recordings {
        sessions_per_class[recording.class_idx] += 1;
    }
    if let Some(class_idx) = sessions_per_class.iter().position(|count| *count == 0) {
        return Err(format!(
            "Training fold contains no '{}' session",
            class_names[class_idx]
        ));
    }

    // Give each class equal total mass and each session within that class equal
    // mass. Otherwise a ten-minute empty capture can dominate a short activity
    // capture even though its adjacent frames add little independent evidence.
    let weighted_samples: Vec<(&Sample, f64)> = recordings
        .iter()
        .flat_map(|recording| {
            let session_weight = 1.0
                / (sessions_per_class[recording.class_idx] as f64 * recording.samples.len() as f64);
            recording
                .samples
                .iter()
                .map(move |sample| (sample, session_weight))
        })
        .collect();
    let total_weight: f64 = weighted_samples.iter().map(|(_, weight)| weight).sum();
    if print_progress {
        eprintln!(
            "Total training samples: {n} across {n_classes} classes: {:?}",
            class_names
        );
    }

    // ── Compute global normalisation stats ──
    let mut global_mean = [0.0f64; N_FEATURES];
    let mut global_var = [0.0f64; N_FEATURES];
    for (sample, weight) in &weighted_samples {
        for (mean, feature) in global_mean.iter_mut().zip(sample.features.iter()) {
            *mean += feature * weight;
        }
    }
    for mean in &mut global_mean {
        *mean /= total_weight;
    }
    for (sample, weight) in &weighted_samples {
        for i in 0..N_FEATURES {
            global_var[i] += weight * (sample.features[i] - global_mean[i]).powi(2);
        }
    }
    let mut global_std = [0.0f64; N_FEATURES];
    for i in 0..N_FEATURES {
        global_std[i] = (global_var[i] / total_weight).sqrt().max(1e-9);
    }

    // ── Compute per-class statistics ──
    let mut class_sums = vec![[0.0f64; N_FEATURES]; n_classes];
    let mut class_sq = vec![[0.0f64; N_FEATURES]; n_classes];
    let mut class_counts = vec![0usize; n_classes];
    let mut class_weight = vec![0.0f64; n_classes];
    for (sample, weight) in &weighted_samples {
        let c = sample.class_idx;
        class_counts[c] += 1;
        class_weight[c] += weight;
        for i in 0..N_FEATURES {
            class_sums[c][i] += weight * sample.features[i];
            class_sq[c][i] += weight * sample.features[i] * sample.features[i];
        }
    }

    let mut class_stats = Vec::new();
    for c in 0..n_classes {
        let weight = class_weight[c];
        let mut mean = [0.0; N_FEATURES];
        let mut stddev = [0.0; N_FEATURES];
        for i in 0..N_FEATURES {
            mean[i] = class_sums[c][i] / weight;
            stddev[i] = ((class_sq[c][i] / weight) - mean[i] * mean[i])
                .max(0.0)
                .sqrt();
        }
        class_stats.push(ClassStats {
            label: class_names[c].clone(),
            count: class_counts[c],
            mean,
            stddev,
        });
    }

    // ── Normalise all samples ──
    let mut norm_samples: Vec<([f64; N_FEATURES], usize, f64)> = weighted_samples
        .iter()
        .map(|(sample, weight)| {
            let mut x = [0.0; N_FEATURES];
            for i in 0..N_FEATURES {
                x[i] = (sample.features[i] - global_mean[i]) / (global_std[i] + 1e-9);
            }
            (x, sample.class_idx, *weight)
        })
        .collect();

    // ── Train logistic regression via mini-batch SGD ──
    let mut weights: Vec<Vec<f64>> = vec![vec![0.0f64; N_FEATURES + 1]; n_classes];
    let lr = 0.1;
    let epochs = 200;
    let batch_size = 32;

    // Shuffle helper (simple LCG for determinism).
    let mut rng_state: u64 = 42;
    let mut rng_next = move || -> u64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng_state >> 33
    };

    for epoch in 0..epochs {
        // Shuffle samples.
        for i in (1..norm_samples.len()).rev() {
            let j = (rng_next() as usize) % (i + 1);
            norm_samples.swap(i, j);
        }

        let mut epoch_loss = 0.0f64;

        for batch_start in (0..norm_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(norm_samples.len());
            let batch = &norm_samples[batch_start..batch_end];

            // Accumulate gradients.
            let mut grad: Vec<Vec<f64>> = vec![vec![0.0f64; N_FEATURES + 1]; n_classes];

            let batch_weight: f64 = batch.iter().map(|(_, _, weight)| weight).sum();
            for (x, target, sample_weight) in batch {
                // Forward: softmax.
                let mut logits: Vec<f64> = vec![0.0; n_classes];
                for (c, logit) in logits.iter_mut().enumerate() {
                    *logit = weights[c][N_FEATURES]; // bias
                    *logit += weights[c][..N_FEATURES]
                        .iter()
                        .zip(x.iter())
                        .map(|(&w, &xi)| w * xi)
                        .sum::<f64>();
                }
                let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = logits.iter().map(|z| (z - max_l).exp()).sum();
                let mut probs: Vec<f64> = vec![0.0; n_classes];
                for c in 0..n_classes {
                    probs[c] = ((logits[c] - max_l).exp()) / exp_sum;
                }

                // Cross-entropy loss.
                epoch_loss += sample_weight * -(probs[*target].max(1e-15)).ln();

                // Gradient: prob - one_hot(target).
                for c in 0..n_classes {
                    let delta = sample_weight * (probs[c] - if c == *target { 1.0 } else { 0.0 });
                    for (g, &xi) in grad[c][..N_FEATURES].iter_mut().zip(x.iter()) {
                        *g += delta * xi;
                    }
                    grad[c][N_FEATURES] += delta; // bias grad
                }
            }

            // Update weights.
            let current_lr = lr * (1.0 - epoch as f64 / epochs as f64); // linear decay
            for c in 0..n_classes {
                for i in 0..=N_FEATURES {
                    weights[c][i] -= current_lr * grad[c][i] / batch_weight;
                }
            }
        }

        if print_progress && (epoch % 50 == 0 || epoch == epochs - 1) {
            let avg_loss = epoch_loss / total_weight;
            eprintln!("  Epoch {epoch:3}: loss = {avg_loss:.4}");
        }
    }

    // ── Evaluate accuracy ──
    let compute_logits = |x: &[f64]| -> Vec<f64> {
        (0..n_classes)
            .map(|c| {
                weights[c][N_FEATURES]
                    + weights[c][..N_FEATURES]
                        .iter()
                        .zip(x.iter())
                        .map(|(&w, &xi)| w * xi)
                        .sum::<f64>()
            })
            .collect()
    };
    let mut correct = 0;
    for (x, target, _) in &norm_samples {
        let logits = compute_logits(x);
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;
        if pred == *target {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / n as f64;
    if print_progress {
        eprintln!(
            "Training accuracy: {correct}/{n} = {:.1}%",
            accuracy * 100.0
        );
    }

    // ── Per-class accuracy ──
    let mut class_correct = vec![0usize; n_classes];
    let mut class_total = vec![0usize; n_classes];
    for (x, target, _) in &norm_samples {
        class_total[*target] += 1;
        let logits = compute_logits(x);
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;
        if pred == *target {
            class_correct[*target] += 1;
        }
    }
    if print_progress {
        for c in 0..n_classes {
            let tot = class_total[c].max(1);
            eprintln!(
                "  {}: {}/{} ({:.0}%)",
                class_names[c],
                class_correct[c],
                tot,
                class_correct[c] as f64 / tot as f64 * 100.0
            );
        }
    }

    Ok(AdaptiveModel {
        class_stats,
        weights,
        global_mean,
        global_std,
        trained_frames: n,
        training_accuracy: accuracy,
        version: ADAPTIVE_FEATURE_SCHEMA_VERSION,
        class_names: class_names.to_vec(),
        expected_node_count,
        validation: None,
    })
}

fn model_path_from(configured: Option<std::ffi::OsString>) -> PathBuf {
    configured
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/adaptive_model.json"))
}

/// Default path for the saved adaptive model, overridable for private products.
pub fn model_path() -> PathBuf {
    model_path_from(std::env::var_os("RUVIEW_ADAPTIVE_MODEL_PATH"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn model_with_validation(balanced_accuracy: f64, recalls: &[(&str, f64)]) -> AdaptiveModel {
        let mut model = AdaptiveModel::default();
        model.version = 4;
        model.class_names = recalls
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect();
        model.trained_frames = 800;
        model.weights = vec![vec![0.0; N_FEATURES + 1]; recalls.len()];
        model.class_stats = recalls
            .iter()
            .map(|(name, _)| ClassStats {
                label: (*name).to_string(),
                count: 400,
                mean: [0.0; N_FEATURES],
                stddev: [1.0; N_FEATURES],
            })
            .collect();
        model.validation = Some(ValidationMetrics {
            held_out_recordings: recalls.len() * 2,
            held_out_frames: 400,
            balanced_accuracy,
            per_class_recall: recalls
                .iter()
                .map(|(name, recall)| ((*name).to_string(), *recall))
                .collect(),
        });
        model
    }

    #[test]
    fn legacy_model_without_session_validation_is_not_runtime_eligible() {
        let mut model = AdaptiveModel::default();
        model.training_accuracy = 1.0;

        let error = model.runtime_eligibility().unwrap_err();

        assert!(error.contains("held-out"), "unexpected error: {error}");
    }

    #[test]
    fn model_must_clear_product_balanced_accuracy_and_each_class_floor() {
        let weak_binary = model_with_validation(0.89, &[("absent", 0.95), ("present_still", 0.83)]);
        assert!(weak_binary.runtime_eligibility().is_err());

        let collapsed_class =
            model_with_validation(0.94, &[("absent", 0.98), ("present_still", 0.89)]);
        assert!(collapsed_class.runtime_eligibility().is_err());

        let eligible = model_with_validation(0.94, &[("absent", 0.95), ("present_still", 0.93)]);
        assert!(eligible.runtime_eligibility().is_ok());
    }

    #[test]
    fn stale_feature_schema_is_never_activated() {
        let mut stale = model_with_validation(0.95, &[("absent", 0.95), ("present", 0.95)]);
        stale.version = 3;

        let error = stale.runtime_eligibility().unwrap_err();

        assert!(error.contains("schema"), "unexpected error: {error}");
    }

    #[test]
    fn offline_and_runtime_features_do_not_depend_on_private_amplitude_payloads() {
        let features = serde_json::json!({
            "variance": 1.0,
            "motion_band_power": 2.0,
            "breathing_band_power": 0.3,
            "spectral_power": 4.0,
            "dominant_freq_hz": 0.25,
            "change_points": 5,
            "mean_rssi": -45.0
        });
        let frame = serde_json::json!({"features": features.clone(), "nodes": []});

        assert_eq!(
            features_from_frame(&frame),
            features_from_runtime(&features, &[100.0, 200.0, 300.0])
        );
    }

    #[test]
    fn every_class_needs_an_independent_held_out_recording() {
        let mut model = model_with_validation(0.80, &[("absent", 0.82), ("present_still", 0.78)]);
        model.validation.as_mut().unwrap().held_out_recordings = 1;

        assert!(model.runtime_eligibility().is_err());
    }

    #[test]
    fn malformed_or_non_finite_model_artifacts_are_never_activated() {
        let mut wrong_shape =
            model_with_validation(0.80, &[("absent", 0.82), ("present_still", 0.78)]);
        wrong_shape.weights[0].pop();
        assert!(wrong_shape.runtime_eligibility().is_err());

        let mut non_finite =
            model_with_validation(0.80, &[("absent", 0.82), ("present_still", 0.78)]);
        non_finite.global_std[3] = f64::NAN;
        assert!(non_finite.runtime_eligibility().is_err());
    }

    #[test]
    fn runtime_loader_rejects_legacy_model_even_when_training_accuracy_is_high() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        let mut model = AdaptiveModel::default();
        model.training_accuracy = 1.0;
        model.save(&path).unwrap();

        assert!(AdaptiveModel::load_runtime(&path).is_err());
    }

    #[test]
    fn runtime_loader_accepts_session_validated_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        let model = model_with_validation(0.95, &[("absent", 0.96), ("present_still", 0.94)]);
        model.save(&path).unwrap();

        assert!(AdaptiveModel::load_runtime(&path).is_ok());
    }

    fn write_recording(dir: &Path, name: &str, variance: f64, amp: f64) {
        write_recording_with_node_count(dir, name, variance, amp, 1);
    }

    fn write_recording_with_node_count(
        dir: &Path,
        name: &str,
        variance: f64,
        amp: f64,
        node_count: usize,
    ) {
        let mut file = std::fs::File::create(dir.join(name)).unwrap();
        for i in 0..40 {
            let nodes: Vec<_> = (0..node_count)
                .map(|node_id| {
                    serde_json::json!({
                        "node_id": node_id + 1,
                        "amplitude": [amp, amp + 0.1, amp + 0.2, amp + 0.3]
                    })
                })
                .collect();
            let frame = serde_json::json!({
                "features": {
                    "variance": variance + (i % 3) as f64 * 0.01,
                    "motion_band_power": variance,
                    "breathing_band_power": variance * 0.5,
                    "spectral_power": variance + 1.0,
                    "dominant_freq_hz": variance * 0.01,
                    "change_points": if variance > 10.0 { 8 } else { 0 },
                    "mean_rssi": if variance > 10.0 { -45.0 } else { -60.0 }
                },
                "nodes": nodes
            });
            writeln!(file, "{frame}").unwrap();
        }
    }

    #[test]
    fn temporal_patch_is_rate_invariant_and_requires_context() {
        let mut one_hz = AdaptiveFeatureExtractor::default();
        let mut ten_hz = AdaptiveFeatureExtractor::default();

        let base = |value: f64| {
            let mut features = [0.0; N_BASE_FEATURES];
            features[0] = value;
            features
        };

        assert!(one_hz.observe_at(base(0.0), 0.0).is_none());
        assert!(one_hz.observe_at(base(1.0), 1.0).is_none());
        let slow = one_hz.observe_at(base(2.0), 2.0).expect("two-second patch");

        let mut fast = None;
        for step in 0..=20 {
            let time = step as f64 / 10.0;
            fast = ten_hz.observe_at(base(time.floor()), time);
        }

        assert_eq!(slow, fast.expect("two-second patch"));
        assert_eq!(2.0, slow[0]);
        assert_eq!(1.0, slow[N_BASE_FEATURES]);
        assert!(
            (slow[N_BASE_FEATURES + N_TEMPORAL_FEATURES] - (2.0_f64 / 3.0).sqrt()).abs() < 1e-9
        );
    }

    #[test]
    fn occupied_recording_names_share_the_presence_class() {
        for name in [
            "train_present_still_desk.jsonl",
            "train_walking_room.jsonl",
            "train_active_dressing.jsonl",
        ] {
            assert_eq!(Some("present".to_string()), classify_recording_name(name));
        }
        assert_eq!(
            Some("absent".to_string()),
            classify_recording_name("train_empty_room.jsonl")
        );
    }

    #[test]
    fn learned_presence_preserves_motion_granularity() {
        assert_eq!(
            ("active".to_string(), true),
            reconcile_presence_prediction("present", "active")
        );
        assert_eq!(
            ("present_still".to_string(), true),
            reconcile_presence_prediction("present", "absent")
        );
        assert_eq!(
            ("absent".to_string(), false),
            reconcile_presence_prediction("absent", "active")
        );
    }

    #[test]
    fn presence_smoother_rejects_a_single_contrary_prediction() {
        let mut smoother = AdaptivePresenceSmoother::default();
        assert!(smoother.update("present", 0.9));

        assert!(
            smoother.update("absent", 0.9),
            "one noisy frame must not clear an occupied room"
        );
        for _ in 0..20 {
            smoother.update("absent", 0.9);
        }
        assert!(!smoother.is_present().unwrap());
    }

    #[test]
    fn private_model_path_does_not_change_the_default() {
        assert_eq!(
            PathBuf::from("/private/model.json"),
            model_path_from(Some("/private/model.json".into()))
        );
        assert_eq!(
            PathBuf::from("data/adaptive_model.json"),
            model_path_from(None)
        );
    }

    #[test]
    fn topology_filter_excludes_incompatible_recordings() {
        let dir = tempfile::tempdir().unwrap();
        for session in 1..=2 {
            write_recording_with_node_count(
                dir.path(),
                &format!("train_absent_dual{session}.jsonl"),
                1.0,
                1.0,
                2,
            );
            write_recording_with_node_count(
                dir.path(),
                &format!("train_present_still_dual{session}.jsonl"),
                100.0,
                20.0,
                2,
            );
        }
        write_recording(dir.path(), "train_absent_single.jsonl", 50.0, 10.0);
        write_recording(dir.path(), "train_present_still_single.jsonl", 50.0, 10.0);

        let model = train_from_recordings_for_node_count(dir.path(), Some(2)).unwrap();

        assert_eq!(Some(2), model.expected_node_count);
        assert_eq!(152, model.trained_frames);
        assert_eq!(4, model.validation.unwrap().held_out_recordings);
    }

    #[test]
    fn one_recording_per_class_cannot_claim_generalization() {
        let dir = tempfile::tempdir().unwrap();
        write_recording(dir.path(), "train_absent_session1.jsonl", 1.0, 1.0);
        write_recording(
            dir.path(),
            "train_present_still_session1.jsonl",
            100.0,
            20.0,
        );

        let model = train_from_recordings(dir.path()).unwrap();

        assert!(model.validation.is_none());
        assert!(model.runtime_eligibility().is_err());
    }

    #[test]
    fn validation_covers_every_recording_and_final_fit_uses_all_frames() {
        let dir = tempfile::tempdir().unwrap();
        for session in 1..=2 {
            write_recording(
                dir.path(),
                &format!("train_absent_session{session}.jsonl"),
                1.0,
                1.0,
            );
            write_recording(
                dir.path(),
                &format!("train_present_still_session{session}.jsonl"),
                100.0,
                20.0,
            );
        }

        let model = train_from_recordings(dir.path()).unwrap();
        let validation = model.validation.as_ref().expect("held-out metrics");

        assert_eq!(validation.held_out_recordings, 4);
        assert_eq!(validation.held_out_frames, 152);
        assert_eq!(model.trained_frames, 152);
        assert!(validation.balanced_accuracy > 0.95);
        assert!(model.runtime_eligibility().is_ok());
    }

    #[test]
    fn recorded_feature_extraction_is_independent_of_node_payload_order() {
        let frame_ab = serde_json::json!({
            "features": { "variance": 1.0 },
            "nodes": [
                { "node_id": 1, "amplitude": [1.0, 2.0] },
                { "node_id": 2, "amplitude": [9.0, 10.0] }
            ]
        });
        let frame_ba = serde_json::json!({
            "features": { "variance": 1.0 },
            "nodes": [
                { "node_id": 2, "amplitude": [9.0, 10.0] },
                { "node_id": 1, "amplitude": [1.0, 2.0] }
            ]
        });

        let ab = features_from_frame(&frame_ab);
        let ba = features_from_frame(&frame_ba);

        assert_eq!(ab, ba, "node serialization order must not change features");
        assert_eq!(ab.len(), N_BASE_FEATURES);
    }

    #[test]
    fn spatial_intelligence_payload_wrapper_uses_the_same_feature_contract() {
        let wrapped = serde_json::json!({
            "captured_at": "2026-08-10T00:00:00Z",
            "payload": {
                "features": { "variance": 3.0 },
                "nodes": [{ "node_id": 1, "amplitude": [2.0, 4.0] }]
            }
        });

        let features = features_from_frame(&wrapped);

        assert_eq!(features[0], 3.0);
        assert_eq!(features[1..], [0.0; N_BASE_FEATURES - 1]);
    }
}

//! Adaptive CSI Activity Classifier
//!
//! Learns environment-specific classification thresholds from labeled JSONL
//! recordings.  Uses a lightweight approach:
//!
//! 1. **Feature statistics**: per-class mean/stddev for each of 7 CSI features
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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── Feature vector ───────────────────────────────────────────────────────────

/// Extended feature vector: 7 server features + 8 subcarrier-derived features = 15.
const N_FEATURES: usize = 15;

/// Default class names for backward compatibility with old saved models.
const DEFAULT_CLASSES: &[&str] = &["absent", "present_still", "present_moving", "active"];

/// Extract extended feature vector from a JSONL frame (features + raw amplitudes).
pub fn features_from_frame(frame: &serde_json::Value) -> [f64; N_FEATURES] {
    // spatial-intelligence stores the original RuView envelope under
    // `payload`; RuView's own recorder writes the envelope at the root. Keep
    // one feature contract so offline evaluation cannot silently turn every
    // wrapped feature into zero.
    let frame = frame.get("payload").unwrap_or(frame);
    let feat = frame
        .get("features")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let nodes = frame.get("nodes").and_then(|n| n.as_array());
    let amps: Vec<f64> = nodes
        .map(|ns| {
            // The room-level adaptive model is trained over every link. Node
            // order comes from a HashMap and is not stable across processes,
            // so only order-invariant summary statistics are derived here.
            ns.iter()
                .filter_map(|node| node.get("amplitude").and_then(|a| a.as_array()))
                .flatten()
                .filter_map(|value| value.as_f64())
                .collect()
        })
        .unwrap_or_default();

    // Server-computed features (0-6).
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

    // Subcarrier-derived features (7-14).
    let (amp_mean, amp_std, amp_skew, amp_kurt, amp_iqr, amp_entropy, amp_max, amp_range) =
        subcarrier_stats(&amps);

    [
        variance,
        mbp,
        bbp,
        sp,
        df,
        cp,
        rssi,
        amp_mean,
        amp_std,
        amp_skew,
        amp_kurt,
        amp_iqr,
        amp_entropy,
        amp_max,
        amp_range,
    ]
}

/// Also keep a simpler version for runtime (no JSONL, just FeatureInfo + amps).
pub fn features_from_runtime(feat: &serde_json::Value, amps: &[f64]) -> [f64; N_FEATURES] {
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
    let (amp_mean, amp_std, amp_skew, amp_kurt, amp_iqr, amp_entropy, amp_max, amp_range) =
        subcarrier_stats(amps);
    [
        variance,
        mbp,
        bbp,
        sp,
        df,
        cp,
        rssi,
        amp_mean,
        amp_std,
        amp_skew,
        amp_kurt,
        amp_iqr,
        amp_entropy,
        amp_max,
        amp_range,
    ]
}

/// Compute statistical features from raw subcarrier amplitudes.
fn subcarrier_stats(amps: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    // HashMap-backed node serialization has no stable order. Sorting makes all
    // summary features bit-reproducible across process restarts; dropping
    // non-finite hardware samples keeps one corrupt bin from poisoning every
    // logit in the room model.
    let mut sorted: Vec<f64> = amps
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if sorted.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let var = sorted.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-9);

    // Skewness (asymmetry).
    let skew = sorted
        .iter()
        .map(|a| ((a - mean) / std).powi(3))
        .sum::<f64>()
        / n;
    // Kurtosis (peakedness).
    let kurt = sorted
        .iter()
        .map(|a| ((a - mean) / std).powi(4))
        .sum::<f64>()
        / n
        - 3.0;

    // IQR (inter-quartile range).
    let q1 = sorted[sorted.len() / 4];
    let q3 = sorted[3 * sorted.len() / 4];
    let iqr = q3 - q1;

    // Spectral entropy (normalised).
    let total_power: f64 = sorted.iter().map(|a| a * a).sum::<f64>().max(1e-9);
    let entropy: f64 = sorted
        .iter()
        .map(|a| {
            let p = (a * a) / total_power;
            if p > 1e-12 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n.ln().max(1e-9); // normalise to [0,1]

    let max_val = sorted.last().copied().unwrap_or(0.0);
    let range = max_val - sorted.first().copied().unwrap_or(0.0);

    (mean, std, skew, kurt, iqr, entropy, max_val, range)
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
            version: 1,
            class_names: default_class_names(),
            validation: None,
        }
    }
}

impl AdaptiveModel {
    /// Whether this model has enough independent evidence to override the
    /// conservative threshold classifier in production.
    pub fn runtime_eligibility(&self) -> Result<(), String> {
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
        if metrics.held_out_recordings < n_classes {
            return Err(format!(
                "held-out validation needs at least one independent recording per class (have {}, need {n_classes})",
                metrics.held_out_recordings
            ));
        }
        if metrics.held_out_frames == 0 {
            return Err("held-out validation contains no frames".into());
        }

        // Binary classifiers must clear a higher bar than four-way models
        // because 50% is already random chance. The fixed 0.60 floor prevents
        // a many-class model from activating on a superficially small margin.
        let chance = 1.0 / n_classes as f64;
        let required_balanced_accuracy = 0.60_f64.max(chance + 0.20);
        if !metrics.balanced_accuracy.is_finite()
            || metrics.balanced_accuracy < required_balanced_accuracy
        {
            return Err(format!(
                "held-out balanced accuracy {:.3} is below activation floor {:.3}",
                metrics.balanced_accuracy, required_balanced_accuracy
            ));
        }

        for class_name in &self.class_names {
            let recall = metrics
                .per_class_recall
                .get(class_name)
                .copied()
                .ok_or_else(|| format!("held-out recall missing for class '{class_name}'"))?;
            if !recall.is_finite() || recall < 0.50 {
                return Err(format!(
                    "held-out recall for class '{class_name}' is {recall:.3}, below 0.500"
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
struct Sample {
    features: [f64; N_FEATURES],
    class_idx: usize,
}

/// Load JSONL recording frames and assign a class label based on filename.
fn load_recording(path: &Path, class_idx: usize) -> Vec<Sample> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    content
        .lines()
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).ok()?;
            // Use extended features (server features + subcarrier stats).
            Some(Sample {
                features: features_from_frame(&v),
                class_idx,
            })
        })
        .collect()
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
    if lower.contains("still") || lower.contains("sitting") || lower.contains("standing") {
        return Some("present_still".into());
    }
    if lower.contains("walking") || lower.contains("moving") {
        return Some("present_moving".into());
    }
    if lower.contains("active") || lower.contains("exercise") || lower.contains("running") {
        return Some("active".into());
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
/// - `*still*` / `*sitting*`  → present_still
/// - `*walking*` / `*moving*` → present_moving
/// - `*active*` / `*exercise*`→ active
/// - Any other `train_<class>_*.jsonl` → <class>
pub fn train_from_recordings(recordings_dir: &Path) -> Result<AdaptiveModel, String> {
    // First pass: scan filenames to discover all unique class names.
    let entries: Vec<_> = std::fs::read_dir(recordings_dir)
        .map_err(|e| format!("Cannot read {}: {}", recordings_dir.display(), e))?
        .flatten()
        .collect();

    let mut class_map: HashMap<String, usize> = HashMap::new();
    let mut class_names: Vec<String> = Vec::new();

    // Collect (entry, class_name) pairs for files that match.
    let mut file_classes: Vec<(PathBuf, String, String)> = Vec::new(); // (path, fname, class_name)
    for entry in &entries {
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.starts_with("train_") || !fname.ends_with(".jsonl") {
            continue;
        }
        if let Some(class_name) = classify_recording_name(&fname) {
            if !class_map.contains_key(&class_name) {
                let idx = class_names.len();
                class_map.insert(class_name.clone(), idx);
                class_names.push(class_name.clone());
            }
            file_classes.push((entry.path(), fname, class_name));
        }
    }

    let n_classes = class_names.len();
    if n_classes == 0 {
        return Err("No training samples found. Record data with train_* prefix.".into());
    }

    // Stable ordering makes both the held-out choice and model fitting
    // reproducible across filesystems whose read_dir order differs.
    file_classes.sort_by(|a, b| a.1.cmp(&b.1));

    // Hold out one whole recording per class. Adjacent CSI frames are highly
    // autocorrelated, so a random frame split leaks a session's room/channel
    // fingerprint into validation and overstates deployment accuracy.
    let mut recordings_per_class: HashMap<String, usize> = HashMap::new();
    for (_, _, class_name) in &file_classes {
        *recordings_per_class.entry(class_name.clone()).or_default() += 1;
    }
    let can_validate_sessions = class_names
        .iter()
        .all(|name| recordings_per_class.get(name).copied().unwrap_or(0) >= 2);
    let mut held_out_path_by_class: HashMap<String, PathBuf> = HashMap::new();
    if can_validate_sessions {
        for (path, _, class_name) in &file_classes {
            // Sorted iteration intentionally leaves the last session per class.
            held_out_path_by_class.insert(class_name.clone(), path.clone());
        }
    }

    // Second pass: load fitting and held-out sessions separately.
    let mut samples: Vec<Sample> = Vec::new();
    let mut validation_samples: Vec<Sample> = Vec::new();
    for (path, fname, class_name) in &file_classes {
        let class_idx = class_map[class_name];
        let loaded = load_recording(path, class_idx);
        let is_held_out = held_out_path_by_class.get(class_name) == Some(path);
        eprintln!(
            "  Loaded {}: {} frames → class '{}'{}",
            fname,
            loaded.len(),
            class_name,
            if is_held_out { " [held out]" } else { "" }
        );
        if is_held_out {
            validation_samples.extend(loaded);
        } else {
            samples.extend(loaded);
        }
    }

    if samples.is_empty() {
        return Err("No training samples found. Record data with train_* prefix.".into());
    }

    let n = samples.len();
    eprintln!(
        "Total training samples: {n} across {n_classes} classes: {:?}",
        class_names
    );

    // ── Compute global normalisation stats ──
    let mut global_mean = [0.0f64; N_FEATURES];
    let mut global_var = [0.0f64; N_FEATURES];
    for s in &samples {
        for (m, &f) in global_mean.iter_mut().zip(s.features.iter()) {
            *m += f;
        }
    }
    for m in global_mean.iter_mut() {
        *m /= n as f64;
    }
    for s in &samples {
        for i in 0..N_FEATURES {
            global_var[i] += (s.features[i] - global_mean[i]).powi(2);
        }
    }
    let mut global_std = [0.0f64; N_FEATURES];
    for i in 0..N_FEATURES {
        global_std[i] = (global_var[i] / n as f64).sqrt().max(1e-9);
    }

    // ── Compute per-class statistics ──
    let mut class_sums = vec![[0.0f64; N_FEATURES]; n_classes];
    let mut class_sq = vec![[0.0f64; N_FEATURES]; n_classes];
    let mut class_counts = vec![0usize; n_classes];
    for s in &samples {
        let c = s.class_idx;
        class_counts[c] += 1;
        for i in 0..N_FEATURES {
            class_sums[c][i] += s.features[i];
            class_sq[c][i] += s.features[i] * s.features[i];
        }
    }

    let mut class_stats = Vec::new();
    for c in 0..n_classes {
        let cnt = class_counts[c].max(1) as f64;
        let mut mean = [0.0; N_FEATURES];
        let mut stddev = [0.0; N_FEATURES];
        for i in 0..N_FEATURES {
            mean[i] = class_sums[c][i] / cnt;
            stddev[i] = ((class_sq[c][i] / cnt) - mean[i] * mean[i]).max(0.0).sqrt();
        }
        class_stats.push(ClassStats {
            label: class_names[c].clone(),
            count: class_counts[c],
            mean,
            stddev,
        });
    }

    // ── Normalise all samples ──
    let mut norm_samples: Vec<([f64; N_FEATURES], usize)> = samples
        .iter()
        .map(|s| {
            let mut x = [0.0; N_FEATURES];
            for i in 0..N_FEATURES {
                x[i] = (s.features[i] - global_mean[i]) / (global_std[i] + 1e-9);
            }
            (x, s.class_idx)
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

            for (x, target) in batch {
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
                epoch_loss += -(probs[*target].max(1e-15)).ln();

                // Gradient: prob - one_hot(target).
                for c in 0..n_classes {
                    let delta = probs[c] - if c == *target { 1.0 } else { 0.0 };
                    for (g, &xi) in grad[c][..N_FEATURES].iter_mut().zip(x.iter()) {
                        *g += delta * xi;
                    }
                    grad[c][N_FEATURES] += delta; // bias grad
                }
            }

            // Update weights.
            let bs = batch.len() as f64;
            let current_lr = lr * (1.0 - epoch as f64 / epochs as f64); // linear decay
            for c in 0..n_classes {
                for i in 0..=N_FEATURES {
                    weights[c][i] -= current_lr * grad[c][i] / bs;
                }
            }
        }

        if epoch % 50 == 0 || epoch == epochs - 1 {
            let avg_loss = epoch_loss / n as f64;
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
    for (x, target) in &norm_samples {
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
    eprintln!(
        "Training accuracy: {correct}/{n} = {:.1}%",
        accuracy * 100.0
    );

    // ── Per-class accuracy ──
    let mut class_correct = vec![0usize; n_classes];
    let mut class_total = vec![0usize; n_classes];
    for (x, target) in &norm_samples {
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

    let mut model = AdaptiveModel {
        class_stats,
        weights,
        global_mean,
        global_std,
        trained_frames: n,
        training_accuracy: accuracy,
        version: 2,
        class_names,
        validation: None,
    };
    if can_validate_sessions {
        model.validation = Some(evaluate_held_out(
            &model,
            &validation_samples,
            held_out_path_by_class.len(),
        ));
    } else {
        eprintln!(
            "Session-held-out validation unavailable: record at least two independent sessions per class"
        );
    }
    Ok(model)
}

/// Evaluate the fitted model only on sessions excluded from fitting.
fn evaluate_held_out(
    model: &AdaptiveModel,
    samples: &[Sample],
    held_out_recordings: usize,
) -> ValidationMetrics {
    let n_classes = model.class_names.len();
    let mut class_correct = vec![0usize; n_classes];
    let mut class_total = vec![0usize; n_classes];
    for sample in samples {
        class_total[sample.class_idx] += 1;
        let (predicted, _) = model.classify(&sample.features);
        if predicted == model.class_names[sample.class_idx] {
            class_correct[sample.class_idx] += 1;
        }
    }

    let mut per_class_recall = HashMap::new();
    for class_idx in 0..n_classes {
        let recall = if class_total[class_idx] == 0 {
            0.0
        } else {
            class_correct[class_idx] as f64 / class_total[class_idx] as f64
        };
        per_class_recall.insert(model.class_names[class_idx].clone(), recall);
    }
    let balanced_accuracy = if n_classes == 0 {
        0.0
    } else {
        per_class_recall.values().sum::<f64>() / n_classes as f64
    };
    eprintln!(
        "Held-out balanced accuracy: {:.1}% across {} recording(s) / {} frames",
        balanced_accuracy * 100.0,
        held_out_recordings,
        samples.len()
    );

    ValidationMetrics {
        held_out_recordings,
        held_out_frames: samples.len(),
        balanced_accuracy,
        per_class_recall,
    }
}

/// Default path for the saved adaptive model.
pub fn model_path() -> PathBuf {
    PathBuf::from("data/adaptive_model.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn model_with_validation(balanced_accuracy: f64, recalls: &[(&str, f64)]) -> AdaptiveModel {
        let mut model = AdaptiveModel::default();
        model.version = 2;
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
            held_out_recordings: recalls.len(),
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
    fn model_must_beat_dynamic_chance_margin_and_each_class_floor() {
        let weak_binary = model_with_validation(0.69, &[("absent", 0.70), ("present_still", 0.68)]);
        assert!(weak_binary.runtime_eligibility().is_err());

        let collapsed_class =
            model_with_validation(0.75, &[("absent", 0.98), ("present_still", 0.49)]);
        assert!(collapsed_class.runtime_eligibility().is_err());

        let eligible = model_with_validation(0.80, &[("absent", 0.82), ("present_still", 0.78)]);
        assert!(eligible.runtime_eligibility().is_ok());
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
        let model = model_with_validation(0.80, &[("absent", 0.82), ("present_still", 0.78)]);
        model.save(&path).unwrap();

        assert!(AdaptiveModel::load_runtime(&path).is_ok());
    }

    fn write_recording(dir: &Path, name: &str, variance: f64, amp: f64) {
        let mut file = std::fs::File::create(dir.join(name)).unwrap();
        for i in 0..40 {
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
                "nodes": [{ "amplitude": [amp, amp + 0.1, amp + 0.2, amp + 0.3] }]
            });
            writeln!(file, "{frame}").unwrap();
        }
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
    fn validation_uses_whole_recording_sessions_not_training_frames() {
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

        assert_eq!(validation.held_out_recordings, 2);
        assert_eq!(validation.held_out_frames, 80);
        assert!(validation.balanced_accuracy > 0.95);
        assert!(model.runtime_eligibility().is_ok());
    }

    #[test]
    fn recorded_feature_extraction_uses_every_node_and_is_order_invariant() {
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
        assert!((ab[7] - 5.5).abs() < 1e-9, "all-node amplitude mean");
        assert!((ab[14] - 9.0).abs() < 1e-9, "all-node amplitude range");
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
        assert_eq!(features[7], 3.0);
    }
}

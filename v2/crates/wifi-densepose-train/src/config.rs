//! Training configuration for WiFi-DensePose.
//!
//! [`TrainingConfig`] is the single source of truth for all hyper-parameters,
//! dataset shapes, loss weights, and infrastructure settings used throughout
//! the training pipeline. It is serializable via [`serde`] so it can be stored
//! to / restored from JSON checkpoint files.
//!
//! # Example
//!
//! ```rust
//! use wifi_densepose_train::config::TrainingConfig;
//!
//! let cfg = TrainingConfig::default();
//! cfg.validate().expect("default config is valid");
//!
//! assert_eq!(cfg.num_subcarriers, 56);
//! assert_eq!(cfg.num_keypoints, 17);
//!
//! // Adapt for a non-MM-Fi source — e.g. an ESP32 HT40 capture (~192 raw
//! // subcarriers) or the ADR-078 multi-band mesh (168). The model still sees
//! // `num_subcarriers`; the loader resamples the native count down to it.
//! let ht40 = TrainingConfig::ht40_192();
//! assert_eq!(ht40.native_subcarriers, 192);
//! assert!(ht40.needs_subcarrier_interp());
//! let mesh = TrainingConfig::for_subcarriers(168, 56);
//! assert_eq!(mesh.native_subcarriers, 168);
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::ConfigError;

// ---------------------------------------------------------------------------
// Allocation-guard upper bounds (ADR-155 §Tier-2)
// ---------------------------------------------------------------------------
//
// `validate()` historically only checked lower bounds, so a config with an
// absurd field (e.g. `window_frames = usize::MAX`) passed validation and only
// blew up later as an OOM / allocation-size overflow deep in the pipeline.
// These constants cap each dimensioning field at a value far above any real
// hardware configuration but well below the point where the product of
// dimensions overflows `usize` on a 64-bit allocation. They guard against
// allocation-overflow, not against "sensible" configs — every real preset
// stays orders of magnitude under these caps.

/// Maximum temporal window length, in frames. Caps the time dimension of every
/// CSI window allocation. Real captures use ≤ a few thousand frames.
pub const MAX_WINDOW_FRAMES: usize = 100_000;

/// Maximum subcarrier count (model or native). Real Wi-Fi captures top out in
/// the low hundreds; this leaves vast headroom while preventing overflow.
pub const MAX_SUBCARRIERS: usize = 100_000;

/// Maximum backbone feature-map channel count. Even large vision backbones use
/// a few thousand channels.
pub const MAX_BACKBONE_CHANNELS: usize = 1_000_000;

/// Maximum heatmap side length (H = W). Caps the square heatmap allocation.
pub const MAX_HEATMAP_SIZE: usize = 100_000;

/// Maximum number of keypoints. COCO uses 17; this is a wide safety margin.
pub const MAX_KEYPOINTS: usize = 10_000;

/// Maximum number of DensePose body-part classes. DensePose uses 24.
pub const MAX_BODY_PARTS: usize = 10_000;

/// Maximum mini-batch size. Guards the batch dimension of every allocation.
pub const MAX_BATCH_SIZE: usize = 1_000_000;

// ---------------------------------------------------------------------------
// TrainingConfig
// ---------------------------------------------------------------------------

/// Complete configuration for a WiFi-DensePose training run.
///
/// All fields have documented defaults that match the paper's experimental
/// setup. Use [`TrainingConfig::default()`] as a starting point, then override
/// individual fields as needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    // -----------------------------------------------------------------------
    // Data / Signal
    // -----------------------------------------------------------------------
    /// Number of subcarriers after interpolation (the *model's* input width).
    ///
    /// The model always sees this many subcarriers regardless of the raw
    /// hardware output; [`crate::subcarrier::interpolate_subcarriers`] resamples
    /// `native_subcarriers` → `num_subcarriers` when they differ. Default: **56**.
    pub num_subcarriers: usize,

    /// Number of subcarriers in the *raw* dataset, before interpolation.
    ///
    /// Common sources: MM-Fi = 114, ESP32 HT20 = 56, ESP32 HT40 ≈ 192 (or 114),
    /// multi-band mesh = 168 (ADR-078). When it equals [`Self::num_subcarriers`]
    /// no interpolation happens ([`Self::needs_subcarrier_interp`]). For the
    /// non-MM-Fi shapes prefer the preset constructors
    /// ([`Self::for_subcarriers`], [`Self::ht40_192`], [`Self::multiband_168`])
    /// over overriding both fields by hand. Default: **114**.
    ///
    /// **Multi-NIC note:** a 2–3-node CSI mesh currently maps onto the existing
    /// `[T, n_tx, n_rx, n_sc]` layout by treating the nodes' receive chains as
    /// extra `n_rx` (i.e. `num_antennas_rx = nodes × per_node_rx`); a dedicated
    /// node dimension is a separate dataset-loader change.
    pub native_subcarriers: usize,

    /// Number of transmit antennas. Default: **3**.
    pub num_antennas_tx: usize,

    /// Number of receive antennas. Default: **3**.
    pub num_antennas_rx: usize,

    /// Temporal sliding-window length in frames. Default: **100**.
    pub window_frames: usize,

    /// Side length of the square keypoint heatmap output (H = W). Default: **56**.
    pub heatmap_size: usize,

    // -----------------------------------------------------------------------
    // Model
    // -----------------------------------------------------------------------
    /// Number of body keypoints (COCO 17-joint skeleton). Default: **17**.
    pub num_keypoints: usize,

    /// Number of DensePose body-part classes. Default: **24**.
    pub num_body_parts: usize,

    /// Number of feature-map channels in the backbone encoder. Default: **256**.
    pub backbone_channels: usize,

    // -----------------------------------------------------------------------
    // Optimisation
    // -----------------------------------------------------------------------
    /// Mini-batch size. Default: **8**.
    pub batch_size: usize,

    /// Initial learning rate for the Adam / AdamW optimiser. Default: **1e-3**.
    pub learning_rate: f64,

    /// L2 weight-decay regularisation coefficient. Default: **1e-4**.
    pub weight_decay: f64,

    /// Total number of training epochs. Default: **50**.
    pub num_epochs: usize,

    /// Number of linear-warmup epochs at the start. Default: **5**.
    pub warmup_epochs: usize,

    /// Epochs at which the learning rate is multiplied by `lr_gamma`.
    ///
    /// Default: **[30, 45]** (multi-step scheduler).
    pub lr_milestones: Vec<usize>,

    /// Multiplicative factor applied at each LR milestone. Default: **0.1**.
    pub lr_gamma: f64,

    /// Maximum gradient L2 norm for gradient clipping. Default: **1.0**.
    pub grad_clip_norm: f64,

    // -----------------------------------------------------------------------
    // Loss weights
    // -----------------------------------------------------------------------
    /// Weight for the keypoint heatmap loss term. Default: **0.3**.
    pub lambda_kp: f64,

    /// Weight for the DensePose body-part / UV-coordinate loss. Default: **0.6**.
    pub lambda_dp: f64,

    /// Weight for the cross-modal transfer / domain-alignment loss. Default: **0.1**.
    pub lambda_tr: f64,

    // -----------------------------------------------------------------------
    // Validation and checkpointing
    // -----------------------------------------------------------------------
    /// Run validation every N epochs. Default: **1**.
    pub val_every_epochs: usize,

    /// Stop training if validation loss does not improve for this many
    /// consecutive validation rounds. Default: **10**.
    pub early_stopping_patience: usize,

    /// Directory where model checkpoints are saved.
    pub checkpoint_dir: PathBuf,

    /// Directory where TensorBoard / CSV logs are written.
    pub log_dir: PathBuf,

    /// Keep only the top-K best checkpoints by validation metric. Default: **3**.
    pub save_top_k: usize,

    // -----------------------------------------------------------------------
    // Device
    // -----------------------------------------------------------------------
    /// Use a CUDA GPU for training when available. Default: **false**.
    pub use_gpu: bool,

    /// CUDA device index when `use_gpu` is `true`. Default: **0**.
    pub gpu_device_id: i64,

    /// Number of background data-loading threads. Default: **4**.
    pub num_workers: usize,

    // -----------------------------------------------------------------------
    // Reproducibility
    // -----------------------------------------------------------------------
    /// Global random seed for all RNG sources in the training pipeline.
    ///
    /// This seed is applied to the dataset shuffler, model parameter
    /// initialisation, and any stochastic augmentation. Default: **42**.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            // Data
            num_subcarriers: 56,
            native_subcarriers: 114,
            num_antennas_tx: 3,
            num_antennas_rx: 3,
            window_frames: 100,
            heatmap_size: 56,
            // Model
            num_keypoints: 17,
            num_body_parts: 24,
            backbone_channels: 256,
            // Optimisation
            batch_size: 8,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            num_epochs: 50,
            warmup_epochs: 5,
            lr_milestones: vec![30, 45],
            lr_gamma: 0.1,
            grad_clip_norm: 1.0,
            // Loss weights
            lambda_kp: 0.3,
            lambda_dp: 0.6,
            lambda_tr: 0.1,
            // Validation / checkpointing
            val_every_epochs: 1,
            early_stopping_patience: 10,
            checkpoint_dir: PathBuf::from("checkpoints"),
            log_dir: PathBuf::from("logs"),
            save_top_k: 3,
            // Device
            use_gpu: false,
            gpu_device_id: 0,
            num_workers: 4,
            // Reproducibility
            seed: 42,
        }
    }
}

impl TrainingConfig {
    /// Load a [`TrainingConfig`] from a JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::FileRead`] if the file cannot be opened and
    /// [`ConfigError::InvalidValue`] if the JSON is malformed.
    pub fn from_json(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::FileRead {
            path: path.to_path_buf(),
            source,
        })?;
        let cfg: TrainingConfig = serde_json::from_str(&contents)
            .map_err(|e| ConfigError::invalid_value("(file)", e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Serialize this configuration to pretty-printed JSON and write it to
    /// `path`, creating parent directories if necessary.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::FileRead`] if the directory cannot be created or
    /// the file cannot be written.
    pub fn to_json(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ConfigError::FileRead {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::invalid_value("(serialization)", e.to_string()))?;
        std::fs::write(path, json).map_err(|source| ConfigError::FileRead {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }

    /// Build a config for a dataset whose raw CSI has `native` subcarriers,
    /// resampling to `target` (the model's input width) before training.
    ///
    /// All other fields take their [`Default`] values. Prefer this over
    /// overriding `native_subcarriers` / `num_subcarriers` directly so the
    /// relationship between the dataset's shape and the model's is explicit.
    #[must_use]
    pub fn for_subcarriers(native: usize, target: usize) -> Self {
        Self {
            native_subcarriers: native,
            num_subcarriers: target,
            ..Self::default()
        }
    }

    /// Preset for the MM-Fi dataset (114 raw subcarriers → 56). Identical to
    /// [`Self::default()`]; provided as a named counterpart to the other
    /// presets.
    #[must_use]
    pub fn mmfi() -> Self {
        Self::default()
    }

    /// Preset for ESP32 HT40 captures (≈192 raw subcarriers → 56). Use
    /// [`Self::for_subcarriers`] if your capture reports a different native
    /// count (some HT40 firmwares yield 114).
    #[must_use]
    pub fn ht40_192() -> Self {
        Self::for_subcarriers(192, 56)
    }

    /// Preset for the ADR-078 multi-band mesh (168 raw subcarriers → 56).
    #[must_use]
    pub fn multiband_168() -> Self {
        Self::for_subcarriers(168, 56)
    }

    /// Returns `true` when the native dataset subcarrier count differs from the
    /// model's target count and interpolation is therefore required.
    pub fn needs_subcarrier_interp(&self) -> bool {
        self.native_subcarriers != self.num_subcarriers
    }

    /// Validate all fields and return an error describing the first problem
    /// found, or `Ok(())` if the configuration is coherent.
    ///
    /// # Validated invariants
    ///
    /// - Subcarrier counts must be non-zero.
    /// - Antenna counts must be non-zero.
    /// - `window_frames` must be at least 1.
    /// - `batch_size` must be at least 1.
    /// - `learning_rate` must be strictly positive.
    /// - `weight_decay` must be non-negative.
    /// - Loss weights must be non-negative and sum to a positive value.
    /// - `num_epochs` must be greater than `warmup_epochs`.
    /// - All `lr_milestones` must be within `[1, num_epochs]` and strictly
    ///   increasing.
    /// - `save_top_k` must be at least 1.
    /// - `val_every_epochs` must be at least 1.
    /// - Dimensioning fields (`window_frames`, subcarrier counts,
    ///   `backbone_channels`, `heatmap_size`, `num_keypoints`,
    ///   `num_body_parts`, `batch_size`) must not exceed their
    ///   allocation-guard upper bounds (see `MAX_*` constants), so an absurd
    ///   value is rejected here rather than causing an OOM / allocation
    ///   overflow later in the pipeline.
    /// - `gpu_device_id` must be non-negative.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Subcarrier counts
        if self.num_subcarriers == 0 {
            return Err(ConfigError::invalid_value("num_subcarriers", "must be > 0"));
        }
        if self.num_subcarriers > MAX_SUBCARRIERS {
            return Err(ConfigError::invalid_value(
                "num_subcarriers",
                format!("must be <= {MAX_SUBCARRIERS} (allocation guard)"),
            ));
        }
        if self.native_subcarriers == 0 {
            return Err(ConfigError::invalid_value(
                "native_subcarriers",
                "must be > 0",
            ));
        }
        if self.native_subcarriers > MAX_SUBCARRIERS {
            return Err(ConfigError::invalid_value(
                "native_subcarriers",
                format!("must be <= {MAX_SUBCARRIERS} (allocation guard)"),
            ));
        }

        // Antenna counts
        if self.num_antennas_tx == 0 {
            return Err(ConfigError::invalid_value("num_antennas_tx", "must be > 0"));
        }
        if self.num_antennas_rx == 0 {
            return Err(ConfigError::invalid_value("num_antennas_rx", "must be > 0"));
        }

        // Temporal window
        if self.window_frames == 0 {
            return Err(ConfigError::invalid_value("window_frames", "must be > 0"));
        }
        if self.window_frames > MAX_WINDOW_FRAMES {
            return Err(ConfigError::invalid_value(
                "window_frames",
                format!("must be <= {MAX_WINDOW_FRAMES} (allocation guard)"),
            ));
        }

        // Heatmap
        if self.heatmap_size == 0 {
            return Err(ConfigError::invalid_value("heatmap_size", "must be > 0"));
        }
        if self.heatmap_size > MAX_HEATMAP_SIZE {
            return Err(ConfigError::invalid_value(
                "heatmap_size",
                format!("must be <= {MAX_HEATMAP_SIZE} (allocation guard)"),
            ));
        }

        // Model dims
        if self.num_keypoints == 0 {
            return Err(ConfigError::invalid_value("num_keypoints", "must be > 0"));
        }
        if self.num_keypoints > MAX_KEYPOINTS {
            return Err(ConfigError::invalid_value(
                "num_keypoints",
                format!("must be <= {MAX_KEYPOINTS} (allocation guard)"),
            ));
        }
        if self.num_body_parts == 0 {
            return Err(ConfigError::invalid_value("num_body_parts", "must be > 0"));
        }
        if self.num_body_parts > MAX_BODY_PARTS {
            return Err(ConfigError::invalid_value(
                "num_body_parts",
                format!("must be <= {MAX_BODY_PARTS} (allocation guard)"),
            ));
        }
        if self.backbone_channels == 0 {
            return Err(ConfigError::invalid_value(
                "backbone_channels",
                "must be > 0",
            ));
        }
        if self.backbone_channels > MAX_BACKBONE_CHANNELS {
            return Err(ConfigError::invalid_value(
                "backbone_channels",
                format!("must be <= {MAX_BACKBONE_CHANNELS} (allocation guard)"),
            ));
        }

        // Optimisation
        if self.batch_size == 0 {
            return Err(ConfigError::invalid_value("batch_size", "must be > 0"));
        }
        if self.batch_size > MAX_BATCH_SIZE {
            return Err(ConfigError::invalid_value(
                "batch_size",
                format!("must be <= {MAX_BATCH_SIZE} (allocation guard)"),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(ConfigError::invalid_value("learning_rate", "must be > 0.0"));
        }
        if self.weight_decay < 0.0 {
            return Err(ConfigError::invalid_value("weight_decay", "must be >= 0.0"));
        }
        if self.grad_clip_norm <= 0.0 {
            return Err(ConfigError::invalid_value(
                "grad_clip_norm",
                "must be > 0.0",
            ));
        }

        // Epochs
        if self.num_epochs == 0 {
            return Err(ConfigError::invalid_value("num_epochs", "must be > 0"));
        }
        if self.warmup_epochs >= self.num_epochs {
            return Err(ConfigError::invalid_value(
                "warmup_epochs",
                "must be < num_epochs",
            ));
        }

        // LR milestones: must be strictly increasing and within bounds
        let mut prev = 0usize;
        for &m in &self.lr_milestones {
            if m == 0 || m > self.num_epochs {
                return Err(ConfigError::invalid_value(
                    "lr_milestones",
                    "each milestone must be in [1, num_epochs]",
                ));
            }
            if m <= prev {
                return Err(ConfigError::invalid_value(
                    "lr_milestones",
                    "milestones must be strictly increasing",
                ));
            }
            prev = m;
        }

        if self.lr_gamma <= 0.0 || self.lr_gamma >= 1.0 {
            return Err(ConfigError::invalid_value(
                "lr_gamma",
                "must be in (0.0, 1.0)",
            ));
        }

        // Loss weights
        if self.lambda_kp < 0.0 {
            return Err(ConfigError::invalid_value("lambda_kp", "must be >= 0.0"));
        }
        if self.lambda_dp < 0.0 {
            return Err(ConfigError::invalid_value("lambda_dp", "must be >= 0.0"));
        }
        if self.lambda_tr < 0.0 {
            return Err(ConfigError::invalid_value("lambda_tr", "must be >= 0.0"));
        }
        let total_weight = self.lambda_kp + self.lambda_dp + self.lambda_tr;
        if total_weight <= 0.0 {
            return Err(ConfigError::invalid_value(
                "lambda_kp / lambda_dp / lambda_tr",
                "at least one loss weight must be > 0.0",
            ));
        }

        // Validation / checkpoint
        if self.val_every_epochs == 0 {
            return Err(ConfigError::invalid_value(
                "val_every_epochs",
                "must be > 0",
            ));
        }
        if self.save_top_k == 0 {
            return Err(ConfigError::invalid_value("save_top_k", "must be > 0"));
        }

        // Device: a CUDA device index can never be negative.
        if self.gpu_device_id < 0 {
            return Err(ConfigError::invalid_value("gpu_device_id", "must be >= 0"));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_config_is_valid() {
        let cfg = TrainingConfig::default();
        cfg.validate().expect("default config should be valid");
    }

    #[test]
    fn json_round_trip() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("config.json");

        let original = TrainingConfig::default();
        original
            .to_json(&path)
            .expect("serialization should succeed");

        let loaded = TrainingConfig::from_json(&path).expect("deserialization should succeed");
        assert_eq!(loaded.num_subcarriers, original.num_subcarriers);
        assert_eq!(loaded.batch_size, original.batch_size);
        assert_eq!(loaded.seed, original.seed);
        assert_eq!(loaded.lr_milestones, original.lr_milestones);
    }

    #[test]
    fn zero_subcarriers_is_invalid() {
        let cfg = TrainingConfig {
            num_subcarriers: 0,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn negative_learning_rate_is_invalid() {
        let cfg = TrainingConfig {
            learning_rate: -0.001,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn warmup_equal_to_epochs_is_invalid() {
        let default = TrainingConfig::default();
        let cfg = TrainingConfig {
            warmup_epochs: default.num_epochs,
            ..default
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn non_increasing_milestones_are_invalid() {
        let cfg = TrainingConfig {
            lr_milestones: vec![30, 20],
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn milestone_beyond_epochs_is_invalid() {
        let default = TrainingConfig::default();
        let beyond = default.num_epochs + 1;
        let cfg = TrainingConfig {
            lr_milestones: vec![30, beyond],
            ..default
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn all_zero_loss_weights_are_invalid() {
        let cfg = TrainingConfig {
            lambda_kp: 0.0,
            lambda_dp: 0.0,
            lambda_tr: 0.0,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn needs_subcarrier_interp_when_counts_differ() {
        let cfg = TrainingConfig {
            num_subcarriers: 56,
            native_subcarriers: 114,
            ..TrainingConfig::default()
        };
        assert!(cfg.needs_subcarrier_interp());

        let cfg2 = TrainingConfig {
            num_subcarriers: 56,
            native_subcarriers: 56,
            ..TrainingConfig::default()
        };
        assert!(!cfg2.needs_subcarrier_interp());
    }

    // ADR-155 §Tier-2: every preset constructor must still validate after the
    // upper-bound (allocation-guard) checks were added.
    #[test]
    fn presets_still_validate() {
        TrainingConfig::default().validate().expect("default");
        TrainingConfig::mmfi().validate().expect("mmfi");
        TrainingConfig::ht40_192().validate().expect("ht40_192");
        TrainingConfig::multiband_168()
            .validate()
            .expect("multiband_168");
        TrainingConfig::for_subcarriers(168, 56)
            .validate()
            .expect("for_subcarriers");
    }

    // ADR-155 §Tier-2: oversized dimensioning fields (config-OOM class) must be
    // rejected, not passed through to an allocation that overflows / OOMs.
    #[test]
    fn oversized_window_frames_is_invalid() {
        let cfg = TrainingConfig {
            window_frames: MAX_WINDOW_FRAMES + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn oversized_subcarriers_are_invalid() {
        let cfg = TrainingConfig {
            num_subcarriers: MAX_SUBCARRIERS + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
        let cfg = TrainingConfig {
            native_subcarriers: MAX_SUBCARRIERS + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn oversized_backbone_channels_is_invalid() {
        let cfg = TrainingConfig {
            backbone_channels: MAX_BACKBONE_CHANNELS + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn oversized_heatmap_size_is_invalid() {
        let cfg = TrainingConfig {
            heatmap_size: MAX_HEATMAP_SIZE + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn oversized_keypoints_and_body_parts_are_invalid() {
        let cfg = TrainingConfig {
            num_keypoints: MAX_KEYPOINTS + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
        let cfg = TrainingConfig {
            num_body_parts: MAX_BODY_PARTS + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn oversized_batch_size_is_invalid() {
        let cfg = TrainingConfig {
            batch_size: MAX_BATCH_SIZE + 1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn negative_gpu_device_id_is_invalid() {
        let cfg = TrainingConfig {
            gpu_device_id: -1,
            ..TrainingConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_fields_have_expected_defaults() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.num_subcarriers, 56);
        assert_eq!(cfg.native_subcarriers, 114);
        assert_eq!(cfg.num_antennas_tx, 3);
        assert_eq!(cfg.num_antennas_rx, 3);
        assert_eq!(cfg.window_frames, 100);
        assert_eq!(cfg.heatmap_size, 56);
        assert_eq!(cfg.num_keypoints, 17);
        assert_eq!(cfg.num_body_parts, 24);
        assert_eq!(cfg.batch_size, 8);
        assert!((cfg.learning_rate - 1e-3).abs() < 1e-10);
        assert_eq!(cfg.num_epochs, 50);
        assert_eq!(cfg.warmup_epochs, 5);
        assert_eq!(cfg.lr_milestones, vec![30, 45]);
        assert!((cfg.lr_gamma - 0.1).abs() < 1e-10);
        assert!((cfg.lambda_kp - 0.3).abs() < 1e-10);
        assert!((cfg.lambda_dp - 0.6).abs() < 1e-10);
        assert!((cfg.lambda_tr - 0.1).abs() < 1e-10);
        assert_eq!(cfg.seed, 42);
    }
}

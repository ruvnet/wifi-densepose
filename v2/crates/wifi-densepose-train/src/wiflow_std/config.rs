//! Configuration and pure-Rust shape/parameter math for WiFlow-STD
//! (ADR-152 §2.2). See the [module docs](crate::wiflow_std) for provenance.
//!
//! Everything here compiles without the `tch-backend` feature so the
//! architecture's invariants (parameter count, output shapes, divisibility
//! constraints) are unit-testable under `--no-default-features`. The
//! 15-keypoint default must yield exactly **2,225,042** parameters — the
//! count verified against the upstream reference (`RESULTS.md`).

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// TCN kernel size — fixed at 3 in the reference architecture.
pub const TCN_KERNEL: usize = 3;

/// Dropout used inside the 2-D conv blocks (`Dropout2d`). The reference
/// hardcodes 0.3 in `convnet.py` (the model-level `dropout` argument is only
/// forwarded to the TCN), so it is a constant here rather than a config field.
pub const CONV_BLOCK_DROPOUT: f64 = 0.3;

// ---------------------------------------------------------------------------
// TcnGroupsMode
// ---------------------------------------------------------------------------

/// How the group count of each depthwise-grouped TCN convolution is chosen
/// (ADR-152 efficiency sweep, `benchmarks/wiflow-std/remote/sweep/model_compact.py`).
///
/// The upstream reference hardcodes `groups = 20`, which does not divide the
/// compact variants' channel counts (e.g. 270, 135, 85). The sweep's rules:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TcnGroupsMode {
    /// Every grouped conv uses [`WiFlowStdConfig::tcn_groups`] verbatim
    /// (upstream behavior; requires divisibility). Default.
    #[default]
    Fixed,
    /// Per-conv groups = `gcd(channels, tcn_groups)` — equals `tcn_groups`
    /// wherever the upstream choice is valid (incl. the 540-channel input
    /// conv) and falls back to the largest common divisor otherwise.
    /// The sweep's `gcd20` mode (`half` / `quarter` presets).
    Gcd,
    /// Per-conv groups = channels (fully depthwise; `tiny` preset).
    Depthwise,
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn default_input_pw_groups() -> usize {
    1
}

fn default_min_feature_width() -> usize {
    15
}

// ---------------------------------------------------------------------------
// WiFlowStdConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters for the WiFlow-STD pose model (ADR-152 §2.2).
///
/// Defaults reproduce the verified upstream architecture exactly (2,225,042
/// parameters, 15 keypoints). For RuView's ESP32 17-keypoint eval set
/// (ADR-152 §2.2(b)) use [`WiFlowStdConfig::for_keypoints`]`(17)` — the
/// keypoint count only changes the final adaptive pooling, not the parameter
/// count, so retrained 15-keypoint weights remain shape-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WiFlowStdConfig {
    /// CSI input feature dimension (subcarriers × antenna paths flattened).
    /// Must be divisible by [`Self::tcn_groups`]. Default: **540**.
    pub subcarriers: usize,

    /// Temporal window length in CSI frames. Default: **20**.
    pub window: usize,

    /// Output channels of each TCN level (dilation doubles per level:
    /// 1, 2, 4, 8, …). Every entry must be divisible by [`Self::tcn_groups`].
    /// Default: **[540, 440, 340, 240]** — the `models/` code values, *not*
    /// upstream `config.py`'s stale `[480, 360, 240]`.
    pub tcn_channels: Vec<usize>,

    /// Group count for the depthwise-grouped TCN convolutions. The reference
    /// hardcodes **20**; exposed so non-540 subcarrier layouts can keep the
    /// divisibility invariant. Default: **20**. Interpreted per
    /// [`Self::tcn_groups_mode`]: the verbatim group count in `Fixed` mode,
    /// the gcd base in `Gcd` mode, ignored in `Depthwise` mode.
    pub tcn_groups: usize,

    /// Group-selection rule for the TCN's grouped convolutions
    /// (ADR-152 efficiency sweep). Default: [`TcnGroupsMode::Fixed`]
    /// (upstream behavior — every grouped conv uses [`Self::tcn_groups`]).
    #[serde(default)]
    pub tcn_groups_mode: TcnGroupsMode,

    /// Group count for the **first** TCN block's pointwise (1×1) and residual
    /// downsample convs (`subcarriers → tcn_channels[0]`). The sweep's `tiny`
    /// variant uses **4** to break the dense-540-input parameter floor
    /// (~117k params, which alone exceeds tiny's budget); every other config
    /// uses **1** (upstream behavior). Must divide both `subcarriers` and
    /// `tcn_channels[0]`. Default: **1**.
    #[serde(default = "default_input_pw_groups")]
    pub input_pw_groups: usize,

    /// Output channels of the 2-D conv encoder blocks. The first entry is
    /// also `ConvBlock1`'s output; each subsequent block downsamples the
    /// subcarrier axis by 2. Default: **[8, 16, 32, 64]**.
    pub conv_channels: Vec<usize>,

    /// Attention head groups for the dual axial attention. Must divide the
    /// last entry of [`Self::conv_channels`]. Default: **8**.
    pub attention_groups: usize,

    /// Number of 2-D keypoints produced. Default: **15** (upstream skeleton);
    /// use **17** for RuView's COCO-skeleton ESP32 eval set. Only changes the
    /// parameter-free final adaptive pool — never the trunk: the stride
    /// schedule is governed by [`Self::min_feature_width`], so 15- and
    /// 17-keypoint variants share the identical conv graph and weights
    /// (matching the validated Python protocol,
    /// `benchmarks/wiflow-std/remote/measb/train_measb.py`, which swaps only
    /// `avg_pool` and loads the pretrained state_dict `strict=True`).
    pub keypoints: usize,

    /// Floor for the conv encoder's width downsampling: each
    /// `AsymmetricConvBlock` halves the width only while the result stays
    /// ≥ this value (see [`Self::conv_strides`]).
    ///
    /// Default: **15** — the upstream constant. Provenance: the reference's
    /// four hardcoded stride-2 blocks exist because its 240-channel TCN
    /// output halves cleanly four times, 240 / 2⁴ = 15. The compact presets'
    /// schedules were derived with this same floor. Override only when
    /// designing a new trunk; do **not** couple it to [`Self::keypoints`] —
    /// the adaptive pool maps the decoder height to any keypoint count.
    #[serde(default = "default_min_feature_width")]
    pub min_feature_width: usize,

    /// Elementwise dropout probability inside the TCN blocks, in `[0, 1)`.
    /// Default: **0.5** (the value used by our verified retraining run).
    pub dropout: f64,
}

impl Default for WiFlowStdConfig {
    fn default() -> Self {
        WiFlowStdConfig {
            subcarriers: 540,
            window: 20,
            tcn_channels: vec![540, 440, 340, 240],
            tcn_groups: 20,
            tcn_groups_mode: TcnGroupsMode::Fixed,
            input_pw_groups: 1,
            conv_channels: vec![8, 16, 32, 64],
            attention_groups: 8,
            keypoints: 15,
            min_feature_width: 15,
            dropout: 0.5,
        }
    }
}

impl WiFlowStdConfig {
    /// Default architecture with a different keypoint count (e.g. 17 for the
    /// ESP32 COCO-skeleton eval set, ADR-152 §2.2(b)).
    ///
    /// The trunk is untouched: [`Self::min_feature_width`] stays at the
    /// upstream floor of 15, so e.g. `for_keypoints(17)` keeps the trained
    /// `[2, 2, 2, 2]` stride schedule (feature width 15) and the adaptive
    /// pool maps 15 → 17 — exactly the validated Python protocol
    /// (`benchmarks/wiflow-std/remote/measb/train_measb.py`).
    pub fn for_keypoints(keypoints: usize) -> Self {
        WiFlowStdConfig {
            keypoints,
            ..Self::default()
        }
    }

    /// **half** compact preset (ADR-152 efficiency sweep, trained
    /// 2026-06-10/11): **843,834** parameters (0.38×), clean-test PCK@20
    /// **96.62%** — strictly dominates the full reference on its own
    /// benchmark. Per-conv groups = `gcd(channels, 20)`; stride schedule
    /// derives to `[2, 2, 2, 1]`. See
    /// `benchmarks/wiflow-std/results/efficiency_sweep.jsonl`.
    pub fn half() -> Self {
        WiFlowStdConfig {
            tcn_channels: vec![270, 220, 170, 120],
            tcn_groups_mode: TcnGroupsMode::Gcd,
            conv_channels: vec![4, 8, 16, 32],
            attention_groups: 4,
            ..Self::default()
        }
    }

    /// **quarter** compact preset (ADR-152 efficiency sweep): **338,600**
    /// parameters (0.15×), clean-test PCK@20 **96.05%**. Per-conv groups =
    /// `gcd(channels, 20)`; stride schedule derives to `[2, 2, 1, 1]`.
    pub fn quarter() -> Self {
        WiFlowStdConfig {
            tcn_channels: vec![135, 110, 85, 60],
            tcn_groups_mode: TcnGroupsMode::Gcd,
            conv_channels: vec![2, 4, 8, 16],
            attention_groups: 2,
            ..Self::default()
        }
    }

    /// **tiny** compact preset (ADR-152 efficiency sweep): **56,290**
    /// parameters (0.025×), clean-test PCK@20 **94.11%** — the smallest
    /// deployable WiFlow-class model (~220 KB fp32). Fully depthwise TCN
    /// groups plus `input_pw_groups = 4` on the first block's pointwise /
    /// downsample convs; stride schedule derives to `[2, 1, 1, 1]`
    /// (feature width 16).
    pub fn tiny() -> Self {
        WiFlowStdConfig {
            tcn_channels: vec![68, 56, 44, 32],
            tcn_groups_mode: TcnGroupsMode::Depthwise,
            input_pw_groups: 4,
            conv_channels: vec![2, 4, 8, 16],
            attention_groups: 2,
            ..Self::default()
        }
    }

    /// Validate all architectural invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidValue`] naming the offending field.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.subcarriers == 0 {
            return Err(ConfigError::invalid_value("subcarriers", "must be >= 1"));
        }
        if self.window == 0 {
            return Err(ConfigError::invalid_value("window", "must be >= 1"));
        }
        if self.tcn_groups == 0 {
            return Err(ConfigError::invalid_value("tcn_groups", "must be >= 1"));
        }
        // In Gcd mode the per-conv group count is gcd(channels, tcn_groups)
        // and in Depthwise mode it is the channel count itself, so the
        // divisibility invariant holds by construction; only Fixed mode
        // (upstream behavior) needs the explicit checks.
        let fixed = self.tcn_groups_mode == TcnGroupsMode::Fixed;
        if fixed && self.subcarriers % self.tcn_groups != 0 {
            return Err(ConfigError::invalid_value(
                "subcarriers",
                format!(
                    "{} is not divisible by tcn_groups={} (grouped conv requirement)",
                    self.subcarriers, self.tcn_groups
                ),
            ));
        }
        if self.tcn_channels.is_empty() {
            return Err(ConfigError::invalid_value(
                "tcn_channels",
                "must contain at least one level",
            ));
        }
        for (i, &c) in self.tcn_channels.iter().enumerate() {
            if c == 0 || (fixed && c % self.tcn_groups != 0) {
                return Err(ConfigError::invalid_value(
                    "tcn_channels",
                    format!(
                        "level {i} has {c} channels; must be > 0 and divisible by tcn_groups={}",
                        self.tcn_groups
                    ),
                ));
            }
        }
        if self.input_pw_groups == 0
            || self.subcarriers % self.input_pw_groups != 0
            || self.tcn_channels[0] % self.input_pw_groups != 0
        {
            return Err(ConfigError::invalid_value(
                "input_pw_groups",
                format!(
                    "{} must be >= 1 and divide both subcarriers={} and tcn_channels[0]={}",
                    self.input_pw_groups, self.subcarriers, self.tcn_channels[0]
                ),
            ));
        }
        if self.conv_channels.is_empty() {
            return Err(ConfigError::invalid_value(
                "conv_channels",
                "must contain at least one block",
            ));
        }
        if self.conv_channels.iter().any(|&c| c == 0) {
            return Err(ConfigError::invalid_value(
                "conv_channels",
                "all blocks must have > 0 channels",
            ));
        }
        let c_last = *self.conv_channels.last().expect("non-empty checked above");
        if self.attention_groups == 0 || c_last % self.attention_groups != 0 {
            return Err(ConfigError::invalid_value(
                "attention_groups",
                format!(
                    "{} must be >= 1 and divide the last conv channel count {c_last}",
                    self.attention_groups
                ),
            ));
        }
        if c_last < 2 || c_last % 2 != 0 {
            return Err(ConfigError::invalid_value(
                "conv_channels",
                format!("last block has {c_last} channels; decoder needs an even count >= 2"),
            ));
        }
        if self.keypoints == 0 {
            return Err(ConfigError::invalid_value("keypoints", "must be >= 1"));
        }
        if self.min_feature_width == 0 {
            return Err(ConfigError::invalid_value(
                "min_feature_width",
                "must be >= 1",
            ));
        }
        if !self.dropout.is_finite() || !(0.0..1.0).contains(&self.dropout) {
            return Err(ConfigError::invalid_value(
                "dropout",
                format!("{} is outside [0, 1)", self.dropout),
            ));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shape inference
    // -----------------------------------------------------------------------

    /// Channel count produced by the TCN stack (last TCN level). This is the
    /// *width* of the image-like tensor fed to the 2-D encoder.
    pub fn tcn_output_channels(&self) -> usize {
        *self.tcn_channels.last().unwrap_or(&0)
    }

    /// Group count of a grouped TCN conv over `channels` channels, per
    /// [`Self::tcn_groups_mode`].
    pub fn tcn_conv_groups(&self, channels: usize) -> usize {
        match self.tcn_groups_mode {
            TcnGroupsMode::Fixed => self.tcn_groups,
            TcnGroupsMode::Gcd => gcd(channels, self.tcn_groups),
            TcnGroupsMode::Depthwise => channels,
        }
    }

    /// Width stride of each `AsymmetricConvBlock`, derived with the sweep's
    /// rule (`model_compact.py::compute_strides`): halve the width
    /// (`w → ceil(w / 2)`, the `(1,3)`-kernel stride-2 output size) only
    /// while the result stays ≥ [`Self::min_feature_width`]. At the upstream
    /// default (240 TCN channels, floor 15) this derives `[2, 2, 2, 2]` —
    /// the hardcoded upstream schedule, exactly.
    ///
    /// Deliberately independent of [`Self::keypoints`]: the keypoint count
    /// only changes the parameter-free adaptive pool, so retargeting the
    /// skeleton (e.g. [`Self::for_keypoints`]`(17)`) keeps the trained graph
    /// and the pool maps `feature_width() → keypoints`.
    pub fn conv_strides(&self) -> Vec<usize> {
        let mut w = self.tcn_output_channels();
        let mut strides = Vec::with_capacity(self.conv_channels.len());
        for _ in &self.conv_channels {
            let next = w.div_ceil(2);
            if next >= self.min_feature_width {
                strides.push(2);
                w = next;
            } else {
                strides.push(1);
            }
        }
        strides
    }

    /// Width of the encoder feature map after the conv blocks.
    ///
    /// `ConvBlock1` preserves width; each `AsymmetricConvBlock` applies a
    /// `(1, 3)` kernel with padding `(0, 1)` and the per-block stride from
    /// [`Self::conv_strides`]. Default: 240 → 120 → 60 → 30 → **15**.
    pub fn feature_width(&self) -> usize {
        let mut w = self.tcn_output_channels();
        for s in self.conv_strides() {
            if s == 2 {
                w = w.div_ceil(2);
            }
        }
        w
    }

    /// Mid-channel count of the decoder's 3×3 conv:
    /// `max(conv_channels.last() / 2, 4)` (the sweep's floor of 4 keeps the
    /// decoder viable at very small widths; identical to the upstream `c / 2`
    /// for every channel count ≥ 8, including the default 64 → 32).
    pub fn decoder_mid(&self) -> usize {
        (self.conv_channels.last().unwrap_or(&0) / 2).max(4)
    }

    /// Output tensor shape `(batch, keypoints, 2)`. The adaptive average pool
    /// maps the feature height to `keypoints` regardless of its size, so the
    /// keypoint count is free (15 and 17 share identical weights).
    pub fn output_shape(&self, batch: usize) -> (usize, usize, usize) {
        (batch, self.keypoints, 2)
    }

    // -----------------------------------------------------------------------
    // Parameter-count formula
    // -----------------------------------------------------------------------

    /// Total trainable parameter count, derived layer-by-layer from the
    /// architecture (BatchNorm weight+bias counted; running stats are buffers
    /// and excluded, matching PyTorch's `numel` convention).
    ///
    /// Pins the port against the verified reference: the 15-keypoint default
    /// must equal **2,225,042** (`RESULTS.md` artifact verification).
    ///
    /// Returns **0** for any config that fails [`Self::validate`]: the
    /// formula is only meaningful for buildable architectures (an invalid
    /// config would otherwise index an empty `conv_channels` or divide by a
    /// zero group count). Call `validate()` first when you need the reason.
    pub fn param_count(&self) -> usize {
        if self.validate().is_err() {
            return 0;
        }

        let mut total = 0;

        // TCN stack: per-conv groups follow tcn_groups_mode; only the first
        // block's pointwise/downsample convs use input_pw_groups.
        let mut c_in = self.subcarriers;
        for (i, &c_out) in self.tcn_channels.iter().enumerate() {
            let pw_groups = if i == 0 { self.input_pw_groups } else { 1 };
            total += tcn_block_params(
                c_in,
                c_out,
                TCN_KERNEL,
                self.tcn_conv_groups(c_in),
                self.tcn_conv_groups(c_out),
                pw_groups,
            );
            c_in = c_out;
        }

        // ConvBlock1 (1 → conv_channels[0]) + asymmetric blocks. Both block
        // kinds have identical parameter shapes (stride changes nothing).
        let mut c_in = 1;
        total += conv_block_params(c_in, self.conv_channels[0]);
        c_in = self.conv_channels[0];
        for &c_out in &self.conv_channels {
            total += conv_block_params(c_in, c_out);
            c_in = c_out;
        }

        // Dual axial attention: width axis + height axis, both c_in → c_in.
        total += 2 * axial_attention_params(c_in, self.attention_groups);

        // Decoder: 3×3 conv (c → decoder_mid) + BN + 1×1 conv (mid → 2) + BN.
        total += decoder_params(c_in, self.decoder_mid());

        total
    }
}

// ---------------------------------------------------------------------------
// Per-component parameter formulas
// ---------------------------------------------------------------------------

/// One `InnerGroupedTemporalBlock`: two (depthwise-grouped conv → BN →
/// pointwise conv → BN) stages plus a 1×1 + BN residual projection when the
/// channel count changes. All convs are bias-free. `g_in`/`g_out` are the
/// group counts of the two grouped convs (each conv groups over its own
/// channel count — they differ in `Gcd`/`Depthwise` mode); `pw_groups`
/// groups the first pointwise conv and the residual projection (the sweep's
/// `input_pw_groups`, block 0 only — 1 everywhere else).
fn tcn_block_params(
    c_in: usize,
    c_out: usize,
    k: usize,
    g_in: usize,
    g_out: usize,
    pw_groups: usize,
) -> usize {
    let grouped1 = c_in * (c_in / g_in) * k; // depthwise-grouped, c_in → c_in
    let bn1g = 2 * c_in;
    let pw1 = c_out * (c_in / pw_groups); // pointwise 1×1
    let bn1p = 2 * c_out;
    let grouped2 = c_out * (c_out / g_out) * k;
    let bn2g = 2 * c_out;
    let pw2 = c_out * c_out;
    let bn2p = 2 * c_out;
    let downsample = if c_in != c_out {
        (c_in / pw_groups) * c_out + 2 * c_out
    } else {
        0
    };
    grouped1 + bn1g + pw1 + bn1p + grouped2 + bn2g + pw2 + bn2p + downsample
}

/// One `ConvBlock1` / `AsymmetricConvBlock`: three (1, 3) convs **with bias**
/// + BN each, plus a bias-free 1×1 + BN residual projection.
fn conv_block_params(c_in: usize, c_out: usize) -> usize {
    let conv1 = c_out * c_in * 3 + c_out;
    let conv_rest = 2 * (c_out * c_out * 3 + c_out);
    let bns = 3 * 2 * c_out;
    let downsample = c_in * c_out + 2 * c_out;
    conv1 + conv_rest + bns + downsample
}

/// One `AxialAttention` axis: bias-free 1×1 qkv conv (c → 3c), BN over the
/// 3c qkv channels, BN over the `groups` similarity maps, BN over the output.
fn axial_attention_params(c: usize, groups: usize) -> usize {
    let qkv = c * 3 * c;
    let bn_qkv = 2 * (3 * c);
    let bn_similarity = 2 * groups;
    let bn_output = 2 * c;
    qkv + bn_qkv + bn_similarity + bn_output
}

/// Decoder: `Conv2d(c → mid, 3×3, bias)` + BN + `Conv2d(mid → 2, 1×1, bias)`
/// + BN, where `mid` = [`WiFlowStdConfig::decoder_mid`].
fn decoder_params(c: usize, mid: usize) -> usize {
    let conv1 = mid * c * 9 + mid;
    let bn1 = 2 * mid;
    let conv2 = 2 * mid + 2;
    let bn2 = 2 * 2;
    conv1 + bn1 + conv2 + bn2
}

// ---------------------------------------------------------------------------
// Tests (pure Rust — run under --no-default-features)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference parameter count verified against the upstream checkpoint
    /// and `torchinfo` (benchmarks/wiflow-std/RESULTS.md, 2026-06-10).
    const REFERENCE_PARAMS: usize = 2_225_042;

    #[test]
    fn default_config_is_valid() {
        WiFlowStdConfig::default()
            .validate()
            .expect("default config must validate");
    }

    #[test]
    fn default_param_count_matches_verified_reference() {
        assert_eq!(WiFlowStdConfig::default().param_count(), REFERENCE_PARAMS);
    }

    #[test]
    fn param_count_is_independent_of_keypoints() {
        // The keypoint count only changes the parameter-free adaptive pool,
        // so 15- and 17-keypoint variants share identical weights.
        let kp17 = WiFlowStdConfig::for_keypoints(17);
        kp17.validate().expect("17-keypoint config must validate");
        assert_eq!(kp17.param_count(), REFERENCE_PARAMS);
    }

    #[test]
    fn per_component_breakdown_matches_hand_calculation() {
        // TCN levels (hand-verified against the reference layer shapes).
        assert_eq!(tcn_block_params(540, 540, 3, 20, 20, 1), 675_000);
        assert_eq!(tcn_block_params(540, 440, 3, 20, 20, 1), 746_180);
        assert_eq!(tcn_block_params(440, 340, 3, 20, 20, 1), 464_780);
        assert_eq!(tcn_block_params(340, 240, 3, 20, 20, 1), 249_380);
        // Conv encoder.
        assert_eq!(conv_block_params(1, 8), 504);
        assert_eq!(conv_block_params(8, 8), 728);
        assert_eq!(conv_block_params(8, 16), 2_224);
        assert_eq!(conv_block_params(16, 32), 8_544);
        assert_eq!(conv_block_params(32, 64), 33_472);
        // Attention + decoder.
        assert_eq!(axial_attention_params(64, 8), 12_816);
        assert_eq!(decoder_params(64, 32), 18_598);
    }

    // -----------------------------------------------------------------------
    // ADR-152 efficiency-sweep compact presets. The parameter pins are
    // GROUND TRUTH measured from the trained PyTorch checkpoints
    // (benchmarks/wiflow-std/results/efficiency_sweep.jsonl, 2026-06-11):
    // any mismatch means the Rust formula or config mapping is wrong.
    // -----------------------------------------------------------------------

    #[test]
    fn half_preset_param_count_matches_trained_checkpoint() {
        let cfg = WiFlowStdConfig::half();
        cfg.validate().expect("half preset must validate");
        assert_eq!(cfg.param_count(), 843_834);
    }

    #[test]
    fn quarter_preset_param_count_matches_trained_checkpoint() {
        let cfg = WiFlowStdConfig::quarter();
        cfg.validate().expect("quarter preset must validate");
        assert_eq!(cfg.param_count(), 338_600);
    }

    #[test]
    fn tiny_preset_param_count_matches_trained_checkpoint() {
        let cfg = WiFlowStdConfig::tiny();
        cfg.validate().expect("tiny preset must validate");
        assert_eq!(cfg.param_count(), 56_290);
    }

    #[test]
    fn preset_tcn_groups_match_sweep_per_block_record() {
        // efficiency_sweep.jsonl "tcn_groups_per_block": (conv1, conv2) of
        // each block — conv1 groups over c_in, conv2 over c_out.
        let half = WiFlowStdConfig::half();
        let groups: Vec<(usize, usize)> = {
            let mut c_in = half.subcarriers;
            half.tcn_channels
                .iter()
                .map(|&c_out| {
                    let g = (half.tcn_conv_groups(c_in), half.tcn_conv_groups(c_out));
                    c_in = c_out;
                    g
                })
                .collect()
        };
        assert_eq!(groups, [(20, 10), (10, 20), (20, 10), (10, 20)]);

        let tiny = WiFlowStdConfig::tiny();
        assert_eq!(tiny.tcn_conv_groups(540), 540); // depthwise input conv
        assert_eq!(tiny.tcn_conv_groups(68), 68);
    }

    #[test]
    fn preset_stride_schedules_match_sweep_record() {
        // efficiency_sweep.jsonl "conv_strides" / "final_width".
        assert_eq!(WiFlowStdConfig::default().conv_strides(), [2, 2, 2, 2]);
        assert_eq!(WiFlowStdConfig::half().conv_strides(), [2, 2, 2, 1]);
        assert_eq!(WiFlowStdConfig::quarter().conv_strides(), [2, 2, 1, 1]);
        assert_eq!(WiFlowStdConfig::tiny().conv_strides(), [2, 1, 1, 1]);
        assert_eq!(WiFlowStdConfig::half().feature_width(), 15);
        assert_eq!(WiFlowStdConfig::quarter().feature_width(), 15);
        assert_eq!(WiFlowStdConfig::tiny().feature_width(), 16);
    }

    #[test]
    fn for_keypoints_17_keeps_trained_trunk_and_pools_15_to_17() {
        // Pin against the validated Python protocol (train_measb.py): K=17
        // swaps only the adaptive pool, never the stride schedule. A derived
        // [2, 2, 2, 1]/width-30 graph here would silently diverge from the
        // trained [2, 2, 2, 2]/width-15 checkpoint.
        let cfg = WiFlowStdConfig::for_keypoints(17);
        assert_eq!(cfg.min_feature_width, 15);
        assert_eq!(cfg.conv_strides(), [2, 2, 2, 2]);
        assert_eq!(cfg.feature_width(), 15);
        assert_eq!(cfg.output_shape(1), (1, 17, 2));
    }

    #[test]
    fn min_feature_width_override_changes_schedule_as_designed() {
        // Raising the floor stops the downsampling earlier (240 → 30).
        let cfg = WiFlowStdConfig {
            min_feature_width: 30,
            ..Default::default()
        };
        cfg.validate().expect("floor 30 validates");
        assert_eq!(cfg.conv_strides(), [2, 2, 2, 1]);
        assert_eq!(cfg.feature_width(), 30);

        // Lowering it lets a small trunk halve further (tiny: 32 → 8).
        let cfg = WiFlowStdConfig {
            min_feature_width: 8,
            ..WiFlowStdConfig::tiny()
        };
        cfg.validate().expect("floor 8 validates");
        assert_eq!(cfg.conv_strides(), [2, 2, 1, 1]);
        assert_eq!(cfg.feature_width(), 8);
    }

    #[test]
    fn rejects_zero_min_feature_width() {
        let cfg = WiFlowStdConfig {
            min_feature_width: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn param_count_returns_zero_for_invalid_configs() {
        // Documented total behavior: configs that fail validate() yield 0
        // instead of panicking (OOB index / division by zero).
        for cfg in [
            WiFlowStdConfig {
                conv_channels: vec![],
                ..Default::default()
            },
            WiFlowStdConfig {
                tcn_groups: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                input_pw_groups: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                tcn_channels: vec![],
                ..Default::default()
            },
        ] {
            assert!(cfg.validate().is_err(), "precondition: {cfg:?} is invalid");
            assert_eq!(cfg.param_count(), 0, "no panic, returns 0: {cfg:?}");
        }
    }

    #[test]
    fn fixed_mode_with_defaults_is_unchanged_by_new_knobs() {
        // The new fields default to upstream behavior: gcd(c, 20) == 20 for
        // every default channel count, so Gcd mode is also a no-op there.
        let mut cfg = WiFlowStdConfig::default();
        assert_eq!(cfg.param_count(), REFERENCE_PARAMS);
        cfg.tcn_groups_mode = TcnGroupsMode::Gcd;
        cfg.validate().expect("gcd mode validates at defaults");
        assert_eq!(cfg.param_count(), REFERENCE_PARAMS);
        assert_eq!(WiFlowStdConfig::default().decoder_mid(), 32);
    }

    #[test]
    fn rejects_bad_input_pw_groups() {
        // 7 divides neither 540 nor 540's first TCN level.
        let cfg = WiFlowStdConfig {
            input_pw_groups: 7,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        // 27 divides subcarriers=540 but not tiny's tcn_channels[0]=68.
        let cfg = WiFlowStdConfig {
            input_pw_groups: 27,
            ..WiFlowStdConfig::tiny()
        };
        assert!(cfg.validate().is_err());
        let zero = WiFlowStdConfig {
            input_pw_groups: 0,
            ..Default::default()
        };
        assert!(zero.validate().is_err());
    }

    #[test]
    fn serde_defaults_for_new_fields_are_backward_compatible() {
        // A config serialized before the compact-variant knobs existed must
        // deserialize to upstream behavior (Fixed mode, input_pw_groups 1).
        let legacy = r#"{
            "subcarriers": 540, "window": 20,
            "tcn_channels": [540, 440, 340, 240], "tcn_groups": 20,
            "conv_channels": [8, 16, 32, 64], "attention_groups": 8,
            "keypoints": 15, "dropout": 0.5
        }"#;
        let cfg: WiFlowStdConfig = serde_json::from_str(legacy).expect("deserialize");
        assert_eq!(cfg, WiFlowStdConfig::default());
        assert_eq!(cfg.param_count(), REFERENCE_PARAMS);
    }

    #[test]
    fn serde_roundtrip_preserves_presets() {
        for cfg in [
            WiFlowStdConfig::half(),
            WiFlowStdConfig::quarter(),
            WiFlowStdConfig::tiny(),
        ] {
            let json = serde_json::to_string(&cfg).expect("serialize");
            let back: WiFlowStdConfig = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, cfg);
        }
    }

    #[test]
    fn output_shape_default_and_esp32() {
        assert_eq!(WiFlowStdConfig::default().output_shape(4), (4, 15, 2));
        assert_eq!(
            WiFlowStdConfig::for_keypoints(17).output_shape(1),
            (1, 17, 2)
        );
    }

    #[test]
    fn feature_width_default_is_15() {
        // 240 → 120 → 60 → 30 → 15 (four stride-(1,2) blocks).
        assert_eq!(WiFlowStdConfig::default().feature_width(), 15);
    }

    #[test]
    fn tcn_output_channels_default_is_240() {
        assert_eq!(WiFlowStdConfig::default().tcn_output_channels(), 240);
    }

    #[test]
    fn rejects_subcarriers_not_divisible_by_groups() {
        let cfg = WiFlowStdConfig {
            subcarriers: 541,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_dimensions() {
        for cfg in [
            WiFlowStdConfig {
                subcarriers: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                window: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                keypoints: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                tcn_groups: 0,
                ..Default::default()
            },
        ] {
            assert!(cfg.validate().is_err(), "expected rejection: {cfg:?}");
        }
    }

    #[test]
    fn rejects_empty_or_indivisible_tcn_channels() {
        let empty = WiFlowStdConfig {
            tcn_channels: vec![],
            ..Default::default()
        };
        assert!(empty.validate().is_err());

        let indivisible = WiFlowStdConfig {
            tcn_channels: vec![540, 441],
            ..Default::default()
        };
        assert!(indivisible.validate().is_err());
    }

    #[test]
    fn rejects_bad_conv_channels() {
        let empty = WiFlowStdConfig {
            conv_channels: vec![],
            ..Default::default()
        };
        assert!(empty.validate().is_err());

        let zero = WiFlowStdConfig {
            conv_channels: vec![8, 0, 64],
            ..Default::default()
        };
        assert!(zero.validate().is_err());

        // Odd last channel breaks the c → c/2 decoder split.
        let odd_last = WiFlowStdConfig {
            conv_channels: vec![8, 16, 33],
            attention_groups: 1,
            ..Default::default()
        };
        assert!(odd_last.validate().is_err());
    }

    #[test]
    fn rejects_attention_group_mismatch() {
        let cfg = WiFlowStdConfig {
            attention_groups: 7, // 64 % 7 != 0
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        let zero = WiFlowStdConfig {
            attention_groups: 0,
            ..Default::default()
        };
        assert!(zero.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_dropout() {
        for d in [1.0, 1.5, -0.1, f64::NAN] {
            let cfg = WiFlowStdConfig {
                dropout: d,
                ..Default::default()
            };
            assert!(cfg.validate().is_err(), "dropout {d} must be rejected");
        }
    }

    #[test]
    fn serde_roundtrip_preserves_config() {
        let cfg = WiFlowStdConfig::for_keypoints(17);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: WiFlowStdConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }
}

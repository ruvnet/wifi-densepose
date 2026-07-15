//! WiFlow-STD forward pass (tch-rs / LibTorch backend, ADR-152 §2.2).
//!
//! Idiomatic reimplementation of the DY2434 reference (Apache-2.0); see the
//! [module docs](crate::wiflow_std) for provenance and the evidence grade.
//! From-scratch init: BN gamma is pinned to 1 (see `layers::bn_cfg`); the
//! axial-attention qkv conv uses `N(0, sqrt(1/in_planes))` per the
//! reference's `attention.py` intent (note the reference's *effective* init
//! differs — its `_initialize_weights` re-inits every `nn.Conv1d`, qkv
//! included, with `kaiming_normal(fan_out)`); conv weights keep tch defaults
//! (kaiming-uniform fan_in), which differ in scale from PyTorch's defaults.
//! These divergences affect from-scratch training dynamics only — BN absorbs
//! them at init, and loaded checkpoints overwrite everything. The
//! retrained PyTorch checkpoint loads via [`WiFlowStdModel::load`] after
//! key-remapped safetensors export
//! (`benchmarks/wiflow-std/export_to_safetensors.py`); numerical parity with
//! the PyTorch forward pass is proven by
//! `tests/test_wiflow_std_parity.rs` (max abs diff ~1.2e-7).

use tch::{nn, Device, Tensor};

use super::config::WiFlowStdConfig;
use super::layers::{ConvBlock, DualAxialAttention, GroupedTemporalBlock};
use crate::error::TrainError;

// ---------------------------------------------------------------------------
// WiFlowStdModel
// ---------------------------------------------------------------------------

/// WiFlow-STD pose model: TCN temporal encoder → asymmetric 2-D conv encoder
/// → dual axial attention → conv decoder → adaptive pool to `(K, 2)` keypoints.
///
/// Input: `[B, subcarriers, window]` CSI amplitudes.
/// Output: `[B, keypoints, 2]` normalised 2-D keypoint coordinates.
pub struct WiFlowStdModel {
    vs: nn::VarStore,
    tcn: Vec<GroupedTemporalBlock>,
    conv_in: ConvBlock,
    conv_blocks: Vec<ConvBlock>,
    attention: DualAxialAttention,
    dec_conv1: nn::Conv2D,
    dec_bn1: nn::BatchNorm,
    dec_conv2: nn::Conv2D,
    dec_bn2: nn::BatchNorm,
    /// Active model configuration.
    pub config: WiFlowStdConfig,
}

impl WiFlowStdModel {
    /// Build a new model with randomly-initialised weights on `device`.
    ///
    /// Call `tch::manual_seed(seed)` before this for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::Config`] if `config.validate()` fails.
    pub fn new(config: &WiFlowStdConfig, device: Device) -> Result<Self, TrainError> {
        config.validate()?;

        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // TCN stack: dilation doubles per level, causal padding. Per-conv
        // groups follow `config.tcn_groups_mode`; only block 0's pointwise/
        // downsample convs use `config.input_pw_groups` (ADR-152 sweep).
        let mut tcn = Vec::with_capacity(config.tcn_channels.len());
        let mut c_in = config.subcarriers;
        for (i, &c_out) in config.tcn_channels.iter().enumerate() {
            let dilation = 1_i64 << i;
            let pw_groups = if i == 0 { config.input_pw_groups } else { 1 };
            tcn.push(GroupedTemporalBlock::new(
                &root / format!("tcn{i}"),
                c_in as i64,
                c_out as i64,
                dilation,
                config.tcn_conv_groups(c_in) as i64,
                config.tcn_conv_groups(c_out) as i64,
                pw_groups as i64,
                config.dropout,
            ));
            c_in = c_out;
        }

        // 2-D conv encoder: ConvBlock1 (stride 1) + asymmetric blocks with
        // the derived stride schedule ([2, 2, 2, 2] at the upstream default).
        let c0 = config.conv_channels[0] as i64;
        let conv_in = ConvBlock::new(&root / "conv_in", 1, c0, 1);
        let mut conv_blocks = Vec::with_capacity(config.conv_channels.len());
        let strides = config.conv_strides();
        let mut c_in = c0;
        for (i, &c_out) in config.conv_channels.iter().enumerate() {
            conv_blocks.push(ConvBlock::new(
                &root / format!("conv{i}"),
                c_in,
                c_out as i64,
                strides[i] as i64,
            ));
            c_in = c_out as i64;
        }

        let attention =
            DualAxialAttention::new(&root / "attention", c_in, config.attention_groups as i64);

        // Decoder: c → decoder_mid (3×3) → 2 (1×1), BN + SiLU after each conv.
        let mid = config.decoder_mid() as i64;
        let dec_conv1 = nn::conv2d(
            &root / "dec_conv1",
            c_in,
            mid,
            3,
            nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let dec_bn1 = nn::batch_norm2d(&root / "dec_bn1", mid, super::layers::bn_cfg());
        let dec_conv2 = nn::conv2d(&root / "dec_conv2", mid, 2, 1, Default::default());
        let dec_bn2 = nn::batch_norm2d(&root / "dec_bn2", 2, super::layers::bn_cfg());

        Ok(WiFlowStdModel {
            vs,
            tcn,
            conv_in,
            conv_blocks,
            attention,
            dec_conv1,
            dec_bn1,
            dec_conv2,
            dec_bn2,
            config: config.clone(),
        })
    }

    /// Forward pass in training mode (dropout active, BN in train mode).
    ///
    /// `csi`: `[B, subcarriers, window]` → `[B, keypoints, 2]`.
    pub fn forward_t(&self, csi: &Tensor) -> Tensor {
        self.forward_impl(csi, true)
    }

    /// Forward pass without gradient tracking (inference mode).
    pub fn forward_inference(&self, csi: &Tensor) -> Tensor {
        tch::no_grad(|| self.forward_impl(csi, false))
    }

    /// Save model weights. The tch `VarStore` dispatches the format on the
    /// file extension: `.safetensors` → safetensors, anything else → torch
    /// `.pt`.
    ///
    /// **Platform constraint:** prefer `.safetensors`. The `.pt` path
    /// (`_save_parameters`/`_load_parameters`) is broken on Windows with
    /// torch 2.11 (GenericDict internal assert on the load roundtrip — see
    /// the `save_and_load_roundtrip` test below), and the verified retrained
    /// checkpoint is shipped as key-remapped safetensors anyway
    /// (`benchmarks/wiflow-std/export_to_safetensors.py`).
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::TrainingStep`] if the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> Result<(), TrainError> {
        self.vs
            .save(path)
            .map_err(|e| TrainError::training_step(format!("save failed: {e}")))
    }

    /// Load model weights from a file (format dispatched on extension; see
    /// the `.pt`-on-Windows caveat on [`Self::save`]).
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::TrainingStep`] if the file cannot be read or the
    /// weights are incompatible with this architecture.
    pub fn load(&mut self, path: &std::path::Path) -> Result<(), TrainError> {
        self.vs
            .load(path)
            .map_err(|e| TrainError::training_step(format!("load failed: {e}")))
    }

    /// Reference to the internal `VarStore` (e.g. to build an optimiser).
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Mutable access to the internal `VarStore`.
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Total number of trainable scalar parameters. Must equal
    /// [`WiFlowStdConfig::param_count`] (2,225,042 at the default config).
    pub fn num_parameters(&self) -> i64 {
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum()
    }

    fn forward_impl(&self, csi: &Tensor, train: bool) -> Tensor {
        // TCN: [B, subcarriers, T] → [B, c_tcn, T].
        let mut h = csi.shallow_clone();
        for block in &self.tcn {
            h = block.forward_t(&h, train);
        }

        // Image-like reshape: [B, c_tcn, T] → [B, 1, T, c_tcn].
        let h = h.transpose(1, 2).unsqueeze(1);

        // 2-D conv encoder: [B, 1, T, S] → [B, C, T, S'].
        let mut h = self.conv_in.forward_t(&h, train);
        for block in &self.conv_blocks {
            h = block.forward_t(&h, train);
        }

        // Swap to [B, C, S', T] for the axial attention + decoder.
        let h = h.permute([0, 1, 3, 2]);
        let h = self.attention.forward_t(&h, train);

        // Decoder: [B, C, S', T] → [B, 2, S', T].
        let h = h
            .apply(&self.dec_conv1)
            .apply_t(&self.dec_bn1, train)
            .silu()
            .apply(&self.dec_conv2)
            .apply_t(&self.dec_bn2, train)
            .silu();

        // [B, 2, S', T] → pool (K, 1) → [B, 2, K] → [B, K, 2].
        let k = self.config.keypoints as i64;
        h.adaptive_avg_pool2d([k, 1])
            .squeeze_dim(-1)
            .transpose(1, 2)
    }
}

// ---------------------------------------------------------------------------
// Tests (require the tch-backend feature + LibTorch)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;

    fn random_csi(cfg: &WiFlowStdConfig, batch: i64) -> Tensor {
        Tensor::rand(
            [batch, cfg.subcarriers as i64, cfg.window as i64],
            (Kind::Float, Device::Cpu),
        )
    }

    #[test]
    fn param_count_matches_pure_rust_formula() {
        tch::manual_seed(0);
        let cfg = WiFlowStdConfig::default();
        let model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("default config builds");
        // Pins the tch graph against the verified reference (2,225,042).
        assert_eq!(model.num_parameters(), cfg.param_count() as i64);
        assert_eq!(model.num_parameters(), 2_225_042);
    }

    /// ADR-152 efficiency-sweep compact presets: the tch graph must realise
    /// exactly the trained checkpoints' measured parameter counts
    /// (benchmarks/wiflow-std/results/efficiency_sweep.jsonl) and produce
    /// the standard [B, 15, 2] output.
    #[test]
    fn compact_preset_param_counts_and_shapes() {
        for (cfg, expected) in [
            (WiFlowStdConfig::half(), 843_834_i64),
            (WiFlowStdConfig::quarter(), 338_600),
            (WiFlowStdConfig::tiny(), 56_290),
        ] {
            tch::manual_seed(0);
            let model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("preset builds");
            assert_eq!(model.num_parameters(), expected);
            assert_eq!(model.num_parameters(), cfg.param_count() as i64);
            let out = model.forward_inference(&random_csi(&cfg, 2));
            assert_eq!(out.size(), &[2, 15, 2]);
        }
    }

    #[test]
    fn forward_output_shape_15_keypoints() {
        tch::manual_seed(0);
        let cfg = WiFlowStdConfig::default();
        let model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("build");
        let out = model.forward_t(&random_csi(&cfg, 2));
        assert_eq!(out.size(), &[2, 15, 2]);
    }

    #[test]
    fn forward_output_shape_17_keypoints_esp32() {
        tch::manual_seed(0);
        let cfg = WiFlowStdConfig::for_keypoints(17);
        let model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("build");
        let out = model.forward_inference(&random_csi(&cfg, 1));
        assert_eq!(out.size(), &[1, 17, 2]);
    }

    #[test]
    fn inference_outputs_are_finite_and_deterministic() {
        tch::manual_seed(7);
        let cfg = WiFlowStdConfig::default();
        let model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("build");
        let csi = random_csi(&cfg, 1);
        let a = model.forward_inference(&csi);
        let b = model.forward_inference(&csi);
        assert!(
            bool::try_from(a.isfinite().all()).unwrap(),
            "non-finite output"
        );
        assert!(
            bool::try_from(a.eq_tensor(&b).all()).unwrap(),
            "inference must be deterministic (dropout disabled)"
        );
    }

    /// Dumps the authoritative tch `VarStore` variable names + shapes. This is
    /// the source of truth for the PyTorch→tch key mapping implemented in
    /// `benchmarks/wiflow-std/export_to_safetensors.py` — rerun it (with
    /// `--nocapture`) whenever the architecture changes.
    #[test]
    fn dump_variable_names() {
        let cfg = WiFlowStdConfig::default();
        let model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("build");
        let vars = model.var_store().variables();
        let mut names: Vec<(String, Vec<i64>)> =
            vars.iter().map(|(n, t)| (n.clone(), t.size())).collect();
        names.sort();
        for (name, shape) in &names {
            println!("{name} {shape:?}");
        }
        println!("total: {} variables", names.len());
        assert!(!names.is_empty());
    }

    #[test]
    fn invalid_config_is_rejected() {
        let cfg = WiFlowStdConfig {
            subcarriers: 541, // not divisible by tcn_groups
            ..Default::default()
        };
        assert!(WiFlowStdModel::new(&cfg, Device::Cpu).is_err());
    }

    #[test]
    fn save_and_load_roundtrip() {
        use tempfile::tempdir;
        tch::manual_seed(42);
        let cfg = WiFlowStdConfig::default();
        let mut model = WiFlowStdModel::new(&cfg, Device::Cpu).expect("build");
        let tmp = tempdir().expect("tempdir");
        // safetensors, not .pt: this torch build's _save_parameters/_load_parameters
        // .pt roundtrip is broken on Windows (GenericDict internal assert)
        let path = tmp.path().join("wiflow_std.safetensors");
        model.save(&path).expect("save");
        model.load(&path).expect("load");
        let out = model.forward_inference(&random_csi(&cfg, 1));
        assert_eq!(out.size(), &[1, 15, 2]);
    }
}

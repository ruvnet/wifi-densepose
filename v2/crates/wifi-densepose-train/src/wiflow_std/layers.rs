//! Building-block layers for the WiFlow-STD model (tch backend, ADR-152 §2.2):
//! grouped causal TCN blocks, asymmetric residual conv blocks, and dual axial
//! attention. Internal to [`super::model`]; see the module docs for provenance.

use tch::{nn, nn::Module, Tensor};

use super::config::{CONV_BLOCK_DROPOUT, TCN_KERNEL};

/// BatchNorm config matching the reference: gamma = 1 (PyTorch default; the
/// reference additionally pins BatchNorm1d weight=1/bias=0). tch-0.24's
/// `BatchNormConfig::default()` would draw gamma from Uniform(0,1), silently
/// halving activations on average in from-scratch training.
pub(super) fn bn_cfg() -> nn::BatchNormConfig {
    nn::BatchNormConfig {
        ws_init: nn::Init::Const(1.0),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// GroupedTemporalBlock (TCN level)
// ---------------------------------------------------------------------------

/// One TCN level: two (depthwise-grouped causal conv → BN → SiLU → pointwise
/// conv → BN → SiLU → dropout) stages with a residual connection (1×1 + BN
/// projection when channels change) and a final SiLU.
///
/// Causality: each grouped conv pads by `(k-1)·dilation` and the trailing
/// padding is chomped off afterwards, exactly like the reference `Chomp1d`.
pub(super) struct GroupedTemporalBlock {
    conv1_group: nn::Conv1D,
    bn1_group: nn::BatchNorm,
    conv1_pw: nn::Conv1D,
    bn1_pw: nn::BatchNorm,
    conv2_group: nn::Conv1D,
    bn2_group: nn::BatchNorm,
    conv2_pw: nn::Conv1D,
    bn2_pw: nn::BatchNorm,
    downsample: Option<(nn::Conv1D, nn::BatchNorm)>,
    dropout: f64,
}

impl GroupedTemporalBlock {
    /// `g_in`/`g_out`: group counts of the two grouped convs (each conv
    /// groups over its own channel count — they differ under the ADR-152
    /// compact variants' `Gcd`/`Depthwise` modes). `pw_groups` groups the
    /// first pointwise conv and the residual projection (`input_pw_groups`
    /// on block 0; 1 everywhere else).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        vs: nn::Path,
        c_in: i64,
        c_out: i64,
        dilation: i64,
        g_in: i64,
        g_out: i64,
        pw_groups: i64,
        dropout: f64,
    ) -> Self {
        let k = TCN_KERNEL as i64;
        let padding = (k - 1) * dilation;
        let grouped_cfg = |groups| nn::ConvConfig {
            padding,
            dilation,
            groups,
            bias: false,
            ..Default::default()
        };
        let pointwise_cfg = |groups| nn::ConvConfig {
            groups,
            bias: false,
            ..Default::default()
        };

        let conv1_group = nn::conv1d(&vs / "conv1_group", c_in, c_in, k, grouped_cfg(g_in));
        let bn1_group = nn::batch_norm1d(&vs / "bn1_group", c_in, bn_cfg());
        let conv1_pw = nn::conv1d(&vs / "conv1_pw", c_in, c_out, 1, pointwise_cfg(pw_groups));
        let bn1_pw = nn::batch_norm1d(&vs / "bn1_pw", c_out, bn_cfg());

        let conv2_group = nn::conv1d(&vs / "conv2_group", c_out, c_out, k, grouped_cfg(g_out));
        let bn2_group = nn::batch_norm1d(&vs / "bn2_group", c_out, bn_cfg());
        let conv2_pw = nn::conv1d(&vs / "conv2_pw", c_out, c_out, 1, pointwise_cfg(1));
        let bn2_pw = nn::batch_norm1d(&vs / "bn2_pw", c_out, bn_cfg());

        let downsample = (c_in != c_out).then(|| {
            (
                nn::conv1d(&vs / "ds_conv", c_in, c_out, 1, pointwise_cfg(pw_groups)),
                nn::batch_norm1d(&vs / "ds_bn", c_out, bn_cfg()),
            )
        });

        GroupedTemporalBlock {
            conv1_group,
            bn1_group,
            conv1_pw,
            bn1_pw,
            conv2_group,
            bn2_group,
            conv2_pw,
            bn2_pw,
            downsample,
            dropout,
        }
    }

    pub(super) fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let res = match &self.downsample {
            Some((conv, bn)) => conv.forward(x).apply_t(bn, train),
            None => x.shallow_clone(),
        };
        let t = x.size()[2];

        // Stage 1: grouped causal conv (chomp trailing padding) + pointwise.
        let out = self
            .conv1_group
            .forward(x)
            .narrow(2, 0, t) // Chomp1d
            .apply_t(&self.bn1_group, train)
            .silu()
            .apply(&self.conv1_pw)
            .apply_t(&self.bn1_pw, train)
            .silu()
            .dropout(self.dropout, train);

        // Stage 2.
        let out = self
            .conv2_group
            .forward(&out)
            .narrow(2, 0, t) // Chomp1d
            .apply_t(&self.bn2_group, train)
            .silu()
            .apply(&self.conv2_pw)
            .apply_t(&self.bn2_pw, train)
            .silu()
            .dropout(self.dropout, train);

        (out + res).silu()
    }
}

// ---------------------------------------------------------------------------
// ConvBlock (ConvBlock1 / AsymmetricConvBlock)
// ---------------------------------------------------------------------------

/// Asymmetric residual conv block: three `(1, 3)` convs (only the subcarrier
/// axis is convolved) with BN, SiLU and channel dropout, plus a 1×1 + BN
/// residual projection. `stride_w == 1` reproduces the reference `ConvBlock1`,
/// `stride_w == 2` the downsampling `AsymmetricConvBlock`.
pub(super) struct ConvBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    bn3: nn::BatchNorm,
    ds_conv: nn::Conv2D,
    ds_bn: nn::BatchNorm,
}

impl ConvBlock {
    pub(super) fn new(vs: nn::Path, c_in: i64, c_out: i64, stride_w: i64) -> Self {
        let asym = |stride_w| nn::ConvConfigND::<[i64; 2]> {
            stride: [1, stride_w],
            padding: [0, 1],
            ..Default::default()
        };
        let conv1 = nn::conv(&vs / "conv1", c_in, c_out, [1, 3], asym(stride_w));
        let bn1 = nn::batch_norm2d(&vs / "bn1", c_out, bn_cfg());
        let conv2 = nn::conv(&vs / "conv2", c_out, c_out, [1, 3], asym(1));
        let bn2 = nn::batch_norm2d(&vs / "bn2", c_out, bn_cfg());
        let conv3 = nn::conv(&vs / "conv3", c_out, c_out, [1, 3], asym(1));
        let bn3 = nn::batch_norm2d(&vs / "bn3", c_out, bn_cfg());

        let ds_conv = nn::conv(
            &vs / "ds_conv",
            c_in,
            c_out,
            [1, 1],
            nn::ConvConfigND::<[i64; 2]> {
                stride: [1, stride_w],
                bias: false,
                ..Default::default()
            },
        );
        let ds_bn = nn::batch_norm2d(&vs / "ds_bn", c_out, bn_cfg());

        ConvBlock {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            ds_conv,
            ds_bn,
        }
    }

    pub(super) fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let identity = self.ds_conv.forward(x).apply_t(&self.ds_bn, train);
        let out = x
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .silu()
            .feature_dropout(CONV_BLOCK_DROPOUT, train) // Dropout2d
            .apply(&self.conv2)
            .apply_t(&self.bn2, train)
            .silu()
            .feature_dropout(CONV_BLOCK_DROPOUT, train)
            .apply(&self.conv3)
            .apply_t(&self.bn3, train);
        (out + identity).silu()
    }
}

// ---------------------------------------------------------------------------
// Axial attention
// ---------------------------------------------------------------------------

/// Single-axis self-attention with BN-normalised qkv, BN-normalised
/// similarity logits and BN-normalised output. `width == true` attends along
/// the last (W) axis, otherwise along the H axis; the other spatial axis is
/// folded into the batch.
pub(super) struct AxialAttention {
    qkv: nn::Conv1D,
    bn_qkv: nn::BatchNorm,
    bn_similarity: nn::BatchNorm,
    bn_output: nn::BatchNorm,
    out_planes: i64,
    groups: i64,
    width: bool,
}

impl AxialAttention {
    pub(super) fn new(vs: nn::Path, planes: i64, groups: i64, width: bool) -> Self {
        // Reference init: N(0, sqrt(1 / in_planes)).
        let qkv = nn::conv1d(
            &vs / "qkv",
            planes,
            planes * 3,
            1,
            nn::ConvConfig {
                bias: false,
                ws_init: nn::Init::Randn {
                    mean: 0.0,
                    stdev: (1.0 / planes as f64).sqrt(),
                },
                ..Default::default()
            },
        );
        let bn_qkv = nn::batch_norm1d(&vs / "bn_qkv", planes * 3, bn_cfg());
        let bn_similarity = nn::batch_norm2d(&vs / "bn_similarity", groups, bn_cfg());
        let bn_output = nn::batch_norm1d(&vs / "bn_output", planes, bn_cfg());

        AxialAttention {
            qkv,
            bn_qkv,
            bn_similarity,
            bn_output,
            out_planes: planes,
            groups,
            width,
        }
    }

    pub(super) fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        // Fold the non-attended spatial axis into the batch:
        // width: [B,C,H,W] → [B,H,C,W]; height: [B,C,H,W] → [B,W,C,H].
        let x = if self.width {
            x.permute([0, 2, 1, 3])
        } else {
            x.permute([0, 3, 1, 2])
        };
        let (n, outer, c, axis) = {
            let s = x.size();
            (s[0], s[1], s[2], s[3])
        };
        let flat = x.contiguous().view([n * outer, c, axis]);

        // BN-normalised qkv: [N', 3·C, axis] → grouped q, k, v.
        let gp = self.out_planes / self.groups; // group planes
        let qkv = flat.apply(&self.qkv).apply_t(&self.bn_qkv, train).reshape([
            n * outer,
            3,
            self.groups,
            gp,
            axis,
        ]);
        let q = qkv.select(1, 0); // [N', g, gp, axis]
        let k = qkv.select(1, 1);
        let v = qkv.select(1, 2);

        // similarity[b,g,i,j] = Σ_c q[b,g,c,i]·k[b,g,c,j], BN over the g maps.
        let logits = q.transpose(2, 3).matmul(&k); // [N', g, axis, axis]
        let similarity = logits
            .apply_t(&self.bn_similarity, train)
            .softmax(-1, logits.kind());

        // out[b,g,c,i] = Σ_j similarity[b,g,i,j]·v[b,g,c,j].
        let sv = v.matmul(&similarity.transpose(2, 3)); // [N', g, gp, axis]
        let out = sv
            .reshape([n * outer, self.out_planes, axis])
            .apply_t(&self.bn_output, train)
            .view([n, outer, self.out_planes, axis]);

        // Restore [B, C, H, W].
        if self.width {
            out.permute([0, 2, 1, 3])
        } else {
            out.permute([0, 2, 3, 1])
        }
    }
}

/// Width-axis then height-axis axial attention (the reference
/// `DualAxialAttention`, stride 1).
pub(super) struct DualAxialAttention {
    width_axis: AxialAttention,
    height_axis: AxialAttention,
}

impl DualAxialAttention {
    pub(super) fn new(vs: nn::Path, planes: i64, groups: i64) -> Self {
        DualAxialAttention {
            width_axis: AxialAttention::new(&vs / "width", planes, groups, true),
            height_axis: AxialAttention::new(&vs / "height", planes, groups, false),
        }
    }

    pub(super) fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let x = self.width_axis.forward_t(x, train);
        self.height_axis.forward_t(&x, train)
    }
}

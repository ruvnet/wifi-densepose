//! EfficientNet implementation.
use crate::nn::{self, ConvConfig, Module, ModuleT};
use crate::Tensor;

const BATCH_NORM_MOMENTUM: f64 = 0.99;
const BATCH_NORM_EPSILON: f64 = 1e-3;

// Based on the Python version from torchvision.
// https://github.com/pytorch/vision/blob/0d75d9e5516f446c9c0ef93bd4ed9fea13992d06/torchvision/models/efficientnet.py#L47
#[derive(Debug, Clone, Copy)]
pub struct MBConvConfig {
    expand_ratio: f64,
    kernel: i64,
    stride: i64,
    input_channels: i64,
    out_channels: i64,
    num_layers: usize,
}

fn make_divisible(v: f64, divisor: i64) -> i64 {
    let min_value = divisor;
    let new_v = i64::max(min_value, (v + divisor as f64 * 0.5) as i64 / divisor * divisor);
    if (new_v as f64) < 0.9 * v {
        new_v + divisor
    } else {
        new_v
    }
}

fn bneck_confs(width_mult: f64, depth_mult: f64) -> Vec<MBConvConfig> {
    let bneck_conf = |e, k, s, i, o, n| {
        let input_channels = make_divisible(i as f64 * width_mult, 8);
        let out_channels = make_divisible(o as f64 * width_mult, 8);
        let num_layers = (n as f64 * depth_mult).ceil() as usize;
        MBConvConfig {
            expand_ratio: e,
            kernel: k,
            stride: s,
            input_channels,
            out_channels,
            num_layers,
        }
    };
    vec![
        bneck_conf(1., 3, 1, 32, 16, 1),
        bneck_conf(6., 3, 2, 16, 24, 2),
        bneck_conf(6., 5, 2, 24, 40, 2),
        bneck_conf(6., 3, 2, 40, 80, 3),
        bneck_conf(6., 5, 1, 80, 112, 3),
        bneck_conf(6., 5, 2, 112, 192, 4),
        bneck_conf(6., 3, 1, 192, 320, 1),
    ]
}

impl MBConvConfig {
    fn b0() -> Vec<Self> {
        bneck_confs(1.0, 1.0)
    }
    fn b1() -> Vec<Self> {
        bneck_confs(1.0, 1.1)
    }
    fn b2() -> Vec<Self> {
        bneck_confs(1.1, 1.2)
    }
    fn b3() -> Vec<Self> {
        bneck_confs(1.2, 1.4)
    }
    fn b4() -> Vec<Self> {
        bneck_confs(1.4, 1.8)
    }
    fn b5() -> Vec<Self> {
        bneck_confs(1.6, 2.2)
    }
    fn b6() -> Vec<Self> {
        bneck_confs(1.8, 2.6)
    }
    fn b7() -> Vec<Self> {
        bneck_confs(2.0, 3.1)
    }
}

/// Conv2D with same padding.
#[derive(Debug)]
struct Conv2DSame {
    conv2d: nn::Conv2D,
    s: i64,
    k: i64,
}

impl Conv2DSame {
    fn new(vs: nn::Path, i: i64, o: i64, k: i64, stride: i64, groups: i64, b: bool) -> Self {
        let conv_config = nn::ConvConfig { stride, groups, bias: b, ..Default::default() };
        let conv2d = nn::conv2d(vs, i, o, k, conv_config);
        Self { conv2d, s: stride, k }
    }
}

impl Module for Conv2DSame {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let s = self.s;
        let k = self.k;
        let size = xs.size();
        let ih = size[2];
        let iw = size[3];
        let oh = (ih + s - 1) / s;
        let ow = (iw + s - 1) / s;
        let pad_h = i64::max((oh - 1) * s + k - ih, 0);
        let pad_w = i64::max((ow - 1) * s + k - iw, 0);
        if pad_h > 0 || pad_w > 0 {
            xs.zero_pad2d(pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2)
                .apply(&self.conv2d)
        } else {
            xs.apply(&self.conv2d)
        }
    }
}

#[derive(Debug)]
struct ConvNormActivation {
    conv2d: Conv2DSame,
    bn2d: nn::BatchNorm,
    activation: bool,
}

impl ConvNormActivation {
    fn new(vs: nn::Path, i: i64, o: i64, k: i64, stride: i64, groups: i64) -> Self {
        let conv2d = Conv2DSame::new(&vs / 0, i, o, k, stride, groups, false);
        let bn_config = nn::BatchNormConfig {
            momentum: 1.0 - BATCH_NORM_MOMENTUM,
            eps: BATCH_NORM_EPSILON,
            ..Default::default()
        };
        let bn2d = nn::batch_norm2d(&vs / 1, o, bn_config);
        Self { conv2d, bn2d, activation: true }
    }

    fn no_activation(self) -> Self {
        Self { activation: false, ..self }
    }
}

impl ModuleT for ConvNormActivation {
    fn forward_t(&self, xs: &Tensor, t: bool) -> Tensor {
        let xs = xs.apply(&self.conv2d).apply_t(&self.bn2d, t);
        if self.activation {
            xs.swish()
        } else {
            xs
        }
    }
}

#[derive(Debug)]
struct SqueezeExcitation {
    fc1: Conv2DSame,
    fc2: Conv2DSame,
}

impl SqueezeExcitation {
    fn new(vs: nn::Path, input_channels: i64, squeeze_channels: i64) -> Self {
        let fc1 = Conv2DSame::new(&vs / "fc1", input_channels, squeeze_channels, 1, 1, 1, true);
        let fc2 = Conv2DSame::new(&vs / "fc2", squeeze_channels, input_channels, 1, 1, 1, true);
        Self { fc1, fc2 }
    }
}

impl ModuleT for SqueezeExcitation {
    fn forward_t(&self, xs: &Tensor, t: bool) -> Tensor {
        let scale = xs
            .adaptive_avg_pool2d([1, 1])
            .apply_t(&self.fc1, t)
            .swish()
            .apply_t(&self.fc2, t)
            .sigmoid();
        scale * xs
    }
}

#[derive(Debug)]
struct MBConv {
    expand_cna: Option<ConvNormActivation>,
    depthwise_cna: ConvNormActivation,
    squeeze_excitation: SqueezeExcitation,
    project_cna: ConvNormActivation,
    config: MBConvConfig,
}

impl MBConv {
    fn new(vs: nn::Path, c: MBConvConfig) -> Self {
        let vs = &vs / "block";
        let exp = make_divisible(c.input_channels as f64 * c.expand_ratio, 8);
        let expand_cna = if exp != c.input_channels {
            Some(ConvNormActivation::new(&vs / 0, c.input_channels, exp, 1, 1, 1))
        } else {
            None
        };
        let start_index = if expand_cna.is_some() { 1 } else { 0 };
        let depthwise_cna =
            ConvNormActivation::new(&vs / start_index, exp, exp, c.kernel, c.stride, exp);
        let squeeze_channels = i64::max(1, c.input_channels / 4);
        let squeeze_excitation =
            SqueezeExcitation::new(&vs / (start_index + 1), exp, squeeze_channels);
        let project_cna =
            ConvNormActivation::new(&vs / (start_index + 2), exp, c.out_channels, 1, 1, 1)
                .no_activation();
        Self { expand_cna, depthwise_cna, squeeze_excitation, project_cna, config: c }
    }
}

impl ModuleT for MBConv {
    fn forward_t(&self, xs: &Tensor, t: bool) -> Tensor {
        let use_res_connect =
            self.config.stride == 1 && self.config.input_channels == self.config.out_channels;
        let ys = match &self.expand_cna {
            Some(expand_cna) => xs.apply_t(expand_cna, t),
            None => xs.shallow_clone(),
        };
        let ys = ys
            .apply_t(&self.depthwise_cna, t)
            .apply_t(&self.squeeze_excitation, t)
            .apply_t(&self.project_cna, t);
        if use_res_connect {
            ys + xs
        } else {
            ys
        }
    }
}

impl Tensor {
    fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }
}

#[derive(Debug)]
struct EfficientNet {
    init_cna: ConvNormActivation,
    blocks: Vec<MBConv>,
    final_cna: ConvNormActivation,
    classifier: nn::Linear,
}

impl EfficientNet {
    fn new(p: &nn::Path, configs: Vec<MBConvConfig>, nclasses: i64) -> Self {
        let f_p = p / "features";
        let first_in_c = configs[0].input_channels;
        let last_out_c = configs.last().unwrap().out_channels;
        let final_out_c = 4 * last_out_c;
        let init_cna = ConvNormActivation::new(&f_p / 0, 3, first_in_c, 3, 2, 1);
        let nconfigs = configs.len();
        let mut blocks = vec![];
        for (index, cnf) in configs.into_iter().enumerate() {
            let f_p = &f_p / (index + 1);
            for r_index in 0..cnf.num_layers {
                let cnf = if r_index == 0 {
                    cnf
                } else {
                    MBConvConfig { input_channels: cnf.out_channels, stride: 1, ..cnf }
                };
                blocks.push(MBConv::new(&f_p / r_index, cnf))
            }
        }
        let final_cna =
            ConvNormActivation::new(&f_p / (nconfigs + 1), last_out_c, final_out_c, 1, 1, 1);
        let classifier =
            nn::linear(p / "classifier" / 1, final_out_c, nclasses, Default::default());
        Self { init_cna, blocks, final_cna, classifier }
    }
}

impl ModuleT for EfficientNet {
    fn forward_t(&self, xs: &Tensor, t: bool) -> Tensor {
        let mut xs = xs.apply_t(&self.init_cna, t);
        for block in self.blocks.iter() {
            xs = xs.apply_t(block, t)
        }
        xs.apply_t(&self.final_cna, t)
            .adaptive_avg_pool2d([1, 1])
            .squeeze_dim(-1)
            .squeeze_dim(-1)
            .apply(&self.classifier)
    }
}

pub fn b0(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b0(), nclasses)
}
pub fn b1(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b1(), nclasses)
}
pub fn b2(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b2(), nclasses)
}
pub fn b3(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b3(), nclasses)
}
pub fn b4(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b4(), nclasses)
}
pub fn b5(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b5(), nclasses)
}
pub fn b6(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b6(), nclasses)
}
pub fn b7(p: &nn::Path, nclasses: i64) -> impl ModuleT {
    EfficientNet::new(p, MBConvConfig::b7(), nclasses)
}

#[allow(clippy::many_single_char_names)]
pub fn conv2d_same(vs: nn::Path, i: i64, o: i64, k: i64, c: ConvConfig) -> impl Module {
    let conv2d = nn::conv2d(vs, i, o, k, c);
    let s = c.stride;
    nn::func(move |xs| {
        let size = xs.size();
        let ih = size[2];
        let iw = size[3];
        let oh = (ih + s - 1) / s;
        let ow = (iw + s - 1) / s;
        let pad_h = i64::max((oh - 1) * s + k - ih, 0);
        let pad_w = i64::max((ow - 1) * s + k - iw, 0);
        if pad_h > 0 || pad_w > 0 {
            xs.zero_pad2d(pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2).apply(&conv2d)
        } else {
            xs.apply(&conv2d)
        }
    })
}

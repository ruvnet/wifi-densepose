//! N-dimensional convolution layers.
use super::Path;
use crate::{TchError, Tensor};
use std::borrow::Borrow;

/// How padding is performed by convolution operations
/// on the edge of the input tensor.
#[derive(Debug, Clone, Copy)]
pub enum PaddingMode {
    Zeros,
    Reflect,
    Replicate,
    Circular,
}

impl PaddingMode {
    fn to_string(self) -> &'static str {
        // This has to match the internal representation used on the C++
        // side.
        match self {
            // The default value when using constant is zero.
            PaddingMode::Zeros => "constant",
            PaddingMode::Reflect => "reflect",
            PaddingMode::Replicate => "replicate",
            PaddingMode::Circular => "circular",
        }
    }

    pub fn f_pad(
        self,
        xs: &Tensor,
        reversed_padding_repeated_twice: &[i64],
    ) -> Result<Tensor, TchError> {
        xs.f_pad(reversed_padding_repeated_twice, self.to_string(), None)
    }

    pub fn pad(self, xs: &Tensor, reversed_padding_repeated_twice: &[i64]) -> Tensor {
        xs.pad(reversed_padding_repeated_twice, self.to_string(), None)
    }
}

/// Generic convolution config.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct ConvConfigND<ND> {
    pub stride: ND,
    pub padding: ND,
    pub dilation: ND,
    pub groups: i64,
    pub bias: bool,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
    pub padding_mode: PaddingMode,
}

/// Convolution config using the same parameters on all dimensions.
pub type ConvConfig = ConvConfigND<i64>;

impl Default for ConvConfig {
    fn default() -> Self {
        ConvConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            ws_init: super::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: super::Init::Const(0.),
            padding_mode: PaddingMode::Zeros,
        }
    }
}

impl Default for ConvConfigND<[i64; 2]> {
    fn default() -> Self {
        ConvConfigND::<[i64; 2]> {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            bias: true,
            ws_init: super::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: super::Init::Const(0.),
            padding_mode: PaddingMode::Zeros,
        }
    }
}

/// The default convolution config without bias.
pub fn no_bias() -> ConvConfig {
    ConvConfig { bias: false, ..Default::default() }
}

// Use const generics when they have landed in stable rust.
/// A N-dimensional convolution layer.
#[derive(Debug)]
pub struct Conv<ND> {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
    reversed_padding_repeated_twice: Vec<i64>,
    config: ConvConfigND<ND>,
}

/// One dimension convolution layer.
pub type Conv1D = Conv<[i64; 1]>;

/// Two dimensions convolution layer.
pub type Conv2D = Conv<[i64; 2]>;

/// Three dimensions convolution layer.
pub type Conv3D = Conv<[i64; 3]>;

/// Creates a new convolution layer for any number of dimensions.
pub fn conv<'a, ND: std::convert::AsRef<[i64]>, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    ksizes: ND,
    config: ConvConfigND<ND>,
) -> Conv<ND> {
    let vs = vs.borrow();
    let bs = if config.bias { Some(vs.var("bias", &[out_dim], config.bs_init)) } else { None };
    let mut weight_size = vec![out_dim, in_dim / config.groups];
    weight_size.extend(ksizes.as_ref().iter());
    let ws = vs.var("weight", weight_size.as_slice(), config.ws_init);
    let mut reversed_padding_repeated_twice = vec![];
    for &v in config.padding.as_ref().iter().rev() {
        reversed_padding_repeated_twice.push(v)
    }
    for &v in config.padding.as_ref().iter().rev() {
        reversed_padding_repeated_twice.push(v)
    }
    Conv { ws, bs, config, reversed_padding_repeated_twice }
}

trait Create: std::convert::AsRef<[i64]> + std::marker::Sized {
    fn make_array(i: i64) -> Self;

    fn conv<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: ConvConfig,
    ) -> Conv<Self> {
        let config = ConvConfigND::<Self> {
            stride: Self::make_array(config.stride),
            padding: Self::make_array(config.padding),
            dilation: Self::make_array(config.dilation),
            groups: config.groups,
            bias: config.bias,
            ws_init: config.ws_init,
            bs_init: config.bs_init,
            padding_mode: config.padding_mode,
        };
        conv(vs, in_dim, out_dim, Self::make_array(ksize), config)
    }
}

impl Create for [i64; 1] {
    fn make_array(i: i64) -> Self {
        [i]
    }
}

impl Create for [i64; 2] {
    fn make_array(i: i64) -> Self {
        [i, i]
    }
}

impl Create for [i64; 3] {
    fn make_array(i: i64) -> Self {
        [i, i, i]
    }
}

/// Creates a new one dimension convolution layer.
pub fn conv1d<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Conv1D {
    <[i64; 1]>::conv(vs, i, o, k, c)
}

/// Creates a new two dimension convolution layer.
pub fn conv2d<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Conv2D {
    <[i64; 2]>::conv(vs, i, o, k, c)
}

/// Creates a new three dimension convolution layer.
pub fn conv3d<'a, T: Borrow<Path<'a>>>(vs: T, i: i64, o: i64, k: i64, c: ConvConfig) -> Conv3D {
    <[i64; 3]>::conv(vs, i, o, k, c)
}

impl super::module::Module for Conv1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (xs, padding) = match self.config.padding_mode {
            PaddingMode::Zeros => (xs.shallow_clone(), self.config.padding),
            p => (p.pad(xs, &self.reversed_padding_repeated_twice), [0]),
        };
        xs.conv1d(
            &self.ws,
            self.bs.as_ref(),
            self.config.stride,
            padding,
            self.config.dilation,
            self.config.groups,
        )
    }
}

impl super::module::Module for Conv2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (xs, padding) = match self.config.padding_mode {
            PaddingMode::Zeros => (xs.shallow_clone(), self.config.padding),
            p => (p.pad(xs, &self.reversed_padding_repeated_twice), [0, 0]),
        };
        xs.conv2d(
            &self.ws,
            self.bs.as_ref(),
            self.config.stride,
            padding,
            self.config.dilation,
            self.config.groups,
        )
    }
}

impl super::module::Module for Conv3D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (xs, padding) = match self.config.padding_mode {
            PaddingMode::Zeros => (xs.shallow_clone(), self.config.padding),
            p => (p.pad(xs, &self.reversed_padding_repeated_twice), [0, 0, 0]),
        };
        xs.conv3d(
            &self.ws,
            self.bs.as_ref(),
            self.config.stride,
            padding,
            self.config.dilation,
            self.config.groups,
        )
    }
}

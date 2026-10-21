//! A two dimension transposed convolution layer.
use super::Path;
use crate::Tensor;
use std::borrow::Borrow;

/// A generic transposed convolution configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct ConvTransposeConfigND<ND> {
    pub stride: ND,
    pub padding: ND,
    pub output_padding: ND,
    pub groups: i64,
    pub bias: bool,
    pub dilation: ND,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

/// A transposed convolution configuration using the same values on each dimension.
pub type ConvTransposeConfig = ConvTransposeConfigND<i64>;

impl Default for ConvTransposeConfig {
    fn default() -> Self {
        ConvTransposeConfigND {
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            ws_init: super::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A generic transposed convolution layer.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct ConvTransposeND<ND> {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
    config: ConvTransposeConfigND<ND>,
}

/// A one dimension transposed convolution layer.
pub type ConvTranspose1D = ConvTransposeND<[i64; 1]>;

/// A two dimension transposed convolution layer.
pub type ConvTranspose2D = ConvTransposeND<[i64; 2]>;

/// A three dimension transposed convolution layer.
pub type ConvTranspose3D = ConvTransposeND<[i64; 3]>;

fn conv_transpose<'a, ND: std::convert::AsRef<[i64]>, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    ksizes: ND,
    config: ConvTransposeConfigND<ND>,
) -> ConvTransposeND<ND> {
    let vs = vs.borrow();
    let bs = if config.bias { Some(vs.var("bias", &[out_dim], config.bs_init)) } else { None };
    let mut weight_size = vec![in_dim, out_dim / config.groups];
    weight_size.extend(ksizes.as_ref().iter());
    let ws = vs.var("weight", weight_size.as_slice(), config.ws_init);
    ConvTransposeND { ws, bs, config }
}

trait Create: std::convert::AsRef<[i64]> + std::marker::Sized {
    fn make_array(i: i64) -> Self;

    fn conv_transpose<'a, T: Borrow<super::Path<'a>>>(
        vs: T,
        in_dim: i64,
        out_dim: i64,
        ksize: i64,
        config: ConvTransposeConfig,
    ) -> ConvTransposeND<Self> {
        let config = ConvTransposeConfigND::<Self> {
            stride: Self::make_array(config.stride),
            padding: Self::make_array(config.padding),
            output_padding: Self::make_array(config.output_padding),
            dilation: Self::make_array(config.dilation),
            groups: config.groups,
            bias: config.bias,
            ws_init: config.ws_init,
            bs_init: config.bs_init,
        };
        conv_transpose(vs, in_dim, out_dim, Self::make_array(ksize), config)
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

/// Creates a one dimension transposed convolution layer.
pub fn conv_transpose1d<'a, T: Borrow<Path<'a>>>(
    vs: T,
    i: i64,
    o: i64,
    k: i64,
    c: ConvTransposeConfig,
) -> ConvTranspose1D {
    <[i64; 1]>::conv_transpose(vs, i, o, k, c)
}

/// Creates a two dimension transposed convolution layer.
pub fn conv_transpose2d<'a, T: Borrow<Path<'a>>>(
    vs: T,
    i: i64,
    o: i64,
    k: i64,
    c: ConvTransposeConfig,
) -> ConvTranspose2D {
    <[i64; 2]>::conv_transpose(vs, i, o, k, c)
}

/// Creates a three dimension transposed convolution layer.
pub fn conv_transpose3d<'a, T: Borrow<Path<'a>>>(
    vs: T,
    i: i64,
    o: i64,
    k: i64,
    c: ConvTransposeConfig,
) -> ConvTranspose3D {
    <[i64; 3]>::conv_transpose(vs, i, o, k, c)
}

impl super::module::Module for ConvTranspose1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv_transpose1d(
            xs,
            &self.ws,
            self.bs.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.groups,
            self.config.dilation,
        )
    }
}

impl super::module::Module for ConvTranspose2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv_transpose2d(
            xs,
            &self.ws,
            self.bs.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.groups,
            self.config.dilation,
        )
    }
}

impl super::module::Module for ConvTranspose3D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::conv_transpose3d(
            xs,
            &self.ws,
            self.bs.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.output_padding,
            self.config.groups,
            self.config.dilation,
        )
    }
}

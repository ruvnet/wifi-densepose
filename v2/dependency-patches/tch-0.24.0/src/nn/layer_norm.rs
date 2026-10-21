//! A layer-normalization layer.
use crate::Tensor;
use std::borrow::Borrow;

/// Layer-normalization config.
#[derive(Debug, Clone, Copy)]
pub struct LayerNormConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub elementwise_affine: bool,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        LayerNormConfig {
            cudnn_enabled: true,
            eps: 1e-5,
            elementwise_affine: true,
            ws_init: super::Init::Const(1.),
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A layer-normalization layer.
#[derive(Debug)]
pub struct LayerNorm {
    config: LayerNormConfig,
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    pub normalized_shape: Vec<i64>,
}

pub fn layer_norm<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    normalized_shape: Vec<i64>,
    config: LayerNormConfig,
) -> LayerNorm {
    let vs = vs.borrow();

    let (ws, bs) = if config.elementwise_affine {
        let ws = vs.var("weight", normalized_shape.as_slice(), config.ws_init);
        let bs = vs.var("bias", normalized_shape.as_slice(), config.bs_init);
        (Some(ws), Some(bs))
    } else {
        (None, None)
    };

    LayerNorm { config, ws, bs, normalized_shape }
}

impl super::module::Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::layer_norm(
            xs,
            self.normalized_shape.as_slice(),
            self.ws.as_ref(),
            self.bs.as_ref(),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}

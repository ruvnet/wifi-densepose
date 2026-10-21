//! A group-normalization layer.
//! Group Normalization <https://arxiv.org/abs/1803.0849>
use crate::Tensor;
use std::borrow::Borrow;

/// Group-normalization config.
#[derive(Debug, Clone, Copy)]
pub struct GroupNormConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub affine: bool,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for GroupNormConfig {
    fn default() -> Self {
        GroupNormConfig {
            cudnn_enabled: true,
            eps: 1e-5,
            affine: true,
            ws_init: super::Init::Const(1.),
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A group-normalization layer.
#[derive(Debug)]
pub struct GroupNorm {
    config: GroupNormConfig,
    pub ws: Option<Tensor>,
    pub bs: Option<Tensor>,
    pub num_groups: i64,
    pub num_channels: i64,
}

pub fn group_norm<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    num_groups: i64,
    num_channels: i64,
    config: GroupNormConfig,
) -> GroupNorm {
    let vs = vs.borrow();
    let (ws, bs) = if config.affine {
        let ws = vs.var("weight", &[num_channels], config.ws_init);
        let bs = vs.var("bias", &[num_channels], config.bs_init);
        (Some(ws), Some(bs))
    } else {
        (None, None)
    };
    GroupNorm { config, ws, bs, num_groups, num_channels }
}

impl super::module::Module for GroupNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::group_norm(
            xs,
            self.num_groups,
            self.ws.as_ref(),
            self.bs.as_ref(),
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}

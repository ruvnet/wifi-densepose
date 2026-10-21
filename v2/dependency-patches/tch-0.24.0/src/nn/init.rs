//! Variable initialization.
use crate::{Device, Kind, TchError, Tensor};

/// Number of features as input or output of a layer.
/// In Kaiming initialization, choosing `FanIn` preserves
/// the magnitude of the variance of the weights in the
/// forward pass, choosing `FanOut` preserves this
/// magnitude in the backward pass.
#[derive(Debug, Copy, Clone)]
pub enum FanInOut {
    FanIn,
    FanOut,
}

impl FanInOut {
    /// Compute the fan-in or fan-out value for a weight tensor of
    /// the specified dimensions.
    /// <https://github.com/pytorch/pytorch/blob/dbeacf11820e336e803bb719b7aaaf2125ae4d9c/torch/nn/init.py#L284>
    pub fn for_weight_dims(&self, dims: &[i64]) -> i64 {
        let receptive_field_size: i64 = dims.iter().skip(2).product();
        match &self {
            FanInOut::FanIn => {
                if dims.len() < 2 {
                    1
                } else {
                    dims[1] * receptive_field_size
                }
            }
            FanInOut::FanOut => {
                if dims.is_empty() {
                    1
                } else {
                    dims[0] * receptive_field_size
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NormalOrUniform {
    Normal,
    Uniform,
}

/// The non-linear function that follows this layer. ReLU is the
/// recommended value.
#[derive(Debug, Copy, Clone)]
pub enum NonLinearity {
    ReLU,
    Linear,
    Sigmoid,
    Tanh,
    SELU,
    ExplicitGain(f64),
}

impl NonLinearity {
    pub fn gain(&self) -> f64 {
        match *self {
            NonLinearity::ReLU => 2f64.sqrt(),
            NonLinearity::Tanh => 5. / 3.,
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.,
            NonLinearity::SELU => 0.75,
            NonLinearity::ExplicitGain(g) => g,
        }
    }
}

/// Variable initializations.
#[derive(Debug, Copy, Clone)]
pub enum Init {
    /// Constant value.
    Const(f64),

    /// Random normal with some mean and standard deviation.
    Randn { mean: f64, stdev: f64 },

    /// Uniform initialization between some lower and upper bounds.
    Uniform { lo: f64, up: f64 },

    /// Kaiming uniform initialization.
    /// See "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
    /// He, K. et al. (2015). This uses a uniform distribution.
    Kaiming { dist: NormalOrUniform, fan: FanInOut, non_linearity: NonLinearity },

    /// Orthogonal initialization
    Orthogonal { gain: f64 },
}

pub const DEFAULT_KAIMING_UNIFORM: Init = Init::Kaiming {
    dist: NormalOrUniform::Uniform,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

pub const DEFAULT_KAIMING_NORMAL: Init = Init::Kaiming {
    dist: NormalOrUniform::Normal,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

/// Creates a new float tensor with the specified shape, device, and initialization.
pub fn f_init(i: Init, dims: &[i64], device: Device, kind: Kind) -> Result<Tensor, TchError> {
    match i {
        Init::Const(cst) => {
            // Optimize the case for which a single C++ code can be done.
            if cst == 0. {
                Tensor::f_zeros(dims, (kind, device))
            } else if (cst - 1.).abs() <= f64::EPSILON {
                Tensor::f_ones(dims, (kind, device))
            } else {
                Tensor::f_ones(dims, (kind, device)).map(|t| t * cst)
            }
        }
        Init::Uniform { lo, up } => Tensor::f_zeros(dims, (kind, device))?.f_uniform_(lo, up),
        Init::Randn { mean, stdev } => {
            if mean == 0. && (stdev - 1.).abs() <= f64::EPSILON {
                Tensor::f_randn(dims, (kind, device))
            } else {
                Tensor::f_randn(dims, (kind, device)).map(|t| t * stdev + mean)
            }
        }
        Init::Kaiming { dist, fan, non_linearity } => {
            let fan = fan.for_weight_dims(dims);
            let gain = non_linearity.gain();
            let std = gain / (fan as f64).sqrt();
            match dist {
                NormalOrUniform::Uniform => {
                    let bound = 3f64.sqrt() * std;
                    Tensor::f_zeros(dims, (kind, device))?.f_uniform_(-bound, bound)
                }
                NormalOrUniform::Normal => {
                    let randn = Tensor::f_randn(dims, (kind, device))?;
                    Ok(randn * std)
                }
            }
        }
        Init::Orthogonal { gain } => {
            if dims.len() < 2 {
                return Err(TchError::Shape(
                    "Only tensors with 2 or more dimensions are supported".to_string(),
                ));
            }
            let rows = dims[0];
            let cols: i64 = dims.iter().skip(1).product();

            let mut flattened =
                Tensor::f_empty([rows, cols], (kind, device))?.f_normal_(0.0, 1.0)?;
            let flattened = if rows < cols { flattened.f_t_()? } else { flattened };

            let (mut q, r) = Tensor::f_linalg_qr(&flattened, "reduced")?;
            let d = r.f_diag(0)?;
            let ph = d.f_sign()?;
            q *= ph;

            let mut q = if rows < cols { q.f_t_()? } else { q };
            crate::no_grad(|| q *= gain);

            q.f_contiguous()
        }
    }
}

/// Creates a new float tensor with the specified shape, device, and initialization.
pub fn init(i: Init, dims: &[i64], device: Device) -> Tensor {
    f_init(i, dims, device, Kind::Float).unwrap()
}

impl Init {
    /// Re-initializes an existing tensor with the specified initialization
    pub fn set(self, tensor: &mut Tensor) {
        match self {
            Init::Const(cst) => {
                let _ = tensor.fill_(cst);
            }
            Init::Uniform { lo, up } => {
                let _ = tensor.uniform_(lo, up);
            }
            Init::Kaiming { dist, fan, non_linearity } => {
                let fan = fan.for_weight_dims(&tensor.size());
                let gain = non_linearity.gain();
                let std = gain / (fan as f64).sqrt();
                match dist {
                    NormalOrUniform::Uniform => {
                        let bound = 3f64.sqrt() * std;
                        let _ = tensor.uniform_(-bound, bound);
                    }
                    NormalOrUniform::Normal => {
                        tensor.copy_(&(tensor.randn_like() * std));
                    }
                }
            }
            Init::Randn { mean, stdev } => {
                tensor.copy_(&(tensor.randn_like() * stdev + mean));
            }
            Init::Orthogonal { gain } => {
                let q =
                    f_init(Init::Orthogonal { gain }, &tensor.size(), tensor.device(), Kind::Float)
                        .unwrap();
                crate::no_grad(|| tensor.view_as(&q).copy_(&q));
            }
        }
    }
}

impl Tensor {
    /// Re-initializes the tensor using the specified initialization.
    pub fn init(&mut self, i: Init) {
        i.set(self)
    }
}

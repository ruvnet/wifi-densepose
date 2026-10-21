use candle_core::backend::BackendStorage;
use candle_core::cpu_backend;
use candle_core::test_utils::to_vec1_round;
use candle_core::{CpuStorage, CustomOp1, DType, Device, Error, Layout, Result, Shape, Tensor};

fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        (v.exp() - T::one()) * alpha
    }
}

struct Elu {
    alpha: f64,
}

impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle_core::map_dtype!(
            "elu",
            s,
            |s| cpu_backend::unary_map(s, l, |v| fwd(v, self.alpha)),
            (BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}

#[test]
fn custom_op1_no_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1_no_bwd(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&elu_t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

// Define a similar struct as Elu but with backward support.
fn bwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        T::one()
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        v.exp() * alpha
    }
}

struct EluBackward {
    alpha: f64,
}

impl CustomOp1 for EluBackward {
    fn name(&self) -> &'static str {
        "elu-bwd"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let storage = candle_core::map_dtype!(
            "elu-bwd",
            s,
            |s| cpu_backend::unary_map(s, l, |v| bwd(v, self.alpha)),
            (BF16, F16, F32, F64)
        );
        Ok((storage, l.shape().clone()))
    }
}

struct EluWithBackward(Elu);

impl EluWithBackward {
    fn new(alpha: f64) -> Self {
        Self(Elu { alpha })
    }
}

impl CustomOp1 for EluWithBackward {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        self.0.cpu_fwd(s, l)
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let alpha = self.0.alpha;
        let bwd = arg.apply_op1(EluBackward { alpha })?;
        Ok(Some(grad_res.mul(&bwd)?))
    }
}

#[test]
fn custom_op1_with_backward() -> Result<()> {
    let cpu = &Device::Cpu;
    let t = candle_core::Var::new(&[-2f32, 0f32, 2f32], cpu)?;
    let elu_t = t.apply_op1(EluWithBackward::new(2.))?;
    assert_eq!(to_vec1_round(&elu_t, 4)?, &[-1.7293, 0.0, 2.0]);

    let grads = elu_t.backward()?;
    let grad_x = grads.get(&t).unwrap();
    assert_eq!(to_vec1_round(grad_x, 4)?, [0.2707, 1.0, 1.0]);

    Ok(())
}

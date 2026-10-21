use super::tensor::Tensor;
use crate::TchError;

pub struct COptimizer {
    c_optimizer: *mut torch_sys::C_optimizer,
}

unsafe impl Send for COptimizer {}

impl std::fmt::Debug for COptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "optimizer")
    }
}

impl COptimizer {
    pub fn adam(
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    ) -> Result<COptimizer, TchError> {
        let c_optimizer =
            unsafe_torch_err!(torch_sys::ato_adam(lr, beta1, beta2, wd, eps, amsgrad));
        Ok(COptimizer { c_optimizer })
    }

    pub fn adamw(
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    ) -> Result<COptimizer, TchError> {
        let c_optimizer =
            unsafe_torch_err!(torch_sys::ato_adamw(lr, beta1, beta2, wd, eps, amsgrad));
        Ok(COptimizer { c_optimizer })
    }

    // Maybe we should use the builder pattern to provide default values for these ?
    pub fn rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: bool,
    ) -> Result<COptimizer, TchError> {
        let centered = i32::from(centered);
        let c_optimizer =
            unsafe_torch_err!(torch_sys::ato_rms_prop(lr, alpha, eps, wd, momentum, centered));
        Ok(COptimizer { c_optimizer })
    }

    pub fn sgd(
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<COptimizer, TchError> {
        let nesterov = i32::from(nesterov);
        let c_optimizer =
            unsafe_torch_err!(torch_sys::ato_sgd(lr, momentum, dampening, wd, nesterov));
        Ok(COptimizer { c_optimizer })
    }

    pub fn add_parameters(&mut self, t: &Tensor, group: usize) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_add_parameters(self.c_optimizer, t.c_tensor, group));
        Ok(())
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_learning_rate(self.c_optimizer, lr));
        Ok(())
    }

    pub fn set_learning_rate_group(&mut self, group: usize, lr: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_learning_rate_group(self.c_optimizer, group, lr));
        Ok(())
    }

    pub fn set_momentum(&mut self, m: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_momentum(self.c_optimizer, m));
        Ok(())
    }

    pub fn set_momentum_group(&mut self, group: usize, m: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_momentum_group(self.c_optimizer, group, m));
        Ok(())
    }

    pub fn set_weight_decay(&mut self, weight_decay: f64) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_weight_decay(self.c_optimizer, weight_decay));
        Ok(())
    }

    pub fn set_weight_decay_group(
        &mut self,
        group: usize,
        weight_decay: f64,
    ) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_set_weight_decay_group(
            self.c_optimizer,
            group,
            weight_decay
        ));
        Ok(())
    }

    pub fn zero_grad(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_zero_grad(self.c_optimizer));
        Ok(())
    }

    pub fn step(&self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::ato_step(self.c_optimizer));
        Ok(())
    }
}

impl Drop for COptimizer {
    fn drop(&mut self) {
        unsafe { torch_sys::ato_free(self.c_optimizer) }
    }
}

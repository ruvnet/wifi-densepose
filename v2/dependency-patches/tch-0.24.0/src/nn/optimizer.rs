//! Optimizers to be used for gradient-descent based training.
use super::var_store::{VarStore, Variables};
use crate::wrappers::optimizer::COptimizer;
use crate::{TchError, Tensor};
use std::sync::{Arc, Mutex};

/// An optimizer to run gradient descent.
#[derive(Debug)]
pub struct Optimizer {
    opt: COptimizer,
    variables: Arc<Mutex<Variables>>,
    variables_in_optimizer: usize,
}

/// Optimizer configurations. These configs can be used to build optimizer.
pub trait OptimizerConfig
where
    Self: std::marker::Sized,
{
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError>;

    /// Builds an optimizer with the specified learning rate handling variables stored in `vs`.
    fn build(self, vs: &VarStore, lr: f64) -> Result<Optimizer, TchError> {
        let mut opt = self.build_copt(lr)?;
        let v = vs.variables_.lock().unwrap();
        for var in &v.trainable_variables {
            opt.add_parameters(&var.tensor, var.group)?;
        }
        Ok(Optimizer {
            opt,
            variables: vs.variables_.clone(),
            variables_in_optimizer: v.trainable_variables.len(),
        })
    }
}

/// Parameters for the SGD optimizer.
#[derive(Debug, Copy, Clone)]
pub struct Sgd {
    pub momentum: f64,
    pub dampening: f64,
    pub wd: f64,
    pub nesterov: bool,
}

impl Default for Sgd {
    fn default() -> Self {
        Sgd { momentum: 0., dampening: 0., wd: 0., nesterov: false }
    }
}

/// Creates the configuration for a Stochastic Gradient Descent (SGD) optimizer.
pub fn sgd(momentum: f64, dampening: f64, wd: f64, nesterov: bool) -> Sgd {
    Sgd { momentum, dampening, wd, nesterov }
}

impl OptimizerConfig for Sgd {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::sgd(lr, self.momentum, self.dampening, self.wd, self.nesterov)
    }
}

/// Parameters for the Adam optimizer.
#[derive(Debug, Copy, Clone)]
pub struct Adam {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
    pub eps: f64,
    pub amsgrad: bool,
}

impl Default for Adam {
    fn default() -> Self {
        Adam { beta1: 0.9, beta2: 0.999, wd: 0., eps: 1e-8, amsgrad: false }
    }
}

/// Creates the configuration for the Adam optimizer.
pub fn adam(beta1: f64, beta2: f64, wd: f64) -> Adam {
    Adam { beta1, beta2, wd, eps: 1e-8, amsgrad: false }
}

impl Adam {
    pub fn beta1(mut self, b: f64) -> Self {
        self.beta1 = b;
        self
    }

    pub fn beta2(mut self, b: f64) -> Self {
        self.beta2 = b;
        self
    }

    pub fn wd(mut self, w: f64) -> Self {
        self.wd = w;
        self
    }

    pub fn eps(mut self, e: f64) -> Self {
        self.eps = e;
        self
    }

    pub fn amsgrad(mut self, a: bool) -> Self {
        self.amsgrad = a;
        self
    }
}

impl OptimizerConfig for Adam {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::adam(lr, self.beta1, self.beta2, self.wd, self.eps, self.amsgrad)
    }
}

/// Parameters for the AdamW optimizer.
#[derive(Debug, Copy, Clone)]
pub struct AdamW {
    pub beta1: f64,
    pub beta2: f64,
    pub wd: f64,
    pub eps: f64,
    pub amsgrad: bool,
}

impl Default for AdamW {
    fn default() -> Self {
        AdamW { beta1: 0.9, beta2: 0.999, wd: 0.01, eps: 1e-8, amsgrad: false }
    }
}

/// Creates the configuration for the AdamW optimizer.
pub fn adamw(beta1: f64, beta2: f64, wd: f64) -> AdamW {
    AdamW { beta1, beta2, wd, eps: 1e-8, amsgrad: false }
}

impl AdamW {
    pub fn beta1(mut self, b: f64) -> Self {
        self.beta1 = b;
        self
    }

    pub fn beta2(mut self, b: f64) -> Self {
        self.beta2 = b;
        self
    }

    pub fn wd(mut self, w: f64) -> Self {
        self.wd = w;
        self
    }

    pub fn eps(mut self, e: f64) -> Self {
        self.eps = e;
        self
    }

    pub fn amsgrad(mut self, a: bool) -> Self {
        self.amsgrad = a;
        self
    }
}

impl OptimizerConfig for AdamW {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::adamw(lr, self.beta1, self.beta2, self.wd, self.eps, self.amsgrad)
    }
}

/// Parameters for the RmsProp optimizer.
#[derive(Debug, Copy, Clone)]
pub struct RmsProp {
    pub alpha: f64,
    pub eps: f64,
    pub wd: f64,
    pub momentum: f64,
    pub centered: bool,
}

impl Default for RmsProp {
    fn default() -> Self {
        RmsProp { alpha: 0.99, eps: 1e-8, wd: 0., momentum: 0., centered: false }
    }
}

/// Creates the configuration for the RmsProp optimizer.
pub fn rms_prop(alpha: f64, eps: f64, wd: f64, momentum: f64, centered: bool) -> RmsProp {
    RmsProp { alpha, eps, wd, momentum, centered }
}

impl OptimizerConfig for RmsProp {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        COptimizer::rms_prop(lr, self.alpha, self.eps, self.wd, self.momentum, self.centered)
    }
}

impl Optimizer {
    fn add_missing_variables(&mut self) {
        let v = self.variables.lock().unwrap();
        if v.trainable_variables.len() > self.variables_in_optimizer {
            for var in &v.trainable_variables[self.variables_in_optimizer..] {
                self.opt.add_parameters(&var.tensor, var.group).unwrap();
            }
            self.variables_in_optimizer = v.trainable_variables.len();
        }
    }

    /// Zeroes the gradient for the tensors tracked by this optimizer.
    pub fn zero_grad(&mut self) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap()
    }

    /// Clips gradient value at some specified maximum value.
    pub fn clip_grad_value(&self, max: f64) {
        let v = self.variables.lock().unwrap();
        for var in v.trainable_variables.iter() {
            let mut grad = var.tensor.grad();
            if grad.defined() {
                let _t = grad.clamp_(-max, max);
            }
        }
    }

    /// Clips gradient L2 norm over all trainable parameters.
    ///
    /// The norm is computed over all gradients together, as if they were
    /// concatenated into a single vector.
    pub fn clip_grad_norm(&self, max: f64) {
        crate::no_grad(|| {
            let v = self.variables.lock().unwrap();
            let mut norms = vec![];
            for var in v.trainable_variables.iter() {
                let grad = var.tensor.grad();
                if grad.defined() {
                    norms.push(grad.norm());
                }
            }
            let total_norm = f64::try_from(Tensor::stack(&norms, 0).norm()).unwrap();
            let clip_coef = max / (total_norm + 1e-6);
            if clip_coef < 1.0 {
                for var in v.trainable_variables.iter() {
                    let mut grad = var.tensor.grad();
                    if grad.defined() {
                        let _t = grad.g_mul_scalar_(clip_coef);
                    }
                }
            }
        })
    }

    /// Performs an optimization step, updating the tracked tensors based on their gradients.
    pub fn step(&mut self) {
        self.add_missing_variables();
        self.opt.step().unwrap()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    pub fn backward_step(&mut self, loss: &Tensor) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap();
        loss.backward();
        self.opt.step().unwrap()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients are clipped based on `max` before being applied.
    pub fn backward_step_clip(&mut self, loss: &Tensor, max: f64) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap();
        loss.backward();
        self.clip_grad_value(max);
        self.opt.step().unwrap()
    }

    /// Applies a backward step pass, update the gradients, and performs an optimization step.
    ///
    /// The gradients L2 norm is clipped based on `max`.
    pub fn backward_step_clip_norm(&mut self, loss: &Tensor, max: f64) {
        self.add_missing_variables();
        self.opt.zero_grad().unwrap();
        loss.backward();
        self.clip_grad_norm(max);
        self.opt.step().unwrap()
    }

    /// Sets the optimizer learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.opt.set_learning_rate(lr).unwrap()
    }

    /// Sets the optimizer momentum.
    pub fn set_momentum(&mut self, m: f64) {
        self.opt.set_momentum(m).unwrap()
    }

    /// Sets the optimizer learning rate for a parameter group.
    pub fn set_lr_group(&mut self, group: usize, lr: f64) {
        self.opt.set_learning_rate_group(group, lr).unwrap()
    }

    /// Sets the optimizer momentum.
    pub fn set_momentum_group(&mut self, group: usize, m: f64) {
        self.opt.set_momentum_group(group, m).unwrap()
    }

    /// Returns all the trainable variables for this optimizer.
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        let variables = self.variables.lock().unwrap();
        variables.trainable_variables.iter().map(|v| v.tensor.shallow_clone()).collect()
    }

    /// Sets the optimizer weight decay.
    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.opt.set_weight_decay(weight_decay).unwrap()
    }

    /// Sets the optimizer weight decay.
    pub fn set_weight_decay_group(&mut self, group: usize, weight_decay: f64) {
        self.opt.set_weight_decay_group(group, weight_decay).unwrap()
    }
}

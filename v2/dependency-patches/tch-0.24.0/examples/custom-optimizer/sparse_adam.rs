use std::sync::{Arc, Mutex};
use tch::nn::{VarStore, Variables};
use tch::{no_grad, Device, Kind, Tensor};

/// Buffer of first/second order moment for the Adam optimizer
struct Buffer {
    pub first_moment: Tensor,
    pub second_moment: Tensor,
    idx: usize,
}

impl Buffer {
    pub fn new(size: &[i64]) -> Buffer {
        Buffer {
            first_moment: Tensor::zeros(size, (Kind::Float, Device::Cpu)),
            second_moment: Tensor::zeros(size, (Kind::Float, Device::Cpu)),
            idx: 1,
        }
    }

    // Return and increment the timestep
    pub fn inc(&mut self) -> usize {
        let old_val = self.idx;
        self.idx += 1;

        old_val
    }
}

/// Sparse-Adam optimizer supporting sparse and dense gradients
///
/// This demonstrates how a custom optimizer can be implemented. It handles a shared copy of the
/// variables store `vars` and updates it by traversing the computation graph.
pub struct SparseAdam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    force_sparse: bool,

    vars: Arc<Mutex<Variables>>,
    buffers: Vec<Buffer>,
}

impl SparseAdam {
    pub fn new(
        vs: &VarStore,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        force_sparse: bool,
    ) -> SparseAdam {
        let vars = vs.variables_.clone();

        // create buffers for every trainable variable
        let buffers = vars
            .lock()
            .unwrap()
            .trainable_variables
            .iter()
            .map(|x| Buffer::new(&x.tensor.size()))
            .collect();

        SparseAdam { lr, beta1, beta2, eps, force_sparse, vars, buffers }
    }

    /// Ensure that the gradient update is not part of the autograd routine
    pub fn step(&mut self) {
        no_grad(|| self._step());
    }

    pub fn _step(&mut self) {
        let mut vars = self.vars.lock().unwrap();

        // iterate through all trainable variables
        for (var, buffer) in vars.trainable_variables.iter_mut().zip(&mut self.buffers) {
            let mut grad = var.tensor.grad();

            // calculate both bias correction values
            let buffer_idx = buffer.inc();
            let bias_correction1 = 1.0 - self.beta1.powf(buffer_idx as f64);
            let bias_correction2 = 1.0 - self.beta2.powf(buffer_idx as f64);

            // check whether the gradient is sparse
            if grad.is_sparse() || self.force_sparse {
                // convert matrix to sparse matrix if necessary
                if !grad.is_sparse() {
                    grad = grad.to_sparse_sparse_dim(1);
                }

                // deduplicate coordinates in the sparse matrix
                let grad = grad.coalesce();
                // get indices and values of the sparse gradient
                let indices = grad.indices().squeeze();
                let values = grad.values();

                // for SGD we would do:
                //tensor.index_add_(0, &indices, &(-self.lr * &values));

                // update both moments
                // old = b*old + (1-b) * new <==> old += (1-b) * (new - old)
                let update_first_moment =
                    (1.0 - self.beta1) * (&values - buffer.first_moment.index_select(0, &indices));
                let update_second_moment = (1.0 - self.beta2)
                    * (&values * &values - buffer.second_moment.index_select(0, &indices));

                let _ = buffer.first_moment.index_add_(0, &indices, &update_first_moment);
                let _ = buffer.second_moment.index_add_(0, &indices, &update_second_moment);

                // first part of update step -lr * m_t / (1-b_1^t)
                let part1 =
                    buffer.first_moment.index_select(0, &indices) * (-self.lr / bias_correction1);
                // second part of update step sqrt(v_t / (1-b_2^t)) + eps
                let part2 = (buffer.second_moment.index_select(0, &indices) / bias_correction2)
                    .sqrt()
                    + self.eps;

                let _ = var.tensor.index_add_(0, &indices, &(part1 / part2));
            } else {
                // update first moment
                buffer.first_moment *= self.beta1;
                buffer.first_moment += (1.0 - self.beta1) * &grad;
                // update second raw moment
                buffer.second_moment *= self.beta2;
                let scaled_grad = grad * (1.0 - self.beta2).sqrt();
                let _ = buffer.second_moment.addcmul_(&scaled_grad, &scaled_grad);

                // first part of update step -lr * m_t / (1-b_1^t)
                let part1 = &buffer.first_moment * (-self.lr / bias_correction1);
                // second part of update step sqrt(v_t / (1-b_2^t)) + eps
                let part2 = (&buffer.second_moment / bias_correction2).sqrt() + self.eps;

                // calculate fraction and update parameters
                let _ = var.tensor.addcdiv_(&part1, &part2);
            }
        }
    }

    // zero the gradient of all trainable variables
    pub fn zero_grad(&mut self) {
        let mut vars = self.vars.lock().unwrap();
        for var in &mut vars.trainable_variables {
            var.tensor.zero_grad();
        }
    }
}

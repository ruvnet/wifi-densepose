//! Basic module traits defining the forward pass.
use crate::{data::Iter2, Device, Tensor};

/// The simplest module trait, defining a forward function.
pub trait Module: std::fmt::Debug + Send {
    fn forward(&self, xs: &Tensor) -> Tensor;
}

/// Module trait with an additional train parameter.
///
/// The train parameter is commonly used to have different behavior between training
/// and evaluation, e.g. when using dropout or batch-normalization.
pub trait ModuleT: std::fmt::Debug + Send {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;

    fn batch_accuracy_for_logits(
        &self,
        xs: &Tensor,
        ys: &Tensor,
        d: Device,
        batch_size: i64,
    ) -> f64 {
        let _no_grad = crate::no_grad_guard();
        let mut sum_accuracy = 0f64;
        let mut sample_count = 0f64;
        for (xs, ys) in Iter2::new(xs, ys, batch_size).return_smaller_last_batch() {
            let acc = self.forward_t(&xs.to_device(d), false).accuracy_for_logits(&ys.to_device(d));
            let size = xs.size()[0] as f64;
            sum_accuracy += f64::try_from(&acc).unwrap() * size;
            sample_count += size;
        }
        sum_accuracy / sample_count
    }
}

impl<T> ModuleT for T
where
    T: Module,
{
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

impl Tensor {
    pub fn apply<M: Module>(&self, m: &M) -> Tensor {
        m.forward(self)
    }

    pub fn apply_t<M: ModuleT>(&self, m: &M, train: bool) -> Tensor {
        m.forward_t(self, train)
    }

    pub fn apply_opt<M: Module>(&self, m: &Option<M>) -> Tensor {
        match m {
            Some(m) => m.forward(self),
            None => self.shallow_clone(),
        }
    }

    pub fn apply_opt_t<M: ModuleT>(&self, m: &Option<M>, train: bool) -> Tensor {
        match m {
            Some(m) => m.forward_t(self, train),
            None => self.shallow_clone(),
        }
    }
}

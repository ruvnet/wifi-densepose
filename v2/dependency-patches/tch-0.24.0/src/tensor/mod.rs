//! A Torch tensor.
use crate::{Device, Kind, TchError};

mod convert;
pub mod display;
pub mod index;
mod iter;
mod npy;
mod ops;
mod safetensors;

pub use super::wrappers::tensor::{
    autocast, no_grad, no_grad_guard, with_grad, NoGradGuard, Reduction, Tensor,
};
pub use index::{IndexOp, NewAxis, TensorIndexer};

pub trait Shape {
    fn to_shape(&self) -> Box<[i64]>;
}

macro_rules! impl_shape {
    ($v:expr) => {
        impl Shape for [i64; $v] {
            fn to_shape(&self) -> Box<[i64]> {
                Box::new(*self)
            }
        }
    };
}

impl_shape!(0);
impl_shape!(1);
impl_shape!(2);
impl_shape!(3);
impl_shape!(4);
impl_shape!(5);
impl_shape!(6);

impl Shape for () {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([])
    }
}

impl Shape for &[i64] {
    fn to_shape(&self) -> Box<[i64]> {
        (*self).into()
    }
}

impl Shape for i64 {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([*self])
    }
}

impl Shape for usize {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([*self as i64])
    }
}

impl Shape for i32 {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([i64::from(*self)])
    }
}

impl Shape for (i64,) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0])
    }
}

impl Shape for (i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1])
    }
}

impl Shape for (i64, i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1, self.2])
    }
}

impl Shape for (i64, i64, i64, i64) {
    fn to_shape(&self) -> Box<[i64]> {
        Box::new([self.0, self.1, self.2, self.3])
    }
}

impl Tensor {
    pub fn f_view<T: Shape>(&self, s: T) -> Result<Tensor, TchError> {
        self.f_view_(&s.to_shape())
    }

    pub fn view<T: Shape>(&self, s: T) -> Tensor {
        self.view_(&s.to_shape())
    }

    pub fn f_zero_pad1d(&self, left: i64, right: i64) -> Result<Tensor, TchError> {
        if self.dim() != 3 {
            return Err(TchError::Shape(format!(
                "expected a 3 dimension tensor, got {:?}",
                self.size()
            )));
        }
        self.f_constant_pad_nd([left, right])
    }

    pub fn zero_pad1d(&self, left: i64, right: i64) -> Tensor {
        self.f_zero_pad1d(left, right).unwrap()
    }

    pub fn f_zero_pad2d(
        &self,
        left: i64,
        right: i64,
        top: i64,
        bottom: i64,
    ) -> Result<Tensor, TchError> {
        if self.dim() != 4 {
            return Err(TchError::Shape(format!(
                "expected a 4 dimension tensor, got {:?}",
                self.size()
            )));
        }
        self.f_constant_pad_nd([left, right, top, bottom])
    }

    pub fn zero_pad2d(&self, left: i64, right: i64, top: i64, bottom: i64) -> Tensor {
        self.f_zero_pad2d(left, right, top, bottom).unwrap()
    }
}

impl<T: crate::kind::Element> From<&[T]> for Tensor {
    fn from(v: &[T]) -> Tensor {
        Tensor::from_slice(v)
    }
}

impl<T: crate::kind::Element> From<T> for Tensor {
    fn from(v: T) -> Tensor {
        Tensor::from_slice(&[v]).view(())
    }
}
impl Tensor {
    /// Casts a tensor to a specified kind.
    pub fn to_kind(&self, kind: Kind) -> Tensor {
        self.totype(kind)
    }

    pub fn f_to_kind(&self, kind: Kind) -> Result<Tensor, TchError> {
        self.f_totype(kind)
    }

    pub fn nll_loss(&self, targets: &Tensor) -> Tensor {
        self.g_nll_loss::<Tensor>(targets, None, Reduction::Mean, -100)
    }
}

impl Tensor {
    /// Computes the cross-entropy loss based on some logits and targets.
    pub fn cross_entropy_for_logits(&self, targets: &Tensor) -> Tensor {
        self.log_softmax(-1, Kind::Float).nll_loss(targets)
    }

    /// Returns the average accuracy for some given logits assuming that
    /// targets represent ground-truth.
    pub fn accuracy_for_logits(&self, targets: &Tensor) -> Tensor {
        self.argmax(-1, false).eq_tensor(targets).to_kind(Kind::Float).mean(Kind::Float)
    }

    pub fn random_batch(&self, batch_size: i64) -> Tensor {
        let len: i64 = self.size()[0];
        let index = Tensor::randint(len, [batch_size], (Kind::Int64, self.device()));
        self.index_select(0, &index)
    }

    pub fn random_batch2(t1: &Tensor, t2: &Tensor, batch_size: i64) -> (Tensor, Tensor) {
        let len1: i64 = t1.size()[0];
        let len2: i64 = t2.size()[0];
        if len1 != len2 {
            panic!("random_batch2: shape mismatch {:?} {:?}", t1.size(), t2.size())
        }
        let device1 = t1.device();
        let device2 = t2.device();
        if device1 != device2 {
            panic!("random_batch2: device mismatch {device1:?} {device2:?}")
        }
        let index = Tensor::randint(len1, [batch_size], (Kind::Int64, device1));
        let batch1 = t1.index_select(0, &index);
        let batch2 = t2.index_select(0, &index);
        (batch1, batch2)
    }

    /// Moves a tensor to a specified device.
    pub fn to_device(&self, device: Device) -> Tensor {
        self.to(device)
    }

    pub fn f_to_device(&self, device: Device) -> Result<Tensor, TchError> {
        self.f_to(device)
    }

    pub fn avg_pool2d_default(&self, ksize: i64) -> Tensor {
        self.avg_pool2d([ksize, ksize], [ksize, ksize], [0, 0], false, true, 1)
    }

    pub fn max_pool2d_default(&self, ksize: i64) -> Tensor {
        self.max_pool2d([ksize, ksize], [ksize, ksize], [0, 0], [1, 1], false)
    }

    /// Flattens a tensor.
    ///
    /// This returns a flattened version of the given tensor. The first dimension
    /// is preserved as it is assumed to be the mini-batch dimension.
    pub fn flat_view(&self) -> Tensor {
        self.view((self.size()[0], -1))
    }

    /// Converts a tensor to a one-hot encoded version.
    ///
    /// If the input has a size [N1, N2, ..., Nk], the returned tensor has a size
    /// [N1, ..., Nk, labels]. The returned tensor uses float values.
    /// Elements of the input vector are expected to be between 0 and labels-1.
    pub fn onehot(&self, labels: i64) -> Tensor {
        Tensor::zeros([self.size(), vec![labels]].concat(), (Kind::Float, self.device()))
            .scatter_value_(-1, &self.unsqueeze(-1).to_kind(Kind::Int64), 1.0)
    }

    /// Copies a tensor to a newly allocated tensor using the same shape and device.
    pub fn copy(&self) -> Tensor {
        let mut result = self.zeros_like();
        result.copy_(self);
        result
    }

    /// Copies the data from a two dimensional slice in a tensor object.
    pub fn from_slice2<T, U>(v: &[U]) -> Tensor
    where
        T: crate::kind::Element,
        U: AsRef<[T]>,
    {
        let inner: Vec<Tensor> = v.iter().map(|v| Tensor::from_slice(v.as_ref())).collect();
        Tensor::stack(&inner, 0)
    }

    pub fn to_mkldnn(&self) -> Tensor {
        self.g_to_mkldnn(self.kind())
    }
}

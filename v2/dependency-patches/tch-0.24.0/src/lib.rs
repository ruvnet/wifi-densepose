#[macro_use]
extern crate lazy_static;

pub mod data;

mod error;
pub use error::TchError;
pub type Result<T> = std::result::Result<T, error::TchError>;

pub(crate) mod wrappers;
pub use wrappers::device::{Cuda, Device};
pub use wrappers::jit::{self, CModule, IValue, TrainableCModule};
pub use wrappers::kind::{self, Kind};
pub use wrappers::layout::Layout;
pub use wrappers::optimizer::COptimizer;
#[cfg(feature = "python-extension")]
pub use wrappers::python;
pub use wrappers::scalar::Scalar;
pub use wrappers::utils;
pub use wrappers::{
    get_num_interop_threads, get_num_threads, manual_seed, set_num_interop_threads,
    set_num_threads, QEngine,
};

mod tensor;
pub use tensor::{
    autocast, display, index, no_grad, no_grad_guard, with_grad, IndexOp, NewAxis, NoGradGuard,
    Reduction, Shape, Tensor, TensorIndexer,
};

pub mod nn;
pub mod vision;

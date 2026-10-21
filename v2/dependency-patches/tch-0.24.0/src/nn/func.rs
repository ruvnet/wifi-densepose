//! Layers defined by closures.
use crate::Tensor;

/// A layer defined by a simple closure.
pub struct Func<'a> {
    f: Box<dyn 'a + Fn(&Tensor) -> Tensor + Send>,
}

impl std::fmt::Debug for Func<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "func")
    }
}

pub fn func<'a, F>(f: F) -> Func<'a>
where
    F: 'a + Fn(&Tensor) -> Tensor + Send,
{
    Func { f: Box::new(f) }
}

impl super::module::Module for Func<'_> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        (*self.f)(xs)
    }
}

/// A layer defined by a closure with an additional training parameter.
#[allow(clippy::type_complexity)]
pub struct FuncT<'a> {
    f: Box<dyn 'a + Fn(&Tensor, bool) -> Tensor + Send>,
}

impl std::fmt::Debug for FuncT<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "funcT")
    }
}

pub fn func_t<'a, F>(f: F) -> FuncT<'a>
where
    F: 'a + Fn(&Tensor, bool) -> Tensor + Send,
{
    FuncT { f: Box::new(f) }
}

impl super::module::ModuleT for FuncT<'_> {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        (*self.f)(xs, train)
    }
}

//! Indexing operations
//!
//! This module defines the `i` indexing operation. This can be used in various
//! scenarios.
//!
//! Using an integer index returns the slice obtained by selecting elements with
//! the specified index. Negative values can be used for the index, and `..` can
//! be used to get all the indexes from a given dimension.
//!
//! ```ignore
//! use crate::tch::{IndexOp, Tensor};
//! let tensor = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).view((2, 3));
//! let t = tensor.i(1);
//! let t = tensor.i((.., -2));
//! ```
//!
//! Indexes like `1..`, `..1`, or `1..2`, can be used to narrow a dimension.
//!
//! ```ignore
//! use crate::tch::{IndexOp, Tensor};
//! let tensor = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).view((2, 3));
//! let t = tensor.i((.., 1..));
//! assert_eq!(t.size(), [2, 2]);
//! assert_eq!(Vec::<i64>::from(t.contiguous().view(-1)), [2, 3, 5, 6]);
//! let t = tensor.i((..1, ..));
//! assert_eq!(t.size(), [1, 3]);
//! assert_eq!(Vec::<i64>::from(t.contiguous().view(-1)), [1, 2, 3]);
//! let t = tensor.i((.., 1..2));
//! assert_eq!(t.size(), [2, 1]);
//! assert_eq!(Vec::<i64>::from(t.contiguous().view(-1)), [2, 5]);
//! let t = tensor.i((.., 1..=2));
//! assert_eq!(t.size(), [2, 2]);
//! assert_eq!(Vec::<i64>::from(t.contiguous().view(-1)), [2, 3, 5, 6]);
//! ```
//!
//! The `NewAxis` index can be used to insert a dimension.
//!
//! ```ignore
//! use crate::tch::{IndexOp, NewAxis, Tensor};
//! let tensor = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).view((2, 3));
//! let t = tensor.i((NewAxis,));
//! assert_eq!(t.size(), &[1, 2, 3]);
//! let t = tensor.i((.., .., NewAxis));
//! assert_eq!(t.size(), &[2, 3, 1]);
//! ```
//!
//! Unlike NumPy, the `i` operation does not support advanced indexing.
//! The result can be different from NumPy with same set of arguments.
//! For example, `tensor.i(..1, vec![0, 3], vec![2, 1, 3])` does narrowing
//! on first dimension, and index selection on second and third dimensions.
//! The analogous NumPy indexing `array[:1, [0, 3], [2, 1, 3]]` throws
//! shape mismatch error due to advanced indexing rule. Another distinction
//! is that `i` guarantees the input and result tensor shares the same
//! underlying storage, while NumPy may copy the tensor in certain scenarios.
use crate::{Result, TchError, Tensor};
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

#[derive(Debug, PartialEq, Eq)]
pub struct NewAxis;

#[derive(Debug, PartialEq)]
pub enum TensorIndexer {
    Select(i64),
    Narrow(Bound<i64>, Bound<i64>),
    IndexSelect(Tensor),
    InsertNewAxis,
}

impl From<NewAxis> for TensorIndexer {
    fn from(_index: NewAxis) -> Self {
        TensorIndexer::InsertNewAxis
    }
}

impl From<i64> for TensorIndexer {
    fn from(index: i64) -> Self {
        TensorIndexer::Select(index)
    }
}

impl From<&[i64]> for TensorIndexer {
    fn from(index: &[i64]) -> Self {
        let tensor = index.into();
        TensorIndexer::IndexSelect(tensor)
    }
}

impl From<Vec<i64>> for TensorIndexer {
    fn from(index: Vec<i64>) -> Self {
        let tensor = Tensor::from_slice(&index);
        TensorIndexer::IndexSelect(tensor)
    }
}

impl From<&Tensor> for TensorIndexer {
    fn from(tensor: &Tensor) -> Self {
        TensorIndexer::IndexSelect(tensor.shallow_clone())
    }
}

macro_rules! impl_from_range {
    ($range_type:ty) => {
        impl From<$range_type> for TensorIndexer {
            fn from(range: $range_type) -> Self {
                use std::ops::Bound::*;

                let start = match range.start_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                let end = match range.end_bound() {
                    Included(idx) => Included(*idx),
                    Excluded(idx) => Excluded(*idx),
                    Unbounded => Unbounded,
                };

                TensorIndexer::Narrow(start, end)
            }
        }
    };
}

impl_from_range!(Range<i64>);
impl_from_range!(RangeFrom<i64>);
impl_from_range!(RangeFull);
impl_from_range!(RangeInclusive<i64>);
impl_from_range!(RangeTo<i64>);
impl_from_range!(RangeToInclusive<i64>);

pub trait IndexOp<T> {
    fn i(&self, index: T) -> Tensor;
    fn f_i(&self, index: T) -> Result<Tensor>;
}

impl<A> IndexOp<A> for Tensor
where
    A: Into<TensorIndexer>,
{
    fn i(&self, index: A) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: A) -> Result<Tensor> {
        self.f_indexer(&[index.into()])
    }
}

impl<A> IndexOp<(A,)> for Tensor
where
    A: Into<TensorIndexer>,
{
    fn i(&self, index: (A,)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A,)) -> Result<Tensor> {
        let idx_a = index.0.into();
        self.f_indexer(&[idx_a])
    }
}

impl<A, B> IndexOp<(A, B)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A, B)) -> Result<Tensor> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        self.f_indexer(&[idx_a, idx_b])
    }
}

impl<A, B, C> IndexOp<(A, B, C)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A, B, C)) -> Result<Tensor> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        self.f_indexer(&[idx_a, idx_b, idx_c])
    }
}

impl<A, B, C, D> IndexOp<(A, B, C, D)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A, B, C, D)) -> Result<Tensor> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        self.f_indexer(&[idx_a, idx_b, idx_c, idx_d])
    }
}

impl<A, B, C, D, E> IndexOp<(A, B, C, D, E)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
    E: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D, E)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A, B, C, D, E)) -> Result<Tensor> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        let idx_e = index.4.into();
        self.f_indexer(&[idx_a, idx_b, idx_c, idx_d, idx_e])
    }
}

impl<A, B, C, D, E, F> IndexOp<(A, B, C, D, E, F)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
    E: Into<TensorIndexer>,
    F: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D, E, F)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A, B, C, D, E, F)) -> Result<Tensor> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        let idx_e = index.4.into();
        let idx_f = index.5.into();
        self.f_indexer(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f])
    }
}

impl<A, B, C, D, E, F, G> IndexOp<(A, B, C, D, E, F, G)> for Tensor
where
    A: Into<TensorIndexer>,
    B: Into<TensorIndexer>,
    C: Into<TensorIndexer>,
    D: Into<TensorIndexer>,
    E: Into<TensorIndexer>,
    F: Into<TensorIndexer>,
    G: Into<TensorIndexer>,
{
    fn i(&self, index: (A, B, C, D, E, F, G)) -> Tensor {
        self.f_i(index).unwrap()
    }

    fn f_i(&self, index: (A, B, C, D, E, F, G)) -> Result<Tensor> {
        let idx_a = index.0.into();
        let idx_b = index.1.into();
        let idx_c = index.2.into();
        let idx_d = index.3.into();
        let idx_e = index.4.into();
        let idx_f = index.5.into();
        let idx_g = index.6.into();
        self.f_indexer(&[idx_a, idx_b, idx_c, idx_d, idx_e, idx_f, idx_g])
    }
}

impl Tensor {
    fn f_indexer(&self, index_spec: &[TensorIndexer]) -> Result<Tensor> {
        use std::ops::Bound::*;
        use TensorIndexer::*;

        // Make sure n. non-newaxis does not exceed n. of dimensions
        let n_newaxis = index_spec.iter().filter(|spec| *spec == &InsertNewAxis).count();

        if index_spec.len() > self.size().len() + n_newaxis {
            return Err(TchError::Shape(format!(
                "too many indices for tensor of dimension {}",
                self.size().len()
            )));
        }

        // Make sure tensors conform the format
        for spec in index_spec.iter() {
            use super::Kind::*;
            if let IndexSelect(tensor) = spec {
                if tensor.size().len() != 1 {
                    return Err(TchError::Shape(
                        "Multi-dimensional tensor is not supported for indexing".to_string(),
                    ));
                }
                match tensor.f_kind()? {
                    Int64 => {}
                    Int16 => {}
                    Int8 => {}
                    Int => {}
                    _ => {
                        return Err(TchError::Kind(format!("the kind of tensors used as indices must be one of {Int64:?}, {Int16:?}, {Int8:?}, {Int:?}")));
                    }
                }
            }
        }

        // Apply indexing from left to right
        let mut curr_tensor = self.shallow_clone();
        let mut curr_idx: i64 = 0;

        for spec in index_spec.iter() {
            let (next_tensor, next_idx) = match spec {
                InsertNewAxis => (curr_tensor.unsqueeze(curr_idx), curr_idx + 1),
                Select(index) => (
                    curr_tensor.select(curr_idx, *index),
                    curr_idx, // not advanced because select() squeezes dimension
                ),
                Narrow(start, end) => {
                    if let Some((start, length)) = match (start, end) {
                        (Unbounded, Unbounded) => None,
                        (Included(start), Unbounded) => {
                            let dim_len = curr_tensor.size()[curr_idx as usize];
                            Some((*start, dim_len - *start))
                        }
                        (Excluded(start), Unbounded) => {
                            let dim_len = curr_tensor.size()[curr_idx as usize];
                            Some((*start + 1, dim_len - *start - 1))
                        }
                        (Unbounded, Included(end)) => Some((0, *end + 1)),
                        (Unbounded, Excluded(end)) => Some((0, *end)),
                        (Included(start), Included(end)) => Some((*start, *end - *start + 1)),
                        (Included(start), Excluded(end)) => Some((*start, *end - *start)),
                        (Excluded(start), Included(end)) => Some((*start + 1, *end - *start)),
                        (Excluded(start), Excluded(end)) => Some((*start + 1, *end - *start - 1)),
                    } {
                        (curr_tensor.f_narrow(curr_idx, start, length.max(0))?, curr_idx + 1)
                    } else {
                        (curr_tensor, curr_idx + 1)
                    }
                }
                IndexSelect(index_tensor) => {
                    let index_tensor = index_tensor.to_device(curr_tensor.device());
                    (curr_tensor.index_select(curr_idx, &index_tensor), curr_idx + 1)
                }
            };
            curr_tensor = next_tensor;
            curr_idx = next_idx;
        }
        Ok(curr_tensor)
    }
}

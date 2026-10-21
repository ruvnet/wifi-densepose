use crate::{DType, DeviceLocation, Layout, MetalError, Shape};

#[derive(Debug, Clone)]
pub struct MatMulUnexpectedStriding {
    pub lhs_l: Layout,
    pub rhs_l: Layout,
    pub bmnk: (usize, usize, usize, usize),
    pub msg: &'static str,
}

/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    // === Dimension Index Errors ===
    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: i32,
        op: &'static str,
    },

    #[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
    DuplicateDimIndex {
        shape: Shape,
        dims: Vec<usize>,
        op: &'static str,
    },

    // === Shape Errors ===
    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape {
        msg: String,
        expected: Shape,
        got: Shape,
    },

    #[error(
        "Shape mismatch, got buffer of size {buffer_size} which is compatible with shape {shape:?}"
    )]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
    ShapeMismatchCat {
        dim: usize,
        first_shape: Shape,
        n: usize,
        nth_shape: Shape,
    },

    #[error("Cannot divide tensor of shape {shape:?} equally along dim {dim} into {n_parts}")]
    ShapeMismatchSplit {
        shape: Shape,
        dim: usize,
        n_parts: usize,
    },

    #[error("{op} can only be performed on a single dimension")]
    OnlySingleDimension { op: &'static str, dims: Vec<usize> },

    #[error("empty tensor for {op}")]
    EmptyTensor { op: &'static str },

    // === Device Errors ===
    #[error("device mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DeviceMismatchBinaryOp {
        lhs: DeviceLocation,
        rhs: DeviceLocation,
        op: &'static str,
    },

    // === Op Specific Errors ===
    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },

    #[error("conv1d invalid args {msg}: inp: {inp_shape:?}, k: {k_shape:?}, pad: {padding}, stride: {stride}")]
    Conv1dInvalidArgs {
        inp_shape: Shape,
        k_shape: Shape,
        padding: usize,
        stride: usize,
        msg: &'static str,
    },

    #[error("{op} invalid index {index} with dim size {size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("cannot set variable {msg}")]
    CannotSetVar { msg: &'static str },

    // Box indirection to avoid large variant.
    #[error("{0:?}")]
    MatMulUnexpectedStriding(Box<MatMulUnexpectedStriding>),

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("{op} expects at least two tensors")]
    OpRequiresAtLeastTwoTensors { op: &'static str },

    #[error("backward is not supported for {op}")]
    BackwardNotSupported { op: &'static str },

    // === Other Errors ===
    #[error("the candle crate has not been built with cuda support")]
    NotCompiledWithCudaSupport,

    #[error("the candle crate has not been built with metal support")]
    NotCompiledWithMetalSupport,

    #[error("cannot find tensor {path}")]
    CannotFindTensor { path: String },

    // === Wrapped Errors ===
    #[error(transparent)]
    Cuda(Box<dyn std::error::Error + Send + Sync>),

    #[error("Metal error {0}")]
    Metal(#[from] MetalError),

    #[error(transparent)]
    TryFromIntError(#[from] core::num::TryFromIntError),

    #[error("npy/npz error {0}")]
    Npy(String),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// SafeTensor error.
    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    #[error("unsupported safetensor dtype {0:?}")]
    UnsupportedSafeTensorDtype(safetensors::Dtype),

    /// Arbitrary errors wrapping.
    #[error(transparent)]
    Wrapped(Box<dyn std::error::Error + Send + Sync>),

    /// Adding path information to an error.
    #[error("path: {path:?} {inner}")]
    WithPath {
        inner: Box<Self>,
        path: std::path::PathBuf,
    },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn wrap(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    pub fn msg(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }

    pub fn with_path<P: AsRef<std::path::Path>>(self, p: P) -> Self {
        Self::WithPath {
            inner: Box::new(self),
            path: p.as_ref().to_path_buf(),
        }
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

pub fn zip<T, U>(r1: Result<T>, r2: Result<U>) -> Result<(T, U)> {
    match (r1, r2) {
        (Ok(r1), Ok(r2)) => Ok((r1, r2)),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

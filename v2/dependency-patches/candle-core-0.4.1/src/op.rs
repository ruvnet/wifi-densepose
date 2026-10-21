#![allow(clippy::redundant_closure_call)]
use crate::{CpuStorage, CudaStorage, Layout, MetalStorage, Result, Shape, Tensor};
use half::{bf16, f16};
use num_traits::float::Float;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl ReduceOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::ArgMax => "argmax",
            Self::ArgMin => "argmin",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }
}

// These ops return the same type as their input type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

// Unary ops with no argument
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Neg,
    Recip,
    Sqr,
    Sqrt,
    Gelu,
    GeluErf,
    Erf,
    Relu,
    Silu,
    Tanh,
    Floor,
    Ceil,
    Round,
}

#[derive(Clone)]
pub enum Op {
    Binary(Tensor, Tensor, BinaryOp),
    Unary(Tensor, UnaryOp),
    Cmp(Tensor, CmpOp),
    // The third argument is the reduced shape with `keepdim=true`.
    Reduce(Tensor, ReduceOp, Vec<usize>),
    Matmul(Tensor, Tensor),
    Gather(Tensor, Tensor, usize),
    ScatterAdd(Tensor, Tensor, Tensor, usize),
    IndexSelect(Tensor, Tensor, usize),
    IndexAdd(Tensor, Tensor, Tensor, usize),
    WhereCond(Tensor, Tensor, Tensor),

    #[allow(dead_code)]
    Conv1D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
    },

    #[allow(dead_code)]
    ConvTranspose1D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    },

    #[allow(dead_code)]
    Conv2D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
    },

    #[allow(dead_code)]
    ConvTranspose2D {
        arg: Tensor,
        kernel: Tensor,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    },

    AvgPool2D {
        arg: Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    MaxPool2D {
        arg: Tensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    UpsampleNearest1D {
        arg: Tensor,
        target_size: usize,
    },
    UpsampleNearest2D {
        arg: Tensor,
        target_h: usize,
        target_w: usize,
    },

    Cat(Vec<Tensor>, usize),

    #[allow(dead_code)] // add is currently unused.
    Affine {
        arg: Tensor,
        mul: f64,
        add: f64,
    },
    ToDType(Tensor),
    Copy(Tensor),
    Broadcast(Tensor),
    Narrow(Tensor, usize, usize, usize),
    SliceScatter0(Tensor, Tensor, usize),
    Reshape(Tensor),
    ToDevice(Tensor),
    Transpose(Tensor, usize, usize),
    Permute(Tensor, Vec<usize>),
    Elu(Tensor, f64),
    Powf(Tensor, f64),
    CustomOp1(Tensor, std::sync::Arc<Box<dyn CustomOp1 + Send + Sync>>),
    CustomOp2(
        Tensor,
        Tensor,
        std::sync::Arc<Box<dyn CustomOp2 + Send + Sync>>,
    ),
    CustomOp3(
        Tensor,
        Tensor,
        Tensor,
        std::sync::Arc<Box<dyn CustomOp3 + Send + Sync>>,
    ),
}

/// Unary ops that can be defined in user-land.
pub trait CustomOp1 {
    // Box<dyn> does not support const yet, so use a function to get the name.
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(&self, _storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _storage: &MetalStorage,
        _layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, _arg: &Tensor, _res: &Tensor, _grad_res: &Tensor) -> Result<Option<Tensor>> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait CustomOp2 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait CustomOp3 {
    fn name(&self) -> &'static str;

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)>;

    /// The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        Err(crate::Error::Cuda(
            format!("no cuda implementation for {}", self.name()).into(),
        ))
    }

    /// The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        Err(crate::Error::Metal(
            format!("no metal implementation for {}", self.name()).into(),
        ))
    }

    fn bwd(
        &self,
        _arg1: &Tensor,
        _arg2: &Tensor,
        _arg3: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        Err(crate::Error::BackwardNotSupported { op: self.name() })
    }
}

pub trait UnaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16) -> bf16;
    fn f16(v1: f16) -> f16;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
    fn u8(v1: u8) -> u8;
    fn u32(v1: u32) -> u32;
    fn i64(v1: i64) -> i64;

    // There is no very good way to represent optional function in traits so we go for an explicit
    // boolean flag to mark the function as existing.
    const BF16_VEC: bool = false;
    fn bf16_vec(_xs: &[bf16], _ys: &mut [bf16]) {}
    const F16_VEC: bool = false;
    fn f16_vec(_xs: &[f16], _ys: &mut [f16]) {}
    const F32_VEC: bool = false;
    fn f32_vec(_xs: &[f32], _ys: &mut [f32]) {}
    const F64_VEC: bool = false;
    fn f64_vec(_xs: &[f64], _ys: &mut [f64]) {}
}

pub trait BinaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn bf16(v1: bf16, v2: bf16) -> bf16;
    fn f16(v1: f16, v2: f16) -> f16;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
    fn u32(v1: u32, v2: u32) -> u32;
    fn i64(v1: i64, v2: i64) -> i64;

    const BF16_VEC: bool = false;
    fn bf16_vec(_xs1: &[bf16], _xs2: &[bf16], _ys: &mut [bf16]) {}
    const F16_VEC: bool = false;
    fn f16_vec(_xs1: &[f16], _xs2: &[f16], _ys: &mut [f16]) {}
    const F32_VEC: bool = false;
    fn f32_vec(_xs1: &[f32], _xs2: &[f32], _ys: &mut [f32]) {}
    const F64_VEC: bool = false;
    fn f64_vec(_xs1: &[f64], _xs2: &[f64], _ys: &mut [f64]) {}
    const U8_VEC: bool = false;
    fn u8_vec(_xs1: &[u8], _xs2: &[u8], _ys: &mut [u8]) {}
    const U32_VEC: bool = false;
    fn u32_vec(_xs1: &[u32], _xs2: &[u32], _ys: &mut [u32]) {}
    const I64_VEC: bool = false;
    fn i64_vec(_xs1: &[i64], _xs2: &[i64], _ys: &mut [i64]) {}
}

pub(crate) struct Add;
pub(crate) struct Div;
pub(crate) struct Mul;
pub(crate) struct Sub;
pub(crate) struct Maximum;
pub(crate) struct Minimum;
pub(crate) struct Exp;
pub(crate) struct Log;
pub(crate) struct Sin;
pub(crate) struct Cos;
pub(crate) struct Abs;
pub(crate) struct Neg;
pub(crate) struct Recip;
pub(crate) struct Sqr;
pub(crate) struct Sqrt;
pub(crate) struct Gelu;
pub(crate) struct GeluErf;
pub(crate) struct Erf;
pub(crate) struct Relu;
pub(crate) struct Silu;
pub(crate) struct Tanh;
pub(crate) struct Floor;
pub(crate) struct Ceil;
pub(crate) struct Round;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr, $f32_vec: ident, $f64_vec: ident) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16(v1: bf16, v2: bf16) -> bf16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f16(v1: f16, v2: f16) -> f16 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }
            #[inline(always)]
            fn i64(v1: i64, v2: i64) -> i64 {
                $e(v1, v2)
            }

            #[cfg(feature = "mkl")]
            const F32_VEC: bool = true;
            #[cfg(feature = "mkl")]
            const F64_VEC: bool = true;
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
                crate::mkl::$f32_vec(xs1, xs2, ys)
            }
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
                crate::mkl::$f64_vec(xs1, xs2, ys)
            }

            #[cfg(feature = "accelerate")]
            const F32_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            const F64_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f32_vec(xs1: &[f32], xs2: &[f32], ys: &mut [f32]) {
                crate::accelerate::$f32_vec(xs1, xs2, ys)
            }
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f64_vec(xs1: &[f64], xs2: &[f64], ys: &mut [f64]) {
                crate::accelerate::$f64_vec(xs1, xs2, ys)
            }
        }
    };
}

bin_op!(Add, "add", |v1, v2| v1 + v2, vs_add, vd_add);
bin_op!(Sub, "sub", |v1, v2| v1 - v2, vs_sub, vd_sub);
bin_op!(Mul, "mul", |v1, v2| v1 * v2, vs_mul, vd_mul);
bin_op!(Div, "div", |v1, v2| v1 / v2, vs_div, vd_div);
bin_op!(
    Minimum,
    "minimum",
    |v1, v2| if v1 > v2 { v2 } else { v1 },
    vs_min,
    vd_min
);
bin_op!(
    Maximum,
    "maximum",
    |v1, v2| if v1 < v2 { v2 } else { v1 },
    vs_max,
    vd_max
);

#[allow(clippy::redundant_closure_call)]
macro_rules! unary_op {
    ($op: ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline(always)]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }
        }
    };

    ($op: ident, $name: literal, $a: ident, $e: expr, $f32_vec:ident, $f64_vec:ident) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            #[inline(always)]
            fn bf16($a: bf16) -> bf16 {
                $e
            }
            #[inline(always)]
            fn f16($a: f16) -> f16 {
                $e
            }
            #[inline(always)]
            fn f32($a: f32) -> f32 {
                $e
            }
            #[inline(always)]
            fn f64($a: f64) -> f64 {
                $e
            }
            #[inline(always)]
            fn u8(_: u8) -> u8 {
                todo!("no unary function for u8")
            }
            #[inline(always)]
            fn u32(_: u32) -> u32 {
                todo!("no unary function for u32")
            }
            #[inline(always)]
            fn i64(_: i64) -> i64 {
                todo!("no unary function for i64")
            }

            #[cfg(feature = "mkl")]
            const F32_VEC: bool = true;
            #[cfg(feature = "mkl")]
            const F64_VEC: bool = true;
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f32_vec(xs: &[f32], ys: &mut [f32]) {
                crate::mkl::$f32_vec(xs, ys)
            }
            #[cfg(feature = "mkl")]
            #[inline(always)]
            fn f64_vec(xs: &[f64], ys: &mut [f64]) {
                crate::mkl::$f64_vec(xs, ys)
            }

            #[cfg(feature = "accelerate")]
            const F32_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            const F64_VEC: bool = true;
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f32_vec(xs: &[f32], ys: &mut [f32]) {
                crate::accelerate::$f32_vec(xs, ys)
            }
            #[cfg(feature = "accelerate")]
            #[inline(always)]
            fn f64_vec(xs: &[f64], ys: &mut [f64]) {
                crate::accelerate::$f64_vec(xs, ys)
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp(), vs_exp, vd_exp);
unary_op!(Log, "log", v, v.ln(), vs_ln, vd_ln);
unary_op!(Sin, "sin", v, v.sin(), vs_sin, vd_sin);
unary_op!(Cos, "cos", v, v.cos(), vs_cos, vd_cos);
unary_op!(Tanh, "tanh", v, v.tanh(), vs_tanh, vd_tanh);
unary_op!(Neg, "neg", v, -v);
unary_op!(Recip, "recip", v, v.recip());
unary_op!(Sqr, "sqr", v, v * v, vs_sqr, vd_sqr);
unary_op!(Sqrt, "sqrt", v, v.sqrt(), vs_sqrt, vd_sqrt);

/// Tanh based approximation of the `gelu` operation
/// GeluErf is the more precise one.
/// <https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions>
impl UnaryOpT for Gelu {
    const NAME: &'static str = "gelu";
    const V: Self = Gelu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f32_const(0.5)
            * v
            * (bf16::ONE
                + bf16::tanh(
                    (bf16::from_f32_const(2.0) / bf16::PI).sqrt()
                        * v
                        * (bf16::ONE + bf16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f32_const(0.5)
            * v
            * (f16::ONE
                + f16::tanh(
                    (f16::from_f32_const(2.0) / f16::PI).sqrt()
                        * v
                        * (f16::ONE + f16::from_f32_const(0.044715) * v * v),
                ))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        0.5 * v
            * (1.0
                + f32::tanh((2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        0.5 * v
            * (1.0
                + f64::tanh((2.0f64 / std::f64::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)))
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    const KERNEL: &'static str = "ugelu";

    #[cfg(feature = "mkl")]
    const F32_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::mkl::vs_gelu(xs, ys)
    }

    #[cfg(feature = "mkl")]
    const F64_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::mkl::vd_gelu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F32_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::accelerate::vs_gelu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F64_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::accelerate::vd_gelu(xs, ys)
    }
}

/// `erf` operation
/// <https://en.wikipedia.org/wiki/Error_function>
impl UnaryOpT for Erf {
    const NAME: &'static str = "erf";
    const KERNEL: &'static str = "uerf";
    const V: Self = Erf;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        Self::f64(v as f64) as f32
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        crate::cpu::erf::erf(v)
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
}

/// Silu operation
impl UnaryOpT for Silu {
    const NAME: &'static str = "silu";
    const V: Self = Silu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v / (bf16::ONE + (-v).exp())
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v / (f16::ONE + (-v).exp())
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
    const KERNEL: &'static str = "usilu";

    #[cfg(feature = "mkl")]
    const F32_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::mkl::vs_silu(xs, ys)
    }

    #[cfg(feature = "mkl")]
    const F64_VEC: bool = true;

    #[cfg(feature = "mkl")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::mkl::vd_silu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F32_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f32_vec(xs: &[f32], ys: &mut [f32]) {
        crate::accelerate::vs_silu(xs, ys)
    }

    #[cfg(feature = "accelerate")]
    const F64_VEC: bool = true;

    #[cfg(feature = "accelerate")]
    #[inline(always)]
    fn f64_vec(xs: &[f64], ys: &mut [f64]) {
        crate::accelerate::vd_silu(xs, ys)
    }
}

impl UnaryOpT for Abs {
    const NAME: &'static str = "abs";
    const KERNEL: &'static str = "uabs";
    const V: Self = Abs;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.abs()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.abs()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.abs()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.abs()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v.abs()
    }
}

impl UnaryOpT for Ceil {
    const NAME: &'static str = "ceil";
    const KERNEL: &'static str = "uceil";
    const V: Self = Ceil;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.ceil()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.ceil()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.ceil()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.ceil()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
}

impl UnaryOpT for Floor {
    const NAME: &'static str = "floor";
    const KERNEL: &'static str = "ufloor";
    const V: Self = Floor;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.floor()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.floor()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.floor()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.floor()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
}

impl UnaryOpT for Round {
    const NAME: &'static str = "round";
    const KERNEL: &'static str = "uround";
    const V: Self = Round;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.round()
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.round()
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.round()
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.round()
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
}

impl UnaryOpT for GeluErf {
    const NAME: &'static str = "gelu_erf";
    const KERNEL: &'static str = "ugelu_erf";
    const V: Self = GeluErf;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        bf16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        f16::from_f64(Self::f64(v.to_f64()))
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        Self::f64(v as f64) as f32
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        (crate::cpu::erf::erf(v / 2f64.sqrt()) + 1.) * 0.5 * v
    }
    #[inline(always)]
    fn u8(_: u8) -> u8 {
        0
    }
    #[inline(always)]
    fn u32(_: u32) -> u32 {
        0
    }
    #[inline(always)]
    fn i64(_: i64) -> i64 {
        0
    }
}

impl UnaryOpT for Relu {
    const NAME: &'static str = "relu";
    const KERNEL: &'static str = "urelu";
    const V: Self = Relu;
    #[inline(always)]
    fn bf16(v: bf16) -> bf16 {
        v.max(bf16::ZERO)
    }
    #[inline(always)]
    fn f16(v: f16) -> f16 {
        v.max(f16::ZERO)
    }
    #[inline(always)]
    fn f32(v: f32) -> f32 {
        v.max(0f32)
    }
    #[inline(always)]
    fn f64(v: f64) -> f64 {
        v.max(0f64)
    }
    #[inline(always)]
    fn u8(v: u8) -> u8 {
        v
    }
    #[inline(always)]
    fn u32(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn i64(v: i64) -> i64 {
        v
    }
}

/// `BackpropOp` is a wrapper around `Option<Op>`. The main goal is to ensure that dependencies are
/// properly checked when creating a new value
#[derive(Clone)]
pub struct BackpropOp(Option<Op>);

impl BackpropOp {
    pub(crate) fn none() -> Self {
        BackpropOp(None)
    }

    pub(crate) fn new1(arg: &Tensor, f: impl Fn(Tensor) -> Op) -> Self {
        let op = if arg.track_op() {
            Some(f(arg.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new2(arg1: &Tensor, arg2: &Tensor, f: impl Fn(Tensor, Tensor) -> Op) -> Self {
        let op = if arg1.track_op() || arg2.track_op() {
            Some(f(arg1.clone(), arg2.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new3(
        arg1: &Tensor,
        arg2: &Tensor,
        arg3: &Tensor,
        f: impl Fn(Tensor, Tensor, Tensor) -> Op,
    ) -> Self {
        let op = if arg1.track_op() || arg2.track_op() || arg3.track_op() {
            Some(f(arg1.clone(), arg2.clone(), arg3.clone()))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn new<A: AsRef<Tensor>>(args: &[A], f: impl Fn(Vec<Tensor>) -> Op) -> Self {
        let op = if args.iter().any(|arg| arg.as_ref().track_op()) {
            let args: Vec<Tensor> = args.iter().map(|arg| arg.as_ref().clone()).collect();
            Some(f(args))
        } else {
            None
        };
        Self(op)
    }

    pub(crate) fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

impl std::ops::Deref for BackpropOp {
    type Target = Option<Op>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

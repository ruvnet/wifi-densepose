use crate::{CpuStorage, Device, Result, Shape, Storage, Tensor};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::f16;

pub use k_quants::GgmlType;

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl QStorage {
    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?)?;
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid dequantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<[u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Metal(_) | QStorage::Cuda(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
        }
    }
    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]) -> Result<()>;
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) -> Result<()> {
        T::from_float(xs, self)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys)?;
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if dims[dims.len() - 1] % block_size != 0 {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if elem_count % block_size != 0 {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        let is_variable = false;
        crate::tensor::from_storage(storage, self.shape.clone(), none, is_variable)
            .to_device(device)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&Device::Cpu)?;
            Self::Tensor(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];
        self_storage.matmul_t((dst_shape.elem_count() / n, k, n), slice, &mut dst_storage)?;
        Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
        }
    }
}

use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use candle_metal_kernels;
use candle_metal_kernels::Kernels;
use metal;
use metal::{Buffer, CommandBuffer, CommandQueue, MTLResourceOptions, NSUInteger};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock, TryLockError};

/// Simple way to catch lock error without
/// depending on T
#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

impl<T> From<TryLockError<T>> for MetalError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => MetalError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => MetalError::LockError(LockError::WouldBlock),
        }
    }
}

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
    #[error("{0:?}")]
    LockError(LockError),
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

type AllocatedBuffers = Arc<RwLock<HashMap<(NSUInteger, MTLResourceOptions), Vec<Arc<Buffer>>>>>;

#[derive(Clone)]
pub struct MetalDevice {
    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    device: metal::Device,

    /// Single command queue for the entire device.
    command_queue: metal::CommandQueue,
    /// One command buffer at a time.
    /// The scheduler works by allowing multiple
    /// [ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// on a single command buffer. Using a single command buffer would be fastest on the GPU but
    /// prevents overlapping of CPU and GPU commands (because command buffer needs to be committed
    /// to start to work).
    /// Despite what the documentation says, command buffers are NOT ordered. They are ordered
    /// for their START time, but there's no guarantee that command buffer1 will finish before
    /// command buffer2 starts (or there are metal bugs there)
    command_buffer: Arc<RwLock<metal::CommandBuffer>>,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    /// Arc, RwLock because of the interior mutability.
    command_buffer_index: Arc<RwLock<usize>>,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    kernels: Arc<candle_metal_kernels::Kernels>,
    /// Simple allocator struct.
    /// The buffers are stored in size buckets since ML tends to use similar shapes over and over.
    /// We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
    /// (could be linked to FFI communication overhead).
    ///
    /// Whenever a buffer has a strong_count==1, we can reuse it, it means it was dropped in the
    /// graph calculation, and only we the allocator kept a reference to it, therefore it's free
    /// to be reused. However, in order for this to work, we need to guarantee the order of
    /// operation, so that this buffer is not being used by another kernel at the same time.
    /// Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
    ///
    /// Whenever we actually allocate a new buffer, we make a full sweep to cleanup unused buffers
    /// (strong_count = 1).
    buffers: AllocatedBuffers,
    /// Seed for random number generation.
    seed: Arc<Mutex<Buffer>>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.device.registry_id())
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    pub fn id(&self) -> NSUInteger {
        self.registry_id()
    }

    pub fn metal_device(&self) -> &metal::Device {
        &self.device
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let mut command_buffer_lock = self.command_buffer.try_write().map_err(MetalError::from)?;
        let mut command_buffer = command_buffer_lock.to_owned();
        let mut index = self
            .command_buffer_index
            .try_write()
            .map_err(MetalError::from)?;
        if *index > self.compute_per_buffer {
            command_buffer.commit();
            command_buffer = self.command_queue.new_command_buffer().to_owned();
            *command_buffer_lock = command_buffer.clone();
            *index = 0;
        }
        *index += 1;
        Ok(command_buffer)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let mut command_buffer = self.command_buffer.try_write().map_err(MetalError::from)?;
        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                panic!("Already committed");
            }
            _ => {}
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        *command_buffer = self.command_queue.new_command_buffer().to_owned();
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLPrivate](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer data cannot be read on the CPU directly.
    ///
    /// [`name`] is only used to keep track of the resource origin in case of bugs
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = (element_count * dtype.size_in_bytes()) as NSUInteger;
        self.allocate_buffer(size, MTLResourceOptions::StorageModePrivate, name)
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer can be read on the CPU but will require manual
    /// synchronization when the CPU memory is modified
    /// Used as a bridge to gather data back from the GPU
    pub fn new_buffer_managed(&self, size: NSUInteger) -> Result<Arc<Buffer>> {
        self.allocate_buffer(size, MTLResourceOptions::StorageModeManaged, "managed")
    }

    /// Creates a new buffer from data.
    /// The buffer is [MTLPrivate](https://developer.apple.com/documentation/metal/mtlstoragemode)
    ///
    /// This method will block the computation because of the
    /// lack of lifetime management through the GPU.
    /// Internal comment for technical details.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        let tmp = self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            size,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let real = self.allocate_buffer(
            size,
            metal::MTLResourceOptions::StorageModePrivate,
            "with_data",
        )?;
        let command_buffer = self.command_buffer()?;
        command_buffer.set_label("with_data");
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("with_data_blit");
        blit.copy_from_buffer(&tmp, 0, &real, 0, tmp.length());
        blit.end_encoding();

        // This is necessary, for mmaped safetensors
        // Because of the unsafe slice cast we're doing.
        // The slice might not live long enough for metal
        // To actually fill the GPU buffer.
        // Putting this wait forces the GPU buffer to be filled
        // with the actual data allowing the CPU storage to do
        // deallocate properly.
        self.wait_until_completed()?;
        Ok(real)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(
            size_in_bytes as NSUInteger,
            MTLResourceOptions::StorageModePrivate,
            "allocate_zeros",
        )?;
        let command_buffer = self.command_buffer()?;
        command_buffer.set_label("zeros");
        let blit = command_buffer.new_blit_command_encoder();
        blit.fill_buffer(
            &buffer,
            metal::NSRange {
                location: 0,
                length: buffer.length(),
            },
            0,
        );
        blit.end_encoding();
        Ok(buffer)
    }

    /// The critical allocator algorithm
    fn allocate_buffer(
        &self,
        size: NSUInteger,
        option: MTLResourceOptions,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.try_write().map_err(MetalError::from)?;
        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);

        for sub in &mut *subbuffers {
            if Arc::strong_count(sub) == 1 {
                return Ok(sub.clone());
            }
        }
        let new_buffer = self.device.new_buffer(size as NSUInteger, option);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(new_buffer)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(self);
        descriptor.set_output_url(path);

        capture
            .start_capture(&descriptor)
            .map_err(MetalError::from)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    /// The actual buffer containing the data.
    buffer: Arc<metal::Buffer>,
    /// a reference to the device owning this buffer
    device: MetalDevice,
    /// The dtype is kept since buffers are untyped.
    dtype: DType,
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype, "affine")?;
        let command_buffer = self.device.command_buffer()?;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "affine_f32",
                DType::F16 => "affine_f16",
                DType::BF16 => "affine_bf16",
                dtype => crate::bail!("Metal contiguous affine {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_affine(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "affine_f32_strided",
                DType::F16 => "affine_f16_strided",
                DType::BF16 => "affine_bf16_strided",
                dtype => crate::bail!("Metal strided affine {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_affine_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn powf(&self, layout: &Layout, pow: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype, "powf")?;
        let command_buffer = self.device.command_buffer()?;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "powf_f32",
                DType::F16 => "powf_f16",
                dtype => crate::bail!("Metal contiguous powf {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_powf(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                pow as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "powf_f32_strided",
                DType::F16 => "powf_f16_strided",
                dtype => crate::bail!("Metal strided powf {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_powf_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &buffer,
                pow as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype, "elu")?;
        let command_buffer = self.device.command_buffer()?;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "elu_f32",
                DType::F16 => "elu_f16",
                dtype => crate::bail!("Metal contiguous elu {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_elu(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                alpha as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "elu_f32_strided",
                DType::F16 => "elu_f16_strided",
                dtype => crate::bail!("Metal strided elu {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_elu_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &buffer,
                alpha as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device.clone();
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !sum_dims.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in sum_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }

        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let (name, check_empty, return_index) = match (op, self.dtype) {
            (ReduceOp::Sum, DType::F32) => ("fast_sum_f32_strided", false, false),
            (ReduceOp::Min, DType::F32) => ("fast_min_f32_strided", true, false),
            (ReduceOp::Max, DType::F32) => ("fast_max_f32_strided", true, false),
            (ReduceOp::ArgMin, DType::F32) => ("fast_argmin_f32_strided", true, true),
            (ReduceOp::ArgMax, DType::F32) => ("fast_argmax_f32_strided", true, true),
            (ReduceOp::Sum, DType::U32) => ("fast_sum_u32_strided", false, false),
            (ReduceOp::Min, DType::U32) => ("fast_min_u32_strided", true, false),
            (ReduceOp::Max, DType::U32) => ("fast_max_u32_strided", true, false),
            (ReduceOp::ArgMin, DType::U32) => ("fast_argmin_u32_strided", true, true),
            (ReduceOp::ArgMax, DType::U32) => ("fast_argmax_u32_strided", true, true),
            (ReduceOp::Sum, DType::F16) => ("fast_sum_f16_strided", false, false),
            (ReduceOp::Min, DType::F16) => ("fast_min_f16_strided", true, false),
            (ReduceOp::Max, DType::F16) => ("fast_max_f16_strided", true, false),
            (ReduceOp::ArgMin, DType::F16) => ("fast_argmin_f16_strided", true, true),
            (ReduceOp::ArgMax, DType::F16) => ("fast_argmax_f16_strided", true, true),
            (ReduceOp::Sum, DType::BF16) => ("fast_sum_bf16_strided", false, false),
            (ReduceOp::Min, DType::BF16) => ("fast_min_bf16_strided", true, false),
            (ReduceOp::Max, DType::BF16) => ("fast_max_bf16_strided", true, false),
            (ReduceOp::ArgMin, DType::BF16) => ("fast_argmin_bf16_strided", true, true),
            (ReduceOp::ArgMax, DType::BF16) => ("fast_argmax_bf16_strided", true, true),
            (ReduceOp::Sum, DType::I64) => ("fast_sum_i64_strided", false, false),
            (ReduceOp::Min, DType::I64) => ("fast_min_i64_strided", true, false),
            (ReduceOp::Max, DType::I64) => ("fast_max_i64_strided", true, false),
            (ReduceOp::ArgMin, DType::I64) => ("fast_argmin_i64_strided", true, true),
            (ReduceOp::ArgMax, DType::I64) => ("fast_argmax_i64_strided", true, true),
            (ReduceOp::Sum, DType::U8) => ("fast_sum_u8_strided", false, false),
            (ReduceOp::Min, DType::U8) => ("fast_min_u8_strided", true, false),
            (ReduceOp::Max, DType::U8) => ("fast_max_u8_strided", true, false),
            (ReduceOp::ArgMin, DType::U8) => ("fast_argmin_u8_strided", true, true),
            (ReduceOp::ArgMax, DType::U8) => ("fast_argmax_u8_strided", true, true),
            (k, dtype) => crate::bail!("Metal reduce op {k:?} {dtype:?} not implemented"),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dtype = if return_index { DType::U32 } else { self.dtype };
        let buffer = device.new_buffer(dst_el, dtype, "reduce")?;
        let command_buffer = self.device.command_buffer()?;
        candle_metal_kernels::call_reduce_strided(
            &device.device,
            &command_buffer,
            &device.kernels,
            name,
            &dims,
            &stride,
            dst_el,
            &self.buffer,
            layout.start_offset() * self.dtype.size_in_bytes(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::new(buffer, device, dtype))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let name = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Le => "le",
            CmpOp::Ge => "ge",
            CmpOp::Lt => "lt",
            CmpOp::Gt => "gt",
        };
        self.binary(name, rhs, lhs_l, rhs_l)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let device = self.device();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, "todtype")?;
        let command_buffer = device.command_buffer()?;
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32",
                (DType::U32, DType::U8) => "cast_u32_u8",
                (DType::U32, DType::I64) => "cast_u32_i64",
                (DType::U32, DType::BF16) => "cast_u32_bf16",

                (DType::U8, DType::U32) => "cast_u8_u32",
                (DType::U8, DType::F32) => "cast_u8_f32",
                (DType::U8, DType::I64) => "cast_u8_i64",
                (DType::U8, DType::BF16) => "cast_u8_bf16",

                (DType::F32, DType::F16) => "cast_f32_f16",
                (DType::F32, DType::BF16) => "cast_f32_bf16",

                (DType::I64, DType::F32) => "cast_i64_f32",

                (DType::F16, DType::BF16) => "cast_f16_bf16",
                (DType::F16, DType::F32) => "cast_f16_f32",

                (DType::BF16, DType::U8) => "cast_bf16_u8",
                (DType::BF16, DType::U32) => "cast_bf16_u32",
                (DType::BF16, DType::F16) => "cast_bf16_f16",
                (DType::BF16, DType::F32) => "cast_bf16_f32",

                (left, right) => {
                    crate::bail!("Metal contiguous to_dtype {left:?} {right:?} not implemented")
                }
            };
            candle_metal_kernels::call_cast_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                layout.start_offset() * self.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32_strided",
                (DType::U32, DType::U8) => "cast_u32_u8_strided",
                (DType::U32, DType::I64) => "cast_u32_i64_strided",
                (DType::U8, DType::U32) => "cast_u8_u32_strided",
                (DType::U8, DType::F32) => "cast_u8_f32_strided",
                (DType::U8, DType::I64) => "cast_u8_i64_strided",
                (DType::F32, DType::F16) => "cast_f32_f16_strided",
                (DType::F16, DType::F32) => "cast_f16_f32_strided",
                (DType::I64, DType::F32) => "cast_i64_f32_strided",
                (DType::F32, DType::BF16) => "cast_f32_bf16_strided",
                (DType::BF16, DType::F32) => "cast_bf16_f32_strided",
                (left, right) => {
                    crate::bail!("Metal strided to_dtype {left:?} {right:?} not implemented")
                }
            };
            candle_metal_kernels::call_cast_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
        }
        command_buffer.set_label("to_dtype");
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, B::KERNEL)?;
        let command_buffer = device.command_buffer()?;
        command_buffer.set_label(B::KERNEL);
        if layout.is_contiguous() && layout.start_offset() == 0 {
            use candle_metal_kernels::unary::contiguous;

            let kernel_name = match (B::KERNEL, dtype) {
                ("ucos", DType::F32) => contiguous::cos::FLOAT,
                ("usin", DType::F32) => contiguous::sin::FLOAT,
                ("usqr", DType::F32) => contiguous::sqr::FLOAT,
                ("usqrt", DType::F32) => contiguous::sqrt::FLOAT,
                ("uneg", DType::F32) => contiguous::neg::FLOAT,
                ("uexp", DType::F32) => contiguous::exp::FLOAT,
                ("ulog", DType::F32) => contiguous::log::FLOAT,
                ("ugelu", DType::F32) => contiguous::gelu::FLOAT,
                ("ugelu_erf", DType::F32) => contiguous::gelu_erf::FLOAT,
                ("uerf", DType::F32) => contiguous::erf::FLOAT,
                ("usilu", DType::F32) => contiguous::silu::FLOAT,
                ("uabs", DType::F32) => contiguous::abs::FLOAT,
                ("uceil", DType::F32) => contiguous::ceil::FLOAT,
                ("ufloor", DType::F32) => contiguous::floor::FLOAT,
                ("uround", DType::F32) => contiguous::round::FLOAT,
                ("urecip", DType::F32) => contiguous::recip::FLOAT,
                ("utanh", DType::F32) => contiguous::tanh::FLOAT,
                ("urelu", DType::F32) => contiguous::relu::FLOAT,
                ("ucos", DType::F16) => contiguous::cos::HALF,
                ("usin", DType::F16) => contiguous::sin::HALF,
                ("usqr", DType::F16) => contiguous::sqr::HALF,
                ("usqrt", DType::F16) => contiguous::sqrt::HALF,
                ("uneg", DType::F16) => contiguous::neg::HALF,
                ("uexp", DType::F16) => contiguous::exp::HALF,
                ("ulog", DType::F16) => contiguous::log::HALF,
                ("ugelu", DType::F16) => contiguous::gelu::HALF,
                ("ugelu_erf", DType::F16) => contiguous::gelu_erf::HALF,
                ("uerf", DType::F16) => contiguous::erf::HALF,
                ("usilu", DType::F16) => contiguous::silu::HALF,
                ("uabs", DType::F16) => contiguous::abs::HALF,
                ("uceil", DType::F16) => contiguous::ceil::HALF,
                ("ufloor", DType::F16) => contiguous::floor::HALF,
                ("uround", DType::F16) => contiguous::round::HALF,
                ("urecip", DType::F16) => contiguous::recip::HALF,
                ("utanh", DType::F16) => contiguous::tanh::HALF,
                ("urelu", DType::F16) => contiguous::relu::HALF,
                (name, dtype) => {
                    crate::bail!("Metal contiguous unary {name} {dtype:?} not implemented")
                }
            };
            candle_metal_kernels::call_unary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::unary::strided;
            let kernel_name = match (B::KERNEL, dtype) {
                ("ucos", DType::F32) => strided::cos::FLOAT,
                ("usin", DType::F32) => strided::sin::FLOAT,
                ("usqr", DType::F32) => strided::sqr::FLOAT,
                ("usqrt", DType::F32) => strided::sqrt::FLOAT,
                ("uneg", DType::F32) => strided::neg::FLOAT,
                ("uexp", DType::F32) => strided::exp::FLOAT,
                ("ulog", DType::F32) => strided::log::FLOAT,
                ("ugelu", DType::F32) => strided::gelu::FLOAT,
                ("ugelu_erf", DType::F32) => strided::gelu_erf::FLOAT,
                ("uerf", DType::F32) => strided::erf::FLOAT,
                ("usilu", DType::F32) => strided::silu::FLOAT,
                ("uabs", DType::F32) => strided::abs::FLOAT,
                ("uceil", DType::F32) => strided::ceil::FLOAT,
                ("ufloor", DType::F32) => strided::floor::FLOAT,
                ("urelu", DType::F32) => strided::relu::FLOAT,
                ("uround", DType::F32) => strided::round::FLOAT,
                ("ucos", DType::F16) => strided::cos::HALF,
                ("usin", DType::F16) => strided::sin::HALF,
                ("usqr", DType::F16) => strided::sqr::HALF,
                ("usqrt", DType::F16) => strided::sqrt::HALF,
                ("uneg", DType::F16) => strided::neg::HALF,
                ("uexp", DType::F16) => strided::exp::HALF,
                ("ulog", DType::F16) => strided::log::HALF,
                ("ugelu", DType::F16) => strided::gelu::HALF,
                ("ugelu_erf", DType::F16) => strided::gelu_erf::HALF,
                ("uerf", DType::F16) => strided::erf::HALF,
                ("usilu", DType::F16) => strided::silu::HALF,
                ("uabs", DType::F16) => strided::abs::HALF,
                ("uceil", DType::F16) => strided::ceil::HALF,
                ("ufloor", DType::F16) => strided::floor::HALF,
                ("urelu", DType::F16) => strided::relu::HALF,
                ("uround", DType::F16) => strided::round::HALF,
                (name, dtype) => {
                    crate::bail!("Metal strided unary {name} {dtype:?} not implemented")
                }
            };
            candle_metal_kernels::call_unary_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
                &buffer,
                0,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.binary(B::KERNEL, rhs, lhs_l, rhs_l)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device.clone();
        let shape = t_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dtype = t.dtype;
        let buffer = self.device.new_buffer(el, dtype, "where")?;
        let command_buffer = self.device.command_buffer()?;
        if t.dtype() != f.dtype() {
            crate::bail!(
                "Invalid where: different dtypes for values {:?} != {:?}",
                t.dtype(),
                f.dtype()
            );
        }
        let name = match (self.dtype, t.dtype()) {
            (DType::U8, DType::F32) => "where_u8_f32",
            (DType::U8, DType::BF16) => "where_u8_bf16",
            (DType::U8, DType::F16) => "where_u8_f16",
            (DType::U8, DType::I64) => "where_u8_i64",
            (DType::U8, DType::U32) => "where_u8_u32",
            (DType::U8, DType::U8) => "where_u8_u8",
            (left, right) => crate::bail!("Metal where_cond {left:?} {right:?} not implemented"),
        };
        candle_metal_kernels::call_where_cond_strided(
            &device.device,
            &command_buffer,
            &device.kernels,
            name,
            dims,
            &self.buffer,
            (
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
            ),
            &t.buffer,
            (t_l.stride(), t_l.start_offset() * t.dtype.size_in_bytes()),
            &f.buffer,
            (f_l.stride(), f_l.start_offset() * f.dtype.size_in_bytes()),
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device, dtype))
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let shape = layout.shape();
        let dims = shape.dims();
        let strides = layout.stride();

        let stride = params.stride;
        let dilation = params.dilation;
        let padding = params.padding;
        let k_size = params.k_size;
        let l_out = (dims[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
        let dst_el = dims[0] * l_out * dims[1] * k_size;
        let dst = self
            .device
            .new_buffer(dst_el, self.dtype, "conv1d_im2col")?;
        let command_buffer = self.device.command_buffer()?;
        let name = match self.dtype {
            DType::F32 => "im2col1d_f32",
            dtype => crate::bail!("Metal conv1d {dtype:?} not implemented"),
        };
        candle_metal_kernels::call_im2col1d_strided(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            layout.shape().dims(),
            strides,
            (k_size, stride, padding, dilation),
            &self.buffer,
            layout.start_offset() * self.dtype.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        let col = Self {
            buffer: dst,
            device,
            dtype: self.dtype,
        };
        let l_out = params.l_out();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), res.dtype())?;
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("Metal conv_transpose1d not implemented")
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let shape = layout.shape();
        let dims = shape.dims();

        let stride = params.stride;
        let dilation = params.dilation;
        let padding = params.padding;
        let h_k = params.k_h;
        let w_k = params.k_w;
        let h = dims[2];
        let w = dims[3];
        let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
        let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;
        let dst_el = dims[0] * h_out * w_out * dims[1] * h_k * w_k;

        let dst = self
            .device
            .new_buffer(dst_el, self.dtype, "conv2d_im2col")?;
        let command_buffer = self.device.command_buffer()?;
        let name = match self.dtype {
            DType::F32 => "im2col_f32",
            dtype => crate::bail!("Metal conv2d {dtype:?} not implemented"),
        };
        candle_metal_kernels::call_im2col_strided(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            layout.shape().dims(),
            layout.stride(),
            (h_k, w_k, stride, padding, dilation),
            &self.buffer,
            layout.start_offset() * self.dtype.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        let col = Self {
            buffer: dst,
            device,
            dtype: self.dtype,
        };
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b, m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
                .transpose(1, 2)?
                .broadcast_as((b, k, n))?;
            col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        let mut res_t = self.device().zeros_impl(res_l.shape(), res.dtype())?;
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("Metal conv_tranpose2d not implemented")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("Metal avg_pool2d not implemented")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("Metal max_pool2d not implemented")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("Metal upsample_nearest1d not implemented")
    }

    fn upsample_nearest2d(&self, inp_l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        // let inp = &inp.slice(inp_l.start_offset()..);
        let shape = inp_l.shape();
        let dims = shape.dims();
        let strides = inp_l.stride();
        if dims.len() != 4 {
            crate::bail!("unexpected input shape for upsample {dims:?}")
        }
        let name = match self.dtype {
            DType::F32 => "upsample_nearest2d_f32",
            dtype => crate::bail!("Metal upsample_nearest2d {dtype:?} not implemented"),
        };

        let dst_el = out_w * out_h * dims[0] * dims[1];
        let buffer = self
            .device
            .new_buffer(dst_el, self.dtype, "upsample_nearest2d")?;
        let command_buffer = self.device.command_buffer()?;
        candle_metal_kernels::call_upsample_nearest_2d(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            dims,
            strides,
            out_w,
            out_h,
            &self.buffer,
            inp_l.start_offset() * self.dtype.size_in_bytes(),
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, self.device.clone(), self.dtype))
    }

    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let ids_el = ids_l.dims()[dim];
        let dst_el = ids_l.shape().elem_count();
        let dtype = self.dtype;
        let device = self.device();
        let buffer = device.new_buffer(dst_el, dtype, "index_select")?;
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "gather_u32_f32",
            (DType::U32, DType::F16) => "gather_u32_f16",
            (left, right) => crate::bail!("Metal gather {left:?} {right:?} not implemented"),
        };
        let command_buffer = self.device.command_buffer()?;
        candle_metal_kernels::call_gather(
            &device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            src_l.dims(),
            ids_el,
            dim,
            &self.buffer,
            src_l.start_offset() * dtype.size_in_bytes(),
            &ids.buffer,
            ids_o1 * ids.dtype.size_in_bytes(),
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn scatter_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let mut acc = self.device.zeros_impl(l.shape(), self.dtype())?;
        self.copy_strided_src(&mut acc, 0, l)?;
        let (ids_offset, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let src_offset = match src_l.contiguous_offsets() {
            Some((o1, _)) => o1,
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "sa_u32_f32",
            _ => Err(MetalError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let command_buffer = self.device.command_buffer()?;
        candle_metal_kernels::call_scatter_add(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            src_l.dims(),
            l.dims(),
            dim,
            &src.buffer,
            src_offset * src.dtype.size_in_bytes(),
            &ids.buffer,
            ids_offset * ids.dtype.size_in_bytes(),
            &acc.buffer,
        )
        .map_err(MetalError::from)?;
        Ok(acc)
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if !(src_l.is_contiguous()
            && src_l.start_offset() == 0
            && ids_l.is_contiguous()
            && ids_l.start_offset() == 0)
        {
            crate::bail!("Metal strided index_select not implemented");
        }
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;
        let dtype = self.dtype;
        let device = self.device();
        let buffer = device.new_buffer(dst_el, dtype, "index_select")?;
        let name = match (ids.dtype, self.dtype) {
            (DType::U8, DType::BF16) => "is_u8_bf16",

            (DType::U32, DType::F32) => "is_u32_f32",
            (DType::U32, DType::F16) => "is_u32_f16",
            (DType::U32, DType::BF16) => "is_u32_bf16",

            (left, right) => {
                crate::bail!("Metal contiguous index_select {left:?} {right:?} not implemented")
            }
        };
        let command_buffer = self.device.command_buffer()?;
        candle_metal_kernels::call_index_select(
            &device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            src_l.dims(),
            ids_el,
            dim,
            &self.buffer,
            &ids.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let mut acc = self.device.zeros_impl(l.shape(), self.dtype())?;
        self.copy_strided_src(&mut acc, 0, l)?;
        let (ids_offset, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let src_offset = match src_l.contiguous_offsets() {
            Some((o1, _)) => o1,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "ia_u32_f32",
            _ => Err(MetalError::UnexpectedDType {
                msg: "index-add ids should be u32",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };
        let command_buffer = self.device.command_buffer()?;
        candle_metal_kernels::call_index_add(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            src_l.dims(),
            l.dims(),
            ids_l.dims(),
            dim,
            &src.buffer,
            src_offset * src.dtype.size_in_bytes(),
            &ids.buffer,
            ids_offset * ids.dtype.size_in_bytes(),
            &acc.buffer,
        )
        .map_err(MetalError::from)?;
        Ok(acc)
    }
    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let buffer = self.device.new_buffer(b * m * n, self.dtype, "matmul")?;
        let name = match self.dtype {
            DType::F32 => "sgemm",
            DType::F16 => "hgemm",
            dtype => {
                return Err(MetalError::Message(format!("matmul doesn't support {dtype:?}")).into())
            }
        };

        let command_buffer = self.device.command_buffer()?;
        command_buffer.set_label("matmul");
        candle_metal_kernels::call_gemm(
            &self.device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            (b, m, n, k),
            lhs_l.stride(),
            lhs_l.start_offset() * self.dtype.size_in_bytes(),
            &self.buffer,
            rhs_l.stride(),
            rhs_l.start_offset() * rhs.dtype.size_in_bytes(),
            &rhs.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, self.device.clone(), self.dtype()))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let command_buffer = self.device.command_buffer()?;
        if src_l.is_contiguous() && self.dtype == dst.dtype() {
            command_buffer.set_label("copy_contiguous");
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("copy_contiguous");
            let src_offset = (src_l.start_offset() * self.dtype.size_in_bytes()) as NSUInteger;
            let length = (src_l.shape().elem_count() * self.dtype.size_in_bytes()) as NSUInteger;
            let dst_offset = (dst_offset * dst.dtype().size_in_bytes()) as NSUInteger;
            blit.copy_from_buffer(&self.buffer, src_offset, dst.buffer(), dst_offset, length);
            blit.end_encoding();
        } else {
            let src_shape = src_l.shape();
            let el_count = src_shape.elem_count();
            if el_count == 0 {
                return Ok(());
            }
            let kernel_name = match self.dtype {
                DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
                DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
                DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
                DType::I64 => candle_metal_kernels::unary::strided::copy::I64,
                DType::U32 => candle_metal_kernels::unary::strided::copy::U32,
                DType::U8 => candle_metal_kernels::unary::strided::copy::U8,
                dtype => crate::bail!("Metal copy_strided {dtype:?} not implemented"),
            };
            candle_metal_kernels::call_unary_strided(
                &self.device.device,
                &command_buffer,
                &self.device.kernels,
                kernel_name,
                src_l.dims(),
                &self.buffer,
                src_l.stride(),
                src_l.start_offset() * self.dtype.size_in_bytes(),
                &dst.buffer,
                dst_offset * dst.dtype.size_in_bytes(),
            )
            .map_err(MetalError::from)?;
            command_buffer.set_label("copy_strided");
        }
        Ok(())
    }
}

impl MetalStorage {
    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn binary(
        &self,
        op: &'static str,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device();
        let shape = lhs_l.shape();
        let el_count = shape.elem_count();
        let command_buffer = device.command_buffer()?;
        let (buffer, dtype) = if (lhs_l.is_contiguous() && lhs_l.start_offset() == 0)
            && (rhs_l.is_contiguous() && rhs_l.start_offset() == 0)
            && &op[..1] != "b"
        {
            use candle_metal_kernels::binary::contiguous;

            let (kernel_name, dtype) = match (op, self.dtype) {
                ("add", DType::F32) => (contiguous::add::FLOAT, self.dtype),
                ("sub", DType::F32) => (contiguous::sub::FLOAT, self.dtype),
                ("mul", DType::F32) => (contiguous::mul::FLOAT, self.dtype),
                ("div", DType::F32) => (contiguous::div::FLOAT, self.dtype),
                ("eq", DType::F32) => (contiguous::eq::FLOAT, DType::U8),
                ("ne", DType::F32) => (contiguous::ne::FLOAT, DType::U8),
                ("le", DType::F32) => (contiguous::le::FLOAT, DType::U8),
                ("lt", DType::F32) => (contiguous::lt::FLOAT, DType::U8),
                ("ge", DType::F32) => (contiguous::ge::FLOAT, DType::U8),
                ("gt", DType::F32) => (contiguous::gt::FLOAT, DType::U8),

                ("add", DType::F16) => (contiguous::add::HALF, self.dtype),
                ("sub", DType::F16) => (contiguous::sub::HALF, self.dtype),
                ("mul", DType::F16) => (contiguous::mul::HALF, self.dtype),
                ("div", DType::F16) => (contiguous::div::HALF, self.dtype),
                ("eq", DType::F16) => (contiguous::eq::HALF, DType::U8),
                ("ne", DType::F16) => (contiguous::ne::HALF, DType::U8),
                ("le", DType::F16) => (contiguous::le::HALF, DType::U8),
                ("lt", DType::F16) => (contiguous::lt::HALF, DType::U8),
                ("ge", DType::F16) => (contiguous::ge::HALF, DType::U8),
                ("gt", DType::F16) => (contiguous::gt::HALF, DType::U8),

                ("add", DType::BF16) => (contiguous::add::BFLOAT, self.dtype),
                ("sub", DType::BF16) => (contiguous::sub::BFLOAT, self.dtype),
                ("mul", DType::BF16) => (contiguous::mul::BFLOAT, self.dtype),
                ("div", DType::BF16) => (contiguous::div::BFLOAT, self.dtype),
                ("eq", DType::BF16) => (contiguous::eq::BFLOAT, DType::U8),
                ("ne", DType::BF16) => (contiguous::ne::BFLOAT, DType::U8),
                ("le", DType::BF16) => (contiguous::le::BFLOAT, DType::U8),
                ("lt", DType::BF16) => (contiguous::lt::BFLOAT, DType::U8),
                ("ge", DType::BF16) => (contiguous::ge::BFLOAT, DType::U8),
                ("gt", DType::BF16) => (contiguous::gt::BFLOAT, DType::U8),

                ("add", DType::I64) => (contiguous::add::I64, self.dtype),
                ("sub", DType::I64) => (contiguous::sub::I64, self.dtype),
                ("mul", DType::I64) => (contiguous::mul::I64, self.dtype),
                ("div", DType::I64) => (contiguous::div::I64, self.dtype),
                ("eq", DType::I64) => (contiguous::eq::I64, DType::U8),
                ("ne", DType::I64) => (contiguous::ne::I64, DType::U8),
                ("le", DType::I64) => (contiguous::le::I64, DType::U8),
                ("lt", DType::I64) => (contiguous::lt::I64, DType::U8),
                ("ge", DType::I64) => (contiguous::ge::I64, DType::U8),
                ("gt", DType::I64) => (contiguous::gt::I64, DType::U8),

                ("add", DType::U32) => (contiguous::add::U32, self.dtype),
                ("sub", DType::U32) => (contiguous::sub::U32, self.dtype),
                ("mul", DType::U32) => (contiguous::mul::U32, self.dtype),
                ("div", DType::U32) => (contiguous::div::U32, self.dtype),
                ("eq", DType::U32) => (contiguous::eq::U32, DType::U8),
                ("ne", DType::U32) => (contiguous::ne::U32, DType::U8),
                ("le", DType::U32) => (contiguous::le::U32, DType::U8),
                ("lt", DType::U32) => (contiguous::lt::U32, DType::U8),
                ("ge", DType::U32) => (contiguous::ge::U32, DType::U8),
                ("gt", DType::U32) => (contiguous::gt::U32, DType::U8),

                ("add", DType::U8) => (contiguous::add::U8, self.dtype),
                ("sub", DType::U8) => (contiguous::sub::U8, self.dtype),
                ("mul", DType::U8) => (contiguous::mul::U8, self.dtype),
                ("div", DType::U8) => (contiguous::div::U8, self.dtype),
                ("eq", DType::U8) => (contiguous::eq::U8, DType::U8),
                ("ne", DType::U8) => (contiguous::ne::U8, DType::U8),
                ("le", DType::U8) => (contiguous::le::U8, DType::U8),
                ("lt", DType::U8) => (contiguous::lt::U8, DType::U8),
                ("ge", DType::U8) => (contiguous::ge::U8, DType::U8),
                ("gt", DType::U8) => (contiguous::gt::U8, DType::U8),

                (name, dtype) => {
                    crate::bail!("Metal contiguous binary {name} {dtype:?} not implemented")
                }
            };
            let buffer = device.new_buffer(el_count, dtype, op)?;
            candle_metal_kernels::call_binary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &rhs.buffer,
                &buffer,
            )
            .map_err(MetalError::from)?;
            (buffer, dtype)
        } else {
            use candle_metal_kernels::binary::strided;

            let (kernel_name, dtype) = match (op, self.dtype) {
                ("badd", DType::F32) => (strided::add::FLOAT, self.dtype),
                ("bsub", DType::F32) => (strided::sub::FLOAT, self.dtype),
                ("bmul", DType::F32) => (strided::mul::FLOAT, self.dtype),
                ("bdiv", DType::F32) => (strided::div::FLOAT, self.dtype),
                ("bminimum", DType::F32) => (strided::min::FLOAT, self.dtype),
                ("bmaximum", DType::F32) => (strided::max::FLOAT, self.dtype),
                ("eq", DType::F32) => (strided::eq::FLOAT, DType::U8),
                ("ne", DType::F32) => (strided::ne::FLOAT, DType::U8),
                ("le", DType::F32) => (strided::le::FLOAT, DType::U8),
                ("lt", DType::F32) => (strided::lt::FLOAT, DType::U8),
                ("ge", DType::F32) => (strided::ge::FLOAT, DType::U8),
                ("gt", DType::F32) => (strided::gt::FLOAT, DType::U8),

                ("badd", DType::F16) => (strided::add::HALF, self.dtype),
                ("bsub", DType::F16) => (strided::sub::HALF, self.dtype),
                ("bmul", DType::F16) => (strided::mul::HALF, self.dtype),
                ("bdiv", DType::F16) => (strided::div::HALF, self.dtype),
                ("bminimum", DType::F16) => (strided::min::HALF, self.dtype),
                ("bmaximum", DType::F16) => (strided::max::HALF, self.dtype),
                ("eq", DType::F16) => (strided::eq::HALF, DType::U8),
                ("ne", DType::F16) => (strided::ne::HALF, DType::U8),
                ("le", DType::F16) => (strided::le::HALF, DType::U8),
                ("lt", DType::F16) => (strided::lt::HALF, DType::U8),
                ("ge", DType::F16) => (strided::ge::HALF, DType::U8),
                ("gt", DType::F16) => (strided::gt::HALF, DType::U8),

                ("badd", DType::BF16) => (strided::add::BFLOAT, self.dtype),
                ("bsub", DType::BF16) => (strided::sub::BFLOAT, self.dtype),
                ("bmul", DType::BF16) => (strided::mul::BFLOAT, self.dtype),
                ("bdiv", DType::BF16) => (strided::div::BFLOAT, self.dtype),
                ("bminimum", DType::BF16) => (strided::min::BFLOAT, self.dtype),
                ("bmaximum", DType::BF16) => (strided::max::BFLOAT, self.dtype),
                ("eq", DType::BF16) => (strided::eq::BFLOAT, DType::U8),
                ("ne", DType::BF16) => (strided::ne::BFLOAT, DType::U8),
                ("le", DType::BF16) => (strided::le::BFLOAT, DType::U8),
                ("lt", DType::BF16) => (strided::lt::BFLOAT, DType::U8),
                ("ge", DType::BF16) => (strided::ge::BFLOAT, DType::U8),
                ("gt", DType::BF16) => (strided::gt::BFLOAT, DType::U8),

                ("badd", DType::I64) => (strided::add::I64, self.dtype),
                ("bsub", DType::I64) => (strided::sub::I64, self.dtype),
                ("bmul", DType::I64) => (strided::mul::I64, self.dtype),
                ("bdiv", DType::I64) => (strided::div::I64, self.dtype),
                ("bminimum", DType::I64) => (strided::min::I64, self.dtype),
                ("bmaximum", DType::I64) => (strided::max::I64, self.dtype),
                ("eq", DType::I64) => (strided::eq::I64, DType::U8),
                ("ne", DType::I64) => (strided::ne::I64, DType::U8),
                ("le", DType::I64) => (strided::le::I64, DType::U8),
                ("lt", DType::I64) => (strided::lt::I64, DType::U8),
                ("ge", DType::I64) => (strided::ge::I64, DType::U8),
                ("gt", DType::I64) => (strided::gt::I64, DType::U8),

                ("badd", DType::U32) => (strided::add::U32, self.dtype),
                ("bsub", DType::U32) => (strided::sub::U32, self.dtype),
                ("bmul", DType::U32) => (strided::mul::U32, self.dtype),
                ("bdiv", DType::U32) => (strided::div::U32, self.dtype),
                ("bminimum", DType::U32) => (strided::min::U32, self.dtype),
                ("bmaximum", DType::U32) => (strided::max::U32, self.dtype),
                ("eq", DType::U32) => (strided::eq::U32, DType::U8),
                ("ne", DType::U32) => (strided::ne::U32, DType::U8),
                ("le", DType::U32) => (strided::le::U32, DType::U8),
                ("lt", DType::U32) => (strided::lt::U32, DType::U8),
                ("ge", DType::U32) => (strided::ge::U32, DType::U8),
                ("gt", DType::U32) => (strided::gt::U32, DType::U8),

                ("badd", DType::U8) => (strided::add::U8, self.dtype),
                ("bsub", DType::U8) => (strided::sub::U8, self.dtype),
                ("bmul", DType::U8) => (strided::mul::U8, self.dtype),
                ("bdiv", DType::U8) => (strided::div::U8, self.dtype),
                ("bminimum", DType::U8) => (strided::min::U8, self.dtype),
                ("bmaximum", DType::U8) => (strided::max::U8, self.dtype),
                ("eq", DType::U8) => (strided::eq::U8, DType::U8),
                ("ne", DType::U8) => (strided::ne::U8, DType::U8),
                ("le", DType::U8) => (strided::le::U8, DType::U8),
                ("lt", DType::U8) => (strided::lt::U8, DType::U8),
                ("ge", DType::U8) => (strided::ge::U8, DType::U8),
                ("gt", DType::U8) => (strided::gt::U8, DType::U8),

                (name, dtype) => {
                    crate::bail!("Metal strided binary {name} {dtype:?} not implemented")
                }
            };
            let buffer = device.new_buffer(el_count, dtype, op)?;
            candle_metal_kernels::call_binary_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                lhs_l.dims(),
                &self.buffer,
                lhs_l.stride(),
                lhs_l.start_offset() * self.dtype.size_in_bytes(),
                &rhs.buffer,
                rhs_l.stride(),
                rhs_l.start_offset() * rhs.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
            (buffer, dtype)
        };
        command_buffer.set_label("binary");
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    pub(crate) fn to_cpu<T: Clone>(&self) -> Result<Vec<T>> {
        let length = self.buffer.length() as usize;
        let size = self.dtype.size_in_bytes();
        if length % size != 0 {
            crate::bail!(
                "The Metal buffer length is not aligned with dtype {:?}",
                self.dtype
            );
        }
        let buffer = self.device.new_buffer_managed(self.buffer.length())?;
        {
            let command_buffer = self.device.command_buffer()?;
            command_buffer.set_label("to_cpu");
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec(&buffer, length / size))
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = metal::Device::all().swap_remove(ordinal);
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer().to_owned();
        command_buffer.enqueue();
        let command_buffer = Arc::new(RwLock::new(command_buffer));
        let command_buffer_index = Arc::new(RwLock::new(0));
        let kernels = Arc::new(Kernels::new());
        let buffers = Arc::new(RwLock::new(HashMap::new()));
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse()?,
            _ => 10,
        };
        let seed = Arc::new(Mutex::new(device.new_buffer_with_data(
            [299792458].as_ptr() as *const c_void,
            4,
            MTLResourceOptions::StorageModeManaged,
        )));
        Ok(Self {
            device,
            command_queue,
            command_buffer,
            command_buffer_index,
            compute_per_buffer,
            buffers,
            kernels,
            seed,
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal {
            gpu_id: self.registry_id() as usize,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.device.registry_id() == rhs.device.registry_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let size = shape.elem_count() * dtype.size_in_bytes();
        let buffer = self.allocate_zeros(size)?;
        Ok(MetalStorage::new(buffer, self.clone(), dtype))
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // TODO Is there a faster way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.ones_impl(shape, dtype)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let buffer = match storage {
            CpuStorage::U8(storage) => self.new_buffer_with_data(storage),
            CpuStorage::U32(storage) => self.new_buffer_with_data(storage),
            CpuStorage::I64(storage) => self.new_buffer_with_data(storage),
            CpuStorage::BF16(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F16(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F32(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F64(storage) => self.new_buffer_with_data(storage),
        }?;
        Ok(Self::Storage::new(buffer, self.clone(), storage.dtype()))
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self::Storage> {
        let name = match dtype {
            DType::F32 => "rand_uniform_f32",
            DType::F16 => "rand_uniform_f16",
            DType::BF16 => "rand_uniform_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self.new_buffer(shape.elem_count(), dtype, "rand_uniform")?;
        let command_buffer = self.command_buffer()?;
        candle_metal_kernels::call_random_uniform(
            &self.device,
            &command_buffer,
            &self.kernels,
            name,
            min as f32,
            max as f32,
            shape.elem_count(),
            &self.seed.lock().unwrap(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::Storage::new(buffer, self.clone(), dtype))
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        let name = match dtype {
            DType::F32 => "rand_normal_f32",
            DType::F16 => "rand_normal_f16",
            DType::BF16 => "rand_normal_bf16",
            dtype => crate::bail!("rand_uniform not implemented for {dtype:?}"),
        };
        let buffer = self.new_buffer(shape.elem_count(), dtype, "rand_normal")?;
        let command_buffer = self.command_buffer()?;
        candle_metal_kernels::call_random_normal(
            &self.device,
            &command_buffer,
            &self.kernels,
            name,
            mean as f32,
            stddev as f32,
            shape.elem_count(),
            &self.seed.lock().unwrap(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::Storage::new(buffer, self.clone(), dtype))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let seed: u32 = seed.try_into().map_err(|_| {
            MetalError::Message("Metal seed must be less than or equal to u32::MAX".to_string())
        })?;

        let seed_buffer = self.seed.try_lock().map_err(MetalError::from)?;
        let contents = seed_buffer.contents();
        unsafe {
            std::ptr::copy([seed].as_ptr(), contents as *mut u32, 4);
        }
        seed_buffer.did_modify_range(metal::NSRange::new(0, 4));

        Ok(())
    }
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

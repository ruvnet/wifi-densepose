use super::{GgmlDType, QStorage};
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{CudaDevice, CudaStorage, Result};

use cudarc::driver::{CudaSlice, DeviceSlice};

pub struct QCudaStorage {
    data: CudaSlice<u8>,
    dtype: GgmlDType,
    device: CudaDevice,
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;

fn dequantize(
    data: &CudaSlice<u8>,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let (kernel_name, is_k, block_dim) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0", false, 32),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1", false, 32),
        GgmlDType::Q5_0 => ("dequantize_block_q5_0", false, 32),
        GgmlDType::Q5_1 => ("dequantize_block_q5_1", false, 32),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0", false, 32),
        GgmlDType::Q2K => ("dequantize_block_q2_K", true, 64),
        GgmlDType::Q3K => ("dequantize_block_q3_K", true, 64),
        GgmlDType::Q4K => ("dequantize_block_q4_K", true, 32),
        GgmlDType::Q5K => ("dequantize_block_q5_K", true, 64),
        GgmlDType::Q6K => ("dequantize_block_q6_K", true, 64),
        GgmlDType::Q8K => ("dequantize_block_q8_K", true, 32),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, candle_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(elem_count).w()?;
    let nb = (elem_count + 255) / 256;
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let params = (data, &dst);
        unsafe { func.launch(cfg, params) }.w()?;
    } else {
        let nb32 = elem_count / 32;
        let params = (data, &dst, nb32 as i32);
        unsafe { func.launch(cfg, params) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mut_mal_vec(
    data: &CudaSlice<u8>,
    y: &cudarc::driver::CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, candle_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows).w()?;
    let block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let params = (data, y, &dst, ncols as i32, nrows as i32);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = el_count * dtype.type_size() / dtype.block_size();
        let data = device.alloc_zeros::<u8>(size_in_bytes).w()?;
        Ok(QCudaStorage {
            data,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.
        use crate::quantized::k_quants::GgmlType;

        let buffer = self.device.dtoh_sync_copy(&self.data).w()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let slice =
                    unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const f32, block_len) };
                out.copy_from_slice(slice)
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out)?;
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out)?;
            }
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => {
                self.device.dtoh_sync_copy(data).w()?
            }
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let data = self.device.htod_sync_copy(data.as_ref()).w()?;
        self.data = data;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if matches!(layout.shape().dims(), [1, 1, _] | [1, _]) {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (with_batch, k) = match rhs_l.shape().dims() {
            [1, 1, k] => (true, k),
            [1, k] => (false, k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != *k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out =
            dequantize_mut_mal_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?;
        let out_shape = if with_batch {
            vec![1, 1, nrows]
        } else {
            vec![1, nrows]
        };
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let data_f32 = self.dequantize(n * k)?;
        let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0);
        let out = storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?;
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

fn read_to_vec<T: Clone>(buffer: &[u8], n: usize) -> Vec<T> {
    let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
    slice.to_vec()
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let data = device.htod_sync_copy(data).w()?;
    Ok(QStorage::Cuda(QCudaStorage {
        data,
        device: device.clone(),
        dtype: T::DTYPE,
    }))
}

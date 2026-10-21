//! Devices on which tensor computations are run.

/// A torch device.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// The main CPU device.
    Cpu,
    /// The main GPU device.
    Cuda(usize),
    /// The main MPS device.
    Mps,
    /// The main Vulkan device.
    Vulkan,
}

/// Cuda related helper functions.
pub enum Cuda {}
impl Cuda {
    /// Returns the number of CUDA devices available.
    pub fn device_count() -> i64 {
        let res = unsafe_torch!(torch_sys::cuda::atc_cuda_device_count());
        i64::from(res)
    }

    /// Returns true if at least one CUDA device is available.
    pub fn is_available() -> bool {
        unsafe_torch!(torch_sys::cuda::atc_cuda_is_available()) != 0
    }

    /// Returns true if CUDA is available, and CuDNN is available.
    pub fn cudnn_is_available() -> bool {
        unsafe_torch!(torch_sys::cuda::atc_cudnn_is_available()) != 0
    }

    /// Sets the seed for the current GPU.
    ///
    /// # Arguments
    ///
    /// * `seed` - An unsigned 64bit int to be used as seed.
    pub fn manual_seed(seed: u64) {
        unsafe_torch!(torch_sys::cuda::atc_manual_seed(seed));
    }

    /// Sets the seed for all available GPUs.
    ///
    /// # Arguments
    ///
    /// * `seed` - An unsigned 64bit int to be used as seed.
    pub fn manual_seed_all(seed: u64) {
        unsafe_torch!(torch_sys::cuda::atc_manual_seed_all(seed));
    }

    /// Waits for all kernels in all streams on a CUDA device to complete.
    ///
    /// # Arguments
    ///
    /// * `device_index` - A signed 64bit int to indice which device to wait for.
    pub fn synchronize(device_index: i64) {
        unsafe_torch!(torch_sys::cuda::atc_synchronize(device_index));
    }

    /// Returns true if cudnn is enabled by the user.
    ///
    /// This does not indicate whether cudnn is actually usable.
    pub fn user_enabled_cudnn() -> bool {
        unsafe_torch!(torch_sys::cuda::atc_user_enabled_cudnn()) != 0
    }

    /// Enable or disable cudnn.
    pub fn set_user_enabled_cudnn(b: bool) {
        unsafe_torch!(torch_sys::cuda::atc_set_user_enabled_cudnn(i32::from(b)))
    }

    /// Sets cudnn benchmark mode.
    ///
    /// When set cudnn will try to optimize the generators durning
    /// the first network runs and then use the optimized architecture
    /// in the following runs. This can result in significant performance
    /// improvements.
    pub fn cudnn_set_benchmark(b: bool) {
        unsafe_torch!(torch_sys::cuda::atc_set_benchmark_cudnn(i32::from(b)))
    }
}

impl Device {
    pub(super) fn c_int(self) -> libc::c_int {
        match self {
            Device::Cpu => -1,
            Device::Cuda(device_index) => device_index as libc::c_int,
            Device::Mps => -2,
            Device::Vulkan => -3,
        }
    }

    pub(super) fn from_c_int(v: libc::c_int) -> Self {
        match v {
            -1 => Device::Cpu,
            -2 => Device::Mps,
            -3 => Device::Vulkan,
            index if index >= 0 => Device::Cuda(index as usize),
            _ => panic!("unexpected device {v}"),
        }
    }

    /// Returns a GPU device if available, else default to CPU.
    pub fn cuda_if_available() -> Device {
        if Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }

    pub fn is_cuda(self) -> bool {
        match self {
            Device::Cuda(_) => true,
            Device::Cpu | Device::Mps | Device::Vulkan => false,
        }
    }
}

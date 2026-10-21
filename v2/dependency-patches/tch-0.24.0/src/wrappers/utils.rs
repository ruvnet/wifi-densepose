use crate::TchError;
use libc::c_char;
use std::io;

// This returns None on the null pointer. If not null, the pointer gets
// freed.
pub(super) unsafe fn ptr_to_string(ptr: *mut c_char) -> Option<String> {
    if !ptr.is_null() {
        let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        libc::free(ptr as *mut libc::c_void);
        Some(str)
    } else {
        None
    }
}

pub(super) fn read_and_clean_error() -> Result<(), TchError> {
    unsafe {
        match ptr_to_string(torch_sys::get_and_reset_last_err()) {
            None => Ok(()),
            Some(c_error) => Err(TchError::Torch(c_error)),
        }
    }
}

macro_rules! unsafe_torch {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::wrappers::utils::read_and_clean_error().unwrap();
        v
    }};
}

macro_rules! unsafe_torch_err {
    ($e:expr) => {{
        let v = unsafe { $e };
        crate::wrappers::utils::read_and_clean_error()?;
        v
    }};
}

// Be cautious when using this function as the returned CString should be stored
// in a variable when using as_ptr. Otherwise dangling pointer issues are likely
// to happen.
pub(super) fn path_to_cstring<T: AsRef<std::path::Path>>(
    path: T,
) -> Result<std::ffi::CString, TchError> {
    let path = path.as_ref();
    match path.to_str() {
        Some(path) => Ok(std::ffi::CString::new(path)?),
        None => Err(TchError::Io(io::Error::other(format!(
            "path {path:?} cannot be converted to UTF-8"
        )))),
    }
}

/// Sets the random seed used by torch.
pub fn manual_seed(seed: i64) {
    unsafe_torch!(torch_sys::at_manual_seed(seed))
}

/// Get the number of threads used by torch for inter-op parallelism.
pub fn get_num_interop_threads() -> i32 {
    unsafe_torch!(torch_sys::at_get_num_interop_threads())
}

/// Get the number of threads used by torch in parallel regions.
pub fn get_num_threads() -> i32 {
    unsafe_torch!(torch_sys::at_get_num_threads())
}

/// Set the number of threads used by torch for inter-op parallelism.
pub fn set_num_interop_threads(n_threads: i32) {
    unsafe_torch!(torch_sys::at_set_num_interop_threads(n_threads))
}

/// Set the number of threads used by torch in parallel regions.
pub fn set_num_threads(n_threads: i32) {
    unsafe_torch!(torch_sys::at_set_num_threads(n_threads))
}

pub fn has_openmp() -> bool {
    unsafe_torch!(torch_sys::at_context_has_openmp())
}

pub fn has_mkl() -> bool {
    unsafe_torch!(torch_sys::at_context_has_mkl())
}
pub fn has_lapack() -> bool {
    unsafe_torch!(torch_sys::at_context_has_lapack())
}
pub fn has_mkldnn() -> bool {
    unsafe_torch!(torch_sys::at_context_has_mkldnn())
}
pub fn has_magma() -> bool {
    unsafe_torch!(torch_sys::at_context_has_magma())
}
pub fn has_cuda() -> bool {
    unsafe_torch!(torch_sys::at_context_has_cuda())
}
pub fn has_cudart() -> bool {
    unsafe_torch!(torch_sys::at_context_has_cudart())
}
pub fn has_cusolver() -> bool {
    unsafe_torch!(torch_sys::at_context_has_cusolver())
}
pub fn has_hip() -> bool {
    unsafe_torch!(torch_sys::at_context_has_hip())
}
pub fn has_ipu() -> bool {
    unsafe_torch!(torch_sys::at_context_has_ipu())
}
pub fn has_xla() -> bool {
    unsafe_torch!(torch_sys::at_context_has_xla())
}
pub fn has_lazy() -> bool {
    unsafe_torch!(torch_sys::at_context_has_lazy())
}
pub fn has_mps() -> bool {
    unsafe_torch!(torch_sys::at_context_has_mps())
}
pub fn version_cudnn() -> i64 {
    unsafe_torch!(torch_sys::at_context_version_cudnn())
}
pub fn version_cudart() -> i64 {
    unsafe_torch!(torch_sys::at_context_version_cudart())
}

/// Check whether the vulkan backend is available. None that this
/// backend is not included by default as of PyTorch 2.0.0.
/// https://pytorch.org/tutorials/prototype/vulkan_workflow.html#building-pytorch-with-vulkan-backend
pub fn has_vulkan() -> bool {
    crate::Tensor::is_vulkan_available()
}

/// Quantization engines
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum QEngine {
    NoQEngine,
    FBGEMM,
    QNNPACK,
}

impl QEngine {
    fn to_cint(self) -> i32 {
        match self {
            QEngine::NoQEngine => 0,
            QEngine::FBGEMM => 1,
            QEngine::QNNPACK => 2,
        }
    }
    pub fn set(self) -> Result<(), TchError> {
        unsafe_torch_err!(torch_sys::at_set_qengine(self.to_cint()));
        Ok(())
    }
}

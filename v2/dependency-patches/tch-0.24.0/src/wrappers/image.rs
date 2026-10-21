use super::tensor::Tensor;
use super::utils::path_to_cstring;
use crate::TchError;
use libc::c_int;
use std::path::Path;

/// On success returns a tensor of shape [width, height, channels].
pub fn load_hwc<T: AsRef<Path>>(path: T) -> Result<Tensor, TchError> {
    let path = path_to_cstring(path)?;
    let c_tensor = unsafe_torch_err!(torch_sys::at_load_image(path.as_ptr()));
    Ok(Tensor { c_tensor })
}

/// On success returns a tensor of shape [width, height, channels].
pub fn load_hwc_from_mem(data: &[u8]) -> Result<Tensor, TchError> {
    let c_tensor =
        unsafe_torch_err!(torch_sys::at_load_image_from_memory(data.as_ptr(), data.len()));
    Ok(Tensor { c_tensor })
}

/// Expects a tensor of shape [width, height, channels].
pub fn save_hwc<T: AsRef<Path>>(t: &Tensor, path: T) -> Result<(), TchError> {
    let path = path_to_cstring(path)?;
    let _ = unsafe_torch_err!(torch_sys::at_save_image(t.c_tensor, path.as_ptr()));
    Ok(())
}

/// Expects a tensor of shape [width, height, channels].
/// On success returns a tensor of shape [width, height, channels].
pub fn resize_hwc(t: &Tensor, out_w: i64, out_h: i64) -> Result<Tensor, TchError> {
    let c_tensor = unsafe_torch_err!({
        torch_sys::at_resize_image(t.c_tensor, out_w as c_int, out_h as c_int)
    });
    Ok(Tensor { c_tensor })
}

use crate::{TchError, Tensor};
use torch_sys::python::{self, C_pyobject};

pub type CPyObject = C_pyobject;

/// Check whether an object is a wrapped tensor or not.
///
/// # Safety
/// Undefined behavior if the given pointer is not a valid PyObject.
pub unsafe fn pyobject_check(pyobject: *mut CPyObject) -> Result<bool, TchError> {
    let v = unsafe_torch_err!(python::thp_variable_check(pyobject));
    Ok(v)
}

impl Tensor {
    /// Wrap a tensor in a Python object.
    pub fn pyobject_wrap(&self) -> Result<*mut CPyObject, TchError> {
        let v = unsafe_torch_err!(python::thp_variable_wrap(self.c_tensor));
        Ok(v)
    }

    /// Unwrap a tensor stored in a Python object. This returns `Ok(None)` if
    /// the object is not a wrapped tensor.
    ///
    /// # Safety
    /// Undefined behavior if the given pointer is not a valid PyObject.
    pub unsafe fn pyobject_unpack(pyobject: *mut CPyObject) -> Result<Option<Self>, TchError> {
        if !pyobject_check(pyobject)? {
            return Ok(None);
        }
        let v = unsafe_torch_err!(python::thp_variable_unpack(pyobject));
        Ok(Some(Tensor::from_ptr(v)))
    }
}

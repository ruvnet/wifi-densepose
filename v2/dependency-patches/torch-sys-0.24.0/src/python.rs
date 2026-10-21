use super::C_tensor;

#[repr(C)]
pub struct C_pyobject {
    _private: [u8; 0],
}

extern "C" {
    pub fn thp_variable_check(obj: *mut C_pyobject) -> bool;
    pub fn thp_variable_wrap(var: *mut C_tensor) -> *mut C_pyobject;
    pub fn thp_variable_unpack(obj: *mut C_pyobject) -> *mut C_tensor;
}

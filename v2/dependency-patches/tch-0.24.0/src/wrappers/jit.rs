//! JIT interface to run model trained/saved using PyTorch Python API.
use super::utils::{path_to_cstring, ptr_to_string};
use super::{device::Device, kind::Kind};
use crate::{nn::Path, TchError, Tensor};
use libc::{c_int, c_void};
use std::borrow::Borrow;
use std::convert::TryFrom;
use torch_sys::*;

/// Argument and output values for JIT models. These represent arbitrary values,
/// e.g. tensors, atomic values, pairs of values, etc.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum IValue {
    None,
    Tensor(crate::Tensor),
    Double(f64),
    Int(i64),
    Bool(bool),
    Tuple(Vec<IValue>),
    IntList(Vec<i64>),
    DoubleList(Vec<f64>),
    BoolList(Vec<bool>),
    String(String),
    StringList(Vec<String>),
    TensorList(Vec<crate::Tensor>),
    GenericList(Vec<IValue>),
    // We use a vec to represent dictionaries as f64 does not implement
    // Eq or Hash out of the box in rust. TODO: improve this?
    GenericDict(Vec<(IValue, IValue)>),
    Object(Object),
    Device(Device),
}

impl IValue {
    fn type_str(self) -> &'static str {
        match self {
            IValue::None => "None",
            IValue::Tensor(_) => "Tensor",
            IValue::Double(_) => "Double",
            IValue::Int(_) => "Int",
            IValue::Bool(_) => "Bool",
            IValue::Tuple(_) => "Tuple",
            IValue::IntList(_) => "IntList",
            IValue::DoubleList(_) => "DoubleList",
            IValue::BoolList(_) => "BoolList",
            IValue::String(_) => "String",
            IValue::StringList(_) => "StringList",
            IValue::TensorList(_) => "TensorList",
            IValue::GenericList(_) => "GenericList",
            IValue::GenericDict(_) => "GenericDict",
            IValue::Object(_) => "Object",
            IValue::Device(_) => "Device",
        }
    }
}

impl From<()> for IValue {
    fn from((): ()) -> Self {
        IValue::None
    }
}

impl<T1: Into<IValue>, T2: Into<IValue>> From<(T1, T2)> for IValue {
    fn from((p1, p2): (T1, T2)) -> Self {
        IValue::Tuple(vec![p1.into(), p2.into()])
    }
}

impl<T1: Into<IValue>, T2: Into<IValue>, T3: Into<IValue>> From<(T1, T2, T3)> for IValue {
    fn from((p1, p2, p3): (T1, T2, T3)) -> Self {
        IValue::Tuple(vec![p1.into(), p2.into(), p3.into()])
    }
}

impl<T1: Into<IValue>, T2: Into<IValue>, T3: Into<IValue>, T4: Into<IValue>> From<(T1, T2, T3, T4)>
    for IValue
{
    fn from((p1, p2, p3, p4): (T1, T2, T3, T4)) -> Self {
        IValue::Tuple(vec![p1.into(), p2.into(), p3.into(), p4.into()])
    }
}

impl<T1, T2, T1E, T2E> TryFrom<IValue> for (T1, T2)
where
    T1: TryFrom<IValue, Error = T1E>,
    TchError: From<T1E>,
    T2: TryFrom<IValue, Error = T2E>,
    TchError: From<T2E>,
{
    type Error = TchError;
    fn try_from(value: IValue) -> Result<Self, TchError> {
        match value {
            IValue::GenericList(mut vec) | IValue::Tuple(mut vec) => {
                if vec.len() == 2 {
                    let t2 = T2::try_from(vec.pop().unwrap())?;
                    let t1 = T1::try_from(vec.pop().unwrap())?;
                    Ok((t1, t2))
                } else {
                    Err(TchError::Kind(format!(
                        "unable to unpack ivalue, expected a tuple of len 2 got {}",
                        vec.len()
                    )))
                }
            }
            _ => Err(TchError::Kind(format!(
                "unable to unpack ivalue, expected a tuple got {}",
                value.type_str()
            ))),
        }
    }
}

impl<T1, T2, T3, T1E, T2E, T3E> TryFrom<IValue> for (T1, T2, T3)
where
    T1: TryFrom<IValue, Error = T1E>,
    TchError: From<T1E>,
    T2: TryFrom<IValue, Error = T2E>,
    TchError: From<T2E>,
    T3: TryFrom<IValue, Error = T3E>,
    TchError: From<T3E>,
{
    type Error = TchError;
    fn try_from(value: IValue) -> Result<Self, TchError> {
        match value {
            IValue::GenericList(mut vec) | IValue::Tuple(mut vec) => {
                if vec.len() == 3 {
                    let t3 = T3::try_from(vec.pop().unwrap())?;
                    let t2 = T2::try_from(vec.pop().unwrap())?;
                    let t1 = T1::try_from(vec.pop().unwrap())?;
                    Ok((t1, t2, t3))
                } else {
                    Err(TchError::Kind(format!(
                        "unable to unpack ivalue, expected a tuple of len 3 got {}",
                        vec.len()
                    )))
                }
            }
            _ => Err(TchError::Kind(format!(
                "unable to unpack ivalue, expected a tuple got {}",
                value.type_str()
            ))),
        }
    }
}

impl<T1, T2, T3, T4, T1E, T2E, T3E, T4E> TryFrom<IValue> for (T1, T2, T3, T4)
where
    T1: TryFrom<IValue, Error = T1E>,
    TchError: From<T1E>,
    T2: TryFrom<IValue, Error = T2E>,
    TchError: From<T2E>,
    T3: TryFrom<IValue, Error = T3E>,
    TchError: From<T3E>,
    T4: TryFrom<IValue, Error = T4E>,
    TchError: From<T4E>,
{
    type Error = TchError;
    fn try_from(value: IValue) -> Result<Self, TchError> {
        match value {
            IValue::GenericList(mut vec) | IValue::Tuple(mut vec) => {
                if vec.len() == 4 {
                    let t4 = T4::try_from(vec.pop().unwrap())?;
                    let t3 = T3::try_from(vec.pop().unwrap())?;
                    let t2 = T2::try_from(vec.pop().unwrap())?;
                    let t1 = T1::try_from(vec.pop().unwrap())?;
                    Ok((t1, t2, t3, t4))
                } else {
                    Err(TchError::Kind(format!(
                        "unable to unpack ivalue, expected a tuple of len 4 got {}",
                        vec.len()
                    )))
                }
            }
            _ => Err(TchError::Kind(format!(
                "unable to unpack ivalue, expected a tuple got {}",
                value.type_str()
            ))),
        }
    }
}

macro_rules! impl_from {
    ($type_:ty, $cons:ident) => {
        impl From<$type_> for IValue {
            fn from(v: $type_) -> Self {
                IValue::$cons(v)
            }
        }

        impl TryFrom<IValue> for $type_ {
            type Error = TchError;
            fn try_from(value: IValue) -> Result<$type_, TchError> {
                match value {
                    IValue::$cons(t) => Ok(t),
                    _ => Err(TchError::Kind(format!(
                        "unable to unpack ivalue, expected {} got {}",
                        std::stringify!($cons),
                        value.type_str()
                    ))),
                }
            }
        }

        // A generic trait for Option<T> would seem nicer but because
        // of E0119, this is currently hard to do.
        // See https://github.com/rust-lang/rust/issues/50133
        impl TryFrom<IValue> for Option<$type_> {
            type Error = TchError;
            fn try_from(value: IValue) -> Result<Self, TchError> {
                match value {
                    IValue::None => Ok(None),
                    IValue::$cons(t) => Ok(Some(t)),
                    _ => Err(TchError::Kind(format!(
                        "unable to unpack ivalue, expected {} or None got {}",
                        std::stringify!($cons),
                        value.type_str()
                    ))),
                }
            }
        }
    };
}

impl_from!(i64, Int);
impl_from!(f64, Double);
impl_from!(bool, Bool);
impl_from!(String, String);
impl_from!(Tensor, Tensor);
impl_from!(Vec<i64>, IntList);
impl_from!(Vec<f64>, DoubleList);
impl_from!(Vec<bool>, BoolList);
impl_from!(Vec<String>, StringList);
impl_from!(Vec<crate::Tensor>, TensorList);
impl_from!(Vec<IValue>, GenericList);
impl_from!(Vec<(IValue, IValue)>, GenericDict);
impl_from!(Object, Object);
impl_from!(Device, Device);

impl From<&str> for IValue {
    fn from(s: &str) -> Self {
        IValue::String(s.to_string())
    }
}

impl IValue {
    #![allow(unused_unsafe)]
    pub(super) fn to_c(&self) -> Result<*mut CIValue, TchError> {
        let c = unsafe_torch_err!(match self {
            IValue::Tensor(tensor) => ati_tensor(tensor.c_tensor),
            IValue::Int(i) => ati_int(*i),
            IValue::None => ati_none(),
            IValue::Double(f) => ati_double(*f),
            IValue::Bool(b) => ati_bool(i32::from(*b)),
            IValue::Tuple(v) => {
                let v = v.iter().map(Self::to_c).collect::<Result<Vec<_>, TchError>>()?;
                let tuple = ati_tuple(v.as_ptr(), v.len() as c_int);
                for x in v {
                    ati_free(x);
                }

                tuple
            }
            IValue::GenericList(v) => {
                let v = v.iter().map(Self::to_c).collect::<Result<Vec<_>, TchError>>()?;
                let list = ati_generic_list(v.as_ptr(), v.len() as c_int);
                for x in v {
                    ati_free(x);
                }
                list
            }
            IValue::IntList(v) => ati_int_list(v.as_ptr(), v.len() as c_int),
            IValue::DoubleList(v) => ati_double_list(v.as_ptr(), v.len() as c_int),
            IValue::BoolList(v) => {
                let v: Vec<libc::c_char> = v.iter().map(|&b| libc::c_char::from(b)).collect();
                ati_bool_list(v.as_ptr(), v.len() as c_int)
            }
            IValue::TensorList(v) => {
                let v = v.iter().map(|t| t.c_tensor).collect::<Vec<_>>();
                ati_tensor_list(v.as_ptr(), v.len() as c_int)
            }
            IValue::String(string) => {
                let c_str = std::ffi::CString::new(string.as_str())?;
                ati_string(c_str.as_ptr())
            }
            IValue::StringList(strings) => {
                let mut v = vec![];
                for s in strings {
                    v.push(std::ffi::CString::new(s.as_str())?);
                }
                let v_ptr: Vec<_> = v.iter().map(|s| s.as_ptr()).collect();
                ati_string_list(v_ptr.as_ptr(), v.len() as c_int)
            }
            IValue::GenericDict(dict) => {
                let v = dict
                    .iter()
                    .flat_map(|(k, v)| vec![Self::to_c(k), Self::to_c(v)])
                    .collect::<Result<Vec<_>, TchError>>()?;
                let dict = ati_generic_dict(v.as_ptr(), dict.len() as c_int);
                for x in v {
                    ati_free(x);
                }
                dict
            }
            IValue::Object(Object { c_ivalue }) => {
                // Clone the object if necessary before passing the pointer to the C++ side.
                unsafe_torch_err!(ati_clone(*c_ivalue))
            }
            IValue::Device(device) => {
                ati_device(device.c_int())
            }
        });
        Ok(c)
    }

    // This consumes the pointer and frees the associated memory (unless it is an Object).
    pub(super) fn from_c(c_ivalue: *mut CIValue) -> Result<Self, TchError> {
        let mut free = true;
        let tag = unsafe_torch_err!(ati_tag(c_ivalue));
        let v = match tag {
            0 => IValue::None,
            1 => {
                let c_tensor = unsafe_torch_err!(ati_to_tensor(c_ivalue));
                IValue::Tensor(crate::Tensor { c_tensor })
            }
            2 => IValue::Double(unsafe_torch_err!(ati_to_double(c_ivalue))),
            3 => IValue::Int(unsafe_torch_err!(ati_to_int(c_ivalue))),
            4 => {
                let b = unsafe_torch_err!(ati_to_bool(c_ivalue));
                if b < 0 {
                    return Err(TchError::Kind(format!("unexpected bool value {b}")));
                }
                IValue::Bool(b != 0)
            }
            5 => {
                let len = unsafe_torch_err!(ati_tuple_length(c_ivalue));
                let mut c_ivalues: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<CIValue>()).collect();
                unsafe_torch_err!(ati_to_tuple(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let vec: Result<Vec<_>, _> =
                    c_ivalues.iter().map(|&c_ivalue| Self::from_c(c_ivalue)).collect();
                IValue::Tuple(vec?)
            }
            6 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_array = vec![0i64; len as usize];
                unsafe_torch_err!(ati_to_int_list(c_ivalue, c_array.as_mut_ptr(), len));
                IValue::IntList(c_array)
            }
            7 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_array = vec![0f64; len as usize];
                unsafe_torch_err!(ati_to_double_list(c_ivalue, c_array.as_mut_ptr(), len));
                IValue::DoubleList(c_array)
            }
            8 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_array = vec![0_i8; len as usize];
                let c_array_ptr = c_array.as_mut_ptr() as *mut libc::c_char;
                unsafe_torch_err!(ati_to_bool_list(c_ivalue, c_array_ptr, len));
                IValue::BoolList(c_array.iter().map(|&x| x != 0).collect())
            }
            9 => {
                let ptr = unsafe_torch_err!(ati_to_string(c_ivalue));
                let string = match unsafe { ptr_to_string(ptr) } {
                    None => return Err(TchError::Kind("nullptr representation".to_string())),
                    Some(s) => s,
                };
                IValue::String(string)
            }
            10 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_tensors: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<C_tensor>()).collect();
                unsafe_torch_err!(ati_to_tensor_list(c_ivalue, c_tensors.as_mut_ptr(), len));
                let vec: Vec<_> = c_tensors.iter().map(|&c_tensor| Tensor { c_tensor }).collect();
                IValue::TensorList(vec)
            }
            12 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_ivalues: Vec<_> =
                    (0..len).map(|_| std::ptr::null_mut::<CIValue>()).collect();
                unsafe_torch_err!(ati_to_generic_list(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let vec: Result<Vec<_>, _> =
                    c_ivalues.iter().map(|&c_ivalue| Self::from_c(c_ivalue)).collect();
                IValue::GenericList(vec?)
            }
            13 => {
                let len = unsafe_torch_err!(ati_length(c_ivalue));
                let mut c_ivalues: Vec<_> =
                    (0..2 * len).map(|_| std::ptr::null_mut::<CIValue>()).collect();
                unsafe_torch_err!(ati_to_generic_dict(c_ivalue, c_ivalues.as_mut_ptr(), len));
                let mut res: Vec<(IValue, IValue)> = vec![];
                for i in 0..(len as usize) {
                    let key = Self::from_c(c_ivalues[2 * i])?;
                    let value = Self::from_c(c_ivalues[2 * i + 1])?;
                    res.push((key, value))
                }
                IValue::GenericDict(res)
            }
            14 => {
                free = false;
                IValue::Object(Object { c_ivalue })
            }
            _ => return Err(TchError::Kind(format!("unhandled tag {tag}"))),
        };
        if free {
            unsafe_torch_err!(ati_free(c_ivalue));
        }
        Ok(v)
    }
}

/// A jit PyTorch module.
///
/// These modules can be created via the
/// [TorchScript python api](https://pytorch.org/docs/stable/jit.html).
#[derive(Debug)]
pub struct CModule {
    pub(super) c_module: *mut CModule_,
}

unsafe impl Send for CModule {}

unsafe impl Sync for CModule {}

impl Drop for CModule {
    fn drop(&mut self) {
        unsafe_torch!(atm_free(self.c_module))
    }
}

impl CModule {
    /// Loads a PyTorch saved JIT model from a file.
    pub fn load<T: AsRef<std::path::Path>>(path: T) -> Result<CModule, TchError> {
        let path = path_to_cstring(path)?;
        let c_module = unsafe_torch_err!(atm_load(path.as_ptr()));
        Ok(CModule { c_module })
    }

    /// Loads a PyTorch saved JIT model from a file onto the given device.
    ///
    /// This function loads the model directly on the specified device,
    /// which means it also allows loading a GPU model on the CPU without having a CUDA enabled GPU.
    pub fn load_on_device<T: AsRef<std::path::Path>>(
        path: T,
        device: Device,
    ) -> Result<CModule, TchError> {
        let path = path_to_cstring(path)?;
        let c_module = unsafe_torch_err!(atm_load_on_device(path.as_ptr(), device.c_int()));
        Ok(CModule { c_module })
    }

    /// Loads a PyTorch saved JIT model from a read instance.
    pub fn load_data<T: std::io::Read>(f: &mut T) -> Result<CModule, TchError> {
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let buffer_ptr = buffer.as_ptr() as *const libc::c_char;
        let c_module = unsafe_torch_err!(atm_load_str(buffer_ptr, buffer.len()));
        Ok(CModule { c_module })
    }

    /// Loads a PyTorch saved JIT model from a read instance.
    ///
    /// This function loads the model directly on the specified device,
    /// which means it also allows loading a GPU model on the CPU without having a CUDA enabled GPU.
    pub fn load_data_on_device<T: std::io::Read>(
        f: &mut T,
        device: Device,
    ) -> Result<CModule, TchError> {
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer)?;
        let buffer_ptr = buffer.as_ptr() as *const libc::c_char;
        let c_module =
            unsafe_torch_err!(atm_load_str_on_device(buffer_ptr, buffer.len(), device.c_int()));
        Ok(CModule { c_module })
    }

    /// Performs the forward pass for a model on some specified tensor inputs. This is equivalent
    /// to calling method_ts with the 'forward' method name, and returns a single tensor.
    pub fn forward_ts<T: Borrow<Tensor>>(&self, ts: &[T]) -> Result<Tensor, TchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.borrow().c_tensor).collect();
        let c_tensor =
            unsafe_torch_err!(atm_forward(self.c_module, ts.as_ptr(), ts.len() as c_int));
        Ok(Tensor { c_tensor })
    }

    /// Performs the forward pass for a model on some specified ivalue inputs. This is equivalent
    /// to calling method_is with the 'forward' method name, and returns an arbitrary ivalue.
    pub fn forward_is<T: Borrow<IValue>>(&self, ts: &[T]) -> Result<IValue, TchError> {
        let ts = ts.iter().map(|x| x.borrow().to_c()).collect::<Result<Vec<_>, TchError>>()?;
        let c_ivalue =
            unsafe_torch_err!(atm_forward_(self.c_module, ts.as_ptr(), ts.len() as c_int));
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::from_c(c_ivalue)
    }

    /// Runs a specified entry point for a model on some given tensor inputs.
    pub fn method_ts<T: Borrow<Tensor>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<Tensor, TchError> {
        let ts: Vec<_> = ts.iter().map(|x| x.borrow().c_tensor).collect();
        let method_name = std::ffi::CString::new(method_name)?;
        let c_tensor = unsafe_torch_err!(atm_method(
            self.c_module,
            method_name.as_ptr(),
            ts.as_ptr(),
            ts.len() as c_int
        ));
        Ok(Tensor { c_tensor })
    }

    /// Runs a specified entry point for a model on some given ivalue inputs.
    pub fn method_is<T: Borrow<IValue>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<IValue, TchError> {
        let ts = ts.iter().map(|x| x.borrow().to_c()).collect::<Result<Vec<_>, TchError>>()?;
        let method_name = std::ffi::CString::new(method_name)?;
        let c_ivalue = unsafe_torch_err!(atm_method_(
            self.c_module,
            method_name.as_ptr(),
            ts.as_ptr(),
            ts.len() as c_int
        ));
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::from_c(c_ivalue)
    }

    /// Create a specified custom JIT class object with the given class name, eg: `__torch__.foo.Bar`
    pub fn create_class_is<T: Borrow<IValue>>(
        &self,
        clz_name: &str,
        ts: &[T],
    ) -> Result<IValue, TchError> {
        let ts = ts.iter().map(|x| x.borrow().to_c()).collect::<Result<Vec<_>, TchError>>()?;
        let clz_name = std::ffi::CString::new(clz_name)?;
        let c_ivalue = unsafe_torch_err!(atm_create_class_(
            self.c_module,
            clz_name.as_ptr(),
            ts.as_ptr(),
            ts.len() as c_int
        ));
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::from_c(c_ivalue)
    }

    /// Switches the module to evaluation mode.
    pub fn f_set_eval(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(atm_eval(self.c_module));
        Ok(())
    }

    /// Switches the module to evaluation mode.
    pub fn set_eval(&mut self) {
        self.f_set_eval().unwrap();
    }

    /// Switches the module to training mode.
    pub fn f_set_train(&mut self) -> Result<(), TchError> {
        unsafe_torch_err!(atm_train(self.c_module));
        Ok(())
    }

    /// Switches the module to training mode.
    pub fn set_train(&mut self) {
        self.f_set_train().unwrap();
    }

    /// Moves the module to a different device and converts the kind.
    pub fn to(&mut self, device: Device, kind: Kind, non_blocking: bool) {
        unsafe_torch!(atm_to(self.c_module, device.c_int(), kind.c_int(), non_blocking));
    }

    /// Saves a module to a given path.
    pub fn save<T: AsRef<std::path::Path>>(&self, path: T) -> Result<(), TchError> {
        let path = path_to_cstring(path)?;
        unsafe_torch_err!(atm_save(self.c_module, path.as_ptr()));
        Ok(())
    }

    /// Loads some named tensors from a module
    pub fn named_parameters(&self) -> Result<Vec<(String, Tensor)>, TchError> {
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(atm_named_parameters(
            self.c_module,
            &mut v as *mut _ as *mut c_void,
            super::tensor::add_callback
        ));
        Ok(v)
    }

    /// Create a new module by tracing the application of the specified function on
    /// the given inputs.
    pub fn create_by_tracing<F>(
        modl_name: &str,
        fn_name: &str,
        inputs: &[Tensor],
        closure: &mut F,
    ) -> Result<CModule, TchError>
    where
        F: FnMut(&[Tensor]) -> Vec<Tensor>,
    {
        let modl_name = std::ffi::CString::new(modl_name)?;
        let fn_name = std::ffi::CString::new(fn_name)?;
        let c_inputs = inputs.iter().map(|tensor| tensor.c_tensor).collect::<Vec<_>>();
        let c_module = unsafe_torch_err!(atm_create_for_tracing(
            modl_name.as_ptr(),
            c_inputs.as_ptr(),
            c_inputs.len() as c_int
        ));
        let outputs = closure(inputs);
        let c_outputs = outputs.iter().map(|tensor| tensor.c_tensor).collect::<Vec<_>>();
        unsafe_torch_err!(atm_end_tracing(
            c_module,
            fn_name.as_ptr(),
            c_outputs.as_ptr(),
            c_outputs.len() as c_int,
        ));
        Ok(CModule { c_module })
    }
}

/// The trainable version of a jit PyTorch module.
///
/// These modules can be created via the
/// [TorchScript python api](https://pytorch.org/docs/stable/jit.html).
#[derive(Debug)]
pub struct TrainableCModule {
    pub(crate) inner: CModule,
}

impl TrainableCModule {
    /// Loads a PyTorch saved JIT module from a file.
    ///
    /// This function also adds the tensors from the JIT module to the VarStore path
    /// passed as argument so that the module can be trained.
    pub fn load<T: AsRef<std::path::Path>>(module_path: T, path: Path) -> Result<Self, TchError> {
        let inner = CModule::load_on_device(module_path, path.device())?;
        for (name, tensor) in inner.named_parameters()? {
            let requires_grad = tensor.requires_grad();
            let _t = path.add(&name.replace('.', "_"), tensor, requires_grad);
        }
        Ok(TrainableCModule { inner })
    }

    /// Loads a PyTorch saved JIT model from a read instance.
    ///
    /// This function also adds the tensors from the JIT module to the VarStore path
    /// passed as argument so that the module can be trained.
    pub fn load_data<T: std::io::Read>(data: &mut T, path: Path) -> Result<Self, TchError> {
        let inner = CModule::load_data_on_device(data, path.device())?;
        for (name, tensor) in inner.named_parameters()? {
            let requires_grad = tensor.requires_grad();
            let _t = path.add(&name.replace('.', "_"), tensor, requires_grad);
        }
        Ok(TrainableCModule { inner })
    }

    pub fn save<T: AsRef<std::path::Path>>(&self, module_path: T) -> Result<(), TchError> {
        self.inner.save(module_path)
    }

    /// Switches the module to training mode.
    pub fn f_set_train(&mut self) -> Result<(), TchError> {
        self.inner.f_set_train()
    }

    /// Switches the module to training mode.
    pub fn set_train(&mut self) {
        self.inner.set_train()
    }

    /// Switches the module to evaluation mode.
    pub fn f_set_eval(&mut self) -> Result<(), TchError> {
        self.inner.f_set_eval()
    }

    /// Switches the module to evaluation mode.
    pub fn set_eval(&mut self) {
        self.inner.set_eval()
    }

    /// Performs the forward pass for a model on some specified tensor inputs.
    pub fn forward_ts<T: Borrow<Tensor>>(&self, ts: &[T]) -> Result<Tensor, TchError> {
        self.inner.forward_ts(ts)
    }

    /// Performs the forward pass for a model on some specified ivalue inputs.
    pub fn forward_is<T: Borrow<IValue>>(&self, ts: &[T]) -> Result<IValue, TchError> {
        self.inner.forward_is(ts)
    }

    /// Runs a specified entry point for a model on some given tensor inputs.
    pub fn method_ts<T: Borrow<Tensor>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<Tensor, TchError> {
        self.inner.method_ts(method_name, ts)
    }

    /// Runs a specified entry point for a model on some given ivalue inputs.
    pub fn method_is<T: Borrow<IValue>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<IValue, TchError> {
        self.inner.method_is(method_name, ts)
    }
}

/// Returns whether profiling mode is set or not.
pub fn f_get_profiling_mode() -> Result<bool, TchError> {
    Ok(unsafe_torch_err!(atm_get_profiling_mode()) != 0)
}

/// Returns whether profiling mode is set or not.
pub fn get_profiling_mode() -> bool {
    f_get_profiling_mode().unwrap()
}

/// Activates or deactivates the profiling mode.
pub fn f_set_profiling_mode(b: bool) -> Result<(), TchError> {
    unsafe_torch_err!(atm_set_profiling_mode(b as c_int));
    Ok(())
}

/// Activates or deactivates the profiling mode.
pub fn set_profiling_mode(b: bool) {
    f_set_profiling_mode(b).unwrap()
}

pub fn f_fuser_cuda_set_enabled(enabled: bool) -> Result<(), TchError> {
    unsafe_torch_err!(atm_fuser_cuda_set_enabled(enabled));
    Ok(())
}

pub fn fuser_cuda_set_enabled(enabled: bool) {
    f_fuser_cuda_set_enabled(enabled).unwrap()
}

pub fn f_fuser_cuda_is_enabled() -> Result<bool, TchError> {
    let b = unsafe_torch_err!(atm_fuser_cuda_is_enabled());
    Ok(b)
}

pub fn fuser_cuda_is_enabled() -> bool {
    f_fuser_cuda_is_enabled().unwrap()
}

pub fn f_set_tensor_expr_fuser_enabled(b: bool) -> Result<(), TchError> {
    unsafe_torch_err!(atm_set_tensor_expr_fuser_enabled(b as c_int));
    Ok(())
}

pub fn set_tensor_expr_fuser_enabled(b: bool) {
    f_set_tensor_expr_fuser_enabled(b).unwrap()
}

pub fn f_get_tensor_expr_fuser_enabled() -> Result<bool, TchError> {
    Ok(unsafe_torch_err!(atm_get_tensor_expr_fuser_enabled()))
}

pub fn get_tensor_expr_fuser_enabled() -> bool {
    f_get_tensor_expr_fuser_enabled().unwrap()
}

/// Enables or disables the graph executor optimizer for the current thread.
///
/// # Arguments
///
/// * `b` - A boolean that if true enables the graph executor optimizer for the current thread.
///
/// This function returns an error if it is not possible to enable or disable the graph executor optimizer.
pub fn f_set_graph_executor_optimize(b: bool) -> Result<(), TchError> {
    unsafe_torch_err!(at_set_graph_executor_optimize(b));
    Ok(())
}

/// Enables or disables the graph executor optimizer for the current thread.
///
/// # Arguments
///
/// * `b` - A boolean that if true enables the graph executor optimizer for the current thread.
///
/// This panics if it is not possible to enable or disable the graph executor optimizer.
pub fn set_graph_executor_optimize(b: bool) {
    f_set_graph_executor_optimize(b).unwrap();
}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq)]
pub struct Object {
    c_ivalue: *mut CIValue,
}

impl Object {
    /// Applies the specified method to the object. The method takes as argument an arbitrary
    /// number of ivalues and returns an ivalue.
    pub fn method_is<T: Borrow<IValue>>(
        &self,
        method_name: &str,
        ts: &[T],
    ) -> Result<IValue, TchError> {
        let ts = ts.iter().map(|x| x.borrow().to_c()).collect::<Result<Vec<_>, TchError>>()?;
        let method_name = std::ffi::CString::new(method_name)?;
        let c_ivalue = unsafe_torch_err!(ati_object_method_(
            self.c_ivalue,
            method_name.as_ptr(),
            ts.as_ptr(),
            ts.len() as c_int
        ));
        for x in ts {
            unsafe { ati_free(x) }
        }
        IValue::from_c(c_ivalue)
    }

    /// Retrieves the specified attribute from an object as an ivalue.
    pub fn getattr(&self, attr_name: &str) -> Result<IValue, TchError> {
        let property_name = std::ffi::CString::new(attr_name)?;
        let c_ivalue =
            unsafe_torch_err!(ati_object_getattr_(self.c_ivalue, property_name.as_ptr()));
        if c_ivalue.is_null() {
            return Err(TchError::Torch(format!(
                "Object.getattr(\"{attr_name}\") returned CIValue nullptr"
            )));
        }
        IValue::from_c(c_ivalue)
    }
}

impl Drop for Object {
    fn drop(&mut self) {
        unsafe_torch!(ati_free(self.c_ivalue))
    }
}

#[cfg(test)]
mod tests {
    use super::IValue;
    use std::f64::consts;

    fn round_trip<T: Into<IValue>>(t: T) {
        let ivalue: IValue = t.into();
        let ivalue2 = IValue::from_c(ivalue.to_c().unwrap()).unwrap();
        assert_eq!(ivalue, ivalue2);
    }
    #[test]
    fn ivalue_round_trip() {
        round_trip(());
        round_trip(true);
        round_trip(false);
        round_trip(-1);
        round_trip(42);
        round_trip(15);
        round_trip("".to_string());
        round_trip("foobar".to_string());
        round_trip((42, consts::PI));
        round_trip(vec![42, 1337]);
        round_trip(vec![consts::E, consts::PI, 299792458.00001]);
        round_trip((vec![true, false, true, true], vec![consts::E, consts::PI, 299792458.00001]));
        round_trip(vec![IValue::from(42), IValue::from("foobar")]);
        round_trip(vec![
            (IValue::from(42), IValue::from("foobar")),
            (IValue::from("foo"), IValue::from("bar")),
        ]);
    }
}

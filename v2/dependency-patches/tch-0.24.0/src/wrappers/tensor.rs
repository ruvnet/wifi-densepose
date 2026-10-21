use super::stream::ReadSeekAdapter;
use super::utils::{path_to_cstring, ptr_to_string};
use super::{
    device::{Cuda, Device},
    kind,
    kind::Kind,
};
use crate::TchError;
use libc::{c_char, c_int, c_void};
use std::borrow::Borrow;
use std::io::{Read, Seek, Write};
use std::path::Path;
use torch_sys::io::ReadStream;
use torch_sys::*;

/// A tensor object.
#[must_use]
pub struct Tensor {
    pub(super) c_tensor: *mut C_tensor,
}

unsafe impl Send for Tensor {}

pub extern "C" fn add_callback(data: *mut c_void, name: *const c_char, c_tensor: *mut C_tensor) {
    let name = unsafe { std::ffi::CStr::from_ptr(name).to_str().unwrap() };
    let name = name.replace('|', ".");
    let v: &mut Vec<(String, Tensor)> = unsafe { &mut *(data as *mut Vec<(String, Tensor)>) };
    v.push((name, Tensor { c_tensor }))
}

impl Tensor {
    /// Creates a new tensor.
    pub fn new() -> Tensor {
        let c_tensor = unsafe_torch!(at_new_tensor());
        Tensor { c_tensor }
    }

    /// Creates a new tensor from the pointer to an existing C++ tensor.
    ///
    /// # Safety
    ///
    /// The caller must ensures that the pointer outlives the Rust
    /// object.
    pub unsafe fn from_ptr(c_tensor: *mut C_tensor) -> Self {
        Self { c_tensor }
    }

    /// Creates a new tensor from the pointer to an existing C++ tensor.
    ///
    /// # Safety
    ///
    /// A shallow copy of the pointer is made so there is no need for
    /// this pointer to remain valid for the whole lifetime of the Rust
    /// object.
    pub unsafe fn clone_from_ptr(c_tensor: *mut C_tensor) -> Self {
        let c_tensor = at_shallow_clone(c_tensor);
        crate::wrappers::utils::read_and_clean_error().unwrap();
        Self { c_tensor }
    }

    /// Returns a pointer to the underlying C++ tensor.
    ///
    /// The caller must ensures that the Rust tensor object outlives
    /// this pointer.
    pub fn as_ptr(&self) -> *const C_tensor {
        self.c_tensor
    }

    /// Returns a mutable pointer to the underlying C++ tensor.
    ///
    /// The caller must ensures that the Rust tensor object outlives
    /// this pointer.
    pub fn as_mut_ptr(&mut self) -> *mut C_tensor {
        self.c_tensor
    }

    /// Returns the number of dimension of the tensor.
    pub fn dim(&self) -> usize {
        unsafe_torch!(at_dim(self.c_tensor))
    }

    /// Returns the shape of the input tensor.
    pub fn size(&self) -> Vec<i64> {
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        let mut sz = vec![0i64; dim];
        unsafe_torch!(at_shape(self.c_tensor, sz.as_mut_ptr()));
        sz
    }

    /// Returns the tensor size for single dimension tensors.
    pub fn size1(&self) -> Result<i64, TchError> {
        match self.size().as_slice() {
            &[s0] => Ok(s0),
            size => Err(TchError::Shape(format!("expected one dim, got {size:?}"))),
        }
    }

    /// Returns the tensor sizes for two dimension tensors.
    pub fn size2(&self) -> Result<(i64, i64), TchError> {
        match self.size().as_slice() {
            &[s0, s1] => Ok((s0, s1)),
            size => Err(TchError::Shape(format!("expected two dims, got {size:?}"))),
        }
    }

    /// Returns the tensor sizes for three dimension tensors.
    pub fn size3(&self) -> Result<(i64, i64, i64), TchError> {
        match self.size().as_slice() {
            &[s0, s1, s2] => Ok((s0, s1, s2)),
            size => Err(TchError::Shape(format!("expected three dims, got {size:?}"))),
        }
    }

    /// Returns the tensor sizes for four dimension tensors.
    pub fn size4(&self) -> Result<(i64, i64, i64, i64), TchError> {
        match self.size().as_slice() {
            &[s0, s1, s2, s3] => Ok((s0, s1, s2, s3)),
            size => Err(TchError::Shape(format!("expected four dims, got {size:?}"))),
        }
    }

    /// Returns the tensor sizes for five dimension tensors.
    pub fn size5(&self) -> Result<(i64, i64, i64, i64, i64), TchError> {
        match self.size().as_slice() {
            &[s0, s1, s2, s3, s4] => Ok((s0, s1, s2, s3, s4)),
            size => Err(TchError::Shape(format!("expected five dims, got {size:?}"))),
        }
    }

    /// Returns the tensor sizes for six dimension tensors.
    pub fn size6(&self) -> Result<(i64, i64, i64, i64, i64, i64), TchError> {
        match self.size().as_slice() {
            &[s0, s1, s2, s3, s4, s5] => Ok((s0, s1, s2, s3, s4, s5)),
            size => Err(TchError::Shape(format!("expected six dims, got {size:?}"))),
        }
    }

    /// Returns the stride of the input tensor.
    pub fn stride(&self) -> Vec<i64> {
        let dim = unsafe_torch!(at_dim(self.c_tensor));
        let mut sz = vec![0i64; dim];
        unsafe_torch!(at_stride(self.c_tensor, sz.as_mut_ptr()));
        sz
    }

    /// Returns the tensor strides for single dimension tensors.
    pub fn stride1(&self) -> Result<i64, TchError> {
        match self.stride().as_slice() {
            &[s0] => Ok(s0),
            size => Err(TchError::Shape(format!("expected one dim, got {size:?}"))),
        }
    }

    /// Returns the tensor strides for two dimension tensors.
    pub fn stride2(&self) -> Result<(i64, i64), TchError> {
        match self.stride().as_slice() {
            &[s0, s1] => Ok((s0, s1)),
            size => Err(TchError::Shape(format!("expected two dims, got {size:?}"))),
        }
    }

    /// Returns the tensor strides for three dimension tensors.
    pub fn stride3(&self) -> Result<(i64, i64, i64), TchError> {
        match self.stride().as_slice() {
            &[s0, s1, s2] => Ok((s0, s1, s2)),
            size => Err(TchError::Shape(format!("expected three dims, got {size:?}"))),
        }
    }

    /// Returns the tensor strides for four dimension tensors.
    pub fn stride4(&self) -> Result<(i64, i64, i64, i64), TchError> {
        match self.stride().as_slice() {
            &[s0, s1, s2, s3] => Ok((s0, s1, s2, s3)),
            size => Err(TchError::Shape(format!("expected four dims, got {size:?}"))),
        }
    }

    /// Returns the tensor strides for five dimension tensors.
    pub fn stride5(&self) -> Result<(i64, i64, i64, i64, i64), TchError> {
        match self.stride().as_slice() {
            &[s0, s1, s2, s3, s4] => Ok((s0, s1, s2, s3, s4)),
            size => Err(TchError::Shape(format!("expected five dims, got {size:?}"))),
        }
    }

    /// Returns the tensor strides for six dimension tensors.
    pub fn stride6(&self) -> Result<(i64, i64, i64, i64, i64, i64), TchError> {
        match self.stride().as_slice() {
            &[s0, s1, s2, s3, s4, s5] => Ok((s0, s1, s2, s3, s4, s5)),
            size => Err(TchError::Shape(format!("expected six dims, got {size:?}"))),
        }
    }

    /// Returns the kind of elements stored in the input tensor. Returns
    /// an error on undefined tensors and unsupported data types.
    pub fn f_kind(&self) -> Result<Kind, TchError> {
        let kind = unsafe_torch!(at_scalar_type(self.c_tensor));
        Kind::from_c_int(kind)
    }

    /// Returns the kind of elements stored in the input tensor. Panics
    /// an error on undefined tensors and unsupported data types.
    pub fn kind(&self) -> Kind {
        self.f_kind().unwrap()
    }

    /// Returns the device on which the input tensor is located.
    pub fn device(&self) -> Device {
        let device = unsafe_torch!(at_device(self.c_tensor));
        Device::from_c_int(device)
    }

    /// Prints the input tensor.
    ///
    /// Caution: this uses the C++ printer which prints the whole tensor even if
    /// it is very large.
    pub fn print(&self) {
        unsafe_torch!(at_print(self.c_tensor))
    }

    /// Returns a double value on tensors holding a single element. An error is
    /// returned otherwise.
    pub fn f_double_value(&self, idx: &[i64]) -> Result<f64, TchError> {
        Ok(unsafe_torch_err!({
            at_double_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32)
        }))
    }

    /// Returns an int value on tensors holding a single element. An error is
    /// returned otherwise.
    pub fn f_int64_value(&self, idx: &[i64]) -> Result<i64, TchError> {
        Ok(unsafe_torch_err!({
            at_int64_value_at_indexes(self.c_tensor, idx.as_ptr(), idx.len() as i32)
        }))
    }

    /// Returns a double value on tensors holding a single element. Panics otherwise.
    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.f_double_value(idx).unwrap()
    }

    /// Returns an int value on tensors holding a single element. Panics otherwise.
    pub fn int64_value(&self, idx: &[i64]) -> i64 {
        self.f_int64_value(idx).unwrap()
    }

    /// Returns true if gradient are currently tracked for this tensor.
    pub fn requires_grad(&self) -> bool {
        unsafe_torch!(at_requires_grad(self.c_tensor)) != 0
    }

    /// Returns the address of the first element of this tensor.
    pub fn data_ptr(&self) -> *mut c_void {
        unsafe_torch!(at_data_ptr(self.c_tensor))
    }

    /// Returns true if the tensor is defined.
    pub fn defined(&self) -> bool {
        unsafe_torch!(at_defined(self.c_tensor) != 0)
    }

    /// Returns true if the tensor is compatible with MKL-DNN (oneDNN).
    pub fn is_mkldnn(&self) -> bool {
        unsafe_torch!(at_is_mkldnn(self.c_tensor) != 0)
    }

    /// Returns true if the tensor is sparse.
    pub fn is_sparse(&self) -> bool {
        unsafe_torch!(at_is_sparse(self.c_tensor) != 0)
    }

    // Returns true if the tensor if contiguous
    pub fn is_contiguous(&self) -> bool {
        unsafe_torch!(at_is_contiguous(self.c_tensor) != 0)
    }

    /// Zeroes the gradient tensor attached to this tensor if defined.
    pub fn zero_grad(&mut self) {
        let mut grad = self.grad();
        if grad.defined() {
            let _ = grad.detach_().zero_();
        }
    }

    /// Runs the backward pass, populating the gradient tensors for tensors
    /// which gradients are tracked.
    ///
    /// Gradients tracking can be turned on via `set_requires_grad`.
    pub fn f_backward(&self) -> Result<(), TchError> {
        unsafe_torch_err!(at_backward(self.c_tensor, 0, 0));
        Ok(())
    }

    /// Runs the backward pass, populating the gradient tensors for tensors
    /// which gradients are tracked.
    ///
    /// Gradients tracking can be turned on via `set_requires_grad`.
    /// Panics if the C++ api returns an exception.
    pub fn backward(&self) {
        self.f_backward().unwrap()
    }

    pub fn f_run_backward<T1, T2>(
        tensors: &[T1],
        inputs: &[T2],
        keep_graph: bool,
        create_graph: bool,
    ) -> Result<Vec<Tensor>, TchError>
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        let mut outputs = vec![std::ptr::null_mut(); inputs.len()];
        let tensors: Vec<_> = tensors.iter().map(|x| x.borrow().c_tensor).collect();
        let inputs: Vec<_> = inputs.iter().map(|x| x.borrow().c_tensor).collect();
        unsafe_torch_err!(at_run_backward(
            tensors.as_ptr(),
            tensors.len() as c_int,
            inputs.as_ptr(),
            inputs.len() as c_int,
            outputs.as_mut_ptr(),
            keep_graph as c_int,
            create_graph as c_int,
        ));
        Ok(outputs.into_iter().map(|c_tensor| Tensor { c_tensor }).collect())
    }

    pub fn run_backward<T1, T2>(
        tensors: &[T1],
        inputs: &[T2],
        keep_graph: bool,
        create_graph: bool,
    ) -> Vec<Tensor>
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        Tensor::f_run_backward(tensors, inputs, keep_graph, create_graph).unwrap()
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn f_copy_data_u8(&self, dst: &mut [u8], numel: usize) -> Result<(), TchError> {
        let elt_size_in_bytes = self.f_kind()?.elt_size_in_bytes();
        if dst.len() < numel * elt_size_in_bytes {
            return Err(TchError::Shape(format!("slice len < {numel}")));
        }
        unsafe_torch_err!(at_copy_data(
            self.c_tensor,
            dst.as_mut_ptr() as *const c_void,
            numel,
            elt_size_in_bytes,
        ));
        Ok(())
    }

    /// Unscale tensor while checking for infinities.
    ///
    /// `found_inf` is a singleton tensor that is used to record the
    /// presence of infinite values. `inv_scale` is a scalar containing
    /// the inverse scaling factor. This method is only available
    /// for CUDA tensors.
    pub fn f_internal_amp_non_finite_check_and_unscale(
        &mut self,
        found_inf: &mut Tensor,
        inv_scale: &Tensor,
    ) -> Result<(), TchError> {
        unsafe_torch_err!(at__amp_non_finite_check_and_unscale(
            self.c_tensor,
            found_inf.c_tensor,
            inv_scale.c_tensor
        ));

        Ok(())
    }

    /// Unscale tensor while checking for infinities.
    ///
    /// `found_inf` is a singleton tensor that is used to record the
    /// presence of infinite values. `inv_scale` is a scalar containing
    /// the inverse scaling factor. This method is only available
    /// for CUDA tensors.
    pub fn internal_amp_non_finite_check_and_unscale(
        &mut self,
        found_inf: &mut Tensor,
        inv_scale: &Tensor,
    ) {
        self.f_internal_amp_non_finite_check_and_unscale(found_inf, inv_scale).unwrap()
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn copy_data_u8(&self, dst: &mut [u8], numel: usize) {
        self.f_copy_data_u8(dst, numel).unwrap()
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn f_copy_data<T: kind::Element>(
        &self,
        dst: &mut [T],
        numel: usize,
    ) -> Result<(), TchError> {
        if T::KIND != self.f_kind()? {
            return Err(TchError::Kind(format!(
                "incoherent elt kind, {:?} != {:?}",
                self.f_kind(),
                T::KIND
            )));
        }
        if dst.len() < numel {
            return Err(TchError::Shape(format!("slice len < {numel}")));
        }
        unsafe_torch_err!(at_copy_data(
            self.c_tensor,
            dst.as_mut_ptr() as *const c_void,
            numel,
            T::KIND.elt_size_in_bytes(),
        ));
        Ok(())
    }

    /// Copies `numel` elements from `self` to `dst`.
    pub fn copy_data<T: kind::Element>(&self, dst: &mut [T], numel: usize) {
        self.f_copy_data(dst, numel).unwrap()
    }

    /// Returns the total number of elements stored in a tensor.
    pub fn numel(&self) -> usize {
        self.size().iter().product::<i64>() as usize
    }

    // This is similar to vec_... but faster as it directly blits the data.
    /// Converts a slice to a tensor.
    pub fn f_from_slice<T: kind::Element>(data: &[T]) -> Result<Tensor, TchError> {
        let data_len = data.len();
        let data = data.as_ptr() as *const c_void;
        let c_tensor = unsafe_torch_err!(at_tensor_of_data(
            data,
            [data_len as i64].as_ptr(),
            1,
            T::KIND.elt_size_in_bytes(),
            T::KIND.c_int(),
        ));
        Ok(Tensor { c_tensor })
    }

    /// Converts a slice to a tensor.
    pub fn from_slice<T: kind::Element>(data: &[T]) -> Tensor {
        Self::f_from_slice(data).unwrap()
    }

    /// Converts some byte data to a tensor with some specified kind and shape.
    pub fn f_from_data_size(data: &[u8], size: &[i64], kind: Kind) -> Result<Tensor, TchError> {
        let data = data.as_ptr() as *const c_void;
        let elt_size_in_bytes = kind.elt_size_in_bytes();
        let c_tensor = unsafe_torch_err!(at_tensor_of_data(
            data,
            size.as_ptr(),
            size.len(),
            elt_size_in_bytes,
            kind.c_int(),
        ));
        Ok(Tensor { c_tensor })
    }

    /// Creates a tensor from data that is assumed to be initialized.
    /// Resize operations are not allowed on this tensor without copying the data first.
    /// An empty strides slice will result in using the default strides.
    /// # Safety
    ///   Behavior is undefined if `data` points to invalid data.
    pub unsafe fn f_from_blob(
        data: *const u8,
        size: &[i64],
        strides: &[i64],
        kind: Kind,
        device: Device,
    ) -> Result<Tensor, TchError> {
        let data = data as *const c_void;
        #[allow(unused_unsafe)]
        let c_tensor = unsafe_torch_err!(at_tensor_of_blob(
            data,
            size.as_ptr(),
            size.len(),
            strides.as_ptr(),
            strides.len(),
            kind.c_int(),
            device.c_int()
        ));
        Ok(Tensor { c_tensor })
    }

    /// Creates a tensor from data that is assumed to be initialized.
    /// Resize operations are not allowed on this tensor without copying the data first.
    /// An empty strides slice will result in using the default strides.
    /// # Safety
    ///   Behavior is undefined if `data` points to invalid data.
    pub unsafe fn from_blob(
        data: *const u8,
        size: &[i64],
        strides: &[i64],
        kind: Kind,
        device: Device,
    ) -> Tensor {
        Self::f_from_blob(data, size, strides, kind, device).unwrap()
    }

    /// Converts some byte data to a tensor with some specified kind and shape.
    pub fn from_data_size(data: &[u8], size: &[i64], kind: Kind) -> Tensor {
        Self::f_from_data_size(data, size, kind).unwrap()
    }

    /// Returns a new tensor that share storage with the input tensor.
    pub fn shallow_clone(&self) -> Tensor {
        let c_tensor = unsafe_torch!(at_shallow_clone(self.c_tensor));
        Tensor { c_tensor }
    }

    /// Gets the sub-tensor at the given index.
    pub fn f_get(&self, index: i64) -> Result<Tensor, TchError> {
        let c_tensor = unsafe_torch_err!(at_get(self.c_tensor, index as c_int));
        Ok(Tensor { c_tensor })
    }

    /// Gets the sub-tensor at the given index.
    pub fn get(&self, index: i64) -> Tensor {
        self.f_get(index).unwrap()
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn f_copy_(&mut self, src: &Tensor) -> Result<(), TchError> {
        unsafe_torch_err!(at_copy_(self.c_tensor, src.c_tensor));
        Ok(())
    }

    /// Copies values from the argument tensor to the input tensor.
    pub fn copy_(&mut self, src: &Tensor) {
        self.f_copy_(src).unwrap()
    }

    /// Loads a tensor from a file.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load<T: AsRef<Path>>(path: T) -> Result<Tensor, TchError> {
        let path = path_to_cstring(path)?;
        let c_tensor = unsafe_torch_err!(at_load(path.as_ptr()));
        Ok(Tensor { c_tensor })
    }

    /// Loads a tensor from a stream.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_from_stream<T: Read + Seek>(stream: T) -> Result<Tensor, TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let boxed_stream: Box<Box<dyn ReadStream>> = Box::new(Box::new(adapter));
        let c_tensor =
            unsafe_torch_err!(at_load_from_stream(Box::into_raw(boxed_stream) as *mut c_void,));
        Ok(Tensor { c_tensor })
    }

    /// Saves a tensor to a file.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), TchError> {
        let path = path_to_cstring(path)?;
        unsafe_torch_err!(at_save(self.c_tensor, path.as_ptr()));
        Ok(())
    }

    /// Saves a tensor to a stream.
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_to_stream<W: Write>(&self, stream: W) -> Result<(), TchError> {
        let boxed_stream: Box<Box<dyn Write>> = Box::new(Box::new(stream));
        unsafe_torch_err!(at_save_to_stream(
            self.c_tensor,
            Box::into_raw(boxed_stream) as *mut c_void,
        ));
        Ok(())
    }

    /// Saves some named tensors to a file
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_multi<S: AsRef<str>, T: AsRef<Tensor>, P: AsRef<Path>>(
        named_tensors: &[(S, T)],
        path: P,
    ) -> Result<(), TchError> {
        let path = path_to_cstring(path)?;
        let c_tensors = named_tensors.iter().map(|nt| nt.1.as_ref().c_tensor).collect::<Vec<_>>();
        let names = named_tensors
            .iter()
            .map(|nt| nt.0.as_ref().replace('.', "|").into_bytes())
            .map(std::ffi::CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let name_ptrs = names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        unsafe_torch_err!(at_save_multi(
            c_tensors.as_ptr(),
            name_ptrs.as_ptr(),
            names.len() as i32,
            path.as_ptr(),
        ));
        Ok(())
    }

    /// Saves some named tensors to a stream
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn save_multi_to_stream<S: AsRef<str>, T: AsRef<Tensor>, W: Write>(
        named_tensors: &[(S, T)],
        stream: W,
    ) -> Result<(), TchError> {
        let boxed_stream: Box<Box<dyn Write>> = Box::new(Box::new(stream));
        let c_tensors = named_tensors.iter().map(|nt| nt.1.as_ref().c_tensor).collect::<Vec<_>>();
        let names = named_tensors
            .iter()
            .map(|nt| nt.0.as_ref().replace('.', "|").into_bytes())
            .map(std::ffi::CString::new)
            .collect::<Result<Vec<_>, _>>()?;
        let name_ptrs = names.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();
        unsafe_torch_err!(at_save_multi_to_stream(
            c_tensors.as_ptr(),
            name_ptrs.as_ptr(),
            names.len() as i32,
            Box::into_raw(boxed_stream) as *mut c_void,
        ));
        Ok(())
    }

    /// Loads some named tensors from a file
    ///
    /// The file format is the same as the one used for modules in the PyTorch C++ API.
    /// It commonly uses the .ot extension.
    pub fn load_multi<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_callback(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback
        ));
        Ok(v)
    }

    /// Loads some named tensors from a file to a given device
    ///
    /// The file format is the same as the one used for modules in the PyTorch C++ API.
    /// It commonly uses the .ot extension.
    pub fn load_multi_with_device<T: AsRef<Path>>(
        path: T,
        device: Device,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_callback_with_device(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback,
            device.c_int(),
        ));
        Ok(v)
    }

    /// Loads some named tensors from a zip file
    ///
    /// The expected file format is a zip archive containing a data.pkl file describing
    /// the embedded tensors. These are commonly used with the .bin extension to export
    /// PyTorch models and weights using the Python api.
    pub fn loadz_multi<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_loadz_callback(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback
        ));
        Ok(v)
    }

    /// Loads some named tensors from a zip file to a given device
    ///
    /// The expected file format is a zip archive containing a data.pkl file describing
    /// the embedded tensors. These are commonly used with the .bin extension to export
    /// PyTorch models and weights using the Python api.
    pub fn loadz_multi_with_device<T: AsRef<Path>>(
        path: T,
        device: Device,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let path = path_to_cstring(path)?;
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_loadz_callback_with_device(
            path.as_ptr(),
            &mut v as *mut _ as *mut c_void,
            add_callback,
            device.c_int(),
        ));
        Ok(v)
    }

    /// Loads some named tensors from a stream
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_multi_from_stream<T: Read + Seek>(
        stream: T,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let boxed_stream: Box<Box<dyn ReadStream>> = Box::new(Box::new(adapter));
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_from_stream_callback(
            Box::into_raw(boxed_stream) as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            add_callback,
            false,
            0,
        ));
        Ok(v)
    }

    /// Loads some named tensors from a stream to a given device
    ///
    /// The file format is the same as the one used by the PyTorch C++ API.
    pub fn load_multi_from_stream_with_device<T: Read + Seek>(
        stream: T,
        device: Device,
    ) -> Result<Vec<(String, Tensor)>, TchError> {
        let adapter = ReadSeekAdapter::new(stream);
        let boxed_stream: Box<Box<dyn ReadStream>> = Box::new(Box::new(adapter));
        let mut v: Vec<(String, Tensor)> = vec![];
        unsafe_torch_err!(at_load_from_stream_callback(
            Box::into_raw(boxed_stream) as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            add_callback,
            true,
            device.c_int(),
        ));
        Ok(v)
    }

    /// Returns a string representation for the tensor.
    ///
    /// The representation will contain all the tensor element hence may be huge for
    /// large tensors.
    pub fn to_string(&self, lw: i64) -> Result<String, TchError> {
        let s =
            unsafe_torch_err!(ptr_to_string(torch_sys::at_to_string(self.c_tensor, lw as c_int)));
        match s {
            None => Err(TchError::Kind("nullptr representation".to_string())),
            Some(s) => Ok(s),
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe_torch!(at_free(self.c_tensor))
    }
}

fn autocast_clear_cache() {
    unsafe_torch!(at_autocast_clear_cache())
}

fn autocast_decrement_nesting() -> isize {
    unsafe_torch!(at_autocast_decrement_nesting() as isize)
}

fn autocast_increment_nesting() -> isize {
    unsafe_torch!(at_autocast_increment_nesting() as isize)
}

fn autocast_is_enabled() -> bool {
    unsafe_torch!(at_autocast_is_enabled() != 0)
}

fn autocast_set_enabled(b: bool) -> bool {
    unsafe_torch!(at_autocast_set_enabled(i32::from(b)) != 0)
}

/// Runs a closure in mixed precision.
pub fn autocast<T, F>(enabled: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    if !Cuda::is_available() {
        return f();
    }

    // Check whether we are using CUDA.
    let prev = autocast_is_enabled();
    autocast_set_enabled(enabled);
    autocast_increment_nesting();

    let result = f();

    if autocast_decrement_nesting() == 0 {
        autocast_clear_cache();
    }
    autocast_set_enabled(prev);

    result
}

fn grad_set_enabled(b: bool) -> bool {
    unsafe_torch!(at_grad_set_enabled(i32::from(b)) != 0)
}

/// Runs a closure without keeping track of gradients.
pub fn no_grad<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let prev = grad_set_enabled(false);
    let result = f();
    let _false = grad_set_enabled(prev);
    result
}

/// Runs a closure explicitly keeping track of gradients, this could be
/// run within a no_grad closure for example.
pub fn with_grad<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let prev = grad_set_enabled(true);
    let result = f();
    let _false = grad_set_enabled(prev);
    result
}

/// A RAII guard that prevents gradient tracking until deallocated.
pub struct NoGradGuard {
    enabled: bool,
}

/// Disables gradient tracking, this will be enabled back when the
/// returned value gets deallocated.
/// Note that it is important to bind this to a name like `_guard`
/// and not to `_` as the latter would immediately drop the guard.
/// See <https://internals.rust-lang.org/t/pre-rfc-must-bind/12658/46>
/// for more details.
pub fn no_grad_guard() -> NoGradGuard {
    NoGradGuard { enabled: grad_set_enabled(false) }
}

impl std::convert::AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let _enabled = grad_set_enabled(self.enabled);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Reduction {
    /// Do not reduce.
    None,
    /// Mean of losses.
    Mean,
    /// Sum of losses.
    Sum,
    /// Escape hatch in case new options become available.
    Other(i64),
}

impl Reduction {
    // This has to stay in sync with
    // pytorch/aten/src/ATen/core/Reduction.h
    pub fn to_int(self) -> i64 {
        match self {
            Reduction::None => 0,
            Reduction::Mean => 1,
            Reduction::Sum => 2,
            Reduction::Other(i) => i,
        }
    }
}

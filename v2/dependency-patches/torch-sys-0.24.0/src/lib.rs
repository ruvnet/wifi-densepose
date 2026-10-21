pub mod cuda;
pub mod io;
#[cfg(feature = "python-extension")]
pub mod python;
mod traits;

use libc::{c_char, c_int, c_uchar, c_void, size_t};
pub use traits::{DoubleList, IntList, IntListOption};

#[repr(C)]
pub struct C_scalar {
    _private: [u8; 0],
}

extern "C" {
    pub fn ats_int(v: i64) -> *mut C_scalar;
    pub fn ats_float(v: f64) -> *mut C_scalar;
    pub fn ats_to_int(arg: *mut C_scalar) -> i64;
    pub fn ats_to_float(arg: *mut C_scalar) -> f64;
    pub fn ats_to_string(arg: *mut C_scalar) -> *mut c_char;
    pub fn ats_free(arg: *mut C_scalar);
}

#[repr(C)]
pub struct C_tensor {
    _private: [u8; 0],
}

extern "C" {
    pub fn at_new_tensor() -> *mut C_tensor;
    pub fn at_shallow_clone(arg: *mut C_tensor) -> *mut C_tensor;
    pub fn at_copy_(dst: *mut C_tensor, src: *mut C_tensor);
    pub fn at_data_ptr(arg: *mut C_tensor) -> *mut c_void;
    pub fn at_defined(arg: *mut C_tensor) -> c_int;
    pub fn at_is_sparse(arg: *mut C_tensor) -> c_int;
    pub fn at_is_mkldnn(arg: *mut C_tensor) -> c_int;
    pub fn at_is_contiguous(args: *mut C_tensor) -> c_int;
    pub fn at_backward(arg: *mut C_tensor, keep_graph: c_int, create_graph: c_int);
    pub fn at_print(arg: *mut C_tensor);
    pub fn at_to_string(arg: *mut C_tensor, line_size: c_int) -> *mut c_char;
    pub fn at_dim(arg: *mut C_tensor) -> size_t;
    pub fn at_get(arg: *mut C_tensor, index: c_int) -> *mut C_tensor;
    pub fn at_requires_grad(arg: *mut C_tensor) -> c_int;
    pub fn at_shape(arg: *mut C_tensor, sz: *mut i64);
    pub fn at_stride(arg: *mut C_tensor, sz: *mut i64);
    pub fn at_double_value_at_indexes(arg: *mut C_tensor, idx: *const i64, idx_len: c_int) -> f64;
    pub fn at_int64_value_at_indexes(arg: *mut C_tensor, idx: *const i64, idx_len: c_int) -> i64;
    pub fn at_get_num_interop_threads() -> c_int;
    pub fn at_get_num_threads() -> c_int;
    pub fn at_set_num_interop_threads(n_threads: c_int);
    pub fn at_set_num_threads(n_threads: c_int);
    pub fn at_set_qengine(qengine: c_int);
    pub fn at_free(arg: *mut C_tensor);
    pub fn at_run_backward(
        arg: *const *mut C_tensor,
        ntensors: c_int,
        inputs: *const *mut C_tensor,
        ninputs: c_int,
        outputs: *mut *mut C_tensor,
        keep_graph: c_int,
        create_graph: c_int,
    );
    pub fn at_copy_data(
        arg: *mut C_tensor,
        vs: *const c_void,
        numel: size_t,
        elt_size_in_bytes: size_t,
    );
    pub fn at_scalar_type(arg: *mut C_tensor) -> c_int;
    pub fn at__amp_non_finite_check_and_unscale(
        t: *mut C_tensor,
        found_inf: *mut C_tensor,
        inf_scale: *mut C_tensor,
    );
    pub fn at_autocast_clear_cache();
    pub fn at_autocast_decrement_nesting() -> c_int;
    pub fn at_autocast_increment_nesting() -> c_int;
    pub fn at_autocast_is_enabled() -> c_int;
    pub fn at_autocast_set_enabled(b: c_int) -> c_int;
    pub fn at_device(arg: *mut C_tensor) -> c_int;
    pub fn at_tensor_of_data(
        vs: *const c_void,
        dims: *const i64,
        ndims: size_t,
        elt_size_in_bytes: size_t,
        kind: c_int,
    ) -> *mut C_tensor;
    pub fn at_tensor_of_blob(
        vs: *const c_void,
        dims: *const i64,
        ndims: size_t,
        strides: *const i64,
        nstrides: size_t,
        kind: c_int,
        device: c_int,
    ) -> *mut C_tensor;
    pub fn at_grad_set_enabled(b: c_int) -> c_int;
    pub fn at_save(arg: *mut C_tensor, filename: *const c_char);
    pub fn at_save_to_stream(arg: *mut C_tensor, stream_ptr: *mut c_void);
    pub fn at_load(filename: *const c_char) -> *mut C_tensor;
    pub fn at_load_from_stream(stream_ptr: *mut c_void) -> *mut C_tensor;
    pub fn at_save_multi(
        args: *const *mut C_tensor,
        names: *const *const c_char,
        n: c_int,
        filename: *const c_char,
    );
    pub fn at_save_multi_to_stream(
        args: *const *mut C_tensor,
        names: *const *const c_char,
        n: c_int,
        stream_ptr: *mut c_void,
    );
    pub fn at_loadz_callback(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
    );
    pub fn at_loadz_callback_with_device(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
        device_id: c_int,
    );
    pub fn at_load_callback(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
    );
    pub fn at_load_callback_with_device(
        filename: *const c_char,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
        device_id: c_int,
    );
    pub fn at_load_from_stream_callback(
        stream_ptr: *mut c_void,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
        enable_device_id: bool,
        device_id: c_int,
    );

    pub fn at_manual_seed(seed: i64);
    pub fn at_set_graph_executor_optimize(b: bool);
    pub fn at_context_has_openmp() -> bool;
    pub fn at_context_has_mkl() -> bool;
    pub fn at_context_has_lapack() -> bool;
    pub fn at_context_has_mkldnn() -> bool;
    pub fn at_context_has_magma() -> bool;
    pub fn at_context_has_cuda() -> bool;
    pub fn at_context_has_cudart() -> bool;
    pub fn at_context_has_cusolver() -> bool;
    pub fn at_context_has_hip() -> bool;
    pub fn at_context_has_ipu() -> bool;
    pub fn at_context_has_xla() -> bool;
    pub fn at_context_has_lazy() -> bool;
    pub fn at_context_has_mps() -> bool;
    pub fn at_context_version_cudnn() -> i64;
    pub fn at_context_version_cudart() -> i64;
}

pub mod c_generated;

extern "C" {
    pub fn get_and_reset_last_err() -> *mut c_char;
}

#[repr(C)]
pub struct C_optimizer {
    _private: [u8; 0],
}

extern "C" {
    pub fn ato_adam(
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    ) -> *mut C_optimizer;
    pub fn ato_adamw(
        lr: f64,
        beta1: f64,
        beta2: f64,
        wd: f64,
        eps: f64,
        amsgrad: bool,
    ) -> *mut C_optimizer;
    pub fn ato_rms_prop(
        lr: f64,
        alpha: f64,
        eps: f64,
        wd: f64,
        momentum: f64,
        centered: c_int,
    ) -> *mut C_optimizer;
    pub fn ato_sgd(
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: c_int,
    ) -> *mut C_optimizer;
    pub fn ato_add_parameters(arg: *mut C_optimizer, ts: *mut C_tensor, group: size_t);
    pub fn ato_set_learning_rate(arg: *mut C_optimizer, lr: f64);
    pub fn ato_set_learning_rate_group(arg: *mut C_optimizer, group: size_t, lr: f64);
    pub fn ato_set_momentum(arg: *mut C_optimizer, momentum: f64);
    pub fn ato_set_momentum_group(arg: *mut C_optimizer, group: size_t, momentum: f64);
    pub fn ato_set_weight_decay(arg: *mut C_optimizer, weight_decay: f64);
    pub fn ato_set_weight_decay_group(arg: *mut C_optimizer, group: size_t, weight_decay: f64);
    pub fn ato_zero_grad(arg: *mut C_optimizer);
    pub fn ato_step(arg: *mut C_optimizer);
    pub fn ato_free(arg: *mut C_optimizer);
    pub fn at_save_image(arg: *mut C_tensor, filename: *const c_char) -> c_int;
    pub fn at_load_image(filename: *const c_char) -> *mut C_tensor;
    pub fn at_load_image_from_memory(
        img_data: *const c_uchar,
        img_data_len: size_t,
    ) -> *mut C_tensor;
    pub fn at_resize_image(arg: *mut C_tensor, out_w: c_int, out_h: c_int) -> *mut C_tensor;
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
pub struct CIValue {
    _private: [u8; 0],
}

#[repr(C)]
pub struct CModule_ {
    _private: [u8; 0],
}

extern "C" {
    // Constructors
    pub fn ati_none() -> *mut CIValue;
    pub fn ati_bool(b: c_int) -> *mut CIValue;
    pub fn ati_int(v: i64) -> *mut CIValue;
    pub fn ati_double(v: f64) -> *mut CIValue;
    pub fn ati_tensor(v: *mut C_tensor) -> *mut CIValue;
    pub fn ati_string(s: *const c_char) -> *mut CIValue;
    pub fn ati_tuple(v: *const *mut CIValue, n: c_int) -> *mut CIValue;
    pub fn ati_generic_list(v: *const *mut CIValue, n: c_int) -> *mut CIValue;
    pub fn ati_generic_dict(v: *const *mut CIValue, n: c_int) -> *mut CIValue;
    pub fn ati_int_list(v: *const i64, n: c_int) -> *mut CIValue;
    pub fn ati_double_list(v: *const f64, n: c_int) -> *mut CIValue;
    pub fn ati_bool_list(v: *const c_char, n: c_int) -> *mut CIValue;
    pub fn ati_string_list(v: *const *const c_char, n: c_int) -> *mut CIValue;
    pub fn ati_tensor_list(v: *const *mut C_tensor, n: c_int) -> *mut CIValue;
    pub fn ati_device(device_idx: c_int) -> *mut CIValue;

    // Type query
    pub fn ati_tag(arg: *mut CIValue) -> c_int;

    // Getters
    pub fn ati_to_int(arg: *mut CIValue) -> i64;
    pub fn ati_to_bool(arg: *mut CIValue) -> c_int;
    pub fn ati_to_double(arg: *mut CIValue) -> f64;
    pub fn ati_to_tensor(arg: *mut CIValue) -> *mut C_tensor;
    pub fn ati_length(arg: *mut CIValue) -> c_int;
    pub fn ati_tuple_length(arg: *mut CIValue) -> c_int;
    pub fn ati_to_tuple(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int);
    pub fn ati_to_generic_list(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int);
    pub fn ati_to_generic_dict(arg: *mut CIValue, outputs: *mut *mut CIValue, n: c_int);
    pub fn ati_to_int_list(arg: *mut CIValue, outputs: *mut i64, n: c_int);
    pub fn ati_to_double_list(arg: *mut CIValue, outputs: *mut f64, n: c_int);
    pub fn ati_to_bool_list(arg: *mut CIValue, outputs: *mut c_char, n: c_int);
    pub fn ati_to_tensor_list(arg: *mut CIValue, outputs: *mut *mut C_tensor, n: c_int);
    pub fn ati_to_string(arg: *mut CIValue) -> *mut c_char;

    pub fn ati_clone(arg: *mut CIValue) -> *mut CIValue;
    pub fn ati_free(arg: *mut CIValue);

    pub fn ati_object_method_(
        arg: *mut CIValue,
        method_name: *const c_char,
        args: *const *mut CIValue,
        n: c_int,
    ) -> *mut CIValue;

    pub fn ati_object_getattr_(arg: *mut CIValue, attr_name: *const c_char) -> *mut CIValue;

    pub fn atm_load(filename: *const c_char) -> *mut CModule_;
    pub fn atm_load_on_device(filename: *const c_char, device: c_int) -> *mut CModule_;
    pub fn atm_load_str(data: *const c_char, sz: size_t) -> *mut CModule_;
    pub fn atm_load_str_on_device(data: *const c_char, sz: size_t, device: c_int) -> *mut CModule_;
    pub fn atm_forward(m: *mut CModule_, args: *const *mut C_tensor, n: c_int) -> *mut C_tensor;
    pub fn atm_forward_(m: *mut CModule_, args: *const *mut CIValue, n: c_int) -> *mut CIValue;
    pub fn atm_method(
        m: *mut CModule_,
        method_name: *const c_char,
        args: *const *mut C_tensor,
        n: c_int,
    ) -> *mut C_tensor;
    pub fn atm_method_(
        m: *mut CModule_,
        method_name: *const c_char,
        args: *const *mut CIValue,
        n: c_int,
    ) -> *mut CIValue;
    pub fn atm_create_class_(
        m: *mut CModule_,
        clz_name: *const c_char,
        args: *const *mut CIValue,
        n: c_int,
    ) -> *mut CIValue;
    pub fn atm_eval(m: *mut CModule_);
    pub fn atm_train(m: *mut CModule_);
    pub fn atm_free(m: *mut CModule_);
    pub fn atm_to(m: *mut CModule_, device: c_int, kind: c_int, non_blocking: bool);
    pub fn atm_save(m: *mut CModule_, filename: *const c_char);
    pub fn atm_get_profiling_mode() -> c_int;
    pub fn atm_set_profiling_mode(profiling_mode: c_int);
    pub fn atm_fuser_cuda_set_enabled(enabled: bool);
    pub fn atm_fuser_cuda_is_enabled() -> bool;
    pub fn atm_named_parameters(
        m: *mut CModule_,
        data: *mut c_void,
        f: extern "C" fn(*mut c_void, name: *const c_char, t: *mut C_tensor),
    );
    pub fn atm_create_for_tracing(
        modl_name: *const c_char,
        inputs: *const *mut C_tensor,
        ninputs: c_int,
    ) -> *mut CModule_;
    pub fn atm_end_tracing(
        m: *mut CModule_,
        fn_name: *const c_char,
        outputs: *const *mut C_tensor,
        noutputs: c_int,
    );
    pub fn atm_set_tensor_expr_fuser_enabled(enabled: c_int);
    pub fn atm_get_tensor_expr_fuser_enabled() -> bool;
}

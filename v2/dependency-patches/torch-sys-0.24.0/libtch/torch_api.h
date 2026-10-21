#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
#include<torch/torch.h>
#include<stdexcept>
using namespace std;
extern thread_local char *torch_last_err;

extern "C" {
typedef torch::Tensor *tensor;
typedef torch::Scalar *scalar;
typedef torch::optim::Optimizer *optimizer;
typedef torch::jit::script::Module *module;
typedef torch::jit::IValue *ivalue;
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
      torch_last_err = strdup(e.what()); \
  }
#else
typedef void *tensor;
typedef void *optimizer;
typedef void *scalar;
typedef void *module;
typedef void *ivalue;
#endif

char *get_and_reset_last_err(); // thread-local
void at_manual_seed(int64_t);
tensor at_new_tensor();
tensor at_tensor_of_blob(void *data, int64_t *dims, size_t ndims, int64_t *strides, size_t nstrides, int type, int device);
tensor at_tensor_of_data(void *vs, int64_t *dims, size_t ndims, size_t element_size_in_bytes, int type);
void at_copy_data(tensor tensor, void *vs, size_t numel, size_t element_size_in_bytes);
tensor at_shallow_clone(tensor);

void *at_data_ptr(tensor);
int at_defined(tensor);
int at_is_mkldnn(tensor);
int at_is_sparse(tensor);
int at_is_contiguous(tensor);
int at_device(tensor);
size_t at_dim(tensor);
void at_shape(tensor, int64_t *);
void at_stride(tensor, int64_t *);
int at_scalar_type(tensor);

void at__amp_non_finite_check_and_unscale(tensor, tensor, tensor);

void at_autocast_clear_cache();
int at_autocast_decrement_nesting();
int at_autocast_increment_nesting();
bool at_autocast_is_enabled();
bool at_autocast_set_enabled(bool b);

void at_backward(tensor, int, int);
int at_requires_grad(tensor);
int at_grad_set_enabled(int);

tensor at_get(tensor, int index);
void at_fill_double(tensor, double);
void at_fill_int64(tensor, int64_t);

double at_double_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
int64_t at_int64_value_at_indexes(tensor, int64_t *indexes, int indexes_len);
void at_set_double_value_at_indexes(tensor, int *indexes, int indexes_len, double v);
void at_set_int64_value_at_indexes(tensor, int *indexes, int indexes_len, int64_t v);

void at_copy_(tensor dst, tensor src);

void at_print(tensor);
char *at_to_string(tensor, int line_size);
void at_save(tensor, char *filename);
void at_save_to_stream(tensor t, void *stream_ptr);
tensor at_load(char *filename);
tensor at_load_from_stream(void *stream_ptr);
tensor at_load_image(char *filename);
tensor at_load_image_from_memory(unsigned char *img_data, size_t img_size);
int at_save_image(tensor, char *filename);
tensor at_resize_image(tensor, int w, int h);

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
void at_save_multi_to_stream(tensor *tensors, char **tensor_names, int ntensors, void *stream_ptr);
/* [at_load_multi] takes as input an array of nullptr for [tensors]. */
void at_load_multi(tensor *tensors, char **tensor_names, int ntensors, char *filename);
/* [at_load_multi_] takes as input an array of allocation [tensors]. */
void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors, char *filename);

void at_loadz_callback(char *filename, void *data, void (*f)(void *, char *, tensor));
void at_loadz_callback_with_device(char *filename, void *data, void (*f)(void *, char *, tensor), int device_id);
void at_load_callback(char *filename, void *data, void (*f)(void *, char *, tensor));
void at_load_callback_with_device(char *filename, void *data, void (*f)(void *, char *, tensor), int device_id);
void at_load_from_stream_callback(void *stream_ptr, void *data, void (*f)(void *, char *, tensor), bool enable_device_id, int device_id);

int at_get_num_interop_threads();

int at_get_num_threads();

void at_set_num_interop_threads(int n_threads);

void at_set_num_threads(int n_threads);

void at_set_qengine(int qengine);

void at_free(tensor);

void at_run_backward(tensor *tensors,
                      int ntensors,
                      tensor *inputs,
                      int ninputs,
                      tensor *outputs,
                      int keep_graph,
                      int create_graph);

optimizer ato_adam(double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay,
                   double eps,
                   bool amsgrad);
optimizer ato_adamw(double learning_rate,
                   double beta1,
                   double beta2,
                   double weight_decay,
                   double eps,
                   bool amsgrad);
optimizer ato_rms_prop(double learning_rate,
                       double alpha,
                       double eps,
                       double weight_decay,
                       double momentum,
                       int centered);
optimizer ato_sgd(double learning_rate,
                  double momentum,
                  double dampening,
                  double weight_decay,
                  int nesterov);
void ato_add_parameters(optimizer, tensor, size_t group);
void ato_set_learning_rate(optimizer, double learning_rate);
void ato_set_momentum(optimizer, double momentum);
void ato_set_learning_rate_group(optimizer, size_t group, double learning_rate);
void ato_set_momentum_group(optimizer, size_t group, double momentum);
void ato_set_weight_decay(optimizer t, double weight_decay);
void ato_set_weight_decay_group(optimizer t, size_t group, double weight_decay);
void ato_zero_grad(optimizer);
void ato_step(optimizer);
void ato_free(optimizer);

scalar ats_int(int64_t);
scalar ats_float(double);
int64_t ats_to_int(scalar);
double ats_to_float(scalar);
char *ats_to_string(scalar);
void ats_free(scalar);

bool at_context_has_openmp();
bool at_context_has_mkl();
bool at_context_has_lapack();
bool at_context_has_mkldnn();
bool at_context_has_magma();
bool at_context_has_cuda();
bool at_context_has_cudart();
bool at_context_has_cudnn();
int64_t at_context_version_cudnn();
int64_t at_context_version_cudart();
bool at_context_has_cusolver();
bool at_context_has_hip();
bool at_context_has_ipu();
bool at_context_has_xla();
bool at_context_has_lazy();
bool at_context_has_mps();


/// Returns the number of CUDA devices available.
int atc_cuda_device_count();

/// Returns true if at least one CUDA device is available.
int atc_cuda_is_available();

/// Returns true if CUDA is available, and CuDNN is available.
int atc_cudnn_is_available();

/// Sets the seed for the current GPU.
void atc_manual_seed(uint64_t seed);

/// Sets the seed for all available GPUs.
void atc_manual_seed_all(uint64_t seed);

/// Waits for all kernels in all streams on a CUDA device to complete.
void atc_synchronize(int64_t device_index);


int atc_user_enabled_cudnn();
void atc_set_user_enabled_cudnn(int b);
void atc_set_benchmark_cudnn(int b);

module atm_load(char *);
module atm_load_on_device(char *, int device);
module atm_load_str(char *, size_t sz);
module atm_load_str_on_device(char *, size_t sz, int device);
tensor atm_forward(module, tensor *tensors, int ntensors);
ivalue atm_forward_(module,
                    ivalue *ivalues,
                    int nivalues);
tensor atm_method(module,
                  char *method_name,
                  tensor *tensors,
                  int ntensors);
ivalue atm_method_(module,
                   char *method_name,
                   ivalue *ivalues,
                   int nivalues);
ivalue atm_create_class_(module,
                   char *clz_name, 
                   ivalue *ivalues, 
                   int nivalues);
void atm_eval(module);
void atm_train(module);
void atm_free(module);
void atm_to(module m, int device, int dtype, bool non_blocking);
void atm_save(module m, char*);
int atm_get_profiling_mode();
void atm_set_profiling_mode(int);
void atm_fuser_cuda_set_enabled(bool);
bool atm_fuser_cuda_is_enabled();
void atm_named_parameters(module, void *data, void (*f)(void *, char *, tensor));

// This function has to be followed by a call to atm_end_tracing.
module atm_create_for_tracing(char *modl_name, tensor *inputs, int ninputs);
void atm_end_tracing(module m, char *fn_name, tensor *outputs, int noutputs);

ivalue ati_none();
ivalue ati_tensor(tensor);
ivalue ati_int(int64_t);
ivalue ati_double(double);
ivalue ati_bool(int);
ivalue ati_string(char *);
ivalue ati_tuple(ivalue *, int);
ivalue ati_generic_list(ivalue *, int);
ivalue ati_generic_dict(ivalue *, int);
ivalue ati_int_list(int64_t *, int);
ivalue ati_double_list(double *, int);
ivalue ati_bool_list(char *, int);
ivalue ati_string_list(char **, int);
ivalue ati_tensor_list(tensor *, int);
ivalue ati_device(int);

tensor ati_to_tensor(ivalue);
int64_t ati_to_int(ivalue);
double ati_to_double(ivalue);
char *ati_to_string(ivalue);
int ati_to_bool(ivalue);
int ati_length(ivalue);
int ati_tuple_length(ivalue);
void ati_to_tuple(ivalue, ivalue *, int);
void ati_to_generic_list(ivalue, ivalue *, int);
void ati_to_generic_dict(ivalue, ivalue *, int);
void ati_to_int_list(ivalue, int64_t *, int);
void ati_to_double_list(ivalue, double *, int);
void ati_to_bool_list(ivalue, char *, int);
void ati_to_tensor_list(ivalue, tensor *, int);

void atm_set_tensor_expr_fuser_enabled(int);
bool atm_get_tensor_expr_fuser_enabled();

int ati_tag(ivalue);

ivalue ati_object_method_(ivalue i, char *method_name, ivalue *ivalues, int nivalues);
ivalue ati_object_getattr_(ivalue i, char *attr_name);

ivalue ati_clone(ivalue);
void ati_free(ivalue);

/// Enables or disables the graph executor optimizer for the current thread.
void at_set_graph_executor_optimize(bool);

// for internal use
bool tch_write_stream_destructor(void *stream_ptr);
bool tch_write_stream_write(void *stream_ptr, const uint8_t *buf, size_t size, size_t *out_size);
bool tch_read_stream_destructor(void *stream_ptr);
bool tch_read_stream_stream_position(void *stream_ptr, uint64_t *pos);
bool tch_read_stream_seek_start(void *stream_ptr, uint64_t pos, uint64_t *new_pos);
bool tch_read_stream_seek_end(void *stream_ptr, int64_t pos, uint64_t *new_pos);
bool tch_read_stream_read(void *stream_ptr, uint8_t *buf, size_t size, size_t *new_pos);

#ifdef __cplusplus
};

std::vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len);
at::Device device_of_int(int d);
c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs, int len);
#endif
#endif

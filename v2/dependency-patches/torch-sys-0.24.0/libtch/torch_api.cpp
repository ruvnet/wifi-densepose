#include "torch_api.h"
#include <ATen/autocast_mode.h>
#include <stdexcept>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/mobile/import_data.h>
#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

thread_local char *torch_last_err = nullptr;

char *get_and_reset_last_err() {
  char *tmp = torch_last_err;
  torch_last_err = nullptr;
  return tmp;
}

void at_manual_seed(int64_t seed) { torch::manual_seed(seed); }

vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i)
    result.push_back(*(vs[i]));
  return result;
}

c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs,
                                                             int len) {
  vector<c10::optional<torch::Tensor>> result;
  for (int i = 0; i < len; ++i) {
    result.push_back(vs[i] != nullptr ? c10::optional<torch::Tensor>(*(vs[i]))
                                      : c10::nullopt);
  }
  return c10::List<c10::optional<torch::Tensor>>(result);
}

at::Device device_of_int(int d) {
  if (d == -3)
    return at::Device(at::kVulkan);
  if (d == -2)
    return at::Device(at::kMPS);
  if (d < 0)
    return at::Device(at::kCPU);
  return at::Device(at::kCUDA, /*index=*/d);
}
tensor at_new_tensor() {
  PROTECT(return new torch::Tensor();)
  return nullptr;
}

tensor at_tensor_of_blob(void *data, int64_t *dims, size_t ndims,
                         int64_t *strides, size_t nstrides, int type,
                         int device) {
  PROTECT(
      at::TensorOptions blobOptions = at::TensorOptions()
                                          .device(device_of_int(device))
                                          .dtype(torch::ScalarType(type));
      if (nstrides == 0) {
        return new torch::Tensor(
            torch::for_blob(data, torch::IntArrayRef(dims, ndims))
                .options(blobOptions)
                .make_tensor());
      } else {
        return new torch::Tensor(torch::from_blob(
            data, torch::IntArrayRef(dims, ndims),
            torch::IntArrayRef(strides, nstrides), blobOptions));
      })

  return nullptr;
}

tensor at_tensor_of_data(void *vs, int64_t *dims, size_t ndims,
                         size_t element_size_in_bytes, int type) {
  PROTECT(
      torch::Tensor tensor = torch::zeros(torch::IntArrayRef(dims, ndims),
                                          torch::ScalarType(type));
      if ((int64_t)element_size_in_bytes != tensor.element_size()) throw std::
          invalid_argument("incoherent element sizes in bytes");
      void *tensor_data = tensor.data_ptr();
      memcpy(tensor_data, vs, tensor.numel() * element_size_in_bytes);
      return new torch::Tensor(tensor);)
  return nullptr;
}

void at_copy_data(tensor tensor, void *vs, size_t numel,
                  size_t elt_size_in_bytes) {
  PROTECT(
      if ((int64_t)elt_size_in_bytes != tensor->element_size()) throw std::
          invalid_argument("incoherent element sizes in bytes");
      if ((int64_t)numel > tensor->numel()) throw std::invalid_argument(
          "target numel is larger than tensor numel");
      if (tensor->device().type() != at::kCPU) {
        torch::Tensor tmp_tensor = tensor->to(at::kCPU).contiguous();
        void *tensor_data = tmp_tensor.data_ptr();
        memcpy(vs, tensor_data, numel * elt_size_in_bytes);
      } else {
        auto tmp_tensor = tensor->contiguous();
        void *tensor_data = tmp_tensor.data_ptr();
        memcpy(vs, tensor_data, numel * elt_size_in_bytes);
      })
}

tensor at_shallow_clone(tensor t) {
  PROTECT(return new torch::Tensor(*t);)
  return nullptr;
}

void *at_data_ptr(tensor t) {
  PROTECT(return t->data_ptr();)
  return nullptr;
}

int at_defined(tensor t) {
  PROTECT(return t->defined();)
  return -1;
}

int at_is_mkldnn(tensor t) {
  PROTECT(return t->is_mkldnn();)
  return -1;
}

int at_is_sparse(tensor t) {
  PROTECT(return t->is_sparse();)
  return -1;
}

int at_is_contiguous(tensor t) {
  PROTECT(return t->is_contiguous();)
  return -1;
}

size_t at_dim(tensor t) {
  PROTECT(return t->dim();)
  return -1;
}

void at_shape(tensor t, int64_t *dims) {
  PROTECT(int i = 0; for (int64_t dim : t->sizes()) dims[i++] = dim;)
}

void at_stride(tensor t, int64_t *dims) {
  PROTECT(int i = 0; for (int64_t dim : t->strides()) dims[i++] = dim;)
}

int at_scalar_type(tensor t) {
  PROTECT(return static_cast<int>(t->scalar_type());)
  return -1;
}

void at__amp_non_finite_check_and_unscale(tensor t, tensor found_inf,
                                          tensor inf_scale) {
  PROTECT(at::_amp_foreach_non_finite_check_and_unscale_(*t, *found_inf,
                                                         *inf_scale);)
}

void at_autocast_clear_cache() { at::autocast::clear_cache(); }

int at_autocast_decrement_nesting() {
  PROTECT(return at::autocast::decrement_nesting();)
  return -1;
}

int at_autocast_increment_nesting() {
  PROTECT(return at::autocast::increment_nesting();)
  return -1;
}

bool at_autocast_is_enabled() {
  PROTECT(return at::autocast::is_enabled();)
  return -1;
}

bool at_autocast_set_enabled(bool b) {
  PROTECT(bool is_enabled = at::autocast::is_enabled();
          at::autocast::set_enabled(b); return is_enabled;)
  return -1;
}

int at_device(tensor t) {
  PROTECT(auto device = t->device(); if (device.type() == at::kCPU) return -1;
          if (device.type() == at::kMPS) return -2;
          if (device.type() == at::kVulkan) return -3;
          if (device.type() == at::kCUDA) return device.index();)
  return -2;
}

void at_backward(tensor t, int keep_graph, int create_graph) {
  PROTECT(t->backward({}, keep_graph, create_graph);)
}

int at_requires_grad(tensor t) {
  PROTECT(return t->requires_grad();)
  return -1;
}

int at_grad_set_enabled(int b) {
  PROTECT(bool is_enabled = torch::autograd::GradMode::is_enabled();
          torch::autograd::GradMode::set_enabled(b); return is_enabled;)
  return -1;
}

tensor at_get(tensor t, int index) {
  PROTECT(return new torch::Tensor((*t)[index]);)
  return nullptr;
}

template <typename T>
T at_value_at_indexes(tensor t, int64_t *indexes, int indexes_len) {
  PROTECT(torch::Tensor tensor = *t; for (int i = 0; i < indexes_len; ++i) {
    tensor = tensor[indexes[i]];
  } return tensor.item<T>();)
  return T();
}

double at_double_value_at_indexes(tensor t, int64_t *indexes, int indexes_len) {
  return at_value_at_indexes<double>(t, indexes, indexes_len);
}

int64_t at_int64_value_at_indexes(tensor t, int64_t *indexes, int indexes_len) {
  return at_value_at_indexes<int64_t>(t, indexes, indexes_len);
}

template <typename T>
void at_set_value_at_indexes(tensor t, int *indexes, int indexes_len, T v) {
  PROTECT(torch::Tensor tensor = *t; for (int i = 0; i < indexes_len; ++i) {
    tensor = tensor[indexes[i]];
  } tensor.fill_(v);)
}

void at_set_double_value_at_indexes(tensor t, int *indexes, int indexes_len,
                                    double v) {
  at_set_value_at_indexes<double>(t, indexes, indexes_len, v);
}

void at_set_int64_value_at_indexes(tensor t, int *indexes, int indexes_len,
                                   int64_t v) {
  at_set_value_at_indexes<int64_t>(t, indexes, indexes_len, v);
}

void at_fill_double(tensor t, double v) { PROTECT(t->fill_(v);) }

void at_fill_int64(tensor t, int64_t v) { PROTECT(t->fill_(v);) }

void at_print(tensor t) {
  PROTECT(torch::Tensor *tensor = (torch::Tensor *)t; cout << *tensor << endl;)
}

char *at_to_string(tensor t, int line_size) {
  PROTECT(std::ostringstream oss; torch::print(oss, *t, line_size);
          return strdup(oss.str().c_str());)
  return nullptr;
}

void at_copy_(tensor dst, tensor src) { PROTECT(dst->copy_(*src);) }

namespace {
class WriteStreamAdapter {
private:
  void *_stream_ptr;

public:
  WriteStreamAdapter(void *stream_ptr) : _stream_ptr(stream_ptr) {}

  ~WriteStreamAdapter() { tch_write_stream_destructor(_stream_ptr); }

  size_t save(const void *data, size_t size) {
    size_t out_size = 0;
    if (!tch_write_stream_write(_stream_ptr, static_cast<const uint8_t *>(data),
                                size, &out_size)) {
      throw std::ios_base::failure("tch_write_stream_write has returned error");
    }
    return out_size;
  }
};

class ReadStreamAdapter : public caffe2::serialize::ReadAdapterInterface {
private:
  void *_stream_ptr;

public:
  ReadStreamAdapter(void *stream_ptr) : _stream_ptr(stream_ptr) {}

  virtual ~ReadStreamAdapter() { tch_read_stream_destructor(_stream_ptr); }

  virtual size_t size() const {
    uint64_t current = 0;
    if (!tch_read_stream_stream_position(_stream_ptr, &current)) {
      throw std::ios_base::failure(
          "tch_read_stream_stream_position has returned error");
    }

    uint64_t size = 0;
    if (!tch_read_stream_seek_end(_stream_ptr, 0, &size)) {
      throw std::ios_base::failure(
          "tch_read_stream_seek_end has returned error");
    }

    uint64_t dummy = 0;
    if (!tch_read_stream_seek_start(_stream_ptr, current, &dummy)) {
      throw std::ios_base::failure(
          "tch_read_stream_seek_start has returned error");
    }

    return size;
  }

  virtual size_t read(uint64_t pos, void *buf, size_t n,
                      const char *what) const {
    uint64_t dummy = 0;
    if (!tch_read_stream_seek_start(_stream_ptr, pos, &dummy)) {
      throw std::ios_base::failure(
          "tch_read_stream_seek_start has returned error");
    }

    size_t size = 0;
    if (!tch_read_stream_read(_stream_ptr, static_cast<uint8_t *>(buf), n,
                              &size)) {
      throw std::ios_base::failure("tch_read_stream_read has returned error");
    }

    return size;
  }
};
} // namespace

void at_save(tensor t, char *filename) { PROTECT(torch::save(*t, filename);) }

void at_save_to_stream(tensor t, void *stream_ptr) {
  PROTECT(auto adapter = std::shared_ptr<WriteStreamAdapter>(
              new WriteStreamAdapter(stream_ptr));
          torch::save(*t, [adapter](const void *data, size_t size) {
            return adapter->save(data, size);
          });)
}

void at_save_multi(tensor *tensors, char **tensor_names, int ntensors,
                   char *filename) {
  PROTECT(torch::serialize::OutputArchive archive;
          for (int i = 0; i < ntensors; ++i) archive.write(
              std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/false);
          archive.save_to(filename);)
}

void at_save_multi_to_stream(tensor *tensors, char **tensor_names, int ntensors,
                             void *stream_ptr) {
  PROTECT(auto adapter = std::shared_ptr<WriteStreamAdapter>(
              new WriteStreamAdapter(stream_ptr));
          torch::serialize::OutputArchive archive;
          for (int i = 0; i < ntensors; ++i) archive.write(
              std::string(tensor_names[i]), *(tensors[i]), /* buffer=*/false);
          archive.save_to([adapter](const void *data, size_t size) {
            return adapter->save(data, size);
          });)
}

void at_load_multi(tensor *tensors, char **tensor_names, int ntensors,
                   char *filename) {
  PROTECT(torch::serialize::InputArchive archive;
          archive.load_from(std::string(filename));
          vector<torch::Tensor> ts(ntensors);
          for (int i = 0; i < ntensors; ++i)
              archive.read(std::string(tensor_names[i]), ts[i]);
          // Only allocate the new tensor now so that if there is an exception
          // raised during [read], no memory has to be freed.
          for (int i = 0; i < ntensors; ++i) tensors[i] =
              new torch::Tensor(ts[i]);)
}

void at_loadz_callback(char *filename, void *data,
                       void (*f)(void *, char *, tensor)) {
  PROTECT(auto params = torch::jit::_load_parameters(filename);
          for (const auto &p : params) {
            f(data, (char *)p.first.c_str(), new torch::Tensor(p.second));
          })
}

void at_loadz_callback_with_device(char *filename, void *data,
                                   void (*f)(void *, char *, tensor),
                                   int device_id) {
  PROTECT(auto params =
              torch::jit::_load_parameters(filename, device_of_int(device_id));
          for (const auto &p : params) {
            f(data, (char *)p.first.c_str(), new torch::Tensor(p.second));
          })
}

void at_load_callback(char *filename, void *data,
                      void (*f)(void *, char *, tensor)) {
  PROTECT(auto module = torch::jit::load(filename);
          for (const auto &p : module.named_parameters()) {
            auto v = p.value;
            f(data, (char *)p.name.c_str(), new torch::Tensor(v));
          })
}

void at_load_callback_with_device(char *filename, void *data,
                                  void (*f)(void *, char *, tensor),
                                  int device_id) {
  PROTECT(auto module = torch::jit::load(filename, device_of_int(device_id));
          for (const auto &p : module.named_parameters()) {
            auto v = p.value;
            f(data, (char *)p.name.c_str(), new torch::Tensor(v));
          })
}

void at_load_from_stream_callback(void *stream_ptr, void *data,
                                  void (*f)(void *, char *, tensor),
                                  bool enable_device_id, int device_id) {
  PROTECT(
      auto adapter = std::shared_ptr<caffe2::serialize::ReadAdapterInterface>(
          new ReadStreamAdapter(stream_ptr));
      auto module = enable_device_id
                        ? torch::jit::load(adapter, device_of_int(device_id))
                        : torch::jit::load(adapter);
      for (const auto &p : module.named_parameters()) {
        auto v = p.value;
        f(data, (char *)p.name.c_str(), new torch::Tensor(v));
      })
}

void at_load_multi_(tensor *tensors, char **tensor_names, int ntensors,
                    char *filename) {
  PROTECT(torch::NoGradGuard no_grad; torch::serialize::InputArchive archive;
          archive.load_from(std::string(filename));
          for (int i = 0; i < ntensors; ++i) {
            if (tensors[i]->device().type() == at::kCPU)
              archive.read(std::string(tensor_names[i]), *(tensors[i]));
            else {
              torch::Tensor tmp_tensor =
                  torch::empty_like(*(tensors[i]), at::device(at::kCPU));
              archive.read(std::string(tensor_names[i]), tmp_tensor);
              tensors[i]->copy_(tmp_tensor);
            }
          })
}

tensor at_load(char *filename) {
  PROTECT(torch::Tensor tensor; torch::load(tensor, filename);
          return new torch::Tensor(tensor);)
  return nullptr;
}

tensor at_load_from_stream(void *stream_ptr) {
  PROTECT(torch::NoGradGuard no_grad; torch::Tensor tensor;
          auto adapter =
              std::shared_ptr<caffe2::serialize::ReadAdapterInterface>(
                  new ReadStreamAdapter(stream_ptr));
          torch::load(
              tensor,
              [adapter](uint64_t pos, void *buf, size_t nbytes) {
                return adapter->read(pos, buf, nbytes, "tensor");
              },
              [adapter]() { return adapter->size(); });
          return new torch::Tensor(tensor);)
  return nullptr;
}

tensor at_load_image(char *filename) {
  PROTECT(
      int w = -1; int h = -1; int c = -1;
      void *data = stbi_load(filename, &w, &h, &c, 3);
      if (data == nullptr) throw std::invalid_argument(stbi_failure_reason());
      torch::Tensor tensor = torch::zeros({h, w, 3}, at::ScalarType::Byte);
      memcpy(tensor.data_ptr(), data, h * w * 3); free(data);
      return new torch::Tensor(tensor);)
  return nullptr;
}

tensor at_load_image_from_memory(unsigned char *img_data, size_t img_size) {
  PROTECT(
      int w = -1; int h = -1; int c = -1;
      void *data = stbi_load_from_memory(img_data, img_size, &w, &h, &c, 3);
      if (data == nullptr) throw std::invalid_argument(stbi_failure_reason());
      torch::Tensor tensor = torch::zeros({h, w, 3}, at::ScalarType::Byte);
      memcpy(tensor.data_ptr(), data, h * w * 3); free(data);
      return new torch::Tensor(tensor);)
  return nullptr;
}

bool ends_with(const char *str, const char *suffix) {
  int suffix_len = strlen(suffix);
  int str_len = strlen(str);
  if (str_len < suffix_len)
    return false;
  for (int i = 1; i <= suffix_len; ++i)
    if (str[str_len - i] != suffix[suffix_len - i])
      return false;
  return true;
}

int at_save_image(tensor tensor, char *filename) {
  PROTECT(auto sizes = tensor->sizes();
          if (tensor->device().type() != at::kCPU) throw std::invalid_argument(
              "the input tensor has to be on cpu");
          if (sizes.size() != 3) throw std::invalid_argument(
              "invalid number of dimensions, should be 3");
          int h = sizes[0]; int w = sizes[1]; int c = sizes[2];
          auto tmp_tensor = tensor->contiguous();
          void *tensor_data = tmp_tensor.data_ptr();
          if (ends_with(filename, ".jpg")) return stbi_write_jpg(
              filename, w, h, c, tensor_data, 90);
          if (ends_with(filename, ".bmp")) return stbi_write_bmp(
              filename, w, h, c, tensor_data);
          if (ends_with(filename, ".tga")) return stbi_write_tga(
              filename, w, h, c, tensor_data);
          return stbi_write_png(filename, w, h, c, tensor_data, 0);)
  return -1;
}

int at_get_num_interop_threads() {
  PROTECT(return at::get_num_interop_threads();)
  return -1;
}

int at_get_num_threads() {
  PROTECT(return at::get_num_threads();)
  return -1;
}

void at_set_num_interop_threads(int n_threads) {
  PROTECT(at::set_num_interop_threads(n_threads);)
}

void at_set_num_threads(int n_threads) {
  PROTECT(at::set_num_threads(n_threads);)
}

void at_set_qengine(int qengine_id) {
  PROTECT(
      at::QEngine qengine = at::QEngine::NoQEngine; switch (qengine_id) {
        case 0:
          break;
        case 1:
          qengine = at::QEngine::FBGEMM;
          break;
        case 2:
          qengine = at::QEngine::QNNPACK;
          break;
      } auto qengines = at::globalContext().supportedQEngines();
      if (std::find(qengines.begin(), qengines.end(), qengine) !=
          qengines.end()) {
        at::globalContext().setQEngine(qengine);
      } else throw std::invalid_argument("unsupported qengine");)
}

tensor at_resize_image(tensor tensor, int out_w, int out_h) {
  PROTECT(
      auto sizes = tensor->sizes();
      if (tensor->device().type() != at::kCPU) throw std::invalid_argument(
          "the input tensor has to be on cpu");
      if (sizes.size() != 3) throw std::invalid_argument(
          "invalid number of dimensions, should be 3");
      int h = sizes[0]; int w = sizes[1]; int c = sizes[2];
      auto tmp_tensor = tensor->contiguous();
      const unsigned char *tensor_data = (unsigned char *)tmp_tensor.data_ptr();
      torch::Tensor out = torch::zeros({out_h, out_w, c}, at::ScalarType::Byte);
      stbir_resize_uint8(tensor_data, w, h, 0, (unsigned char *)out.data_ptr(),
                         out_w, out_h, 0, c);
      return new torch::Tensor(out);)
  return nullptr;
}

void at_free(tensor t) { delete (t); }

void at_run_backward(tensor *tensors, int ntensors, tensor *inputs, int ninputs,
                     tensor *outputs, int keep_graph, int create_graph) {
  PROTECT(
      vector<torch::autograd::Edge> roots;
      for (int i = 0; i < ntensors; ++i)
          roots.push_back(torch::autograd::impl::gradient_edge(*tensors[i]));

      vector<torch::autograd::Edge> inputs_; for (int i = 0; i < ninputs; ++i) {
        if (!inputs[i]->requires_grad())
          throw std::invalid_argument(
              "one of the input tensor does not use set_requires_grad");
        inputs_.push_back(torch::autograd::impl::gradient_edge(*inputs[i]));
      }

      vector<torch::autograd::Variable>
          grads;
      for (int i = 0; i < ntensors; ++i)
          grads.push_back(torch::ones_like(*tensors[i]));

      auto vl = torch::autograd::Engine::get_default_engine().execute(
          roots, grads, keep_graph, create_graph, false, inputs_);
      for (int i = 0; i < ninputs; ++i) {
        outputs[i] = static_cast<tensor>(new torch::autograd::Variable(vl[i]));
      })
}

optimizer ato_adam(double learning_rate, double beta1, double beta2,
                   double weight_decay, double eps, bool amsgrad) {
  PROTECT(auto options = torch::optim::AdamOptions(learning_rate)
                             .betas(std::tuple<double, double>(beta1, beta2))
                             .weight_decay(weight_decay)
                             .eps(eps)
                             .amsgrad(amsgrad);
          return new torch::optim::Adam(vector<torch::Tensor>(), options);)
  return nullptr;
}

optimizer ato_adamw(double learning_rate, double beta1, double beta2,
                    double weight_decay, double eps, bool amsgrad) {
  PROTECT(auto options = torch::optim::AdamWOptions(learning_rate)
                             .betas(std::tuple<double, double>(beta1, beta2))
                             .weight_decay(weight_decay)
                             .eps(eps)
                             .amsgrad(amsgrad);
          return new torch::optim::AdamW(vector<torch::Tensor>(), options);)
  return nullptr;
}

optimizer ato_rms_prop(double learning_rate, double alpha, double eps,
                       double weight_decay, double momentum, int centered) {
  PROTECT(auto options = torch::optim::RMSpropOptions(learning_rate)
                             .alpha(alpha)
                             .eps(eps)
                             .weight_decay(weight_decay)
                             .momentum(momentum)
                             .centered(centered != 0);
          return new torch::optim::RMSprop(vector<torch::Tensor>(), options);)
  return nullptr;
}

optimizer ato_sgd(double learning_rate, double momentum, double dampening,
                  double weight_decay, int nesterov) {
  PROTECT(auto options = torch::optim::SGDOptions(learning_rate)
                             .momentum(momentum)
                             .dampening(dampening)
                             .weight_decay(weight_decay)
                             .nesterov(nesterov);
          return new torch::optim::SGD(vector<torch::Tensor>(), options);)
  return nullptr;
}

void ato_add_parameters(optimizer t, tensor tensor, size_t group) {
  PROTECT(auto &groups = t->param_groups(); while (groups.size() <= group) {
    groups.push_back(
        torch::optim::OptimizerParamGroup({}, t->defaults().clone()));
  } groups[group]
                                                .params()
                                                .push_back(*tensor);)
}

template <class T> void set_lr(optimizer t, double learning_rate) {
  torch::optim::OptimizerOptions *d = &(t->defaults());
  if (auto p = dynamic_cast<T *>(d)) {
    p->lr(learning_rate);
    for (auto &param_group : t->param_groups()) {
      torch::optim::OptimizerOptions *d = &(param_group.options());
      if (auto p2 = dynamic_cast<T *>(d)) {
        p2->lr(learning_rate);
      } else
        throw std::invalid_argument("unexpected param group type");
    }
  }
}

void ato_set_learning_rate(optimizer t, double learning_rate) {
  PROTECT(set_lr<torch::optim::AdamOptions>(t, learning_rate);
          set_lr<torch::optim::AdamWOptions>(t, learning_rate);
          set_lr<torch::optim::RMSpropOptions>(t, learning_rate);
          set_lr<torch::optim::SGDOptions>(t, learning_rate);)
}

template <class T>
void set_lr_group(optimizer t, size_t group, double learning_rate) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions *d = &(param_group.options());
  if (auto p = dynamic_cast<T *>(d)) {
    p->lr(learning_rate);
  }
}

void ato_set_learning_rate_group(optimizer t, size_t group,
                                 double learning_rate) {
  PROTECT(set_lr_group<torch::optim::AdamOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::AdamWOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::RMSpropOptions>(t, group, learning_rate);
          set_lr_group<torch::optim::SGDOptions>(t, group, learning_rate);)
}

void ato_set_momentum(optimizer t, double momentum) {
  PROTECT(
      torch::optim::OptimizerOptions *d = &(t->defaults());
      if (auto adam = dynamic_cast<torch::optim::AdamOptions *>(d)) {
        auto betas = adam->betas();
        adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto adam2 = dynamic_cast<torch::optim::AdamOptions *>(d)) {
            adam2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions *>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto adamw2 = dynamic_cast<torch::optim::AdamWOptions *>(d)) {
            adamw2->betas(std::tuple<double, double>(momentum, get<1>(betas)));
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions *>(d)) {
        rms->momentum(momentum);
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto rms2 = dynamic_cast<torch::optim::RMSpropOptions *>(d)) {
            rms2->momentum(momentum);
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else if (auto sgd = dynamic_cast<torch::optim::SGDOptions *>(d)) {
        sgd->momentum(momentum);
        for (auto &param_group : t->param_groups()) {
          torch::optim::OptimizerOptions *d = &(param_group.options());
          if (auto sgd2 = dynamic_cast<torch::optim::SGDOptions *>(d)) {
            sgd2->momentum(momentum);
          } else
            throw std::invalid_argument("unexpected param group type");
        }
      } else throw std::invalid_argument("unexpected optimizer");)
}

void ato_set_momentum_group(optimizer t, size_t group, double momentum) {
  PROTECT(
      auto &param_group = t->param_groups().at(group);
      torch::optim::OptimizerOptions *d = &(param_group.options());

      if (auto adam = dynamic_cast<torch::optim::AdamOptions *>(d)) {
        auto betas = adam->betas();
        adam->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      } else if (auto adamw = dynamic_cast<torch::optim::AdamWOptions *>(d)) {
        auto betas = adamw->betas();
        adamw->betas(std::tuple<double, double>(momentum, get<1>(betas)));
      } else if (auto rms = dynamic_cast<torch::optim::RMSpropOptions *>(d)) {
        rms->momentum(momentum);
      } if (auto sgd = dynamic_cast<torch::optim::SGDOptions *>(d)) {
        sgd->momentum(momentum);
      } else throw std::invalid_argument("unexpected optimizer");)
}

template <class T> void set_weight_decay(optimizer t, double weight_decay) {
  torch::optim::OptimizerOptions *d = &(t->defaults());
  if (auto p = dynamic_cast<T *>(d)) {
    p->weight_decay(weight_decay);
    for (auto &param_group : t->param_groups()) {
      torch::optim::OptimizerOptions *d = &(param_group.options());
      if (auto p2 = dynamic_cast<T *>(d)) {
        p2->weight_decay(weight_decay);
      } else
        throw std::invalid_argument("unexpected param group type");
    }
  }
}

void ato_set_weight_decay(optimizer t, double weight_decay) {
  PROTECT(set_weight_decay<torch::optim::AdamOptions>(t, weight_decay);
          set_weight_decay<torch::optim::AdamWOptions>(t, weight_decay);
          set_weight_decay<torch::optim::RMSpropOptions>(t, weight_decay);
          set_weight_decay<torch::optim::SGDOptions>(t, weight_decay);)
}

template <class T>
void set_weight_decay_group(optimizer t, size_t group, double weight_decay) {
  auto &param_group = t->param_groups().at(group);
  torch::optim::OptimizerOptions *d = &(param_group.options());
  if (auto p = dynamic_cast<T *>(d)) {
    p->weight_decay(weight_decay);
  }
}

void ato_set_weight_decay_group(optimizer t, size_t group,
                                double weight_decay) {
  PROTECT(
      set_weight_decay_group<torch::optim::AdamOptions>(t, group, weight_decay);
      set_weight_decay_group<torch::optim::AdamWOptions>(t, group,
                                                         weight_decay);
      set_weight_decay_group<torch::optim::RMSpropOptions>(t, group,
                                                           weight_decay);
      set_weight_decay_group<torch::optim::SGDOptions>(t, group, weight_decay);)
}

void ato_zero_grad(optimizer t) { PROTECT(t->zero_grad();) }

void ato_step(optimizer t) { PROTECT(t->step();) }

void ato_free(optimizer t) { delete (t); }

scalar ats_int(int64_t v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

scalar ats_float(double v) {
  PROTECT(return new torch::Scalar(v);)
  return nullptr;
}

int64_t ats_to_int(scalar s) {
  PROTECT(return s->toLong();)
  return -1;
}

double ats_to_float(scalar s) {
  PROTECT(return s->toDouble();)
  return 0.;
}

char *ats_to_string(scalar s) {
  PROTECT(using namespace at; std::ostringstream oss; oss << (*s);
          return strdup(oss.str().c_str());)
  return nullptr;
}

void ats_free(scalar s) { delete (s); }

int atc_cuda_device_count() {
  PROTECT(return torch::cuda::device_count();)
  return -1;
}

int atc_cuda_is_available() {
  PROTECT(return torch::cuda::is_available();)
  return -1;
}

int atc_cudnn_is_available() {
  PROTECT(return torch::cuda::cudnn_is_available();)
  return -1;
}

void atc_manual_seed(uint64_t seed) {
  PROTECT(return torch::cuda::manual_seed(seed);)
}

void atc_manual_seed_all(uint64_t seed) {
  PROTECT(return torch::cuda::manual_seed_all(seed);)
}

void atc_synchronize(int64_t device_index) {
  PROTECT(return torch::cuda::synchronize(device_index);)
}

int atc_user_enabled_cudnn() {
  PROTECT(return at::globalContext().userEnabledCuDNN();)
  return -1;
}

void atc_set_user_enabled_cudnn(int b) {
  PROTECT(at::globalContext().setUserEnabledCuDNN(b);)
}

void atc_set_benchmark_cudnn(int b) {
  PROTECT(at::globalContext().setBenchmarkCuDNN(b);)
}

bool at_context_has_openmp() {
  PROTECT(return at::globalContext().hasOpenMP();)
  return 0;
}

bool at_context_has_mkl() {
  PROTECT(return at::globalContext().hasMKL();)
  return 0;
}

bool at_context_has_lapack() {
  PROTECT(return at::globalContext().hasLAPACK();)
  return 0;
}

bool at_context_has_mkldnn() {
  PROTECT(return at::globalContext().hasMKLDNN();)
  return 0;
}

bool at_context_has_magma() {
  PROTECT(return at::globalContext().hasMAGMA();)
  return 0;
}

bool at_context_has_cuda() {
  PROTECT(return at::globalContext().hasCUDA();)
  return 0;
}

bool at_context_has_cudart() {
  PROTECT(return at::globalContext().hasCUDART();)
  return 0;
}

bool at_context_has_cudnn() {
  PROTECT(return at::globalContext().hasCuDNN();)
  return 0;
}

int64_t at_context_version_cudnn() {
  PROTECT(return at::globalContext().versionCuDNN();)
  return 0;
}

int64_t at_context_version_cudart() {
  PROTECT(return at::globalContext().versionCUDART();)
  return 0;
}

bool at_context_has_cusolver() {
  PROTECT(return at::globalContext().hasCuSOLVER();)
  return 0;
}

bool at_context_has_hip() {
  PROTECT(return at::globalContext().hasHIP();)
  return 0;
}

bool at_context_has_ipu() {
  PROTECT(return at::globalContext().hasIPU();)
  return 0;
}

bool at_context_has_xla() {
  PROTECT(return at::globalContext().hasXLA();)
  return 0;
}

bool at_context_has_lazy() {
  PROTECT(return at::globalContext().hasLazy();)
  return 0;
}

bool at_context_has_mps() {
  PROTECT(return at::globalContext().hasMPS();)
  return 0;
}

module atm_load(char *filename) {
  PROTECT(return new torch::jit::script::Module(torch::jit::load(filename));)
  return nullptr;
}

module atm_load_on_device(char *filename, int device) {
  PROTECT(return new torch::jit::script::Module(
                     torch::jit::load(filename, device_of_int(device)));)
  return nullptr;
}

module atm_load_str(char *data, size_t sz) {
  PROTECT(std::istringstream stream(std::string(data, sz));
          return new torch::jit::script::Module(torch::jit::load(stream));)
  return nullptr;
}

module atm_load_str_on_device(char *data, size_t sz, int device) {
  PROTECT(std::istringstream stream(std::string(data, sz));
          return new torch::jit::script::Module(
              torch::jit::load(stream, device_of_int(device)));)
  return nullptr;
}

tensor atm_forward(module m, tensor *tensors, int ntensors) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < ntensors; ++i) inputs.push_back(*(tensors[i]));
          torch::jit::IValue output = m->forward(std::move(inputs));
          if (!output.isTensor()) throw std::invalid_argument(
              "forward did not return a tensor");
          return new torch::Tensor(output.toTensor());)
  return nullptr;
}

ivalue atm_forward_(module m, ivalue *ivalues, int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < nivalues; ++i) inputs.push_back(*(ivalues[i]));
          torch::jit::IValue output = m->forward(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

tensor atm_method(module m, char *method_name, tensor *tensors, int ntensors) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < ntensors; ++i) inputs.push_back(*(tensors[i]));
          torch::jit::IValue output =
              m->get_method(method_name)(std::move(inputs));
          if (!output.isTensor()) throw std::invalid_argument(
              "method did not return a tensor");
          return new torch::Tensor(output.toTensor());)
  return nullptr;
}

ivalue atm_method_(module m, char *method_name, ivalue *ivalues, int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < nivalues; ++i) inputs.push_back(*(ivalues[i]));
          torch::jit::IValue output =
              m->get_method(method_name)(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue atm_create_class_(module m, char *clz_name, ivalue *ivalues,
                         int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          for (int i = 0; i < nivalues; ++i) inputs.push_back(*(ivalues[i]));

          c10::QualifiedName base(clz_name);
          torch::jit::IValue obj =
              m->create_class(c10::QualifiedName(clz_name), std::move(inputs));
          return new torch::jit::IValue(obj);)
  return nullptr;
}

void atm_eval(module m) { PROTECT(m->eval();) }

void atm_train(module m) { PROTECT(m->train();) }

void atm_free(module m) { delete (m); }

void atm_save(module m, char *filename) { PROTECT(m->save(filename);) }

void atm_to(module m, int device, int dtype, bool non_blocking) {
  PROTECT(m->to(device_of_int(device), at::ScalarType(dtype), non_blocking);)
}

int atm_get_profiling_mode() {
  PROTECT(return torch::jit::getProfilingMode();)
  return 0;
}

void atm_set_tensor_expr_fuser_enabled(int enabled) {
  PROTECT(torch::jit::setTensorExprFuserEnabled((bool)enabled);)
}

bool atm_get_tensor_expr_fuser_enabled() {
  PROTECT(return torch::jit::tensorExprFuserEnabled();)
  return false;
}

void atm_set_profiling_mode(int b) {
  PROTECT(torch::jit::getProfilingMode() = (bool)b;)
}

void atm_fuser_cuda_set_enabled(bool b) {
  PROTECT(torch::jit::fuser::cuda::setEnabled(b);)
}

bool atm_fuser_cuda_is_enabled() {
  PROTECT(return torch::jit::fuser::cuda::isEnabled();)
  return false;
}

module atm_create_for_tracing(char *modl_name, tensor *inputs, int ninputs) {
  PROTECT(torch::jit::script::Module modl(modl_name);
          if (torch::jit::tracer::isTracing()) throw std::invalid_argument(
              "cannot nest tracing calls");
          auto state = std::make_shared<torch::jit::tracer::TracingState>();
          torch::jit::tracer::setTracingState(state);
          auto *_modl_value = state->graph->insertInput(0, "self")->setType(
              modl._ivalue()->type());
          for (int i = 0; i < ninputs; ++i) {
            auto value = state->graph->addInput();
            value->setType(torch::jit::TensorType::get());
            state->setValue(*inputs[i], value);
          } return new torch::jit::script::Module(modl);)
  torch::jit::tracer::abandon();
  return nullptr;
}

void atm_end_tracing(module m, char *fn_name, tensor *outputs, int noutputs) {
  PROTECT(
      auto state = torch::jit::tracer::getTracingState();
      if (state == nullptr) throw std::invalid_argument("not in tracing mode");
      for (int i = 0; i < noutputs; ++i) {
        state->graph->registerOutput(state->getOutput(*outputs[i], i));
      } torch::jit::FixupTraceScopeBlocks(state->graph, m);
      torch::jit::NormalizeOps(state->graph);
      torch::jit::tracer::setTracingState(nullptr);
      auto fn = m->_ivalue()->compilation_unit()->create_function(fn_name,
                                                                  state->graph);
      m->type()->addMethod(fn);)
}

void atm_named_parameters(module m, void *data,
                          void (*f)(void *, char *, tensor)) {
  PROTECT(for (const auto &p : m->named_parameters()) {
    auto v = p.value;
    f(data, (char *)p.name.c_str(), new torch::Tensor(v));
  })
}

ivalue ati_tensor(tensor t) {
  PROTECT(return new torch::jit::IValue(*t);)
  return nullptr;
}

ivalue ati_int(int64_t i) {
  PROTECT(return new torch::jit::IValue(i);)
  return nullptr;
}

ivalue ati_double(double d) {
  PROTECT(return new torch::jit::IValue(d);)
  return nullptr;
}

ivalue ati_bool(int i) {
  PROTECT(return new torch::jit::IValue((bool)i);)
  return nullptr;
}

ivalue ati_string(char *s) {
  PROTECT(string str(s); return new torch::jit::IValue(str);)
  return nullptr;
}

ivalue ati_none() {
  PROTECT(return new torch::jit::IValue();)
  return nullptr;
}

ivalue ati_tuple(ivalue *is, int nvalues) {
  PROTECT(vector<torch::jit::IValue> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(torch::ivalue::Tuple::create(vec));)
  return nullptr;
}

ivalue ati_generic_list(ivalue *is, int nvalues) {
  PROTECT(c10::List<torch::jit::IValue> vec(c10::AnyType::get());
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(c10::List<torch::jit::IValue>(vec));)
  return nullptr;
}

using generic_dict = c10::Dict<torch::jit::IValue, torch::jit::IValue>;

ivalue ati_generic_dict(ivalue *is, int nvalues) {
  PROTECT(
      bool all_keys_are_str = true; for (int i = 0; i < nvalues; ++i) {
        if (!is[2 * i]->isString())
          all_keys_are_str = false;
      } bool all_values_are_tensor = true;
      for (int i = 0; i < nvalues; ++i) {
        if (!is[2 * i + 1]->isTensor())
          all_values_are_tensor = false;
      } if (all_keys_are_str && all_values_are_tensor) {
        generic_dict dict(c10::StringType::get(), c10::TensorType::get());
        for (int i = 0; i < nvalues; ++i)
          dict.insert(is[2 * i]->toString(), is[2 * i + 1]->toTensor());
        return new torch::jit::IValue(dict);
      } else if (all_keys_are_str) {
        generic_dict dict(c10::StringType::get(), c10::AnyType::get());
        for (int i = 0; i < nvalues; ++i)
          dict.insert(is[2 * i]->toString(), *(is[2 * i + 1]));
        return new torch::jit::IValue(dict);
      } else {
        generic_dict dict(c10::AnyType::get(), c10::AnyType::get());
        for (int i = 0; i < nvalues; ++i)
          dict.insert(*(is[2 * i]), *(is[2 * i + 1]));
        return new torch::jit::IValue(dict);
      })
  return nullptr;
}

ivalue ati_int_list(int64_t *is, int nvalues) {
  PROTECT(c10::List<int64_t> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_double_list(double *is, int nvalues) {
  PROTECT(c10::List<double> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i]);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_bool_list(char *is, int nvalues) {
  PROTECT(c10::List<bool> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(is[i] != 0);
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_string_list(char **is, int nvalues) {
  PROTECT(c10::List<string> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(string(is[i]));
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_tensor_list(tensor *is, int nvalues) {
  PROTECT(c10::List<at::Tensor> vec;
          for (int i = 0; i < nvalues; ++i) vec.push_back(*(is[i]));
          return new torch::jit::IValue(vec);)
  return nullptr;
}

ivalue ati_device(int device_idx) {
  PROTECT(return new torch::jit::IValue(device_of_int(device_idx));)
  return nullptr;
}

int ati_tag(ivalue i) {
  PROTECT(
      if (i->isNone()) return 0; else if (i->isTensor()) return 1;
      else if (i->isDouble()) return 2; else if (i->isInt()) return 3;
      else if (i->isBool()) return 4; else if (i->isTuple()) return 5;
      else if (i->isIntList()) return 6; else if (i->isDoubleList()) return 7;
      else if (i->isBoolList()) return 8; else if (i->isString()) return 9;
      else if (i->isTensorList()) return 10; else if (i->isList()) return 12;
      else if (i->isGenericDict()) return 13; else if (i->isObject()) return 14;
      throw std::invalid_argument(("unsupported tag " + i->tagKind()).c_str());
      return -1;)
  return -1;
}

int64_t ati_to_int(ivalue i) {
  PROTECT(return i->toInt();)
  return -1;
}

double ati_to_double(ivalue i) {
  PROTECT(return i->toDouble();)
  return 0;
}

int ati_to_bool(ivalue i) {
  PROTECT(return i->toBool();)
  return -1;
}

char *ati_to_string(ivalue i) {
  PROTECT(auto str = i->toStringRef(); return strdup(str.c_str());)
  return nullptr;
}

tensor ati_to_tensor(ivalue i) {
  PROTECT(return new torch::Tensor(i->toTensor());)
  return nullptr;
}

int ati_length(ivalue i) {
  PROTECT(if (i->isTuple()) return i->toTuple()->elements().size();
          else if (i->isIntList()) return i->toIntList().size();
          else if (i->isDoubleList()) return i->toDoubleList().size();
          else if (i->isBoolList()) return i->toBoolList().size();
          else if (i->isString()) return i->toStringRef().size();
          else if (i->isTensorList()) return i->toTensorList().size();
          else if (i->isList()) return i->toList().size();
          else if (i->isGenericDict()) return i->toGenericDict().size();
          throw std::invalid_argument(
              ("unsupported tag for length " + i->tagKind()).c_str());
          return -1;)
  return -1;
}

int ati_tuple_length(ivalue i) {
  PROTECT(return i->toTuple()->elements().size();)
  return -1;
}

void ati_to_tuple(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(auto vec = i->toTuple()->elements(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected tuple size");
  } for (int i = 0; i < noutputs;
         ++i) outputs[i] = new torch::jit::IValue(vec[i]);)
}

void ati_to_generic_list(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(auto vec = i->toList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] =
                                      new torch::jit::IValue(vec[i]);)
}

void ati_to_generic_dict(ivalue i, ivalue *outputs, int noutputs) {
  PROTECT(
      auto dict = i->toGenericDict(); if (dict.size() != noutputs) {
        throw std::invalid_argument("unexpected dict size");
      } int k = 0;
      for (auto it = dict.begin(); it != dict.end(); ++it) {
        outputs[k++] = new torch::jit::IValue(it->key());
        outputs[k++] = new torch::jit::IValue(it->value());
      })
}

void ati_to_int_list(ivalue i, int64_t *outputs, int noutputs) {
  PROTECT(auto vec = i->toIntList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<int> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = vec[i];)
}

void ati_to_double_list(ivalue i, double *outputs, int noutputs) {
  PROTECT(auto vec = i->toDoubleList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<double> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = vec[i];)
}

void ati_to_bool_list(ivalue i, char *outputs, int noutputs) {
  PROTECT(auto vec = i->toBoolList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<bool> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = vec[i];)
}

void ati_to_tensor_list(ivalue i, tensor *outputs, int noutputs) {
  PROTECT(auto vec = i->toTensorList(); if (vec.size() != noutputs) {
    throw std::invalid_argument("unexpected list<tensor> size");
  } for (int i = 0; i < noutputs; ++i) outputs[i] = new torch::Tensor(vec[i]);)
}

ivalue ati_object_method_(ivalue i, char *method_name, ivalue *ivalues,
                          int nivalues) {
  PROTECT(std::vector<torch::jit::IValue> inputs;
          inputs.push_back(*i); // self parameter
          for (int j = 0; j < nivalues; ++j) inputs.push_back(*(ivalues[j]));
          torch::jit::IValue output = i->toObjectRef().type()->getMethod(
              method_name)(std::move(inputs));
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue ati_object_getattr_(ivalue i, char *attr_name) {
  PROTECT(torch::jit::IValue output = i->toObjectRef().getAttr(attr_name);
          return new torch::jit::IValue(output);)
  return nullptr;
}

ivalue ati_clone(ivalue i) {
  PROTECT(return new torch::jit::IValue(*i);)
  return nullptr;
}

void ati_free(ivalue i) { delete (i); }

void at_set_graph_executor_optimize(bool o) {
  torch::jit::setGraphExecutorOptimize(o);
}

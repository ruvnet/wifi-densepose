# Loading and Running a Quantized PyTorch Model in Rust

This builds upon the [main JIT tutorial](https://github.com/LaurentMazare/tch-rs/blob/master/examples/jit/README.md) 
to load and run quantized PyTorch models.

After a model has been successfully trained, they must often be deployed to
resource-constrained devices (such as cheap cloud computing instances or mobile 
phones). A common technique used to speed up inference on devices lacking a GPU
is to quantize the model's weight matrices and perform some or all of its operations
using INT8 tensors (rather than FP32).

## Quantizing a Python PyTorch Model
The details of quantizing a PyTorch model are described in the PyTorch [documentation](https://pytorch.org/docs/stable/quantization.html). In this example we simply export a model available within torchvision.
```python
import torch
import torchvision

model = torchvision.models.quantization.resnet18(
    pretrained=True, quantize=True
)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet18_int8.pt")
```

## Loading a Quantized Torch Script Model from Rust
The process for loading a quantized Torch Script model is almost identical to that
outlined in the [main jit tutorial](https://github.com/LaurentMazare/tch-rs/blob/master/examples/jit/README.md).
The only difference is that we must specify a quantization engine (or backend) to 
use during inference. The two engines currently supported by PyTorch are [FBGEMM](https://github.com/pytorch/FBGEMM) 
(for inference on x86 architectures) and [QNNPACK](https://github.com/pytorch/QNNPACK) 
(for inference on ARM architectures).

This can be done using the following:
```rust
tch::QEngine::FBGEMM.set()?,
// or
tch::QEngine::QNNPACK.set()?,
```

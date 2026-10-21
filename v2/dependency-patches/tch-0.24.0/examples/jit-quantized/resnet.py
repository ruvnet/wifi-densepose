import torch
import torchvision


def save_resnet18_script_module(
    filename: str, pretrained: bool, quantize: bool
) -> None:
    """Save quantizable resnet18 as traced script module."""
    model = torchvision.models.quantization.resnet18(
        pretrained=pretrained, quantize=quantize
    )
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(filename)


# Save fp32 model
save_resnet18_script_module("resnet18_fp32.pt", True, False)

# Save int8 model
save_resnet18_script_module("resnet18_int8.pt", True, True)

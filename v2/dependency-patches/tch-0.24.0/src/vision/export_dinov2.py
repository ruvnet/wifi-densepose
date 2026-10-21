import torch
from safetensors.torch import save_file

def normalize_key(k):
    if k.startswith("backbone."):
        k = k[9:]
    if k.startswith("linear_head."):
        k = k[7:]
    return k

for model_size in ["small", "base", "large", "giant"]:
    letter = model_size[0]
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{letter}14_lc', layers=1)
    weights = dinov2_vits14.state_dict()
    weights = {normalize_key(k): v for k, v in weights.items()}
    save_file(weights, f"dinov2_vit{letter}14.safetensors")

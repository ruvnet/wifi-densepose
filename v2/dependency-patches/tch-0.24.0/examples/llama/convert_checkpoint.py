# Adapted from https://github.com/Lightning-AI/lit-llama/blob/main/scripts/convert_checkpoint.py
import sys
import torch
from typing import Dict
from pathlib import Path
from safetensors.torch import save_file

def convert_state_dict(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype = torch.float16) -> Dict[str, torch.Tensor]:
    print("start conv")

    def get_and_remove(key):
        v = state_dict[key].to(dtype)
        del state_dict[key]
        return v

    converted = {}
    converted["transformer.wte.weight"] = get_and_remove("tok_embeddings.weight")
    converted["lm_head.weight"] = get_and_remove("output.weight")
    converted["transformer.ln_f.scale"] = get_and_remove("norm.weight")

    for layer_idx in sorted(set([k.split(".")[1] for k in state_dict if k.startswith("layers")])):
        print(layer_idx)

        # attention
        # the wq, wk, wv from the FB model are stacked in our model as c_attn
        converted[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.cat(
            (
                get_and_remove(f"layers.{layer_idx}.attention.wq.weight"),
                get_and_remove(f"layers.{layer_idx}.attention.wk.weight"),
                get_and_remove(f"layers.{layer_idx}.attention.wv.weight"),
                )
        )
        converted[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = get_and_remove(
            f"layers.{layer_idx}.attention.wo.weight"
            )
        # mlp
        converted[f"transformer.h.{layer_idx}.mlp.c_fc1.weight"] = get_and_remove(
            f"layers.{layer_idx}.feed_forward.w1.weight"
            )
        converted[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = get_and_remove(
            f"layers.{layer_idx}.feed_forward.w2.weight"
            )
        converted[f"transformer.h.{layer_idx}.mlp.c_fc2.weight"] = get_and_remove(
            f"layers.{layer_idx}.feed_forward.w3.weight"
            )
        # rms norm
        converted[f"transformer.h.{layer_idx}.rms_1.scale"] = get_and_remove(f"layers.{layer_idx}.attention_norm.weight")
        converted[f"transformer.h.{layer_idx}.rms_2.scale"] = get_and_remove(f"layers.{layer_idx}.ffn_norm.weight")
    return converted

def convert_weights(llama_ckpt, *, output_st: Path = Path("llama.safetensors"), dtype: str = "float16") -> None:
    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    checkpoint = torch.load(llama_ckpt, map_location="cpu")
    converted = convert_state_dict(checkpoint, dtype=dt)
    del checkpoint
    save_file(converted, output_st)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(f"usage: convert_checkpoint.py ..../LLaMA/7B/consolidated.00.pth")
    convert_weights(sys.argv[1])

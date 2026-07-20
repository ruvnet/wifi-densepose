"""ADR-152 §2.2: export the retrained WiFlow-STD PyTorch checkpoint to
safetensors with tch-rs (VarStore) variable names, plus a numerical-parity
fixture for the Rust port.

Outputs (all under results/, gitignored):
  retrained_wiflow_std.safetensors  -- 248 f32 tensors named exactly as the
                                       Rust WiFlowStdModel VarStore expects
                                       (see wiflow_std/model.rs
                                       `dump_variable_names` for the
                                       authoritative name dump)
  parity_fixture.npz                -- deterministic input (seed 42,
                                       shape (2, 540, 20), uniform [0,1]) and
                                       the Python model's eval-mode output
  parity_fixture.json               -- same data as flattened f32 lists, for
                                       the dependency-free Rust test
                                       (tests/test_wiflow_std_parity.rs)

PyTorch -> tch key mapping (derived from the VarStore dump, not guessed):

  tcn.network.{i}.conv1_group.weight        -> tcn{i}.conv1_group.weight
  tcn.network.{i}.bn*_{group,pw}.<leaf>     -> tcn{i}.bn*_{group,pw}.<leaf>
  tcn.network.{i}.downsample.0.weight       -> tcn{i}.ds_conv.weight
  tcn.network.{i}.downsample.1.<leaf>       -> tcn{i}.ds_bn.<leaf>
  up.block.{0,1,4,5,8,9}.<leaf>             -> conv_in.{conv1,bn1,conv2,bn2,conv3,bn3}.<leaf>
  up.downsample.{0,1}.<leaf>                -> conv_in.{ds_conv,ds_bn}.<leaf>
  residual_blocks.{i}.block.{...}.<leaf>    -> conv{i}.{conv1..bn3}.<leaf>
  residual_blocks.{i}.downsample.{0,1}      -> conv{i}.{ds_conv,ds_bn}
  attention.{width,height}_axis.qkv_transform.weight
                                            -> attention.{width,height}.qkv.weight
  attention.{width,height}_axis.bn_*        -> attention.{width,height}.bn_*
  decoder.{0,1,3,4}.<leaf>                  -> {dec_conv1,dec_bn1,dec_conv2,dec_bn2}.<leaf>
  *.num_batches_tracked                     -> dropped (tch BatchNorm has no such buffer)

Legacy upstream names (att. -> attention., final_conv. -> decoder.) are
remapped first, exactly as eval_repro.py does for the released checkpoint.

Usage:
  .venv/Scripts/python.exe export_to_safetensors.py
"""

import json
import os
import re

import numpy as np
import torch
from safetensors.torch import save_file

from _bench_common import RESULTS, import_upstream, remap_legacy_keys

import_upstream()  # sys.path + models stub

from models.pose_model import WiFlowPoseModel  # noqa: E402

CHECKPOINT = os.path.join(RESULTS, "retrained_best_pose_model.pth")

# Sequential index -> tch sub-name inside one ConvBlock1/AsymmetricConvBlock:
# [Conv2d(0), BN(1), SiLU(2), Dropout2d(3), Conv2d(4), BN(5), SiLU(6),
#  Dropout2d(7), Conv2d(8), BN(9)]
_BLOCK_IDX = {"0": "conv1", "1": "bn1", "4": "conv2", "5": "bn2",
              "8": "conv3", "9": "bn3"}
_DS_IDX = {"0": "ds_conv", "1": "ds_bn"}
_DECODER_IDX = {"0": "dec_conv1", "1": "dec_bn1", "3": "dec_conv2",
                "4": "dec_bn2"}


def _conv_block(new_prefix: str, rest: str) -> str:
    m = re.fullmatch(r"block\.(\d+)\.(.+)", rest)
    if m:
        return f"{new_prefix}.{_BLOCK_IDX[m.group(1)]}.{m.group(2)}"
    m = re.fullmatch(r"downsample\.(\d+)\.(.+)", rest)
    if m:
        return f"{new_prefix}.{_DS_IDX[m.group(1)]}.{m.group(2)}"
    raise KeyError(f"unmapped conv-block key: {new_prefix} / {rest}")


def map_key(key: str) -> str:
    """Map one PyTorch state_dict key to the tch VarStore name."""
    m = re.fullmatch(r"tcn\.network\.(\d+)\.(.+)", key)
    if m:
        i, rest = m.groups()
        rest = (rest.replace("downsample.0.", "ds_conv.")
                    .replace("downsample.1.", "ds_bn."))
        return f"tcn{i}.{rest}"

    m = re.fullmatch(r"up\.(.+)", key)
    if m:
        return _conv_block("conv_in", m.group(1))

    m = re.fullmatch(r"residual_blocks\.(\d+)\.(.+)", key)
    if m:
        return _conv_block(f"conv{m.group(1)}", m.group(2))

    m = re.fullmatch(r"attention\.(width|height)_axis\.(.+)", key)
    if m:
        axis, rest = m.groups()
        rest = rest.replace("qkv_transform.", "qkv.")
        return f"attention.{axis}.{rest}"

    m = re.fullmatch(r"decoder\.(\d+)\.(.+)", key)
    if m:
        return f"{_DECODER_IDX[m.group(1)]}.{m.group(2)}"

    raise KeyError(f"unmapped checkpoint key: {key}")


def main():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "tcn.network.0.conv1_group.weight" not in {
        k for k in state
    } | {k.replace("att.", "attention.") for k in state}:
        # tolerate trainer wrappers like {"model_state_dict": ...}
        for wrapper in ("model_state_dict", "state_dict", "model"):
            if isinstance(state, dict) and wrapper in state:
                state = state[wrapper]
                break

    # Legacy upstream names predate the published code (_bench_common).
    state = remap_legacy_keys(state)

    mapped = {}
    dropped = 0
    for k, v in state.items():
        if k.endswith("num_batches_tracked"):
            dropped += 1
            continue
        tch_key = map_key(k)
        if tch_key in mapped:
            raise KeyError(f"duplicate mapped key: {k} -> {tch_key}")
        mapped[tch_key] = v.detach().to(torch.float32).contiguous()

    n_params = sum(v.numel() for k, v in mapped.items()
                   if "running_" not in k)
    print(f"checkpoint tensors: {len(state)} "
          f"(dropped {dropped} num_batches_tracked)")
    print(f"mapped tensors: {len(mapped)}, "
          f"non-buffer params: {n_params/1e6:.6f}M")
    assert len(mapped) == 248, f"expected 248 tch variables, got {len(mapped)}"
    assert n_params == 2_225_042, f"param count mismatch: {n_params}"

    st_path = os.path.join(RESULTS, "retrained_wiflow_std.safetensors")
    save_file(mapped, st_path)
    print(f"wrote {st_path}")

    # ---- parity fixture --------------------------------------------------
    model = WiFlowPoseModel(dropout=0.5)
    model.load_state_dict(state, strict=True)
    model.eval()

    gen = torch.Generator().manual_seed(42)
    x = torch.rand(2, 540, 20, generator=gen, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    print(f"fixture input {tuple(x.shape)} -> output {tuple(y.shape)}, "
          f"output range [{y.min().item():.6f}, {y.max().item():.6f}]")

    np.savez(os.path.join(RESULTS, "parity_fixture.npz"),
             input=x.numpy(), output=y.numpy())
    fixture = {
        "seed": 42,
        "input_shape": list(x.shape),
        "input": x.flatten().tolist(),
        "output_shape": list(y.shape),
        "output": y.flatten().tolist(),
    }
    json_path = os.path.join(RESULTS, "parity_fixture.json")
    with open(json_path, "w") as f:
        json.dump(fixture, f)
    print(f"wrote {os.path.join(RESULTS, 'parity_fixture.npz')}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

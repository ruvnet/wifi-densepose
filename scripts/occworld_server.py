"""
OccWorld inference server — Unix-socket newline-delimited JSON IPC.

Usage:
    ~/ml-env/bin/python3 occworld_server.py [SOCKET_PATH]

Default socket: /tmp/occworld.sock

Request JSON (one line):
    {
      "past_frames": [{"width":200,"height":200,"depth":16,"voxels":[...u8...]},...],
      "voxel_resolution_m": 0.4,
      "scene_bounds": {"x_min":-40,"x_max":40,"y_min":-40,"y_max":40,"z_min":-1,"z_max":5.4},
      "prediction_steps": 15
    }

Response JSON (one line):
    {
      "future_frames": [...],
      "trajectory_priors": [...],
      "confidence": 0.82,
      "model_id": "occworld-patched-v0",
      "inference_ms": 375
    }
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys

# Phase 3 — RuViewOccDataset available for callers that want to build
# training tensors directly from WorldGraph snapshots (see occworld_retrain.py).
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from ruview_occ_dataset import RuViewOccDataset, snapshot_to_voxels, record_snapshot  # noqa: F401
    _DATASET_AVAILABLE = True
except ImportError:
    _DATASET_AVAILABLE = False
import time
import traceback
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("occworld_server")

# ---------------------------------------------------------------------------
# OccWorld repo path
# ---------------------------------------------------------------------------
OCCWORLD_ROOT = os.path.expanduser("~/projects/OccWorld")
if OCCWORLD_ROOT not in sys.path:
    sys.path.insert(0, OCCWORLD_ROOT)

# nuScenes 16-class label where class 7 = "pedestrian" and class 17 = "empty"
PERSON_CLASSES = {7}   # pedestrian in labels_16 scheme
FREE_CLASS = 17

# Default config dimensions (from config/occworld.py)
NUM_FRAMES = 15    # model.num_frames
OFFSET = 1         # model.offset — one conditioning frame prepended
H, W, D = 200, 200, 16   # spatial grid
NUM_CLASSES = 18   # model output classes
POSE_DIM = 128     # base_channel * 2

# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patched_forward_inference(self, x: torch.Tensor) -> dict:
    """
    Drop-in replacement for TransVQVAE.forward_inference.

    The original calls:
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
    but PlanUAutoRegTransformer.forward(tokens, pose_tokens) does not accept
    a `hidden` keyword and returns a (queries, pose_queries) tuple.

    Fix: pass pose_tokens=zeros, unpack tuple.
    """
    from copy import deepcopy
    from einops import rearrange

    bs, F, H_, W_, D_ = x.shape
    output_dict: dict = {}
    output_dict["target_occs"] = x[:, self.offset:]

    z, shape = self.vae.forward_encoder(x)
    z = self.vae.vqvae.quant_conv(z)
    z_q, loss, (perplexity, min_encodings, min_encoding_indices) = (
        self.vae.vqvae.forward_quantizer(z, is_voxel=False)
    )
    min_encoding_indices = rearrange(
        min_encoding_indices, "(b f) h w -> b f h w", b=bs
    )
    output_dict["ce_labels"] = (
        min_encoding_indices[:, self.offset:].detach().flatten(0, 1)
    )
    z_q = rearrange(z_q, "(b f) c h w -> b f c h w", b=bs)

    tokens = z_q[:, : self.num_frames]  # (bs, num_frames, C, H, W)
    # Build zero pose_tokens matching transformer's expected pose_shape (bs, F, pose_dim)
    bs_, F_, C_, H_t, W_t = tokens.shape
    pose_tokens = torch.zeros(bs_, F_, C_, device=tokens.device, dtype=tokens.dtype)

    # Transformer returns (queries, pose_queries) tuple
    z_q_predict, _pose_out = self.transformer(tokens, pose_tokens=pose_tokens)

    z_q_predict = z_q_predict.flatten(0, 1)
    output_dict["ce_inputs"] = z_q_predict
    z_q_predict = z_q_predict.argmax(dim=1)
    z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
    z_q_predict = rearrange(z_q_predict, "bf h w c -> bf c h w")
    z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
    z_q_predict = self.vae.forward_decoder(
        z_q_predict, shape, output_dict["target_occs"].shape
    )
    output_dict["logits"] = z_q_predict
    pred = z_q_predict.argmax(dim=-1).detach().cuda()
    output_dict["sem_pred"] = pred
    pred_iou = deepcopy(pred)
    pred_iou[pred_iou != FREE_CLASS] = 1
    pred_iou[pred_iou == FREE_CLASS] = 0
    output_dict["iou_pred"] = pred_iou
    return output_dict


def _patched_forward(self, x: torch.Tensor, metas=None) -> dict:
    """
    Drop-in replacement for TransVQVAE.forward.

    The original routes through forward_inference_with_plan when pose_encoder
    exists, which requires metas (ego-vehicle pose data).  For our WiFi-CSI
    use-case there is no ego pose, so we always call forward_inference directly.
    """
    if self.training:
        return self.forward_train(x)
    return self.forward_inference(x)


def apply_patches(model: Any) -> Any:
    """Monkey-patch forward and forward_inference to fix the transformer API mismatch."""
    import types

    model.forward_inference = types.MethodType(_patched_forward_inference, model)
    model.forward = types.MethodType(_patched_forward, model)
    log.info("Applied patches: forward (bypass plan path) + forward_inference (pose_tokens zero-init, tuple unpack)")
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | None = None) -> Any:
    """
    Build TransVQVAE from the OccWorld config, optionally loading weights.
    Returns model in eval mode on CUDA (or CPU if CUDA unavailable).
    checkpoint_path=None -> dummy mode with random weights (for testing).
    """
    t0 = time.monotonic()

    # Import OccWorld modules (mmengine registry populated on import)
    from mmengine.registry import MODELS  # noqa: F401
    import model as _model_pkg  # noqa: F401 — registers VAERes2D, TransVQVAE …
    import model.VAE.vae_2d_resnet  # noqa: F401
    import model.transformer.PlanUtransformer  # noqa: F401
    import model.transformer.pose_encoder  # noqa: F401
    import model.transformer.pose_decoder  # noqa: F401

    # Load config dict from occworld.py (has the `model` dict)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "occworld_cfg",
        os.path.join(OCCWORLD_ROOT, "config", "occworld.py"),
    )
    cfg_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(cfg_mod)  # type: ignore[union-attr]
    model_cfg = cfg_mod.model

    net = MODELS.build(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path and os.path.isfile(checkpoint_path):
        log.info("Loading checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Strip common "model." prefix from distributed training saves
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            log.warning("Missing keys (%d): %s …", len(missing), missing[:3])
        if unexpected:
            log.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:3])
        mode_tag = "checkpoint"
    else:
        if checkpoint_path:
            log.warning("Checkpoint not found at %s — running in DUMMY mode", checkpoint_path)
        else:
            log.info("No checkpoint supplied — running in DUMMY mode (random weights)")
        mode_tag = "dummy"

    net = net.to(device)
    net.eval()
    net = apply_patches(net)

    elapsed = time.monotonic() - t0
    n_params = sum(p.numel() for p in net.parameters())
    log.info(
        "Model ready [%s] | params=%.2fM | device=%s | load_time=%.1fs",
        mode_tag,
        n_params / 1e6,
        device,
        elapsed,
    )

    if device == "cuda":
        vram = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        log.info("VRAM allocated=%.2f GB  reserved=%.2f GB", vram, reserved)

    return net


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def voxels_to_tensor(past_frames: list[dict]) -> torch.Tensor:
    """
    Convert list of frame dicts to model input tensor.

    Each frame dict: {"width": W, "height": H, "depth": D, "voxels": [u8 flat]}
    Returns: torch.Tensor shape (1, F, H, W, D)  dtype=long  on CUDA/CPU.
    """
    arrays = []
    for f in past_frames:
        w, h, d = f["width"], f["height"], f["depth"]
        vox = np.array(f["voxels"], dtype=np.int64).reshape(h, w, d)
        arrays.append(vox)

    # Stack to (F, H, W, D), add batch dim -> (1, F, H, W, D)
    tensor = torch.from_numpy(np.stack(arrays, axis=0)).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tensor.to(device)


def decode_trajectories(
    future_sem_pred: torch.Tensor,
    scene_bounds: dict,
    voxel_resolution_m: float,
) -> list[dict]:
    """
    Convert predicted semantic voxel frames to trajectory_priors.

    For each future frame find voxels labelled as person class (7),
    compute centroid in world coordinates, emit as a waypoint.

    future_sem_pred: (B, F, H, W, D) long tensor
    Returns list of trajectory dicts, one per detected person cluster.
    """
    pred = future_sem_pred[0]  # (F, H, W, D)
    n_future = pred.shape[0]

    x_min = scene_bounds.get("x_min", -40.0)
    y_min = scene_bounds.get("y_min", -40.0)
    z_min = scene_bounds.get("z_min", -1.0)

    trajectories: list[dict] = []
    waypoints_by_id: dict[int, list[dict]] = {}  # simple single-track approach

    for t in range(n_future):
        frame = pred[t]  # (H, W, D)
        person_mask = torch.zeros_like(frame, dtype=torch.bool)
        for cls in PERSON_CLASSES:
            person_mask |= frame == cls

        if not person_mask.any():
            continue

        # Centroid of all person voxels in this frame
        indices = person_mask.nonzero(as_tuple=False).float()  # (N, 3) [h, w, d]
        centroid = indices.mean(dim=0)  # [h_c, w_c, d_c]

        world_x = float(x_min + centroid[1].item() * voxel_resolution_m)
        world_y = float(y_min + centroid[0].item() * voxel_resolution_m)
        world_z = float(z_min + centroid[2].item() * voxel_resolution_m)

        waypoints_by_id.setdefault(0, []).append(
            {"frame": t, "x": world_x, "y": world_y, "z": world_z}
        )

    for track_id, wps in waypoints_by_id.items():
        trajectories.append(
            {
                "track_id": track_id,
                "class": "pedestrian",
                "waypoints": wps,
            }
        )

    return trajectories


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: Any, tensor: torch.Tensor, scene_bounds: dict,
                  voxel_resolution_m: float) -> dict:
    """
    Run forward pass and return response payload dict.
    tensor: (1, F, H, W, D)
    """
    # TransVQVAE expects (B, num_frames+offset, H, W, D)
    # If caller sends fewer frames pad with zeros; if more, truncate
    target_f = model.num_frames + model.offset  # typically 16
    bs, f, h, w, d = tensor.shape

    if f < target_f:
        pad = torch.zeros(bs, target_f - f, h, w, d, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=1)
    elif f > target_f:
        tensor = tensor[:, :target_f]

    t0 = time.monotonic()
    with torch.no_grad():
        output_dict = model(tensor)
    inference_ms = (time.monotonic() - t0) * 1000.0

    sem_pred = output_dict["sem_pred"]  # (B, F_out, H, W, D)

    # Confidence: fraction of non-free voxels across all predicted frames
    total_vox = sem_pred.numel()
    occupied = (sem_pred != FREE_CLASS).sum().item()
    confidence = float(occupied / total_vox) if total_vox > 0 else 0.0

    # Encode future frames as flat voxel lists (uint8 serialisable)
    future_frames = []
    pred_cpu = sem_pred[0].cpu().numpy().astype(np.uint8)  # (F, H, W, D)
    for t in range(pred_cpu.shape[0]):
        frame_arr = pred_cpu[t]
        fh, fw, fd = frame_arr.shape
        future_frames.append(
            {
                "width": fw,
                "height": fh,
                "depth": fd,
                "voxels": frame_arr.flatten().tolist(),
            }
        )

    trajectory_priors = decode_trajectories(sem_pred, scene_bounds, voxel_resolution_m)

    return {
        "future_frames": future_frames,
        "trajectory_priors": trajectory_priors,
        "confidence": round(confidence, 4),
        "model_id": "occworld-patched-v0",
        "inference_ms": round(inference_ms, 1),
    }


# ---------------------------------------------------------------------------
# Server loop
# ---------------------------------------------------------------------------

def handle_connection(conn: socket.socket, model: Any) -> None:
    """Read one newline-terminated JSON request, write one JSON response."""
    try:
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break

        if not buf.strip():
            return

        line = buf.split(b"\n")[0]
        request = json.loads(line.decode("utf-8"))

        past_frames = request["past_frames"]
        voxel_res = float(request.get("voxel_resolution_m", 0.4))
        scene_bounds = request.get(
            "scene_bounds",
            {"x_min": -40, "x_max": 40, "y_min": -40, "y_max": 40, "z_min": -1, "z_max": 5.4},
        )

        tensor = voxels_to_tensor(past_frames)
        response = run_inference(model, tensor, scene_bounds, voxel_res)

    except Exception:  # noqa: BLE001
        log.exception("Inference error")
        response = {
            "error": traceback.format_exc(),
            "future_frames": [],
            "trajectory_priors": [],
            "confidence": 0.0,
            "model_id": "occworld-patched-v0",
            "inference_ms": 0.0,
        }

    try:
        payload = (json.dumps(response) + "\n").encode("utf-8")
        conn.sendall(payload)
    except BrokenPipeError:
        pass
    finally:
        conn.close()


def main() -> None:
    socket_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/occworld.sock"
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None

    log.info("OccWorld inference server starting")
    log.info("Socket path : %s", socket_path)
    log.info("Checkpoint  : %s", checkpoint_path or "(none — dummy mode)")

    model = load_model(checkpoint_path)

    # Remove stale socket file
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen(8)
    os.chmod(socket_path, 0o660)

    # Graceful shutdown
    _running = {"value": True}

    def _shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        log.info("Received signal %d — shutting down", signum)
        _running["value"] = False
        server_sock.close()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Listening on %s", socket_path)

    while _running["value"]:
        try:
            conn, _ = server_sock.accept()
        except OSError:
            break
        handle_connection(conn, model)

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    log.info("Server stopped")


if __name__ == "__main__":
    main()

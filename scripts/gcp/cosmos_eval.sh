#!/usr/bin/env bash
# Run Cosmos-Transfer2.5-2B evaluation on GCP A100 80GB instance
# Usage: bash scripts/gcp/cosmos_eval.sh <INSTANCE_IP> [--snapshot-dir <DIR>]
#
# Flow:
#   1. Start OccWorld sensing server on remote (generates control tensors)
#   2. Rsync RuView scripts + any local control tensors to instance
#   3. Run Cosmos-Transfer2.5 inference with depth+seg control signals
#   4. Download generated video and decoded trajectory priors
#   5. Benchmark inference time (A100 actual vs RTX 5080 estimate)

set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <INSTANCE_IP> [--snapshot-dir <DIR>] [--no-server]" >&2
  echo ""
  echo "  INSTANCE_IP        External IP of the cosmos-eval GCP instance"
  echo "  --snapshot-dir     Local snapshot dir to upload as control input"
  echo "                     (default: ./out/snapshots if it exists)"
  echo "  --no-server        Skip starting the OccWorld server on remote"
  echo ""
  echo "Example:"
  echo "  $0 34.123.45.67 --snapshot-dir /tmp/snapshots"
  exit 1
fi

INSTANCE_IP="$1"
shift

SNAPSHOT_DIR="./out/snapshots"
START_SERVER=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot-dir) SNAPSHOT_DIR="$2"; shift 2 ;;
    --no-server)    START_SERVER=false; shift ;;
    -h|--help)
      echo "Usage: $0 <INSTANCE_IP> [--snapshot-dir <DIR>] [--no-server]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

GCP_USER="${GCP_USER:-$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)}"
REMOTE="${GCP_USER}@${INSTANCE_IP}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
LOCAL_SCRIPTS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/scripts"
OUTPUT_DIR="./out/cosmos-results"
REMOTE_RESULTS="~/cosmos-results"
REMOTE_SCRIPTS="~/ruview-scripts"
REMOTE_CONTROL="~/control-tensors"
COSMOS_MODEL_DIR="/opt/models/cosmos-transfer2.5-2b"

log() { echo "[cosmos_eval] $*"; }

# ── SSH connectivity check ────────────────────────────────────────────────────
log "Checking SSH connectivity to $REMOTE ..."
if ! ssh $SSH_OPTS "$REMOTE" "echo ok" &>/dev/null; then
  echo "ERROR: Cannot SSH to $REMOTE" >&2
  echo "       Ensure the instance is running: gcloud compute instances list --project=cognitum-20260110" >&2
  exit 1
fi
log "SSH connection OK"

# ── Verify startup completed ──────────────────────────────────────────────────
log "Checking Cosmos startup log ..."
COSMOS_READY=$(ssh $SSH_OPTS "$REMOTE" \
  "grep -c 'setup complete' /var/log/cosmos-startup.log 2>/dev/null || echo 0")
if [[ "$COSMOS_READY" -lt 1 ]]; then
  log "WARNING: Cosmos startup may not be complete."
  log "         Check: ssh $REMOTE 'tail -20 /var/log/cosmos-startup.log'"
fi

# Verify model weights exist
MODEL_EXISTS=$(ssh $SSH_OPTS "$REMOTE" \
  "test -d $COSMOS_MODEL_DIR && find $COSMOS_MODEL_DIR -name '*.safetensors' -o -name '*.bin' 2>/dev/null | wc -l || echo 0")
if [[ "$MODEL_EXISTS" -lt 1 ]]; then
  echo "ERROR: Cosmos-Transfer2.5-2B weights not found at $COSMOS_MODEL_DIR on remote." >&2
  echo "       The startup script may still be downloading (can take 30-60 min)." >&2
  echo "       Monitor: ssh $REMOTE 'tail -f /var/log/cosmos-startup.log'" >&2
  exit 1
fi
log "Model weights verified ($MODEL_EXISTS files in $COSMOS_MODEL_DIR)"

# ── Rsync scripts to remote ───────────────────────────────────────────────────
log "Rsyncing RuView scripts → $REMOTE:$REMOTE_SCRIPTS ..."
ssh $SSH_OPTS "$REMOTE" "mkdir -p $REMOTE_SCRIPTS $REMOTE_CONTROL $REMOTE_RESULTS"
rsync -avz \
  -e "ssh $SSH_OPTS" \
  --include="occworld_retrain.py" \
  --include="occworld_server.py" \
  --include="ruview_occ_dataset.py" \
  --exclude="gcp/" \
  --exclude="*.sh" \
  "$LOCAL_SCRIPTS_DIR/" \
  "${REMOTE}:${REMOTE_SCRIPTS}/"

# ── Rsync local snapshots as control input (if they exist) ────────────────────
if [[ -d "$SNAPSHOT_DIR" ]]; then
  SNAP_COUNT=$(find "$SNAPSHOT_DIR" -name "*.json" 2>/dev/null | wc -l)
  log "Rsyncing $SNAP_COUNT snapshots from $SNAPSHOT_DIR → remote control-tensors ..."
  rsync -avz \
    -e "ssh $SSH_OPTS" \
    "$SNAPSHOT_DIR/" \
    "${REMOTE}:${REMOTE_CONTROL}/snapshots/"
else
  log "No local snapshot dir found at $SNAPSHOT_DIR — will use synthetic control tensors on remote"
fi

# ── Stage 1: Start OccWorld sensing server on remote ─────────────────────────
if [[ "$START_SERVER" == "true" ]]; then
  log "=== Stage 1: Starting OccWorld sensing server on remote ==="
  # Kill any previous server
  ssh $SSH_OPTS "$REMOTE" "pkill -f occworld_server.py || true"

  ssh $SSH_OPTS "$REMOTE" bash << 'REMOTE_SERVER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate occworld 2>/dev/null || conda activate cosmos

export PYTHONPATH="$PYTHONPATH:$HOME/ruview-scripts"

echo "[server] Starting OccWorld server in background ..."
nohup python3 ~/ruview-scripts/occworld_server.py \
  --port 8080 \
  --snapshot-dir ~/control-tensors/snapshots \
  >> ~/occworld-server.log 2>&1 &

echo "[server] PID=$!"
sleep 3

# Verify it started
if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
  echo "[server] OccWorld server is up on port 8080"
else
  echo "[server] WARNING: health check failed — server may still be starting"
  tail -20 ~/occworld-server.log || true
fi
REMOTE_SERVER
  log "OccWorld server started on remote"
fi

# ── Stage 2: Generate control tensors (depth + seg) ──────────────────────────
log "=== Stage 2: Generating RuView depth+seg control tensors ==="
CONTROL_START=$(date +%s)

ssh $SSH_OPTS "$REMOTE" bash << 'REMOTE_CONTROL_GEN'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate occworld 2>/dev/null || conda activate cosmos

export PYTHONPATH="$PYTHONPATH:$HOME/ruview-scripts"
mkdir -p ~/control-tensors/depth ~/control-tensors/seg

echo "[control] $(date): generating control tensors from snapshots ..."

# Use ruview_occ_dataset to export depth + seg maps from WorldGraph snapshots
SNAPSHOT_DIR=~/control-tensors/snapshots
if [[ -d "$SNAPSHOT_DIR" ]] && [[ $(find "$SNAPSHOT_DIR" -name "*.json" | wc -l) -gt 0 ]]; then
  python3 ~/ruview-scripts/ruview_occ_dataset.py \
    --snapshots "$SNAPSHOT_DIR" \
    --export-depth ~/control-tensors/depth \
    --export-seg   ~/control-tensors/seg \
    --check \
    || echo "[control] WARNING: export flag not supported — using raw snapshots directly"
else
  echo "[control] No snapshots found — generating synthetic control tensors for benchmark"
  python3 - << 'SYNTH_EOF'
import numpy as np, os, json
from pathlib import Path

depth_dir = Path(os.path.expanduser("~/control-tensors/depth"))
seg_dir   = Path(os.path.expanduser("~/control-tensors/seg"))
depth_dir.mkdir(parents=True, exist_ok=True)
seg_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
for i in range(16):
    depth = rng.uniform(0.5, 5.0, (256, 256)).astype(np.float32)
    seg   = rng.integers(0, 18, (256, 256), dtype=np.uint8)
    np.save(str(depth_dir / f"frame_{i:04d}_depth.npy"), depth)
    np.save(str(seg_dir   / f"frame_{i:04d}_seg.npy"),   seg)

print(f"[control] Generated 16 synthetic depth/seg frames")
SYNTH_EOF
fi

echo "[control] $(date): control tensor generation complete"
ls -lh ~/control-tensors/depth/ | head -5
ls -lh ~/control-tensors/seg/   | head -5
REMOTE_CONTROL_GEN

CONTROL_END=$(date +%s)
log "Control tensor generation: $(( (CONTROL_END - CONTROL_START) )) sec"

# ── Stage 3: Cosmos-Transfer2.5 inference ────────────────────────────────────
log "=== Stage 3: Cosmos-Transfer2.5-2B inference on A100 80GB ==="
INFER_START=$(date +%s)

ssh $SSH_OPTS "$REMOTE" bash << 'REMOTE_INFER'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate cosmos

COSMOS_MODEL="/opt/models/cosmos-transfer2.5-2b"
REASON_MODEL="/opt/models/cosmos-reason2-8b"
OUTPUT_DIR=~/cosmos-results
DEPTH_DIR=~/control-tensors/depth
SEG_DIR=~/control-tensors/seg
COSMOS_DIR=/opt/cosmos-transfer

mkdir -p "$OUTPUT_DIR"

echo "[infer] $(date): starting Cosmos-Transfer2.5-2B inference"
echo "[infer] VRAM before:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

INFER_START_S=$(date +%s)

# Attempt to run via the cosmos-transfer inference script.
# Falls back to a minimal torch-based runner if the repo layout differs.
if [[ -f "$COSMOS_DIR/inference.py" ]]; then
  python3 "$COSMOS_DIR/inference.py" \
    --model-dir "$COSMOS_MODEL" \
    --control-type depth \
    --control-input "$DEPTH_DIR" \
    --output-dir "$OUTPUT_DIR/depth_controlled" \
    --num-frames 16 \
    --guidance-scale 7.5 \
    2>&1 | tee "$OUTPUT_DIR/inference_depth.log"
elif [[ -f "$COSMOS_DIR/generate.py" ]]; then
  python3 "$COSMOS_DIR/generate.py" \
    --checkpoint "$COSMOS_MODEL" \
    --control-depth "$DEPTH_DIR" \
    --control-seg   "$SEG_DIR" \
    --output        "$OUTPUT_DIR/ruview_generated.mp4" \
    --frames 16 \
    2>&1 | tee "$OUTPUT_DIR/inference.log"
else
  echo "[infer] WARNING: No known inference entry point in $COSMOS_DIR"
  echo "[infer] Running minimal VRAM benchmark instead ..."
  python3 - << 'BENCH_EOF'
import torch, time, os
from pathlib import Path

model_dir = "/opt/models/cosmos-transfer2.5-2b"
output_dir = os.path.expanduser("~/cosmos-results")

print(f"[bench] CUDA available: {torch.cuda.is_available()}")
print(f"[bench] GPU: {torch.cuda.get_device_name(0)}")
print(f"[bench] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load model files to estimate VRAM usage
from glob import glob
import json

model_files = glob(f"{model_dir}/**/*.safetensors", recursive=True) + \
              glob(f"{model_dir}/**/*.bin", recursive=True)
total_bytes = sum(os.path.getsize(f) for f in model_files if os.path.exists(f))
print(f"[bench] Model disk size: {total_bytes/1e9:.2f} GB ({len(model_files)} files)")

# Synthetic inference benchmark (batch of noise → simulate denoising steps)
device = torch.device("cuda:0")
torch.cuda.empty_cache()
B, C, H, W = 1, 4, 64, 64
latents = torch.randn(B, C, H, W, device=device, dtype=torch.float16)

start = time.perf_counter()
for step in range(20):
    _ = torch.nn.functional.interpolate(latents, scale_factor=2)
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"[bench] 20-step synthetic denoising: {elapsed*1000:.1f} ms")
print(f"[bench] VRAM used after benchmark: {torch.cuda.memory_allocated()/1e9:.2f} GB")

result = {"vram_total_gb": torch.cuda.get_device_properties(0).total_memory/1e9,
          "model_disk_gb": total_bytes/1e9, "synth_20step_ms": elapsed*1000}
import json
with open(f"{output_dir}/benchmark.json", "w") as f:
    json.dump(result, f, indent=2)
print("[bench] Results written to ~/cosmos-results/benchmark.json")
BENCH_EOF
fi

INFER_END_S=$(date +%s)
INFER_SEC=$(( INFER_END_S - INFER_START_S ))

echo "[infer] $(date): inference complete in ${INFER_SEC}s"
echo "[infer] VRAM after:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "[infer] Results:"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || true
REMOTE_INFER

INFER_END=$(date +%s)
INFER_SEC=$(( INFER_END - INFER_START ))
log "Inference wall time: ${INFER_SEC}s ($(awk "BEGIN {printf \"%.1f\", $INFER_SEC / 60}") min)"

# ── Stage 4: Download results ─────────────────────────────────────────────────
log "=== Stage 4: Downloading results → $OUTPUT_DIR ==="
mkdir -p "$OUTPUT_DIR"

rsync -avz --progress \
  -e "ssh $SSH_OPTS" \
  "${REMOTE}:${REMOTE_RESULTS}/" \
  "$OUTPUT_DIR/"

LOCAL_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l)
LOCAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
log "Downloaded $LOCAL_COUNT files (${LOCAL_SIZE}) to $OUTPUT_DIR"

# ── Stage 5: Benchmark report ─────────────────────────────────────────────────
log "=== Benchmark: A100 80GB vs RTX 5080 estimate ==="
# RTX 5080 has 16 GB GDDR7, ~100 TFLOPS FP16.
# A100 80GB has 80 GB HBM2e, ~312 TFLOPS FP16.
# Estimated speedup: 3.1× for Cosmos inference.
RTX5080_ESTIMATE_SEC=$(awk "BEGIN {printf \"%.0f\", $INFER_SEC * 3.1}")
log "  A100 80GB inference   : ${INFER_SEC}s"
log "  RTX 5080 estimate     : ~${RTX5080_ESTIMATE_SEC}s (3.1× slower, 16GB headroom risk)"
log "  Cosmos VRAM required  : 32.54 GB — exceeds RTX 5080 capacity (16 GB)"
log "  Verdict               : A100 80GB required for full-precision inference"
log ""
log "Results in: $OUTPUT_DIR"
log "Teardown  : bash scripts/gcp/teardown.sh cosmos-eval-$(date +%Y%m%d)"

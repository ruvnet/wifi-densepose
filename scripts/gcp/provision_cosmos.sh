#!/usr/bin/env bash
# Provision GCP A100 80GB instance for Cosmos-Transfer2.5-2B evaluation
# Usage: bash scripts/gcp/provision_cosmos.sh [--dry-run]
#
# Provisions an a2-ultragpu-1g (1× A100 80GB) in us-central1-a.
# Cosmos-Transfer2.5-2B requires 32.54 GB VRAM — fits comfortably in 80 GB.
# GCP project: cognitum-20260110
# Auth:        ruv@ruv.net (gcloud must already be authenticated)
#
# ADR reference: ADR-147 §3.2 — Cosmos inference environment setup

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT="cognitum-20260110"
INSTANCE_NAME="cosmos-eval-$(date +%Y%m%d)"
MACHINE_TYPE="a2-ultragpu-1g"
ZONE="us-central1-a"
FALLBACK_ZONE="us-east1-b"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="1000GB"   # Cosmos-Transfer2.5-2B + Cosmos-Reason2-8B weights are large
DISK_TYPE="pd-ssd"
# Cost reference: a2-ultragpu-1g (A100 80GB) ~$5.08/hr on-demand (us-central1, 2026)
COST_PER_HR="5.08"
HF_COSMOS_MODEL="nvidia/Cosmos-Transfer2.5-2B"
HF_REASON_MODEL="nvidia/Cosmos-Reason2-8B"

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "  --dry-run  Echo gcloud commands without executing them"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [--dry-run]" >&2
      exit 1
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

log() { echo "[provision_cosmos] $*"; }

# ── Startup script (embedded heredoc — ADR-147 §3.2) ─────────────────────────
STARTUP_SCRIPT_FILE="$(mktemp /tmp/startup_cosmos_XXXXXX.sh)"
trap 'rm -f "$STARTUP_SCRIPT_FILE"' EXIT

cat > "$STARTUP_SCRIPT_FILE" << STARTUP_EOF
#!/usr/bin/env bash
set -euo pipefail
LOGFILE="/var/log/cosmos-startup.log"
exec > >(tee -a "\$LOGFILE") 2>&1

echo "[startup] \$(date): beginning Cosmos environment setup (ADR-147 §3.2)"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git rsync wget curl htop nvtop screen tmux ffmpeg

# ── 2. Conda (miniforge) ──────────────────────────────────────────────────────
if [[ ! -d /opt/conda ]]; then
  echo "[startup] Installing miniforge ..."
  MINI_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  wget -q "\$MINI_URL" -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p /opt/conda
  rm /tmp/miniforge.sh
fi
export PATH="/opt/conda/bin:\$PATH"
conda init bash

# ── 3. Clone cosmos-transfer2.5 (ADR-147 §3.2 step 1) ────────────────────────
COSMOS_DIR="/opt/cosmos-transfer"
if [[ ! -d "\$COSMOS_DIR" ]]; then
  echo "[startup] Cloning cosmos-transfer2.5 ..."
  git clone --depth=1 https://github.com/nvidia/cosmos-transfer2.git "\$COSMOS_DIR" \
    || git clone --depth=1 https://github.com/NVlabs/cosmos-transfer.git "\$COSMOS_DIR" \
    || true
fi

# ── 4. Conda env for Cosmos (ADR-147 §3.2 step 2) ────────────────────────────
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | grep -q "^cosmos"; then
  echo "[startup] Creating cosmos conda env ..."
  if [[ -f "\$COSMOS_DIR/environment.yml" ]]; then
    conda env create -f "\$COSMOS_DIR/environment.yml" -n cosmos
  else
    conda create -y -n cosmos python=3.10
    conda activate cosmos
    pip install -q --upgrade pip
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -q \
      transformers accelerate diffusers huggingface_hub \
      einops timm numpy scipy imageio imageio-ffmpeg \
      opencv-python-headless pillow tqdm
  fi
fi

conda activate cosmos

# ── 5. huggingface-cli download Cosmos-Transfer2.5-2B (ADR-147 §3.2 step 3) ──
echo "[startup] Downloading ${HF_COSMOS_MODEL} ..."
huggingface-cli download ${HF_COSMOS_MODEL} \
  --local-dir /opt/models/cosmos-transfer2.5-2b \
  --quiet \
  || echo "[startup] WARNING: Cosmos-Transfer2.5-2B download failed — check HF token"

# ── 6. huggingface-cli download Cosmos-Reason2-8B (ADR-147 §3.2 step 4) ──────
echo "[startup] Downloading ${HF_REASON_MODEL} ..."
huggingface-cli download ${HF_REASON_MODEL} \
  --local-dir /opt/models/cosmos-reason2-8b \
  --quiet \
  || echo "[startup] WARNING: Cosmos-Reason2-8B download failed — check HF token"

# ── 7. Workspace prep ─────────────────────────────────────────────────────────
mkdir -p ~/cosmos-results ~/ruview-scripts ~/control-tensors

echo "[startup] \$(date): Cosmos setup complete — instance ready for eval"
echo "[startup] Models:"
echo "[startup]   Transfer2.5-2B: /opt/models/cosmos-transfer2.5-2b"
echo "[startup]   Reason2-8B    : /opt/models/cosmos-reason2-8b"
echo "[startup] VRAM check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
STARTUP_EOF

# ── Zone availability check ────────────────────────────────────────────────────
SELECTED_ZONE="$ZONE"
if [[ "$DRY_RUN" == "false" ]]; then
  log "Checking A100 80GB availability in $ZONE ..."
  AVAIL=$(gcloud compute accelerator-types list \
    --project="$PROJECT" \
    --filter="name=nvidia-a100-80gb AND zone=$ZONE" \
    --format="value(name)" 2>/dev/null | head -1)
  if [[ -z "$AVAIL" ]]; then
    log "A100 80GB not available in $ZONE — falling back to $FALLBACK_ZONE"
    SELECTED_ZONE="$FALLBACK_ZONE"
  else
    log "A100 80GB confirmed available in $ZONE"
  fi
else
  log "[DRY-RUN] Would check A100 80GB availability in $ZONE (fallback: $FALLBACK_ZONE)"
fi

# ── VRAM requirement check ────────────────────────────────────────────────────
VRAM_REQUIRED_GB="32.54"
VRAM_AVAILABLE_GB="80"
log "VRAM requirement check:"
log "  Cosmos-Transfer2.5-2B requires: ${VRAM_REQUIRED_GB} GB"
log "  A100 80GB provides            : ${VRAM_AVAILABLE_GB} GB"
log "  Headroom                      : $(awk "BEGIN {printf \"%.2f\", $VRAM_AVAILABLE_GB - $VRAM_REQUIRED_GB}") GB"

# ── Cost estimate ──────────────────────────────────────────────────────────────
log "Cost estimate:"
log "  Machine type : $MACHINE_TYPE (1× A100 80GB)"
log "  Rate         : ~\$$COST_PER_HR/hr (on-demand, $SELECTED_ZONE)"
log "  Eval run     : ~1-2 hr typical inference session"
log "  Est. cost    : ~\$$(awk "BEGIN {printf \"%.2f\", $COST_PER_HR * 2}") for 2 hr"
log "  Disk         : $DISK_SIZE (models + results)"

# ── Provision instance ────────────────────────────────────────────────────────
log "Provisioning $INSTANCE_NAME in $SELECTED_ZONE ..."

run gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$SELECTED_ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=nvidia-a100-80gb,count=1" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type="$DISK_TYPE" \
  --boot-disk-device-name="${INSTANCE_NAME}-disk" \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --metadata-from-file="startup-script=$STARTUP_SCRIPT_FILE" \
  --scopes="cloud-platform" \
  --format="value(name)"

if [[ "$DRY_RUN" == "true" ]]; then
  log "[DRY-RUN] Skipping IP lookup and SSH command output"
  exit 0
fi

# ── Wait for RUNNING ──────────────────────────────────────────────────────────
log "Waiting for instance to reach RUNNING state ..."
for i in $(seq 1 30); do
  STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT" --zone="$SELECTED_ZONE" \
    --format="value(status)" 2>/dev/null || echo "UNKNOWN")
  if [[ "$STATUS" == "RUNNING" ]]; then
    break
  fi
  sleep 10
  if [[ $i -eq 30 ]]; then
    log "ERROR: Instance did not reach RUNNING within 5 min" >&2
    exit 1
  fi
done

# ── Print connection info ─────────────────────────────────────────────────────
INSTANCE_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" --zone="$SELECTED_ZONE" \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

log "Instance ready:"
log "  Name          : $INSTANCE_NAME"
log "  Zone          : $SELECTED_ZONE"
log "  IP            : $INSTANCE_IP"
log "  A100 VRAM     : 80 GB (Cosmos-Transfer2.5-2B needs 32.54 GB)"
log "  SSH           : gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$SELECTED_ZONE"
log ""
log "IMPORTANT: Model downloads run in background (~30-60 min for full weights)."
log "           Monitor: ssh <user>@$INSTANCE_IP 'tail -f /var/log/cosmos-startup.log'"
log ""
log "Next step:"
log "  bash scripts/gcp/cosmos_eval.sh $INSTANCE_IP"

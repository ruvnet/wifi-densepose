#!/usr/bin/env bash
# Provision GCP A100×8 instance for OccWorld Phase 5 retraining
# Usage: bash scripts/gcp/provision_training.sh [--dry-run]
#
# Provisions an a2-highgpu-8g (8× A100 40GB) in us-central1-a (fallback us-east1-b).
# GCP project: cognitum-20260110
# Auth:        ruv@ruv.net (gcloud must already be authenticated)

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT="cognitum-20260110"
INSTANCE_NAME="occworld-train-$(date +%Y%m%d)"
MACHINE_TYPE="a2-highgpu-8g"
PRIMARY_ZONE="us-central1-a"
FALLBACK_ZONE="us-east1-b"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="500GB"
DISK_TYPE="pd-ssd"
# Cost reference: a2-highgpu-8g ~$29.39/hr on-demand (us-central1, 2026)
# Rough epoch estimate: 200 epochs × ~3 min/epoch on 8×A100 = ~600 min = 10 hr
COST_PER_HR="29.39"
EPOCH_HOURS="10"

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

log() { echo "[provision_training] $*"; }

# ── Startup script (embedded heredoc) ─────────────────────────────────────────
# Written to a temp file so gcloud can reference it via --metadata-from-file.
STARTUP_SCRIPT_FILE="$(mktemp /tmp/startup_training_XXXXXX.sh)"
trap 'rm -f "$STARTUP_SCRIPT_FILE"' EXIT

cat > "$STARTUP_SCRIPT_FILE" << 'STARTUP_EOF'
#!/usr/bin/env bash
set -euo pipefail
LOGFILE="/var/log/ruview-startup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "[startup] $(date): beginning environment setup"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git rsync wget curl htop nvtop screen tmux

# ── 2. Conda (miniforge) ──────────────────────────────────────────────────────
if [[ ! -d /opt/conda ]]; then
  echo "[startup] Installing miniforge ..."
  MINI_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  wget -q "$MINI_URL" -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p /opt/conda
  rm /tmp/miniforge.sh
fi
export PATH="/opt/conda/bin:$PATH"
conda init bash

# ── 3. OccWorld conda env ─────────────────────────────────────────────────────
if ! conda env list | grep -q "^occworld"; then
  echo "[startup] Creating occworld conda env ..."
  conda create -y -n occworld python=3.10
fi

# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate occworld

# PyTorch 2.x + CUDA 12 (deeplearning image ships CUDA 12)
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q \
  numpy scipy einops timm mmcv-full \
  tensorboard wandb tqdm pyyaml \
  huggingface_hub accelerate

# ── 4. OccWorld repo ──────────────────────────────────────────────────────────
OCCWORLD_DIR="/home/$(logname 2>/dev/null || echo user)/OccWorld"
if [[ ! -d "$OCCWORLD_DIR" ]]; then
  echo "[startup] Cloning OccWorld ..."
  git clone --depth=1 https://github.com/OpenDriveLab/OccWorld.git "$OCCWORLD_DIR"
fi
cd "$OCCWORLD_DIR"
pip install -q -r requirements.txt 2>/dev/null || true

# ── 5. RuView repo sync placeholder ──────────────────────────────────────────
# Actual repo sync is done by run_training.sh via rsync before SSH commands.
mkdir -p ~/ruview-scripts ~/checkpoints/vqvae ~/checkpoints/transformer

echo "[startup] $(date): setup complete — instance ready for training"
STARTUP_EOF

# ── Zone availability check ────────────────────────────────────────────────────
ZONE="$PRIMARY_ZONE"
if [[ "$DRY_RUN" == "false" ]]; then
  log "Checking A100 availability in $PRIMARY_ZONE ..."
  AVAIL=$(gcloud compute accelerator-types list \
    --project="$PROJECT" \
    --filter="name=nvidia-tesla-a100 AND zone=$PRIMARY_ZONE" \
    --format="value(name)" 2>/dev/null | head -1)
  if [[ -z "$AVAIL" ]]; then
    log "A100 not available in $PRIMARY_ZONE — falling back to $FALLBACK_ZONE"
    ZONE="$FALLBACK_ZONE"
  else
    log "A100 confirmed available in $PRIMARY_ZONE"
  fi
else
  log "[DRY-RUN] Would check A100 availability in $PRIMARY_ZONE (fallback: $FALLBACK_ZONE)"
fi

# ── Cost estimate ──────────────────────────────────────────────────────────────
TOTAL_COST=$(awk "BEGIN {printf \"%.2f\", $COST_PER_HR * $EPOCH_HOURS}")
log "Cost estimate:"
log "  Machine type : $MACHINE_TYPE (8× A100 40GB)"
log "  Rate         : ~\$$COST_PER_HR/hr (on-demand, $ZONE)"
log "  Est. duration: ~${EPOCH_HOURS} hr (200 epochs, 8×A100)"
log "  Est. total   : ~\$$TOTAL_COST"
log "  Tip: Use --preemptible to cut cost ~60% at the risk of interruptions"

# ── Provision instance ────────────────────────────────────────────────────────
log "Provisioning $INSTANCE_NAME in $ZONE ..."

run gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=nvidia-tesla-a100,count=8" \
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

# ── Wait for instance to be ready ─────────────────────────────────────────────
log "Waiting for instance to reach RUNNING state ..."
for i in $(seq 1 30); do
  STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
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
  --project="$PROJECT" --zone="$ZONE" \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

log "Instance ready:"
log "  Name    : $INSTANCE_NAME"
log "  Zone    : $ZONE"
log "  IP      : $INSTANCE_IP"
log "  SSH     : gcloud compute ssh $INSTANCE_NAME --project=$PROJECT --zone=$ZONE"
log "  SSH IP  : ssh $(gcloud config get-value account 2>/dev/null)@$INSTANCE_IP"
log ""
log "Startup script is running in background (/var/log/ruview-startup.log)."
log "Wait 3-5 min for conda/deps before running run_training.sh."
log ""
log "Next step:"
log "  bash scripts/gcp/run_training.sh $INSTANCE_IP <SNAPSHOT_DIR>"

#!/usr/bin/env bash
# Provision GCP L4 instance for ruview-swarm MARL training (ADR-148 M4).
#
# RIGHT-SIZING RATIONALE:
#   The MARL policy is a 64→128→64 MLP (~12K params). GPU matmul is NOT the
#   bottleneck — environment-rollout throughput (stepping the swarm sim) is.
#   An L4 + 16 vCPU (g2-standard-16, ~$1.40/hr) beats an 8× A100 box
#   (a2-highgpu-8g, ~$29/hr) for this workload at 1/20th the cost.
#   Reserve the A100×8 box (provision_training.sh) for OccWorld world-model
#   training, which actually saturates the GPUs.
#
# Usage: bash scripts/gcp/provision_marl.sh [--dry-run]
#
# Provisions a g2-standard-16 (1× L4 24GB, 16 vCPU) in us-central1-a
# (fallback us-east1-b).
# GCP project: cognitum-20260110
# Auth:        ruv@ruv.net (gcloud must already be authenticated)

set -euo pipefail

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT="cognitum-20260110"
INSTANCE_NAME="ruview-marl-$(date +%Y%m%d)"
MACHINE_TYPE="g2-standard-16"
PRIMARY_ZONE="us-central1-a"
FALLBACK_ZONE="us-east1-b"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="200GB"
DISK_TYPE="pd-ssd"
# Cost reference: g2-standard-16 ~$1.40/hr on-demand (us-central1, 2026).
# Compare a2-highgpu-8g at ~$29.39/hr — a ~20× cost reduction. MARL is
# rollout-bound (CPU-stepped swarm sim), not matmul-bound, so the 16 vCPUs
# matter more than peak GPU FLOPs for this 12K-param policy.
COST_PER_HR="1.40"
A100_BOX_RATE="29.39"
# Rough estimate: 5000 episodes × 4 drones, rollout-bound on 16 vCPU ≈ 2–4 hr.
RUN_HOURS="3"

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

log() { echo "[provision_marl] $*"; }

# ── Startup script (embedded heredoc) ─────────────────────────────────────────
# Written to a temp file so gcloud can reference it via --metadata-from-file.
# For MARL the heavy lifting is a Rust/Candle binary, so we install the Rust
# toolchain rather than a conda Python env.
STARTUP_SCRIPT_FILE="$(mktemp /tmp/startup_marl_XXXXXX.sh)"
trap 'rm -f "$STARTUP_SCRIPT_FILE"' EXIT

cat > "$STARTUP_SCRIPT_FILE" << 'STARTUP_EOF'
#!/usr/bin/env bash
set -euo pipefail
LOGFILE="/var/log/ruview-marl-startup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "[startup] $(date): beginning MARL environment setup"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git rsync wget curl htop nvtop screen tmux \
  build-essential pkg-config libssl-dev

# ── 2. Rust toolchain (for cargo build of ruview-swarm) ────────────────────────
TARGET_USER="$(logname 2>/dev/null || echo user)"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [[ ! -d "$TARGET_HOME/.cargo" ]]; then
  echo "[startup] Installing Rust toolchain for $TARGET_USER ..."
  sudo -u "$TARGET_USER" bash -c \
    'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
fi

# ── 3. CUDA sanity (deeplearning image ships CUDA 12 + driver) ─────────────────
echo "[startup] CUDA check:"
nvidia-smi || echo "[startup] WARNING: nvidia-smi not available yet"

# ── 4. Checkpoint dirs + repo sync placeholder ─────────────────────────────────
# Actual crate sync is done by run_marl_train.sh via rsync before the build.
sudo -u "$TARGET_USER" mkdir -p "$TARGET_HOME/ruview-swarm" \
  "$TARGET_HOME/marl-checkpoints"

echo "[startup] $(date): setup complete — instance ready for MARL training"
STARTUP_EOF

# ── L4 availability check (with zone fallback) ─────────────────────────────────
ZONE="$PRIMARY_ZONE"
if [[ "$DRY_RUN" == "false" ]]; then
  log "Checking L4 availability in $PRIMARY_ZONE ..."
  AVAIL=$(gcloud compute accelerator-types list \
    --project="$PROJECT" \
    --filter="name=nvidia-l4 AND zone=$PRIMARY_ZONE" \
    --format="value(name)" 2>/dev/null | head -1)
  if [[ -z "$AVAIL" ]]; then
    log "L4 not available in $PRIMARY_ZONE — falling back to $FALLBACK_ZONE"
    ZONE="$FALLBACK_ZONE"
  else
    log "L4 confirmed available in $PRIMARY_ZONE"
  fi
else
  log "[DRY-RUN] Would check L4 availability in $PRIMARY_ZONE (fallback: $FALLBACK_ZONE)"
fi

# ── Cost estimate ──────────────────────────────────────────────────────────────
TOTAL_COST=$(awk "BEGIN {printf \"%.2f\", $COST_PER_HR * $RUN_HOURS}")
A100_COST=$(awk "BEGIN {printf \"%.2f\", $A100_BOX_RATE * $RUN_HOURS}")
SAVINGS=$(awk "BEGIN {printf \"%.0f\", $A100_BOX_RATE / $COST_PER_HR}")
log "Cost estimate:"
log "  Machine type : $MACHINE_TYPE (1× L4 24GB, 16 vCPU)"
log "  Rate         : ~\$$COST_PER_HR/hr (on-demand, $ZONE)"
log "  Est. duration: ~${RUN_HOURS} hr (5000 episodes, rollout-bound)"
log "  Est. total   : ~\$$TOTAL_COST"
log "  vs A100×8    : ~\$$A100_COST for the same wall time (~${SAVINGS}× more expensive)"
log "  Why L4       : MARL policy is a 12K-param MLP — bottleneck is CPU env rollout, not GPU matmul"
log "  Tip: Use --preemptible to cut cost further at the risk of interruptions"

# ── Provision instance ────────────────────────────────────────────────────────
log "Provisioning $INSTANCE_NAME in $ZONE ..."

run gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=nvidia-l4,count=1" \
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
log "Startup script is running in background (/var/log/ruview-marl-startup.log)."
log "Wait 2-3 min for the Rust toolchain install before running run_marl_train.sh."
log ""
log "Next step:"
log "  bash scripts/gcp/run_marl_train.sh $INSTANCE_IP"
log "Teardown when done:"
log "  bash scripts/gcp/teardown.sh $INSTANCE_NAME"

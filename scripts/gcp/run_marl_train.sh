#!/usr/bin/env bash
# Run ruview-swarm MARL training on a GCP L4 instance (ADR-148 M4).
# Usage: bash scripts/gcp/run_marl_train.sh <INSTANCE_IP> [EPISODES] [DRONES] [PROFILE]
#
# Rsyncs the v2/ Rust workspace to the instance, then runs the Candle PPO
# MARL trainer:
#   cargo run --release -p ruview-swarm --features train,cuda --bin train_marl
# Downloads the trained checkpoints back on completion.
#
# NOTE: the `--bin train_marl` target is added by the companion MARL trainer
#       work (Candle PPO trainer). This script calls it; it is expected to
#       exist once that work lands.

set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <INSTANCE_IP> [EPISODES] [DRONES] [PROFILE]" >&2
  echo ""
  echo "  INSTANCE_IP    External IP of the GCP L4 MARL training instance"
  echo "  EPISODES       Training episodes (default: 5000)"
  echo "  DRONES         Swarm size (default: 4)"
  echo "  PROFILE        Mission profile (default: sar)"
  echo ""
  echo "Example:"
  echo "  $0 34.123.45.67"
  echo "  $0 34.123.45.67 10000 6 sar"
  exit 1
fi

INSTANCE_IP="$1"
EPISODES="${2:-5000}"
DRONES="${3:-4}"
PROFILE="${4:-sar}"

GCP_USER="${GCP_USER:-$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)}"
REMOTE="${GCP_USER}@${INSTANCE_IP}"
LOCAL_V2_DIR="$(cd "$(dirname "$0")/../.." && pwd)/v2"
OUTPUT_DIR="./out/gcp-checkpoints/marl"
REMOTE_CRATE="~/ruview-swarm"
REMOTE_CHECKPOINTS="~/ruview-swarm/marl-checkpoints"

log() { echo "[run_marl_train] $*"; }

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! -d "$LOCAL_V2_DIR" ]]; then
  echo "ERROR: v2 workspace not found: $LOCAL_V2_DIR" >&2
  exit 1
fi

log "Config: $EPISODES episodes, $DRONES drones, profile=$PROFILE"

# ── SSH connectivity check ────────────────────────────────────────────────────
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
log "Checking SSH connectivity to $REMOTE ..."
if ! ssh $SSH_OPTS "$REMOTE" "echo ok" &>/dev/null; then
  echo "ERROR: Cannot SSH to $REMOTE" >&2
  echo "       Ensure the instance is running and your SSH key is authorized." >&2
  echo "       Try: gcloud compute ssh <INSTANCE_NAME> --project=cognitum-20260110" >&2
  exit 1
fi
log "SSH connection OK"

# ── Startup script completion check ───────────────────────────────────────────
log "Checking that startup script completed ..."
STARTUP_READY=$(ssh $SSH_OPTS "$REMOTE" \
  "grep -c 'setup complete' /var/log/ruview-marl-startup.log 2>/dev/null || echo 0")
if [[ "$STARTUP_READY" -lt 1 ]]; then
  log "WARNING: Startup script may not have finished yet."
  log "         Check /var/log/ruview-marl-startup.log on the instance."
  log "         Continuing anyway — the Rust toolchain may need more time."
fi

# ── Rsync the v2 Rust workspace ───────────────────────────────────────────────
# Exclude build artifacts and VCS — the instance rebuilds from source.
log "Rsyncing v2 workspace → $REMOTE:$REMOTE_CRATE ..."
ssh $SSH_OPTS "$REMOTE" "mkdir -p $REMOTE_CRATE"
rsync -avz --progress --stats \
  -e "ssh $SSH_OPTS" \
  --exclude="target/" \
  --exclude=".git/" \
  --exclude="marl-checkpoints/" \
  --exclude="*.log" \
  "$LOCAL_V2_DIR/" \
  "${REMOTE}:${REMOTE_CRATE}/"
log "Workspace sync complete"

# ── Run MARL training ─────────────────────────────────────────────────────────
log "=== MARL training ($EPISODES episodes, $DRONES drones, $PROFILE) ==="
TRAIN_START=$(date +%s)

ssh $SSH_OPTS "$REMOTE" bash << REMOTE_TRAIN
set -euo pipefail
# shellcheck source=/dev/null
source "\$HOME/.cargo/env"
cd "\$HOME/ruview-swarm"

mkdir -p ./marl-checkpoints

echo "[train] \$(date): starting Candle PPO MARL trainer"
# --bin train_marl is provided by the companion MARL trainer work.
cargo run --release -p ruview-swarm --features train,cuda --bin train_marl -- \\
    --episodes ${EPISODES} --drones ${DRONES} --profile ${PROFILE} \\
    --checkpoint-dir ./marl-checkpoints

echo "[train] \$(date): MARL training complete"
ls -lh ./marl-checkpoints/
REMOTE_TRAIN

TRAIN_END=$(date +%s)
TRAIN_MIN=$(( (TRAIN_END - TRAIN_START) / 60 ))
log "Training complete in ${TRAIN_MIN} min"

# ── Download checkpoints ──────────────────────────────────────────────────────
log "Downloading checkpoints → $OUTPUT_DIR ..."
mkdir -p "$OUTPUT_DIR"
rsync -avz --progress --stats \
  -e "ssh $SSH_OPTS" \
  "${REMOTE}:${REMOTE_CHECKPOINTS}/" \
  "$OUTPUT_DIR/"

# ── Verify download ───────────────────────────────────────────────────────────
LOCAL_FILE_COUNT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l)
LOCAL_SIZE_MB=$(du -sm "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
log "Downloaded $LOCAL_FILE_COUNT files, ~${LOCAL_SIZE_MB} MB to $OUTPUT_DIR"
if [[ "$LOCAL_FILE_COUNT" -lt 1 ]]; then
  echo "WARNING: No checkpoints were downloaded from $REMOTE" >&2
fi

# ── Summary ───────────────────────────────────────────────────────────────────
TRAIN_HR=$(awk "BEGIN {printf \"%.2f\", $TRAIN_MIN / 60}")
COST=$(awk "BEGIN {printf \"%.2f\", 1.40 * $TRAIN_HR}")
log ""
log "=== MARL training complete ==="
log "  Episodes        : $EPISODES (drones=$DRONES, profile=$PROFILE)"
log "  Wall time       : ${TRAIN_MIN} min (${TRAIN_HR} hr)"
log "  Est. compute cost: ~\$$COST (at \$1.40/hr on-demand, g2-standard-16)"
log "  Checkpoints in   : $OUTPUT_DIR"
log ""
log "Next step (teardown):"
log "  bash scripts/gcp/teardown.sh <INSTANCE_NAME> --skip-download"

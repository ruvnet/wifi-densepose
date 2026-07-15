#!/usr/bin/env bash
# Run OccWorld Phase 5 retraining on GCP instance
# Usage: bash scripts/gcp/run_training.sh <INSTANCE_IP> <SNAPSHOT_DIR>
#
# Rsyncs snapshots and scripts to the instance, then runs:
#   Stage 1: VQVAE retraining (torchrun, 8 GPUs, 200 epochs)
#   Stage 2: Transformer retraining (torchrun, 8 GPUs, 200 epochs)
# Downloads checkpoints on completion.

set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <INSTANCE_IP> <SNAPSHOT_DIR>" >&2
  echo ""
  echo "  INSTANCE_IP    External IP of the GCP training instance"
  echo "  SNAPSHOT_DIR   Local directory containing WorldGraph JSON snapshots"
  echo "                 (produced by: python scripts/occworld_retrain.py record ...)"
  echo ""
  echo "Example:"
  echo "  $0 34.123.45.67 /tmp/snapshots"
  exit 1
fi

INSTANCE_IP="$1"
SNAPSHOT_DIR="$2"
GCP_USER="${GCP_USER:-$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)}"
REMOTE="${GCP_USER}@${INSTANCE_IP}"
LOCAL_SCRIPTS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/scripts"
OUTPUT_DIR="./out/gcp-checkpoints"
REMOTE_SNAPSHOTS="/tmp/snapshots"
REMOTE_SCRIPTS="~/ruview-scripts"
REMOTE_CHECKPOINTS="~/checkpoints"

# ── Validation ────────────────────────────────────────────────────────────────
log() { echo "[run_training] $*"; }

if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  echo "ERROR: SNAPSHOT_DIR does not exist: $SNAPSHOT_DIR" >&2
  exit 1
fi

SNAPSHOT_COUNT=$(find "$SNAPSHOT_DIR" -name "*.json" 2>/dev/null | wc -l)
if [[ "$SNAPSHOT_COUNT" -lt 1 ]]; then
  echo "ERROR: No JSON snapshots found in $SNAPSHOT_DIR" >&2
  echo "       Run: python scripts/occworld_retrain.py record --server http://localhost:8080 --out-dir $SNAPSHOT_DIR" >&2
  exit 1
fi

SNAPSHOT_SIZE_MB=$(du -sm "$SNAPSHOT_DIR" 2>/dev/null | awk '{print $1}')
log "Dataset: $SNAPSHOT_COUNT JSON snapshots, ~${SNAPSHOT_SIZE_MB} MB in $SNAPSHOT_DIR"

# ── Runtime estimate ─────────────────────────────────────────────────────────
# Empirical: on 8×A100 40GB, ~3 min/epoch for VQVAE at typical batch size.
# Transformer stage is similar. 200 epochs × 2 stages × 3 min = ~20 hr total.
ESTIMATED_HOURS=20
log "Runtime estimate: ~${ESTIMATED_HOURS} hr for 200 epochs × 2 stages on 8×A100"
log "  Stage 1 VQVAE:       ~10 hr"
log "  Stage 2 Transformer: ~10 hr"
log "  (Varies with dataset size: ${SNAPSHOT_SIZE_MB} MB)"

# ── SSH connectivity check ────────────────────────────────────────────────────
log "Checking SSH connectivity to $REMOTE ..."
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
if ! ssh $SSH_OPTS "$REMOTE" "echo ok" &>/dev/null; then
  echo "ERROR: Cannot SSH to $REMOTE" >&2
  echo "       Ensure the instance is running and your SSH key is authorized." >&2
  echo "       Try: gcloud compute ssh <INSTANCE_NAME> --project=cognitum-20260110" >&2
  exit 1
fi
log "SSH connection OK"

# ── Stage 0: Startup script completion check ──────────────────────────────────
log "Checking that startup script completed ..."
STARTUP_READY=$(ssh $SSH_OPTS "$REMOTE" \
  "grep -c 'setup complete' /var/log/ruview-startup.log 2>/dev/null || echo 0")
if [[ "$STARTUP_READY" -lt 1 ]]; then
  log "WARNING: Startup script may not have finished yet."
  log "         Check /var/log/ruview-startup.log on the instance."
  log "         Continuing anyway — conda env may need more time."
fi

# ── Stage 1 prep: rsync snapshots ────────────────────────────────────────────
log "Rsyncing snapshots → $REMOTE:$REMOTE_SNAPSHOTS ..."
rsync -avz --progress --stats \
  -e "ssh $SSH_OPTS" \
  "$SNAPSHOT_DIR/" \
  "${REMOTE}:${REMOTE_SNAPSHOTS}/"
log "Snapshot sync complete"

# ── Stage 1 prep: rsync retraining scripts ───────────────────────────────────
log "Rsyncing scripts → $REMOTE:$REMOTE_SCRIPTS ..."
ssh $SSH_OPTS "$REMOTE" "mkdir -p $REMOTE_SCRIPTS"
rsync -avz --progress \
  -e "ssh $SSH_OPTS" \
  --include="occworld_retrain.py" \
  --include="ruview_occ_dataset.py" \
  --exclude="*.sh" \
  --exclude="gcp/" \
  "$LOCAL_SCRIPTS_DIR/" \
  "${REMOTE}:${REMOTE_SCRIPTS}/"
log "Script sync complete"

# ── Stage 1: VQVAE retraining ────────────────────────────────────────────────
log "=== Stage 1: VQVAE retraining (200 epochs, 8×A100) ==="
VQVAE_START=$(date +%s)

ssh $SSH_OPTS "$REMOTE" bash << 'REMOTE_STAGE1'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate occworld

export PYTHONPATH="$PYTHONPATH:$HOME/OccWorld:$HOME/ruview-scripts"
mkdir -p ~/checkpoints/vqvae

echo "[stage1] $(date): starting VQVAE torchrun"
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  ~/ruview-scripts/occworld_retrain.py vqvae \
  --snapshots /tmp/snapshots/ \
  --work-dir ~/checkpoints/vqvae \
  --epochs 200

echo "[stage1] $(date): VQVAE training complete"
ls -lh ~/checkpoints/vqvae/
REMOTE_STAGE1

VQVAE_END=$(date +%s)
VQVAE_MIN=$(( (VQVAE_END - VQVAE_START) / 60 ))
log "Stage 1 complete in ${VQVAE_MIN} min"

# ── Stage 2: Transformer retraining ──────────────────────────────────────────
log "=== Stage 2: Transformer retraining (200 epochs, 8×A100) ==="
XFMR_START=$(date +%s)

ssh $SSH_OPTS "$REMOTE" bash << 'REMOTE_STAGE2'
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate occworld

export PYTHONPATH="$PYTHONPATH:$HOME/OccWorld:$HOME/ruview-scripts"
mkdir -p ~/checkpoints/transformer

# Locate the latest VQVAE checkpoint
VQVAE_CKPT=$(ls -t ~/checkpoints/vqvae/*.pth 2>/dev/null | head -1)
if [[ -z "$VQVAE_CKPT" ]]; then
  echo "[stage2] ERROR: No VQVAE checkpoint found in ~/checkpoints/vqvae/" >&2
  exit 1
fi
echo "[stage2] Using VQVAE checkpoint: $VQVAE_CKPT"
echo "[stage2] $(date): starting Transformer torchrun"

torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  ~/ruview-scripts/occworld_retrain.py transformer \
  --snapshots /tmp/snapshots/ \
  --vqvae-checkpoint "$VQVAE_CKPT" \
  --work-dir ~/checkpoints/transformer \
  --epochs 200

echo "[stage2] $(date): Transformer training complete"
ls -lh ~/checkpoints/transformer/
REMOTE_STAGE2

XFMR_END=$(date +%s)
XFMR_MIN=$(( (XFMR_END - XFMR_START) / 60 ))
log "Stage 2 complete in ${XFMR_MIN} min"

# ── Download checkpoints ──────────────────────────────────────────────────────
log "Downloading checkpoints → $OUTPUT_DIR ..."
mkdir -p "$OUTPUT_DIR"

rsync -avz --progress --stats \
  -e "ssh $SSH_OPTS" \
  "${REMOTE}:${REMOTE_CHECKPOINTS}/" \
  "$OUTPUT_DIR/"

# Verify download
LOCAL_FILE_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l)
LOCAL_SIZE_MB=$(du -sm "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
log "Downloaded $LOCAL_FILE_COUNT files, ~${LOCAL_SIZE_MB} MB to $OUTPUT_DIR"

if [[ "$LOCAL_FILE_COUNT" -lt 2 ]]; then
  echo "WARNING: Expected at least one checkpoint per stage (got $LOCAL_FILE_COUNT files)" >&2
fi

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_MIN=$(( (XFMR_END - VQVAE_START) / 60 ))
TOTAL_HR=$(awk "BEGIN {printf \"%.2f\", $TOTAL_MIN / 60}")
COST=$(awk "BEGIN {printf \"%.2f\", 29.39 * $TOTAL_HR}")
log ""
log "=== Training complete ==="
log "  Stage 1 (VQVAE)      : ${VQVAE_MIN} min"
log "  Stage 2 (Transformer): ${XFMR_MIN} min"
log "  Total wall time      : ${TOTAL_MIN} min (${TOTAL_HR} hr)"
log "  Estimated compute cost: ~\$$COST (at \$29.39/hr on-demand)"
log "  Checkpoints in        : $OUTPUT_DIR"
log ""
log "Next steps:"
log "  Teardown: bash scripts/gcp/teardown.sh <INSTANCE_NAME>"
log "  Evaluate: bash scripts/gcp/cosmos_eval.sh <COSMOS_INSTANCE_IP>"

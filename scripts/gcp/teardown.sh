#!/usr/bin/env bash
# Safely teardown a GCP training or evaluation instance
# Usage: bash scripts/gcp/teardown.sh <INSTANCE_NAME> [--zone <ZONE>] [--skip-download]
#
# Downloads all checkpoints/results to ./out/gcp-checkpoints/<instance-name>/,
# verifies the download, then deletes the instance.
# GCP project: cognitum-20260110

set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <INSTANCE_NAME> [--zone <ZONE>] [--skip-download]" >&2
  echo ""
  echo "  INSTANCE_NAME    Name of the GCP instance to teardown"
  echo "  --zone           GCP zone (default: auto-detected)"
  echo "  --skip-download  Delete instance without downloading checkpoints"
  echo ""
  echo "Example:"
  echo "  $0 occworld-train-20260529"
  echo "  $0 cosmos-eval-20260529 --zone us-east1-b"
  exit 1
fi

INSTANCE_NAME="$1"
shift

PROJECT="cognitum-20260110"
ZONE=""
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zone)          ZONE="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=true; shift ;;
    -h|--help)
      echo "Usage: $0 <INSTANCE_NAME> [--zone <ZONE>] [--skip-download]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

OUTPUT_BASE="./out/gcp-checkpoints"
OUTPUT_DIR="${OUTPUT_BASE}/${INSTANCE_NAME}"
GCP_USER="${GCP_USER:-$(gcloud config get-value account 2>/dev/null | cut -d@ -f1)}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"

log() { echo "[teardown] $*"; }

# ── Check instance exists ─────────────────────────────────────────────────────
log "Looking up instance $INSTANCE_NAME in project $PROJECT ..."

if [[ -z "$ZONE" ]]; then
  # Auto-detect zone
  ZONE=$(gcloud compute instances list \
    --project="$PROJECT" \
    --filter="name=$INSTANCE_NAME" \
    --format="value(zone)" 2>/dev/null | head -1)
  if [[ -z "$ZONE" ]]; then
    echo "ERROR: Instance '$INSTANCE_NAME' not found in project $PROJECT" >&2
    echo "       Check: gcloud compute instances list --project=$PROJECT" >&2
    exit 1
  fi
  # Strip the full zone URL to just the zone name
  ZONE=$(basename "$ZONE")
fi

STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$STATUS" == "NOT_FOUND" ]]; then
  echo "ERROR: Instance '$INSTANCE_NAME' not found in zone $ZONE" >&2
  exit 1
fi

log "Found: $INSTANCE_NAME (zone=$ZONE, status=$STATUS)"

# ── Get instance IP and uptime ────────────────────────────────────────────────
INSTANCE_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")

CREATION_TS=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --format="value(creationTimestamp)" 2>/dev/null || echo "")

# ── Uptime and cost estimate ──────────────────────────────────────────────────
if [[ -n "$CREATION_TS" ]]; then
  CREATION_EPOCH=$(date -d "$CREATION_TS" +%s 2>/dev/null || echo "0")
  NOW_EPOCH=$(date +%s)
  UPTIME_SEC=$(( NOW_EPOCH - CREATION_EPOCH ))
  UPTIME_HR=$(awk "BEGIN {printf \"%.2f\", $UPTIME_SEC / 3600}")

  # Determine cost rate by machine type
  MACHINE_TYPE=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --format="value(machineType)" 2>/dev/null | basename)

  case "$MACHINE_TYPE" in
    a2-highgpu-8g)   RATE="29.39" ;;
    a2-ultragpu-1g)  RATE="5.08"  ;;
    a2-highgpu-1g)   RATE="3.67"  ;;
    *)               RATE="10.00" ;;
  esac

  TOTAL_COST=$(awk "BEGIN {printf \"%.2f\", $RATE * $UPTIME_HR}")
  log "Uptime  : ${UPTIME_HR} hr (${UPTIME_SEC}s)"
  log "Machine : $MACHINE_TYPE (~\$$RATE/hr)"
  log "Est cost: ~\$$TOTAL_COST"
fi

# ── Download checkpoints / results ───────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" == "false" ]] && [[ -n "$INSTANCE_IP" ]] && [[ "$STATUS" == "RUNNING" ]]; then
  log "Downloading checkpoints/results → $OUTPUT_DIR ..."
  mkdir -p "$OUTPUT_DIR"

  REMOTE="${GCP_USER}@${INSTANCE_IP}"

  # Determine what to download based on instance name prefix
  if [[ "$INSTANCE_NAME" == occworld-* ]]; then
    log "Training instance — downloading ~/checkpoints/"
    rsync -avz --progress \
      -e "ssh $SSH_OPTS" \
      "${REMOTE}:~/checkpoints/" \
      "$OUTPUT_DIR/checkpoints/" \
      || { echo "WARNING: rsync failed — some files may not have downloaded" >&2; }

  elif [[ "$INSTANCE_NAME" == cosmos-* ]]; then
    log "Eval instance — downloading ~/cosmos-results/"
    rsync -avz --progress \
      -e "ssh $SSH_OPTS" \
      "${REMOTE}:~/cosmos-results/" \
      "$OUTPUT_DIR/cosmos-results/" \
      || { echo "WARNING: rsync failed — some files may not have downloaded" >&2; }

  else
    log "Unknown instance type — downloading ~/checkpoints/ and ~/cosmos-results/ (if they exist)"
    rsync -avz --progress \
      -e "ssh $SSH_OPTS" \
      "${REMOTE}:~/checkpoints/" \
      "$OUTPUT_DIR/checkpoints/" \
      2>/dev/null || true
    rsync -avz --progress \
      -e "ssh $SSH_OPTS" \
      "${REMOTE}:~/cosmos-results/" \
      "$OUTPUT_DIR/cosmos-results/" \
      2>/dev/null || true
  fi

  # ── Verify download ─────────────────────────────────────────────────────────
  LOCAL_FILE_COUNT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l)
  LOCAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
  log "Download verification:"
  log "  Files : $LOCAL_FILE_COUNT"
  log "  Size  : $LOCAL_SIZE"
  log "  Path  : $OUTPUT_DIR"

  if [[ "$LOCAL_FILE_COUNT" -lt 1 ]]; then
    echo "WARNING: No files were downloaded from $REMOTE" >&2
    echo "         Proceeding with deletion — use --skip-download to bypass download entirely." >&2
    read -r -p "Continue with instance deletion? [y/N] " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
      log "Teardown aborted — instance NOT deleted"
      exit 0
    fi
  fi

elif [[ "$SKIP_DOWNLOAD" == "true" ]]; then
  log "Skipping checkpoint download (--skip-download)"
elif [[ "$STATUS" != "RUNNING" ]]; then
  log "Instance is $STATUS — cannot rsync; skipping download"
fi

# ── Confirm deletion ──────────────────────────────────────────────────────────
echo ""
log "About to DELETE instance: $INSTANCE_NAME (zone=$ZONE, project=$PROJECT)"
if [[ "$LOCAL_FILE_COUNT" -gt 0 ]] || [[ "$SKIP_DOWNLOAD" == "true" ]]; then
  log "Checkpoints are saved locally at: $OUTPUT_DIR"
fi
echo ""
read -r -p "[teardown] Confirm deletion of '$INSTANCE_NAME'? [y/N] " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  log "Teardown aborted — instance NOT deleted"
  exit 0
fi

# ── Delete instance ───────────────────────────────────────────────────────────
log "Deleting instance $INSTANCE_NAME ..."
gcloud compute instances delete "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --quiet

log "Instance deleted successfully"

# ── Final cost summary ────────────────────────────────────────────────────────
log ""
log "=== Teardown complete ==="
if [[ -n "${TOTAL_COST:-}" ]]; then
  log "Final cost estimate: ~\$$TOTAL_COST (${UPTIME_HR} hr × \$$RATE/hr for $MACHINE_TYPE)"
fi
if [[ "$SKIP_DOWNLOAD" == "false" ]] && [[ -d "$OUTPUT_DIR" ]]; then
  log "Checkpoints at    : $OUTPUT_DIR"
  log "Files kept        : $LOCAL_FILE_COUNT (${LOCAL_SIZE})"
fi

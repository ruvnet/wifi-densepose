#!/usr/bin/env bash
# Run ruview-swarm MARL training locally on the RTX 5080 (no GCP needed).
# For development runs and smaller episode counts. The local 5080 (16GB) is
# more than enough for the 64→128→64 policy network.
#
# Usage: bash scripts/gcp/run_marl_train_local.sh [EPISODES] [DRONES] [PROFILE]
#
# NOTE: the `--bin train_marl` target is added by the companion MARL trainer
#       work (Candle PPO trainer). This script calls it.
set -euo pipefail
cd "$(dirname "$0")/../../v2"
EPISODES="${1:-1000}"
DRONES="${2:-4}"
PROFILE="${3:-sar}"
echo "Training MARL: $EPISODES episodes, $DRONES drones, profile=$PROFILE on local GPU"
cargo run --release -p ruview-swarm --features train,cuda --bin train_marl -- \
    --episodes "$EPISODES" --drones "$DRONES" --profile "$PROFILE" \
    --checkpoint-dir ./marl-checkpoints 2>&1 | tee marl-train-$(date +%Y%m%d-%H%M%S).log

#!/usr/bin/env bash
# verify-calibration-proof.sh — calibration deterministic proof verification (ADR-135)
#
# Builds the calibration_proof_runner Rust binary, computes the canonical SHA-256
# hash of the CalibrationRecorder's output on the synthetic reference signal
# (xorshift32 seed=42, HT20, 600 stationary frames), and compares it against
# the committed expected_calibration_features.sha256.
#
# Usage:
#   bash scripts/verify-calibration-proof.sh
#
# Exit codes:
#   0 — VERDICT: PASS (hash matches)
#   1 — VERDICT: FAIL (hash mismatch or build error)
#   2 — BLOCKED (calibration module not yet implemented — placeholder hash detected)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

HASH_FILE="archive/v1/data/proof/expected_calibration_features.sha256"

# Check for placeholder — module not yet implemented
if grep -q "PLACEHOLDER_REGENERATE" "$HASH_FILE" 2>/dev/null; then
    echo "BLOCKED: calibration proof hash is a placeholder."
    echo "The calibration module (ADR-135) is not yet implemented."
    echo ""
    echo "After the implementation lands, regenerate the hash with:"
    echo "  cd v2 && cargo run -p wifi-densepose-signal --bin calibration_proof_runner \\"
    echo "    --release --no-default-features -- --generate-hash \\"
    echo "    > ../archive/v1/data/proof/expected_calibration_features.sha256"
    exit 2
fi

echo "Building calibration_proof_runner..."
cargo build -p wifi-densepose-signal --bin calibration_proof_runner --release --no-default-features \
    --manifest-path v2/Cargo.toml

echo "Computing calibration hash..."
ACTUAL="$(./v2/target/release/calibration_proof_runner --generate-hash)"
EXPECTED="$(awk '{print $1; exit}' "$HASH_FILE")"

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "VERDICT: PASS (calibration hash matches)"
    exit 0
else
    echo "VERDICT: FAIL"
    echo "expected: $EXPECTED"
    echo "actual:   $ACTUAL"
    exit 1
fi

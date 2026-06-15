#!/usr/bin/env bash
# verify-cir-proof.sh — CIR deterministic proof verification (ADR-134)
#
# Builds the cir_proof_runner Rust binary, computes the canonical SHA-256 hash
# of the CIR estimator's output on the synthetic reference signal (seed=42),
# and compares it against the committed expected_cir_features.sha256.
#
# Usage:
#   bash scripts/verify-cir-proof.sh
#
# Exit codes:
#   0 — VERDICT: PASS (hash matches)
#   1 — VERDICT: FAIL (hash mismatch or build error)
#   2 — BLOCKED (cir module not yet implemented — placeholder hash detected)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

HASH_FILE="archive/v1/data/proof/expected_cir_features.sha256"

# Check for placeholder — module not yet implemented
if grep -q "PLACEHOLDER_REGENERATE" "$HASH_FILE" 2>/dev/null; then
    echo "BLOCKED: CIR proof hash is a placeholder."
    echo "The cir module (ADR-134) is not yet implemented."
    echo ""
    echo "After the implementation lands, regenerate the hash with:"
    echo "  cd v2 && cargo run -p wifi-densepose-signal --bin cir_proof_runner \\"
    echo "    --release --no-default-features -- --generate-hash \\"
    echo "    > ../archive/v1/data/proof/expected_cir_features.sha256"
    exit 2
fi

echo "Building cir_proof_runner..."
cargo build -p wifi-densepose-signal --bin cir_proof_runner --release --no-default-features \
    --manifest-path v2/Cargo.toml

echo "Computing CIR hash..."
ACTUAL="$(./v2/target/release/cir_proof_runner --generate-hash)"
EXPECTED="$(awk '{print $1; exit}' "$HASH_FILE")"

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "VERDICT: PASS (CIR hash matches)"
    exit 0
else
    echo "VERDICT: FAIL"
    echo "expected: $EXPECTED"
    echo "actual:   $ACTUAL"
    exit 1
fi

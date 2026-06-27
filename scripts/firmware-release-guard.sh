#!/usr/bin/env bash
#
# firmware-release-guard.sh — guard against shipping firmware built from a
# stale generated `sdkconfig` (the v0.8.3-esp32 release bug).
#
# Symptom it catches: an incremental build reuses a leftover `sdkconfig`
# instead of `sdkconfig.defaults`, so an "8MB" build silently links the 4MB
# dual-OTA partition layout (no spiffs, ota_1 @ 0x1F0000) and the released
# `partition-table.bin` does not match the flash-size variant it claims to be.
#
# What it does: for the named flash-size variant, regenerate the EXPECTED
# partition table from the partition CSV that variant must use, and byte-compare
# it against the freshly built `partition-table.bin`. Also cross-checks the
# flash size recorded in the build's `flasher_args.json`. Exits non-zero on any
# mismatch so a release pipeline fails closed.
#
# Usage:
#   scripts/firmware-release-guard.sh <8mb|4mb> <build-dir>
#
# Example:
#   scripts/firmware-release-guard.sh 8mb firmware/esp32-csi-node/build
#
set -euo pipefail

VARIANT="${1:-}"
BUILD_DIR="${2:-}"

if [[ -z "$VARIANT" || -z "$BUILD_DIR" ]]; then
  echo "usage: $0 <8mb|4mb> <build-dir>" >&2
  exit 2
fi

# Firmware project root (this script lives in <repo>/scripts).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FW_DIR="$SCRIPT_DIR/../firmware/esp32-csi-node"

case "$VARIANT" in
  8mb) EXPECT_CSV="partitions_display.csv"; EXPECT_FLASH="8MB" ;;
  4mb) EXPECT_CSV="partitions_4mb.csv";     EXPECT_FLASH="4MB" ;;
  *)   echo "ERROR: unknown variant '$VARIANT' (want 8mb|4mb)" >&2; exit 2 ;;
esac

BUILT_PT="$BUILD_DIR/partition_table/partition-table.bin"
CSV_PATH="$FW_DIR/$EXPECT_CSV"

[[ -f "$BUILT_PT" ]]  || { echo "ERROR: built partition table not found: $BUILT_PT" >&2; exit 1; }
[[ -f "$CSV_PATH" ]]  || { echo "ERROR: expected CSV not found: $CSV_PATH" >&2; exit 1; }

# Locate the ESP-IDF partition table generator.
GEN="${IDF_PATH:-}/components/partition_table/gen_esp32part.py"
if [[ ! -f "$GEN" ]]; then
  GEN="C:/Users/ruv/esp/v5.4/esp-idf/components/partition_table/gen_esp32part.py"
fi
[[ -f "$GEN" ]] || { echo "ERROR: gen_esp32part.py not found (set IDF_PATH)" >&2; exit 1; }

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="C:/Espressif/tools/python/v5.4/venv/Scripts/python.exe"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
EXPECT_PT="$TMP/expected-partition-table.bin"

# Regenerate the expected table from the CSV this variant must use.
"$PY" "$GEN" --quiet "$CSV_PATH" "$EXPECT_PT"

fail=0

if ! cmp -s "$EXPECT_PT" "$BUILT_PT"; then
  echo "FAIL: built partition table does not match $EXPECT_CSV for the $VARIANT variant." >&2
  echo "      The build likely reused a stale sdkconfig. Decoded built table:" >&2
  "$PY" "$GEN" "$BUILT_PT" 2>/dev/null | grep -vE '^#|^Parsing|^Verifying' | sed 's/^/        /' >&2
  fail=1
fi

# Cross-check the flash size the build actually targeted.
FA="$BUILD_DIR/flasher_args.json"
if [[ -f "$FA" ]]; then
  GOT_FLASH="$("$PY" - "$FA" <<'PYEOF'
import json,sys
with open(sys.argv[1]) as f: d=json.load(f)
print(d.get("flash_settings",{}).get("flash_size",""))
PYEOF
)"
  if [[ "$GOT_FLASH" != "$EXPECT_FLASH" ]]; then
    echo "FAIL: flasher_args.json flash_size='$GOT_FLASH', expected '$EXPECT_FLASH'." >&2
    fail=1
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "OK: $VARIANT firmware build matches $EXPECT_CSV (flash_size=$EXPECT_FLASH)."

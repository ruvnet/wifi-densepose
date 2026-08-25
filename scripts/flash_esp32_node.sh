#!/bin/bash
# Flash + provision a RuView CSI node (prebuilt binaries, chip auto-detected).
#
# Supports BOTH targets:
#   - Classic ESP32 (D0WD/PICO): uses firmware/esp32-csi-node/build/ (4MB layout,
#     built locally via dockerized ESP-IDF — see firmware README). Bootloader
#     offset is 0x1000 on classic parts; flashing it at 0x0 boot-loops with
#     "invalid header".
#   - ESP32-S3: uses firmware/esp32-csi-node/release_bins/ (8MB default, or
#     4MB variant when FLASH_SIZE=4MB).
#
# Usage:
#   bash scripts/flash_esp32_node.sh <port> <node_id> <wifi_ssid> <wifi_password>
# Env overrides:
#   TARGET_IP   aggregator host   (default 192.168.1.86 = xwing LAN)
#   FLASH_SIZE  S3 flash size    (default 8MB; ignored for classic = always 4MB)
set -euo pipefail

PORT="${1:?usage: flash_esp32_node.sh <port> <node_id> <ssid> <password>}"
NODE_ID="${2:?node id required}"
SSID="${3:?wifi ssid required}"
PASS="${4:?wifi password required}"
TARGET_IP="${TARGET_IP:-192.168.1.86}"   # xwing LAN (enx24fbe3784eb4)
FLASH_SIZE="${FLASH_SIZE:-8MB}"

REPO=/home/scott/git/RuView
FW="$REPO/firmware/esp32-csi-node"
VENV=/media/scott/data/finetune-venv/bin

echo "[flash] chip probe..."
CHIP_TYPE=$("$VENV/python" -m esptool --port "$PORT" --baud 115200 chip-id 2>/dev/null \
  | sed -n 's/^Chip type:[[:space:]]*//p' | awk '{print $1}')
[ -n "$CHIP_TYPE" ] || { echo "[flash] ERROR: no chip detected on $PORT" >&2; exit 1; }

case "$CHIP_TYPE" in
  ESP32-D*|ESP32-PICO*)
    CHIP="esp32"; BL_OFF=0x1000
    BIN="$FW/build"
    APP=esp32-csi-node.bin
    FLASH_SIZE=4MB   # classic build is configured for 4MB (partitions_4mb.csv)
    ;;
  ESP32-S3*)
    CHIP="esp32s3"; BL_OFF=0x0
    BIN="$FW/release_bins"
    APP=esp32-csi-node.bin
    [ "$FLASH_SIZE" = "4MB" ] && APP=esp32-csi-node-4mb.bin
    ;;
  *)
    echo "[flash] ERROR: unsupported chip '$CHIP_TYPE' (classic ESP32 and S3 supported)" >&2
    exit 1
    ;;
esac
PT=partition-table.bin
[ "$FLASH_SIZE" = "4MB" ] && [ "$CHIP" = "esp32s3" ] && PT=partition-table-4mb.bin
echo "[flash] detected $CHIP_TYPE -> chip=$CHIP bootloader@$BL_OFF bins=$BIN ($FLASH_SIZE)"

# idf.py build trees nest bootloader/partition_table in subdirs; release_bins is flat.
find_bin() {
  local f="$1"
  if [ -f "$BIN/$f" ]; then echo "$BIN/$f"
  elif [ -f "$BIN/bootloader/$f" ]; then echo "$BIN/bootloader/$f"
  elif [ -f "$BIN/partition_table/$f" ]; then echo "$BIN/partition_table/$f"
  else return 1; fi
}
BL_BIN=$(find_bin bootloader.bin) || { echo "[flash] ERROR: missing $BIN/bootloader.bin (build it first — see firmware README)" >&2; exit 1; }
PT_BIN=$(find_bin "$PT")          || { echo "[flash] ERROR: missing $BIN/$PT (build it first — see firmware README)" >&2; exit 1; }
OTA_BIN="$BIN/ota_data_initial.bin"
APP_BIN="$BIN/$APP"
[ -f "$APP_BIN" ] || { echo "[flash] ERROR: missing $APP_BIN (build it first — see firmware README)" >&2; exit 1; }
[ -f "$OTA_BIN" ] || OTA_BIN=$(find_bin ota_data_initial.bin) || { echo "[flash] ERROR: missing ota_data_initial.bin" >&2; exit 1; }

echo "[flash] writing firmware ($FLASH_SIZE layout)..."
"$VENV/python" -m esptool --chip "$CHIP" --port "$PORT" --baud 460800 \
  write_flash --flash_mode dio --flash_size "$FLASH_SIZE" \
  "$BL_OFF" "$BL_BIN" \
  0x8000  "$PT_BIN" \
  0xf000  "$OTA_BIN" \
  0x20000 "$APP_BIN"

echo "[provision] WiFi=$SSID node=$NODE_ID target=$TARGET_IP:5005..."
"$VENV/python" "$FW/provision.py" \
  --port "$PORT" \
  --ssid "$SSID" --password "$PASS" \
  --target-ip "$TARGET_IP" --target-port 5005 \
  --node-id "$NODE_ID"

echo "[done] node flashed. It will join WiFi and stream CSI to $TARGET_IP:5005"
echo "[hint] watch RuView logs: docker logs -f ruview-sensing-server-1 | grep -i source"

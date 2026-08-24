#!/bin/bash
# Flash + provision a RuView ESP32-S3 CSI node (prebuilt release binaries).
# Usage:
#   bash scripts/flash_esp32_node.sh /dev/ttyACM0 <NODE_ID> <WIFI_SSID> <WIFI_PASSWORD>
# Defaults: aggregator target = xwing LAN IP 192.168.1.86:5005, 8MB flash.
set -euo pipefail

PORT="${1:?usage: flash_esp32_node.sh <port> <node_id> <ssid> <password>}"
NODE_ID="${2:?node id required}"
SSID="${3:?wifi ssid required}"
PASS="${4:?wifi password required}"
TARGET_IP="${TARGET_IP:-192.168.1.86}"   # xwing LAN (enx24fbe3784eb4)
FLASH_SIZE="${FLASH_SIZE:-8MB}"

REPO=/home/scott/git/RuView
BIN="$REPO/firmware/esp32-csi-node/release_bins"
VENV=/media/scott/data/finetune-venv/bin

APP=esp32-csi-node.bin
PT=partition-table.bin
[ "$FLASH_SIZE" = "4MB" ] && { APP=esp32-csi-node-4mb.bin; PT=partition-table-4mb.bin; }

echo "[flash] chip probe..."
"$VENV/python" -m esptool --chip esp32s3 --port "$PORT" --baud 115200 chip_id

echo "[flash] writing firmware ($FLASH_SIZE layout)..."
"$VENV/python" -m esptool --chip esp32s3 --port "$PORT" --baud 460800 \
  write_flash --flash_mode dio --flash_size "$FLASH_SIZE" \
  0x0     "$BIN/bootloader.bin" \
  0x8000  "$BIN/$PT" \
  0xf000  "$BIN/ota_data_initial.bin" \
  0x20000 "$BIN/$APP"

echo "[provision] WiFi=$SSID node=$NODE_ID target=$TARGET_IP:5005..."
"$VENV/python" "$REPO/firmware/esp32-csi-node/provision.py" \
  --port "$PORT" --chip esp32s3 \
  --ssid "$SSID" --password "$PASS" \
  --target-ip "$TARGET_IP" --target-port 5005 \
  --node-id "$NODE_ID"

echo "[done] node flashed. It will join WiFi and stream CSI to $TARGET_IP:5005"
echo "[hint] watch RuView logs: docker logs -f ruview-sensing-server-1 | grep -i source"

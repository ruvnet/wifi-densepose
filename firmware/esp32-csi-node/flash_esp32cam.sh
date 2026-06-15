#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/ttyUSB0}"
BAUD="${BAUD:-460800}"
IDF_PATH="${IDF_PATH:-/run/media/deck/SDCARD/offload/toolchains/esp/esp-idf}"
BUILD_DIR="build-esp32cam"
SDKCONFIG_PATH="${BUILD_DIR}/sdkconfig"

if [[ ! -f "${IDF_PATH}/export.sh" ]]; then
  echo "ESP-IDF export.sh not found at ${IDF_PATH}" >&2
  echo "Set IDF_PATH to your ESP-IDF checkout and retry." >&2
  exit 1
fi

source "${IDF_PATH}/export.sh" >/dev/null

idf.py -B "${BUILD_DIR}" \
  -D "SDKCONFIG=${SDKCONFIG_PATH}" \
  -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults.esp32cam" \
  set-target esp32 build

python -m esptool --chip esp32 --port "${PORT}" --baud "${BAUD}" \
  write_flash --flash_mode dio --flash_size 4MB \
  0x1000  "${BUILD_DIR}/bootloader/bootloader.bin" \
  0x8000  "${BUILD_DIR}/partition_table/partition-table.bin" \
  0xf000  "${BUILD_DIR}/ota_data_initial.bin" \
  0x20000 "${BUILD_DIR}/esp32-csi-node.bin"

echo "ESP32-CAM RuView firmware flashed on ${PORT}."
echo "Provision it next, for example:"
echo "  python provision.py --port ${PORT} --chip esp32 --node-id 2 --ssid <ssid> --password <password> --target-ip <deck-ip> --target-port 5005"

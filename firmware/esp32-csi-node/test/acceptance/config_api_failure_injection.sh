#!/usr/bin/env bash
#
# Failure injection for the node configuration API.
#
#   ./config_api_failure_injection.sh [serial-device] [psk-file]
#
#   serial-device  default /dev/ttyUSB0              (or $RUVIEW_SERIAL)
#   psk-file       default $HOME/.ruview-ota-psk      (or $RUVIEW_PSK_FILE)
#
# The ESP32-S3 must be attached to THIS host over USB: the script flashes the
# NVS partition and reads the boot log from the serial port. The WiFi
# password and the OTA PSK are read from files and never echoed.
#
# The NVS partition is filled so that the cfg_trial namespace has no room for
# the recovery bank, while the WiFi namespaces keep the room the node needs to
# boot. The right amount of filler is found by trying a few sizes.
#
# Requires: curl, jq, python3, esptool, and the esp_idf_nvs_partition_gen
# module for the interpreter in $PYTHON (default: python3).
#
# Set RUVIEW_ASSERT_ONLY=<node-url> to skip the flash and the boot and only
# run the assertion against a node that is already in the failure state (its
# cfg_trial namespace is full). That is how the scenario is checked from a
# second machine, or re-checked without reflashing.

set -uo pipefail

SERIAL="${1:-${RUVIEW_SERIAL:-/dev/ttyUSB0}}"
PSK_FILE="${2:-${RUVIEW_PSK_FILE:-$HOME/.ruview-ota-psk}}"
PYTHON="${PYTHON:-python3}"
SIZES="${SIZES:-40 60 80 100 120 150}"
ESPTOOL="${ESPTOOL:-esptool}"
ESPTOOL_BAUD="${ESPTOOL_BAUD:-460800}"
HERE="$(cd "$(dirname "$0")" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for tool in curl jq; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing dependency: $tool" >&2; exit 2; }
done
if [ -z "${RUVIEW_ASSERT_ONLY:-}" ]; then
    # the flash and the boot need the device, the generator and esptool;
    # in assert-only mode none of that is used.
    command -v "$ESPTOOL" >/dev/null 2>&1 || { echo "missing dependency: $ESPTOOL" >&2; exit 2; }
    "$PYTHON" -c 'import esp_idf_nvs_partition_gen' 2>/dev/null || {
        echo "missing dependency: the $PYTHON interpreter cannot import esp_idf_nvs_partition_gen" >&2
        exit 2; }
    [ -e "$SERIAL" ] || { echo "no such serial device: $SERIAL" >&2; exit 2; }
fi
[ -r "$PSK_FILE" ] || { echo "cannot read psk file: $PSK_FILE" >&2; exit 2; }

PSK="$(tr -d '\r\n' < "$PSK_FILE")"
[ -n "$PSK" ] || { echo "psk file is empty: $PSK_FILE" >&2; exit 2; }

# esptool v5 renamed the subcommand to write-flash and the flash options to
# --flash-mode/--flash-size/--flash-freq; v4 used write_flash and the
# underscore spelling. v5 still accepts the old names as deprecated
# aliases, but use whatever the installed version prefers.
if "$ESPTOOL" write-flash --help >/dev/null 2>&1; then
    WRITE_CMD="write-flash"
    FLASH_MODE_OPT="--flash-mode"
    FLASH_SIZE_OPT="--flash-size"
    FLASH_FREQ_OPT="--flash-freq"
else
    WRITE_CMD="write_flash"
    FLASH_MODE_OPT="--flash_mode"
    FLASH_SIZE_OPT="--flash_size"
    FLASH_FREQ_OPT="--flash_freq"
fi

NVS_BIN="$WORK/nvs-trial-full.bin"

get_config() { # canonical (key-sorted) JSON of the stored config
    curl -sS --max-time 20 -H "Authorization: Bearer $PSK" "$1/config" \
        | jq -e -S -c '.config'
}

post_config() { # post_config <url> <json> -> "<http code>\t<body>"
    local tmp code body
    tmp="$(mktemp)" || return 1
    code="$(curl -sS --max-time 25 -o "$tmp" -w '%{http_code}' \
        -X POST -H "Authorization: Bearer $PSK" \
        -H 'Content-Type: application/json' \
        --data "$2" "$1/config" 2>/dev/null)" || true
    [ -n "$code" ] || code="000"
    body="$(tr -d '\r\n' < "$tmp")"
    rm -f "$tmp"
    printf '%s\t%s' "$code" "$body"
}

flash_nvs() { # flash_nvs <filler count>
    "$PYTHON" "$HERE/make_trial_full_nvs.py" "$1" \
        --out-bin "$NVS_BIN" --out-csv "$WORK/nvs-trial-full.csv" \
        --python "$PYTHON" >/dev/null || return 1
    "$ESPTOOL" --chip esp32s3 --port "$SERIAL" --baud "$ESPTOOL_BAUD" \
        "$WRITE_CMD" "$FLASH_MODE_OPT" dio "$FLASH_SIZE_OPT" 16MB \
        "$FLASH_FREQ_OPT" 80m \
        0x9000 "$NVS_BIN" >/dev/null 2>&1
}

boot_and_find_node() { # prints the node url on stdout, empty on failure
    local out ip err
    sleep 2
    err="$(mktemp)"
    out="$("$PYTHON" "$HERE/serial_boot_ip.py" --port "$SERIAL" --timeout 30 \
        --log "$WORK/boot.log" 2>"$err")" || true
    case "$out" in
        IP=*) ip="${out#IP=}"; echo "http://$ip:8032"; rm -f "$err"; return 0 ;;
        DIED) echo "DIED" >&2; rm -f "$err"; return 1 ;;
        *) echo "TIMEOUT" >&2
           [ -s "$err" ] && cat "$err" >&2
           rm -f "$err"; return 1 ;;
    esac
}

for n in $SIZES; do
    echo "=== filler = $n entries ==="
    if [ -n "${RUVIEW_ASSERT_ONLY:-}" ]; then
        # assert-only mode: the node is already in the failure state (its
        # cfg_trial namespace is full), so skip the flash and the boot.
        node="$RUVIEW_ASSERT_ONLY"
        echo "  assert-only against $node"
    else
        if ! flash_nvs "$n"; then
            echo "  flash failed"
            continue
        fi
        node="$(boot_and_find_node)" || { echo "  the node did not come up -> try less"; continue; }
        if [ "$node" = "DIED" ]; then
            echo "  the node DIED on boot (no NVS room for the WiFi) -> try less"
            continue
        fi
        echo "  node at $node"
        sleep 3
    fi

    before="$(get_config "$node")"
    if [ -z "$before" ]; then
        echo "  the node did not answer GET /config"
        continue
    fi
    echo "  before : $(jq -r '"ssid=\(.wifi_ssid) vital_window=\(.vital_window) top_k_count=\(.top_k_count) csi_channel=\(.csi_channel)"' <<<"$before")"

    out="$(post_config "$node" '{"wifi_ssid":"ruview-acceptance-deadnet"}')"
    IFS=$'\t' read -r code body <<<"$out"
    echo "  POST  -> HTTP $code  $body"

    after="$(get_config "$node")"
    echo "  after : $(jq -r '"ssid=\(.wifi_ssid) vital_window=\(.vital_window) top_k_count=\(.top_k_count) csi_channel=\(.csi_channel)"' <<<"$after")"

    if [ "${code:-0}" -ge 500 ] && [ "$before" = "$after" ]; then
        echo ""
        echo "  [PASS] the failure to write the recovery bank answered $code and no live value changed"
        echo "  (the node keeps answering at $node)"
        exit 0
    fi
    if [ "$code" = "200" ]; then
        echo "  the bank wrote without error -> the namespace still had room -> try more"
    else
        echo "  HTTP $code with changed values -> not the right scenario yet"
    fi
done
echo "could not reach the scenario"
exit 1

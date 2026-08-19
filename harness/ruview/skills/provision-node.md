---
name: provision-node
description: Build, flash, and provision an ESP32-S3/C6 CSI node for RuView — firmware variant choice, ESP-IDF Windows-subprocess flow, NVS/WiFi/channel/MAC-filter overrides.
---

# provision-node

Bring an ESP32 sensing node online.

## 1. Pick a firmware variant

- **s3-8mb** (display build) — ESP32-S3 N16R8 / 16MB; AMOLED optional. The display-detect
  fix (#1000) means a *bare* board still captures CSI (MGMT+DATA).
- **s3-4mb** (no-display) — ESP32-S3 4MB; dual-OTA, display disabled.
- **c6** — ESP32-C6 + Seeed MR60BHA2 (60 GHz mmWave + WiFi CSI). The mmwave probe
  requires a validated MR60 header (#1107) so an empty UART never false-detects.

Prebuilt binaries: GitHub release `v0.8.1-esp32` (hardware-validated on S3 QFN56 rev v0.2).

## 2. Flash

ESP-IDF v5.4 on Windows is **subprocess-only** (Git Bash/MSYS is unsupported — strip
`MSYSTEM*` env vars). Offsets for the S3 image:

```
esptool --chip esp32s3 -p <PORT> -b 460800 write_flash \
  0x0 bootloader.bin  0x8000 partition-table.bin \
  0xf000 ota_data_initial.bin  0x20000 esp32-csi-node-s3-8mb.bin
```

(`ruview_node_flash` returns the exact pinned command rather than running an
unattended flash.)

## 3. Provision

```
python firmware/esp32-csi-node/provision.py --port <PORT> \
  --ssid "<SSID>" --password "<secret>" --target-ip <server-ip> --target-port 5005
# optional ADR-060 overrides:
python firmware/esp32-csi-node/provision.py --port <PORT> --channel 6 --filter-mac AA:BB:CC:DD:EE:FF
```

Never echo or commit the WiFi password.

## 4. Confirm CSI is flowing

`ruview_node_monitor {port}` — PASS criteria: serial shows `CSI cb #...` callbacks and
(on a bare board) `CSI filter upgraded to MGMT+DATA`. No callbacks → the node isn't
capturing; do not proceed to calibration.

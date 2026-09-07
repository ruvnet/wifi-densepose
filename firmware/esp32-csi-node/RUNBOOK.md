# Firmware runbook — ESP32-C6 CSI node

Operational procedure. `README.md` explains the project; this says exactly what
to run, in what order, and what to check. Follow it top to bottom.

If you are an agent: **read this file before touching firmware.** Everything
below was learned by getting it wrong at least once.

---

## 1. Build

**One command. Copy it.** From the **repository root**, not this directory:

```bash
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(pwd)/firmware/esp32-csi-node:/project" -w /project \
  espressif/idf:v5.4 bash -c \
  "cat sdkconfig.defaults sdkconfig.defaults.16mb sdkconfig.defaults.esp32c6 \
     > sdkconfig.defaults.build && \
   SDKCONFIG_DEFAULTS='sdkconfig.defaults.build' idf.py set-target esp32c6 && \
   idf.py build"
```

Takes ~3 minutes cold, well under a minute incremental.

### The three defaults files are NOT all automatic

IDF picks up `sdkconfig.defaults` and `sdkconfig.defaults.<target>` on its own.
It does **not** pick up `sdkconfig.defaults.16mb`, and that file is where
`CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE=y` lives.

Build without it and you get an image that is correct in every visible respect
and **cannot roll back a bad OTA** — while `ota_rollback_boot_check()` sits in
the app expecting the capability. Silent, and only discoverable by pushing a
deliberately bad image.

That is why the command above concatenates all three explicitly rather than
relying on discovery.

### Never `rm -rf sdkconfig` without a backup

The README's headline example opens with `rm -rf build sdkconfig`. `sdkconfig`
is **untracked and gitignored**. `partitions_16mb.csv` was lost to exactly this,
and `CLAUDE.md` warns about it *even when a README documents it*.

```bash
cp sdkconfig "sdkconfig.backup-$(date +%Y%m%d-%H%M%S)"   # then regenerate
```

A stale `sdkconfig` is a real hazard, not a hypothetical: one found on
2026-09-03 said `FLASHSIZE=4MB`, `partitions_4mb.csv`,
`DYNAMIC_TX_BUFFER_NUM=64` — against a fleet running 16MB and 128. Anything
built from it would silently not be the configuration under test.

### Verify before you flash anything

```bash
grep -E "^CONFIG_(IDF_TARGET|ESPTOOLPY_FLASHSIZE|PARTITION_TABLE_CUSTOM_FILENAME|\
ESP_WIFI_DYNAMIC_TX_BUFFER_NUM|BOOTLOADER_APP_ROLLBACK_ENABLE)=" sdkconfig
```

All five must read:

| key | expected |
|---|---|
| `CONFIG_IDF_TARGET` | `"esp32c6"` |
| `CONFIG_ESPTOOLPY_FLASHSIZE` | `"16MB"` |
| `CONFIG_PARTITION_TABLE_CUSTOM_FILENAME` | `"partitions_16mb.csv"` |
| `CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM` | `128` |
| `CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE` | `y` |

**Do not use the native `C:\Espressif\...esp-idf-v5.5.5`.** Wrong version, its
venv does not match the default python, and a container-built `build/` refuses
a native build outright. Docker only.

---

## 2. What OTA can and cannot reach

This is the distinction that decides whether a change needs a cable.

| region | offset | reachable by OTA? |
|---|---|---|
| bootloader | `0x0` | **NO — USB only** |
| partition table | `0x8000` | **NO — USB only** |
| otadata | `0xF000` | rewritten by OTA |
| ota_0 | `0x20000` | yes (4 MB) |
| ota_1 | `0x420000` | yes (4 MB) |
| coredump | `0x820000` | no |
| storage (fat) | `0x830000` | no |
| nvs | `0x9000` | never written by a flash — provisioning survives |

**OTA replaces the app partition only.** Anything in the bootloader —
`CONFIG_BOOTLOADER_*`, notably `APP_ROLLBACK_ENABLE` — or any partition-table
change **requires USB on every board**.

There is **no bootloader version on the wire**. A node's sync packet reports app
health, not bootloader capability, so you cannot determine remotely which boards
have a rollback-capable bootloader. Track it per board, or do a USB pass across
all of them.

Corollary: "every node reports health" proves the **app** is current and proves
nothing about the bootloader. Do not answer a bootloader question with app
evidence.

---

## 3. Deploying

### Over the air — app-only changes

Works end-to-end and is the default for anything that is not a bootloader or
partition change. Roll to **one node**, soak it, then the rest.

### Over USB — bootloader, partition table, or recovery

```bash
python -m esptool --chip esp32c6 -b 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 16MB --flash_freq 80m \
  0x0     build/bootloader/bootloader.bin \
  0x8000  build/partition_table/partition-table.bin \
  0xf000  build/ota_data_initial.bin \
  0x20000 build/esp32-csi-node.bin
```

`0x9000` is not in that list, so SSID, password and `node_id` survive.

### Safe single-board diagnostic flash, with a revert path

Nodes run from whichever slot `otadata` selects. **Parse it before flashing** —
two 32-byte entries at offsets 0 and 0x1000; active slot is
`(max ota_seq - 1) % 2`.

1. Back up `otadata` (8 KB) **and** the live app slot with `esptool read-flash`.
2. Flash the diagnostic build — it lands in `ota_0` and rewrites `otadata`.
3. **Revert = write the 8 KB `otadata` backup back.** Production in `ota_1` was
   never touched, so no rebuild is needed.

Validate any backup image before trusting it: byte 0 must be `0xE9`, and the
`esp_app_desc_t` magic at offset `0x20` must be `0xABCD5432`.

**Never flash the whole fleet from a workstation, and never use the server's OTA
endpoint for a diagnostic.** USB, one board, revert path secured first.

---

## 4. Serial

CH343 USB-serial at 115200.

To read a running board **without resetting it**, open with
`DtrEnable=$false; RtsEnable=$false`. esptool asserts RTS and *will* reset the
board, which loses the state you were trying to observe.

---

## 5. Changing config without a cable

Most parameters do not need a flash at all:

```bash
export RUVIEW_OTA_PSK_FILE=<path to the psk file>       # never inline the key
python config_push.py --node <node-ip> --get
python config_push.py --node <node-ip> --set led_brightness=10
```

Keys marked `assoc: true` in `config_api.c` (`wifi_ssid`, `wifi_password`,
`channel_hop_count`, `dwell_ms`, `csi_channel`) are applied as a **trial**: the
node banks its current values, reboots, and keeps the change only if it
re-associates. Everything else applies without that risk.

`config_push.py` has **no unset**. To restore a default you must set it
explicitly. Current defaults:

| key | default | source |
|---|---|---|
| `vital_interval_ms` | 1000 | `Kconfig.projbuild` |
| `swarm_heartbeat_sec` | 30 | `nvs_config.c` |
| `swarm_ingest_sec` | 5 | `nvs_config.c` |
| `beacon_period_ms` | 0 = derive from fleet size | `nvs_config.c` |

---

## 6. Reading a node remotely

`GET /api/v1/mesh` returns per-node `health`: `min_heap_kib`, `die_c`,
`thermal_state`, `tx_dbm`, `reset_reason`, and from sync proto v3 the TX-path
counters `send_fail` / `rate_skip` / `early_drop`.

`min_heap_kib` is a **minimum-since-boot watermark**, not an instantaneous
reading. It resets on reboot and only ever falls. Sampling it within a minute of
a restart tells you nothing — that mistake produced a wrong "no effect" verdict
on 2026-09-03; the real answer, once it had settled, was −56 KiB.

`/api/v1/nodes` `rssi_dbm` is **not** the AP link — it is whatever link landed
most recently and swings 20+ dB with no physical change. Filter `/api/v1/links`
by the AP's BSSIDs instead, and read `csi_fps_ema` alongside, because link count
alone hides a starved node.

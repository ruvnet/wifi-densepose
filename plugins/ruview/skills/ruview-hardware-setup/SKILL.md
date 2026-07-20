---
name: ruview-hardware-setup
description: ESP32-S3 / ESP32-C6 firmware build, flash, WiFi provisioning, and serial monitoring for RuView CSI sensing nodes. Use when setting up physical hardware, reflashing a node, or debugging a device that isn't streaming CSI.
allowed-tools: Bash Read Write Edit Glob Grep
---

# RuView Hardware Setup

Bring a RuView sensing node online: build firmware → flash → provision WiFi → confirm CSI stream.

## Supported devices

| Device | Flash | Chip | Role |
|--------|-------|------|------|
| ESP32-S3 (8MB) | 8 MB | Xtensa dual-core | WiFi CSI sensing node (default) |
| ESP32-S3 SuperMini | 4 MB | Xtensa dual-core | Compact CSI node — use `sdkconfig.defaults.4mb` |
| ESP32-C6 + Seeed MR60BHA2 | — | RISC-V + 60 GHz FMCW | mmWave HR/BR/presence |

**Not supported:** original ESP32, ESP32-C3 (single-core).

**⚠️ Ask about board form factor before flashing.** If the user's board is a coin-sized clone (ESP32-S3-Zero, SuperMini, or similar — not a full DevKitC/XIAO-style board with a real USB connector and visible regulator), warn them before they walk away from it: this firmware runs the WiFi radio continuously (`WIFI_PS_NONE`) plus a full DSP pipeline (`edge_tier=2`), which is sustained high current draw that full-size dev boards handle fine but tiny clones with minimal copper/budget regulators may not. At least one field report: boards ran hot during a normal session and failed to power on again afterward (regulator damage suspected). Tell them to give the board airflow (don't stack/enclose it) and check it by touch during the first several minutes of any new deployment.

## 1. Build firmware (Windows — Python subprocess, NOT bash directly)

ESP-IDF v5.4 does not support MSYS2/Git Bash. Use the Espressif Python venv as a subprocess with `MSYSTEM*` env vars stripped. The proven command lives in `CLAUDE.local.md` — reproduce it:

```bash
/c/Espressif/tools/python/v5.4/venv/Scripts/python.exe -c "
import subprocess, os
env = os.environ.copy()
for k in ['MSYSTEM','MSYSTEM_CHOST','MSYSTEM_PREFIX','MINGW_PREFIX','CHERE_INVOKING']:
    env.pop(k, None)
env['IDF_PATH'] = r'C:\Users\ruv\esp\v5.4\esp-idf'
env['IDF_PYTHON_ENV_PATH'] = r'C:\Espressif\tools\python\v5.4\venv'
env['IDF_TOOLS_PATH'] = r'C:\Espressif'
env['PATH'] = (
    r'C:\Espressif\tools\xtensa-esp-elf\esp-14.2.0_20241119\xtensa-esp-elf\bin;'
    r'C:\Espressif\tools\cmake\3.30.2\cmake-3.30.2-windows-x86_64\bin;'
    r'C:\Espressif\tools\ninja\1.12.1;'
    r'C:\Espressif\tools\idf-exe\1.0.3;'
    r'C:\Espressif\tools\ccache\4.10.2\ccache-4.10.2-windows-x86_64;'
    r'C:\Espressif\tools\python\v5.4\venv\Scripts;'
    + env['PATH']
)
python = r'C:\Espressif\tools\python\v5.4\venv\Scripts\python.exe'
idf_py = os.path.join(env['IDF_PATH'], 'tools', 'idf.py')
r = subprocess.run([python, idf_py, 'build'],   # flash: [python, idf_py, '-p', 'COM8', 'flash']
    cwd=r'C:\Users\ruv\Projects\wifi-densepose\firmware\esp32-csi-node',
    env=env, capture_output=True, text=True, timeout=300)
print(r.stdout[-3000:]); print(r.stderr[-2000:]); print('RC:', r.returncode)
"
```

- **8MB build:** uses `sdkconfig.defaults.template` (no mock — real WiFi CSI).
- **4MB build:** `cp firmware/esp32-csi-node/sdkconfig.defaults.4mb firmware/esp32-csi-node/sdkconfig.defaults` first, then build.
- Build outputs: `firmware/esp32-csi-node/build/{bootloader/bootloader.bin, partition_table/partition-table.bin, esp32-csi-node.bin, ota_data_initial.bin}`.

## 2. Flash to the device

Same subprocess pattern, swap `[python, idf_py, 'build']` → `[python, idf_py, '-p', 'COM8', 'flash']`. Or with esptool directly:

```bash
python -m esptool --chip esp32s3 --port COM8 --baud 460800 \
  write_flash 0x0 firmware/esp32-csi-node/build/bootloader/bootloader.bin \
  0x8000 firmware/esp32-csi-node/build/partition_table/partition-table.bin \
  0xf000 firmware/esp32-csi-node/build/ota_data_initial.bin \
  0x20000 firmware/esp32-csi-node/build/esp32-csi-node.bin
```

(The default device port in this workspace is **COM8**. Some docs reference COM9 — confirm with the user.)

## 3. Provision WiFi + sink address

Runs directly — no ESP-IDF env needed:

```bash
python firmware/esp32-csi-node/provision.py --port COM8 \
  --ssid "YourWiFi" --password "secret" --target-ip 192.168.1.20 --target-port 5005 --node-id 1

# Optional ADR-060 overrides:
python firmware/esp32-csi-node/provision.py --port COM8 --channel 6 --filter-mac AA:BB:CC:DD:EE:FF
```

`--help` lists the full flag set (TDM mesh slotting, edge tier, detection thresholds, vitals window, hop channels, Cognitum Seed, swarm intervals) — see the `ruview-configure` skill for the table. **Gotcha (issue #391):** flashing replaces the *entire* `csi_cfg` NVS namespace — any key not on the CLI is erased; pass the full set you want. On Windows, `provision.py --help` needs `PYTHONUTF8=1` to print (non-ASCII in the help text).

## 4. Confirm CSI stream

```bash
# Serial monitor (use pyserial — idf.py monitor hangs in a subprocess)
/c/Espressif/tools/python/v5.4/venv/Scripts/python.exe -c "
import serial, time
ser = serial.Serial('COM8', 115200, timeout=1); start = time.time()
while time.time() - start < 15:
    line = ser.readline()
    if line: print(line.decode('utf-8', errors='replace').strip())
ser.close()
"
```

Then start the sink and watch frames arrive:
```bash
cd v2 && cargo run -p wifi-densepose-sensing-server   # listens for ESP32 UDP CSI
```

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MSys/Mingw is no longer supported` | ESP-IDF detected Git Bash | Use the Python-subprocess command above with `MSYSTEM*` stripped |
| `cmd.exe /C` hangs | Interactive prompt from Git Bash | Don't use `cmd.exe /C` — use the Python subprocess |
| `cmake not found` | Wrong path | It's `cmake\3.30.2\cmake-3.30.2-windows-x86_64\bin`, not `cmake\3.30.2\bin` |
| `python_env not found` | Missing env var | Set `IDF_PYTHON_ENV_PATH=C:\Espressif\tools\python\v5.4\venv` |
| No CSI frames at the sink | WiFi not provisioned, wrong channel, or MAC filter too tight | Re-run `provision.py`; try `--channel` matching your AP; drop `--filter-mac` |
| False fall alerts | Old `fall_thresh` default | Issue #263 raised it to 15.0 rad/s² + debounce — reflash latest firmware |

## Firmware release process (for maintainers)

1. Build 8MB from `sdkconfig.defaults.template` (no mock)
2. Build 4MB from `sdkconfig.defaults.4mb` (no mock)
3. Save 6 binaries: `esp32-csi-node.bin`, `bootloader.bin`, `partition-table.bin`, `ota_data_initial.bin`, `esp32-csi-node-4mb.bin`, `partition-table-4mb.bin`
4. `git tag v0.X.Y-esp32 && git push origin v0.X.Y-esp32`
5. `gh release create v0.X.Y-esp32 <binaries> --title "..." --notes-file ...`
6. Verify on real hardware (COM8) before publishing — **always test with real WiFi CSI, not mock mode** (mock missed the Kconfig threshold bug)

## Reference

- `CLAUDE.local.md` — exact ESP-IDF build env, paths, QEMU CI notes
- `firmware/esp32-csi-node/` — C firmware (channel hopping, NVS config, TDM protocol)
- `docs/adr/ADR-028-esp32-capability-audit.md`, `docs/build-guide.md`, `docs/TROUBLESHOOTING.md`

# ESP32-S3 → RuView: End-to-End Quickstart (macOS)

Battle-tested runbook for bringing a fresh **ESP32-S3** from box to live CSI on a Mac
(Apple Silicon). Goes: tooling → flash → provision → boot → verify → live sensing.

> Verified 2026-06-12 on an ESP32-S3 (QFN56 rev v0.2, 8 MB PSRAM, **16 MB flash**,
> USB-Serial/JTAG) with prebuilt firmware **v0.6.7**. macOS, zsh.

---

## 0. Tooling (one-time)

The repo's Python tooling isn't on PATH by default. Use a project venv:

```bash
cd <repo>/RuView
python3 -m venv .venv
.venv/bin/pip install esptool pyserial esp-idf-nvs-partition-gen websockets
.venv/bin/esptool version   # expect v5.x
```

> **Nix users:** `esptool` is in nixpkgs (add to your home-manager packages). The venv
> still works regardless; `provision.py` also needs `esp-idf-nvs-partition-gen` which is
> easiest via pip.

You also need `cargo` (for the sensing-server) — already present if you build the v2 workspace.

---

## 1. Plug in the board

- **Use a data USB cable** — charge-only cables are the #1 trap (port never appears).
- ESP32-S3 boards expose either a **CP210x UART port** (`/dev/cu.SLAB_USBtoUART`) or a
  **native USB-Serial/JTAG port** (`/dev/cu.usbmodem*`). This guide hit the native-USB kind.

Find the port:

```bash
ls /dev/cu.*
```

A new `/dev/cu.usbmodemXXXX` (or `/dev/cu.SLAB_USBtoUART`) entry = board found.

> ⚠️ **The port number changes** after a chip reset (e.g. `usbmodem1234561` → `usbmodem1101`).
> Re-check `ls /dev/cu.*` if a command suddenly can't open the port.

---

## 2. Confirm the chip + flash size

```bash
.venv/bin/esptool flash-id
```

If it prints **"No serial data received" / "Could not connect"**, the chip isn't in
download mode. **Enter it manually:**

> **Hold `BOOT` (a.k.a. `0`) → tap `RESET` (a.k.a. `EN`/`RST`) → release `BOOT`.**
> Then rerun within a few seconds.

Expected output identifies the chip (`ESP32-S3`), MAC, and **Detected flash size**
(4 MB / 8 MB / 16 MB). Note it — the next step depends on it.

---

## 3. Flash the firmware (prebuilt — no build needed for S3)

Prebuilt binaries live in `firmware/esp32-csi-node/release_bins/`. Flash offsets are fixed
(`partitions_display.csv`). Use the **8 MB image** even on a 16 MB chip (extra flash is just
left unused):

```bash
.venv/bin/esptool --chip esp32s3 --port /dev/cu.usbmodemXXXX --baud 460800 \
  write-flash --flash-mode dio --flash-size 8MB \
  0x0     firmware/esp32-csi-node/release_bins/bootloader.bin \
  0x8000  firmware/esp32-csi-node/release_bins/partition-table.bin \
  0xf000  firmware/esp32-csi-node/release_bins/ota_data_initial.bin \
  0x20000 firmware/esp32-csi-node/release_bins/esp32-csi-node.bin
```

For **4 MB** boards: use `esp32-csi-node-4mb.bin` + `partition-table-4mb.bin` and
`--flash-size 4MB`.

> If flashing fails to connect, do the BOOT+RESET dance again, then rerun.

---

## 4. Provision WiFi + aggregator target (NVS — no reflash)

The firmware needs your WiFi credentials and the IP of the machine running the
sensing-server (where it streams CSI over UDP :5005).

**Get your Mac's LAN IP** (this is `--target-ip`):

```bash
ipconfig getifaddr en0   # e.g. 192.168.16.105
```

**Provision:**

```bash
.venv/bin/python firmware/esp32-csi-node/provision.py --port /dev/cu.usbmodemXXXX \
  --ssid "YOUR_SSID" --password 'YOUR_PASSWORD' \
  --target-ip 192.168.16.105
```

### Gotcha that will bite you

**2.4 GHz only.** The ESP32-S3 has no 5 GHz radio. Your SSID must broadcast a 2.4 GHz
band. Most dual-band routers use the *same* SSID on both bands — that's fine, the chip
picks 2.4 GHz automatically. Check with:
```bash
system_profiler SPAirPortDataType | grep -E ":$|Channel" | grep -B1 "2GHz"
```

> `provision.py` writes the password in plaintext to `nvs_config.csv` in the repo dir.
> **Delete it after:** `rm nvs_config.csv`.

---

## 5. Boot the board (the serial-monitor trap)

On a **native USB-Serial/JTAG** board, opening a serial monitor toggles DTR/RTS, which
**resets the chip into download mode** — so `miniterm` shows
`boot:0x22 (DOWNLOAD(USB/UART0))` and the app never runs. Symptoms: silent monitor, or
`waiting for download`.

**Don't debug over serial on these boards.** Just:

```bash
# Unplug USB, plug back in (clean power-on → app boots normally).
# Do NOT open miniterm.
```

The board auto-connects to WiFi and starts streaming. (If you *must* read logs and the
board has a CP210x UART port, use that port instead — it auto-resets cleanly.)

---

## 6. Verify CSI is reaching the Mac

Listen on UDP :5005, then re-plug the board:

```bash
python3 - <<'EOF'
import socket, struct, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", 5005)); s.settimeout(45)
print("listening on UDP 5005 — (re)plug the board now...")
n=0; magics={}
t0=time.time()
try:
    while time.time()-t0 < 45:
        d,addr = s.recvfrom(4096); n+=1
        if len(d)>=4:
            m=hex(struct.unpack("<I",d[:4])[0]); magics[m]=magics.get(m,0)+1
        if n==1: print("FIRST PACKET from", addr, len(d),"bytes")
except socket.timeout: pass
print("total:",n,"magics:",magics)
EOF
```

Success looks like packets from the board's IP with magics:
- `0xc5110001` — CSI frames (ADR-018, ~20 Hz) ← the important one
- `0xc5110002` — vitals packets (1 Hz)
- `0xc5110006`, `0xc511a110`, etc. — mesh/sync/aux

Zero packets → board didn't join WiFi (wrong creds / 5 GHz-only SSID) or firewall. Check
`arp -a | grep -i <board-MAC-prefix>` to see if it even got a DHCP lease.

---

## 7. Run the sensing-server

Build once (release):

```bash
cd <repo>/RuView/v2
cargo build -p wifi-densepose-sensing-server --release
```

Run it — **`--source esp32`, not `auto`**:

```bash
./target/release/sensing-server --http-port 3001 --source esp32
```

> - **`--source auto` quits** with "No real CSI source detected" if its ~2 s probe misses
>   frames (it refuses to silently fall back to synthetic). Force `esp32`.
> - **`--http-port`**: default is 3000; pick another (e.g. 3001) if something else owns it.
>   Find a port hog with `lsof -nP -iTCP:3000 -sTCP:LISTEN`.
> - Ports: HTTP `:<http-port>`, WebSocket **`:8765`** (`/ws/sensing`), UDP **`:5005`** (CSI in).
> - Run **without `--model`** — the HF model ships as JSONL RVF which the loader doesn't yet
>   accept, and a bad `--model` degrades the pipeline to null output. Heuristic mode works fine.

---

## 8. Open the UI / inspect routes

UI: **`http://localhost:3001/ui/`** (8 tabs: Dashboard, Hardware, Live Demo, …, Sensing #7).

Key HTTP routes:

| Route | Gives |
|-------|-------|
| `/health` | `{clients, source, status, tick}` — liveness |
| `/api/v1/sensing/latest` | full snapshot: `features` (variance, motion/breathing band power, dominant freq), per-node `amplitude`, `classification`, `estimated_persons` |
| `/api/v1/vital-signs` | HR/RR estimates + `buffer_status` (warming-up indicator) + `signal_quality` |
| `/api/v1/model/info` | loaded RVF model (will say `no_model` without `--load-rvf`) |
| `ws://localhost:8765/ws/sensing` | live 20 Hz push stream the UI consumes |

### What's real vs cosmetic (don't get fooled)

- ✅ **Real:** `features` (variance, band powers), per-subcarrier `amplitude`, presence/motion
  classification — all derived live from your board's CSI.
- 🟡 **Rough:** vital signs (HR/RR) — live extractor, but low-confidence until buffers fill
  (~15–30 s, sit still ~3 m from board) and there's no trained model.
- ❌ **Static marketing copy:** dashboard hero stats ("24 body regions, 100 Hz, 87 % accuracy,
  $30"), and the pose skeleton (signal heuristic, **not** learned keypoints — repo ships no
  pose weights).

---

## 9. Live "is it really sensing me?" test

Poll the feature stream and move around — `motion_band_power` and `variance` should track you:

```bash
.venv/bin/python - <<'EOF'
import urllib.request, json, time
for i in range(1, 26):
    try:
        d = json.load(urllib.request.urlopen("http://localhost:3001/api/v1/sensing/latest", timeout=1))
        f = d["features"]
        print(f"t={i:>2}s  var={f['variance']:7.2f}  motion={f['motion_band_power']:7.2f}  breath={f['breathing_band_power']:7.2f}")
    except Exception as e:
        print(f"t={i:>2}s  (partial frame: {e})")
    time.sleep(1)
EOF
```

Wave near the board → `motion`/`var` jump (seen 22 → 59). Hold still / leave → they settle.
That swing = the full chain (ESP32 → CSI → DSP → API) working end-to-end.

> Occasional `partial frame: 'features'`/`'estimated_persons'` errors are a harmless polling
> artifact — `/api/v1/sensing/latest` sometimes returns the per-node vitals shape instead of
> the full snapshot. Not a board problem.

---

## Troubleshooting cheat-sheet

| Symptom | Cause | Fix |
|---------|-------|-----|
| No `/dev/cu.*` for the board | charge-only cable / no driver | data cable; CP210x driver if SLAB port |
| esptool "Could not connect" | not in download mode | hold BOOT, tap RESET, release BOOT, retry |
| Port name changed mid-flow | chip reset re-enumerates USB | `ls /dev/cu.*`, use new name |
| `miniterm` silent / `waiting for download` | native-USB DTR/RTS reset trap | don't use serial; unplug/replug to boot |
| 0 UDP packets on :5005 | 5 GHz SSID / wrong creds / firewall | 2.4 GHz SSID, re-provision, allow UDP 5005 |
| server exits "No real CSI source" | `--source auto` probe too short | `--source esp32` |
| port 3000 in use | another app | `--http-port 3001` |

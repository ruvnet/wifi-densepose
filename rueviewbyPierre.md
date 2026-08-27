# RuView — by Pierre

A log of the implementations, fixes, and deployment done on this RuView install:
a **3-node WiFi-CSI sensing mesh** (Raspberry Pi aggregator + ESP32 nodes) with a
live, dynamic, hardware-agnostic web **Observatory**.

> Most of this lands on branch `feat/presence-tuning-observatory-nodes` (PR #4).
> The ST7789 display + calibration deadlock fix were merged earlier (PR #3 / `main`).

---

## Hardware in this deployment

| Role | Device | Notes |
|------|--------|-------|
| Aggregator / "the computer" | Raspberry Pi (this host, `192.168.1.195`) | Runs the sensing-server + web UI; sits logically between the nodes |
| Node 1 | Waveshare **ESP32-S3-LCD-1.47** (8MB PSRAM, 16MB flash) | Has the 1.47" ST7789 screen; 2.4 GHz CSI |
| Node 2 | **ESP32-C6** (C6FH4, 4MB) | WiFi-6 CSI node |
| Node 3 | **ESP32-C6** (C6FH8, 8MB) | WiFi-6 CSI node |

All nodes join WiFi **`skynet` (2.4 GHz)** and stream CSI over UDP to the Pi at
`192.168.1.195:5005`. Nodes are flashed in boot mode over USB; **no eFuses are ever burned.**

---

## Firmware

### ST7789 display support (Waveshare ESP32-S3-LCD-1.47)
The stock firmware only drove the SH8601 AMOLED (a different board). Added a
**Kconfig-selectable ST7789 SPI HAL** (`CONFIG_DISPLAY_PANEL_ST7789`):
- Pins: `SCLK=40 MOSI=45 DC=41 CS=42 RST=39 BL=48`, 172×320, built-in `esp_lcd` ST7789 driver.
- Synchronous draw via an `on_color_trans_done` semaphore (no tearing with the shared LVGL flush).
- Boot R/G/B test pattern; orientation/colour are Kconfig knobs (gap/invert/mirror/swap, BGR).
- Touchless-panel gating in `display_task.c`; full CSI capture kept on (light SPI panel, no #396 contention).
- Build variant: `sdkconfig.defaults.s3-lcd147`. The SH8601 build is unchanged/default.

### Native ESP-IDF build environment
Docker IDF image pulls fail on this Pi (containerd snapshotter "failed precondition"
digest bug). Worked around with a **native ESP-IDF v5.4 install at `~/esp/esp-idf`**
(toolchain in `~/.espressif`). Build:
```bash
source ~/esp/esp-idf/export.sh
cd firmware/esp32-csi-node
idf.py -DSDKCONFIG_DEFAULTS=sdkconfig.defaults.s3-lcd147 set-target esp32s3 && idf.py build   # S3 + display
idf.py -DSDKCONFIG_DEFAULTS=sdkconfig.defaults.esp32c6  set-target esp32c6 && idf.py build    # C6
idf.py -p /dev/ttyACM0 flash
python provision.py --port /dev/ttyACM0 --ssid skynet --password '***' --target-ip 192.168.1.195 --node-id <N>
```

---

## Sensing server (Rust, `wifi-densepose-sensing-server`)

Deployed **natively** as a systemd service (`ruview-sensing`) instead of Docker
(see build-env note). Serves REST + WebSocket + the web UI (`--ui-path /home/pi/RuView/ui`).

### Calibration — three stacked bugs fixed (PR #3 + follow-ups)
Room calibration (`/api/v1/calibration/start|stop`) was completely non-functional:
1. **0-frame deadlock** — `maybe_feed_calibration` only fed frames when status was
   already `Collecting`, but feeding is what *causes* `Collecting`. Now feeds while
   `Uncalibrated`/`Collecting`.
2. **Subcarrier width mismatch** — the field model defaulted to `n_subcarriers=56`,
   but ESP32 CSI is 256 (S3 HT40) / 64 (C6). `calibration_start` now sizes the model
   to the live CSI width; mismatched-width frames are skipped.
3. **Unreachable threshold** — `min_calibration_frames` was 12,000 (~10–20 min);
   lowered to 200 for an interactive ~10–30 s empty-room baseline.

Verified on hardware: `start → frame_count climbs → stop → Fresh`.

### Per-environment presence tuning + gating
- **`RUVIEW_PRESENCE_FLOOR`** env var (default `0.03`) — the smoothed-motion floor below
  which the scene reads "absent". The empty-room noise floor is environment-specific
  (this room sits ~0.15), so it's tunable; deployed at **`0.22`** to kill constant
  false-presence. Motion-classify bands rebase on it.
- **Vitals gated on presence** — no phantom HR/RR on an empty room.
- **Signal field gated on presence** — empty room renders a calm floor instead of
  per-frame-normalised noise (was the Observatory floor "flicker").

---

## Observatory web UI (`/ui/observatory.html`)

### Fully dynamic, hardware-agnostic nodes
A node is just a node — the UI makes **no assumptions about hardware**.
- The scene renders **one marker per `node_id` currently reporting** in the live feed
  (any count N, not a fixed pool). Markers are reconciled every frame, so nodes
  **appear/disappear live** as they join/leave — no reload.
- Labels are just `Node <id>`; optionally **tag a name + X/Y/Z position** per node in
  **Settings → SCENE → Room & Nodes** (persisted by id; untagged nodes auto-layout
  evenly around the room).
- Room dimensions (X/Y metres) are editable there too.

> Node *positions* are render-only today — they don't yet feed the server's fusion
> geometry (`SENSING_NODE_POSITIONS` is startup-only). That's the logical next step.

### Hands-off resilience
- **Service worker** (`ui/sw.js`, v4) rewritten to **network-first** (was cache-first,
  which served stale UI forever). On activate it **deletes all caches** and
  **`clients.navigate()`s open tabs**, so UI updates self-apply in one reload.
- **WebSocket auto-reconnects** with exponential backoff. Previously any drop (server
  restart, blip) permanently fell back to demo with 1 node until a manual reload;
  now it recovers on its own. *Verified: restarted the server, the open tab
  auto-recovered to all 3 nodes with no reload.*
- Crash-proof init: settings/loop are wired before networking, and per-frame updaters
  are guarded, so a data/init error can never freeze the page.

### Note on the "router" + waves
The 3D router prop and WiFi-wave ripples are **decorative animation only** — not real
positions or measured signal. The waves now emanate per real node and are gated to the
connected set.

---

## What the system can and can't do (honest)

- ✅ **Presence / motion / breathing** across the mesh — real and solid.
- ✅ **Count & show however many nodes are reporting**, live, with optional tagging.
- ⚠️ **Coarse motion location** only — sub-metre at best, improves with more nodes.
- ❌ **No furniture / room-geometry imaging** — WiFi CSI senses *change*, not static
  structure; that needs mmWave/UWB/LiDAR.
- ❌ **No real pose/skeleton** — needs a trained model (none loaded; the shipped `.rvf`
  is a stub). The pose view is a placeholder.
- ⚠️ **Heterogeneous CSI widths don't fuse** — the S3 (256) and C6 (64) report different
  CSI widths, so multistatic fusion only combines matching-width nodes (the C6 pair).
  Going all-one-chip is the clean path to fused localization.

---

## Operate it

```bash
# Service
sudo systemctl status|restart|stop ruview-sensing
sudo journalctl -u ruview-sensing -f

# Live state
curl -s http://localhost:3000/api/v1/nodes            # who's reporting
curl -s http://localhost:3000/health                  # source/status

# Tune presence floor (no rebuild — edit the systemd env + restart)
#   Environment=RUVIEW_PRESENCE_FLOOR=0.22   in /etc/systemd/system/ruview-sensing.service

# Empty-room calibration (room empty)
curl -X POST http://localhost:3000/api/v1/calibration/start   # wait ~30s
curl -X POST http://localhost:3000/api/v1/calibration/stop
```

**Dashboard:** `http://192.168.1.195:3000/ui/index.html` ·
**Observatory:** `http://192.168.1.195:3000/ui/observatory.html`

Adding a node: flash + provision with the next `--node-id`, point it at
`192.168.1.195`; it shows up in the Observatory on its own.

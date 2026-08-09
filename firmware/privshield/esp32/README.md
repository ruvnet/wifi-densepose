# VEIL on ESP32 — feasibility and honest scope

**Status: `SYNTHETIC / L0` (build-only).** Everything in this directory is an
ESP-IDF component *skeleton*. Nothing here has been flashed, run, or captured on
silicon. Hardware-touching paths are marked `TODO(hw)`. Per `CLAUDE.md`, no
runtime or on-air claim is valid without a captured hardware log — none exists.

This is a **defensive-security, compliance-only** effort. Nothing here jams,
transmits into a band to deny it, or amplifies energy. The ESP32 either
*observes* the channel or *toggles the control pins of a passive external
surface*.

---

## The direct question: "can we use the ESP32 to scramble signals?"

**Short answer: not the way you probably mean, and yes in three narrow
supporting roles.**

The ESP32 **cannot shape its own transmitted 802.11 beamforming feedback.** The
VEIL shield works by perturbing the *compressed beamforming feedback report* (the
Givens/phi-psi angles a station sends back to an AP) with a keyed orthogonal
rotation. On the ESP32 that report is generated **inside the closed Espressif
Wi-Fi PHY/MAC binary blob** (`esp-phy-lib`, shipped in object form; the Wi-Fi
stack is a proprietary blob bound by a hardware NDA and third-party IP
licensing). There is **no ESP-IDF API to intercept, replace, or rotate the
compressed-BF-report the PHY emits.** `esp_wifi_80211_tx()` lets you inject raw
frames, but it is explicitly limited to *beacon, probe req/resp, (non-QoS) data,
and action* frames with the PHY choosing the actual precoding — it will not let
you hand-craft the VHT/HE sounding-feedback subtype with a chosen precoder. So
the ESP32 is **not** a beamforming-feedback protector.

**Feasibility grade for "ESP32 as a self-protecting VEIL node": F (infeasible).**
The one waveform we need to touch is behind a blob with no hook.

**Feasibility grade for "ESP32 as a VEIL supporting device": B (feasible,
build-only).** Three legitimate roles below, best-first.

---

## What the ESP32 can and cannot do

| Capability | ESP-IDF surface | VEIL-relevant? | Verdict |
|---|---|---|---|
| Read CSI (channel state) | `esp_wifi_set_csi_config` / `esp_wifi_set_csi_rx_cb` / `esp_wifi_set_csi` | Yes — detect *being sensed* | **CAN** (observe only) |
| Promiscuous / sniffer RX | `esp_wifi_set_promiscuous` | Yes — more CSI, frame cadence | **CAN** (observe only) |
| Inject raw mgmt/data frames | `esp_wifi_80211_tx` (beacon, probe, action, non-QoS data only) | Marginal; not for BF feedback | **CAN (limited)** |
| Drive external GPIO/SPI hardware | `gpio_*`, `spi_master_*` | Yes — control an external RIS | **CAN** |
| Shape its own **beamforming feedback** (compressed BF report angles) | *none* — generated in closed PHY blob | This is the actual VEIL waveform | **CANNOT** |
| Choose/replace its own **precoding matrix** | *none* — PHY-internal | Yes, but inaccessible | **CANNOT** |
| Modify the Wi-Fi PHY / `esp-phy-lib` | *none* — object-only, NDA | — | **CANNOT** |

Bottom line: the ESP32 **cannot scramble its own WiFi beamforming feedback**, but
it **can** (a) tell an AP-side shield *when* to act, and (b) drive an **external
passive surface** that scrambles the channel in the *sensing* direction. The
latter is the only honest sense in which an ESP32 "helps scramble" a signal, and
it does so without the ESP32 emitting any RF of its own.

---

## The three legitimate roles

### 1. `veil_sensing_detector/` — sensing-solicitation detector (strongest, clearly compliant)
Uses the CSI callback (+ promiscuous RX) to estimate how often the node is being
sounded/solicited, and raises an engage **trigger** (GPIO / MQTT / ESP-NOW) that
tells the *AP-side* VEIL shield (running the portable `../core/veil_shield.c`) to
turn on. Pure observe-plus-control-signal; the ESP32 shapes nothing on air. This
is the role we would actually build first.

### 2. `veil_ris_controller/` — external RIS driver (the honest "help scramble")
Drives a **reconfigurable intelligent surface** over GPIO/SPI. Following the
PrivISAC pattern, each surface element has two phase states designed offline so
the array response is ~identical in the *communication* direction (throughput
preserved) but differs sharply in the *sensing* direction (an eavesdropper's
channel is perturbed). The ESP32 is just a keyed pin-driver; the surface is
**passive** (re-reflects ambient energy, adds none), which is what keeps this on
the compliant side of the jamming line. The switching **schedule is keyed** via
the portable core's `veil_rng` (SplitMix64), so an authorized sensor holding the
key can reconstruct and tolerate the schedule while an eavesdropper cannot.

### 3. `esp_wifi_80211_tx` action-frame signaling (minor)
Not a separate component. The trigger in role 1 could ride an action frame via
`esp_wifi_80211_tx` instead of GPIO/MQTT/ESP-NOW. Useful only as a transport for
the control signal — it does **not** touch beamforming feedback.

---

## Not recommended: decoy / cover-traffic

One could have the ESP32 emit extra frames (via `esp_wifi_80211_tx`) to inject
motion-like or clutter-like variation into an observer's CSI ("cover traffic").
**We do not implement this and do not recommend it.** It is (a) **legally
sensitive** — deliberately adding channel-occupying transmissions to degrade
another party's reception sits close to the *jamming* line and can violate
radio regulations depending on rate, power, and intent; and (b) **low-value** —
it costs airtime, harms your own network, and a determined observer can often
filter periodic decoys. It is documented here only so the option is explicitly
weighed and rejected in favor of the passive-RIS approach (role 2), which
perturbs the *sensing* direction without occupying spectrum.

---

## Build notes

Both components are standard ESP-IDF components (`idf_component_register`) and
are intended to be dropped into an ESP-IDF project's `components/` (or referenced
via `EXTRA_COMPONENT_DIRS`). `veil_ris_controller` compiles the portable core
(`../core/veil_shield.c`) directly to reuse `veil_rng`. They **build** as
skeletons; they do not run — every RF/GPIO/SPI/network path is a `TODO(hw)` stub.

---

## Sources

- ESP-IDF Wi-Fi API (`esp_wifi_80211_tx` supported frame types; CSI APIs):
  <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/network/esp_wifi.html>
- ESP-IDF Wi-Fi CSI (Vendor Features — `esp_wifi_set_csi*`, promiscuous CSI):
  <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi-driver/wifi-vendor-features.html>
- ESP32-C6 beamforming-feedback limitations (IDFGH-15163):
  <https://github.com/espressif/esp-idf/issues/15839>
- Closed Wi-Fi PHY blob (`esp-phy-lib`, object-only, NDA):
  <https://github.com/espressif/esp-phy-lib>
- ESP32 Wi-Fi binary-blob reverse-engineering context (why the PHY is not modifiable):
  <https://esp32-open-mac.be/posts/0005-the-road-ahead/>
- Raw 802.11 TX capability/limits reference (`esp32-80211-tx`):
  <https://github.com/Jeija/esp32-80211-tx>
- PrivISAC — RIS-based privacy-preserving ISAC (sensing vs. comm direction):
  <https://arxiv.org/abs/2601.04488>
- Wi-BFI — beamforming-feedback extraction (why unprotected BF reports leak):
  <https://arxiv.org/pdf/2309.04408>

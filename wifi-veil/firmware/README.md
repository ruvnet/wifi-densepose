# WiFi Veil privacy shield — end-to-end hardware implementation

This tree is the **hardware/firmware realization** of the WiFi Veil compliant-waveform
privacy shield (crate `wifi-veil`, ADR-288; hardware program
ADR-290). It takes WiFi Veil from a synthetic reference model toward real silicon
across multiple hardware providers.

> **Evidence discipline (read this first).** Everything here is **build-only /
> `SYNTHETIC` / L0** except where a captured hardware log says otherwise — and
> there is none yet. Per CLAUDE.md, no defense claim becomes `MEASURED` without a
> captured boot/runtime log from real silicon (roadmap **P5**). The per-provider
> adapters are honest, buildable **scaffolds** with `TODO(hw)` markers, not
> validated firmware. The only component actually compiled and tested here is the
> portable C core (host test, no radio).
>
> **Compliant waveform controls only — never jamming.** Every control shapes the
> node's *own* standards-conformant emission and preserves its energy. Nothing
> here transmits to interfere with another station.

## Architecture

```
        ┌────────────────────────────────────────────────────────┐
        │  core/  — portable C shield  (validated, host-tested)   │
        │  keyed Givens rotation over the fine subspace;          │
        │  SplitMix64 key schedule byte-consistent with the Rust  │
        │  crate; orthogonal ⇒ energy-preserving (not jamming)    │
        └───────────────┬───────────────────────────┬────────────┘
                        │ links against              │
        ┌───────────────▼───────┐   ┌────────────────▼───────────┐
        │ protector adapters    │   │ supporting roles           │
        │ (shape TX feedback)   │   │                            │
        │  • openwifi/ (SDR)    │   │  • esp32/ sensing detector │
        │  • openwrt/  (mac80211)│  │    → trigger the shield     │
        │  • nexmon/   (Broadcom)│  │  • esp32/ RIS controller    │
        └───────────────────────┘   │    → external scramble     │
                                     └────────────────────────────┘
```

- **`core/`** — the shared, hardware-agnostic keyed-rotation implementation.
  Pure C99, no malloc, no libc I/O, only `<math.h>`. **Validated here**:
  `cd core && make test` (energy conservation, reversibility, wrong-key-fails,
  and a PRNG stream that matches the Rust crate exactly). This is what makes the
  on-air behavior identical across every provider and consistent with the
  reference crate.
- **Protector adapters** apply the core's rotation to the transmitted
  beamforming feedback / spatial mapping. Feasibility differs sharply by
  platform (see the matrix) — full control needs an open PHY (openwifi);
  commodity paths are partial and firmware-deep.
- **Supporting roles** are where cheap commodity hardware (ESP32) genuinely
  helps *without* being able to shape its own feedback: detecting sensing to
  trigger the shield, or driving an external reconfigurable surface (RIS).

## Layout

| Path | Provider | Role |
|---|---|---|
| `core/` | portable C | keyed-rotation shield core (validated host test) |
| `openwifi/` | Xilinx Zynq + AD9361 (open PHY/MAC) | full protector + the P5 measurement path |
| `openwrt/` | Linux `mac80211` (mt76 / ath9k…) | commodity protector (partial; sounding/MU control feasible) |
| `nexmon/` | Broadcom/Cypress (RPi) | C-firmware-patch protector (research-grade, partial) |
| `esp32/` | Espressif ESP-IDF | sensing detector + RIS controller (NOT a feedback protector) |

## Feasibility matrix

Grades reflect *capability to actually shape the beamforming-feedback surface*
(the waveform WiFi Veil must touch), **not** effort. Each grade is taken from that
provider's own README, produced by a hardware research agent; the effort/blocker
reality is in the "Why" column. All rows are `SYNTHETIC / L0` — build-only, no
silicon, no captured log.

| Provider | Grade | Can it shape the BF-feedback surface? | Why |
|---|:---:|---|---|
| **openwifi** (Zynq + AD9361, open PHY/MAC) | **B** | **Yes — the only full path.** Capability ceiling **A**; graded B for effort **D**. | Only platform exposing the whole PHY/MAC on FPGA, so a keyed rotation *and its inverse* are physically reachable. But it ships SISO 802.11a/g/n with **no native explicit beamforming** (no NDP sounding, no SVD `V`, no compressed report), so WiFi Veil is realized as the client-transparent per-packet keyed unitary on the TX spatial-mapping stage — which requires **new HDL + a 2nd TX chain + a Vivado rebuild**. Carries the P5 measurement protocol. |
| **openwrt** (Linux `mac80211`; mt76 / ath9k / ath1x) | **C** | **Partial — coarse compliant knobs only.** | The per-packet keyed unitary on the compressed-BF angles / LTF precoder is generated **inside the WiFi MCU firmware blob** on every mainstream AP part (Qualcomm ath10k/11k/12k, MediaTek mt76/mt7915) — userspace never touches the pre-TX `V`. Reachable from userspace: TX antenna-map perturbation, hostapd sounding-cadence jitter, beamformer-capability toggles. **ath9k** (802.11n, register-open) is the one credible driver-patch route toward B. |
| **nexmon** (Broadcom/Cypress C-firmware patch; e.g. BCM43455c0) | **C** | **Read = A (solved); write = C/C-.** | *Reading* the compressed-BF angles is already solved (nexmon_csi + Wi-BFI, no firmware change). *Shaping the transmitted* report is graded C: the report is emitted by the proprietary **D11 real-time core** ~10 µs after the NDP, from hardware-updated internal memory — *below* the ARM firmware where Nexmon's C hooks live. Plausible, deep, firmware-version-specific, unproven here. |
| **esp32** (Espressif ESP-IDF) | **F** / **B** | **F** as a self-protecting node; **B** as a supporting device. | The BF-report is emitted by the **closed `esp-phy-lib` blob** with no ESP-IDF hook to intercept or rotate it (`esp_wifi_80211_tx` won't hand-craft sounding feedback) — so **F (infeasible)** for shaping its own feedback. It earns **B (build-only)** in three legitimate, compliance-only supporting roles: **sensing detector** (CSI-rate trigger for the AP-side shield) and **RIS controller** (drive an external passive reconfigurable surface — the honest way ESP32 "helps scramble", via an external surface, never its own PHY). |

**Reading the grades.** Only **openwifi** can host the full keyed-reversible WiFi Veil
design end-to-end (and only after real HDL work). **openwrt** and **nexmon** are
partial: the exact angles are blob-/ucode-locked on commodity silicon, leaving
either coarse compliant perturbations (openwrt) or a deep, unproven ucode-adjacent
hook (nexmon). **esp32 cannot shield its own feedback at all** — it contributes as
a detector or an external-RIS driver. The direct answer to *"can OpenWRT/open WiFi
software implement this, and can ESP32 scramble signals?"* is: **partially via
OpenWRT (full only on an open PHY like openwifi), and ESP32 only indirectly via an
external surface — never by shaping its own transmission.**

## Two firmware variants

- **Keyed-reversible** (WiFi Veil's ~98%-throughput design): the protector rotates and
  the associated receiver undoes it with the shared key — needs changes on
  **both** ends + key agreement. Best result; needs an open PHY (openwifi) for a
  true demo, or the client-transparent AP-side variant below.
- **Client-transparent per-packet unitary** (LeakyBeam family): only the AP
  changes; clients are unmodified. Rides the 802.11 spatial-mapping mechanism the
  standard marks "not restricted".

## Roadmap position

This tree is roadmap **P4** (firmware feedback shaping — build). **P5** is the
two-node hardware measurement that produces the first `MEASURED` numbers with a
captured log; the openwifi `MEASUREMENT.md` defines that protocol. See
`docs/research/privacy-shield/07-implementation-and-roadmap.md`.

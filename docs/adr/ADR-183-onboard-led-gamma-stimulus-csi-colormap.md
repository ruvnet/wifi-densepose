# ADR-183: Onboard LED as a 40 Hz Gamma Stimulus, Colour-Mapped from Live CSI via `ruv-neural-viz`

| Field | Value |
|-------|-------|
| **Status** | Accepted — implemented & hardware-confirmed on ESP32-S3 N16R8 (COM8) |
| **Date** | 2026-06-17 |
| **Deciders** | ruv |
| **Codename** | **GAMMA-VIZ** |
| **Builds on** | `ruv-neural-viz::ColorMap` (now `no_std` — ruvnet/ruv-neural#3 / RuView#1126), the ESP32 edge `motion_energy` metric (`edge_processing.c`), PR #962 (WS2812 on GPIO 48) |

## Context

Two threads converged. (1) `ruv-neural-viz::ColorMap` — the viridis/cool-warm
palette the rUv-Neural stack uses to render brain-topology graphs — was `std`-only,
so it couldn't run on the ESP32. (2) The onboard WS2812 on the S3 CSI node was dead
weight: the firmware only cleared it on boot (and on the wrong pin for N16R8 — GPIO
38 vs the actual 48, see #962).

The ask: make the LED do something real and honest, using the project's own visual
capability — not a decorative blink. The natural fit is a **40 Hz gamma stimulus**
(the GENUS gamma-entrainment frequency from Alzheimer's light-therapy research)
whose **colour is driven by live sensed motion**, so the node's front panel is both
a known bio-stimulus waveform and a truthful readout of what the CSI is detecting.

## Decision

### Part A — make `ColorMap` `no_std`

`colormap.rs` is self-contained (no cross-crate deps), so expose it on `no_std`
targets. The only blockers were two `std`-only `f64` ops:

- `f64::round` / `f64::abs` → replaced with `core`+`alloc`-safe helpers `fround`
  (round via `f64 as i64` truncation — a `core` cast, no `libm`) and `fabs`.
- `Vec`/`String`/`format!` → from `alloc`.

The graph-bound modules (`animation`/`ascii`/`export`/`layout`) and their heavy deps
move behind a default `std` feature; `--no-default-features` builds the crate `no_std`
and exposes only `colormap`. Output is **byte-identical** (8/8 colormap tests pass with
the same RGB values), so this is a pure portability change.

### Part B — the LED stimulus (firmware)

`firmware/esp32-csi-node/main/main.c`, on boot:

- WS2812 on **GPIO 48** (N16R8 / DevKitC-1 v1.1; GPIO 8 on C6).
- An `esp_timer` periodic at **12 500 µs toggles a square wave → 40 Hz, 50 % duty**
  (full-on / full-off — a *perceptible* gamma flicker, not a colour drift).
- **ON-phase colour = live CSI motion.** Each ON phase reads `edge_get_vitals().motion_energy`,
  normalises it (`/ LED_MOTION_FULLSCALE`, clamped `[0,1]`), and indexes a **60-step
  viridis LUT generated from `ColorMap::viridis().map()`** — still = dark purple,
  strong motion = yellow.

The LUT is baked from the real crate (Part A makes the same `ColorMap` embeddable
for a future direct FFI path once the ESP Rust toolchain is in CI). The colours are
therefore provably `ruv-neural-viz`'s, and the motion is provably real.

## Honesty (what it is and is not)

- **40 Hz is a real square-wave stimulus** (12.5 ms on / 12.5 ms off), not a label on
  a colour sweep. It is *not* tied to any measured 40 Hz brain rhythm — it is an
  *output* stimulus at the gamma frequency, not a readout of neural gamma.
- **Colour is a real CSI readout** — `motion_energy` is the on-device phase-variance
  motion metric the node already computes; no fabrication. At rest the LED sits at the
  purple (low) end and flickers there.
- No therapeutic claim is made. 40 Hz GENUS entrainment is cited as the *origin of the
  frequency choice*, not as a validated medical effect of this device.

## Consequences

**Positive**
- The LED is now an honest front-panel: gamma-frequency flicker + a live motion readout.
- `ColorMap` is embeddable (`no_std`), unblocking on-device use of the rUv-Neural
  palette beyond this LED.
- Confirms #962's GPIO-48 fix visually (the LED lights on N16R8).

**Negative / risks**
- Changes *default* firmware behaviour: the onboard LED now animates instead of staying
  off (minor power + a visible flicker some may not want). Gate behind a Kconfig
  (`CONFIG_LED_GAMMA_VIZ`) if a dark default is preferred — follow-up.
- A 40 Hz flicker can be an issue for photosensitive users; document on the enclosure.
- `LED_MOTION_FULLSCALE` (0.25) is hand-tuned, not calibrated per-environment.
- The colour uses a baked LUT, not the live Rust `ColorMap` (FFI path deferred — needs
  the ESP Rust/xtensa toolchain, not yet in CI).

## Validation

- `ruv-neural-viz`: `cargo build` (std) ✓, `cargo test colormap` 8/8 ✓ (identical RGB),
  `cargo build --no-default-features` compiles `no_std` ✓.
- Firmware: built (1.13 MB), flashed to ESP32-S3 N16R8 (COM8). Boot log:
  `Onboard WS2812: 40 Hz gamma flicker (GENUS), colour=CSI motion via ruv-neural-viz, GPIO 48`;
  CSI continues (27–38 pps), `motion=0.00` at rest → purple flicker as designed.
- Full on-device (xtensa) Rust build of `ColorMap` not run — ESP Rust toolchain absent.

## References
- ruvnet/ruv-neural#3 (ColorMap no_std), RuView#1126 (submodule bump), #962 (GPIO 48).
- Singer/Tsai GENUS 40 Hz gamma entrainment (origin of the frequency, not a device claim).

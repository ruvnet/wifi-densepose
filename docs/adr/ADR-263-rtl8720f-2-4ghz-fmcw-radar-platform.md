# ADR-263: Adopt RTL8720F 2.4 GHz FMCW radar as an optional RuView sensing platform

- **Status**: proposed
- **Date**: 2026-07-18
- **Deciders**: ruv
- **Tags**: realtek, rtl8720f, ameba, fmcw, radar, cfr, csi, hardware
- **Relates to**: ADR-018, ADR-063, ADR-064, ADR-095, ADR-097, ADR-260, ADR-262

## Context

Realtek's `RTL8720F-2.4G-Radar-Advantages_EN.pptx` describes an RTL8720F mode that shares the
2.4 GHz radio between Wi-Fi, Bluetooth, and an active FMCW radar. It offers two data products that
are useful to RuView:

1. **CFR (Channel Frequency Report)**, described by Realtek as the same concept as Wi-Fi CSI.
2. **Near and far Range-FFT reports**, preserving near-field content while extending observation to
   approximately 5–6 m.

The proposed radio uses one transmit and one receive antenna, 20/40/70 MHz sweeps, configurable
8/16/32/64 microsecond chirp symbols, a maximum 2.56 ms FMCW packet, and a configurable frame
interval above 15 ms. The deck recommends 40 MHz outside Japan and 20 MHz in Japan. It also
describes EDCCA/CTS channel access, Wi-Fi/BT/radar time division, interference reporting, and
priority arbitration in the driver.

This is not a drop-in replacement for ESP32 CSI:

- it is **active monostatic FMCW**, while the ESP32 path observes Wi-Fi packet CSI;
- one Tx/one Rx has no angle-of-arrival or native multi-target separation;
- the stated 40 MHz range resolution is about 3.15 m, despite a finer 0.59 m Range-FFT report step;
- the presentation is a capability description, not an SDK contract. It contains no header names,
  function signatures, callback ABI, binary layouts, toolchain version, licensing terms, or public
  RTL8720F board package.

Realtek's public Ameba RTOS repository is the base. Release v1.2.1 includes the CSI API and fixes a
CSI application-buffer semaphore issue, but does not expose the radar application surface. Open
upstream PR #1336 (2026-07-18 snapshot) adds RTL8720F project artifacts, `AT+RAD`, `AT+RADDBG`, and
the public configuration call `wifi_radar_config(struct rtw_radar_action_parm *)`. Its public
parameter struct confirms mode, channel, 70/40/20 MHz bandwidth selector, trigger period, and
enable/config actions. Report reception still crosses non-public/placeholder HAL symbols such as
`wifi_hal_radar_recv_data(frame_num, frame_type, data)`, so the report layout and buffer lifetime
remain vendor-gated. Therefore the integration stays split at that boundary.

## Decision

RuView will support RTL8720F radar as an **optional, capability-negotiated source**, without
replacing the ESP32 firmware or treating radar CFR as byte-compatible with ADR-018 CSI.

The integration has three layers:

1. **Realtek device firmware**: a small application built in the vendor-supported Ameba SDK calls
   the radar API, owns coexistence configuration, and emits versioned reports. This code lives under
   `firmware/rtl8720f-radar/` only after the redistributable SDK/API is available.
2. **Transport-neutral wire contract**: CFR and Range-FFT reports are framed independently from the
   vendor ABI and sent over UDP, USB CDC, or UART. ADR-264 defines this boundary.
3. **Rust host adapter**: `wifi-densepose-hardware` parses reports from bytes and converts CFR into
   the existing CSI-domain representation, while Range-FFT remains a radar modality and feeds the
   RuField/RuView cross-modality bridge from ADR-260/262.

The two report types remain semantically distinct:

| RTL8720F output | RuView representation | Permitted use |
|---|---|---|
| CFR | `CsiFrame` through a Realtek calibration adapter | CSI feature extraction after validation |
| Range-FFT near/far | `RadarFrame` / RuField `mmwave_radar`-class event with a 2.4 GHz descriptor | range, motion, presence, fusion |
| Vendor AI presence probability | derived observation with model/version provenance | advisory input, never ground truth |
| Interference report | quality/provenance metadata | reject, down-weight, or mark contaminated frames |

The modality registry should eventually distinguish `fmcw_radar_2_4ghz` from `mmwave_radar`; until
that RuField schema revision is accepted, the adapter must attach `carrier_hz = 2.4e9` and must not
claim millimetre-wave provenance.

## Delivery phases and gates

### P0 — Vendor enablement

Obtain the PR #1336-or-newer RTL8720F SDK package, radar API headers/libraries, a supported evaluation
board, flashing/debug instructions, report definitions, and written redistribution terms.

**Gate:** compile and run Realtek's unmodified radar example and capture CFR plus near/far
Range-FFT output. Until this passes, device firmware is `VENDOR_BLOCKED`, not implemented.

### P1 — Host-first contract

Implement ADR-264 types, parsers, fixtures, fuzz tests, and replay support without linking vendor
code. Use the Rust `Rtl8720fSimulator` as the only pre-hardware live source. It emits deterministic
CFR, near/far Range-FFT, interference, and capabilities frames through the same ADR-264 encoder and
parser used by hardware. Every simulated frame sets `RadarFlags::SYNTHETIC`; simulation results are
never reported as device measurements.

**Gate:** malformed inputs never panic; encode/decode round trips; unknown versions and report
types fail closed.

### P2 — RTL8720F firmware adapter

Wrap only the minimum vendor API surface: initialization, profile configuration, start/stop,
callback acquisition, interference status, and report serialization. Keep vendor types out of the
wire protocol.

**Gate:** 30-minute simultaneous Wi-Fi telemetry and radar capture with no watchdog reset, bounded
loss, monotonic sequence numbers, and explicit coexistence/interference statistics.

### P3 — Calibration and signal validation

Calibrate CFR phase/amplitude, Range-FFT bin spacing, static leakage, and clock drift. Compare
reported range against measured targets at multiple distances and bandwidths.

**Gate:** publish measured error distributions. Do not infer accuracy from report-bin spacing and
do not advertise multi-person pose or vital signs from the vendor deck.

### P4 — Fusion and productization

Feed calibrated CFR through the CSI path and Range-FFT through RuField, retaining source, mode,
bandwidth, calibration, firmware, and interference provenance.

**Gate:** ablation shows whether the radar stream improves a named RuView metric over ESP32 CSI
alone. If it does not, ship it only as an independent presence/range sensor.

## Consequences

### Positive

- One low-cost radio can provide active radar and CSI-like CFR while retaining Wi-Fi connectivity.
- Range-FFT adds an independent physical measurement for presence/range fusion.
- The vendor SDK is isolated from the Rust sensing core and from the stable on-wire contract.
- Capability negotiation permits future Realtek parts without another application-level fork.

### Negative

- The first implementation is blocked on access to the actual RTL8720F radar SDK/API and hardware.
- Active 2.4 GHz transmission changes coexistence, privacy, power, and regional compliance concerns.
- 1T1R and limited sweep bandwidth cannot provide the spatial resolution of multi-antenna mmWave.
- A second embedded toolchain and firmware release process must be maintained.

### Neutral

- ESP32 remains the default CSI node.
- Existing consumers receive normalized frames and do not link against Realtek code.
- Vendor AI output is optional metadata; RuView retains responsibility for its own validation.

## Rejected alternatives

1. **Map Range-FFT directly to `CsiFrame`.** Rejected because range bins and channel-frequency
   samples have different axes and physical meaning.
2. **Link the Realtek SDK into the Rust server.** Rejected because it couples host builds to a
   proprietary embedded ABI and toolchain.
3. **Wait to define any interface until hardware arrives.** Rejected because the host protocol,
   parser safety, replay, and provenance can be developed and reviewed independently.
4. **Replace ESP32 nodes.** Rejected because the modes are complementary and availability differs.

## Open vendor questions

- Exact RTL8720F part/board identifier and production availability.
- SDK repository/tag, compiler, RTOS, binary blobs, license, and redistribution permissions.
- Radar initialization/configuration/callback API signatures and threading/ISR constraints.
- CFR and near/far Range-FFT element type, complex ordering, scaling, endianness, and timestamps.
- Whether CFR is calibrated complex data and whether phase remains coherent across frames.
- Maximum report rates, buffer ownership, DMA/cache constraints, and Wi-Fi throughput impact.
- Region/channel enforcement and whether 70 MHz operation is allowed by the supplied firmware.
- Secure boot, signed OTA, unique device identity, and firmware attestation support.

## Sources

- Realtek Semiconductor, `RTL8720F-2.4G-Radar-Advantages_EN.pptx`, slides 3 and 10–19,
  supplied 2026-07-18. This is product material, not measured RuView validation.
- [Ameba-AIoT/ameba-rtos releases](https://github.com/Ameba-AIoT/ameba-rtos/releases), reviewed
  2026-07-18; v1.2.1 is the current QC release and includes a CSI buffer-semaphore fix.
- [Ameba-AIoT/ameba-rtos PR #1336](https://github.com/Ameba-AIoT/ameba-rtos/pull/1336), reviewed
  2026-07-18; exposes RTL8720F build assets, `wifi_radar_config`, and radar AT commands while report
  internals remain in binary/private layers.
- ADR-063 (mmWave sensor fusion), ADR-095/097 (source normalization), and ADR-260/262 (RuField
  multimodal event model and live bridge).

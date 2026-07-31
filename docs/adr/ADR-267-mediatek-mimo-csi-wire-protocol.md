# ADR-267: MediaTek MIMO CSI Wire Protocol

- **Status**: accepted
- **Date**: 2026-07-18
- **Deciders**: RuView maintainers
- **Tags**: mediatek, csi, protocol, rust, udp, replay

## Context

The MediaTek simulator, captured regression fixtures, and a future `mt76` agent
need one safe host-side representation. Copying an undocumented firmware layout
would couple RuView to a private ABI and make malformed kernel/network data risky.
MIMO CSI also requires explicit Tx/Rx/subcarrier dimensions and per-Rx-chain RSSI.

## Decision

Define `MTC1` version 1 as a little-endian, self-delimiting envelope:

- 72-byte fixed header with magic, version, report kind, total length, sequence,
  monotonic timestamp, device ID, chipset profile, frequency, bandwidth, flags,
  Tx/Rx dimensions, numeric format, PPDU type, subcarrier count, noise floor,
  scale, subcarrier spacing, calibration ID, and payload length.
- CSI payload begins with one signed RSSI byte per Rx chain, followed by
  `tx_count * rx_count * subcarrier_count` complex values in Tx-major,
  Rx-major, subcarrier-major order.
- Supported numeric formats are complex signed i16 and complex finite f32.
- Capability reports use bounded opaque TLVs until a public driver contract exists.
- CRC-32/IEEE covers header and payload; the final four bytes carry the checksum.
- One envelope maps to one UDP datagram, capped at the IPv4 UDP payload maximum
  of 65,507 bytes. Replay files prefix each envelope with a little-endian `u32`.
- Parsers reject unknown versions/types/formats, invalid dimensions/bandwidth,
  multiplication overflow, inconsistent payload lengths, non-finite floats,
  bad CRC, trailing datagram bytes, and frames above the cap.
- Flags distinguish calibrated, saturated, time-synchronized, dropped-predecessor,
  and synthetic frames. Synthetic provenance cannot be cleared by downstream code.

## Consequences

### Positive

- Deterministic simulator and future hardware use identical parsing and APIs.
- Explicit dimensions prevent ambiguous antenna or subcarrier interpretation.
- CRC, finite-value checks, and hard caps make network/replay ingestion robust.
- The format supports MT7981, MT7986, and MT7996 profiles without claiming their
  undocumented firmware layouts.

### Negative

- A translation/copy step is required from a future kernel report.
- Maximum-size Wi-Fi 7 matrices may need segmentation in a later protocol version.

### Neutral

- Version 1 models one link per report; MLO correlation is a future extension.
- Capability TLVs are intentionally conservative until hardware metadata is known.

## Links

- [ADR-266: MediaTek Filogic CSI platform](ADR-266-mediatek-filogic-csi-platform.md)
- [ADR-018: ESP32 binary CSI framing](ADR-018-esp32-csi-frame-protocol.md)
- [ADR-264: RTL8720F radar wire protocol](ADR-264-rtl8720f-radar-wire-protocol.md)

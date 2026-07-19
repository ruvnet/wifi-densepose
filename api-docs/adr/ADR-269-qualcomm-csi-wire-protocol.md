# ADR-269: Qualcomm CSI Wire Protocol

- **Status**: accepted
- **Date**: 2026-07-18
- **Tags**: qualcomm, csi, protocol, rust, udp, replay

## Decision

Define `QCS1` version 1 as a vendor-boundary envelope, not a Qualcomm firmware ABI.
It uses a 72-byte little-endian header plus payload and CRC-32/IEEE. The header
records report kind, total length, sequence, monotonic timestamp, device ID,
chipset profile, center frequency, bandwidth, flags, Tx/Rx counts, numeric format,
PPDU type, subcarrier count, noise floor, scale, subcarrier spacing, calibration
ID and payload length.

CSI payloads contain one signed RSSI byte per receive chain followed by
`tx * rx * subcarriers` complex i16 or finite f32 values in Tx-major, Rx-major,
subcarrier-major order. Capability reports carry bounded opaque bytes. One QCS1
frame maps to one UDP datagram; replay files prefix each frame with a little-endian
u32 length.

Parsers fail closed on unknown enums, bad CRC, truncation, trailing datagram data,
non-finite values, inconsistent dimensions, chipset chain/bandwidth violations,
payload mismatches, arithmetic overflow and the IPv4 UDP payload ceiling. A
synthetic flag provides end-to-end simulator provenance.

Version 1 profiles are QCA9300, QCN9074 and QCN9274. QCA9300 is capped at three
chains and 40 MHz; modern profiles are capped at four chains and 160 MHz.

## Consequences

- Simulator, replay and future hardware adapters share one validated Rust API.
- No private firmware layout is represented or redistributed.
- 320 MHz/EHT matrices require segmentation or a later protocol revision.

## Links

- [ADR-268: Qualcomm platform strategy](ADR-268-qualcomm-atheros-csi-platform.md)
- [ADR-267: MediaTek MTC1 protocol](ADR-267-mediatek-mimo-csi-wire-protocol.md)


# ADR-264: Versioned wire protocol for RTL8720F CFR and Range-FFT reports

- **Status**: proposed
- **Date**: 2026-07-18
- **Deciders**: ruv
- **Tags**: realtek, rtl8720f, protocol, cfr, range-fft, udp, serial
- **Depends on**: ADR-263
- **Relates to**: ADR-018, ADR-095, ADR-097, ADR-099, ADR-260

## Context

ADR-263 adopts RTL8720F radar behind an anti-corruption boundary. The Realtek presentation names
CFR, near Range-FFT, far Range-FFT, and interference reports, but does not specify their binary ABI.
RuView needs a stable, testable contract that can be implemented before the vendor SDK arrives and
that will not expose vendor structs, pointer layouts, padding, or callback lifetime rules over the
network.

ADR-018 already defines ESP32 CSI framing. Reusing its magic or pretending that Realtek radar is an
ESP32 packet would make source detection ambiguous and erase radar-specific calibration metadata.

## Decision

Define a new little-endian `RtlRadarFrameV1` envelope with its own magic and explicit payload type.
This is a RuView protocol, not a claim about Realtek's native memory layout.

### Envelope

All integer fields are little-endian. Floating-point payloads use IEEE-754 binary32. No C struct is
sent by `memcpy`; firmware serializes each field explicitly.

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | magic | ASCII `RTR1` (`0x31525452`) |
| 4 | 1 | version | `1` |
| 5 | 1 | report_type | 1 CFR, 2 range-near, 3 range-far, 4 interference, 5 capabilities |
| 6 | 2 | header_len | complete header size, initially 56 |
| 8 | 4 | frame_len | header + payload + CRC |
| 12 | 4 | sequence | wraps modulo 2^32 |
| 16 | 8 | timestamp_us | monotonic device time at acquisition |
| 24 | 8 | device_id | stable pseudonymous identifier, not a MAC address |
| 32 | 4 | center_freq_khz | RF centre frequency |
| 36 | 2 | bandwidth_mhz | 20, 40, or 70 |
| 38 | 2 | flags | calibration/interference/saturation/time-sync flags |
| 40 | 2 | element_count | complex samples or range bins |
| 42 | 1 | element_format | 0 bytes/TLV, 1 complex-i16, 2 complex-f32, 3 power-u16, 4 power-f32 |
| 43 | 1 | antenna_count | expected to be 1 for the deck's 1T1R configuration |
| 44 | 4 | scale | quantized-to-physical multiplier; `1.0` for float payloads |
| 48 | 4 | bin_spacing | Hz for CFR, metres for Range-FFT |
| 52 | 4 | calibration_id | device calibration revision/hash prefix |
| 56 | variable | payload | determined by type, count, and format |
| final-4 | 4 | crc32 | IEEE CRC-32 over header and payload |

If vendor evidence shows that 56 bytes is too costly, a later protocol version may introduce a
compact header. V1 favors auditable provenance over premature byte savings.

### Payload semantics

- **CFR** contains ordered complex channel-frequency samples. The adapter must know the frequency
  origin/order and must not fabricate missing phase. Uncalibrated frames carry the uncalibrated flag
  and cannot enter phase-sensitive processing.
- **Range-near/range-far** contains ordered range bins. Near and far are separate report types so
  filtering and leakage behavior are never hidden from consumers.
- **Interference** contains a versioned TLV set for channel-busy, detected-during-chirp, estimated
  interference power, and packet jitter. Unknown TLVs are skipped by length.
- **Capabilities** is emitted at boot and on request. It declares supported report types, bandwidths,
  chirp lengths, maximum elements/report, maximum frame rate, firmware version, and SDK identifier.

### Transport

The identical envelope is supported over:

- UDP datagrams for normal RuView ingestion;
- USB CDC or UART with COBS framing and a zero-byte delimiter;
- file replay as a length-prefixed sequence of envelopes.

One envelope must fit one UDP datagram. Fragmentation is not part of V1; firmware rejects a profile
whose maximum report exceeds the configured MTU and reports the required size through capabilities.

### Parser and trust rules

The host parser:

1. validates magic, version, lengths, enum values, element count/format multiplication, and CRC
   before allocating or decoding the payload;
2. caps frames at 64 KiB and elements at a configured hardware maximum;
3. rejects non-finite float metadata/payload values;
4. tracks sequence gaps and timestamp regressions per device;
5. preserves unknown flags but never interprets them as trusted;
6. attaches transport source, firmware/SDK version, calibration ID, and interference state to
   provenance;
7. labels fixture/generated frames as synthetic.

No vendor-provided presence probability bypasses RuView privacy, provenance, or quality gates.

## Consequences

### Positive

- Firmware, transport, parser, replay, and fusion can evolve independently.
- Fuzzing and golden fixtures require no Realtek SDK or board.
- CFR and Range-FFT retain correct axes and calibration provenance.
- A boot-time capabilities frame makes SDK/API drift observable.

### Negative

- Serialization adds CPU and bandwidth overhead compared with dumping a vendor buffer.
- V1 fields may need revision after the actual API and report limits are disclosed.
- UDP provides integrity/error detection, not authenticity or confidentiality.

### Neutral

- Authentication can be layered with ADR-032 device identity or a signed RuField receipt without
  changing report semantics.
- ESP32 ADR-018 framing remains unchanged.

## Implementation plan

1. Add `rtl8720f` types/parser module to `wifi-densepose-hardware` behind no vendor dependency.
2. Add golden CFR, near/far Range-FFT, interference, and capabilities fixtures.
3. Add property/fuzz tests for length arithmetic, enum handling, CRC, and float validation.
4. Add a replay CLI that prints normalized metadata without running inference.
5. Once SDK access exists, implement the embedded serializer and verify captured frames against the
   host golden decoder.
6. Revise this proposed ADR with measured element counts, rates, and API names before acceptance.

Host-side steps 1–3 are implemented in `wifi-densepose-hardware::rtl8720f`: typed report and
element enums, semantic type/format validation, bounded length arithmetic, CRC verification,
finite-float checks, encode/decode round trips, corruption/truncation tests, and deterministic
arbitrary-input panic checks. Cross-language vectors remain blocked on the vendor SDK callback ABI.
Bit 15 of `flags` is reserved by RuView as `SYNTHETIC`; the Rust simulator always sets it and real
firmware must never set it. The simulator is deterministic by seed and exercises the production
encoder/parser rather than a parallel mock representation.

## Acceptance criteria

- Rust encode/decode round-trip for every report type.
- Cross-language golden vector produced by the RTL8720F firmware.
- Zero parser panics over the fuzz corpus and arbitrary byte input.
- Detection of single-bit corruption, truncation, count overflow, timestamp regression, and gaps.
- Captured CFR frequency order and Range-FFT bin spacing verified against vendor documentation and a
  measured target.

## Sources

- Realtek Semiconductor, `RTL8720F-2.4G-Radar-Advantages_EN.pptx`, slides 11–19, supplied
  2026-07-18.
- ADR-018 (ESP32 framing), ADR-095/097 (hardware normalization), ADR-260 (multimodal event model),
  and ADR-263 (platform decision).

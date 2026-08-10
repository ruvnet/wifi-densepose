# ADR-289: Wideband 802.11ax CSI ingest — FeitCSI/AX210 adapter and subcarrier-agnostic plumbing

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-10
- **Deciders**: ruv
- **Tags**: hardware, csi, 80211ax, ax210, feitcsi, ingest, mat

## Context

RuView's CSI ingest (`wifi-densepose-mat/src/integration/hardware_adapter.rs`)
supports ESP32 serial streams, the legacy Intel 5300 tool, and Atheros/Nexmon
paths. All of these are 802.11n-class: ≤40 MHz bandwidth, ≤114 subcarriers,
2.4/5 GHz.

The 2026 research sweep found the field's center of gravity has moved to
Intel AX200/AX210 NICs via PicoScenes (closed-source core) and FeitCSI
(open-source, GPL): 802.11ax CSI at up to 160 MHz / 1992 subcarriers,
including the 6 GHz band. This is both the research-grade tier today and the
shape of the data 802.11bf silicon will deliver from ~2026 onward. RuView's
`wifi-densepose-hardware` crate already models 802.11bf session types, but no
ingest path can carry wideband CSI into the pipeline.

Without a wideband path, RuView cannot develop against the best available
signal, cannot compare ESP32-grade results to wideband upper bounds, and will
meet 802.11bf silicon with no tested plumbing for >114-subcarrier frames.

## Options considered

1. **PicoScenes `.csi` ingest.** Rejected for now: the format is produced by a
   closed-source core and is versioned/complex; parsing it without a
   maintained spec invites silent corruption.
2. **Raw pcap + radiotap parsing.** Rejected: duplicates what FeitCSI already
   does on-device, and pulls a packet-capture dependency into the pipeline.
3. **FeitCSI file/stream ingest.** Chosen: FeitCSI is open-source (its header
   layout is auditable against the source), targets AX200/AX210, covers
   20–160 MHz including 6 GHz, and emits a compact binary record per frame.

## Decision

Extend `v2/crates/wifi-densepose-mat/src/integration` with:

### 1. `feitcsi` record parser

- A validated parser for FeitCSI's binary CSI record layout (header with
  CSI buffer length, rate/bandwidth/channel metadata, antenna counts, RSSI,
  timestamp, followed by interleaved complex CSI). The parser is written
  against the documented layout, is version-checked, and rejects
  records whose declared dimensions disagree with the buffer length —
  untrusted file/stream input is validated at the boundary.
- Bounded allocation: a hard cap on subcarrier count (4096) and antenna
  count (8) so a corrupt length field cannot cause unbounded allocation.

### 2. `DeviceType::FeitCsi` in the hardware adapter

- File-replay mode (read a recorded FeitCSI capture deterministically) and a
  streaming mode fed by an external process writing to a path/pipe. No
  privileged operations inside the crate: RuView does not configure the NIC;
  FeitCSI's own tooling owns that, per least-authority.

### 3. Subcarrier-agnostic plumbing

- Ingest carries native subcarrier dimensionality end-to-end and converts to
  pipeline width explicitly via the existing interpolation/decimation stage,
  recording the native → pipeline mapping in frame metadata so downstream
  consumers know the true spectral resolution. Bandwidth (20–160 MHz) and
  band (2.4/5/6 GHz) become first-class frame metadata.

## Consequences

- RuView gains a research-grade wideband development path and a tested
  ingest shape for future 802.11bf reporting (truncated CIR is a natural
  extension of the same plumbing).
- GPL FeitCSI is used as an external tool, never linked: only its output
  format is parsed. No licensing contamination of the MIT workspace.
- The parser tracks an external project's format; version checks fail loudly
  on mismatch rather than misparse.
- ESP32 remains the deployed sensor tier; wideband is a development/
  validation tier. Accuracy claims from wideband captures must be tagged with
  the capture hardware.

## Validation

- `cargo test -p wifi-densepose-mat` — parser tests over synthetic fixtures:
  valid records at 20/80/160 MHz shapes, truncated buffer, dimension
  mismatch, version mismatch, allocation-cap enforcement; adapter replay
  determinism.
- `cargo bench -p wifi-densepose-mat` — criterion benchmark for record parse
  throughput at 1992-subcarrier frames.
- Hardware validation on real AX210 silicon is explicitly out of scope for
  this PR and remains required (per CLAUDE.md) before any capture-path
  hardware claim; the file-replay path is testable without silicon.

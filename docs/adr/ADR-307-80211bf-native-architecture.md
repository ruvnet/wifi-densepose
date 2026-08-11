# ADR-307: 802.11bf-native architecture — standardized WLAN sensing as native measurement types

- **Status**: Proposed (ADR-297 phase 2)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: 80211bf, wlan-sensing, standards, measurement-types, hal, phase-2

## Context

This ADR is a child of **ADR-297** and owns primitive #10, *802.11bf-native
architecture*. In the ADR-297 phasing it is a phase-2 integration primitive: it
sits on the phase-1 spine (authenticated identity ADR-302, spatial ontology
ADR-303, evidence engine ADR-301) and **feeds ADR-317** (the RuView sensor HAL),
which is the clause of the acceptance test that "identifies the hardware." It is
authored as **Proposed**.

**IEEE 802.11bf-2025 ("WLAN Sensing") was published 2025-09-26** — verified
against the IEEE SA record in `wifi-densepose-hardware` (`ieee80211bf/mod.rs`
header, "evidence grade MEASURED", ADR-152 §1.1). Standardization is complete
for sub-7 GHz and >45 GHz (DMG) bands: formal sensing measurement setup,
measurement instances, feedback/reporting, and sensing-by-proxy (SBP). This
changes RuView's strategic frame: rather than treating every WiFi measurement as
an *opportunistic* extraction from incidental traffic, RuView can be the **open
reference sensing stack around the standard** — the day commodity silicon
exposes it.

Substantial scaffolding already exists and must be **reused/extended, not
rebuilt**. `v2/crates/wifi-densepose-hardware/src/ieee80211bf/` already models
the standardized procedure surface as forward-compatible types (ADR-152/153):

- `types` — `SpecProfile` version gates, `SensingRole`/`TransceiverRole`,
  `MeasurementSetupParams`, `SensingCapabilities` negotiation, and required
  `ConsentMode` governance metadata on every setup.
- `messages` — `SensingMeasurementSetupRequest/Response`,
  `SensingMeasurementInstance`, `SensingMeasurementReport`, `CsiReportPayload`,
  `SbpRequest/Response`, `SensingSessionTermination`.
- `session` — a deterministic FSM (`Idle → SetupNegotiating → Active →
  Terminating → Idle`) with rejection paths, single-role enforcement, and SBP
  proxy mode; `table` (responder-side setup registry); `transport` (the
  `SensingTransport` seam, a `SimTransport` test double, and an
  `OpportunisticCsiBridge` that maps today's opportunistic CSI onto the
  standardized report path).

The module's own honesty note is authoritative and carried forward here: it is
**not a certified 802.11bf implementation**, and **no commodity silicon — ESP32
included — implements the standard yet**; the OTA frame binding lands when a
chipset exposes it. Wideband ingest plumbing is already in place too: **ADR-289**
(FeitCSI/AX210) carries native subcarrier dimensionality end-to-end and records
the native→pipeline mapping, and noted that "truncated CIR is a natural
extension of the same plumbing."

What is missing is architectural, not protocol scaffolding: normalized CSI is
still treated as *the* WiFi input. The standardized sensing measurements
(TB/non-TB soundings, truncated CIR / PDP reports) are modeled as protocol
messages but are **not yet first-class native measurement types** that flow
through calibration (ADR-298), fusion (ADR-308), and the ontology (ADR-303) on
equal footing with normalized CSI.

## Options considered

1. **Keep 802.11bf as a protocol model only; always down-convert its reports to
   normalized CSI at ingest.** Rejected: truncated CIR/PDP carry range-resolved
   multipath structure that flattening to a CSI matrix discards; it also wastes
   the standard's native report semantics.
2. **Fork a parallel "bf pipeline" alongside the CSI pipeline.** Rejected:
   duplicates calibration, fusion, ontology, and evidence plumbing, and re-opens
   the O(surfaces²) translation problem ADR-303 exists to close.
3. **Promote standardized sensing measurements to native measurement types
   inside the existing pipeline**, with normalized CSI as one measurement type
   among several. Chosen.

## Decision

Adopt an **802.11bf-native architecture**: standardized WLAN sensing
measurements become **additional native measurement types**, alongside — not
replacing — normalized CSI.

### 1. Native measurement types

- Define the standardized reports the `ieee80211bf` module already models
  (TB and non-TB soundings; truncated CIR; PDP) as first-class
  `MeasurementType` variants that the pipeline carries end-to-end, each tagged
  with its `SpecProfile` and band. Normalized CSI remains one such type; the
  `OpportunisticCsiBridge` remains the path for silicon that only offers
  incidental CSI.
- Truncated CIR/PDP reuse the **ADR-289** subcarrier-agnostic / native-
  dimensionality plumbing (truncated CIR is the stated natural extension); the
  native→pipeline mapping is recorded in frame metadata so downstream stages
  know the true range/spectral resolution of a bf report vs. an interpolated CSI
  frame.

### 2. Ontology and governance binding

- Each standardized measurement becomes an ADR-303 `Observation` node from an
  ADR-302-authenticated `Sensor`, carrying `SemanticProvenance` and exactly one
  `EvidenceLevel` (L0–L5, ADR-282). The `ieee80211bf` `ConsentMode` metadata —
  required on every setup — composes with the ADR-277 policy engine, so a
  standardized session is admitted under the same governance as any other
  sensing task (ADR-280).
- SBP (sensing-by-proxy) sessions attribute the report to the proxying and the
  sensing entities distinctly, so provenance is not laundered through the proxy.

### 3. HAL feed (ADR-317)

- The capability set a device advertises — which `MeasurementType`s, bands,
  bandwidths, roles, and `SpecProfile` it supports — is exactly the descriptor
  **ADR-317** (HAL) needs to "identify the hardware." ADR-307 defines that
  capability descriptor as the projection of `SensingCapabilities`; ADR-317
  consumes it. A device that implements no bf profile advertises only the
  opportunistic-CSI capability.

## Consequences

- RuView is positioned as the open reference stack *around* the standard: when a
  chipset exposes 802.11bf, its native reports flow through calibration, fusion,
  ontology, and evidence with no bespoke pipeline — the plumbing is already
  tested against `SimTransport` and synthetic fixtures.
- Normalized CSI is demoted from "the WiFi input" to "one measurement type,"
  which is the correct framing for a multi-measurement future and prevents the
  bf path from being a second-class citizen.
- **No hardware claim is made or implied.** No commodity silicon implements
  802.11bf yet; this ADR wires the *types and flow*, tested in simulation. Any
  OTA/native-report accuracy claim requires real silicon evidence (a captured
  log) per CLAUDE.md, and any wideband number must be tagged with the capture
  hardware (ADR-289). No benchmark number is invented here.
- This ADR does not re-open ADR-152/153's decision to avoid OTA frame binding
  until silicon exists; it consumes that surface and adds the pipeline
  integration.

## Validation

- `cargo test -p wifi-densepose-hardware` — existing `ieee80211bf` FSM,
  table, and transport tests continue to pass; new tests assert that a
  `SensingMeasurementReport` (TB and non-TB) and a truncated-CIR/PDP report
  round-trip through the pipeline as native `MeasurementType`s.
- `cargo test -p wifi-densepose-mat` — truncated CIR ingest reuses the ADR-289
  subcarrier-agnostic path and records the native→pipeline mapping; dimension/
  version validation on standardized reports mirrors the FeitCSI parser gates.
- Ontology/governance tests: each standardized measurement becomes an ADR-303
  `Observation` from an ADR-302-authenticated `Sensor` with one `EvidenceLevel`;
  `ConsentMode` composes with ADR-277 admission; SBP attributes proxy vs. sensor
  provenance distinctly.
- HAL contract test: the ADR-317 capability descriptor is derivable from
  `SensingCapabilities`; a bf-less device advertises only opportunistic CSI.
- All measurement-type flows are simulation-tested (`SimTransport`, synthetic
  fixtures); OTA binding and any hardware accuracy claim remain out of scope
  until real silicon exposes the standard.

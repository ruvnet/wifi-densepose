# ADR-270: Vendor RF Sensing Integration Program

- **Status**: accepted
- **Date**: 2026-07-18
- **Deciders**: RuView maintainers
- **Tags**: vendors, csi, telemetry, simulator, rust, hardware-validation

## Context

RuView is evaluating Qualcomm, RF Solutions, Origin AI, Plume, Linksys,
Electric Imp, Mist/Juniper, Luma, Google Nest, NETGEAR and Wifigarden. These
names do not represent equivalent integration surfaces: some expose raw CSI,
some expose derived sensing events or network telemetry, and some expose no
supported developer interface. A repeated implementation process must not turn
brand compatibility, Linux connectivity or synthetic fixtures into a false CSI
claim.

## Decision

Adopt a Rust-first provider portfolio with explicit capability negotiation:

- `ComplexCsi`: calibrated per-packet complex channel matrices.
- `DerivedSensing`: vendor-produced motion, occupancy or location events.
- `RfTelemetry`: RSSI, radio, client and topology observations.
- `NetworkOnly`: useful as excitation/AP infrastructure but not a sensor.
- `Unsupported`: no stable, lawful or supportable integration surface.

Every provider follows the same gated loop:

1. Verify an authoritative API/SDK, exact model/chipset and licensing boundary.
2. Write provider and wire/contract ADRs before coupling core code to a vendor.
3. Implement bounded Rust types, explicit capabilities and synthetic provenance.
4. Test deterministic replay, corruption, loss, reconnect, backpressure, schema
   evolution and secrets handling.
5. Promote to hardware support only after lawful physical capture on an exact
   model/firmware, calibration and repeatability tests, and fixture publication
   rights. Simulator success never satisfies this gate.
6. Publish code/release and an upstream or vendor collaboration announcement
   that states the measured-versus-simulated boundary.

### Portfolio decisions

| Provider | Classification | Decision |
|---|---|---|
| Qualcomm QCA9300 | `ComplexCsi` candidate | Implement first physical baseline via established ath9k research tooling; QCS1 adapter ships simulator-first. |
| Qualcomm QCN9074/QCN9274 | experimental `ComplexCsi` | Simulator and protocol now; require confirmed ath11k/ath12k firmware export before hardware claim. |
| Origin AI | commercial `DerivedSensing`, possible CSI | Pursue NDA sandbox/API and raw-data rights; isolate proprietary engine behind provider trait/service boundary. |
| Plume/OpenSync | `RfTelemetry`; Plume Sense is gated `DerivedSensing` | Build optional OVSDB/control-plane adapter; negotiate Sense separately and do not infer raw CSI. |
| Mist/Juniper | `RfTelemetry` + location | Conditional read-only REST/webhook adapter for occupancy, RSSI and coordinates; no CSI claim. |
| NETGEAR | partner-gated `RfTelemetry` | Insight adapter only after API access; exact legacy OpenWrt models remain community experiments. |
| Luma | discontinued OpenWrt salvage target | Generic OpenWrt telemetry/pcap fixture only when already owned; no procurement or Luma CSI source. |
| Google Nest Wifi | `NetworkOnly` | Use as traffic/AP infrastructure; Device Access does not expose router CSI or radio telemetry. |
| Linksys | `Unsupported` for sensing | Linksys Aware reached end of support in 2024; record capability probe only, if needed. |
| Electric Imp | scalar IoT/RSSI telemetry | Optional agent/impCentral bridge for existing fleets; reject as CSI acquisition hardware. |
| RF Solutions | non-Wi-Fi RF/IoT telemetry | Exclude from sensing backend; optional RIoT environmental fusion is a separate future concern. |
| Wifigarden | commercial OEM, capability unknown | Hold implementation pending chipset, schema, offline, calibration and data-rights disclosure. |

### Provider boundary

Core code consumes a vendor-neutral `RfSource`-style contract whose capability
set prevents RSSI, location or derived occupancy from being represented as CSI.
Cloud adapters use bounded async queues, regional endpoints, secret-provider
credentials and explicit data provenance. Proprietary device SDKs live behind a
feature-gated FFI or sidecar boundary and are never redistributed without rights.

## Consequences

### Positive

- The integration loop can be repeated without duplicating unsafe parsers.
- Product integrations remain useful even when only telemetry is available.
- Public releases make hardware confidence and simulator confidence distinct.

### Negative

- Several named vendors cannot produce a legitimate CSI implementation today.
- Commercial providers require contracts, subscriptions, test vectors or NDAs.
- Exact hardware revisions and firmware provenance increase validation effort.

### Neutral

- A no-go or telemetry-only ADR is a completed research outcome, not a failed port.
- Vendor status and APIs must be rechecked before each implementation begins.

## Implementation Status

The ADR-270 provider contract is implemented in Rust. Each portfolio entry has
a descriptor, bounded decoder or explicit fail-closed access state, deterministic
contract fixtures where lawful, registry coverage, and API exposure:

- Origin AI: contract-configured derived-sensing decoder and request plan.
- Plume/OpenSync: read-only OVSDB request plan and RF telemetry decoder.
- Mist/Juniper: regional request configuration, paginated RF/location decoder.
- NETGEAR Insight: regional partner request configuration and telemetry decoder.
- Electric Imp and RF Solutions: bounded scalar telemetry bridges.
- Luma: explicitly experimental generic OpenWrt telemetry bridge.
- Google Nest: network-only contract events; never represented as CSI.
- Linksys: `Unsupported` decoder because Linksys Aware is end-of-support.
- Wifigarden: `ContractRequired` decoder pending a disclosed SDK/schema.

`vendor-rf-sim` generates deterministic, provenance-labelled events for the
eight providers with a defined event contract and refuses to fabricate Linksys
or Wifigarden events. The sensing server exposes provider descriptors and latest
events under `/api/v1/rf/vendors` and accepts validated canonical simulator
events over its existing UDP port. Physical/vendor-cloud validation remains
separate from implementation completeness and is reflected by
`hardware_validated: false` until performed.

## Evidence and Links

- [ADR-268: Qualcomm strategy](ADR-268-qualcomm-atheros-csi-platform.md)
- [OpenSync developer sandbox](https://www.opensync.io/developer)
- [Origin AI Wi-Fi sensing architecture](https://www.originwirelessai.com/wifi-sensing/)
- [Juniper Mist webhook hierarchy](https://www.juniper.net/documentation/us/en/software/mist/automation-integration/topics/topic-map/webhook-hierarchy.html)
- [Linksys product end-of-life](https://www.linksys.com/pages/linksys-product-end-of-life)
- [Google Nest Device Access supported devices](https://developers.google.com/nest/device-access/supported-devices)
- [OpenWrt Luma WRTQ-329ACN](https://openwrt.org/toh/hwdata/luma/luma_wrtq-329acn)
- [NETGEAR Insight compatible devices](https://kb.netgear.com/000048452/What-devices-can-I-discover-monitor-and-manage-with-Insight)
- [Electric Imp imp005 hardware guide](https://developer.electricimp.com/hardware/imp/imp005_hardware_guide)
- [RF Solutions company portfolio](https://www.rfsolutions.co.uk/about-us-i1/)
- [Wifigarden service terms](https://policies.wifigarden.com/en-us/terms-of-service)


# ADR-153: IEEE 802.11bf-2025 Forward-Compatibility Protocol Model for wifi-densepose-hardware

- **Status**: accepted
- **Date**: 2026-06-10
- **Deciders**: ruv
- **Tags**: hardware, protocol, sensing, 802.11bf, forward-compatibility

## Context

IEEE 802.11bf-2025 (WLAN Sensing) is an **Active Standard**: board approval
2025-05-28, published 2025-09-26 (verified against the IEEE SA record,
<https://standards.ieee.org/ieee/802.11bf/11574/>). Its scope modifies the
MAC, HE and EHT PHY service interfaces, plus DMG and EDMG PHYs, for WLAN
sensing in **1–7.125 GHz** and **above 45 GHz** bands, with formal sensing
measurement setup, measurement instance, feedback/reporting, and
sensing-by-proxy (SBP) procedures (ADR-152 F4, evidence grade MEASURED).

No commodity silicon implements the standard yet — ESP32 parts included.
ADR-152 §2.4 therefore decided "track silicon; no code now", with RuView's
opportunistic CSI extraction remaining the mechanism. That left a gap: when
silicon does land, RuView would have no typed model of the standard's
procedures to bind to, and the integration would start from zero.

ADR-152 §2.4 originally classified 802.11bf as a hardware watch item with no
implementation work until commodity silicon exposes standardized sensing
measurements. This ADR amends that clause: OTA binding remains deferred, but
a pure Rust protocol model, session FSM, transport seam, and opportunistic
CSI bridge will be implemented now so RuView consumers can target a stable
standardized sensing interface before silicon arrives.

The user directed (2026-06-10) that this **forward-compatibility protocol
model** — a protocol surface, not a conformance implementation — be built
now.

## Decision

Implement an `ieee80211bf` **forward-compatibility protocol model** in
`wifi-densepose-hardware` (pure Rust, no internal deps, simulation-testable,
no OTA path):

> This module is not a certified 802.11bf implementation. It models the
> public procedure shape needed by RuView and RuvSense, while intentionally
> avoiding OTA frame binding until chipset support and vendor APIs exist.

1. **`types.rs`** — typed structures for the standard's sensing procedures
   (sub-7 GHz focus; DMG stubbed): Sensing Measurement Setup (setup ID,
   initiator/responder and transmitter/receiver roles, bandwidth,
   periodicity, threshold-based reporting parameters), Sensing Measurement
   Instance, Sensing Measurement Report (CSI-variant payload), SBP
   request/response, termination. Two future-proofing requirements:

   - **Version gates** — every negotiated surface is tagged with a spec
     profile, because vendors will expose partial or renamed capabilities
     first:

     ```rust
     pub enum SpecProfile {
         DraftCompatible,
         Ieee80211Bf2025,
         VendorExtension(String),
     }
     ```

   - **Capability negotiation** — no hardcoded ESP32 assumptions in the
     future-silicon path:

     ```rust
     pub struct SensingCapabilities {
         pub sub_7_ghz: bool,
         pub dmg: bool,
         pub edmg: bool,
         pub csi_report: bool,
         pub threshold_reporting: bool,
         pub sensing_by_proxy: bool,
         pub max_bandwidth_mhz: u16,
         pub max_period_ms: u32,
         pub max_active_setups: u16,
     }
     ```

   - **Privacy and governance fields** — sensing is presence inference, not
     just radio telemetry. Every `SensingMeasurementSetup` carries policy
     metadata (required, not optional), for enterprise, elderly-care,
     retail, workplace, and municipal deployments:

     ```rust
     pub enum ConsentMode {
         LabOnly,
         ExplicitConsent,
         ManagedEnterprisePolicy,
         Disabled,
     }
     ```

2. **`session.rs`** — deterministic event-driven session state machine:
   `Idle → SetupNegotiating → Active → Terminating → Idle`, with explicit
   rejection paths (unsupported parameters, setup-ID collision) and timeout
   handling.
3. **`transport.rs`** — a `SensingTransport` trait abstracting frame
   exchange; a `SimTransport` test double; and an `OpportunisticCsiBridge`
   adapter mapping today's ESP32 CSI extraction onto the report path
   (measurement instances ≈ CSI frame batches), so current hardware sits
   behind the standardized interface. **Replaceability benchmark
   (acceptance test):** RuvSense must consume either ESP32 opportunistic CSI
   or future 802.11bf chipset reports through the same `SensingTransport`
   and `SensingMeasurementReport` path, with no consumer-side rewrite — a
   future chipset adapter replaces `OpportunisticCsiBridge` without changing
   consumers.

Constraints: input validation at boundaries (typed errors, no panics on
adversarial input), files under 500 lines, all protocol tests runnable
without hardware.

### Acceptance checklist

| Area            | Acceptance test                                                      |
| --------------- | -------------------------------------------------------------------- |
| Types           | Serde round trip for setup, instance, report, SBP, termination       |
| FSM             | Idle → setup → active → terminating → idle                           |
| Rejection       | Unsupported bandwidth, invalid period, duplicate setup ID            |
| Timeout         | Negotiation timeout returns typed error and resets to Idle           |
| Threshold       | Report emitted only when threshold condition is crossed              |
| SBP             | Proxy request maps to responder path without direct sensor coupling  |
| Bridge          | ESP32 CSI batch becomes standardized measurement report              |
| Safety          | No panics on malformed inputs                                        |
| CI              | All protocol tests run without hardware                              |
| Maintainability | Each file under 500 lines                                            |

### Non-Goals

This ADR does not claim IEEE 802.11bf conformance, certification, or OTA
interoperability. It creates a typed protocol compatibility layer so RuView
can consume standardized sensing reports when commodity silicon exposes
them. Vendor-specific frame exchange, firmware hooks, trigger-frame
sounding, and certification test vectors remain future ADRs.

## Consequences

### Positive
- RuView can adopt standardized WLAN sensing the day any chipset exposes
  802.11bf measurements — the data model, session FSM, and transport seam
  already exist and are tested.
- The `OpportunisticCsiBridge` gives current ESP32 nodes a standardized-shape
  interface now, decoupling RuvSense consumers from the extraction mechanism.
- Simulation transport enables protocol-level tests in CI without hardware.
- `SpecProfile` + `SensingCapabilities` give a clean escape hatch for the
  partial/renamed vendor capabilities that will certainly arrive first.
- Consent/policy metadata is structural from day one, not retrofitted.

### Negative
- Code written against a standard with zero silicon risks drift: vendor
  implementations may interpret parameters differently; the layer may need
  rework at first real binding (drift risk scored 7/10 at acceptance).
- Adds maintenance surface to wifi-densepose-hardware before any
  user-visible benefit (maintenance cost scored 3/10 — small without OTA).

### Neutral
- ADR-152 §2.4's "watch item" remains: revisit when silicon/certification
  appears (re-check by 2026-12). This ADR changes only the "no code now"
  clause.

## Links

- ADR-152 — WiFi-Pose SOTA 2026 Intake (F4, §2.4 — amended by this ADR)
- ADR-028 — ESP32 capability audit (opportunistic CSI extraction baseline)
- ADR-029 — RuvSense multistatic sensing mode (consumer of sensing reports)
- IEEE 802.11bf-2025 — Active Standard, board approval 2025-05-28, published
  2025-09-26: <https://standards.ieee.org/ieee/802.11bf/11574/>

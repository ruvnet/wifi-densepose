# ADR-282: Ecosystem positioning — RuView is the camera-free RF perception runtime, not the whole spatial OS

| Field | Value |
|-------|-------|
| **Status** | Accepted (positioning + evidence-ladder policy; ladder implemented as `frame::EvidenceLevel`) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 |
| **Relates to** | ADR-260/262 (RuField), ADR-261 (RuVector), ADR-182 (MetaHarness-minted harness), ADR-279 (provenance/evidence types), ADR-187 (honest labeling precedent) |

## 1. Context

RuView currently occupies a valuable but ambiguous position: the README's breadth invites reading every capability as field-validated, and the platform sometimes speaks as if it were the complete spatial intelligence operating system. The defensible identity is narrower and stronger.

## 2. Decision — the layered identity

> **RuView is an open, edge-native RF perception runtime that turns heterogeneous radio measurements into governed spatial observations.**

It is *not* positioned as a complete world model, robotics platform, digital twin, or universal spatial OS. The stack divides:

| Layer | Responsibility | Owner |
|---|---|---|
| Applications | healthcare, buildings, robotics, security, retail, industrial | application systems |
| Agent & decision | query planning, active sensing, automation, policy | **MetaHarness** |
| Spatial memory & reasoning | persistent objects, Gaussian fields, scene graphs, temporal memory | **RuVector** (fed by `ruview-unified::gaussian`) |
| Governed sensing plane | evidence, privacy, calibration, lineage, sensing tasks | **RuField** (bridged per ADR-262; contracts in ADR-277/279/280) |
| Perception & edge inference | native capture, adapters, shared encoder, task heads, uncertainty, P0 containment | **RuView** |
| Radio & physical sensors | WiFi CSI/CIR/BF, radar, UWB, BLE CS, cellular SRS | hardware |

Competitive posture follows from the layer: **complement vision platforms** (coverage where cameras are unavailable, unwanted, or ineffective — never "replaces cameras universally"); one shared encoder + spatial field across CSI and radar; BLE/UWB as geometric anchors with WiFi for ambient sensing; and against 6G ISAC, be the practical open implementation of the sensing data plane on hardware that exists today.

## 3. Decision — strengths to invest, weaknesses to fix

Invest (already differentiated): low-cost ambient perception on commodity radios; camera-free coverage (with the explicit caveat that camera-free ≠ privacy-preserving — that is what ADR-277/280 gates are for); edge-first execution; existing application surfaces (HA/Matter/HomeKit), to be extended toward ROS 2, OpenUSD, MQTT Sparkplug, OPC UA, BIM/digital-twin connectors as demand proves out.

Fix (each has a concrete ADR): platform/world-model claim mixing → this ADR's ladder; no persistent spatial representation → ADR-275 (feed RuVector, don't contain everything in the sensing server); ESP32-specific pipeline risk → ADR-279 adapters; stream-only operation → ADR-280 sensing tasks.

## 4. Decision — the public evidence ladder (mandatory)

`frame::EvidenceLevel` is now a type, and its use is policy:

| Level | Meaning |
|---|---|
| L0 | Simulation only |
| L1 | Captured replay |
| L2 | Controlled laboratory |
| L3 | Held-out room + subject validation |
| L4 | Multi-site field pilot |
| L5 | Production operational evidence |

Rules: (a) every capability row in README/registry carries exactly one level; (b) `ProvenanceClass::Synthetic` frames are L0 *by type* and measured frames are ≥ L1 — the constructor rejects both aliasing directions (ADR-279 invariant 6); (c) a level upgrade requires the corresponding artifact (a replay corpus, a lab protocol, a strict-split manifest per ADR-279 §4, a pilot report); (d) the hidden real-world test set used for L3+ claims is never accessible to synthetic generation, augmentation, or calibration. Everything shipped in ADR-273..281 is **L0** except the adapter/contract layers, which are code-level (no accuracy claim to grade).

## 5. Commercial focus (bounded claims per vertical)

Elder care (decision support and anomaly escalation, **not** diagnosis); smart buildings (occupancy/utilization; value = energy + space + safety − cost); industrial safety (works in dust/darkness/occlusion; **not** a certified safety system until field-validated); security (through-wall occupancy with the surveillance-governance gates of ADR-277/280 as a feature, not friction); robotics (RuView is probabilistic exteroception, never ground truth).

## 6. The moat

Not any single detector: the *combination* of broad hardware support (ADR-279 adapters), heterogeneous data with provenance, cross-environment pretrained encoders under anti-leakage evaluation (ADR-273 §4), calibration/uncertainty discipline, privacy-preserving edge execution (ADR-277/280), cryptographic evidence (RuField bridge), persistent spatial memory (ADR-275 → RuVector), and open integration. Harder to reproduce than any model.

## 7. Acceptance test (ecosystem-fit)

RuView fits the mature stack when a **frozen** encoder ingests WiFi CSI, radar, and Bluetooth measurements from previously unseen hardware, emits RuField-compliant observations, updates a persistent RuVector spatial model, and supports an agent query with: ≤ 0.5 m p90 localization; < 20 % degradation across unseen rooms; explicit uncertainty on every result; complete calibration + provenance lineage; no P0 RF leaving the edge; replay/lab/live evidence clearly separated; successful fusion with a standard robotics or digital-twin platform. Tracked as the L4 gate; the synthetic analogue machinery already exists (`tests/e2e_acceptance.rs`).

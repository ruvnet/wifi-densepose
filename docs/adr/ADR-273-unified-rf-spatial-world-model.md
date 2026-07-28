# ADR-273: Unified RF Spatial World Model — one shared representation, not another isolated RF classifier

| Field | Value |
|-------|-------|
| **Status** | Accepted — **P1 implemented** (new v2 workspace crate `ruview-unified`; 66 unit + 3 acceptance-pipeline tests, 0 failed; criterion benches) |
| **Date** | 2026-07-26 |
| **Deciders** | ruv |
| **Codebase target** | `v2/crates/ruview-unified/` (new leaf crate; single internal dep on `wifi-densepose-core` for `CsiFrame`) |
| **Sub-ADRs** | ADR-274 (universal RF encoder + adapter registry), ADR-275 (RF-aware Gaussian spatial memory), ADR-276 (physics-guided synthetic RF worlds), ADR-277 (edge sensing control plane), ADR-278 (radar inverse rendering research program) |
| **Relates to** | ADR-152 (WiFi-Pose SOTA intake: geometry conditioning), ADR-153 (802.11bf protocol model), ADR-260/262 (RuField MFS + bridge), ADR-135/136 (calibration + canonical frame provenance), ADR-024 (AETHER), ADR-027 (MERIDIAN domain generalization) |
| **Scope** | Decide the target architecture for RuView + RuVector sensing through 2026-H2: one persistent, queryable spatial world model that vision, WiFi CSI, cellular CFR/SRS, radar, geometry, semantics, uncertainty, and time all update — and the priority order for building it. |

---

## 0. PROOF discipline

Every number in this ADR family is one of:

- **MEASURED-SYNTHETIC** — produced by this repo's tests/benches on data from the ADR-276 physics generator. Reproducible: `cd v2 && cargo test -p ruview-unified` / `cargo bench -p ruview-unified`. **No claim of real-world accuracy is made or implied.**
- **MEASURED-CODE** — a structural property of the implementation (parameter counts, gradient-check error, determinism), verified by a named test.
- **EXTERNAL-UNVERIFIED** — a number reported by an external paper/preprint (WiFo-2, WiLHPE, RISE, DiffRadar, HybridSim, OAI SRS demo, …) that this repo has **not** reproduced. These motivated design choices; they are never presented as our results.

## 1. Context

Through mid-2026 the field moved decisively away from task-specific RF classifiers:

1. **RF foundation models** (WiFo-2 scaling across 11.6 B CSI points/12 tasks; WiLLM's dataset adapters + shared self-supervised transformer; age-aware CSI fusion) — the architectural signal: *standardize heterogeneous CSI, pretrain with masked reconstruction, attach small task adapters* (all EXTERNAL-UNVERIFIED).
2. **Gaussian fields as spatial memory** (EmbodiedSplat online semantic 3-D Gaussian mapping; TGSFormer bounded temporal Gaussian memory; July's physics-informed channel-gain mapping with incremental Gaussian insertion) — the missing bridge between RuView sensing and a queryable digital twin.
3. **Synthetic RF worlds** (WaveVerse phase-coherent ray tracing; HybridSim's 92 % vs 54 % synthetic-to-real gap when *physics parameters*, not textures, are randomized) — the fastest path out of data scarcity.
4. **Standards became actionable**: IEEE 802.11bf-2025 published (2025-09), 802.11bk (320 MHz positioning), ETSI ISAC architecture (2026-02) + security report (19 privacy/security issue classes), 3GPP Rel-20 sensing studies, OAI SRS xApp localization demo.
5. **Generalization lessons**: PerceptAlign (condition on TX/RX geometry), RePos (factor root-relative pose from absolute localization), JITOMA (task-gated scene memory).

RuView already has the ingredients (calibration ADR-151, canonical frames ADR-136, ruvsense multistatic stack, RuField bridge ADR-262) but they update **separate** state. The decision is to converge on **one shared representation with persistent scene memory**.

## 2. Decision

Build the unified model as five pillars in strict priority order (scored 35 % business value / 25 % readiness / 20 % defensibility / 20 % strategic learning):

| # | Pillar | Score | Sub-ADR | P1 status |
|---|--------|-------|---------|-----------|
| 1 | Universal RF foundation encoder + hardware adapter registry | 4.7 | ADR-274 | **implemented** |
| 2 | RF-aware Gaussian spatial memory | 4.5 | ADR-275 | **implemented** |
| 3 | Age/geometry/uncertainty-aware inference (folded into the encoder contract) | 4.4 | ADR-274 §3 | **implemented** |
| 4 | Physics-guided synthetic RF world generator | 4.1 | ADR-276 | **implemented** |
| 5 | Edge sensing control plane (802.11bf / ETSI ISAC aligned) | 3.9* | ADR-277 | **implemented** (policy engine; O-RAN xApp is roadmap) |
| 6 | Radar inverse rendering + differentiable RF SLAM | 3.6 | ADR-278 | research program (not implemented) |

\* the 3.9-scored item is the O-RAN SRS xApp; its *policy plane* and its *SRS adapter seam* ship in P1 because they are cheap and gate everything else.

The representation contract every pillar shares:

```text
z = Encoder(RF tokens) ⊙ σ(AgeEncoder(age)) + GeometryEncoder(sensor_pose)
```

served from one canonical tensor (`RfTensor`, ADR-274 §2) and persisted into one scene memory (`GaussianMap` + task-gated `SceneGraph`, ADR-275).

## 3. Architecture (implemented, `v2/crates/ruview-unified/src/`)

```text
vendor captures ──▶ adapters.rs (WiFi CSI / FMCW cube / UWB CIR / 5G SRS)
                        │  normalize: layout → gain → phase   (ADR-274 §2.3)
                        ▼
                 tensor.rs  RfTensor (links × 56 bins × 8 snapshots, complex)
                        │
                 tokenizer.rs  amplitude/delay/Doppler/phase/age/geometry/
                        │      clock/uncertainty tokens (CFO-aligned,
                        │      median-scale-normalized)
                        ▼
                 encoder.rs + pretrain.rs   masked-reconstruction pretraining,
                        │      exact hand-derived backprop (gradient-checked)
                        ▼
        ┌── heads.rs  ≤1 % task adapters (presence/activity/localization/anomaly)
        │
        ├── gaussian/  RF-aware Gaussian memory: fusion, decay, channel-gain
        │              queries, inverse updates, task-gated scene graph
        │
        └── policy.rs  purposes/zones/retention/identity gating; BoundedEvent
                       is the only exportable type (raw RF unrepresentable)
```

`synth/` (ADR-276) generates the labeled physics worlds that train and gate all of it; `eval.rs` implements the anti-leakage protocol below.

## 4. The non-negotiable evaluation protocol (anti-leakage)

The biggest failure mode in this field is **domain leakage disguised as accuracy**: random frame splits let a model recognize the room, session, person, device, or trajectory. Bigger models make it worse. Therefore:

- **No result counts unless the test set holds out complete** rooms, days, people, chipsets, firmware versions, and antenna layouts. `eval::StrictSplit` constructs such splits and `verify()` independently proves disjointness (`eval.rs`; test `verify_catches_a_manufactured_leak`).
- Track **relative degradation** known→unknown (`relative_degradation`, gate < 20 %), **calibration** (`expected_calibration_error`), and **abstention quality** (`selective_metrics` — an uncertain result must become *no decision*, not a confident guess).
- Every synthetic number is labeled SYNTHETIC in test output and in these ADRs.

## 5. Acceptance gates — P1 (synthetic analogue) results

The ADR's acceptance test (frozen shared encoder, adapters < 1 % of backbone, unseen rooms/chipsets/layouts) is implemented end-to-end in `tests/e2e_acceptance.rs`. **MEASURED-SYNTHETIC** results on the ADR-276 generator (8 rooms × 20 windows × 3 links, seed 273273):

| Gate (ADR target) | P1 synthetic result | Verdict |
|---|---|---|
| Presence F1 ≥ 0.90, unseen rooms | **1.0000** (rooms 6–7 held out of pretraining *and* head training) | pass |
| Presence F1 ≥ 0.90, unseen chipset | **1.0000** (`chip-2` held out; per-room random gain/phase/CFO/noise) | pass |
| Cross-environment degradation < 20 % | **0.0000** | pass |
| Adapter budget < 1 % of backbone | presence 129 / activity 268 / localization 387 / anomaly 2 params vs 40,856-param backbone (< 408) | pass (MEASURED-CODE) |
| Edge latency p95 < 50 ms | **2.0 ms** debug profile (tokenize+encode); 105 µs encode / 67 µs tokenize release (criterion) | pass |
| Held-out ECE | **0.0122**; abstention risk monotone in threshold | pass |
| Raw RF never crosses the trust boundary | structural: only `policy::BoundedEvent` exports (no tensor-carrying variant exists) | pass |
| Every output carries uncertainty, provenance, model version, purpose | enforced at `BoundedEvent::new` (construction fails otherwise) | pass |

**Honest reading**: a synthetic world where presence ⇔ a moving scatterer is *separable by construction*; F1 = 1.0 here validates the **pipeline and the anti-leakage machinery**, not real-world performance. The real-data gate (5 unseen rooms, 2 unseen chipsets, 2 unseen layouts, measured CSI) is P2 and remains open.

## 6. Consequences

- RuView gains a single, tested substrate that all future sensing work (vision fusion, SRS xApp, radar) updates instead of forking.
- The synthetic-first discipline means every accuracy claim is grade-labeled; publishing an unlabeled number is now a process violation.
- The Gaussian memory becomes the integration point for RuVector (vector retrieval → graph constraints → geometric verification; the LLM plans the query, the renderer verifies the answer).
- Cost: a new crate to maintain (~4.6 k lines incl. tests); mitigations: zero heavy deps, deterministic tests, files < 500 lines each.

## 7. Roadmap after P1

| Phase | Content | Gate |
|-------|---------|------|
| P2 | Replay real `.csi.jsonl` (rvCSI / ADR-262 corpus) through the WiFi adapter; calibrate the anomaly head on real empty-room captures | strict-split F1/ECE on measured data, reported with degradation vs synthetic |
| P3 | Wire `GaussianMap` into `wifi-densepose-sensing-server` behind the ADR-277 boundary; RuVector embedding of Gaussian clusters | live map consistency + bounded-event-only egress audit |
| P4 | OAI SRS xApp feeding `CellularSrsAdapter` (the adapter + registry seam already exists) | 0.5 m p90 localization under *non-random* splits |
| P5 | ADR-278 radar inverse rendering reproduction (RISE first) | 

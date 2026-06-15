# ADR Corpus Census

Full per-ADR census underpinning ADR-164. **162 ADR entries across 156 distinct files** (the 5 duplicate-number collisions / 6 displaced files have been RESOLVED — displaced files renumbered to ADR-166…171 per ADR-164 G1; the ADR-134 identity split is tracked separately under G3). Source of truth for the gap-analysis lenses. Where the census is uncertain it is marked *needs verification*.

| ADR | Title | Status | impl_state | Flags |
|-----|-------|--------|-----------|-------|
| ADR-001 | WiFi-Mat Disaster Detection Architecture | Accepted | implemented | data/hardware-gated (rubble-penetration unproven without field hardware) |
| ADR-002 | RuVector RVF Integration Strategy | Superseded by ADR-016 + ADR-017 | superseded | umbrella ADR; child ADRs 003/007/008/009/010 still Proposed |
| ADR-003 | RVF Cognitive Containers for CSI Data | Proposed | proposed-only | proposed-but-looks-abandoned (parent 002 superseded, never advanced) |
| ADR-004 | HNSW Vector Search for Signal Fingerprinting | Partially realized by ADR-024; extended by ADR-027 | partial | realized indirectly via downstream ADRs, not directly |
| ADR-005 | SONA Self-Learning Pose Estimation | Partially realized in ADR-023; extended by ADR-027 | partial | realized indirectly via ADR-023 (MicroLoRA/EWC++) |
| ADR-006 | GNN-Enhanced CSI Pattern Recognition | Partially realized in ADR-023; extended by ADR-027 | partial | realized indirectly via ADR-023 (2-layer GCN), scope narrowed |
| ADR-007 | Post-Quantum Cryptography for Secure Sensing | Proposed | proposed-only | proposed-but-looks-abandoned (parent 002 superseded) |
| ADR-008 | Distributed Consensus for Multi-AP | Proposed | proposed-only | proposed-but-looks-abandoned (parent 002 superseded) |
| ADR-009 | RVF WASM Runtime for Edge Deployment | Proposed | proposed-only | contradicts shipped wifi-densepose-wasm crate it proposes to replace |
| ADR-010 | Witness Chains for Audit-Trail Integrity | Proposed | proposed-only | witness-bundle (ADR-028) fills this role instead |
| ADR-011 | Python Proof-of-Reality / Mock Elimination | Proposed (URGENT) | partial | proof pipeline (verify.py/ADR-028) live despite Proposed status; credibility-gated |
| ADR-012 | ESP32 CSI Sensor Mesh | Accepted — Partially Implemented | partial | hardware-gated; mesh partial, single-node firmware working per ADR-018 |
| ADR-013 | Feature-Level Sensing on Commodity Gear | Accepted — Implemented (36/36 tests) | implemented | — |
| ADR-014 | SOTA Signal Processing | Accepted | implemented | — |
| ADR-015 | Public Dataset Training Strategy | Accepted | implemented | data-gated (MM-Fi/Wi-Pose availability/licensing) |
| ADR-016 | RuVector Training-Pipeline Integration | Accepted | implemented | supersedes ADR-002 (but file never mentions 002 — unsupported claim) |
| ADR-017 | RuVector Signal + MAT Integration | Accepted | implemented | CLAUDE.md still lists as Proposed; supersedes 002 only via "Correction" prose |
| ADR-018 | ESP32 Dev Implementation | Proposed | partial | status stale — ADR-012 cites it as working firmware/aggregator |
| ADR-019 | Sensing-Only UI Mode with Gaussian Splat Viz | Accepted | implemented | status in table format not ## header |
| ADR-020 | Migrate AI/Model Inference to Rust (RuVector + ONNX) | Accepted | partial | table-format status; overlaps ADR-019 backend-decoupling scope |
| ADR-021 | Vital Sign Detection via rvdna Pipeline | Partially Implemented | partial | wifi-densepose-vitals crate exists |
| ADR-022 | Enhanced Windows WiFi Fidelity via Multi-BSSID | Partially Implemented | partial | wifi-densepose-wifiscan crate exists |
| ADR-023 | Trained DensePose Model w/ RuVector Signal Intelligence | Proposed | proposed-only | data/hardware-gated; scaffold w/ random weights |
| ADR-024 | Project AETHER — Contrastive CSI Embedding | Proposed | proposed-only | CLAUDE.md lists Accepted; pose_tracker.rs uses AETHER re-ID — contradiction |
| ADR-025 | macOS CoreWLAN WiFi Sensing (ORCA) | Proposed | proposed-only | hardware-gated (Mac Mini M2 Pro); RSSI-only |
| ADR-026 | Survivor Track Lifecycle Management (MAT) | Accepted | implemented | explicit Supersedes: None |
| ADR-027 | Project MERIDIAN — Cross-Env Domain Generalization | Proposed | proposed-only | CLAUDE.md lists Accepted — contradiction; data-gated |
| ADR-028 | ESP32 Capability Audit & Witness Record | Accepted | implemented | audit/witness record; pins commit 96b01008 |
| ADR-029 | RuvSense — Sensing-First RF Multistatic Mode | Proposed | stale-or-contradicted | repo has ruvsense/ (16 modules); ADR-032 hardens it |
| ADR-030 | RuvSense Persistent Field Model | Proposed | stale-or-contradicted | field_model/longitudinal/cross_room modules exist; ADR-032 secures |
| ADR-031 | RuView — Cross-Viewpoint Fusion | Proposed | stale-or-contradicted | ruvector/src/viewpoint/ exists; near-duplicate of ADR-029 |
| ADR-032 | Multistatic Mesh Security Hardening | Accepted | implemented | hardens Proposed 029/030/031 — status-graph inversion |
| ADR-033 | CRV Signal Line Sensing (Coordinate Remote Viewing) | Proposed | proposed-only | speculative/metaphor-driven; abandonment risk |
| ADR-034 | Expo React Native Mobile App (FieldView) | Accepted | unknown | no mobile-app crate/dir in CLAUDE.md — unverified |
| ADR-035 | Live Sensing UI Accuracy & Data Source Transparency | Accepted | implemented | bug-fix; heuristic pose superseded in spirit by 023/036 |
| ADR-036 | RVF Model Training Pipeline & UI Integration | Proposed | proposed-only | overlaps ADR-023 scope |
| ADR-037 | Multi-Person Pose from Single ESP32 CSI Stream | Proposed | proposed-only | explicit Supersedes: None; HW limitation noted |
| ADR-038 | Sublinear GOAP for Roadmap Optimization | Proposed | proposed-only | meta/process ADR; own corpus census may be stale |
| ADR-039 | ESP32-S3 Edge Intelligence Pipeline | Accepted (hardware-validated) | implemented | hardware-validated |
| ADR-040 | WASM Programmable Sensing (Tier 3) | Accepted | implemented | depends on ADR-039; WASM3 optional |
| ADR-041 | WASM Module Collection — Sensing Registry | Accepted (Phase 1) | partial | ~57 modules catalog/proposed; exotic modules speculative |
| ADR-042 | Coherent Human Channel Imaging (CHCI) | Proposed | proposed-only | hardware-gated (custom PCB/TCXO); superseded-in-intent by ADR-153 |
| ADR-043 | Sensing Server UI API Completion | Accepted | implemented | internal route count contradiction (14 vs 17) |
| ADR-044 | Geospatial Satellite Integration | Accepted | unknown | no Date/Deciders; wifi-densepose-geo crate not in CLAUDE.md table |
| ADR-045 | AMOLED Display Support for ESP32-S3 | Proposed | proposed-only | hardware-gated (LilyGO T-Display-S3); ADR-048 depends on it |
| ADR-046 | Android TV Box / Armbian Deployment Target | Proposed | proposed-only | proposed-but-looks-abandoned; Phase 2 speculative |
| ADR-047 | RuView Observatory — Three.js Visualization | Accepted (Implemented) | implemented | — |
| ADR-048 | Adaptive CSI Activity Classifier | Accepted | implemented | depends on Proposed ADR-045 |
| ADR-049 | Cross-Platform WiFi Detection & Graceful Degradation | Proposed | proposed-only | targets Python v1 legacy; abandonment risk |
| ADR-050 | Provisioning Tool Enhancements | Proposed | partial | keeps 050 (collision resolved); partially fulfilled by ADR-060 |
| ADR-166 | Quality Engineering Response — Security Hardening | Accepted | partial | renumbered from ADR-050 (collision resolved); unverified claims (54K fps); findings #6-8 unconfirmed |
| ADR-167 | DDD Bounded Contexts (appendix to ADR-052) | (none — appendix, no Status) | unknown | renumbered from ADR-052 (collision resolved); missing-status; cross-ref errors (cites 044 for provisioning) |
| ADR-052 | Tauri Desktop Frontend — Hardware Mgmt & Viz | Proposed | partial | keeps 052 (collision resolved); superseded_by ADR-054; status drift |
| ADR-053 | UI Design System — Dark Professional | Accepted | implemented | depends on Proposed ADR-052 |
| ADR-054 | RuView Desktop Full Implementation | Accepted — in progress | partial | command matrix mostly Stub; espflash version drift vs 052 |
| ADR-055 | Integrated Sensing Server in Desktop App | Accepted | implemented | — |
| ADR-056 | RuView Desktop Complete Capabilities Reference | Accepted | partial | reference doc; "complete" overstates impl state |
| ADR-057 | Firmware CSI Build Guard & sdkconfig.defaults | Accepted | implemented | minor C6 CSI matrix tension vs CLAUDE.md |
| ADR-058 | Dual-Modal WASM Browser Pose (Video + CSI) | Proposed | partial | data-gated; ships placeholder weights |
| ADR-059 | Live ESP32 CSI Pipeline Integration | Accepted | implemented | hardware-gated (physical ESP32-S3 + UDP:5005) |
| ADR-060 | Provision Channel Override & MAC Filtering | Accepted | implemented | fulfills part of Proposed ADR-050(prov) without superseding |
| ADR-061 | QEMU ESP32-S3 Emulation for Firmware Testing | Accepted | implemented | RF-PHY paths untestable in QEMU |
| ADR-062 | QEMU ESP32-S3 Swarm Configurator | Accepted | implemented | — |
| ADR-063 | 60 GHz mmWave Sensor Fusion with WiFi CSI | Proposed | proposed-only | hardware-gated (ESP32-C6+MR60BHA2); superseded-in-scope by 064 |
| ADR-064 | Multimodal Ambient Intelligence (CSI+mmWave+env) | Proposed | proposed-only | hardware-gated; mixes build-now + speculative tiers |
| ADR-065 | Hotel Guest Happiness Scoring | Proposed | proposed-only | hardware-gated (Cognitum Seed Pi Zero 2 W) |
| ADR-066 | ESP32 CSI Swarm with Cognitum Seed Coordinator | Proposed | proposed-only | hardware-gated; overlaps 068/069 |
| ADR-067 | RuVector v2.0.4→v2.0.5 Upgrade | Proposed | proposed-only | CLAUDE.md still v2.0.4 — not adopted |
| ADR-068 | Per-Node State Pipeline for Multi-Node Sensing | Accepted | implemented | — |
| ADR-069 | ESP32 CSI → Cognitum Seed RVF Ingest Pipeline | Accepted | implemented | hardware-gated (live Cognitum Seed fw v0.8.1) |
| ADR-070 | Self-Supervised Pretraining from Live CSI + Seed | Accepted | partial | hardware-gated (live 2-node + Seed capture) |
| ADR-071 | ruvllm Training Pipeline for CSI Models | Proposed | proposed-only | overlaps 072/079 + libtorch pipeline |
| ADR-072 | WiFlow Pose Estimation Architecture | Proposed | partial | data-gated; referenced as implemented in CLAUDE.md (WiFlow-STD) — stale header |
| ADR-073 | Multi-Frequency Mesh Scanning | Proposed | proposed-only | hardware-gated (2-node multi-AP) |
| ADR-074 | Spiking Neural Network for CSI Sensing | Proposed | proposed-only | proposed-but-looks-abandoned (no in-repo SNN signal) |
| ADR-075 | Min-Cut Person Separation from Subcarrier Corr | Proposed | proposed-only | fixes #348; 077/078 depend on it though Proposed |
| ADR-076 | CSI Spectrogram Embeddings via CNN + Graph Transformer | Proposed | proposed-only | — |
| ADR-077 | Novel RF Sensing Applications | Accepted | partial | depends on Proposed 075/076; data-gated |
| ADR-078 | Multi-Frequency Mesh Sensing Applications | Proposed | proposed-only | hardware-gated; depends on Proposed 073 |
| ADR-079 | Camera Ground-Truth Training Pipeline | Accepted | partial | P7-P9 Pending; internal PCK contradiction (2.5% vs 35.3% vs 0%); #640 = 0% |
| ADR-080 | QE Analysis Remediation Plan | Proposed | proposed-only | unfixed security HIGH findings (XFF bypass, stack traces, JWT-in-URL) |
| ADR-081 | Adaptive CSI Mesh Firmware Kernel | Accepted — L1-5 host-tested | partial | mesh RX + Ed25519 signing deferred to Phase 3.5 |
| ADR-082 | Pose Tracker Confirmed-Track Output Filter | Accepted — implemented | implemented | fixes #420 |
| ADR-083 | Per-Cluster Pi Compute Hop | Proposed — pending field evidence | proposed-only | hardware-gated (status explicitly pending field evidence) |
| ADR-084 | RaBitQ Similarity Sensor (4 pipeline points) | Accepted — merged PR #435 | implemented | acceptance on synthetic data; <1pp regression deferred to soak |
| ADR-085 | RaBitQ Similarity Sensor — Pipeline Expansion (7 sites) | Proposed | proposed-only | proposed-but-looks-abandoned (refines 084, never advanced) |
| ADR-086 | Edge Novelty Gate — RaBitQ on Sensor MCU | Proposed | proposed-only | hardware-gated (no_std port + real-deployment suppression rates) |
| ADR-089 | nvsim — NV-Diamond Magnetometer Simulator | Accepted — Passes 1-5 merged | partial | Pass 6 (proof bundle + bench) pending |
| ADR-090 | nvsim — Full Hamiltonian/Lindblad Solver | Proposed — conditional | proposed-only | explicitly deferred decision-to-defer |
| ADR-091 | Stand-off Radar — 77 GHz / sub-THz Research | Proposed — research only | proposed-only | hardware-gated (COTS sub-THz); ITAR/dual-use |
| ADR-092 | nvsim Dashboard — Vite + Dual-Transport | Implemented (2026-04-27) | implemented | 4/12 gates need external infra; PR #436 open |
| ADR-093 | nvsim Dashboard Gap Analysis | Implemented (2026-04-27) | implemented | P2.7/P2.8 polish deferred |
| ADR-094 | Live 3D Point Cloud Viewer — GH Pages | Proposed (2026-04-29) | proposed-only | governs viewer deploy only, not crate data contract |
| ADR-095 | rvCSI — Edge RF Sensing Runtime Platform | Proposed | implemented | header stale — ADR-097 confirms built, published 0.3.1 |
| ADR-096 | rvCSI — Crate Topology, napi-c Shim, napi-rs | Proposed | implemented | header stale — 9 crates published 0.3.1 |
| ADR-097 | Adopt rvCSI as RuView's primary CSI runtime | Proposed | proposed-only | RuView vendors but does not yet consume — adoption open |
| ADR-098 | Evaluate ruvnet/midstream | Rejected (with carve-outs) | proposed-only | rejection; carve-outs revived by ADR-099 |
| ADR-099 | Adopt midstream — introspection + low-latency tap | Proposed | proposed-only | tension with ADR-098 (which rejected midstream) |
| ADR-100 | Cognitum Cog Packaging Specification | Accepted | implemented | first cog shipped 2026-05-19 (ADR-101) |
| ADR-101 | Pose Estimation Cog (WiFi-DensePose side) | Accepted — v0.0.1 shipped | implemented | hardware-gated; signed binaries on GCS |
| ADR-102 | Edge Module Registry Integration | Accepted | implemented | serves 105-cog catalog |
| ADR-103 | Learned Multi-Person Counter (cog-person-count) | Proposed | proposed-only | data/hardware-gated; claim gutted by ADR-159 |
| ADR-104 | RuView MCP Server + CLI Distribution | Accepted | partial | depends on Proposed ADR-103 for count tool |
| ADR-105 | Federated learning for RuView CSI personalization | Proposed | proposed-only | head of 105-108 chain, none implemented |
| ADR-106 | Differential privacy + biometric isolation | Proposed | proposed-only | extends Proposed 105 |
| ADR-107 | Cross-installation federation w/ secure aggregation | Proposed | proposed-only | classical DH later superseded by 108 |
| ADR-108 | Kyber PQ key exchange for federation | Proposed | proposed-only | extends Proposed 107 (parent unimplemented) |
| ADR-109 | Dilithium PQ signatures for cog distribution | Proposed | proposed-only | extends ADR-100; sister of 108 |
| ADR-110 | ESP32-C6 firmware extension (Wi-Fi 6 CSI, 802.15.4, TWT, LP) | Accepted — P1-P10 complete v0.7.0 | implemented | HE-CSI needs ESP-IDF ≥5.5 (v5.4 downconverts to HT) |
| ADR-113 | Multistatic anchor placement strategy | Proposed | proposed-only | amends ADR-029; simulation-derived not HW-validated |
| ADR-114 | cog-quantum-vitals | Proposed | proposed-only | hardware-gated (nvsim today, real NV-diamond in prod); R13 NEGATIVE |
| ADR-115 | Home Assistant via MQTT auto-discovery + Matter bridge | Accepted (MQTT) / Proposed (Matter) | partial | mixed status; Matter deferred to v0.7.1 |
| ADR-116 | HA + Matter as a Cognitum Seed cog (cog-ha-matter) | Proposed — P2 scaffold compiles | partial | provisional; Matter deferred to v0.8 |
| ADR-117 | pip wifi-densepose via PyO3 + maturin | Proposed | proposed-only | current PyPI v1.1.0 stale; tracking issue TBD |
| ADR-118 | BFLD — Beamforming Feedback Layer for Detection | Proposed | proposed-only | umbrella; sub-ADRs 119-123 |
| ADR-119 | BFLD Frame Format and Wire Protocol | Proposed | proposed-only | child of Proposed 118 |
| ADR-120 | BFLD Privacy Class and Hash Rotation | Proposed | proposed-only | child of Proposed 118 |
| ADR-121 | BFLD Identity Risk Scoring and Coherence Gate | Proposed | proposed-only | abandonment risk; data-gated (KIT BFId dataset) |
| ADR-122 | BFLD RuView Surface — HA/Matter/MQTT | Proposed | proposed-only | abandonment risk; depends on Soul Signature + cog-ha-matter |
| ADR-123 | BFLD Capture Path — Pi5/Nexmon, ESP32 feasibility | Proposed | proposed-only | hardware-gated (ESP32 cannot sniff CBFR) |
| ADR-124 | rvagent — MCP + ruvector npm lib (SENSE-BRIDGE) | Proposed | proposed-only | abandonment risk; not published; open questions |
| ADR-125 | RuView ↔ Apple Home native HAP bridge | Proposed | proposed-only | abandonment risk; hardware-gated (same-L2 pairing) |
| ADR-126 | HOMECORE — Rust+WASM+TS port of Home Assistant | Proposed | proposed-only | multi-quarter; series map cites missing 131/132 + mis-numbered 134 |
| ADR-127 | HOMECORE-CORE — state machine, registries, event bus | Proposed | proposed-only | future-dated Q3 2026 |
| ADR-128 | HOMECORE-PLUGINS — WASM integration plugin system | Proposed | proposed-only | future-dated; depends on 127 ABI freeze |
| ADR-129 | HOMECORE-AUTO — automation engine + template eval | Proposed | proposed-only | future-dated; broken cross-ref to ADR-134 |
| ADR-130 | HOMECORE-API — wire-compatible REST + WS | Proposed | proposed-only | future-dated; wire-compat needs HA companion-app suite |
| ADR-133 | HOMECORE-ASSIST — voice/intent + Ruflo bridge | Proposed | partial | missing tracking issue; P1 partial build, P2 deferred |
| ADR-134 | First-Class Channel Impulse Response (CIR) Support | Proposed | proposed-only | DUPLICATE IDENTITY (126/129 cite 134 as HOMECORE-MIGRATE); hardware-gated |
| ADR-135 | Empty-Room Baseline Calibration | Proposed | proposed-only | hardware-gated (COM9/COM12 + 802.15.4 sync) |
| ADR-136 | RuView Rust Streaming Engine — Architecture/Contracts | Proposed | partial | status-contradiction: §8 says Built (commit 11f89727f, 9 tests) |
| ADR-137 | Fusion Engine Quality Scoring | Proposed | partial | status-contradiction: Built (commit 4fa3847ac, 6 tests) |
| ADR-138 | WiFi-7 MLO LinkGroup + ArrayCoordinator gating | Proposed | partial | status-contradiction: Built (commit fc7674bde, 8 tests) |
| ADR-139 | WorldGraph — Environmental Digital Twin | Proposed | partial | status-contradiction: Built (commit 521a012d8, 7 tests) |
| ADR-140 | Semantic State Record + Ruflo Agent Bridge | Proposed | partial | status-contradiction: Built (commit 169a355bd, 4 tests); Rest kind not built |
| ADR-141 | BFLD Privacy Control Plane | Proposed | partial | header stale vs Implementation note (commit 7d88eb84c, 6 tests) |
| ADR-142 | Evolution Tracker + Temporal VoxelMap | Proposed | partial | header stale vs note (commit 1f8e180d6, 6 tests) |
| ADR-143 | RF SLAM v2 — Reflector Discovery + Anchor Learning | Proposed | partial | header stale (commit 2d4f3dea5); v2 dormant behind 7-day validation |
| ADR-144 | UWB Range-Constraint Fusion | Proposed | partial | header stale (commit b10bc2e9a); no UWB radio in fleet |
| ADR-145 | Ablation Evaluation Harness | Proposed | partial | referenced as existing by 149/150/151; F4/UWB variant HW-gated |
| ADR-146 | RF Encoder Multi-Task Heads + Uncertainty | Proposed | proposed-only | no Impl note (unlike 141-144); depends on tch/libtorch |
| ADR-169 | adam-mode — light theme toggle | Proposed | proposed-only | renumbered from ADR-147 (collision resolved); referenced by ADR-170 yoga |
| ADR-147 | Occupancy World Model (OccWorld/RoboOccWorld) | Accepted | partial | keeps 147 (collision resolved); self-revised from Cosmos; Phase B gated |
| ADR-168 | Benchmark Proof — OccWorld on RTX 5080 | (none) | unknown | renumbered from ADR-147 (collision resolved); MISSING STATUS; baseline-without-fine-tuning (random weights) |
| ADR-148 | Drone Swarm Control System | In Progress | partial | keeps 148 (collision resolved); re-routes 147 Cosmos item to 149 |
| ADR-170 | yoga-mode — pose detection/scoring demo | Proposed | proposed-only | renumbered from ADR-148 (collision resolved); no tracking issue |
| ADR-149 | AetherArena — Spatial-Intelligence Benchmark (HF) | Accepted | partial | keeps 149 (collision resolved); external repo out-of-tree; Wi-Pose dropped |
| ADR-171 | Drone Swarm Benchmarking Methodology | Accepted (peer-reviewed) | partial | renumbered from ADR-149 (collision resolved); critiques 148's own numbers |
| ADR-150 | RuView RF Foundation Encoder | Proposed | partial | status Proposed but cites measured 81.63% in-domain vs ~11.6% cross-subject |
| ADR-151 | Per-Room Calibration & Specialized Model Training | Accepted — Stages 1-5 impl | partial | HF-backbone distillation pending |
| ADR-152 | WiFi-Pose SOTA 2026 Intake | Proposed | partial | header stale; §2.1-2.3/2.6 impl, WiFlow-STD ~96% PCK; 1/25 claim REFUTED |
| ADR-153 | IEEE 802.11bf-2025 Forward-Compat Protocol Model | accepted | implemented | amends ADR-152 §2.4; OTA/silicon binding deferred |
| ADR-154 | Signal/DSP Beyond-SOTA Sweep — M0 | Proposed | partial | header likely stale; discloses dead CIR coherence gate; ~45 deferred |
| ADR-155 | NN/Training Beyond-SOTA Sweep — M1 | Proposed | partial | header likely stale; retracts synthetic-val/fake-gradient/self-cert proof |
| ADR-156 | RuVector/Cross-Viewpoint Fusion Sweep — M2 | Proposed | partial | header likely stale; one staged finding is numeric no-op |
| ADR-157 | Hardware/Sensing-Acquisition Sweep — M3 | Proposed | partial | header likely stale; headline negative result (layer already hardened) |
| ADR-158 | MAT/World-Model Cluster Sweep — Anti-AI-Slop | accepted | implemented | life-safety; fixes triage inflation; some paths DATA-GATED |
| ADR-159 | Cognitum Appliance Cluster Sweep — Anti-AI-Slop | accepted | implemented | person-count training_class1_accuracy = 0.343; description renamed |
| ADR-160 | Edge Skill Library (wasm-edge) — Honest Labeling | accepted | implemented | medical/affect/weapon NOT validated — relabelled |
| ADR-161 | HOMECORE Server — WS Auth Bypass, Reply-Theater | accepted | implemented | CRITICAL WS auth bypass fix; amends 130/129/128 |
| ADR-162 | HOMECORE Plugin Security + Bounded RunModes | accepted | implemented | security-critical; enforces ADR-161 deferrals |
| ADR-163 | Edge-Latency Measurement — CLAIMED→MEASURED | accepted | implemented | ESP32/Xtensa figure remains UNMEASURED (hardware-gated) |

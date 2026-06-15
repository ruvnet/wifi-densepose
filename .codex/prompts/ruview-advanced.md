# /ruview-advanced — advanced RuView capabilities

Drive RuView's research-grade / multi-node features. Topic: `$ARGUMENTS` (one of `multistatic`, `cross-viewpoint`, `tomography`, `field-model`, `intention`, `adversarial`, `security`; if empty, ask).

- **multistatic** (ADR-029) — treat every WiFi link in range (incl. neighbours' APs) as a bistatic radar pair, then fuse. `v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs` (attention-weighted fusion, geometric diversity), `phase_align.rs` (iterative LO phase-offset, circular mean), `multiband.rs`, `coherence.rs` / `coherence_gate.rs` (Z-score scoring; Accept / PredictOnly / Reject / Recalibrate).
- **cross-viewpoint** (ADR-016 viewpoint module) — combine 2+ nodes geometrically. `v2/crates/wifi-densepose-ruvector/src/viewpoint/`: `attention.rs` (CrossViewpointAttention, GeometricBias, softmax with `G_bias`), `geometry.rs` (GeometricDiversityIndex, Cramér–Rao bounds, Fisher Information), `coherence.rs` (phase-phasor coherence, hysteresis gate), `fusion.rs` (MultistaticArray aggregate root). Explore geometry first: `node scripts/mesh-graph-transformer.js`, `node scripts/deep-scan.js`.
- **tomography** — `ruvsense/tomography.rs` reconstructs a voxel occupancy grid via an ISTA L1 solver (sparse — most voxels empty); pair with cross-viewpoint geometry for through-wall volumetric imaging. RuVector solver crates back the 114→56 subcarrier sparse interpolation.
- **field-model** (ADR-030) — `ruvsense/field_model.rs` builds an SVD eigenstructure of the room, persists it (RVF, ideally on a Cognitum Seed); new frames are projected against it and the residual is the perturbation. Survives restarts; answers "what's different from the empty-room baseline?"
- **intention** — `ruvsense/intention.rs`, pre-movement lead signals 200–500 ms ahead.
- **adversarial** — `ruvsense/adversarial.rs`, rejects physically impossible signals + cross-checks multi-link consistency.
- **security** (ADR-032, multistatic mesh hardening) — using neighbour APs and pooling links across a mesh expands the attack surface. Mitigations: `adversarial.rs` + `coherence_gate.rs` quarantine (Reject / Recalibrate) + Ed25519 witness chain (ADR-028). Run a security review (`docs/security-audit-wasm-edge-vendor.md`); see `/ruview-verify`.

Also relevant: ADR-031 (sensing-first RF mode), ADR-081 (adaptive CSI mesh firmware kernel), ADR-083 (per-cluster π compute hop), ADR-095/096 (on-ESP32 temporal modeling, sparse GQA).

Validate: `cd v2 && cargo test -p wifi-densepose-signal --no-default-features && cargo test -p wifi-densepose-ruvector --no-default-features`, then `cd .. && python archive/v1/data/proof/verify.py`.

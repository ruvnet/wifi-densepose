# ADR Corpus Gap Analysis — Five-Lens Findings

Research notes backing ADR-164. Each lens output is reproduced verbatim. Census: 162 ADR entries across 156 distinct files (6 duplicate-number collisions verified on disk).

---

## Lens 1: status-distribution

Confirmed: ADR-168-benchmark-proof.md (was ADR-147-benchmark-proof.md) and ADR-134-csi-to-cir have no `Status` line in their headers (the 167-ddd hits are Rust code in the body, not a header; the ADR-167 appendix, was ADR-052-ddd, lacks a real Status header per its first lines). Findings are evidence-grounded. Final analysis below.

### ADR Corpus — Status & Implementation Distribution

Census: **162 ADR entries** across **156 distinct files** (6 duplicate-number collisions verified on disk).

#### Status tally

| Status bucket | Count |
|---|---|
| Accepted (incl. "Accepted — partial/in-progress/Phase 1" variants) | ~56 |
| Proposed (incl. "Proposed — conditional/research-only") | ~88 |
| Superseded | 1 (ADR-002) |
| Rejected | 1 (ADR-098) |
| Missing / no Status header | 3 (ADR-168-benchmark-proof [was 147], ADR-167-ddd appendix [was 052], ADR-134-CIR) |
| Mixed/dual status in one ADR | 3 (ADR-115, ADR-149-AetherArena vs swarm, ADR-133) |

#### impl_state tally

| impl_state | Count |
|---|---|
| implemented | ~36 |
| partial | ~50 |
| proposed-only | ~64 |
| stale-or-contradicted | 3 (ADR-029, 030, 031) |
| unknown | 5 (ADR-034, 044, 167-ddd [was 052], 168-proof [was 147], …) |
| superseded | 1 (ADR-002) |

**Headline:** ~114 of 162 ADRs (70%) are decisions that never fully landed (proposed-only + partial + stale + unknown). The dominant failure mode is **stale Status headers** — Accepted/implemented work still labeled "Proposed."

#### SEVERITY: CRITICAL — Status header missing or structurally absent (cannot triage)

- **ADR-168-benchmark-proof.md** (renumbered from ADR-147 to resolve the 147 collision) — *No `Status` header at all* (grep confirmed). Not a true ADR; it's a benchmark artifact (OccWorld @ ~213ms on RTX 5080, random weights) that was misfiled under the ADR-147 number. **Action: relocate to `docs/proof/` or `benchmarks/`, remove ADR number.**
- **ADR-134-csi-to-cir-time-domain-multipath.md** — *No `Status` header* (grep confirmed) in the header region. Body says Proposed but the field is not in canonical position. Compounded by a **number collision**: ADR-126/129 reference "ADR-134" as HOMECORE-MIGRATE, but the on-disk file is CIR. **Action: add canonical `## Status` line; resolve the 134 identity split.**
- **ADR-167-ddd-bounded-contexts.md** (renumbered from ADR-052 to resolve the 052 collision; still an appendix to parent ADR-052) — Appendix doc with no Status/Date header (grep found only Rust code, no header field). **Action: mark explicitly "Appendix to ADR-052 (no independent status)".**

#### SEVERITY: CRITICAL — Duplicate ADR numbers (6 collisions, all verified on disk)

| Number | Colliding files | Action | Resolution |
|---|---|---|---|
| **147** | adam-mode-light-theme · nvidia-cosmos/OccWorld · benchmark-proof | Renumber 2 of 3 | **RESOLVED** — 147 keeps nvidia-cosmos/OccWorld; benchmark-proof → **ADR-168**, adam-mode → **ADR-169** |
| **148** | drone-swarm-control-system · yoga-mode-pose-system | Renumber 1 | **RESOLVED** — 148 keeps drone-swarm; yoga-mode → **ADR-170** |
| **149** | AetherArena-leaderboard · swarm-benchmarking | Renumber 1 | **RESOLVED** — 149 keeps AetherArena; swarm-benchmarking → **ADR-171** |
| **050** | provisioning-tool-enhancements · quality-engineering-security-hardening | Renumber 1 | **RESOLVED** — 050 keeps provisioning (5 refs vs 1); quality-engineering → **ADR-166** |
| **052** | tauri-desktop-frontend · ddd-bounded-contexts (appendix) | Demote appendix | **RESOLVED** — 052 keeps tauri; ddd appendix renumbered → **ADR-167** (still linked to parent 052) |
| **134** | csi-to-cir (on disk) · HOMECORE-MIGRATE (referenced, no file) | Resolve identity | Identity split (not a filename collision); resolved separately via G3 → ADR-165 |

These broke the ADR index and `/adr` tooling — two ADRs answering to one number is a corpus-integrity defect, not cosmetics. The five filename collisions are now resolved (six displaced files renumbered 166–171); see ADR-164 Gap Register G1.

#### SEVERITY: HIGH — Status header stale vs. shipped reality (Proposed header on landed code)

These are the most dangerous: an auditor reading the header concludes "not built" when code + tests exist. Ranked by blast radius:

1. **ADR-136 → ADR-145** (streaming-engine series, 10 ADRs) — every header says `Proposed` but each `§ Implementation Status` reports **"Built" with pinned commits + passing tests** (136: 11f89727f; 137: 4fa3847ac; 138: fc7674bde; 139: 521a012d8; 140: 169a355bd; 141: 7d88eb84c; 142: 1f8e180d6; 143: 2d4f3dea5; 144: b10bc2e9a; 145 referenced as landed by 149/150/151). **Bulk action: flip headers to "Accepted — partial (integration glue pending)".**
2. **ADR-029 / 030 / 031** (RuvSense/field-model/cross-viewpoint) — `Proposed` but repo has `signal/src/ruvsense/` (16 modules) and `ruvector/src/viewpoint/`, and **Accepted ADR-032 hardens them** — an Accepted ADR depending on Proposed parents (status-graph inversion).
3. **ADR-095 / 096** (rvCSI) — `Proposed` but ADR-097 confirms built, extracted to own repo, published 0.3.1 to crates.io/npm.
4. **ADR-152** — `Proposed` but CLAUDE.md + recent commits report §2.1–2.3/2.6 implemented, WiFlow-STD MEASURED-EQUIVALENT ~96% PCK.
5. **ADR-154/155/156/157** (beyond-SOTA sweeps) — `Proposed` but each describes fixes **already landed with revert-verified regression tests**.
6. **ADR-024 (AETHER) / 027 (MERIDIAN) / 072 (WiFlow)** — `Proposed` but CLAUDE.md lists them Accepted and code references them as implemented.
7. **ADR-017** — header Accepted but CLAUDE.md still calls it "Proposed" (inverse drift).
8. **ADR-018** — `Proposed` but ADR-012 cites it as the working firmware/aggregator impl.

#### SEVERITY: HIGH — Status ahead of its dependencies (Accepted depends on Proposed)

- **ADR-032** Accepted → depends on Proposed 029/030/031.
- **ADR-053** Accepted → depends on Proposed ADR-052.
- **ADR-048** Accepted → depends on Proposed ADR-045.
- **ADR-077** Accepted → depends on Proposed ADR-075/076.

#### SEVERITY: MEDIUM — Proposed-but-looks-abandoned (decisions that will likely never land)

Cluster heads where the whole chain is Proposed with zero implementation evidence:
- **ADR-003/007/008/009/010** — RuVector child ADRs orphaned after parent ADR-002 was superseded by 016/017.
- **ADR-105/106/107/108** — entire federation chain, none implemented.
- **ADR-118/119/120/121/122/123** — entire BFLD chain, all ACs unchecked, tracking issues TBD.
- **ADR-124/125/126/127/128/129/130/133** — HOMECORE/bridge chain, multi-quarter future-dated, all TBD.
- **ADR-033** (remote-viewing), **ADR-042** (CHCI, superseded-in-intent by 153), **ADR-046** (Android TV), **ADR-049** (Python v1 legacy), **ADR-067** (RuVector v2.0.5 upgrade not adopted), **ADR-074** (SNN), **ADR-085** (RaBitQ expansion), **ADR-011** (Proposed-URGENT despite proof pipeline live).

#### SEVERITY: MEDIUM — Accepted but no implementation evidence (unverified "done")

- **ADR-034** (FieldView mobile app) — Accepted, no crate/dir in CLAUDE.md.
- **ADR-044** (wifi-densepose-geo) — bare Accepted, no Date/Deciders, crate not in CLAUDE.md table.

#### Ranked actionable backlog (do in this order)

1. **Resolve 6 duplicate ADR numbers + 3 missing-header files** (CRITICAL — breaks the index/tooling). **Number collisions RESOLVED:** renumbered 147×2 (benchmark-proof→168, adam-mode→169), 148 (yoga→170), 149 (swarm-benchmarking→171), 050 (quality-engineering→166), 052 ddd appendix→167. Remaining: resolve the 134 identity split (done via G3→165); add Status headers to 168-proof, 134, 167-ddd (owner-gated).
2. **Bulk-flip the 10 streaming-engine headers (ADR-136–145)** from Proposed → "Accepted — partial" — they have commit-pinned, test-backed Implementation Status notes. Highest ROI: one batch fixes the largest stale-status cluster.
3. **Fix the status-graph inversions** (032/053/048/077 depend on Proposed parents; promote parents 029/030/031/045/052/075/076 to match their built reality, or downgrade the dependents).
4. **Reconcile CLAUDE.md vs ADR headers** for 017, 024, 027, 072, 152 (doc says one thing, header another).
5. **Triage the abandoned chains** (003/007–010, 105–108, 118–123, HOMECORE 124–133) — either mark `Rejected`/`Deferred` or open tracking issues; leaving ~50 Proposed-forever ADRs pollutes every gap analysis.
6. **Verify or downgrade unsubstantiated Accepteds** (034, 044).

**Bottom line:** the corpus's #1 defect is not abandoned work — it's **bookkeeping drift**. ~24 ADRs are mislabeled Proposed while code+tests exist, 6 numbers collide, and 3 files lack Status headers entirely. The status field cannot currently be trusted as a source of truth for what is built.

---

## Lens 2: supersession-chains

I have all evidence needed. ADR-016 doesn't mention ADR-002 at all (only 014/015); ADR-017 discusses ADR-002 as a "correction" but never uses the word "supersede." This confirms the broken reciprocity. Final analysis below.

### Supersession & Lifecycle Integrity — ADR Corpus

Total declared supersession edges in census: **3** (ADR-002→016, ADR-002→017, ADR-052-tauri→054). All three were file-verified. Findings below; severity graded **CRITICAL / HIGH / MEDIUM / LOW**.

#### Reconstructed chains
- **ADR-002** (RuVector RVF Integration Strategy) → superseded-by **ADR-016 + ADR-017** (dual realization). Self-declared `supersedes` on 016/017.
- **ADR-052-tauri** (Tauri Desktop Frontend) → superseded-by **ADR-054** (declared in 052's `superseded_by`).
- No other formal `supersedes`/`superseded_by` links exist. No cycles detected (the only multi-node graph, ADR-002→{016,017}, is a DAG; ADR-052→054 is a single edge). **No cycles — clean.**

#### Broken / asymmetric links

**1. ADR-002 → ADR-016 / ADR-017: one-directional, never reciprocated. (HIGH)**
ADR-002 header declares "Superseded by [ADR-016] and [ADR-017]" (`docs/adr/ADR-002-ruvector-rvf-integration-strategy.md:4`). But neither successor claims it:
- **ADR-016** (`ADR-016-ruvector-integration.md`) never mentions ADR-002 anywhere — its `## References` lists only ADR-014/015. It does not assert supersession; the census `supersedes:["ADR-002"]` for ADR-016 is **unsupported by the file**.
- **ADR-017** (`ADR-017-ruvector-signal-mat-integration.md`) discusses ADR-002 only as a `## Correction to ADR-002 Dependency Strategy` (line 532) — corrects "fictional crate names" — but **never uses the word "supersede."** Census `supersedes:["ADR-002"]` is again file-unsupported.
- Net: ADR-002 points up at two ADRs that don't point back. The supersession is asserted by the superseded ADR alone — backwards from convention, and unverifiable from the successors.

**2. ADR-002 partial-supersession leaves 5 orphaned children stranded. (HIGH)**
ADR-002 is an umbrella whose children ADR-003, 007, 008, 009, 010 are still `Proposed`. ADR-016/017 only realize the *training/signal/MAT* integration points (mincut, attention, solver, etc.). The RVF-container (003), PQ-crypto (007), Raft consensus (008), WASM edge runtime (009), and witness-chains (010) decisions are **neither implemented nor formally superseded** — ADR-017:555 explicitly acknowledges 008/009 "described in ADR-002" are not carried forward. Marking the parent fully "Superseded" silently buries 5 live-but-abandoned child decisions. ADR-010's role is additionally filled de facto by ADR-028's witness-bundle without any supersession link.

**3. ADR-052-tauri → ADR-054: declared by predecessor, not acknowledged by successor. (HIGH)**
Census records ADR-052-tauri `superseded_by:["ADR-054"]`. **ADR-054 (`ADR-054-desktop-full-implementation.md`) contains zero references to ADR-052** (grep for `ADR-052|replac|supersed` returns nothing). ADR-054 is titled "RuView Desktop **Full Implementation**" and is "in progress" — functionally it's the implementation plan *for* 052, not a replacement. The supersession edge is unconfirmed by the successor and arguably mis-modeled (an in-progress impl doesn't supersede its own design ADR).

#### Orphaned superseded ADRs still marked accepted/active
**4. No classic orphan (superseded ADR still `Accepted`), but two soft variants: (MEDIUM)**
- **ADR-052-tauri** is `Proposed` *and* `superseded_by ADR-054`, yet downstream ADR-053/055/056 (all `Accepted`) build on it and treat the desktop app as shipped (v0.3.0). A Proposed-and-superseded ADR anchoring three Accepted descendants is a lifecycle inconsistency: the live decision-of-record is ambiguous (052? 054? 056?).
- **ADR-002** is correctly `Superseded`, so not an orphan — but ADR-038's roadmap census still counts it among 37 active ADRs, so stale references persist downstream.

#### De-facto supersessions never recorded (missing links) — MEDIUM
These pairs behave as supersession in the corpus but carry **no** `supersedes`/`superseded_by` fields, so the chain graph understates reality:
- **ADR-098 ⇄ ADR-099** (`MEDIUM`): ADR-098 **Rejected** midstream; ADR-099 revives its carve-outs ("Adopt midstream…"). A rejection partially reversed by a later Proposed ADR — neither links the other via supersession fields (only prose tension).
- **ADR-063 → ADR-064**, **ADR-035 → ADR-023/036**, **ADR-042 → ADR-153**, **ADR-050-provisioning → ADR-060**, **ADR-117 retracts PyPI v1.1.0**, PQ chain **ADR-100 → ADR-109 / ADR-107→108**: each is "superseded in scope/intent" per flags but has empty supersession fields. Not broken links (none falsely asserted), but **missing** ones — the lifecycle graph is incomplete.

#### Cycles
**None.** ADR-002→{016,017} and ADR-052→054 are acyclic; no ADR is reachable from its own successor.

#### Severity summary
| # | Issue | Severity |
|---|-------|----------|
| 1 | ADR-002 supersession not reciprocated by ADR-016/017 (016 never mentions 002; 017 only "corrects") | HIGH |
| 2 | ADR-002 children 003/007/008/009/010 stranded under a "fully superseded" parent | HIGH |
| 3 | ADR-052→054 edge unacknowledged by ADR-054 (no reference); likely mis-modeled (impl, not replacement) | HIGH |
| 4 | ADR-052-tauri Proposed+superseded yet anchors Accepted 053/055/056 | MEDIUM |
| 5 | De-facto supersessions (098/099, 063/064, 042/153, 050/060, 035/023, 100/109) unrecorded | MEDIUM |
| — | Cycles | none (clean) |

Highest-priority fix: add reciprocal `supersedes: ADR-002` language to ADR-016/017 (or downgrade the claim), and split ADR-002's status to "partially superseded" so children 003/007/008/009/010 aren't silently closed.

---

## Lens 3: contradictions

All findings confirmed against source. Here is the contradictions lens.

### Contradictions & Retractions Lens — ADR Corpus

The sweep (ADR-154–163) is itself a structured retraction layer: each "Beyond-SOTA / anti-AI-slop" ADR exists *because* an earlier accepted claim was found false. Findings graded **CRITICAL** (life-safety, security, or a published accuracy number that was meaningless) / **HIGH** (a capability/number retracted or directly contradicted) / **MEDIUM** (status or scope conflict) / **LOW** (cosmetic/doc drift).

#### A. Accepted/published claims later RETRACTED or REFUTED

**[CRITICAL] ADR-155 retracts every prior NN accuracy/TTA/proof claim.** ADR-155 §2.2 discloses `bin/train.rs` validated a *real* MM-Fi training run against a **synthetic** val set, and windows leak at stride-1 (~99% overlap) — *"any PCK it printed was meaningless on two counts."* §2.3: `rapid_adapt.rs` `contrastive_step`/`entropy_step` wrote a **fake gradient** (`grad += v * 0.01`) unrelated to the objective — every "TTA improves the metric" result was unsupported. §2.4: the deterministic proof **self-certified** (`generate_expected_hash` blessed whatever the pipeline emitted; PASS counted any loss decrease incl. 1e-9 float noise; missing hash defaulted to PASS). This retroactively voids accuracy claims made anywhere in the corpus that depended on the training/proof path prior to commit landing ADR-155.

**[CRITICAL] ADR-154 retracts the ADR-134 CIR coherence gate as live.** ADR-152/CLAUDE.md present CIR (ADR-134) as a contributing signal in the multistatic coherence gate. ADR-154 §2 proves it was **DEAD in production for every canonical frame**: the HT20 CIR estimator returns `SubcarrierMismatch` on all 56-tone canonical frames (`cir_gate_ht20_is_dead_on_canonical56`: 0 Ok / 8 mismatch), so `coherence = 0.7·freq + 0.3·dominant_tap_ratio` silently degraded to freq-only (`cir_gate_dead_ht20_equals_gate_off`, |Δ|<1e-9). Any ADR claiming CIR-enhanced coherence/ToF before this fix overstated reality.

**[CRITICAL] ADR-079 internal accuracy contradiction (self-flagged in census, confirmed).** Context states proxy PCK@20 = **2.5%** (lines 11, 25) and "10-20x improvement: 2.5% → 35%+". The baseline table (line 497) reports proxy PCK@20 = **35.3%** — i.e. the *baseline already equals the stated target* — while per-joint upper body (nose/shoulders/wrists) is **0%** (line 503). The headline 10–20x improvement number is therefore self-refuting against its own baseline table. CLAUDE.local.md adds the local-Windows attempt (#640) measured **0% PCK**. An Accepted ADR with three mutually inconsistent values for its own central metric.

**[HIGH] ADR-152 self-refutes one verified research claim (F4).** ADR-152 grades 25 claims 3-vote; §F4 records the "Espressif `esp_wifi_sensing` is **drop-in compatible with RuView nodes**" claim **REFUTED 0-3** (WiFi-6 parts use a different CSI acquisition config struct). ADR-110 ("ESP32-C6 Wi-Fi 6 CSI") and the CLAUDE.md hardware table treat C6/Wi-Fi-6 CSI as a smooth extension; ADR-152 also notes HE-CSI needs ESP-IDF ≥5.5 (v5.4 silently downconverts to HT). The "WiFlow-STD MEASURED-EQUIVALENT ~96% PCK@20" line in CLAUDE.md is *not* yet supported: §2.2/§F1 mark external pose numbers (incl. the 97.25% WiFlow-STD figure) **CLAIMED**, and §F1 explicitly forbids citing 97.25% as comparable until measurements (a)–(c) are run. CLAUDE.md asserting "MEASURED-EQUIVALENT" contradicts the ADR's own gating.

**[HIGH] ADR-150 retracts the implied cross-subject capability of the encoder line.** AETHER/MERIDIAN ADRs (024/027) and the foundation-encoder framing imply subject-invariant embeddings work. ADR-150 measures **81.63% in-domain vs ~11.6% leakage-free cross-subject** torso-PCK, and reports DANN **failed** (27.26%→27.54%, empirically ~0 gain) and bigger capacity *hurt* (transformer 24.8% < conv 27.3%). §1.1/§4 conclude the cross-subject acceptance gate "is **unlikely to be met without new multi-subject** data" — a direct retraction of the "more capacity / adversarial alignment solves cross-environment loss" premise underlying ADR-027.

**[HIGH] ADR-159 refutes the "never identified anyone" accusation but simultaneously retracts cog-person-count's marketing.** ADR-159 ships real SHA-pinned Candle models, but discloses person-count `training_class1_accuracy = 0.343` (presence-only, classes 0/1), and **renames** the Cargo description from "learned multi-person counter" → "presence detector + (data-gated) person count," clamping/`low_confidence`-flagging multi-occupant counts. This retracts ADR-103's "learned multi-person counter (SOTA WiFi CSI counting)" claim and ADR-104's count tool, which depended on it.

**[HIGH] ADR-161 retracts HOMECORE server security + functionality claims.** ADR-130 (HOMECORE-API, wire-compatible, Ed25519-JWT) implied a secured server. ADR-161 fixes a **CRITICAL WebSocket auth bypass** (any non-empty token accepted), "reply-theater" (WS responses computed then discarded), and documented-but-no-op automation — then ADR-162 enforces the ADR-161 deferrals (plugin Ed25519 sig verification, capability isolation, bounded RunModes that were "parsed-but-unenforced/unbounded-parallel"), retracting ADR-128/129's implied plugin-signing and automation guarantees.

**[MEDIUM] ADR-163 converts CLAIMED latency budgets to MEASURED — retracting prior budget citations.** ADR-160/159 cited wasm-edge/cog latency *budgets*. ADR-163 adds host benches and explicitly states the **ESP32/Xtensa-on-hardware figure remains UNMEASURED** — so any doc citing the device latency budget as achieved is unsupported.

**[MEDIUM] ADR-098 → ADR-099 partial reversal.** ADR-098 **Rejected** midstream as a system component; ADR-099 (Proposed) **adopts** midstream's temporal-compare (DTW) + temporal-attractor-studio as a parallel tap. Framed as "complementary," but it revives the exact carve-outs ADR-098 declined to integrate — a live decision conflict pending resolution.

**[MEDIUM] ADR-147 (OccWorld) self-retracts Cosmos.** The accepted ADR-147 title/decision was revised from "NVIDIA Cosmos WFM Integration" to OccWorld after a hardware finding (Cosmos needs 32.5 GB VRAM); Cosmos is retracted as primary. The companion ADR-168-benchmark-proof (renumbered from ADR-147) reports 213 ms/inference on **random weights, no checkpoint** — a baseline-without-fine-tuning number that must not be cited as a quality/target metric.

#### B. Pairs making CONFLICTING decisions on the same topic

**[HIGH] RVF-WASM edge runtime — ADR-009 vs shipped `wifi-densepose-wasm`.** ADR-009 (Proposed) decides to **replace** the existing wifi-densepose-wasm approach with an `.rvf.edge` container runtime. The crate it proposes to replace is shipped and in the CLAUDE.md crate table (and is the dependency base for ADR-058/059 browser pose). ADR-009 is an unrealized decision directly contradicting shipped architecture.

**[HIGH] Witness/audit mechanism — ADR-010 vs ADR-028.** ADR-010 (Proposed) decides RuVector witness *chains* as "the primary tamper-evident audit mechanism." ADR-028 (Accepted, implemented) established a different **witness-bundle** mechanism (verify.py / SHA-256 / VERIFY.sh) that fills this role. Two competing "primary audit" decisions; ADR-010 is stranded.

**[HIGH] Multistatic "sensing-first RF mode" — ADR-029 vs ADR-031 near-duplicate scope.** Both decide a "sensing-first RF mode for multistatic fidelity": ADR-029 (RuvSense, signal/src/ruvsense/) and ADR-031 (RuView cross-viewpoint fusion, ruvector/src/viewpoint/). Overlapping problem statements (occlusion/depth/multi-person via multistatic attention+geometry), separate crate homes, both still nominally "Proposed" while both are implemented. Unreconciled dual ownership of the multistatic-fusion decision.

**[MEDIUM] Person-counting decision conflict — ADR-037 vs ADR-075 vs ADR-103.** Three different decisions to replace the same fixed-threshold counter: ADR-037 (4-phase neural decomposition), ADR-075 (spectral min-cut over subcarrier-correlation graph, fixes #348), ADR-103 (learned Cog `cog-person-count`). ADR-075's bug (#348) overlaps ADR-069's driver. None supersedes the others; ADR-159 then guts ADR-103's claim (above).

**[MEDIUM] PQ-crypto signing — ADR-007 vs ADR-109.** ADR-007 (Proposed) decides Ed25519 + ML-DSA-65 hybrid for sensing-data signing; ADR-109 (Proposed) decides Ed25519 + **Dilithium-3** hybrid for cog signing (Dilithium = ML-DSA family but a different parameter pick/scope). Two PQ-signature decisions over adjacent surfaces with non-identical algorithm choices, neither reconciled.

**[MEDIUM] Federation key-exchange self-supersession — ADR-107 vs ADR-108.** ADR-107 adopts classical Diffie-Hellman in secure-aggregation Layer 4; ADR-108 replaces it with Kyber-768 because the DH choice is "quantum-vulnerable." ADR-108 supersedes a core element of ADR-107 while ADR-107 is still only Proposed — a decision corrected before it was ever accepted.

**[MEDIUM] Provisioning path forked three ways — ADR-050(prov) vs ADR-060 vs ADR-052/054.** ADR-050 (provisioning-tool-enhancements, Proposed) scopes channel+MAC-filter flags; ADR-060 (Accepted) actually implements them; ADR-052/054 move provisioning into a Rust-native Tauri desktop path. Three live decisions for "how RuView provisions nodes," with ADR-060 partially fulfilling ADR-050 without superseding it.

#### C. Status-graph contradictions (Accepted depending on / contradicting Proposed)

**[MEDIUM] Accepted ADRs hardening/depending on Proposed ones.** ADR-032 (Accepted, security hardening) hardens ADR-029/030/031 which remain "Proposed" — an accepted decision presupposing un-accepted ones exist. Same pattern: ADR-048 (Accepted) depends on ADR-045 (Proposed); ADR-053 (Accepted) depends on ADR-052 (Proposed); ADR-077 (Accepted) depends on ADR-075/076 (Proposed); ADR-104 (Accepted) depends on ADR-103 (Proposed). These are status contradictions, not capability retractions, but they signal the same "header lags reality" hygiene problem the sweep is correcting.

**[LOW] Header-stale-vs-implementation (pervasive).** ADR-029/030/031, 072, 095/096, 136–145, 150, 152, 154–157 all carry `Status: Proposed` while their own appended Implementation-Status notes (or downstream ADRs / CLAUDE.md) report them built+tested with commits. ADR-024/027 say Proposed; CLAUDE.md lists them Accepted; pose_tracker.rs already uses AETHER re-ID. Cosmetic but corpus-wide; it is the mechanism by which retracted/overstated claims survive (a green "built" note under a "Proposed" header is exactly where ADR-155's self-certifying proof hid).

#### Cited source files (absolute)
- C:\Users\ruv\Projects\wifi-densepose\docs\adr\ADR-079-camera-ground-truth-training.md (lines 11/25/497/503 — 2.5% vs 35.3% vs 0%)
- C:\Users\ruv\Projects\wifi-densepose\docs\adr\ADR-150-rf-foundation-encoder.md (81.63% vs 11.6%; DANN ~0)
- C:\Users\ruv\Projects\wifi-densepose\docs\adr\ADR-152-wifi-pose-sota-2026-intake.md (F4 REFUTED 0-3; 97.25% CLAIMED-not-MEASURED)
- C:\Users\ruv\Projects\wifi-densepose\docs\adr\ADR-154-signal-dsp-beyond-sota.md (§2 dead CIR gate)
- C:\Users\ruv\Projects\wifi-densepose\docs\adr\ADR-155-nn-training-beyond-sota.md (§2.2–2.4 synthetic-val / fake gradient / self-certifying proof)
- C:\Users\ruv\Projects\wifi-densepose\docs\adr\ADR-159-cognitum-appliance-beyond-sota.md (person-count 0.343; description renamed)

**Top-severity summary:** the four CRITICAL items (ADR-155 fake-gradient+synthetic-val+self-certifying proof; ADR-154 dead CIR gate; ADR-079 self-inconsistent PCK; ADR-161 WS auth bypass) are the corpus's load-bearing "AI-slop" admissions — each is an *accepted-or-shipped* surface whose stated accuracy/security/function was provably false until the sweep landed. Every accuracy number predating ADR-155 in any other ADR should be treated as CLAIMED, not MEASURED, until re-derived through the post-155 leak-free split.

---

## Lens 4: coverage-gaps

Confirmed — ADR-094 governs the pointcloud *viewer deployment* (proposed-only), not the crate's sensing-data-production contract. I have all evidence needed.

### Coverage Gaps — Crates/Capabilities vs Governing ADRs

Severity: **CRITICAL** (shipped code with no/broken governing ADR), **HIGH** (architect would expect an ADR, none exists), **MEDIUM** (governed only by a remediation/deploy ADR, no creation/architecture ADR), **LOW** (minor).

#### A. Shipped crates whose cited ADR does not exist (CRITICAL)

Two crates are built and in-tree but reference ADR numbers that point to *different* on-disk ADRs or to files that never existed (confirmed: no `ADR-131*.md` or `ADR-132*.md` exists; `ADR-134` on disk is CIR, not HOMECORE-MIGRATE):

- **`v2/crates/homecore-recorder`** — Cargo.toml header: *"SQLite state history + semantic search (ADR-132)"*. **No ADR-132 exists.** The HOMECORE series map (ADR-126 §4) lists ADR-132 HOMECORE-RECORDER as planned, but it was never written. A shipped persistence/history crate has zero governing decision record. **CRITICAL** — this is the recorder, the durable-state surface, ungoverned.
- **`v2/crates/homecore-migrate`** — Cargo.toml header: *"Implements ADR-134 (HOMECORE-MIGRATE)"*. **On-disk ADR-134 is "First-Class CIR Support"** (census + glob confirm). ADR-129/126 also cite ADR-134 as HOMECORE-MIGRATE. The crate implements a migration tool from Python HA reading `.storage/*.json` — a data-integrity-sensitive importer — governed by a phantom ADR identity. **CRITICAL** (compounds the documented ADR-134 duplicate-number collision).

These are not stale-header issues like the ADR-136..146 cluster (where the ADR exists and is just marked Proposed); here the cited governing ADR **is absent or is a different decision**.

#### B. Shipped crates with NO governing ADR at all (HIGH)

- **`v2/crates/wifi-densepose-engine`** — *"streaming-engine integration layer — composes the ADR-135..146 building blocks into one trust-traceable pipeline cycle."* It composes ~12 ADRs' outputs into the live pipeline-cycle aggregate, but **no ADR governs the composition/orchestration contract itself** (ordering, back-pressure, the "one pipeline cycle" boundary). ADR-136 defines frame contracts/stages but not the integrator crate. An architect would expect an ADR for the seam that wires 135–146 onto the live 20 Hz path — exactly the "integration glue not yet on live path" caveat repeated across ADR-136..146. **HIGH.**

#### C. Capabilities governed only by a remediation/deploy ADR — no creation/architecture ADR (MEDIUM)

- **`v2/crates/wifi-densepose-wasm-edge` (~70 edge skills)** — The only ADRs touching it are **ADR-160** (honest *relabeling*/soundness cleanup) and **ADR-163** (latency *measurement*). Both are anti-slop remediation ADRs that presuppose ~70 skills already shipped. There is **no creation/architecture ADR** defining the skill taxonomy, ABI, event-ID allocation, or budget tiers for this crate. (Contrast ADR-041, which *does* catalog the 60-module registry — but for the ESP32/WASM3 on-device path of ADR-040, a different artifact.) A whole ~70-module crate's design rationale lives nowhere. **MEDIUM-HIGH.**
- **`v2/crates/wifi-densepose-occworld-candle`** — *"OccWorld TransVQVAE inference ported to Candle (Rust-native, no Python IPC)."* ADR-147 (OccWorld) decided a **Python-subprocess** thin client and explicitly deferred a Rust backend swap to "Phase B / RoboOccWorld." A native Candle reimplementation is a material architecture change (new dep surface, no IPC, weight-loading path) that **no ADR records the decision to build now**. **MEDIUM.**
- **`v2/crates/wifi-densepose-pointcloud`** — ADR-094 governs only the *GitHub-Pages viewer deployment* (Proposed). The crate as a **point-cloud data-production/format contract** (what it emits, schema, real-data-stream toggle wiring) has no governing decision beyond the demo-deploy doc. **MEDIUM.**
- **`v2/crates/homecore-hap`** — header cites ADR-125 P1 scaffold; ADR-125 (Apple Home HAP bridge) exists and covers it. **Governed — no gap.** (Listed to scope out the false positive.)
- **`v2/crates/wifi-densepose-geo`** — governed by ADR-044 (geospatial). Governed, but ADR-044 is a bare "Accepted" with no implementation evidence and is cross-referenced incorrectly by ADR-052 (cites ADR-044 for provisioning). **LOW** (governed but the ADR itself is thin).

#### D. Decision areas an architect would expect an ADR for, but none exists (HIGH)

1. **Persistence/storage strategy for HOMECORE state history** — `homecore-recorder` ships SQLite with an "HA-compat schema," but no ADR decides SQLite-vs-alternatives, retention, or the semantic-search index. Recorder is the durability backbone; an unrecorded storage choice is a classic missing-ADR. **HIGH** (ties to gap A).
2. **Python-HA → HOMECORE migration/import contract** — `homecore-migrate` reads foreign `.storage` JSON (untrusted input, schema-drift risk) with no governing ADR (the cited one is CIR). Migration correctness and trust boundary are exactly what an ADR should pin. **HIGH** (ties to gap A).
3. **The streaming-engine *integrator* contract** (`wifi-densepose-engine`) — see B. **HIGH.**
4. **Cross-crate workspace dependency/publishing ADR** — CLAUDE.md lists a hand-maintained 12-step publishing order and a 15-crate table, but the workspace now has **38 crates** (glob count) including ungoverned ones (engine, worldmodel, worldgraph, occworld-candle, geo, wasm-edge, homecore-*, cog-*, ruview-swarm, pointcloud, nvsim-server, desktop). No ADR governs crate-graph topology / publish boundaries at this scale — the publishing list in CLAUDE.md is already stale against reality. **MEDIUM-HIGH.**
5. **No ADR ties the streaming-engine (`engine`) to the cog/appliance deploy surface** — ADR-101/102/159 govern cogs; ADR-136..146 govern the engine; nothing decides how the trust-traceable engine output becomes a deployed cog. The seam between the two largest subsystems is ungoverned. **MEDIUM.**

#### E. Scoped-out false positives (verified governed)

- `wifi-densepose-worldmodel` → ADR-147 (OccWorld bridge). Governed.
- `wifi-densepose-worldgraph` → ADR-139. Governed.
- `cog-ha-matter` → ADR-116; `cog-person-count` → ADR-103; `cog-pose-estimation` → ADR-101. Governed.
- `ruview-swarm` → ADR-148. `nvsim`/`nvsim-server` → ADR-089/092. `wifi-densepose-bfld` → ADR-118–123/141. `wifi-densepose-calibration` → ADR-151. All governed.
- `wifi-densepose-desktop` → ADR-052/054 (contested status, but an ADR exists). Not a coverage gap (it's a status-drift issue, out of this lens's scope).

#### Top-priority remediation (concrete)
1. Write **ADR-132 HOMECORE-RECORDER** (or renumber) so the shipped `homecore-recorder` SQLite/history crate has a real governing decision — **CRITICAL**.
2. Resolve the **ADR-134 identity collision** and create a real **HOMECORE-MIGRATE** ADR for `homecore-migrate` (untrusted-import trust boundary) — **CRITICAL**.
3. Write a **streaming-engine integrator ADR** for `wifi-densepose-engine` (the 135–146 composition contract / live-path seam) — **HIGH**.
4. Write a **creation/architecture ADR for `wifi-densepose-wasm-edge`** (taxonomy/ABI/event-IDs) — currently only remediation ADRs (160/163) exist — **MEDIUM-HIGH**.
5. Add an ADR for the **`occworld-candle` Rust-native backend swap** (ADR-147 only blessed the Python-subprocess path) — **MEDIUM**.

Evidence files: `v2/crates/homecore-recorder/Cargo.toml`, `v2/crates/homecore-migrate/Cargo.toml`, `v2/crates/wifi-densepose-engine/Cargo.toml`, `v2/crates/wifi-densepose-occworld-candle/Cargo.toml`, `docs/adr/ADR-134-csi-to-cir-time-domain-multipath.md` (CIR, not MIGRATE), `docs/adr/ADR-160-edge-skill-library-honest-labeling.md`, `docs/adr/ADR-094-pointcloud-github-pages-deployment.md`; absence confirmed via glob: no `ADR-131*`, no `ADR-132*`.

---

## Lens 5: data-hardware-gated

Evidence confirmed. I have enough to write the section.

### Open / Gated Decisions — The Real Backlog (data/hardware-gated, in-progress, or unresolved)

Severity = how load-bearing the unresolved decision is to a shipped claim. Sweep coverage = whether ADR-154–163 touched it.

#### CRITICAL — life-safety or shipped-claim surface, still gated

**ADR-079 — Camera Ground-Truth Training Pipeline.** *Accepted, but core decision unvalidated.* P7–P9 (real paired-data collection, training, cross-room LoRA) are **Pending** (file lines 476–478). Blocker: a real synchronized camera+ESP32 paired-capture session and GPU training run — neither done. The ADR's own baseline table is self-contradictory: text says proxy PCK@20=2.5% (lines 11, 25) yet line 497 reports 35.3% (the *target*) with line 503 confessing **upper-body joints at 0%** — the proxy has no real spatial signal. CLAUDE.local.md records the local-Windows attempt (#640) at 0% PCK. The fleet (ruvultra RTX 5080, cognitum-seed-1) is the unblock, but the decision is accepted-on-paper, not proven. **Sweep: NOT addressed** — 154–163 never touch the camera-teacher path. Real open backlog item.

**ADR-158 — MAT/World-Model sweep (life-safety).** *Accepted/implemented for the correctness fixes, but capability remains DATA-GATED.* The sweep honestly fixed the dangerous bugs (unified the two divergent triage engines so survivor count can't inflate from repeat detection — lines 46–56, 184–186), but explicitly grades the actual capabilities as unproven: **RF-through-rubble survivor detection = DATA-GATED** (needs instrumented rubble trials, line 37); **learned multi-person counter = DATA-GATED** on labelled multi-occupant CSI (lines 41, 173); PicoScenes/Intel-5300/Atheros live capture DATA-GATED on NIC/driver hardware (lines 177–179). **Sweep: addressed the slop, honestly deferred the capability.** This is the model the rest should follow — code is real, accuracy claim is withheld pending absent hardware. Severity CRITICAL because it is the life-safety surface; the residual gate is acceptable and labeled.

#### HIGH — shipped/benchmarked claim with an explicit residual gate

**ADR-152 — WiFi-Pose SOTA 2026 Intake.** Status header stale (says Proposed; commits + line 58 report §2.1–2.3/2.6 implemented and WiFlow-STD **MEASURED-EQUIVALENT 96.09% PCK@20** on RTX 5080). Residual gates are real and disclosed: (1) **1 of 25 verified claims REFUTED 0-3** — "ESP WiFi-6 drop-in compatible with RuView nodes" is false (WiFi-6 parts use a different CSI acquisition struct, lines 31, 123); (2) external pose numbers (PerceptAlign −60% cross-domain; UNSW MAE pose transfer) remain **CLAIMED until reproduced on our hardware** (lines 21, 27, 119–122); (3) measurement (b)/(c) open — line 111 confirms pretrained init gives optimization transfer but **no feature transfer**, and no run beat a mean-pose baseline on single-subject data, so **no CSI→pose capability is citable** until multi-subject/multi-position data exists. Blocker: heterogeneous multi-subject CSI dataset (data-gated, per ADR-150 §F3). **Sweep: this ADR *is* the prove-everything discipline applied to research intake** — gates labeled, not buried.

**ADR-072 / ADR-150 — WiFlow pose + RF foundation encoder.** ADR-072 >80% PCK@20 target unverifiable without camera labels (resolved-path via ADR-079, itself gated above). ADR-150 cites measured 81.63% in-domain vs **~11.6% leakage-free cross-subject** — the cross-subject collapse is real and the stated lever (ADR-152 F3) is *more heterogeneous data*, not capacity. Blocker: multi-subject/room dataset + libtorch GPU training. **Sweep: NOT directly addressed** (155 fixed PCK/OKS metric-integrity plumbing, which makes these numbers *trustworthy* but doesn't close the data gap).

#### HIGH — security/privacy decisions still Proposed-only (no sweep touched the gate itself)

**ADR-080 — QE Remediation.** Tracks unfixed security HIGH findings (X-Forwarded-For bypass, leaked stack traces, JWT-in-URL CWE-598), gate FAILED, status Proposed, no done-marking. The HOMECORE sweep (ADR-161/162) fixed *HOMECORE*'s WS-auth bypass and plugin signing — a **different** server boundary. **Sweep: did NOT cover ADR-080's sensing-server findings.** Genuine open security backlog.

**ADR-105→109, ADR-118–125 (BFLD/federation/fabric chains).** Entire federation chain (105–109) and BFLD surface (118–125) are Proposed-only, all ACs unchecked, several "tracking issue TBD." Blockers: KIT BFId dataset (ADR-121 calibration), Pi5/Nexmon CBFR capture hardware (ADR-123 — ESP32 *structurally cannot* sniff CBFR), Soul-Signature + cog-ha-matter dependencies (ADR-122/125). **Sweep: NOT addressed** — 154–163 stop at HOMECORE/MAT/cog/edge; the privacy control *plane* (ADR-141, built) exists but the BFLD *capture/scoring* chain it would gate does not. Backlog, honestly gated by absent hardware.

#### MEDIUM — hardware-gated, honestly deferred BY the sweep (lowest risk)

**ADR-163 — Edge-latency measurement.** *Accepted/implemented* for host benches, but the **ESP32/Xtensa on-hardware `process_frame` figure is explicitly UNMEASURED / PENDING (hardware)** (lines 31–32, 79–83, 92–93). Blocker: `wasm32-unknown-unknown` built + flashed to ESP32-S3 and timed on-device; host x86_64 median is "an upper bound on algorithm work, not the ESP32 number." This is the **gold-standard deferral**: the gate is stated everywhere, no claim overreaches. **Sweep: this *is* a sweep ADR honestly deferring its own residual.**

**ADR-160 — wasm-edge skill labeling.** Medical/affect/weapon capabilities explicitly **NOT validated** — relabelled/disclaimed/feature-gated rather than implemented, reference-standard-gated. **Sweep: addressed by relabeling, capability honestly deferred.**

**ADR-110 — ESP32-C6 firmware.** Implemented, but HE-CSI requires ESP-IDF ≥5.5 (v5.4 silently downconverts to HT) — capability hardware/toolchain-gated per WITNESS §B1. Not a sweep target; gate is a noted hardware constraint, not slop.

**Other purely hardware/data-gated Proposed decisions (no sweep involvement, no overreach):** ADR-023 (paired data+GPU), ADR-027/MERIDIAN (multi-env data), ADR-042 CHCI (custom PCB/TCXO — largely superseded by 153), ADR-063/064 (ESP32-C6+MR60BHA2 mmWave), ADR-065/066 (live Cognitum Seed deploy), ADR-070 (live 2-node+Seed capture), ADR-073/078 (multi-AP mesh deployment), ADR-083 (pending field evidence), ADR-086 (real-deployment suppression rates), ADR-091 (COTS sub-THz + ITAR-clear use case), ADR-103 (labelled count data), ADR-113 (Fresnel-sim, not hardware-validated), ADR-114 (real NV-diamond device), ADR-134/135 (COM9/COM12 hardware-test feature), ADR-143 v2 (7-day fleet validation campaign, dead-code until then), ADR-144 (no UWB radio in fleet).

#### Cross-cutting finding
The sweep (ADR-154–163) is **narrowly scoped**: it hardened MAT (158), Cognitum cogs (159), wasm-edge (160), HOMECORE server+plugins (161/162), and latency debt (163) — converting CLAIMED→MEASURED or DATA-GATED with honest labels. It **did not** touch the two largest *capability* gaps: the **camera-teacher training validation (ADR-079/072/150)** and the **federation/BFLD privacy chains (105–109, 118–125)** — both remain data/hardware-gated and Proposed-only. The single hard contradiction worth flagging to a human: **ADR-079's baseline table reports the target (35.3%) as if achieved while the prose and #640 evidence say 2.5%/0%** — that is the one place a reader could mistake an aspiration for a measurement.

# ADR-164: ADR Corpus Gap Analysis & Remediation Backlog

- **Status:** proposed
- **Date:** 2026-06-12
- **Deciders:** ruv
- **Tags:** governance, meta

## Context

The corpus has grown to **162 ADR entries across 156 distinct files** (ADR-001 through ADR-171; the 5 duplicate-number collisions / 6 displaced files originally noted here were RESOLVED by renumbering the displaced files to ADR-166…171 — see Gap Register G1). It now spans nine subsystems — signal/DSP, NN/training, ESP32 firmware, RuvSense multistatic, RuView desktop, Cognitum cogs, HOMECORE (HA reimplementation), BFLD privacy, and the streaming engine — written over roughly a year by many agent-driven sessions.

Two forces motivate a corpus-wide gap analysis *now*:

1. **The beyond-SOTA / anti-AI-slop sweep (ADR-154–163) just landed.** That sweep is itself a structured retraction layer: each ADR exists *because* an earlier accepted-or-shipped claim was found false (a dead CIR coherence gate, a fake-gradient TTA path, a self-certifying proof, a WebSocket auth bypass, an inflated survivor count). The sweep hardened five subsystems but was narrowly scoped — it never touched the two largest capability gaps (camera-teacher training validation; federation/BFLD privacy chains). A ledger is needed to record what the sweep retracted and what it left open.
2. **The status field can no longer be trusted as a source of truth.** A five-lens audit (status-distribution, supersession-chains, contradictions, coverage-gaps, data-hardware-gated) found ~24 ADRs mislabeled `Proposed` while their own commit-pinned Implementation-Status notes report them built and tested; 6 ADR numbers collide; 3 files have no Status header at all. An auditor reading headers would conclude "not built" for landed code, and "built/Accepted" for unvalidated capability.

The detailed lens outputs and the full per-ADR census live in `docs/adr/gap-analysis/` (`lens-findings.md`, `census.md`). This ADR is the authoritative summary and remediation backlog.

## Decision

**This ADR is the authoritative gap ledger and remediation backlog for the ADR corpus as of 2026-06-12.** It does not change any subsystem behavior. It records, with cited ADR ids:

- the status/impl distribution and the bookkeeping-drift problem;
- a prioritized Gap Register with a recommended action per gap;
- supersession-integrity defects;
- the contradiction/retraction list (the anti-slop centerpiece);
- shipped capabilities with no governing ADR;
- the genuinely open data/hardware-gated backlog.

Until the Gap Register items are worked, **treat the ADR Status header as advisory, not authoritative**, and treat any accuracy number authored before ADR-155 landed as CLAIMED (not MEASURED) until re-derived through the post-155 leak-free validation split.

## Status Distribution

Counts are approximate (`~`) where a status string is non-canonical or dual-valued; the per-ADR breakdown is in `census.md`.

| Status bucket | Count | impl_state | Count |
|---|---|---|---|
| Accepted (incl. partial/in-progress/Phase-1 variants) | ~56 | implemented | ~36 |
| Proposed (incl. conditional/research-only) | ~88 | partial | ~50 |
| Superseded | 1 (ADR-002) | proposed-only | ~64 |
| Rejected | 1 (ADR-098) | stale-or-contradicted | 3 (029/030/031) |
| Missing / no Status header | 3 (ADR-168-proof [was 147], ADR-167-ddd [was 052], ADR-134) | unknown | 5 (034/044/167-ddd/168-proof/…) |
| Mixed/dual status in one ADR | 3 (115, 149×2, 133) | superseded | 1 (ADR-002) |

**Headline:** ~114 of 162 ADRs (≈70%) are decisions that never fully landed (proposed-only + partial + stale + unknown). The dominant failure mode is **stale Status headers**, not abandoned work.

## Gap Register

Severity: CRITICAL (corpus integrity / tooling-breaking / life-safety / security) · HIGH · MEDIUM · LOW. Action vocabulary: *implement · supersede · mark-stale · write-missing-ADR · close-as-gated · renumber · reconcile-docs*.

| ID | Gap | Severity | Affected ADRs | Recommended action |
|----|-----|----------|---------------|--------------------|
| G1 | ~~6 duplicate ADR numbers (two ADRs answer to one number; breaks index/`/adr` tooling)~~ **RESOLVED (duplicate-number item)** | CRITICAL | 050×2, 052×2, 147×3, 148×2, 149×2; 134 (identity split, separate) | ~~renumber 2-of-3 at 147, 1 each at 050/148/149; demote 052-ddd to appendix; resolve 134 identity~~ **DONE: displaced files renumbered to the next free numbers (166–171), keepers = first-committed file per number (date ties broken by inbound-ref count / parent-appendix relationship): 050 keeps provisioning-tool-enhancements → quality-engineering-security-hardening = ADR-166; 052 keeps tauri-desktop-frontend → ddd-bounded-contexts appendix = ADR-167 (still linked to parent 052); 147 keeps nvidia-cosmos/OccWorld → benchmark-proof = ADR-168, adam-mode-light-theme = ADR-169; 148 keeps drone-swarm-control-system → yoga-mode-pose-system = ADR-170; 149 keeps public-community-leaderboard-huggingface → swarm-benchmarking-evaluation-methodology = ADR-171. In-file headers, intra-file self-refs, all inbound cross-references (README index, census, lens-findings, user-guide, CHANGELOG, proof-of-capabilities, research docs), and this register updated. `ls docs/adr/ADR-*.md | … | uniq -d` is now EMPTY. The ADR-134 identity split is NOT a filename collision; resolved separately under G3 (→ ADR-165).** |
| G2 | 3 files with no Status header (cannot triage) — **INVESTIGATED in `docs/adr-gap-remediation-1`: only 2 genuinely lack one, both owner-gated** | CRITICAL | ADR-168-benchmark-proof (was 147), ADR-167-ddd-appendix (was 052), ~~134-CIR~~ | add canonical `## Status`; relocate ADR-168-proof to `benchmarks/`; label ADR-167-ddd as appendix — **NOTE: ADR-134-CIR DOES have a Status (`\| Status \| Proposed \|` in its header table) — mislabeled here. The two real misses (ADR-168-benchmark-proof [was 147], ADR-167-ddd [was 052]) were inside the owner-gated duplicate-number collisions (147×3, 052×2); those collisions are now resolved (G1) but the missing Status headers themselves remain owner-gated, so left untouched pending owner. The early ADRs (048/049/068/070 etc.) use `\| Status \|` not `\| **Status** \|` — different-format-but-present, not missing. Net: 0 headers added.** |
| G3 | ~~Shipped crates cite a non-existent or wrong-identity governing ADR~~ **RESOLVED in `docs/adr-gap-remediation-1`** | CRITICAL | homecore-recorder→"ADR-132" (no file); homecore-migrate→"ADR-134" (file is CIR) | ~~write-missing-ADR (HOMECORE-RECORDER, HOMECORE-MIGRATE)~~ DONE: wrote ADR-132 (recorder, Accepted) + ADR-165 (migrate, Accepted — P1 scaffold); repointed migrate's ADR-134 refs → ADR-165 |
| G4 | Anti-slop retractions: accuracy/security/function provably false until sweep landed | CRITICAL | 155, 154, 079, 161 (see Contradictions) | already fixed in-code by 154/155/161/162; this ledger records the retraction |
| G5 | ~~10 streaming-engine ADRs marked `Proposed` while §Impl-Status reports Built + commits + tests~~ **RESOLVED in `docs/adr-gap-remediation-1`** | HIGH | 136–145 | ~~mark-stale → "Accepted — partial (integration glue pending)" (one batch)~~ DONE: all 10 (136–145) flipped to "Accepted — partial"; each retains its commit-pinned Implementation-Status note. NB: notes describe *building blocks built + tested*, **not** live-path integration — "partial" is the honest label, not full "Accepted" |
| G6 | Stale `Proposed` headers on built+published code | HIGH | 029/030/031, 095/096, 152, 154–157, 024/027/072, 150 | mark-stale; reconcile with downstream/CLAUDE.md evidence |
| G7 | Status-graph inversion: Accepted ADR depends on Proposed parent | HIGH | 032→029/030/031; 053→052; 048→045; 077→075/076; 104→103 | promote parents to match built reality, or downgrade dependents |
| G8 | ADR-002 supersession not reciprocated by successors; 5 children stranded | HIGH | 002→016/017; children 003/007/008/009/010 | reconcile-docs (add reciprocal language or downgrade); split 002 to "partially superseded" |
| G9 | Streaming-engine integrator crate has no governing ADR (composition/back-pressure/live-path seam) | HIGH | wifi-densepose-engine (composes 135–146) | write-missing-ADR |
| G10 | CLAUDE.md doc-vs-header drift (doc says one status, header another) | HIGH | 017, 024, 027, 072, 152 | reconcile-docs |
| G11 | ~~Open security HIGH findings, gate FAILED, never marked done~~ **RESOLVED (2026-06-13, branch `fix/adr-080-sensing-server-security`)** | HIGH | 080 (XFF bypass, leaked stack traces, JWT-in-URL CWE-598) | ~~implement (sensing-server boundary — NOT covered by HOMECORE sweep 161/162)~~ DONE: verified all three against the *current Rust* `wifi-densepose-sensing-server`. **#2 leaked errors** was the one live exposure — 6 `main.rs` handlers serialized internal `Display`/`JoinError` into response bodies; fixed via a new `error_response` module (generic body + correlation id, detail logged server-side only). **#1 XFF** and **#3 JWT-in-URL** were verified *absent* on the Rust boundary (no IP-rate-limit/allowlist reads XFF; token is header-only, WS handlers take no query token) and pinned with regression tests that fail if either is re-introduced. ADR-080 P0 §1–3 marked RESOLVED. |
| G12 | ADR-052→054 edge unacknowledged by successor; likely mis-modeled (impl, not replacement) | MEDIUM | 052-tauri, 054 | reconcile-docs (054 is the impl plan *for* 052, not a replacement) |
| G13 | Capability governed only by remediation/deploy ADR, no creation/architecture ADR | MEDIUM | wasm-edge (only 160/163); occworld-candle (147 blessed Python path only); pointcloud (094 = viewer deploy only) | write-missing-ADR (taxonomy/ABI for wasm-edge; Candle backend swap; pointcloud data contract) |
| G14 | Conflicting decisions on one topic, none superseding the others | MEDIUM | person-count 037/075/103; PQ-sign 007/109; fed key-exchange 107/108; provisioning 050/060/052; audit 010/028; RVF-WASM 009-vs-shipped | reconcile (pick one, supersede the rest) |
| G15 | ~50 Proposed-forever chains pollute every gap analysis | MEDIUM | 003/007–010, 105–109, 118–125, HOMECORE 124–133, 033/046/049/067/074/085 | close-as-gated or mark Deferred/Rejected + open tracking issues |
| G16 | De-facto supersessions never recorded (lifecycle graph incomplete) | MEDIUM | 098/099, 063/064, 042/153, 050/060, 035/023, 100/109, 117 retracts PyPI v1.1.0 | reconcile (add supersedes/superseded_by fields) |
| G17 | Accepted but no implementation evidence ("unverified done") | MEDIUM | 034 (FieldView app — no crate); 044 (wifi-densepose-geo — bare Accepted, no Date/Deciders) | implement or downgrade to Proposed |
| G18 | Workspace has ~38 crates; CLAUDE.md publishing list (12-step) and crate table (15) are stale | MEDIUM | corpus-wide (crate-graph topology) | write-missing-ADR (crate-graph / publish boundaries) + reconcile CLAUDE.md |

## Supersession Integrity

Only **3 formal supersession edges** exist; all three are defective (see G8/G12; full detail in `lens-findings.md` Lens 2):

- **ADR-002 → ADR-016 / ADR-017** is one-directional. ADR-016 never mentions ADR-002 (its References list only 014/015); ADR-017 only *corrects* ADR-002's "fictional crate names" and never says "supersede." The census `supersedes:["ADR-002"]` on 016/017 is **file-unsupported** — the superseded ADR points up at two successors that do not point back.
- **ADR-002 is an umbrella** whose children 003/007/008/009/010 are still `Proposed`. ADR-016/017 realize only the training/signal/MAT integration points; the RVF-container (003), PQ-crypto (007), Raft (008), WASM-edge-runtime (009), and witness-chains (010) decisions are **neither implemented nor formally superseded**. Marking the parent fully "Superseded" silently buries 5 live-but-abandoned child decisions. Recommended: split ADR-002 to "partially superseded."
- **ADR-052-tauri → ADR-054** is declared by the predecessor but ADR-054 contains zero references to ADR-052. ADR-054 ("Full Implementation", in progress) is the impl plan *for* 052, not a replacement — likely a mis-modeled edge.
- **No cycles** detected. The graph is clean structurally; the defect is missing reciprocity and ~7 unrecorded de-facto supersessions (G16).

## Contradictions & Retractions (anti-slop centerpiece)

The four CRITICAL items are the corpus's load-bearing AI-slop admissions — each an accepted-or-shipped surface whose stated accuracy/security/function was provably false until the sweep landed. **Every accuracy number predating ADR-155 should be treated as CLAIMED until re-derived through the post-155 leak-free split.** Source-cited evidence is in `lens-findings.md` Lens 3.

- **[CRITICAL] ADR-155** retracts every prior NN accuracy/TTA/proof claim: real MM-Fi training validated against a *synthetic* val set with stride-1 (~99%) window leakage (§2.2); a *fake gradient* `grad += v*0.01` in the TTA path (§2.3); a *self-certifying* proof that blessed whatever the pipeline emitted and PASSed on 1e-9 float noise (§2.4).
- **[CRITICAL] ADR-154** proves the ADR-134 CIR coherence gate was **dead in production for every canonical 56-tone frame** (`SubcarrierMismatch`, 0 Ok / 8 mismatch), silently degrading coherence to freq-only. Any "CIR-enhanced coherence/ToF" claim before this fix overstated reality.
- **[CRITICAL] ADR-079** carries three mutually inconsistent values for its own central metric: proxy PCK@20 = 2.5% (prose) vs 35.3% (baseline table — equal to the *target*) vs 0% upper-body joints; #640 measured 0% on real local data. An Accepted ADR whose headline 10–20x improvement is self-refuting.
- **[CRITICAL] ADR-161** fixes a HOMECORE WebSocket **auth bypass** (any non-empty token accepted) + reply-theater + no-op automation; **ADR-162** then enforces plugin Ed25519 signature verification, capability isolation, and bounded RunModes — retracting ADR-128/129/130's implied security guarantees.
- **[HIGH]** ADR-152 self-refutes 1 of 25 claims (ESP WiFi-6 "drop-in" REFUTED 0-3); CLAUDE.md's "WiFlow-STD MEASURED-EQUIVALENT ~96% PCK" contradicts §F1's own gating (97.25% is CLAIMED until measurements (a)–(c) run). ADR-150 retracts the implied cross-subject capability (81.63% in-domain vs ~11.6% leakage-free cross-subject; DANN ~0 gain). ADR-159 ships real models but discloses person-count `training_class1_accuracy = 0.343` and renames "learned multi-person counter" → "presence detector," gutting ADR-103/104's claim.
- **[MEDIUM]** ADR-163 leaves the ESP32/Xtensa on-hardware latency figure UNMEASURED; ADR-098↔099 partial reversal on midstream; ADR-147 self-retracts Cosmos for OccWorld.

## Coverage Gaps (shipped capability, no/broken governing ADR)

- ~~**CRITICAL — `homecore-recorder`** (SQLite state history + semantic search) cites "ADR-132", which **does not exist**. The durable-state backbone is ungoverned. → write HOMECORE-RECORDER ADR.~~ **RESOLVED in `docs/adr-gap-remediation-1`:** ADR-132 written (`ADR-132-homecore-recorder-history-semantic-search.md`, Status: Accepted — reverse-documented from the shipped crate).
- ~~**CRITICAL — `homecore-migrate`** (reads untrusted Python-HA `.storage/*.json`) cites "ADR-134", but on-disk ADR-134 is CIR. A data-integrity-sensitive importer governed by a phantom identity. → resolve 134 collision + write HOMECORE-MIGRATE ADR (trust boundary).~~ **RESOLVED in `docs/adr-gap-remediation-1`:** ADR-165 written (`ADR-165-homecore-migrate-from-home-assistant.md`, Status: Accepted — P1 scaffold); crate's `ADR-134` refs repointed → ADR-165; on-disk ADR-134 (CIR) left intact. ADR-126's series-map row (which labels the *role* "ADR-134 HOMECORE-MIGRATE") is owner-gated and unchanged.
- **HIGH — `wifi-densepose-engine`** composes ADR-135..146 onto the live 20 Hz path but **no ADR governs the integrator contract** (ordering, back-pressure, "one pipeline cycle" boundary).
- **MEDIUM — `wasm-edge`** (~70 skills) governed only by remediation ADRs 160/163 — no creation/taxonomy/ABI ADR. **`occworld-candle`** is a Rust-native backend swap ADR-147 explicitly deferred. **`pointcloud`** has only a viewer-deploy ADR (094), no data-format contract.
- **MEDIUM — workspace topology:** ~38 crates exist; the CLAUDE.md 15-crate table and 12-step publishing order are stale, and no ADR governs crate-graph/publish boundaries at this scale.
- Verified-governed (scoped out): worldmodel→147, worldgraph→139, cog-*→101/103/116, ruview-swarm→148, nvsim→089/092, bfld→118-123/141, calibration→151, homecore-hap→125, geo→044, desktop→052/054.

## Open / Gated Backlog (genuinely unresolved, honestly labeled)

The ADR-154–163 sweep was narrowly scoped. The two largest **capability** gaps it did not touch:

- **CRITICAL — Camera-teacher training validation (ADR-079 / 072 / 150).** P7–P9 Pending; blocker is a real synchronized camera+ESP32 paired-capture session + GPU training on the fleet (ruvultra RTX 5080). Cross-subject collapse (11.6%) is data-gated on a heterogeneous multi-subject CSI dataset, per ADR-150 §F3 / ADR-152 F3 (the lever is *more data*, not capacity). Accepted-on-paper, not proven.
- **HIGH — Federation + BFLD privacy chains (ADR-105–109, 118–125).** All Proposed-only, ACs unchecked. Blockers: KIT BFId dataset (121), Pi5/Nexmon CBFR capture hardware (123 — ESP32 structurally cannot sniff CBFR), Soul-Signature + cog-ha-matter (122/125). The privacy control *plane* (ADR-141) is built; the *capture/scoring* chain it gates is not.
- ~~**HIGH — Sensing-server security (ADR-080).** Distinct from the HOMECORE boundary the sweep fixed; XFF bypass / stack-trace leakage / JWT-in-URL remain open.~~ **RESOLVED (2026-06-13, G11):** verified against the current Rust sensing-server — stack-trace leakage was the one live finding (fixed via `error_response` generic bodies); XFF bypass and JWT-in-URL were verified absent and regression-pinned. See ADR-080 P0 §1–3.
- **MEDIUM — gold-standard deferrals (model to follow):** ADR-163 (ESP32 on-hardware latency UNMEASURED), ADR-160 (medical/affect/weapon NOT validated, relabelled), ADR-158 (RF-through-rubble + learned counter DATA-GATED). Code is real, the claim is withheld pending absent hardware/labelled data — labels are honest.
- **MEDIUM — purely hardware/data-gated Proposed decisions (no overreach):** ADR-023, 027, 042, 063/064, 065/066, 070, 073/078, 083, 086, 091, 103, 110 (HE-CSI needs ESP-IDF ≥5.5), 113, 114, 134/135, 143-v2, 144. *needs verification* where flags rely on downstream prose rather than direct file inspection.

## Consequences

**Positive.** One authoritative ledger replaces scattered, drifting status fields. The anti-slop retractions are recorded in a citable place, so the "AI slop" accusation is met with a structured admission + fix-trail rather than denial. The Gap Register is a concrete, severity-ordered work queue. Batch-fixing G5 (10 streaming-engine headers) and G1/G2 (numbering + missing headers) is high-ROI and unblocks ADR tooling.

**Negative.** This ADR is a snapshot; it goes stale the moment the next ADR lands. Counts marked `~` are approximate and a few impl_state values are *needs verification* (downstream-prose-derived, not file-confirmed). Acting on the register (renumbering, status flips, supersession edits) touches ~30 files and risks transient cross-reference breakage if not done atomically.

**Neutral.** No subsystem behavior changes. Renumbering decisions (which of the colliding files keeps each number) are deferred to the follow-up remediation PR — this ADR records the collision, not the resolution. Whether to close abandoned chains as `Rejected` vs `Deferred` is a judgment call left to the deciders per chain.

## Links

- `docs/adr/gap-analysis/census.md` — full per-ADR census (162 entries).
- `docs/adr/gap-analysis/lens-findings.md` — five-lens findings (status-distribution, supersession-chains, contradictions, coverage-gaps, data-hardware-gated), verbatim.
- Anti-slop sweep: ADR-154, ADR-155, ADR-156, ADR-157, ADR-158, ADR-159, ADR-160, ADR-161, ADR-162, ADR-163.
- Most-cited defects: ADR-079, ADR-134, ADR-002, ADR-136–145, ADR-152.
- Governance: CLAUDE.md (crate table + publishing order — stale per G18); ADR-038 (prior roadmap census, now stale).

# ADR-149: AetherArena ("AA") — The Official Spatial-Intelligence Benchmark (Hugging Face)

> **Scope note:** AetherArena is a **standalone, project-agnostic benchmark** for spatial intelligence — open to *any* project, team, or modality, not a RuView-branded board. RuView contributes the initial scoring harness and enters as one baseline among others; it gets no special treatment. This ADR lives in the RuView repo only because RuView is donating the seed harness — the benchmark itself is independent.

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-05-30 |
| **Deciders** | ruv |
| **Gate decisions** | Name **locked**: `ruvnet/aether-arena` ("AA"), positioned as the official cross-project Spatial-Intelligence Benchmark. v0 ranked metrics **locked**: pose, presence, edge-latency, determinism. Dataset legality **resolved**: MM-Fi (CC BY-NC 4.0) only for v0; Wi-Pose dropped (research-use, no redistribution). |
| **Codebase target** | New repo `ruvnet/aether-arena` (leaderboard + HF Space); reuses `wifi-densepose-train` (`src/ruview_metrics.rs`, `src/ablation.rs`, `src/eval.rs`, `src/proof.rs`) and `wifi-densepose-cli` as the scoring engine |
| **Relates to** | ADR-011 (Deterministic Proof Harness), ADR-015 (Public Dataset Training Strategy — MM-Fi / Wi-Pose), ADR-024 (Contrastive CSI Embedding / HF model release), ADR-027 (Cross-Environment Domain Generalization / MERIDIAN), ADR-031 (RuView Sensing-First RF Mode — `RuViewTier` acceptance), ADR-079 (Camera-Supervised Pose Fine-tune — PCK@20), ADR-120 / ADR-141 (BFLD Privacy), ADR-145 (Ablation Eval Harness — the scoring substrate) |

---

## 1. Context

### 1.1 The Gap

RuView has a mature, deterministic evaluation surface but **no public face for it**. Two assets already exist:

1. **A grading harness.** `wifi-densepose-train/src/ruview_metrics.rs` rolls pose (PCK@0.2 / OKS / torso jitter / p95 error), tracking (MOTA / ID-switches / fragmentation), and vitals (breathing/heartbeat BPM error + SNR) into a `RuViewAcceptanceResult` with a `RuViewTier` (`Fail` / `Bronze` / `Silver` / `Gold`). ADR-145's `src/ablation.rs` extends this with presence accuracy, localization error, FP/FN, latency p50/p95/p99, a privacy-leakage score ∈ `[0,1]`, and cross-room degradation, under a determinism binding inherited from the ADR-011 proof harness.

2. **A determinism substrate.** `proof.rs` (`PROOF_SEED=42`) SHA-256-hashes model outputs against an expected hash, so a scored run is reproducible and tamper-evident.

What is missing is a **public, multi-entrant ranking**. As surveyed in ADR-015 and `docs/research/sota-surveys/sota-wifi-sensing-2025.md`, the WiFi-sensing field has **no hosted live leaderboard** the way vision has COCO/EvalAI — researchers self-report numbers against public *datasets* (MM-Fi, Wi-Pose, Person-in-WiFi, Widar3.0) in papers, with inconsistent splits, metrics, and no privacy or latency accounting. RuView's own pose number (PCK@20 ≈ 2.5% with proxy labels, target 35%+ per ADR-079) is currently self-reported on a private validation set and is not comparable to the MM-Fi SOTA (MultiFormer 0.7225).

### 1.2 The Opportunity

The harness that already gates RuView releases is exactly the engine a community leaderboard needs: a single, deterministic, privacy- and latency-aware scoring function. Publishing it as an open leaderboard:

- Establishes **AetherArena as the field's standard yardstick** for spatial intelligence, with RuView's `RuViewTier` + ADR-145 metric set contributed as its initial basis (pose + tracking + vitals + **privacy-leakage** + latency + determinism — a combination no existing benchmark scores). The standard is AA's; RuView donates the seed.
- Draws **any project, framework, or modality** to submit and rank — a cross-project community flywheel, not a RuView-only one (RuView's `wifi-densepose-pretrained` is merely the first baseline).
- Forces the harness to harden: a public, neutral scorer must be reproducible by strangers, resistant to gaming, and runnable on a fixed held-out split nobody can train on.

### 1.3 Constraints & Risks Up Front

- **Leakage of the held-out split** is the existential risk for any leaderboard. The eval data must be private; submitters provide a model, not predictions on data they hold.
- **Compute cost.** Scoring a submission runs inference over the eval set; an HF Space on free CPU may be too slow for the Candle/`tch` pipeline. Tiering of compute (CPU smoke vs GPU full score) is required.
- **Privacy / consent of the eval data.** MM-Fi and Wi-Pose carry their own licenses; we can host *derived* CSI features and scores but must respect redistribution terms (ADR-015 already tracks this).
- **Trust.** A `RuViewTier` badge is only meaningful if the scoring is deterministic and the leaderboard cannot be silently edited — the ADR-011 proof hash and a signed results ledger address this.

---

## 2. Decision

**Create AetherArena ("AA") — the official, project-agnostic Spatial-Intelligence Benchmark: a public, open-entry leaderboard for camera-free spatial perception (pose, presence, occupancy, tracking, vitals) as a standalone repo `ruvnet/aether-arena` paired with a Hugging Face Space. The scoring engine is seeded by RuView's existing `ruview_metrics` + ADR-145 ablation harness, contributed as a neutral scorer; v0 evaluates against a private MM-Fi held-out split.**

AA is **not a RuView leaderboard**. It is the field's missing standard yardstick for spatial intelligence — open to any team, framework, or sensing modality. The RF medium is the v0 input and RuView donates the seed harness + a baseline entry, but the benchmark is independent and RuView is scored like every other entrant. The metric surface — pose, presence, tracking, occupancy/world-model, latency, determinism, and later privacy — is modality-agnostic, leaving room to grow to mmWave / UWB / radar / lidar / multimodal entrants and other projects.

The leaderboard does **not** fork or re-implement the scoring logic. It is a thin orchestration + presentation layer over the published `wifi-densepose-cli` scorer, so the public number a model earns is identical to the number RuView uses internally to gate releases. **This makes the leaderboard governance, not marketing.**

The whole design reduces to a precise four-part structure:

> **Public leaderboard. Private evaluation split. Open scorer. Signed results.**

- **Public leaderboard** — anyone can see the ranking and submit.
- **Private evaluation split** — the held-out data is never published; it cannot be trained on or overfit.
- **Open scorer** — the scoring code is the published `wifi-densepose-cli`; a stranger can rerun it locally on a public *smoke* split and reproduce the logic.
- **Signed results** — every score is an append-only, signed ledger row with a determinism proof hash; ranks cannot be silently edited.

### 2.1 Name — DECIDED: `ruvnet/aether-arena` ("AA")

**Locked.** Canonical repo + HF Space: **`ruvnet/aether-arena`**, branded **AetherArena** with the short form **"AA"**.

- **"Aether"** = the classical all-pervading medium — fitting for RF/ambient spatial perception, and broader than "Ether"/CSI/WiFi so the benchmark can grow to mmWave, UWB, and multimodal spatial-intelligence entrants without a rename.
- **"Arena"** = open competitive entry.
- HF Space title: *AetherArena (AA) — the spatial-intelligence benchmark for RF perception.*
- `ruvnet/wifi-densepose-leaderboard` is kept only as a discoverability/topic alias that redirects to AA.

(Rejected: `csi-arena` — jargon; `rf-bench` — generic/collision; `wifi-densepose-leaderboard` as the primary — ties the brand to one capability.)

### 2.2 Architecture

```
 Submitter                        ruvnet/aether-arena                     RuView harness
 ─────────                        ──────────────────                     ──────────────
 push model.safetensors  ──►  HF Space (Gradio): submit form       ┌─ wifi-densepose-cli score
 + model card (adapter,        │  • validates manifest             │   ├─ load model snapshot
   input contract, license)    │  • queues job                ──►  │   ├─ replay private MM-Fi/
                                │  • runs scorer in container       │   │   Wi-Pose split (PROOF_SEED)
                                │  • appends signed result          │   ├─ ruview_metrics → RuViewTier
                                ▼                                   │   ├─ ablation.rs → p50/p95,
                          leaderboard.parquet  ◄────────────────────┘   │   privacy-leakage, cross-room
                          (HF dataset, append-only,                     └─ emit result + SHA-256 proof
                           one signed row per submission)
```

1. **Submission contract.** A submitter pushes a model artifact (`model.safetensors` / `.rvf` / LoRA adapter) plus a `ruview-arena.toml` manifest declaring: input feature set (which ADR-145 `FeatureSet` it consumes — F0 CSI / F1 CIR / F2 Doppler / F3 BFLD), tensor I/O contract, license, and optional category (pose / presence / tracking / vitals / multi-task).
2. **Scoring.** The Space runs the **published `wifi-densepose-cli`** in a pinned container against a **private held-out split** of MM-Fi / Wi-Pose (and RuView's own paired-capture set per ADR-079). Output is the existing `RuViewAcceptanceResult` + the ADR-145 scalar set, plus the ADR-011 SHA-256 reproducibility hash.
3. **Ledger.** Each scored submission appends **one signed row** to an append-only HF dataset (`ruvnet/aether-arena-results`, Parquet): `{submitter, model_ref, category, feature_set, tier, pck20, oks, mota, vitals_bpm_err, latency_p50, latency_p95, privacy_leakage, cross_room_deg, proof_sha256, scored_at, harness_version}`. Append-only + signed = no silent edits.
4. **Presentation.** Gradio leaderboard with category tabs (Pose / Presence / Tracking / Vitals / Edge-latency / **Privacy**), `RuViewTier` badges, and a "privacy-respecting" filter (leakage ≤ threshold) — the differentiator no other WiFi benchmark has.

### 2.2.1 Submission Lifecycle (quarantine before scoring)

A submission is an untrusted artifact, so it moves through an explicit state machine — artifacts are isolated and validated **before** any scoring touches the private split. This is both the abuse-handling boundary and the UI flow:

| State | Meaning |
|-------|---------|
| `submitted` | manifest received, job queued |
| `validated` | schema, license, and artifact type accepted |
| `quarantined` | artifact scanned; loaded into the sandbox (network disabled, read-only FS, runtime prepared) |
| `smoke_scored` | passes the **public** smoke split (cheap CPU correctness check) |
| `full_scored` | **private** held-out split score produced |
| `published` | signed row appended to the ledger; appears on the board |
| `rejected` | failed a gate — terminal, with a machine-readable reason |

Only `quarantined` → `smoke_scored` → `full_scored` ever runs the model, always inside the sandbox of §2.4. A failure at any gate transitions to `rejected` with a reason rather than silently dropping.

### 2.3 Categories & Metrics (reuse, do not invent)

| Category | Primary metric (existing) | Source |
|----------|---------------------------|--------|
| Pose | PCK@20, OKS | `ruview_metrics::evaluate_joint_error` |
| Tracking | MOTA, ID-switches | `ruview_metrics::evaluate_tracking` |
| Vitals | breathing/HR BPM error, SNR | `ruview_metrics::evaluate_vital_signs` |
| Presence | accuracy, FP/FN | ADR-145 `ablation.rs` |
| Edge latency | p50 / p95 / p99 ms | ADR-145 `LatencyProfile` |
| **Privacy** | leakage score ∈ `[0,1]` (membership-inference) | ADR-145 §10 |
| Cross-room | degradation ratio | ADR-027 / ADR-145 |
| Overall | `RuViewTier` Bronze/Silver/Gold + `arena_score` (§2.5) | `determine_tier()` |

### 2.3.1 Phased Launch — v0 ships narrow

**A narrow leaderboard that works beats a broad one with half-real metrics.** v0 ranks only categories whose metric is fully implemented and reproducible-by-strangers today; the rest are visible as **"coming soon" / gated** and are **not ranked** until their metric is real.

| Category | v0 status | Gate to activate |
|----------|-----------|------------------|
| Presence | **Ranked** | — (implemented) |
| Pose (PCK@20 / OKS) | **Ranked** | — (implemented) |
| Edge latency (p50/p95/p99) | **Ranked** | — (implemented) |
| Determinism proof | **Ranked** (pass/fail gate) | — (ADR-011, implemented) |
| Tracking (MOTA) | Optional in v0 | enough multi-person eval clips in the private split |
| Vitals (BPM error) | Optional in v0 | paired vital-sign ground truth in the split |
| **Privacy leakage** | **Coming soon — gated, not ranked** | ADR-145 §10 membership-inference attacker implemented + published |
| Cross-room generalization | Coming soon | multi-room held-out split assembled (ADR-027) |

**v0 launch language (explicit, to stay honest and non-contradictory):** *AetherArena v0 starts with pose, presence, edge latency, and deterministic reproducibility. Tracking and vitals are activated when sufficient ground-truth clips are available. Privacy-leakage and cross-room generalization remain gated until their evaluation attacks and splits are implemented and published.* Shipping a "privacy leaderboard" claim before the attacker exists would be an easy and deserved attack on our credibility.

### 2.4 Threat Model

The leaderboard is only credible if its failure modes cannot be hidden. Explicit threats and the control that neutralizes each:

| Threat | Control |
|--------|---------|
| Model exfiltrates / phones home the eval data | Scorer container runs with **no network, read-only eval FS, resource caps** (sandboxed) |
| Submitter overfits the public split | **Private held-out split** — never published; scoring runs on data the submitter has never seen |
| Model fingerprints / detects the eval set | **Seasonal rotation** of a fraction of the held-out split (mirrors ADR-120 hash rotation) |
| Maintainer silently edits a score / rank | **Witness chain**: append-only, hash-chained ledger (`ledger/ledger_tools.py`) — each row references the prior row's hash, so any edit breaks every subsequent link and `verify` fails |
| A score can't be reproduced / hides nondeterminism | **Witness + repeatability analysis**: each score is a witness (`inputs_sha256` binding it to the exact inputs + `proof_sha256` of the quantised result + `harness_version`); `aa_score_runner --repeat N` runs the harness N× and fails if it ever produces ≥2 distinct proof hashes |
| Scorer version drift changes ranks invisibly | **`harness_version` pinned per witness**; a scorer change moves the proof hash and fails the CI determinism gate until regenerated + reviewed |
| Slow model brute-forces accuracy | **Latency is a ranked axis** (p50/p95/p99) with hard caps + the `latency_factor` in `arena_score` |
| "Gold accuracy, leaks identity" win | **Privacy is a (gated) axis**; once active, `privacy_factor` penalizes leakage in `arena_score` |
| Malicious model artifact (RCE in the scorer) | Untrusted artifact loaded in the sandboxed container only; pinned, minimal runtime; no host mounts |

### 2.5 Overall Score (anti-"accuracy-at-any-cost")

Categories are ranked independently (tabs), **and** an optional headline `arena_score` composes them so a model cannot win on raw accuracy while being slow, leaky, or non-reproducible:

```
arena_score = quality_score × latency_factor × privacy_factor × determinism_gate
```

| Component | Rule |
|-----------|------|
| `quality_score` | normalized blend of PCK@20 / OKS / MOTA / vitals for the category, ∈ `[0,1]` |
| `latency_factor` | `1.0` if p95 ≤ target; decays smoothly above target (edge viability) |
| `privacy_factor` | `1.0 − privacy_leakage` once the Privacy axis is active; **fixed at `1.0` in v0** (privacy gated/unranked) |
| `determinism_gate` | `1.0` if the ADR-011 proof hash matches; **`0` if it fails** — a non-reproducible run cannot rank at all |

The multiplicative form means any single hard failure (non-deterministic, or — later — high leakage) collapses the headline score, even at SOTA accuracy. In v0, `privacy_factor` is pinned to `1.0` so the headline number is honest about what is actually measured.

**`arena_score` is a gate, not the only headline.** Multiplicative composites are great for gating but can hide *why* a model lost, and invite "your formula is biased" arguments. So the board ranks **category performance first** and exposes the composite alongside, never instead:

| Surface | What it shows |
|---------|---------------|
| **Primary rank** | the category metric (e.g. PCK@20 for Pose) — this is the sort key per tab |
| **Integrity badge** | determinism proof pass/fail |
| **Edge badge** | p95 latency band |
| **Overall score** | `arena_score` as an *optional* governance-weighted composite |

> The leaderboard ranks category performance first, then exposes `arena_score` as a governance-weighted composite so accuracy, latency, reproducibility, and privacy are visible rather than collapsed into a single opaque number.

### 2.6 Dataset Legality (investigated — resolved for v0)

Confirmed against ADR-015 §dataset-licenses:

| Dataset | License | What AA may do |
|---------|---------|----------------|
| **MM-Fi** | **CC BY-NC 4.0** | ✅ v0 eval source. Non-commercial use + derivatives **permitted with attribution**. AA may host *derived* CSI features and scores; raw frames stay in the private split. AA must be operated **non-commercially** and carry MM-Fi attribution. |
| **Wi-Pose** | **"Research use"** (no clean redistribution grant) | ⚠️ **Not hosted.** Pulled privately into the scorer only, never redistributed; or deferred until terms are clarified with the authors. **Dropped from v0.** |
| Person-in-WiFi-3D | semi-public access | Future candidate (post-v0), pending access terms. |

**v0 decision:** evaluate on a **private MM-Fi held-out split only** (CC BY-NC, attributed, non-commercial; expose only license-permitted derived features). Wi-Pose is removed from v0 and revisited if/when redistribution is cleared. This keeps the existential "can we even host this" risk at zero for launch.

> **Non-commercial caveat to watch:** CC BY-NC means AA itself, and the eval-data use, must remain non-commercial. Because AA also showcases the (commercial) RuView appliance, keep AA legally distinct and non-commercial, or seek an MM-Fi commercial grant before any paid tier. Flagged for the maintainer.

### 2.7 Non-Gameability Is a Launch Gate

Per the explicit directive, AA does not launch unless the harness is demonstrably hard to game. The controls (private split §2.4, seasonal rotation §2.4, model-not-prediction submission §2.2, sandbox §2.4, pinned `harness_version` §2.4, signed append-only ledger §2.3-§2.4, multiplicative `arena_score` §2.5, `determinism_gate=0` on proof-hash failure §2.5) are **not optional hardening — they are acceptance criteria** (see §7). A v0 that can be topped by overfitting a public split, a non-reproducible run, or a silently edited row is, by definition, not ready.

### 2.8 Neutrality & Governance (because it's "official" and cross-project)

The hardest credibility problem for an *official* benchmark seeded by one entrant: **"RuView built the scorer, so of course RuView wins."** If AA is to be the field's standard rather than RuView marketing, neutrality must be structural, not promised:

| Neutrality risk | Control |
|-----------------|---------|
| RuView's entry gets special treatment | RuView is submitted through the **same** public pipeline (§2.2.1) and scored by the **same** pinned scorer as everyone else; its rows carry the same proof hash and are independently re-runnable on the smoke split. |
| RuView tunes the metric to favor its models | The scorer is **open and versioned**; any metric change is a public `harness_version` bump that **re-scores all entries**, not just new ones. Metric changes go through a public changelog. |
| "Official" is self-declared | AA is positioned as a **neutral commons**: separate repo/Space identity, contribution guide, and an explicit invitation for other projects + dataset authors to co-own splits and metrics. RuView is the *donor of the seed harness*, not the owner of the standard. |
| Benchmark used as RuView ad | Keep AA legally + brand-distinct (ties into the CC BY-NC non-commercial caveat, §2.6); the README leads with the standard, not the product. |
| Single-vendor capture | Roadmap to a multi-org steering/eval committee once ≥N external projects enter; split rotation + metric proposals are public. |

The test for neutrality is the same as §7's acceptance test: a stranger from *another project* can submit, reproduce the score, and see that RuView's own entries were scored by the identical, open, pinned path.

---

## 3. Consequences

### 3.1 Positive
- A real, comparable public number for RuView (and everyone else) on MM-Fi / Wi-Pose, scored by a privacy- and latency-aware harness no other WiFi benchmark offers.
- Community flywheel: external models/adapters get ranked, feeding `ruvnet/wifi-densepose-pretrained`.
- Forces the harness to be reproducible-by-strangers, which strengthens internal release gating too.

### 3.2 Negative / Costs
- **New repo + HF Space to maintain**, incl. a scoring container and queue. Ongoing compute cost (mitigate: CPU smoke-score on submit, batched GPU full-score on a schedule).
- **Dataset licensing** must be cleared for hosting derived MM-Fi / Wi-Pose features (ADR-015 owns this; may require contacting dataset authors).
- **Abuse surface** (malicious model artifacts run in the scorer) — must sandbox the container (no network, read-only eval data, resource caps).

### 3.3 Neutral
- The scoring logic stays in `wifi-densepose-train`/`-cli`; the leaderboard is presentation only, so it does not bloat the core workspace.

---

## 4. Alternatives Considered

1. **Submit RuView to existing venues only (MM-Fi GitHub, Papers-with-Code).** Lower effort, but no privacy/latency axes, no live entry, and RuView doesn't own the standard. *Complementary, not exclusive — we should still post MM-Fi numbers.*
2. **A static numbers page in the RuView README.** Zero infra, but not multi-entrant and not a leaderboard.
3. **EvalAI / Kaggle competition.** Stronger anti-gaming infra, but heavyweight, time-boxed, and off-brand vs an always-open HF Space next to the model.

---

## 5. Open Questions

1. **Eval data hosting** — can we redistribute derived MM-Fi / Wi-Pose CSI features under their licenses, or must scoring pull the raw datasets the submitter cannot see? (Owner: ADR-015 follow-up.)
2. **Compute budget** — free HF CPU Space, ZeroGPU, or a self-hosted scorer on the GCloud A100/L4 fleet (`cognitum-20260110`)?
3. **Name lock** — confirm `aether-arena` vs `wifi-densepose-leaderboard`.
4. **Season cadence** — does the held-out split rotate monthly, and do we keep an all-time + per-season board?
5. **Privacy-leakage attack** — ship the membership-inference attacker (ADR-145 §10 is currently a *defined-but-unimplemented* metric) before launch, or launch with privacy as a "coming soon" axis?

---

## 6. Implementation Sketch (if accepted)

- **P1** — Stand up `ruvnet/aether-arena` repo + skeleton Gradio HF Space; define `ruview-arena.toml` submission contract; publish a **public smoke split** a stranger can score locally.
- **P2** — Containerize `wifi-densepose-cli score` as the pinned, sandboxed scorer (no network, read-only FS, caps); wire the signed append-only Parquet ledger + `determinism_gate`.
- **P3 — v0 LAUNCH (narrow).** Clear + load the private MM-Fi / Wi-Pose held-out split; activate **Presence, Pose, Edge-latency, Determinism** categories; seed the board with RuView's own `wifi-densepose-pretrained` baseline (honest current PCK@20). Tracking/Vitals optional. Privacy + Cross-room shown as **gated / coming soon**.
- **P4** — *(post-launch, gated)* Implement the ADR-145 §10 privacy-leakage membership-inference attacker; only then activate + rank the **Privacy** category and switch `privacy_factor` on in `arena_score`.
- **P5** — Assemble the multi-room split → activate **Cross-room**. Submit RuView's MM-Fi number to Papers-with-Code in parallel (alternative #1).

## 7. Acceptance Test (definition of done for v0)

v0 launches **only when a stranger can:**

1. **Submit** a model (artifact + `ruview-arena.toml`) through the Space with no insider help,
2. **Get a deterministic score** back (same model + same harness version → same numbers),
3. **See the signed row** appended to the public results ledger,
4. **Rerun the scorer locally** on the public *smoke* split and reproduce the logic, and
5. **Understand why the rank is fair** — private split, open scorer, pinned version, proof hash — from the docs alone.

If any of these five fails, v0 is not ready.

## 8. Suggested Announcement (draft)

> **I'm proposing AetherArena** — a public leaderboard for WiFi sensing, RF perception, and ambient intelligence.
>
> The problem with this field is not just model quality. It is *measurement* quality. Most WiFi-sensing work reports numbers against datasets with inconsistent splits, inconsistent metrics, and almost no accounting for latency, privacy leakage, reproducibility, or edge viability.
>
> AetherArena fixes that. Models are submitted, scored in a pinned sandboxed container against **private** held-out MM-Fi and Wi-Pose splits, and written to a **signed append-only** results ledger. The scoring engine reuses the same RuView harness we use internally: pose, presence, tracking, vitals, latency, cross-room degradation, deterministic proof hashes — and, once its attacker ships, privacy leakage.
>
> The goal is not to make RuView look good. The goal is to make the *category* measurable. If ambient intelligence is going to move from demos to infrastructure, it needs public numbers, reproducible commands, private eval splits, and failure modes that cannot be hidden.

### Strategic note — three layers of the credibility story

| Layer | Asset |
|-------|-------|
| Retrieval credibility | ruflo BEIR harness |
| Sensing credibility | **AetherArena (this ADR)** |
| Product credibility | RuView appliance + Arista-style deployments |

# ADR-132: HOMECORE-RECORDER — State History + Semantic Search

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-RECORDER** |
| **Crate** | `v2/crates/homecore-recorder` |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master — series map row ADR-132), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE state machine), [ADR-124](ADR-124-rvagent-mcp-ruvector-npm-integration.md) (ruvector/SENSE-BRIDGE), [ADR-130](ADR-130-homecore-rest-websocket-api.md) (HOMECORE-API query surface, downstream) |
| **Tracking issue** | [#800](https://github.com/ruvnet/RuView/pull/800) (HOMECORE intake) |

> **Documented retroactively (2026-06-12).** The `homecore-recorder` crate shipped under
> the ADR-126 series map (which planned an "ADR-132 HOMECORE-RECORDER") but the standalone
> ADR file was never written; the crate's `Cargo.toml`, `README.md`, `lib.rs`, `schema.rs`,
> and `semantic.rs` all cite "ADR-132". This ADR reverse-documents the decision that the
> shipped, tested code already embodies (ADR-164 Gap G3 / Coverage-Gaps Lens §A). It does
> **not** introduce new design; it records what is built. Date reflects the crate's intake
> era (first commit `e96ebaea8`, 2026-05-25); real-impl pass landed in `7c8071145`
> (2026-06-11).

---

## 1. Context

ADR-126 (the HOMECORE master) decided to reimplement Home Assistant (HA) natively in Rust.
HA persists every state change to a SQLite *recorder* database; downstream features
(history graphs, the logbook, long-term statistics, automation conditions that reference
past state) all read that store. HOMECORE therefore needs a durable state-history backbone.

Two forces shape the decision:

1. **Migration / coexistence.** Users adopting HOMECORE will have an existing HA
   `recorder` database. Reusing HA's on-disk schema (rather than inventing a new one) lets
   HOMECORE read an existing HA `home-assistant_v2.db` directly and lets HA-aware tooling
   read HOMECORE's store. This is the same trust boundary that `homecore-migrate`
   (ADR-165) handles for `.storage/*.json`.
2. **Semantic queries.** HA history is queried with SQL `BETWEEN`/`WHERE` clauses. The
   HOMECORE platform already carries ruvector (ADR-124) for vector search, so the recorder
   can additionally embed state changes and answer natural-language queries
   ("which kitchen devices were warm at 3 PM?") via k-NN — a capability HA does not have.

The recorder is the **durable-state surface**: if it is wrong, history, logbook, and
historical-condition automations are all wrong. ADR-164 flagged it as a CRITICAL coverage
gap precisely because such a load-bearing crate had no governing ADR.

## 2. Decision

Ship `homecore-recorder` as a SQLite state-history recorder with an HA-compatible schema
and an optional ruvector-backed semantic index, in three phases. P1 and P2 are built and
tested; P3 is planned.

### 2.1 Storage — SQLite with the HA recorder schema (P1, shipped)

- Persist via `sqlx` with the SQLite backend only (no Postgres, no TLS feature set).
- Mirror HA recorder **schema v48** so the store is bidirectionally readable
  (`src/schema.rs`):
  - `state_attributes` — shared attribute JSON blobs, deduped by an FNV-1a 64-bit hash
    stored as a signed `i64` (matches HA's dedup key);
  - `states` — one row per state write (`entity_id`, `state`, `attributes_id` FK,
    `last_changed_ts`/`last_updated_ts` as REAL Unix seconds, `context_id` UUID);
  - `events` — domain events (`event_type`, `event_data` JSON, `time_fired_ts`);
  - `recorder_runs` — boot/shutdown bookends for history-gap detection.
- All DDL uses `CREATE TABLE IF NOT EXISTS`, so schema application is idempotent and safe
  on every startup.
- Default persistence path `.homecore/home.db` (configurable).

### 2.2 Capture — listener on the HOMECORE event bus (P1, shipped)

- `RecorderListener` subscribes to the HOMECORE event bus (ADR-127) and captures
  `StateChanged` events, writing snapshots through `Recorder` (`src/listener.rs`,
  `src/db.rs`).
- A `DedupEngine` (`src/dedup.rs`) skips redundant writes when the state hash is unchanged,
  matching HA's stateful-listener behaviour.

### 2.3 Semantic search — ruvector HNSW (P2, shipped, feature-gated)

- Behind the `ruvector` Cargo feature, the `Recorder` additionally calls a `SemanticIndex`
  implementation (`src/semantic.rs`) that embeds state attributes and stores vectors in a
  `ruvector-core` HNSW index for k-NN search.
- P2 embeddings are **hash-based** (sha2) — a deliberate, honest placeholder. They give a
  working HNSW surface without claiming sentence-level semantic quality.
- When the feature is off, `NullSemanticIndex` satisfies the `SemanticIndex` trait bound
  with no allocation, so the structural recorder ships independently of ruvector.

### 2.4 Real sentence embeddings (P3, planned — not yet built)

- Replace the hash embeddings with ruvector-attention sentence embeddings (dim → 384). Not
  implemented; tracked as a follow-up. The README and `Cargo.toml` label this P3 explicitly.

### 2.5 Test evidence (as shipped)

- P1: 14 tests (`cargo test -p homecore-recorder --no-default-features`).
- P2: 20 tests (`cargo test -p homecore-recorder --features ruvector`).

## 3. Consequences

**Positive.**

- HA-schema compatibility makes migration (ADR-165) and coexistence cheap: HOMECORE can
  read an existing HA `recorder.db`, and any SQLite tool can read HOMECORE's history.
- The semantic index is **additive** and feature-gated: the durable structural recorder has
  no hard dependency on ruvector, so the storage backbone ships first.
- Standard SQLite means no proprietary export format; history is directly queryable.

**Negative / honest limits.**

- P2 semantic search uses **hash embeddings**, not real sentence embeddings — query quality
  is limited until P3. This is disclosed in the crate docs and here; it must not be cited as
  semantic-quality-validated.
- No per-crate benchmarks exist yet; the latency figures in the README
  (state-write p50 < 2 ms, semantic search < 10 ms on 1 M records) are design targets /
  estimates, **needs verification** with a criterion baseline.
- Pinning to HA schema v48 couples HOMECORE to a specific HA recorder schema generation;
  future HA schema bumps require an explicit migration step.

**Neutral.**

- This ADR governs the recorder crate only. The query/REST surface over recorder data is
  HOMECORE-API (ADR-130, P3); automation conditions on historical state are
  HOMECORE-automation (ADR-129, P3).

## 3a. Security review (2026-06, post-ADR-154–159 sweep)

A beyond-SOTA security review of `homecore-recorder` covered SQL injection, retention/purge
correctness, fail-closed write integrity, semantic-store NaN poisoning, and PII exposure.

**Confirmed clean (with evidence):**

- **SQL injection — clean.** Every query in `db.rs` uses bound `?` parameters; no user- or
  entity-influenceable value is interpolated into SQL via `format!`/concatenation. The only
  `format!` builds the `LIKE` *pattern* string, which is itself **bound** as a parameter with
  `ESCAPE '\\'` and `% _ \` escaping — so a metacharacter payload is matched literally. Pinned
  by `malicious_entity_id_is_stored_literally_not_executed` (a `'; DROP TABLE states; --` state
  value leaves the table intact and round-trips verbatim) and
  `like_metacharacters_in_query_are_literal_not_wildcards`.
- **NaN-index poisoning — structurally impossible.** Embeddings are SHA-256 → `i32` →
  `f32`; an `i32`→`f32` cast is always finite (never NaN/Inf), and an all-zero-digest is
  guarded by the `norm > 1e-10` check. Empty-index search, empty-string query, and `k=0` were
  probed and all return `Ok(0)` with no panic. (Unlike the calibration/vitals/geo paths, no raw
  sensor float ever reaches the index.)
- **Fail-closed writes.** A removal event returns `Ok(None)`; semantic-index failure is logged,
  not propagated, so it never blocks the durable SQLite write; `EntityId` parse failure falls
  back to a sentinel rather than panicking.

**Fixed (real bounding bugs):**

- **Memory-DoS — `get_state_history` was unbounded.** No `LIMIT`, so a wide time window over a
  high-frequency entity loaded an unbounded row set into memory. Now capped at
  `MAX_HISTORY_ROWS` (1,000,000); sibling search paths were already `k`-bounded.
- **Disk-DoS / documented-but-missing `purge`.** The README advertised `Recorder::purge`, but
  no retention path existed → unbounded disk growth. Added a **transactional** `purge(older_than)`
  with an **exclusive** cutoff (idempotent, no off-by-one) that deletes old `states`/`events` and
  GCs orphaned `state_attributes` blobs (dedup-shared blobs kept until their last referrer is gone).

`homecore-recorder` tests: 19 → 25 (`--no-default-features`) / 25 → 31 (`--features ruvector`),
0 failed. Python deterministic proof unchanged (recorder is off the signal proof path).

## 4. Links

- Crate: `v2/crates/homecore-recorder/` — `Cargo.toml`, `README.md`, `src/lib.rs`,
  `src/db.rs`, `src/schema.rs`, `src/dedup.rs`, `src/listener.rs`, `src/semantic.rs`.
- [ADR-126](ADR-126-ruview-native-ha-port-master.md) — HOMECORE master (series map: ADR-132 = HOMECORE-RECORDER).
- [ADR-165](ADR-165-homecore-migrate-from-home-assistant.md) — HOMECORE-MIGRATE (reads HA `.storage`; P2 exports a side-by-side recorder DB).
- [ADR-164](ADR-164-adr-corpus-gap-analysis.md) — gap analysis that surfaced this missing ADR (Gap G3).
- [Home Assistant Recorder integration](https://www.home-assistant.io/integrations/recorder/).

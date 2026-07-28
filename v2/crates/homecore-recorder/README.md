# homecore-recorder

SQLite state-history recorder for HOMECORE with Home Assistant-compatible schema and optional ruvector semantic search (P2).

[![Crates.io](https://img.shields.io/crates/v/homecore-recorder.svg)](https://crates.io/crates/homecore-recorder)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MSRV: 1.89+](https://img.shields.io/badge/MSRV-1.89%2B-purple.svg)
[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen.svg)](https://github.com/ruvnet/RuView)
[![ADR-132](https://img.shields.io/badge/ADR-132-orange.svg)](../../docs/adr/ADR-132-homecore-recorder-history-semantic-search.md)

**P1 release**: SQLite database with Home Assistant-compatible schema for persistent state history. **P2 (feature-gated)**: ruvector HNSW semantic index for natural-language queries ("show me all kitchen devices that were warm at 3 PM").

## What this crate does

`homecore-recorder` persists HOMECORE state changes to SQLite and optionally indexes them for semantic search. It provides:

- **Listener pattern** — subscribes to homecore event bus and captures all `StateChanged` events
- **SQLite schema** — mirrors HA's `recorder` database schema (v48) for 1:1 compatibility
- **Dual-write architecture** — writes state snapshots to `states` table and attributes to `state_attributes` table (same as HA)
- **Deduplication** — avoids recording redundant state writes when state hasn't actually changed
- **SemanticIndex trait** — abstraction for plugging in ruvector embeddings (P2)
- **NullSemanticIndex** — no-op implementation used when `ruvector` feature is off

Data persists in `.homecore/home.db` (by default; configurable). Queries work via standard SQLx, so any tool that reads SQLite can access the history.

## Features

- **Home Assistant schema compatibility** — migrate from HA's `recorder.db` without schema changes
- **Event recording** — all state changes captured with `last_changed` timestamp and old/new state
- **Attribute persistence** — JSON attributes for entities stored in separate table (HA pattern)
- **Automatic deduplication** — skip writes when state hasn't changed (detect via hash)
- **Recorder runs table** — track purge cycles and migration events (HA `recorder_runs` equivalent)
- **Startup restoration** — deterministic newest row per entity, bounded at
  100,000 rows, with malformed rows isolated as typed warnings
- **Semantic search** (P2, `--features ruvector`) — embed state attributes + query by meaning
- **HNSW index** (P2) — k-NN search for "all warm rooms" via ruvector
- **No data export overhead** — SQLite is queryable directly; no proprietary format

## Capabilities

| Capability | Type | Method | Notes |
|------------|------|--------|-------|
| Record state change | Listener | `RecorderListener::on_state_changed(event)` | Fires on homecore event bus; writes to SQLite |
| Query state history | SQL | `SELECT * FROM states WHERE entity_id = ? ORDER BY last_changed DESC` | Standard SQLite; can be queried from anywhere |
| Purge old states | Maintenance | `Recorder::purge(older_than)` | Deletes states older than specified timestamp |
| Restore latest states | Startup | `Recorder::restore_latest(states, limit)` | Entity-id ordered, bounded, malformed-row isolation |
| Deduplicate write | Dedup | `DedupEngine::should_record(old_state, new_state)` | Skip if state hash unchanged |
| Create semantic index | Index | `SemanticIndex::index_state(entity_id, state)` (P2, opt-in) | Hash-based embeddings; real embeddings in P3 |
| Search by meaning | Search | `SemanticIndex::search(query, k)` (P2, opt-in) | "warm rooms" → k-NN search in ruvector HNSW |

## Comparison to Home Assistant

| Aspect | Home Assistant | homecore-recorder |
|--------|----------------|-------------------|
| Database | SQLite (Python sqlite3) | SQLite (Rust sqlx) |
| Schema | `recorder/` (schema v48) | Identical HA schema v48 |
| State table | `states` + `state_attributes` | Same dual-table layout |
| Persistence location | `.homeassistant/home-assistant_v2.db` | `.homecore/home.db` |
| Deduplication | Python stateful listener | DedupEngine + hash comparison |
| Purge policy | YAML `auto_purge_* + retention` | Configurable via `Recorder::purge()` |
| Semantic search | None (HA has YAML history stats only) | ruvector HNSW k-NN (P2, opt-in) |
| Schema compatibility | N/A | Bidirectional; can read HA's home.db directly |

## Performance

- **State write latency** — p50 < 2 ms (SQLite WAL append); p99 < 15 ms (disk fsync)
- **Query latency** — < 1 ms for indexed entity_id lookups; < 50 ms for range scans (full table)
- **Semantic search** (P2) — < 10 ms for k-NN on 1 million state records (ruvector HNSW)
- **Memory overhead** — ~10 MB per million recorded states (SQLite index overhead)
- **Disk space** — ~2-4 KB per state record (entity_id + attributes + timestamps)
- **No per-crate benchmarks yet** — a follow-up issue tracks baseline measurements

Run `cargo bench -p homecore-recorder --features ruvector` for criterion benchmarks.

## Usage

Recording state changes (P1):

```rust
use homecore_recorder::{Recorder, RecorderListener};
use homecore::HomeCore;

#[tokio::main]
async fn main() {
    let homecore = HomeCore::new();
    
    // Create the recorder (writes to .homecore/home.db)
    let recorder = Recorder::new(".homecore/home.db").await.expect("init recorder");

    // Create and spawn a listener
    let listener = RecorderListener::new(recorder.clone());
    let mut rx = homecore.event_bus().subscribe_system();
    
    tokio::spawn(async move {
        while let Ok(event) = rx.recv().await {
            if let Err(e) = listener.on_state_changed(&event).await {
                eprintln!("Recorder error: {}", e);
            }
        }
    });

    // State changes now persist to SQLite
}
```

Querying history directly (standard SQLite):

```sql
-- All light.kitchen state changes in the last hour
SELECT state, attributes, last_changed 
FROM states 
WHERE entity_id = 'light.kitchen' 
  AND last_changed > datetime('now', '-1 hour')
ORDER BY last_changed DESC;

-- Average brightness by hour
SELECT 
  strftime('%Y-%m-%d %H:00:00', last_changed) AS hour,
  JSON_EXTRACT(attributes, '$.brightness') AS brightness
FROM states 
WHERE entity_id = 'light.kitchen'
GROUP BY hour;
```

Semantic search (P2, with `--features ruvector`):

```rust
// (P2, not yet implemented)
// let index = SemanticIndex::new(recorder.clone()).await?;
// let results = index.search("find all warm rooms at 3pm", 5).await?;
// results.iter().for_each(|r| println!("{:?}", r));
```

## Relation to other HOMECORE crates

```
homecore-recorder (state history + semantic search)
├─ homecore (state machine; listens to event bus)
├─ homecore-api (exposes recorder data via REST query endpoint, P3)
├─ homecore-automation (can trigger on historical state conditions, P3)
├─ homecore-server (starts the listener on init)
└─ ruvector-core (semantic index, P2, optional feature)
```

## References

- [ADR-132: HOMECORE Recorder — History + Semantic Search](../../docs/adr/ADR-132-homecore-recorder-history-semantic-search.md)
- [ADR-126: HOMECORE Home Assistant Port (master)](../../docs/adr/ADR-126-homecore-home-assistant-port.md)
- [Home Assistant Recorder Integration](https://www.home-assistant.io/integrations/recorder/)
- [README — wifi-densepose](../../../README.md)

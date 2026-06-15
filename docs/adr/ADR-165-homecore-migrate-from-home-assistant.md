# ADR-165: HOMECORE-MIGRATE — Migration Tooling from Python Home Assistant

| Field | Value |
|-------|-------|
| **Status** | Accepted — P1 scaffold (full conversion deferred to P2) |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-MIGRATE** |
| **Crate** | `v2/crates/homecore-migrate` |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master — series map row "ADR-134 HOMECORE-MIGRATE"), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE), [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) (HOMECORE-RECORDER — P2 side-by-side export target) |
| **Tracking issue** | [#800](https://github.com/ruvnet/RuView/pull/800) (HOMECORE intake) |

> **Number-collision resolution (2026-06-12).** The HOMECORE series in ADR-126 §4 planned
> "ADR-134 = HOMECORE-MIGRATE", and the `homecore-migrate` crate cites "ADR-134" throughout.
> But the on-disk `ADR-134-csi-to-cir-time-domain-multipath.md` is a **different, unrelated
> decision** (First-Class CIR Support, a signal-processing tier). The migrate crate was
> therefore governed by a phantom identity (ADR-164 Gap G3 / Coverage-Gaps Lens §A). This
> ADR takes the next free number (**165**) and becomes the real governing record for
> HOMECORE-MIGRATE; the `ADR-134` references inside `v2/crates/homecore-migrate/` are
> repointed to ADR-165. The real ADR-134 (CIR) is untouched. ADR-126's series-map row still
> labels the *role* "ADR-134 HOMECORE-MIGRATE" for historical traceability; that registry
> renumber is owner-gated and left for the follow-up. This ADR reverse-documents the shipped
> P1 scaffold; it introduces no new design.

---

## 1. Context

ADR-126 decided to reimplement Home Assistant (HA) natively in Rust. A user adopting
HOMECORE has an existing HA install whose configuration lives in two places on disk:

- `.storage/*.json` — versioned JSON envelopes (`{ version, minor_version, data }`) holding
  the entity registry, device registry, and config entries;
- top-level YAML — `secrets.yaml`, `automations.yaml`.

To migrate, HOMECORE must read this foreign, **untrusted** on-disk state. It is untrusted in
the security sense: the schema can drift between HA releases, and silently mis-parsing a
registry would corrupt the imported home. ADR-164 flagged this as a CRITICAL coverage gap —
a data-integrity-sensitive importer governed by a non-existent ADR identity.

The decision an ADR must pin here is the **trust boundary and import contract**: which HA
files are read, how schema versions are validated, and what happens on an unknown version.

## 2. Decision

Ship `homecore-migrate` as a CLI + library that reads an existing HA filesystem and imports
its configuration into HOMECORE. P1 is a **scaffold**: it parses and inspects everything and
converts the entity registry; full conversion of the remaining artifacts is deferred to P2.

### 2.1 Storage reader + versioned format gate (P1, shipped)

- `HaStorageDir` / `HaStorageEnvelope` read HA's `.storage/` directory; `read_envelope(path)`
  deserializes a `.storage/*.json` envelope (`src/storage.rs`).
- Versioned parsers live under `storage_format::v<N>` (e.g. `v13` for the entity registry)
  (`src/storage_format/`).
- **Schema-version validation is the load-bearing safety rule (§6 Q5 of this ADR):** an
  unknown `minor_version` is a **hard error** (`MigrateError::UnsupportedSchemaVersion`),
  never a silent best-effort parse. Better to refuse than to corrupt.

### 2.2 Per-artifact parsers (P1, shipped)

- `entity_registry::load()` — `core.entity_registry` → `Vec<homecore::EntityEntry>`
  (ready for import).
- `device_registry::load()` — `core.device_registry` → `Vec<DeviceImport>` (P1 diagnostic;
  full conversion P2).
- `config_entries::load()` — `core.config_entries` → domain counts + integration names
  (the format is undocumented per §6 Q5; treated diagnostically).
- `secrets::load_secrets()` — `secrets.yaml` → `HashMap<String, String>` (resolution P2).
- `automations::load()` — `automations.yaml` → count + ID/alias list (conversion P2).

### 2.3 CLI (P1, shipped)

- `homecore-migrate inspect <ha-dir>` previews what will be migrated (entity/device/config
  counts, redacted secret/automation lists) (`src/cli.rs`, `src/main.rs`).
- `import-entities` and `export-for-sidecar` are declared but their full behaviour is P2.

### 2.4 Structured errors (P1, shipped)

- `MigrateError` carries context (`path`, line/field) for I/O, JSON, YAML, missing-field,
  unsupported-schema-version, and entity-id parse failures (`src/lib.rs`).
- **Secret-leak hardening (security review, 2026-06).** `secrets.yaml` parse failures must
  NOT use the generic `MigrateError::YamlParse { source }` variant: `serde_yaml`'s message
  for a typed-tag coercion error (e.g. `port: !!int <value>`) embeds the offending scalar
  verbatim (`invalid value: string "<the-secret-value>"`), and that error propagates through
  the `InspectSecrets` CLI path to stderr — leaking a secret value despite the CLI's
  deliberate `<redacted>` design. `read_secrets` now maps such failures to a dedicated
  redacting variant `MigrateError::SecretsParse { path, line, column }` that carries only the
  file path and a coarse location (`serde_yaml::Error::location()`), never the scalar content.
  Pinned by `secrets::tests::malformed_secrets_error_never_contains_secret_value` (asserts the
  rendered error **and its full `#[source]` chain** never contain the secret value).
  **Review dimensions confirmed clean with evidence:** source is never mutated (no
  `fs::write`/`remove`/`create` anywhere — P1 reads source, writes nothing); paths are
  user-supplied dirs joined with fixed filenames (no `..`/absolute traversal beyond the
  user's own privileges); malformed/typed/truncated `.storage` JSON and YAML **error, never
  panic** (every production `unwrap`/`expect` is test-only); unknown schema `minor_version`
  hard-errors fail-closed; no SQL/shell/path injection surface (the tool emits diagnostics
  only, persists nothing in P1).

### 2.5 Deferred to P2+ (NOT built — honestly labelled)

- Convert `config_entries` → HOMECORE plugin manifests.
- Convert `automations.yaml` → `homecore-automation` YAML.
- Side-by-side runtime mode (requires `homecore-recorder`, ADR-132; behind the `recorder`
  Cargo feature, currently a no-op stub).
- `!secret` reference resolution in non-secrets YAML files.

### 2.6 Test evidence (as shipped)

- 21 tests (`cargo test -p homecore-migrate`) — 19 as originally shipped plus 2 added by the
  2026-06 security review (`secrets::tests::malformed_secrets_error_never_contains_secret_value`,
  `malformed_secrets_error_reports_location`).

## 3. Consequences

**Positive.**

- The trust boundary is explicit: unknown HA schema versions are rejected, not guessed, so a
  schema drift fails loudly instead of corrupting an imported home.
- Reusing HA's own `.storage` and YAML formats means no intermediate export step; the tool
  reads a live HA install directly.
- P1 `inspect` gives users a no-risk dry run before any write.

**Negative / honest limits.**

- P1 is a **scaffold**: only the entity registry is conversion-ready. Device registry,
  config-entry→plugin, automation, and secret-resolution conversions are P2 and **not yet
  built** — the Status field and crate docs say so.
- The side-by-side recorder export depends on ADR-132 and is currently a feature-gated
  no-op.
- Performance figures in the README (envelope parse < 5 ms, 1 000-entity load < 50 ms) are
  estimates, **needs verification** with a benchmark.

**Neutral.**

- This resolves only the *identity* of the migrate decision (134→165). The broader 6-way
  duplicate-number cleanup (incl. ADR-126's series-map registry row) is owner-gated.

## 4. Links

- Crate: `v2/crates/homecore-migrate/` — `Cargo.toml`, `README.md`, `src/lib.rs`,
  `src/storage.rs`, `src/storage_format/`, `src/entity_registry.rs`,
  `src/device_registry.rs`, `src/config_entries.rs`, `src/secrets.rs`,
  `src/automations.rs`, `src/cli.rs`, `src/main.rs`.
- [ADR-126](ADR-126-ruview-native-ha-port-master.md) — HOMECORE master (series map: HOMECORE-MIGRATE).
- [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) — HOMECORE-RECORDER (P2 side-by-side export target).
- [ADR-134](ADR-134-csi-to-cir-time-domain-multipath.md) — First-Class CIR Support (the *unrelated* decision the crate was mistakenly citing).
- [ADR-164](ADR-164-adr-corpus-gap-analysis.md) — gap analysis that surfaced this collision (Gap G3).
- [Home Assistant `.storage` format](https://developers.home-assistant.io/docs/storage/).

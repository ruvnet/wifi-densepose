//! Versioned format parsers for HA `.storage/` files.
//!
//! Each sub-module handles one `(version, minor_version)` generation of a
//! particular storage key. Adding support for a new HA schema version means
//! adding a new `v<N>.rs` module; the dispatch function in each parser module
//! routes to the right implementation.
//!
//! Per ADR-165 §6 Q5: unknown `minor_version` values produce a hard
//! `MigrateError::UnsupportedSchemaVersion` — we do NOT silently fall back
//! to an older parser, because schema changes can be load-bearing (new fields,
//! renamed keys, semantic reinterpretations).

pub mod v13;

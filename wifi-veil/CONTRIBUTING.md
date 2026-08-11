# Contributing to WiFi Veil

Thanks for your interest. WiFi Veil is a privacy-defense project with a strict
honesty and safety contract — please read this before opening a PR.

## Non-negotiable rules

- **Compliant waveform controls only — never jamming.** Do not add, suggest, or
  scaffold interference-based "defenses." Every control must shape the node's
  *own* standards-conformant emission and preserve its energy.
- **Never present WiFi sensing as camera-grade.** Accuracy/defense statements
  must be tagged `SYNTHETIC`, `CLAIMED`, or `MEASURED`. A number is only
  `MEASURED` with a reproducer; hardware claims require a captured real-silicon
  log. Everything in this repo today is `SYNTHETIC / L0`.
- **The proof witness is load-bearing.** The default scene is pinned by a
  deterministic FNV-1a witness (`src/proof.rs`). If a change intentionally moves
  it, re-pin the constant *in the same PR* and explain why; an accidental change
  is a failing test, not a witness to bump.

## Development

The Rust crate is dependency-free and builds offline.

```bash
cargo test                        # 43 tests + the pinned witness
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo build --lib --target wasm32-unknown-unknown   # WASM leaf must stay green

cd firmware/core && make test     # portable C core host test
node harness/bin/cli.js guidance --topic overview   # harness (dependency-free)
```

Run the honesty / anti-slop guard before pushing (CI runs it too):

```bash
bash scripts/ci-guard.sh
```

It statically enforces the invariants that keep this project honest: no
telemetry / build artifacts / lockfile / scratch files committed; no debug or
mock-probe markers in source; the `SYNTHETIC` evidence label present on every
firmware provider README; the "never jamming" compliance disclaimer present; no
dishonest hardware-validation claims (honest negated/`TODO(hw)` mentions are
fine); and no stale monorepo identifiers in the code surface.

CI (`.github/workflows/ci.yml`) runs the same gates. Keep changes the smallest
coherent unit, read before editing, and never commit telemetry (`.claude-flow/`),
build artifacts, credentials, or CSI/person data.

## Architecture decisions

Substantive design changes should reference or add an ADR under
[`docs/adr/`](docs/adr/). Treat source, tests, and accepted ADRs as
authoritative over comments and generated text.

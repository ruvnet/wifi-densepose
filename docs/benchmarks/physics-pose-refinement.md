# Physics pose refinement evidence ledger

ADR-323 performance and accuracy targets are gates, not measured claims. Append
rows; never replace prior measurements. Every row must identify the repository
commit, lockfile hash, Rust toolchain, target, engine/features, configuration
hash, corpus/split hash, command, sample count, and evidence label.

## Runtime measurements

| Date | Commit | Lock SHA-256 | Target/toolchain | Engine/config | Tracks | Samples | p50 | p95 | p99/max | RSS delta | Evidence | Reproducer |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 2026-08-15 | `de27336` + uncommitted ADR-323 changes | `552737eab9092b59ea9dd2b2caf68389f0b0966679f0fbb33ff2b1b3d42e2668` | Windows x86_64, Intel Core Ultra 9 285H, rustc 1.91.1 | deterministic kinematic shadow, config `ef3cf581f75124c1d45a8d6bedcef32e4d1bacb39ee0dfcd4e520171fda2d8cf` | 1 | 20,000 | 0.0080 ms | 0.0097 ms | 0.0195/0.5465 ms | not measured | **MEASURED**, local host only; not Pi 5 evidence | `cargo run --release -p wifi-densepose-physics --example latency_probe -- 20000` |
| 2026-08-15 | `de27336` + uncommitted ADR-323 changes | `552737eab9092b59ea9dd2b2caf68389f0b0966679f0fbb33ff2b1b3d42e2668` | Windows x86_64, Intel Core Ultra 9 285H, rustc 1.91.1 | deterministic kinematic shadow after final local optimization, same config | 1 | 20,000 | 0.0075 ms | 0.0084 ms | 0.0117/0.1579 ms | not measured | **MEASURED**, local host only; not Pi 5 evidence | same release probe command |
| 2026-08-15 | `de27336` + uncommitted ADR-323 changes | `552737eab9092b59ea9dd2b2caf68389f0b0966679f0fbb33ff2b1b3d42e2668` | Windows x86_64, Intel Core Ultra 9 285H, rustc 1.91.1 | final deterministic kinematic shadow, config `a44dc696234f31eda54cd4b436bc2d2c69b9638565b729ac9f07435cedfd0dcc` | 1 | 20,000 | 0.0071 ms | 0.0084 ms | 0.0147/1.5994 ms | not measured | **MEASURED**, local host only; not Pi 5 evidence | same release probe command |

The probe measures a warm, one-track `PhysicsEngine::process` call. It excludes
transport, publication, resident-memory delta, dynamics, and learned inference.
It is not evidence for the Pi 5 gate.

Criterion separately measured `kinematic_one_track` at
`[11.911, 12.757, 14.069] us` across 100 samples (approximately 369,000 timed
iterations). That benchmark includes observation construction and canonical
hashing in the timed routine and uses fresh engine state; it is **MEASURED** on
the same local host and is not a percentile or Pi 5 claim.

## Accuracy measurements

| Date | Commit | Corpus/split | Variant | Coverage | MPJPE | PCK threshold/result | Foot slide | Jerk | Fall/prone delta | Evidence |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|

No measured accuracy evidence has been recorded. The deterministic tests are
L0/SYNTHETIC contract evidence only and cannot satisfy G2.

## Validation and supply-chain record

- The default dependency graph is checked to exclude Burn, Rapier, Tch, and
  ONNX Runtime. Dynamics and learned backends remain opt-in.
- Burn CPU serialization/inference tests pass on the authoring host only with
  Cargo's `--ignore-rust-version`; the resolved CubeCL graph requires Rust 1.92.
  The workspace file pins Rust 1.89 and the host provides Rust 1.91.1. This is
  diagnostic, not release approval.
- `cargo audit 0.22.1` used RustSec database commit
  `69f93cf294852cfa9b53751f4ca86de3283dd290` (feed timestamp 2026-08-12).
  ADR-323 updates remove resolved advisories in `event-listener`, `rkyv`, and
  `wasmtime`. The workspace still has five advisories in pre-existing
  `quick-xml` and `rsa` dependency paths; the default physics graph contains
  none of them. The optional Burn training graph includes yanked `spin 0.9.8`.
- `cargo-deny` is not installed on the authoring host, so the required license
  and policy gate is not claimed complete.
- Strict Clippy passes with warnings denied for core/physics default and
  dynamics builds, the diagnostic learned-CPU build, and the Cog itself with
  dependency linting excluded. Focused core, physics, dynamics, learned, Cog,
  sensing-server adapter/live-audit/HTTP, schema, golden, strict-split,
  feature-boundary, and fuzz-build checks pass.
- The repository-wide rustfmt gate is already red across unrelated crates. The
  sensing-server library has existing warning debt, and unscoped Cog Clippy is
  blocked by existing `wifi-densepose-ruvector` warnings. The prescribed
  `cargo test --workspace --no-default-features` did not reach a terminal result
  in either a 904-second cold or 604-second warm serial run on this Windows
  host. None of these broader gates is represented as green.
- The standalone fuzz lock SHA-256 is
  `d386c4edb130bb6b2d1a4ef77334c78e25e0695e90a9d97c01284876acb8c2c6`.

## Required commands

```text
cargo bench -p wifi-densepose-physics
node scripts/pose-physics/verify-feature-boundary.mjs
bash scripts/verify-pose-physics-splits.sh <manifest.json>
bash scripts/replay-pose-physics-golden.sh <golden-results.jsonl>
```

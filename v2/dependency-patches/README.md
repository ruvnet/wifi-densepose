# ZIP9 compatibility patches

These are narrow source patches to the existing crates.io Candle0.4.1, Candle0.9.2, tch0.24.0 and torch-sys0.24.0 releases. The model/framework versions, numerical operators and public APIs are retained. Root Cargo patches redirect those exact package versions to these sources; every ZIP edge resolves to patched ZIP9.0.5 in the root lockfile.

`patches.json` records downloaded `.crate` SHA256, each retained upstream-file hash, patched-file hash and the changed-file list. `compatibility.patch` contains the complete review delta. Sources came from `https://static.crates.io/crates/<name>/<name>-<version>.crate`. MIT/Apache licences remain in each directory. Cargo.lock and cargo VCS metadata were omitted; tch's 87MB `third_party` model/PyTorch test snapshot was omitted because it is not a Rust library or build input. Its asset-dependent upstream tests are not represented as passed. These snapshots are not independently published packages.

## Changes

- Four normalized Cargo manifests request ZIP9; old manifests are retained as upstream provenance only.
- NPZ name decoding propagates ZIP9's checked UTF8 result; names are borrowed from Cow without truncation. NPZ writers use SimpleFileOptions.
- PTH file-name iteration collects checked results before reading entries.
- The optional libtorch download build script uses ZIP9's checked extraction instead of joining mangled names manually. This is a build-time path, not a default server runtime.
- No advisory ignore, audit gate change, framework-major migration or tensor algorithm change.

## Verification

Run `python3 scripts/verify_zip_parent_patches.py`, then root `cargo audit --file v2/Cargo.lock`. Focused tests include the actual current Candle consumer and NPZ Unicode/nested-name/value roundtrip, sensing-server companion/TLS, NPY loader and Wasmtime sandbox tests. `cargo check -p wifi-densepose-train --features tch-backend,tch/doc-only` checks tch and torch-sys Rust/build-script compatibility without downloading or claiming libtorch execution.

The legacy optional Candle0.4 backend cannot currently compile against the main workspace's half2.7/rand0.9 combination: its unchanged random operators require rand0.8. Pinning half2.3 in the full workspace conflicts with burn's half2.7 requirement, so no such pin is applied. `cargo test --manifest-path v2/dependency-patches/compat-tests/Cargo.toml` exercises the patched Candle0.4 NPZ API in an isolated compatible half2.3 environment. Its own lockfile is retained. This does not claim that the legacy optional backend is qualified in the main workspace. Full PyTorch/CUDA/model numerical qualification remains unexecuted; the default Mac server does not include those optional backends.

Remove each snapshot when an upstream compatible package release supplies ZIP>=9. Source changes require the same focused format, build and integrity review. The separate Apple helper binary must be rebuilt from the clean source commit and verified against its embedded manifest before distribution.

## Review disposition (2026-10-22)

This remediation is separated from companion PR #2249 and is based directly on
main. It is a proposal for review, not an approved vendoring policy. Companion
pairing, sensor preview and MAT routing do not require the copied ML sources.
Removing the remediation restores inherited dependency advisories; it does not
make those advisories acceptable or waive the hard audit gate.

Alternatives checked against published registry dependency metadata on this date:

- Latest Candle core 0.11.0 still requires ZIP ^8.6.0; latest tch/torch-sys 0.26.0
  still require ZIP ^0.6. A framework upgrade alone therefore does not supply the
  ZIP9 fix and would add separate model/API qualification.
- Updating only TLS/QUIC/Wasmtime within their existing compatible ranges and
  removing ndarray-npy's unused NPZ feature is a smaller independently reviewable
  change. It cannot clear the remaining Candle/tch ZIP findings by itself.
- Disabling optional ML features is not a lockfile-audit fix: Cargo resolves optional
  workspace packages too. A separate minimal helper workspace/lock could isolate
  shipped dependencies, but needs its own build, audit and release contract; it
  must not silently replace the existing whole-workspace audit.
- Upstream compatible parent releases or maintained, immutable parent patches
  could avoid in-tree snapshots. Neither is established as a qualified replacement
  here. A ZIP version override alone is incompatible with existing callers; the
  checked-in compatibility patch shows the required API adaptations.

Recommended disposition: hold this full snapshot proposal for maintainer review;
prefer a separate small compatible dependency update first, then choose an
upstream patch or explicitly maintained ZIP compatibility strategy. No dependency
on this PR is imposed on #2249's source branch. Merge/release readiness still
requires a truthful disposition of the existing audit failures.

Registry evidence: https://crates.io/api/v1/crates/candle-core/0.11.0/dependencies,
https://crates.io/api/v1/crates/tch/0.26.0/dependencies,
https://crates.io/api/v1/crates/torch-sys/0.26.0/dependencies.

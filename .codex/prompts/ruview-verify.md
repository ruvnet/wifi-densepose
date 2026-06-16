# /ruview-verify — run the RuView trust pipeline

Verify a RuView build. Scope: `$ARGUMENTS` (one of `tests`, `proof`, `bundle`, `all`; default `all`).

1. **tests** — `cd v2 && cargo test --workspace --no-default-features` → must be 1,400+ passed, 0 failed (~2 min). Single-crate: `cargo test -p wifi-densepose-signal --no-default-features`, etc.
2. **proof** — `cd .. && python archive/v1/data/proof/verify.py` → must print `VERDICT: PASS`. If a hash mismatch from a legitimate numpy/scipy bump: `python archive/v1/data/proof/verify.py --generate-hash`, then re-run. Optional: `cd archive/v1 && python -m pytest tests/ -x -q`.
3. **bundle** — `bash scripts/generate-witness-bundle.sh` produces `dist/witness-bundle-ADR028-<sha>.tar.gz` (WITNESS-LOG-028.md, ADR-028 audit, proof, rust test log, firmware hash manifest, crate versions, VERIFY.sh). Then `cd dist/witness-bundle-ADR028-*/ && bash VERIFY.sh` → must be 7/7 PASS.
4. **all** — do 1→3 in order.

If this follows a code change, walk the pre-merge checklist from `CLAUDE.md`: Rust tests pass; Python proof passes; README updated if scope changed; CLAUDE.md updated if scope changed; CHANGELOG `[Unreleased]` entry; `docs/user-guide.md` updated if new data sources/CLI flags/setup; ADR count bumped in README if a new ADR added; witness bundle regenerated if tests/proof hash changed; Docker image rebuilt only if Dockerfile/deps/runtime changed; crate publishing only if a published crate's public API changed (publish in dependency order — see CLAUDE.md); `.gitignore` updated for new artifacts; security review for new hardware/network-boundary modules.

For security-related changes also run `npx @claude-flow/cli@latest security scan`. QEMU firmware CI (ADR-061): local helpers `scripts/qemu-esp32s3-test.sh`, `qemu-mesh-test.sh`, `qemu-chaos-test.sh`, `install-qemu.sh`.

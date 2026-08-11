# Changelog

All notable changes to WiFi Veil are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Standalone repository layout extracted from the RuView monorepo: the
  dependency-free `wifi-veil` Rust crate at the repo root, the `veil` terminal
  TUI, the self-contained WiFi Veil Console (`ui/veil-console.html`), the
  end-to-end `firmware/` hardware program (host-validated portable C core plus
  per-provider scaffolds), and the `wifi-veil-harness` npm MetaHarness.
- Continuous integration: Rust build/test/clippy/fmt + WASM leaf build, the C
  core host test, and the harness smoke run.

### Notes
- All defense figures remain `SYNTHETIC` / evidence level **L0**. No result is
  `MEASURED` until a two-node hardware capture with a witness exists (roadmap
  **P5**). Compliant waveform controls only — never jamming.

## [0.1.0]
- Initial VEIL reference: keyed Givens-rotation shield, passive re-identification
  attacker, throughput/compliance models, optimizer, and a pinned deterministic
  proof witness (ADR-288). npm MetaHarness (ADR-289). E2E hardware program and
  portable C core (ADR-290).

---
name: code-review
description: Authoritative review guidelines for the RuView camera-free RF perception system, including Rust v2 crates, ESP32 firmware, metaharnesses, and CI pipelines. Use when reviewing pull requests or evaluating code changes in this repository.
---

# RuView Code Review Guidelines

When reviewing pull requests in the RuView repository, evaluate all changes against the repository operating contract (see `AGENTS.md` and `CLAUDE.md`):

## 1. Security & Operating Boundary
- **No Leaked Data or Secrets**: Never allow commits with secrets, `.env` files, raw LLM transcripts, private memory indexes, or raw CSI/personal data.
- **Least Authority & Input Validation**: Validate process, path, network, hardware, FFI, and MCP inputs at every boundary. Disallow sandbox or permission bypass flags.
- **Minimal Coherent Scope**: Verify that changes are tightly scoped and do not overwrite or reset unrelated edits in the worktree.

## 2. Technical Quality Bars by Subsystem

### A. Rust Production Code (`v2/crates/`)
- **Safety & Error Handling**: Favor safe Rust. Any `unsafe` block must be strictly justified with safety invariants documented. Handle errors with `Result`/`Option` rather than indiscriminate panics or `.unwrap()`.
- **Concurrency & Async**: Verify lock ordering and async bounds to prevent deadlocks and async cancellation leaks.
- **Testing**: Ensure unit and integration tests are added or updated. Must pass `cargo test --workspace --no-default-features`.

### B. Firmware (`firmware/esp32-csi-node/`)
- **Real Hardware Evidence**: A passing build or simulator is not hardware validation. Any claim of hardware functionality requires captured boot/runtime log evidence from real ESP32-S3/C6 silicon.
- **Resource Constraints**: Respect heap and stack limits in FreeRTOS tasks. Minimize blocking in Wi-Fi and CSI capture callbacks.

### C. Contributor Harnesses (`harness/ruview/`, `harness/homecore/`)
- **Dependency Hygiene**: Metaharnesses must maintain zero runtime dependencies where mandated (ADR-283 / ADR-285).
- **Manifest & Security**: Verify that modified packaged files pass `npm test`, `npm run test:security`, `npm run brain:verify`, and `npm run manifest:verify`.
- **Learning Promotion**: Ensure candidate flywheel learnings are proposal-only and never self-promoted without explicit human review.

### D. Perception & Claims Validation (`archive/v1/`, docs)
- **Claim Taxonomy**: All accuracy and performance claims must be explicitly tagged as `MEASURED` (with a reproducer), `CLAIMED`, or `SYNTHETIC`.
- **Baseline Rigor**: Pose PCK claims require the mean-pose baseline and a leakage-free held-out split. Never present WiFi sensing as camera-grade.
- **Deterministic Proof**: If modifying `archive/v1/`, the proof script must pass with `VERDICT: PASS`.

## 3. Review Output Format
Provide structured, evidence-based review feedback:
1. **Recommendation**: `Approval recommended` or `Changes requested`.
2. **Subsystem Summary**: Note affected areas (`v2`, `firmware`, `harness`, `workflows`).
3. **Findings**: Group findings with severity tags:
   - `CRITICAL`: Security vulnerabilities, secret leaks, breaking regressions, invalid claim tags.
   - `MAJOR`: Missing input validation, memory leaks, unhandled errors, lack of hardware proof.
   - `MINOR` / `SUGGESTION`: Code style, documentation, optimization.
4. **Actionable Guidance**: Include reproducible test commands or code replacements.

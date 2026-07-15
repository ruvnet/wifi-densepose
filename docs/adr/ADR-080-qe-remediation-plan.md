# ADR-080: QE Analysis Remediation Plan

- **Status:** Proposed — P0 security findings #1–#3 **RESOLVED** on the shipped Rust sensing-server boundary (2026-06-13; closes ADR-164 G11)
- **Date:** 2026-04-06
- **Source:** [QE Analysis Gist (2026-04-05)](https://gist.github.com/proffesor-for-testing/a6b84d7a4e26b7bbef0cf12f932925b7)
- **Full Reports:** [proffesor-for-testing/RuView `qe-reports` branch](https://github.com/proffesor-for-testing/RuView/tree/qe-reports/docs/qe-reports)

## Context

An 8-agent QE swarm analyzed ~305K lines across Rust, Python, C firmware, and TypeScript on 2026-04-05. The overall score was **55/100 (C+) — Quality Gate FAILED**. This ADR captures the findings and establishes a remediation plan.

## Decision

Address the 15 prioritized issues from the QE analysis in three waves: P0 (immediate), P1 (this sprint), P2 (this quarter).

## Security P0 closure note (2026-06-13) — Rust sensing-server boundary

The three P0 security findings below were logged against the **Python v1** API
(`archive/v1/src/…`). ADR-164 G11 re-scoped them to the *shipped* boundary:
`wifi-densepose-sensing-server` (Rust). They were verified against the current
Rust crate and closed on branch `fix/adr-080-sensing-server-security`. Each fix
(or already-fixed finding) is pinned by a test that fails on the old behavior.
**The Python v1 paths remain as-is** — v1 is archived and not the shipped
surface; this closure governs the live Rust server only.

## P0 — Fix Immediately

### 1. Rate Limiter Bypass / XFF spoofing (Security HIGH) — **RESOLVED (verified absent on Rust boundary)**

- **Original location (v1):** `archive/v1/src/middleware/rate_limit.py:200-206`
- **Problem:** Trusts `X-Forwarded-For` without validation. Any client bypasses rate limits via header spoofing.
- **Rust verification (2026-06-13):** The Rust sensing-server has **no XFF-trusting control to bypass** — there is no IP-based rate-limiter and no IP-allowlist, and neither security middleware reads a forwarded header. `bearer_auth.rs` authenticates on the token alone (`require_bearer` inspects only the `AUTHORIZATION` header); `host_validation.rs` decides on the `Host` header only. A repo-wide grep for `x-forwarded-for|forwarded|peer_addr|client_ip|real-ip` over `wifi-densepose-sensing-server` returns nothing. The only "rate limiter" is the MQTT *sample-rate* gate (`mqtt/state.rs`), a per-entity publish throttle with no IP/header input.
- **Resolution:** No code change needed (no vulnerable surface). Regression tests pin the immunity: `bearer_auth::tests::xff_header_never_affects_auth_decision` (spoofed XFF never flips a 401↔200 decision) and `host_validation::tests::forwarded_headers_never_bypass_host_allowlist` (spoofed `X-Forwarded-Host: localhost` never lets a foreign `Host: evil.com` past the allowlist). Residual: if an IP-based control is ever added, it must derive the peer from the socket (`ConnectInfo<SocketAddr>`) and only honor XFF from an explicit `--trusted-proxy` CIDR — captured as guidance in the test docstrings.

### 2. Exception Details Leaked in Responses (Security HIGH, CWE-209) — **RESOLVED**

- **Original location (v1):** `archive/v1/src/api/routers/pose.py:140`, `stream.py:297`, +5 endpoints
- **Problem:** Internal error/stack-trace detail serialized into client responses.
- **Rust finding (2026-06-13):** Six handlers in `wifi-densepose-sensing-server/src/main.rs` serialized the internal error `Display` into the JSON body: `edge_registry_endpoint` returned a panicked `spawn_blocking` `JoinError` (`"task … panicked"`) in a `500` and the raw upstream error in a `503`; `delete_model`/`delete_recording`/`start_recording` returned `std::io::Error` strings (OS detail / path); `calibration_start`/`calibration_stop` returned the `FieldModel` error chain.
- **Fix:** New `src/error_response.rs` module — `internal_error` / `internal_error_json` / `upstream_unavailable` log the full detail **server-side only** (tagged with a correlation id) and return a generic body (`{"error":"internal_error","correlation_id":…}`) with no `panicked`, no file paths, no Debug chain. All six call-sites rewired. Pinned by `error_response::tests::internal_error_body_does_not_leak_detail` (leak-substring guard, verified to fail on the reverted old body) + 4 sibling tests.

### 3. WebSocket JWT in URL (Security HIGH, CWE-598) — **RESOLVED (verified absent on Rust boundary)**

- **Original location (v1):** `archive/v1/src/api/routers/stream.py:74`, `archive/v1/src/middleware/auth.py:243`
- **Problem:** Tokens in query strings visible in logs/proxies/browser history.
- **Rust verification (2026-06-13):** The Rust sensing-server never reads a token from the URL. `require_bearer` (`bearer_auth.rs`) inspects only the `Authorization` header; the WebSocket handlers (`ws_sensing_handler`/`ws_introspection_handler`/`ws_pose_handler`) take a bare `WebSocketUpgrade` with no `Query` extractor; the single `Query` in the crate (`EdgeRegistryParams`) is a non-secret `refresh` flag.
- **Resolution:** No code change needed (no query-token path exists). Regression test `bearer_auth::tests::query_string_token_is_never_accepted` proves `?token=`/`?access_token=` in the URL never authenticates (stays `401`) while the same token in the header succeeds (`200`) — verified to fail if a query-token path is re-introduced.

### 4. Rust Tests Not in CI

- **Problem:** 2,618 tests across 153K lines of Rust — zero run in any GitHub Actions workflow. Regressions ship undetected.
- **Fix:** Add `cargo test --workspace --no-default-features` to CI. 1-2 hour task.

### 5. WebSocket Path Mismatch (Bug)

- **Location:** `ui/mobile/src/services/ws.service.ts:104` constructs `/ws/sensing`, but `constants/websocket.ts:1` defines `WS_PATH = '/api/v1/stream/pose'`.
- **Problem:** Mobile WebSocket silently fails.
- **Fix:** Align paths. Verify which endpoint the server actually serves.

## P1 — Fix This Sprint

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 6 | God file: 4,846 lines, CC=121 | `sensing-server/src/main.rs` | Untestable monolith |
| 7 | O(L×V) voxel scan per frame | `ruvsense/tomography.rs:345-383` | ~10ms wasted; use DDA ray march |
| 8 | Sequential neural inference | `wifi-densepose-nn inference.rs:334-336` | 2-4× GPU latency penalty |
| 9 | 720 `.unwrap()` in Rust | Workspace-wide | Each = potential panic in RT paths |
| 10 | 112KB alloc/frame in Python | `csi_processor.py:412-414` | Deque→list→numpy every frame |

## P2 — Fix This Quarter

| # | Issue | Impact |
|---|-------|--------|
| 11 | 11/12 Python modules have zero unit tests (12,280 LOC) | Services, middleware, DB untested |
| 12 | Firmware at 19% coverage (WASM runtime, OTA, swarm) | Security-critical code untested |
| 13 | MAT screen auto-falls back to simulated data | Disaster responders could monitor fake data |
| 14 | Token blacklist never consulted during auth | Revoked tokens remain valid |
| 15 | 50ms frame budget never benchmarked | Real-time requirement unverified |

## Bright Spots

- 79 ADRs (exceptional governance)
- Witness bundle system (ADR-028) with SHA-256 proof
- 2,618 Rust tests with mathematical rigor
- Daily security scanning (Bandit, Semgrep, Safety)
- Ed25519 WASM signature verification on firmware
- Clean mobile state management with good test coverage

## Full QE Reports (9 files, 4,914 lines)

| Report | What it covers |
|--------|---------------|
| `EXECUTIVE-SUMMARY.md` | Top-level synthesis with all scores and priority matrix |
| `00-qe-queen-summary.md` | Master coordination, quality posture, test pyramid |
| `01-code-quality-complexity.md` | Cyclomatic complexity, code smells, top 20 hotspots |
| `02-security-review.md` | 15 security findings (3 HIGH, 7 MEDIUM), OWASP coverage |
| `03-performance-analysis.md` | 23 perf findings (4 CRITICAL), frame budget analysis |
| `04-test-analysis.md` | 3,353 tests inventoried, duplication, quality grading |
| `05-quality-experience.md` | API/CLI/Mobile/DX UX assessment |
| `06-product-assessment-sfdipot.md` | SFDIPOT analysis, 57 test ideas, 14 session charters |
| `07-coverage-gaps.md` | Coverage matrix, top 20 risk gaps, 8-week roadmap |

## Consequences

- **P0 fixes** eliminate 3 security vulnerabilities and 2 functional bugs
- **P1 fixes** improve performance, reliability, and maintainability
- **P2 fixes** close coverage gaps and harden the system for production
- Target score improvement: 55 → 75+ after P0+P1 completion

---

*Generated from QE swarm analysis (fleet-02558e91) on 2026-04-05*

# ADR-292: Source provenance state machine — synthetic can never present as live

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: provenance, honesty, ui, sensing-server, security

## Context

An August 2026 external review found two provenance defects on the release
path:

1. The pose-fusion simulator starts in demo mode; on any page port other than
   3000 the WebSocket target falls back to `localhost:8765`, and if the
   connection fails the simulator keeps running while the status still reads
   "ready" — producing a convincing moving visualization with no live CSI
   (issue 1557).
2. The main sensing client labels the source **live** when the authenticated
   status endpoint returns an error for lack of authorization, until a real
   frame happens to correct it (issue 1526).

The common root cause: source state is a boolean (live vs not), so "unknown"
collapses to "live". CLAUDE.md requires MEASURED/CLAIMED/SYNTHETIC labeling
and forbids presenting synthetic output as real.

## Decision

Define one canonical, mutually exclusive `SourceState` enum shared by the
sensing server and every UI/client that renders a source:

- `Synthetic` — generated data (simulator/replay of synthetic fixtures).
- `LiveVerified` — frames from an authenticated, attested source.
- `LiveUnverified` — frames arriving but provenance not yet confirmed.
- `Stale` — last frame older than a configured freshness window.
- `Disconnected` — no source.

Rules enforced structurally:

- **`Unknown` is not a state.** Any ambiguous condition resolves to
  `LiveUnverified`, `Stale`, or `Disconnected` — never `LiveVerified`.
- A status-endpoint error resolves to `Disconnected`/`LiveUnverified`, never
  live-verified.
- The simulator constructs `Synthetic` and cannot transition to any `Live*`
  state without a verified frame.
- `Synthetic` is watermarked in every view and every export.
- Transitions are a pure function of (last-frame-age, auth-status,
  source-kind) so they are unit-testable without a clock or a socket.

Scope of this PR: the shared `SourceState` type + transition function + tests
in the sensing server, and wiring of the two identified surfaces (pose-fusion
simulator status, sensing client source label). Broader UI adoption follows.

## Consequences

- Closes the "synthetic shown as live" and "unknown shown as live" classes.
- A small breaking change to any consumer currently reading a boolean source
  flag; mitigated by exposing a compatibility accessor during migration.

## Validation

- Unit tests for every transition, especially: auth-error → not-live;
  simulator → never live without a verified frame; freshness expiry → `Stale`;
  watermark present on synthetic export.
- `cargo test -p wifi-densepose-sensing-server`.

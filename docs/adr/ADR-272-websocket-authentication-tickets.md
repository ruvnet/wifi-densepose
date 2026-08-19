# ADR-272: WebSocket authentication tickets

- **Status**: accepted
- **Date**: 2026-07-22
- **Deciders**: RuView maintainers
- **Tags**: auth, websocket, security, sensing-server
- **Related**: ADR-271 (Cognitum OAuth resource server), ADR-055 (integrated sensing server), PR #1313 (the exemption this supersedes), cognitum-one/dashboard ADR-060

## Context

`bearer_auth` gates `/api/v1/*`. WebSocket upgrade endpoints were exempt, for a
real reason: a browser's `WebSocket` constructor cannot attach an
`Authorization` header to the handshake, so a gated socket is simply
unreachable from page JavaScript. `/ws/sensing` and `/ws/introspection` sat
outside `PROTECTED_PREFIX` entirely; `/api/v1/stream/pose` was added to an
explicit `EXEMPT_PATHS` list by PR #1313.

The reasoning was sound. The consequence was not, and it was measured rather
than argued. On a server with `RUVIEW_API_TOKEN` set — an operator who believes
authentication is ON — a real WebSocket handshake carrying **no credential at
all**:

```
/ws/sensing            -> 101 Switching Protocols
/ws/introspection      -> 101 Switching Protocols
/api/v1/stream/pose    -> 101 Switching Protocols
/api/v1/models         -> 401 Unauthorized        (control)
```

**The control plane was locked and the data plane was open.** `/ws/sensing`
carries the live sensing output — presence, pose, breathing and heart rate.
`/ws/introspection` exposes internal pipeline state. For the ADR-055 desktop
topology (server bundled in the app, loopback only) that is bounded. For the
LAN/hub deployment RuView also supports, anyone who can reach the port can
watch the sensor.

ADR-271 sharpened the contrast rather than causing it: the REST surface is now
genuinely strong — offline-verified Cognitum tokens, scope-separated
destructive routes — which makes an ungated data plane the obvious way in.

*Precision about the evidence:* the handshake completing was verified. A
payload frame was not captured in that window, so the finding is "the
connection is established without a credential", not "data was read".

## Decision

Gate every WebSocket upgrade. Accept **either** of two credentials, chosen to
match what each kind of client can actually do.

### 1. Native clients send a bearer on the upgrade

The Python client, the Rust CLI and the TypeScript MCP client are not browsers
and have never been subject to the header limitation. They **can** send a normal
`Authorization: Bearer` on the handshake, so the server accepts one there;
routing them through a ticket would add a round-trip and a second credential
path for no benefit.

> **Correction, 2026-07-23.** This section previously stated that those clients
> **do** send a bearer. The published Python client does not:
> `python/wifi_densepose/client/ws.py` calls `websockets.connect(url,
> ping_interval, ping_timeout, max_size)` and passes no headers at all — the
> file contains zero occurrences of `extra_headers` or `Authorization`. So every
> `wifi-densepose[client]` consumer **401s the moment an operator enables
> auth**, and this ADR told them they would be fine.
>
> The server side of the decision stands — a bearer on the upgrade is accepted,
> and that is the right contract for a non-browser client. What is missing is
> the client implementing it, tracked as ruvnet/RuView#1395. Until then the only
> remedy available to those users is
> `RUVIEW_WS_LEGACY_UNAUTHENTICATED=1`, which reopens the exposure this ADR
> exists to close — so it is a migration aid with a deadline, not an answer.

### 2. Browsers exchange their credential for a single-use ticket

`POST /api/v1/ws-ticket` is an ordinary authenticated request — where headers
*do* work — and returns an opaque ticket the page appends as
`?ticket=<value>` on the socket URL.

**A credential in a URL is normally a mistake.** URLs reach access logs,
`Referer` headers and browser history. Three properties bound this one, and all
three are load-bearing:

| Property | Why it matters |
|---|---|
| **Single use** — consumed on the first upgrade attempt, valid or not | A ticket found in a log is already spent |
| **~30 second TTL** | Long enough to open a socket; not long enough to harvest |
| **Not the credential** — authorizes one WebSocket | Cannot be replayed against `/api/v1/*`, cannot be refreshed, carries no reusable identity |

The long-lived bearer token is still never placed in a URL.

A ticket **inherits the issuing principal's scopes**, so a `sensing:read`
session cannot mint one that outranks itself, and a ticket from a token without
`sensing:read` is refused at the upgrade.

### 3. WebSocket paths are matched by **prefix**, not by an allowlist

Anything under `/ws/` is treated as an upgrade path, plus the one endpoint that
lives outside it (`/api/v1/stream/pose`).

This is the most important detail in the ADR. An allowlist means every
WebSocket route added later is ungated until someone remembers to extend it —
the same bug, reintroduced on a delay. It is not hypothetical:
`/ws/train/progress` (ADR-186, arriving with PR #1387) is already referenced by
`ui/services/training.service.js` and would have shipped unauthenticated under
an allowlist. Prefix matching gates it on arrival.

New WebSocket routes should live under `/ws/` and inherit gating for free.

### 4. A migration escape hatch, deliberately uncomfortable

`RUVIEW_WS_LEGACY_UNAUTHENTICATED=1` restores the previous behaviour. Gating
these paths **breaks a browser UI that has not yet been updated to fetch a
ticket**, and not every deployment can update server and UI in lockstep.

It is a migration aid, not a supported configuration:

- It logs a warning on every boot naming the actual exposure — "the live
  sensing stream — presence, pose and vital signs — is readable by anyone who
  can reach this port" — rather than something an operator can skim past.
- Its blast radius is exactly the WebSocket paths. A test pins that it does not
  weaken `/api/v1/*`.
- It is read **once at construction**, so changing the environment cannot
  silently open the paths on a running server.

The alternative — a clean break with no hatch — was considered and rejected as
sequencing, not principle: a hard break tempts an operator into turning auth off
entirely, which is strictly worse than a narrow, loudly-announced exception.
The hatch should be removed once the shipped UI fetches tickets.

### 5. Deployments with auth off are unchanged

No credential configured ⇒ the middleware is the same no-op it has always been.
Pinned by a test.

## Consequences

- The measured hole is closed: all three paths now return `401` to a
  credential-less handshake, while a bearer or a valid ticket returns `101`.
- Browser UIs need updating. Shipped in the same change for
  `sensing.service.js`, `websocket-client.js` and `observatory/js/main.js` via
  a shared `withWsTicket()` helper; a ticket is minted per connection attempt
  and never cached, because it is single-use and short-lived.
- A UI running against a server that predates this ADR still works: the helper
  treats `404` from `/api/v1/ws-ticket` as "no ticket needed".
- One more round-trip before a browser opens a socket. Negligible against a
  stream that then runs for minutes.
- Tickets live in memory, capped at 512 outstanding and self-healing as they
  expire, so an authenticated but misbehaving caller cannot grow the store
  without bound. In-memory is correct rather than convenient: a ticket
  surviving a restart would outlive the server that vouched for it.

## Supersedes

PR #1313's `enabled_exempts_pose_stream_websocket`, which asserted the
exemption. Its premise about browsers was correct and is preserved here; its
conclusion is replaced. The test was renamed and inverted rather than deleted,
with the history in its doc comment, and the half that still matters — the
WebSocket rule must not leak to other `/api/v1/*` paths — is kept.

## Deliberately not done

- **`/health*` stays ungated.** Orchestrator probes hit it anonymously, and
  that is the point of a liveness endpoint. `/health/metrics` is included in
  that exemption; if metrics ever carry occupancy-derived values this should be
  revisited, because that would make them sensing data wearing an ops label.
- **`/ui/*` stays ungated.** It is static assets; the data behind them is
  gated.
- **No revocation of an issued ticket.** It expires in seconds and is
  single-use; a revocation path would be more machinery than the exposure
  justifies.
- **No ticket for native clients.** They can send a header, so they should.

## Implementation

`v2/crates/wifi-densepose-sensing-server/src/ws_ticket.rs` (store),
`src/bearer_auth.rs` (gating), `src/main.rs` (`POST /api/v1/ws-ticket`),
`ui/services/ws-ticket.js` plus the three call sites.

Tests: 12 store, 9 gating, 4 path-matching. Store coverage includes single-use
enforcement, replay refusal, expiry refusal *and* pruning, 256-bit
unpredictability, cap enforcement and self-healing, and `?myticket=x` not being
read as `?ticket=x`. Gating coverage includes every known WS path refusing an
unauthenticated upgrade, bearer acceptance, ticket single-use, a ticket being
useless against REST, the escape hatch working *and* not weakening REST, and
auth-off behaviour unchanged.

# ADR-271: RuView as a Cognitum OAuth resource server

- **Status**: accepted
- **Date**: 2026-07-22
- **Deciders**: RuView maintainers
- **Tags**: auth, oauth, cognitum, security, sensing-server
- **Related**: ADR-055 (integrated sensing server), ADR-102 (edge module registry), ADR-066 (ESP32 seed pairing), cognitum-one/dashboard ADR-060 (OAuth scopes beyond `inference`), cognitum-one/meta-llm ADR-045 (Bearer at completions)

## Context

`/api/v1/*` on `wifi-densepose-sensing-server` is gated by `RUVIEW_API_TOKEN`
(`bearer_auth.rs`): a single shared secret, compared in constant time, with no
expiry, no rotation and no per-user attribution. `homecore-api` has a second,
unrelated scheme (`LongLivedTokenStore` over `HOMECORE_TOKENS`) whose own doc
comment describes it as "no expiry, no rotation, no per-user attribution yet".

That is proportionate for the ADR-055 topology — server bundled in the desktop
app, spawned as a child, localhost only. It is not proportionate for the other
deployment RuView actually has: a sensing server on a Pi or hub, reachable on a
LAN, potentially serving more than one person, exposing live presence, pose,
breathing and heart-rate data plus destructive operations (model training,
model delete, recording delete).

Cognitum operates a live OAuth 2.1 authorization server at `auth.cognitum.one`.
Users of RuView are already Cognitum account holders. The obvious question is
whether RuView can accept that identity instead of a shared string.

### The direction of the integration is the thing most likely to be misread

Every existing Cognitum OAuth integration in the org — meta-proxy, musica,
metaharness, the dashboard CLI — is an OAuth **client**: it obtains a token so
the application can *call* a Cognitum service (the completions plane).

RuView is the opposite. It makes **no authenticated calls to any Cognitum API**.
Its only outbound Cognitum dependency is the ADR-102 registry fetch, which is an
anonymous GET against a public GCS bucket. What RuView wants is to be a
**resource server**: a user signs in to their *own* RuView instance with their
Cognitum identity, and RuView verifies the token they present.

So the client-side prior art in the org, while useful for a future `ruview
login` command, addresses a plane RuView does not have. The only relevant
precedent is `meta-llm/src/auth/oauthBearer.ts` (ADR-045) — the org's sole
resource-server-side verifier of these tokens. It is TypeScript; **RuView is the
first Rust one.**

### Facts about the tokens, verified against a live production token

- **ES256 JWT**, signed by a single P-256 key published at
  `https://auth.cognitum.one/.well-known/jwks.json`.
- **15-minute lifetime**, with an opaque refresh token that **rotates with reuse
  detection** (presenting a spent one ends the session).
- Claims: `typ`, `sub`, `account_id`, `org_id`, `workspace_id`, `client_id`,
  `scope`, `family_id`, `jti`, `iat`, `exp`, `setup`, `workload`.
- **No `aud` claim.** No `/oauth/introspect`. No `/userinfo`. It is an OAuth 2.1
  authorization server, not an OpenID Provider, deliberately.

## Decision

Verify Cognitum access tokens **offline**, in a new `ruview-auth` crate, and
gate RuView's own API surface on the **scope** they carry.

### 1. Offline verification is a requirement, not an optimisation

RuView runs on Pi-class hardware that loses WAN, and there is no introspection
endpoint to call even when the network is up. Verification is therefore an
ES256 signature check against a `kid`-indexed JWKS cache. Two consequences we
accept explicitly:

- **Revocation window = token lifetime.** A compromised access token stays
  usable until `exp`. This is the same position meta-llm takes, for the same
  reason, and it is why §3 refuses long-lived credentials.
- **A JWKS refetch failure is survivable while a key set is cached.** A key that
  verified a minute ago has not stopped being valid because the network blipped;
  failing closed there would log every user out of their own sensing server
  whenever their internet wobbled. We fail closed in exactly one case: no key
  set has *ever* been fetched.

### 2. The accept-rule is ported from meta-llm, not designed

```
typ == "access"  AND  NOT setup  AND  NOT workload
AND account_id is a non-empty string
AND exp is in the future
AND the scope required by the route is held
```

**Note there is no `iss` check.** An earlier revision of this section listed
"`iss` matches the configured issuer verbatim" — that rule was implemented,
shipped, and rejected EVERY real token, because Cognitum access tokens carry no
`iss` claim (see §"Facts about the tokens" above, which contradicted this
paragraph for a day). Removed in the code; removed here. The JWKS is the issuer
binding.

Divergence from `oauthBearer.ts` would be a bug rather than a preference: a
token meta-llm rejects must not be one RuView accepts. The algorithm is **fixed
to ES256 by our code** — the header's `alg` is only ever compared against that
allowlist, never used to select an algorithm.

### 3. Long-lived setup and workload credentials are refused outright

Identity also issues 365-day *setup* and machine *workload* credentials. Their
revocation state lives in identity's `oauth_setup_tokens` table. RuView — like
meta-llm — has no database and no way to check it, so accepting one would mean
honouring a credential that may already have been revoked. A 15-minute token
needs no revocation round-trip because it expires faster than revocation
propagates; a 365-day one does.

### 4. Scope is the capability boundary, because nothing else can be

Tokens carry no `aud`, so RuView cannot verify a token was minted *for* RuView.
`client_id` cannot substitute: clients borrow each other's registrations when
their own has not been deployed (musica ships `DEFAULT_CLIENT_ID = "meta-proxy"`).

This is not a defect to route around. Cross-product **identity** is intended —
one Cognitum account, every Cognitum product. Cross-product **capability** is
not, and scope is what carries the difference.

RuView registers two scopes (dashboard ADR-060, identity migration `0016`):

| Scope | Grants |
|---|---|
| `sensing:read` | sensing/pose streams, one-shot inference, reading model and recording metadata |
| `sensing:admin` | every mutating route not explicitly allowlisted as read-safe — training (`/api/v1/train/*` AND `/api/v1/adaptive/train`), model and recording deletion, config writes |

**The gate is fail-closed for writes, and that polarity is load-bearing.** An
earlier revision enumerated admin routes by prefix and let everything else fall
through to `sensing:read`. `POST /api/v1/adaptive/train` — which trains a
classifier, overwrites the on-disk model and swaps the live one — does not match
`/api/v1/train/`, so it was reachable with `sensing:read`, the scope
`wifi-densepose login` requests by default. Found by adversarial review. Now:
reads are open, writes require admin unless the exact path is on a short
allowlist of non-destructive mutations. A route added tomorrow is admin-gated
until someone classifies it.

**No hierarchy**: `sensing:admin` does not imply `sensing:read`. Consent means
exactly what it said, and a token needing both must have consented to both.
`client_id` is retained on the principal for logging and attribution only —
never as an authorization input.

### 5. Additive and fail-closed, never a silent downgrade

`RUVIEW_API_TOKEN` and `HOMECORE_TOKENS` deployments keep working unchanged.
OAuth is opt-in; with it unconfigured, behaviour is byte-identical to today.
When OAuth *is* configured but unusable (JWKS unreachable at boot, required
scope not registered), the server must refuse to serve `/api/v1/*` rather than
fall through to an open or single-secret state.

### 6. `ureq`, and a transport seam

`wifi-densepose-sensing-server` deliberately chose `ureq` as "the smallest" HTTP
client. Introducing `reqwest` for a JWKS fetch would silently reverse that for
the whole dependency graph. The fetch sits behind a `JwksFetcher` trait — the
`ureq` implementation is a default-on feature, and a host may supply its own and
take no HTTP dependency at all.

## Consequences

- Requests become attributable: `sub`, `account_id`, `org_id`, `workspace_id`,
  `jti`. This closes the gap `homecore-api`'s `tokens.rs` has been deferring as
  "P3", using claims rather than new RuView machinery.
- Destructive operations can be separated from observation for the first time.
- **The 15-minute lifetime is the main operational cost.** A long-running client
  must refresh, and because refresh tokens rotate with reuse detection, a
  concurrent or naively retried refresh **ends the session** — single-flight is a
  correctness requirement, not an optimisation. This lands with the login flow,
  not this crate.
- Hosts without a battery-backed clock will fail `exp`/`iat` until NTP lands.
  The verifier reports that distinguishably so it is diagnosable rather than
  presenting as a generic 401.
- A new dependency, `jsonwebtoken` — the same crate, same major version, that
  identity itself uses to sign these tokens.

## ~~Known incomplete: the browser cannot obtain an OAuth token~~ — CLOSED 2026-07-23

> **Superseded within this same PR.** The text below described the state when
> this ADR was first written. It is retained because the reasoning still
> explains *why* the browser half was built, but every factual claim in it is
> now false — in particular `grep -ril "oauth|cognitum|pkce" ui/` now returns
> `ui/sw.js`, `ui/sw.test.mjs` and `ui/utils/quick-settings.js`. An adversarial
> review caught the ADR still asserting the old state; see "Browser sign-in"
> below for what actually ships.

<details>
<summary>Original text (no longer accurate)</summary>

`wifi-densepose login` writes to `~/.ruview/credentials.json` — a file a browser
cannot read. The UI's `ws-ticket.js` reads a bearer from
`localStorage['ruview-api-token']`, which is populated **only** by the
QuickSettings manual-paste panel. There is no "Sign in with Cognitum" control,
no redirect flow, and `grep -ril "oauth|cognitum|pkce" ui/` returns nothing.

So a user who signs in via the CLI gets **no benefit in the browser UI**, and
the WebSocket ticket mechanism this ADR's sibling (ADR-272) introduces "for
browsers" is today only exercisable with the legacy static shared secret that
OAuth was meant to replace. The server-side gating is correct and complete; the
browser half of the story these ADRs tell is not built.

</details>

## Browser sign-in

`/oauth/start`, `/oauth/callback`, `/oauth/logout` and `/oauth/status`, plus a
"Cognitum Account" panel in QuickSettings. The server runs the authorization
code + PKCE flow itself and hands the browser a **signed session cookie** —
never the access token. The browser gets an assertion that this server already
verified a token, which is nothing replayable anywhere else.

Three things about it are load-bearing and were each found the hard way:

- **The cookie carries the granted scope**, and the gate re-checks it per
  request. A `sensing:read` session cannot delete a model.
- **`__Host-` is deliberately NOT used.** That prefix requires `Secure`, and
  RuView is routinely reached over plain HTTP on a LAN; a cookie the browser
  refuses to set is worse than one without the prefix. The cost is real and is
  recorded as P3 under "Open problems" below.
- **The service worker must never cache `/oauth/*` or authenticated `/api/*`.**
  The Cache API is not the HTTP cache and ignores `Cache-Control` entirely, so
  a cached `/oauth/status` froze sign-in until a hard reload, and cached API
  responses could be replayed to a different user after sign-out. `ui/sw.js` is
  now deny-by-default with an allowlist.

### Still incomplete

`redirect_uri` defaults to `http://127.0.0.1:8080/oauth/callback` and is
overridden only by `RUVIEW_PUBLIC_BASE_URL`. Browser sign-in therefore works
only on a host reached at exactly that origin: an operator browsing
`http://localhost:8080` or `http://192.168.1.50:8080` cannot complete the flow
(PKCE keeps the code unexchangeable, so this is a broken flow, not a token
leak). Deriving it from the request is the fix; deferred deliberately, since
deriving a redirect URI from attacker-controllable headers is its own class of
bug and deserves its own decision.

The credential `wifi-densepose login` stores is also **not yet consumed by any
shipped client** — no CLI subcommand, MCP server or Python client reads
`~/.ruview/credentials.json`. The token is obtainable and verifiable; wiring the
clients to send it is separate work.

## Open problems — RESOLVED 2026-07-23

Three findings from the 2026-07-23 adversarial review. All three are now
**fixed**; the analysis is retained because it explains why each fix has the
shape it does, and each is guarded by a test that was confirmed to fail against
the old behaviour.

### P1 — the JWKS fetch blocks a tokio worker, and the stale path is unbounded — **FIXED**

`verify.rs:182` calls `JwksCache::decoding_key_for`, which performs a blocking
`ureq` request (`jwks.rs:181`, 3s connect + 3s read) directly on the async
worker running `require_bearer`. The same codebase already knows this is wrong:
`main.rs:9265` wraps the token exchange in `spawn_blocking`, commenting "the
same mistake this codebase had to fix in `jwks.rs`". The hot verification path
did not get the same treatment.

Worse, the rate limiter does not cover the case that matters.
`state.fetched_at` is updated **only on success** (`jwks.rs:188`); the error arm
leaves it untouched. So once the TTL elapses after the last *successful* fetch,
`fresh` is permanently `false`, the `may_force` guard at `:170` is never
consulted, and **every** request performs its own blocking fetch attempt.

This fires with no attacker present. On a Pi that loses WAN — the documented
deployment reality — 300 seconds later every API call and every UI poll starts a
blocking outbound attempt, and with few tokio workers the whole server stalls,
including `/health`. An attacker can reach the same state deliberately by
flooding tokens carrying an unknown `kid`.

**Proposed fix, in dependency order:**

1. **Rate-limit attempts, not successes.** Add `last_attempt_at`, recorded
   before the fetch regardless of outcome, and consult it on the stale path too.
   This alone converts "every request fetches" into "one request per interval".
2. **Get the blocking call off the runtime.** Either wrap the call in
   `spawn_blocking` at the `verify` boundary, or give `JwksCache` an async
   transport behind the existing transport seam. The seam already exists —
   `JwksCache::new` takes a boxed transport — so this is an added
   implementation, not a redesign.
3. **Single-flight the refresh.** Concurrent misses for the same `kid` should
   await one shared fetch rather than each issuing their own.
4. **Refresh ahead of expiry** from a background task, so the request path
   normally never fetches at all.

Steps 1 and 2 are the ones that remove the stall; 3 and 4 are optimisations.
The test that must accompany this: a transport whose fetch blocks on a barrier,
asserting that a second concurrent verification is not serialised behind it —
the current suite is entirely single-threaded and could not observe a
reintroduction (`jwks::tests` contains no concurrency primitive at all).

### P2 — a 15-minute access token becomes a 12-hour session — **FIXED**

`issue()` sets `exp: now() + SESSION_TTL_SECS` with `SESSION_TTL_SECS = 12 *
3600`, deliberately not inheriting the access token's ~15-minute lifetime. The
session cookie is an assertion that this server verified a token, so it is not
*wrong* for it to outlive the token — but 12 hours is a long time to hold an
authority that cannot be revoked. Cognitum publishes no introspection endpoint
(see "Facts about the tokens"), so RuView has no way to ask whether the grant
behind a session still stands. A disabled account keeps sensing access, and
`sensing:admin` if it had it, until the cookie expires on its own.

**Correction.** An earlier revision of this section said capping the session at
`sensing:read` was "considered and rejected, because the dashboard genuinely
performs admin operations". That was wrong, and a cross-vendor pre-merge sweep
caught it: `/oauth/start` (`main.rs:9206`) already requests `SENSING_READ` and
nothing else, deliberately — "admin work goes through the CLI, which requires an
explicit `--admin`". So a browser session is **already** read-only, and the
consequence I claimed capping would cause is simply the current behaviour.

Two things follow, and both are stated here rather than left for the next reader
to trip over:

1. **The UI's admin controls do not work from a browser OAuth session.**
   `model.service.js:136` issues `DELETE /api/v1/models/{id}`; from a
   Cognitum-signed-in browser that returns 401. Admin work requires either the
   CLI (`wifi-densepose login --admin`) or a manually pasted admin bearer in the
   QuickSettings token field. This is a gap in the browser feature, not a
   regression — browser sign-in is new here, and the token-paste path still
   carries whatever authority the pasted token has.

2. **The step-up control below is therefore a guard ahead of need, not an active
   one.** No browser session currently holds `sensing:admin`, so
   `session.has_scope(SENSING_ADMIN)` is false and the freshness branch never
   fires in production. Its tests pass because the crate-internal test seam
   mints an admin cookie the real flow does not produce. That is worth naming
   plainly: it is correct code guarding a case that cannot yet arise, and it
   becomes load-bearing the moment anyone widens the requested scope — which is
   the right time for the guard to already exist, but it is not evidence that
   the control is exercised today.

### Decision, 2026-07-23: the browser is read-only, permanently

**Browser-side admin is not wanted.** `BROWSER_SIGNIN_SCOPE` stays
`sensing:read`, and the escalate-on-demand design sketched while this was still
open is **not** being built.

The reasoning holds up on its own terms rather than being a concession to
scope: the destructive operations — training, model delete, recording delete —
already have a home in the CLI, where `--admin` is explicit, typed by a person,
and scoped to the session that needed it. Routing them through a browser would
mean either asking every user to consent to delete capability in order to watch
a stream, or building a second consent flow to avoid that. Neither is worth it
for operations that are administrative by nature and rare by frequency.

What this settles:

- **The UI's admin controls are unreachable from a Cognitum browser session**
  and that is now intended, not a gap. `model.service.js` issuing
  `DELETE /api/v1/models/{id}` returns 401. The manual token-paste field still
  works and carries whatever authority the pasted token has, so nothing that
  worked before this change stops working.
- **The client-side step-up redirect has been removed** from
  `ui/services/api.service.js`. It caught a challenge that can never be issued,
  and it ended in a promise that never settles — so had any other 401 ever grown
  that header, every caller would have hung forever. Dead code with a trap in it
  is worse than no code.
- **`ADMIN_REVERIFY_SECS` stays as a server-side backstop.** It is fail-closed
  and costs nothing, so if the requested scope is ever widened the freshness
  requirement is already there rather than something to remember. It is
  documented at its definition as a backstop, so nobody mistakes its passing
  tests for evidence that it is exercised.

**Three options, with the tradeoff each carries:**

| Option | Effect | Cost |
|---|---|---|
| **A. Shorten the TTL** (e.g. 12h → 4h) | Bounds exposure by a factor of 3, one constant | Re-auth is a full-page navigation, which interrupts a live streaming dashboard. Mostly silent while the Cognitum session is alive, but not free. |
| **B. Server-side session store** with the refresh token, revalidated periodically | Real revocation: a disabled grant fails at the next refresh | The server now stores refresh tokens — a new and higher-value secret at rest — and refresh rotates with reuse detection, so a bug logs users out. |
| **C. Re-verify on privileged operations only** | `sensing:admin` requires a fresh token; reads keep the long session | Best blast-radius-per-unit-cost, but needs a UI affordance for step-up auth that does not exist. |

**Chosen: A, at one hour** — `SESSION_TTL_SECS` is 3600, down from 12 hours.

C was implemented too, and then the browser-read-only decision above made it a
backstop rather than an active control: with no browser session holding
`sensing:admin`, there is no privileged operation to re-verify. It is kept
because it is fail-closed and free, not because it is doing work today.

B is not built. It is only worth its cost — storing refresh tokens at rest,
against an authorization server that rotates them with reuse detection — if
RuView later needs true cross-device sign-out. Shortening the window addresses
the same risk for a fraction of the exposure.

That leaves a residual this ADR should not pretend away: **within one hour, a
revoked Cognitum grant still reads sensing data through an existing browser
session.** Cognitum publishes no introspection endpoint, so nothing short of B
closes that, and one hour is the size of the hole we accepted.

### P3 — dropping `__Host-` costs cookie origin-integrity, not just `Secure` — **FIXED**

The decision above frames omitting `__Host-` as trading away a `Secure`
requirement that RuView cannot meet on a plain-HTTP LAN. That framing is
incomplete: `__Host-` also guarantees the cookie was set by *this* origin with
`Path=/` and no `Domain`. Without it, cookies are not port-scoped and are not
integrity-protected against a same-host writer.

`read_cookie` returns the **first** match in the header, and RFC 6265 §5.4 sends
longer-`Path` cookies first. So an attacker who can set a cookie on the same
host — any other service on any port on that appliance, or a plain-HTTP MITM
injecting `Set-Cookie` — can plant `ruview_session=<their own validly signed
session>; Path=/ui`. The victim's browser then sends both, the attacker's first,
and it verifies correctly because it *is* genuinely signed. The victim ends up
operating inside the attacker's session; `/oauth/status` reports the attacker's
account, and anything the victim records is attributed to them.

Note the shape: the signature is doing its job. Forgery was never the threat
`__Host-` addresses, so "the signature is what protects the value" does not
answer this.

**Proposed fix (cheap, no prefix needed):** have `read_cookie` collect *all*
values for the name and accept only if exactly one verifies — or, more strictly,
reject outright when more than one `ruview_session` is present, since a browser
should never legitimately send two. Add `Secure` and the `__Host-` prefix
conditionally when the server knows it is behind TLS, keeping the plain-HTTP LAN
case working.

## Alternatives considered

**Keep `RUVIEW_API_TOKEN` only.** Zero work, and adequate for a single-user
localhost install. Rejected because it cannot express who did what, cannot be
revoked without a restart, and cannot separate "watch the stream" from "delete
the model" — all of which matter the moment the server is on a LAN.

**Exchange the OAuth token for a `cog_` key.** The pattern ADR-316 (meta-proxy)
and ADR-119 (metaharness) originally described. Rejected: it cannot work.
`/v1/me/keys` requires a *Firebase* ID token, not an OAuth token — meta-proxy
hit the resulting 401 in production, replaced the approach with Bearer-direct
under ADR-045, and deleted `mint.rs` as dead code.

**Call identity to introspect each token.** Rejected: no introspection endpoint
exists, and a network round-trip per request would be wrong for an edge sensing
server regardless.

**Wait for an `aud` claim before shipping.** Rejected as sequencing. `aud` would
touch every issued token and every verifier in the org; scope is additive and
independently correct. Tracked separately; adding `aud` later strengthens this
design rather than invalidating it.

**Use OAuth for the ESP32 device plane too.** Rejected as a category error.
Devices have no browser, no user and no human present; they already pair with a
`seed_token` bearer (ADR-066) plus a device-bound PSK. Cognitum OAuth is for the
API plane only.

## Implementation

`v2/crates/ruview-auth` — `jwks` (fetch, TTL cache, `kid` index, one
rate-limited forced refetch on an unknown `kid` so rotation is picked up without
waiting out the TTL), `verify` (the §2 accept-rule), `principal` (the verified
caller and its scopes).

41 tests pass under both `cargo test --no-default-features` (the repo's
canonical gate) and default features. The matrix signs real ES256 tokens with a
runtime-generated key — no key material is committed — and covers `alg:none`,
forged signatures, spliced payloads, unknown `kid`, expiry on both sides of the
leeway, `typ` confusion, `setup`/`workload` smuggled onto a `typ=access` token, missing and
empty `account_id`, and scope escalation.

The load-bearing case is
`g2_a_genuinely_valid_token_from_another_cognitum_product_cannot_reach_the_sensing_surface`:
a correctly signed, unexpired, right-issuer, right-`typ` token bearing
`client_id=meta-proxy` and `scope=inference` is rejected. Nothing about its
signature or identity claims distinguishes it — only scope does. A naive
verifier accepts it, and an `inference` token becomes a key to someone's home
sensor.

**Not in this crate**: WebSocket authentication (ADR-272) and any outbound
Cognitum call.

### Amendment, 2026-07-22 — the login flow lives here after all, behind a feature

The paragraph above originally also excluded the login flow. That was written
to keep the sensing server lean, which is the right goal but not a reason to put
the code somewhere else: the Tauri desktop app needs the same flow, and a second
copy of a PKCE + rotating-refresh implementation is exactly the kind of
duplication that drifts apart and then disagrees about something subtle.

So `login` is a **non-default feature** of this crate. A server built with
default features gets the verifier and nothing more — no `reqwest`, no tokio
networking, no browser launcher. The CLI opts in with
`features = ["login"]`, and the desktop app can do the same.

Shipped as `wifi-densepose login` / `logout` / `whoami`. Two properties worth
restating because they are easy to get wrong:

* **Refresh is serialised and never retried.** Identity rotates refresh tokens
  with reuse detection, so a concurrent refresh looks like replay and a retry
  *is* replay — either revokes the session family. `Session::ensure_fresh`
  holds an async mutex across the network call, re-checks expiry after
  acquiring it, and persists the rotated token before returning it.
* **Least scope by default.** `login` requests `sensing:read`; `--admin` is an
  explicit escalation and requests both scopes, since there is no hierarchy
  server-side.

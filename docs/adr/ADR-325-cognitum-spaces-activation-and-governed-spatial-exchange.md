# ADR-325: Cognitum Spaces activation and governed spatial exchange

- **Status**: Accepted — read path implemented; write path remains policy-gated
- **Date**: 2026-08-17
- **Deciders**: ruv
- **Tags**: cognitum-spaces, oauth, spatial-state, privacy, ruvector, policy, autogenous
- **Relates to**: ADR-271, ADR-277, ADR-304, ADR-306, ADR-312, ADR-318, ADR-319, ADR-321; Cognitum API ADR-094; Autogenous ADR-402

## Context

RuView produces camera-free RF perception locally. Cognitum Spaces provides a
tenant-scoped cloud projection of physical places. Autogenous ADR-402 proposes
using that projection as a spatial-intelligence input for agent coordination.
The useful product is not another sensor dashboard: it is a governed chain from
local perception to spatial state, persistent memory, explanation, and action.

Four product pillars define the requested integration:

1. **Spatial state** — sites, buildings, floors, rooms/spaces, zones, entities,
   semantic events, and alerts.
2. **RuView perception** — camera-free sensing is normalized locally before any
   permitted P2/P3 semantic event synchronizes.
3. **Persistent memory** — RuVector grounds anomaly explanations in
   tenant-scoped spatial history.
4. **Governed action** — agents observe or recommend by default; consequential
   execution requires explicit policy authorization.

The live API audit on 2026-08-17 established the current production boundary:

- `GET https://api.cognitum.one/v1/spaces` exists and returns a bounded list;
- an unauthenticated request is rejected;
- the current account has no paired sites, so the authenticated result is an
  empty list rather than fabricated sample state;
- the projection declares HomeCore Edge authoritative and excludes raw CSI,
  CIR, RF tensors, recordings, pose frames, vital waveforms, and identity
  observations;
- the first deployed Function revision accepted only legacy `cog_` API keys;
- the gateway was configured to authenticate private Function hops, but the
  direct Function endpoint was still publicly invokable; that bypass has now
  been closed and the exact gateway runtime service account is the only
  invoker;
- OAuth protected-resource metadata and a RuView-scoped OAuth accept path were
  absent.

The Autogenous review at commit
`f7fa308b261bac89a8909edae8a3fdbbfb8ce66c` found additional integration risks:

- its Spaces client only listed spaces; no governed ingest contract existed;
- it trusted a loose TypeScript cast, with no response-size, timeout, redirect,
  or strict semantic-boundary validation;
- its observation conversion dropped tenant/message/sequence identity;
- missing confidence became zero but could still enter fusion;
- provenance could be substituted for calibration identity;
- a Spaces-derived belief could be converted back into an observation and
  counted as independent corroboration, laundering one source into two;
- its API-key exchange returns a `cognitum-cli` OAuth token, but the live Spaces
  endpoint accepted only a `cog_` key. Calling this “OAuth Spaces access” was a
  contract mismatch.

## Decision

Adopt a one-way-by-default, typed spatial exchange with separate activation,
data, memory, and action authorities.

```text
RuView RF capture (P0/P1, local)
  -> calibrated/OOD-gated semantic observation
  -> ontology + evidence + witness envelope (P2/P3)
  -> HomeCore authoritative edge state
  -> Cognitum Spaces tenant/workspace projection
  -> RuView bounded read client / Autogenous spatial context
  -> RuVector tenant-scoped memory and explanation
  -> recommendation
  -> ruvview-policy authorization + approval + receipt
  -> optional consequential action
```

Cloud state is a projection of edge state, not a second sensor and not an
independent corroborating modality.

### 1. Activation and data-plane credentials are distinct

RuView uses Cognitum's existing Authorization Code + PKCE flow with the public
`ruview` client. A user explicitly requests `spaces:read` with
`wifi-densepose login --spaces`. The authorization-server registration is a
ceiling; ordinary sensing login does not silently gain cloud access.

The Spaces resource server accepts either:

- a legacy API key carrying `spaces:read` (or the migration-compatible
  predecessor `devices:manage`); or
- a Cognitum OAuth access token that passes every condition below.

OAuth acceptance is conjunctive:

| Check | Required value |
|---|---|
| Signature | ES256 against `https://auth.cognitum.one/.well-known/jwks.json` |
| Issuer | exact `https://auth.cognitum.one` |
| Audience | exact `ruview` |
| Client claim | exact `ruview` |
| Token type | ordinary `access`; setup/workload tokens denied |
| Lifetime | current `exp`/`nbf`, five-second clock tolerance only |
| Scope | exact token `spaces:read` member |
| Tenant binding | valid non-empty UUID `org_id` and `workspace_id` |

An API key is not called OAuth. An OAuth token is not stored in
`COGNITUM_SPACES_API`. The compatibility environment variable contains an API
key only and is never printed, logged, or committed.

OAuth consent grants identity-bound read access. It does **not** grant device
pairing, data publication, deployment, billing, spending, leases, learning
promotion, automation installation, commands, or actuator authority.

### 2. The gateway owns the private credential relay

The public gateway strips inbound `X-Cognitum-User-Authorization` and
`X-Serverless-Authorization`. For a locked Function upstream it then:

1. retains a legacy `cog_` credential in `X-API-Key`, or, for the exact Spaces
   route only, retains a non-key bearer in a gateway-owned internal header;
2. replaces `Authorization` with the gateway's Google invoker ID token;
3. fails closed with `503` if it cannot mint that hop identity;
4. forwards only to the configured Function origin.

The Function's Cloud Run invoker check is enabled. `allUsers` has no invoker
binding; only the exact `apigateway-sa` service account may invoke it. This is
required because otherwise a caller could bypass Cloud Armor and spoof an
internal relay header.

The API publishes RFC 9728 protected-resource metadata naming the authorization
server and `spaces:read` scope. Discovery describes capability; it does not
grant it.

### 3. Tenant isolation is part of authentication

Legacy API-key documents are queried by their existing owner-bound `tenantId`.
OAuth requests are conjunctively queried by both signed `org_id` and
`workspace_id` using stored `tenantId` and `workspaceId` fields. The public
tenant identifier is projected from signed `org_id`. A request cannot supply
either selector in a query string.

No cross-tenant aggregation exists on this path. Pagination, search, memory,
and event endpoints added later must carry the same authoritative principal;
client-provided tenant filters may only narrow within it, never replace it.

### 4. Spatial model and ownership

The canonical RuView vocabulary remains ADR-306:

```text
Site -> Building -> Floor -> Space -> Zone
                                  -> Sensor / Person / Object / Track
                                  -> Observation -> Event -> Alert
```

Cognitum may call a bounded room a “space”; RuView does not create a second
room type. Stable external IDs are namespaced and validated before entering the
ontology. HomeCore remains authoritative for local registry state and local
automation. Cognitum owns tenant/workspace projection and activation. RuVector
owns indexed spatial history, not tenancy or authorization.

The current live endpoint exposes the first `Space` slice only. Sites, floors,
zones, entities, events, and alerts are contract milestones, not inferred from
missing fields. A client must represent absence as unknown/unavailable and must
not fabricate parents, coordinates, people, alerts, or provenance.

### 5. Privacy boundary and synchronization eligibility

Only allow-listed P2/P3 semantic projections may cross the cloud boundary.

| Class | Examples | Cloud default |
|---|---|---|
| P0 | raw CSI, CIR, RF tensors, packet captures | prohibited |
| P1 | pose frames, vital waveforms, identity observations, recordings | prohibited |
| P2 | occupancy count, bounded activity/fall possibility, anomaly score | permitted when policy allows |
| P3 | versions, connection health, signed capability metadata | permitted |

The client independently rejects forbidden raw-field names anywhere in the
response. This is defense in depth, not a substitute for server-side
projection. It also enforces HTTPS except for loopback tests, refuses redirects,
uses bounded connect/total timeouts, caps responses at 1 MiB, caps the list at
100 spaces, bounds nesting/arrays/strings, validates confidence, and rejects
non-P2/P3 space records.

Cloud-bound envelopes must preserve, when available:

- tenant/workspace/site/space/device identity;
- `messageId` and monotonic `eventSequence`;
- `observedAt`, `expiresAt`, freshness, and connection state;
- privacy class and semantic schema version;
- calibrated confidence and explicit uncertainty/abstention;
- model, HomeCore, hardware-manifest, calibration, evidence, and witness
  provenance.

Provenance is never used as a calibration identifier. Missing confidence,
calibration, timestamp, or tenant identity stays missing and cannot satisfy an
admission rule.

### 6. No feedback laundering or false corroboration

A Spaces record derived from RuView evidence carries derivation lineage. If it
returns to RuView or Autogenous, it is a **projection/recollection** of that
lineage, not a new observation. It cannot:

- increment corroborating-sensor count;
- raise evidence level;
- be fused as an independent modality;
- reset freshness to retrieval time;
- erase abstention, contradiction, or uncertainty;
- generate a second belief that cites the first as support.

Deduplication keys include tenant, source/witness identity, message ID, and
sequence. Cycles are detected and rejected. Independent corroboration requires
a distinct authenticated source and evidence chain.

### 7. Persistent memory is tenant-scoped and explanation-oriented

RuVector indexes accepted semantic state under at least:

```text
(tenant_id, workspace_id, site_id, space_id, schema_version, time_bucket)
```

It stores bounded semantic features, uncertainty, evidence references, and
witness digests. It does not store OAuth/API credentials or prohibited raw
payloads. Retrieval always applies the authenticated tenant/workspace filter
before similarity ranking.

An anomaly explanation names:

- the current semantic state and its uncertainty;
- the relevant learned baseline/window from ADR-312;
- comparable tenant-local history;
- the measured deviation and contradictory evidence;
- the provenance/witness chain;
- the evidence label (`MEASURED`, `SYNTHETIC`, or `CLAIMED`).

Memory supplies context, not permission. A historically common action is not
automatically authorized.

### 8. Agents observe and recommend; policy authorizes action

Autogenous and other agents receive read-only spatial context by default. Their
normal outputs are observations, explanations, proposals, and recommendations.

Any consequential action must cross the ADR-321 `ruview-policy` gate with:

- an exact action class and target;
- a fresh capability certificate;
- KNOWN/DEGRADED/UNKNOWN domain state;
- bounded uncertainty and sufficient evidence;
- tenant/workspace authorization;
- expiry, nonce, idempotency key, and replay protection;
- required human/policy approval;
- a terminal witness receipt for allow or deny.

Missing policy, unknown action class, stale state, incomplete provenance, or an
unavailable approval service denies. OAuth `spaces:read` can never authorize an
action. This ADR adds no actuator method to the Spaces client.

## Implementation

### RuView

- `ruview-cognitum-spaces` is a reusable, read-only client with typed/redacted
  credentials and a bounded response decoder.
- `wifi-densepose login --spaces` explicitly requests `spaces:read` through the
  existing PKCE flow and credential store.
- `wifi-densepose spaces` refreshes OAuth through the existing single-flight,
  persist-before-return mechanism, verifies that the stored grant contains
  `spaces:read`, and lists validated state. `COGNITUM_SPACES_API` remains an
  explicit compatibility path.

### Cognitum Identity

- the `ruview` public client allow-list includes `spaces:read`;
- RFC 8414 metadata advertises it;
- refresh preserves the originally granted scope;
- no new client secret or password grant is introduced.

### Cognitum API

- the gateway preserves caller OAuth through an internal, spoof-resistant
  relay while authenticating the private Function hop;
- Spaces verifies the signed OAuth principal and queries by tenant + workspace;
- legacy API-key behavior remains available;
- OpenAPI documents both alternatives and RFC 9728 metadata supports discovery;
- the Function remains gateway-only at Cloud Run IAM.

### Autogenous

Autogenous must consume an explicitly typed credential. It must not imply that
`/v1/cli/session/exchange` produces a RuView-audience token: that exchange
currently produces `client_id=cognitum-cli` and cannot pass the Spaces policy.
An external RuView PKCE token may be supplied after activation, or a scoped API
key may be used as the compatibility path. Response validation and lineage
rules in this ADR apply before agent belief formation.

## Threat model

| Threat | Required control |
|---|---|
| Direct Function bypass | invoker IAM check; gateway SA only; no `allUsers` |
| Forged internal OAuth header | strip inbound relay headers; gateway writes after route classification |
| Token substitution | ES256/JWKS plus exact issuer, audience, client, type, scope, and tenant claims |
| Cross-tenant enumeration | principal-derived Firestore selector; bounded non-enumerating errors |
| Redirect/token exfiltration | redirects disabled; HTTPS required; fixed path |
| Oversized/malformed response | byte/depth/count/string bounds before use |
| Raw-data regression | server allow-list plus client forbidden-field rejection |
| Secret disclosure | redacting types; no token logs/URLs; `.env` untracked |
| Feedback amplification | lineage preservation, dedupe, cycle rejection, no independent corroboration |
| Memory leakage | tenant filter before vector search; no global nearest-neighbor pass |
| Agent overreach | observe/recommend default; ADR-321 fail-closed action gate |
| Stale/replayed state | expiry, sequence, message ID, freshness, witness receipt |
| JWKS outage/rotation | bounded cache; fail closed; refresh after unknown `kid`; no algorithm fallback |

## Deployment and rollback

Rollout order is dependency-safe:

1. merge and deploy Identity scope/metadata;
2. deploy the Spaces Function with OAuth verification while API-key behavior
   remains unchanged;
3. deploy the gateway relay and protected-resource metadata;
4. verify gateway API-key access, OAuth denial matrices, direct-URL `403`, and
   tenant isolation;
5. merge/release the RuView client and CLI activation;
6. enable Autogenous consumption only after its strict validation/lineage gates
   pass.

Rollback disables OAuth advertisement/relay and returns clients to scoped API
keys. It must not restore public Function invocation. Revoking an OAuth session
or API key must not alter paired-site state.

## Validation and acceptance

Required automated gates:

- Identity: metadata test, migration application, PKCE authorize/token/refresh
  scope preservation, cross-client scope denial;
- API Function: valid claim matrix and rejection for wrong issuer/audience/
  client/type/scope/tenant, API-key regression, tenant query assertion, bounded
  projection tests, build and dependency audit;
- gateway: spoofed relay stripped, caller OAuth preserved, Google hop identity
  substituted, OpenAPI security alternatives, RFC 9728 metadata, build and
  dependency audit;
- RuView: semantic decoder bounds/privacy tests, redaction tests, login scope
  tests, CLI compile, and live empty/non-empty response tests without fixtures
  masquerading as production;
- policy: no Spaces read can invoke an actuator; denial receipts are witnessed.

Production readback must prove:

- unauthenticated gateway request returns `401`;
- legacy scoped API key returns the authenticated tenant list;
- valid RuView OAuth returns only its workspace;
- wrong client, missing `spaces:read`, setup/workload token, and second-tenant
  token are denied;
- direct Function URL returns `403` even with a valid application credential;
- response remains `no-store` and excludes P0/P1;
- no secret appears in logs, diffs, artifacts, or issue/PR text.

Performance, detection quality, and action-safety numbers are not claimed by
this decision. Any such number requires a named reproducer and the repository's
evidence labels. An empty production tenant is a successful isolation/read-path
test, not sensing-quality evidence.

## Consequences

### Positive

- One Cognitum identity can explicitly activate RuView's cloud spatial read
  capability without sharing a long-lived static bearer.
- Tenant and workspace become cryptographically bound inputs to the data query.
- RuView and Autogenous gain useful spatial context without importing raw RF or
  inventing independent evidence.
- RuVector can ground explanations in local/tenant history while action remains
  separately governed.
- The direct-origin bypass is closed permanently, independent of OAuth rollout.

### Costs and limitations

- Two credential types coexist during migration and must stay visibly distinct.
- OAuth depends on Identity JWKS availability and correct key rotation.
- The current API exposes spaces only; the full hierarchy/events/alerts model
  remains staged work.
- OAuth workspace IDs will return only documents populated with `workspaceId`;
  legacy owner-only documents require an explicit migration, never a broad query.
- No write, command, or agent execution surface is implemented by this ADR.

## Alternatives considered

**Keep API keys only.** Rejected as the target: keys are useful for service
compatibility but do not provide user activation, consent, short lifetime, or
refresh/revocation semantics.

**Treat the CLI API-key exchange token as a Spaces OAuth token.** Rejected: it
is minted for `cognitum-cli`, not `ruview`, and accepting it would remove the
audience/client boundary.

**Trust the gateway without verifying OAuth in Spaces.** Rejected: hop identity
and user authorization are distinct, and authorization must remain valid if the
route topology changes.

**Make Spaces state independent corroboration.** Rejected: it is derived from
the same RuView/HomeCore lineage and would double-count evidence.

**Allow agents to execute from `spaces:read`.** Rejected: read consent is not
action authority, and perception confidence alone cannot authorize consequence.

**Synchronize raw RF for better cloud models.** Rejected by default: it violates
the edge privacy boundary and is unnecessary for the semantic product.

## References

- Autogenous ADR-402, `docs/adr/ADR-402-ruview-cognitum-spaces-spatial-intelligence.md`
- Cognitum API ADR-094, `docs/adr/ADR-094-cognitum-spaces-homecore-edge-boundary.md`
- RFC 7636, Proof Key for Code Exchange
- RFC 8414, OAuth 2.0 Authorization Server Metadata
- RFC 9700, OAuth 2.0 Security Best Current Practice
- RFC 9728, OAuth 2.0 Protected Resource Metadata

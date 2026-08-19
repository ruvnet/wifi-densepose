# ADR-325: Cognitum Spaces activation and governed spatial exchange

- **Status**: Accepted — legacy and versioned reads, OAuth activation, local spatial memory, governed-action policy, metaharness support, and npm distribution are implemented; HTTPS production evidence is complete
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

The contributor metaharness exposes this as CLI verb `spaces` and MCP tool
`ruview_spaces_list`. It delegates to the same Rust client rather than parsing
or refreshing OAuth independently. The tool never accepts a bearer token or API
key. MCP use requires an operator-provided `credential-use` grant, and MCP calls
cannot select the credential path or API origin. The adapter requires an
installed `wifi-densepose` binary rather than executing Cargo build scripts
from an auto-detected checkout while holding credential authority. Because
refresh tokens rotate, a read may atomically update the local OAuth credential
before contacting Spaces; this authentication side effect is disclosed and
does not add cloud write authority.

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
- the dependency-free contributor metaharness adds `spaces` /
  `ruview_spaces_list`, invokes only the OAuth branch, bounds and revalidates
  child output, fixes the production API origin, strips the API-key compatibility
  environment, requires an installed binary, and default-denies MCP access
  without `credential-use`.

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
- bounded semantic-state `PUT` is available only to an explicitly scoped API-key
  publisher and is not exposed by the RuView OAuth client;
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
4. verify gateway API-key access, OAuth denial matrices, direct-origin platform
   denial (`401` or `403` before application code), and tenant isolation;
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
- the direct Function origin is rejected by the Google platform with `401` or
  `403` before application code, even with a valid application credential;
- response remains `no-store` and excludes P0/P1;
- no secret appears in logs, diffs, artifacts, or issue/PR text.

Performance, detection quality, and action-safety numbers are not claimed by
this decision. Any such number requires a named reproducer and the repository's
evidence labels. An empty production tenant is a successful isolation/read-path
test, not sensing-quality evidence.

## Production evidence (2026-08-18)

The bounded Spaces read slice and RuView activation path are deployed. The exact
production release chain is:

- Spaces run `32148530629`, revision `spacesapi-00003-xij`, source
  `fc333e634cd918b9d6fdde4eecbe7beac1043ab8`, Node 22, runtime service account
  `spacesapi-runtime@cognitum-20260110.iam.gserviceaccount.com`, with
  `apigateway-sa@cognitum-20260110.iam.gserviceaccount.com` as sole invoker;
- gateway run `32151485401`, revision `apigateway-00180-peh`, source
  `c4e99ebb4ce0d4e1407f435f905621476c1f0166`, image digest
  `sha256:bacb81281a54256ff6fdaac253175e76ce6fc225f399163ca0a807a2839bd6a3`;
- Identity run `32163542502`, revision `identity-00052-fid`, source
  `fb6320827b879e481cad6caf184d3cbccd8279c4`, image digest
  `sha256:0cd5896518bd8ecf042d2f3e9aea58a32e65a68dbddaab1e54f8ae6da2bfab06`,
  and runtime service account
  `identity-runtime-prod@cognitum-20260110.iam.gserviceaccount.com`.

The live API-key matrix returned `200` with an empty bounded list,
`Cache-Control: private, no-store`, and no prohibited P0/P1 projection fields.
No credential returned `401`. A direct-origin request received a Google
Frontend Bearer challenge (`401`) before application code.

Two independent RuView Authorization Code + PKCE principals also passed the
live matrix. Each token used ES256, exact issuer/audience/client checks,
`sensing:read spaces:read`, signed UUID organization/workspace claims, refresh
rotation, and revocation. Each gateway read returned `200`, an empty bounded
list, and `private, no-store`; a corrupted signature returned `401`; and the
principals had distinct pseudonymous tenant/workspace fingerprints. This proves
the production empty-tenant behavior and independent claim binding. Non-empty
cross-tenant isolation remains emulator/staging evidence because production was
not mutated to manufacture a fixture.

Identity metadata deliberately advertises `spaces:read` for RuView but not
`spaces:write`. The deployed semantic-state `PUT` remains an API-key-only
publisher surface. RuView therefore has no OAuth write, command, policy-approval,
or actuator capability.

That receipt was for the initial flat Space slice. The following production
expansion supersedes only its hierarchy/event/alert deferral. MQTT, commands,
actuators, real-hardware accuracy, and the long-duration operational trial
remain outside the completed claim.

## Completed implementation and production expansion (2026-08-19)

- Cognitum API PRs #211 and #212 shipped the eight `/v1/spatial` collections,
  transactional hierarchy integrity, stable pagination, event/alert retention,
  strict P2/P3 admission, API-key-only writes, OAuth/API-key reads, and the
  additive-only Firestore release authority. Function run `32279092861`
  promoted active Node 22 revision `spacesapi-00005-kaf`.
- Edge PRs #214, #215, and #216 preserved canonical UUID routing, kept SQLi
  denial, and removed secret-valued API-key rate selection. Gateway run
  `32284410107` promoted the reviewed immutable digest to 100% production
  traffic. Every versioned collection returned HTTP 200 through the public
  edge; the hierarchy composite index is `READY` and both retention TTL fields
  are `ACTIVE`.
- The dedicated RuView service credential was rotated to exactly
  `spaces:read` and `spaces:write`; its predecessor returns 401. A non-mutating
  invalid-body probe reached write validation without persisting customer data.
  Other potentially affected owner keys and residual log retention remain
  tracked in Cognitum API #217.
- A live RuView Authorization Code + S256 PKCE consent requested exactly
  `sensing:read spaces:read`. Its in-memory token read versioned `sites` with
  HTTP 200 and schema `1.0`; the verifier then revoked the temporary refresh
  credential and persisted no token.
- RuView PR #1650 merged `ruview-cognitum-spaces`,
  `ruview-spatial-memory`, the ADR-327 policy extension, CLI paging, and the
  guarded `ruview_spaces_list` metaharness surface. PR #1651 removed stale
  feature-branch guidance and refreshed the signed package manifest.
- The contributor metaharness fixes the API origin, accepts bounded resource,
  limit, and opaque-cursor inputs, strips API-key compatibility authority over
  MCP, invokes only the hardened OAuth CLI, and rejects raw sensing or malformed
  hierarchy/event/alert output. Its test, security, reviewed-brain, flywheel,
  manifest, audit, exact-tarball, and claim-check gates pass.
- Release run `32286297277` rebuilt and smoke-tested the exact package and
  provenance-published `@ruvnet/ruview` 0.5.0. The public npm registry resolves
  0.5.0 as `latest`; no workstation publish was used.
- `ruview-spatial-memory` keeps one RuVector HNSW index per authenticated
  tenant/workspace with replay, derivation, retention, cascading-erasure,
  bounded-explanation, encrypted-snapshot, and reload-verified rotation gates.
  This is local `SYNTHETIC` evidence, not a production sensing claim.
- `ruview-policy` keeps observe/recommend/execute intents distinct, requires
  exact host grants plus signed approval for consequence, rejects nonce replay,
  and emits signed hash-chained receipts. `spaces:read` is explicitly denied as
  execution authority.
- Focused Rust gates and the Linux workspace/CLI/security lanes pass. Earlier
  Windows whole-workspace attempts ended in host compiler failure or timeout;
  those attempts are not reclassified as green evidence.
- No OAuth write/action scope, actuator callback, MQTT deployment claim, sensing
  accuracy claim, or real-hardware claim is introduced.

## Consequences

### Positive

- One Cognitum identity can explicitly activate RuView's cloud spatial read
  capability without sharing a long-lived static bearer.
- Tenant and workspace become cryptographically bound inputs to the data query.
- RuView and Autogenous gain useful spatial context without importing raw RF or
  inventing independent evidence.
- The design keeps a path for RuVector-grounded explanations and separately
  governed action without treating either as part of the deployed read slice.
- The direct-origin bypass is closed permanently, independent of OAuth rollout.

### Costs and limitations

- Two credential types coexist during migration and must stay visibly distinct.
- OAuth depends on Identity JWKS availability and correct key rotation.
- Production exposes both the legacy Space twins and the versioned hierarchy,
  anonymous entities, semantic events, and alerts over HTTPS. MQTT remains a
  design contract without deployment evidence.
- OAuth workspace IDs will return only documents populated with `workspaceId`;
  legacy owner-only documents require an explicit migration, never a broad query.
- The RuView client exposes no write, command, or agent execution surface. The
  separate API-key semantic-state ingress is neither OAuth activation nor
  actuator authority.

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
- Cognitum API hierarchy/events/alerts follow-up,
  `https://github.com/cognitum-one/api/issues/206`
- RuView metaharness OAuth surface,
  `https://github.com/ruvnet/RuView/issues/1643`
- RuVector spatial-history follow-up,
  `https://github.com/ruvnet/RuView/issues/1640`
- governed-action and witness-receipt follow-up,
  `https://github.com/ruvnet/RuView/issues/1641`
- RFC 7636, Proof Key for Code Exchange
- RFC 8414, OAuth 2.0 Authorization Server Metadata
- RFC 9700, OAuth 2.0 Security Best Current Practice
- RFC 9728, OAuth 2.0 Protected Resource Metadata

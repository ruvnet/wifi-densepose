# ADR-327: Governed action intents, approvals, replay protection, and witness receipts

- **Status**: Accepted — implementation complete; repository-wide and deployment gates pending
- **Date**: 2026-08-19
- **Decision owners**: RuView maintainers
- **Extends**: ADR-318, ADR-319, ADR-321, ADR-325
- **Implements**: ruvnet/RuView#1641
- **Tags**: policy, governed-action, approval, idempotency, witness, cognitum-spaces

## Context

The current `ruview-policy` crate evaluates assurance for an action class, but it
does not define a complete action intent, tenant/workspace binding, policy
version, approval, nonce/idempotency replay behavior, or signed terminal receipt.
An agent recommendation can therefore be mistaken for execution authority, and
`spaces:read` could be accidentally treated as a general capability.

The system needs a framework that can prove why an action was allowed or denied
without adding any actuator. Real actuation remains a separate integration and
requires its own threat model and device evidence.

## Decision

### 1. Typed intent and registered policy

A governed `ActionIntent` binds:

- intent ID, tenant, workspace, action name/class, and exact target;
- requested policy version and parameter/evidence digests;
- creation/expiry, replay nonce, and requesting principal;
- the recommendation/explanation that motivated review, never a hidden command.

The gate accepts only a registered action policy. Unknown action, action-class
mismatch, policy-version mismatch, target mismatch, invalid timestamps, and
missing exact host authority deny before assurance is evaluated. Tenant and
workspace are part of the signed intent/receipt and nonce key. `spaces:read` is
explicitly tested as insufficient for an `alerts:execute` rule.

### 2. Assurance and approval

The existing ADR-321 certificate/domain/uncertainty/evidence gate remains the
assurance authority. The registered policy declares a bounded minimum of
distinct enrolled approvers. An absent, rejected, duplicated, expired,
wrong-intent, wrong-policy-version, or unverifiable approval denies. Approval
resolution fails closed.

Agents observe, explain, or recommend by default. `evaluate` returns a decision
receipt; it does not call an actuator. An executor may consume an `allow` receipt
only if a separate adapter verifies the receipt, target, expiry, and its own
device-specific authority.

### 3. Replay and idempotency

The bounded in-memory gate stores terminal receipts by intent ID and tracks
nonces by `(tenant, workspace, nonce)`.

- exact intent replay returns the original terminal receipt;
- changed reuse of an intent ID returns a fail-closed idempotency error;
- reuse of a nonce by another intent returns a fail-closed replay error;
- expired intents and approvals deny;
- failed or denied attempts are terminal and auditable.

The current state store is bounded and in-memory, intended for local/runtime use
rather than cross-process replay protection. A production executor must place
the same intent/nonce/receipt invariants behind a transactional durable store;
this ADR does not claim that adapter exists.

### 4. Witnessed terminal receipt

Every evaluated observe/recommend/execute request produces a canonical receipt
containing the intent digest, decision/reason, policy version, tenant/workspace,
decision/expiry time, intent ID and nonce, approval count, and previous receipt
digest. The receipt is signed through the `ruview-attest` signer interface and
can be independently verified. Hash chaining makes removal/reordering visible.
Malformed input, ID conflict, nonce replay, capacity exhaustion, and sequence
exhaustion are errors before receipt creation and must be audited by the host.

The reference keyed-BLAKE3 signer remains `SYNTHETIC` evidence only, as documented
by ADR-319. Production asymmetric signing and key custody must be supplied by the
deployment adapter; no symmetric test MAC is represented as hardware identity.

## Consequences

### Positive

- Recommendation, authorization, and execution are distinct typed stages.
- Default-deny covers missing policy, stale evidence, unavailable approval, and replay.
- Every decision has a terminal, verifiable explanation.
- `spaces:read` cannot silently expand into consequence.

### Costs and limitations

- Executors must implement a separate receipt-verifying adapter.
- Distributed replay protection needs a transactional durable store.
- This ADR implements no actuator, command transport, pairing mutation, or device control.
- Simulator tests are not hardware validation.

## Validation

- unknown/missing policy, stale intent, policy-version/target mismatch,
  insufficient authority, and `spaces:read`-only denial;
- certificate/domain/uncertainty/evidence denial matrix from ADR-321;
- missing/rejected/expired/duplicate/wrong-intent approval tests;
- exact idempotent replay, changed reuse, nonce replay, and bounded-store tests;
- receipt signature, canonical digest, chain linkage, and tamper rejection;
- tests proving evaluation exposes no actuator callback or network/file side effect.

The focused `ruview-policy` suite passes on 2026-08-19. The reference signer
tests are `SYNTHETIC`; they are not hardware-identity evidence. The non-terminal
whole-workspace Windows gate still requires authoritative Linux CI evidence.

Any future actuator adds a separate ADR, credential boundary, failure/rollback
plan, allow/deny integration tests, and captured target-device evidence.

## Alternatives considered

**Let agents call actuators after a recommendation.** Rejected: recommendation
quality is not authorization.

**Treat OAuth scopes as action policy.** Rejected: `spaces:read` expresses read
consent only and carries no target-specific assurance or approval.

**Emit receipts only for successful actions.** Rejected: denial and unavailable
approval are security-relevant terminal facts.

## References

- ADR-318: Capability certificates
- ADR-319: Witness chain
- ADR-321: Decision policy action authorization
- ADR-325: Cognitum Spaces activation and governed exchange
- ADR-326: Tenant-scoped RuVector spatial memory
- ruvnet/RuView#1641

# ADR-284: Bounded nightly SOTA research agent

| Field | Value |
|---|---|
| Status | Accepted - implementation gated off by default |
| Date | 2026-07-29 |
| Builds on | ADR-283 |

## Context

RuView needs a repeatable way to notice relevant state-of-the-art work and turn
it into reviewable repository activity. A nightly model with simultaneous
network, repository-write, policy-evolution, and execution authority would
create an unacceptable prompt-injection and supply-chain boundary. It could
also confuse generated confidence with scientific evidence or silently turn a
research suggestion into production code.

Cognitum exposes an OpenAI-compatible completion service and a public
application registry. The contributor harness already commits a Darwin genome
and a signed Flywheel replay gate. Those components can support nightly
research without granting unattended learning promotion.

## Decision

Add a scheduled GitHub Actions workflow that runs daily at `03:17 UTC`, remains
disabled until a maintainer enables a repository variable, and supports a
manual evidence-only dry run.

The live flow has seven jobs:

1. Collect bounded public Cognitum-registry and recent arXiv evidence.
2. Ask Cognitum `cognitum-mid` for one proposal that is locally validated
   against a strict schema.
3. Score proposal completeness using the frozen Darwin policy and verify an
   honest-null Flywheel replay.
4. Deduplicate or create one issue.
5. For a locally classified low-risk proposal only, ask Cognitum for a tiny
   declarative transform and bounded test vectors. Trusted repository templates
   turn that data into the prototype module, tests, JSON, and README.
6. In a job with no external secret or GitHub write token, revalidate every
   artifact, verify the Flywheel replay, and perform static and syntax checks
   without executing generated code.
7. In a job with no model credential, re-hash the validated artifacts, create a
   new branch, open one draft PR, link it to the issue, and explicitly dispatch
   the credential-free contributor-harness verifier.

The jobs exchange bounded JSON artifacts. Cognitum receipts retain the
provider, endpoint, exact resolved tier/model, request ID, a recomputable
routing attestation, and digest metadata. Credential-free validation rebuilds
the deterministic request and verifies its digest. The raw-output digest is
audit metadata only because raw model transcripts are not retained.

## Security and evidence policy

Retrieved titles, abstracts, descriptions, and links are untrusted `CLAIMED`
evidence. Source hosts, paths, media types, redirects, time, byte counts,
records, and citations are validated. The fixed trusted prompt states that
evidence has no instruction authority. Model output is parsed as one JSON
object and locally reconstructs risk, citations, implementation disposition,
and fingerprint.

Risk classification is deliberately conservative. Security, authentication,
cryptography, workflow, dependency, release, deployment, production, firmware,
hardware, network-server, native/Wasmtime plugin, HomeKit pairing, STT/TTS, and
satellite-voice proposals are issue-only.

Autonomous implementation is restricted to new files beneath a fingerprinted
`examples/research-sota/nightly/` directory. The model cannot supply paths or
source text. It selects only a schema-bounded scalar transform and matching
test vectors; repository-owned templates deterministically emit exactly five
files. It cannot edit existing files or add dependencies. File count, size,
line count, paths, symlink ancestry, numeric bounds, operation schema,
secret-shaped values, canonical template digests, and accuracy claims are
checked. Emitted source receives syntax checking, but is not executed.

The deterministic score is named `PROPOSAL_COMPLETENESS`. It is explicitly not
a novelty, scientific-quality, safety, or performance score.

## Darwin and Flywheel boundary

Nightly automation reads the committed Darwin genome as frozen prompt policy.
It never calls Darwin evolution or any Cognitum evolve, pod, guidance-mutation,
brain-write, or promotion endpoint.

Flywheel evaluates the unchanged policy with the repository's honest-null
fixture. The signed replay must verify, report zero verified improvements, and
report no promotion. This canary proves only that the committed Flywheel gate
stayed root-only, rejected its candidate, and did not promote under the frozen
fixture. It does not evaluate the proposal. The no-learning/no-promotion
boundary for the nightly run comes from the workflow's static authority split,
closed commands, and artifact validation.

## Credentials and publication

Scheduled enablement requires:

- repository secret `COGNITUM_NIGHTLY_API_KEY`, limited to
  `completions:mid`; and
- repository variable `RUVIEW_NIGHTLY_SOTA_ENABLED=true`.

Model jobs receive no write-capable GitHub token. GitHub mutation jobs receive
no model key. Validation receives neither. The publish job has the additional
`actions:write` permission solely to dispatch the read-only
`ruview-harness-flywheel.yml` verifier with Darwin disabled, because a PR
created by the workflow token may not trigger ordinary pull-request workflows.

The agent creates draft PRs only. It cannot approve, merge, release, promote a
Darwin candidate, or update canonical shared-brain records. Before any branch
write, it re-fetches the issue and repository rules. Publication requires the
issue to remain open, bot-authored, correctly labelled, and fingerprint-bound;
`main` must require at least one approving review and the
`Verify contributor harness` job-name check. The publisher requires that exact
check name and GitHub Actions integration ID from GitHub's
effective-active-rules endpoint. GitHub hides
ruleset bypass actors from read-only tokens, so the workflow is not given an
administrative token to inspect them. Its safety does not depend on that
metadata: the publisher can create only a non-default branch and draft PR and
contains no merge, approval, or `main`-push path. Branch protection and
maintainer review remain the authority boundary.

## Consequences

RuView gains a low-volume research flywheel with durable evidence, stable
deduplication, and inspectable failure artifacts. A compromised paper,
registry record, or model can at worst propose bounded new example files that
still require static gates and human review.

The tradeoff is intentionally limited autonomy: production ideas become issues,
generated prototypes are not executed, and a missing credential, service
outage, schema drift, or validation ambiguity stops the run rather than
guessing. Maintainers must explicitly enable the schedule and permit Actions to
create pull requests.

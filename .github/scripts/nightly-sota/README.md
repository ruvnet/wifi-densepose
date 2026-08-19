# Nightly SOTA research agent

`nightly-sota-agent.yml` turns recent public research into at most one
repository issue and, for low-risk topics, one draft offline-prototype pull
request. It is intentionally not a general-purpose autonomous coding agent.

## Enablement

The committed schedule is `03:17 UTC` every day. Scheduled runs stay disabled
until both repository settings exist:

1. Actions secret `COGNITUM_NIGHTLY_API_KEY`, issued with only the Cognitum
   `completions:mid` scope.
2. Actions variable `RUVIEW_NIGHTLY_SOTA_ENABLED=true`.

The key must not receive guidance-write, evolve, pods, brain, Flywheel-write,
or administrative scopes. First run the workflow manually in `dry-run` mode;
that mode only collects a bounded evidence artifact and never reads the secret
or writes an issue. Manual `live` mode is restricted to the repository owner.

The repository must also allow GitHub Actions to create pull requests. Normal
branch protection must require at least one approving review and the
`Verify contributor harness` status check. The publisher requires that exact
job-name check to be bound to the GitHub Actions app,
uses GitHub's effective-active-rules endpoint, and stops before prototype
generation when either requirement is absent. It does not request an
administrative token to inspect hidden ruleset bypass actors; safety does not
depend on that metadata because the publisher has no merge or `main`-push path.

## Authority split

| Job | External credential | Repository authority | Result |
|---|---|---|---|
| `collect` | none | contents read | Normalized public Cognitum registry and recent arXiv evidence |
| `propose` | Cognitum completions key | contents read | One schema-checked proposal |
| `score` | none | contents read | Frozen Darwin digest, completeness score, honest-null Flywheel replay |
| `issue` | GitHub token | issue write, PR read | One deduplicated issue |
| `implement` | Cognitum completions key | contents read | Declarative transform and test vectors |
| `validate` | none | contents read | Schema, template, syntax, claim, path, digest, and replay checks |
| `publish` | GitHub token | branch/issue/draft-PR/Actions write | One draft PR and an explicit read-only harness-verifier dispatch |

The Cognitum key and a write-capable GitHub token never coexist in one job.
Model output is never executable code. Repository-owned templates emit the
prototype module and tests, which this workflow syntax-checks but never runs.

## Hard boundaries

- Public HTTPS sources are fixed to the Cognitum application registry and the
  arXiv Atom API. Redirects, oversized responses, unexpected media types, and
  schema drift fail closed.
- Retrieved text is `CLAIMED`, untrusted evidence. It is quoted inside a fixed
  trusted prompt and cannot grant authority.
- The Darwin genome is read-only. Scheduled jobs never invoke Darwin evolution.
- Flywheel runs a separate committed honest-null canary. A valid canary stays
  root-only, rejects its candidate, and reports zero verified improvements and
  no promotion. It does not evaluate the nightly proposal. The workflow's
  static authority split and artifact gates are what prevent nightly learning
  or promotion.
- High-risk topics stop at an issue. This includes production, security,
  authentication, release/deployment, workflows, dependencies, firmware,
  hardware, networking, native plugins, HomeKit pairing, and voice protocols.
- Low-risk model output is a closed transform DSL: bounded scalar test vectors
  and 1-8 allowlisted operations (`center`, `normalize-peak`, `absolute`,
  `square`, `difference`, `moving-average`, or `clip`). Local trusted templates
  emit exactly five `.md`, `.json`, and `.mjs` files below
  `examples/research-sota/nightly/<fingerprint>/`. Existing files, symlinked
  parents, dependencies, binaries, executable modes, and more than 400 lines
  are rejected.
- Publication is a draft PR. The agent cannot approve, merge, release, promote,
  or modify the reviewed shared brain.

## Deduplication and failure behavior

The stable fingerprint hashes sorted evidence IDs, finding class, and subsystem.
Issues and PRs carry an exact hidden marker. Only markers on
`github-actions[bot]` records with the automation label are trusted for
deduplication, so copied issue text cannot suppress future runs.

A failure leaves the last completed bounded artifact for seven days. Model,
protection-preflight, or validation failures may leave an issue without a PR;
maintainers can inspect the run and decide whether to continue manually. The
workflow does not retry a failed model call, force-push a branch, close an
issue, or delete a branch.

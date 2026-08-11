# ADR-296: Repository CSI data-incident controls — ignore rules and a pre-commit/CI policy check

- **Status**: Accepted — controls implemented; tree remediation gated on owner sign-off
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: privacy, data-governance, ci, security, incident

## Context

The external review found ~64.6 MB of tracked raw CSI recordings under
`data/recordings/` and `v2/data/recordings/` (largest an ~61.8 MB overnight
capture). CLAUDE.md explicitly prohibits committing CSI or person data. The
`.gitignore` rule pointed only at a pre-rename path
(`rust-port/wifi-densepose-rs/data/recordings/`) and did not cover the active
directories, which is how the captures were committed. Raw CSI is person data
(it encodes breathing, movement, presence), so this is a data incident, not a
formatting nit.

## Decision

**Implemented now (mechanical, no data-ownership judgment):**

- Fix `.gitignore` to cover `data/recordings/`, `v2/data/recordings/`, the
  legacy path, and `*.csi.jsonl` / `*.csi.meta.json` globs (done in this PR).
- Add a policy check (pre-commit hook + CI job) that fails when CSI-format
  files (`*.csi.jsonl`, `*.csi.meta.json`) or large JSONL captures are staged
  or present as tracked files, with a message pointing here. Tests may use
  only synthetic or expressly-consented minimal fixtures.

**Explicitly gated on data-owner sign-off (NOT done autonomously):**

- Removing the existing recordings from the tree, and any history rewrite, are
  outward-facing/destructive and require the data owner to first establish
  provenance, consent, purpose, retention authority, and redistribution
  rights. The review is correct that rewriting `origin` does not erase forks
  and clones; coordination is required. This ADR records the controls and the
  required follow-up; it does not delete the data.

## Consequences

- No new CSI captures can be committed (ignore + policy check).
- The existing tracked recordings remain until the owner decides; the incident
  is documented and the guard prevents worsening it.
- CI gains one fast policy job; contributors get a local pre-commit check.

## Validation

- Policy-check unit tests: a staged `*.csi.jsonl` fails; a synthetic fixture
  under an allowed test path passes; the check is deterministic and offline.
- Manual confirmation that the new ignore globs cover both active directories.

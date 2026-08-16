# ADR-299: Repository CSI data-incident controls — ignore rules and a pre-commit/CI policy check

- **Status**: Accepted — controls and current-tree remediation implemented; history coordination pending
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

**Owner-authorized current-tree remediation (2026-08-15):**

- The data owner authorized removal of the six known CSI capture and metadata
  files from the current tree. The removal is recoverable from Git history and
  does not claim to erase existing clones, forks, caches, or release artifacts.
- Any history rewrite remains a separate coordinated incident-response action.
  It requires an inventory of affected refs and releases, downstream notice,
  credential and artifact review, and an explicit execution plan.

## Consequences

- No new CSI captures can be committed (ignore + policy check).
- The six known tracked recordings are absent from the current tree. Historical
  copies remain until a separately authorized and coordinated history rewrite.
- CI gains one fast policy job; contributors get a local pre-commit check.

## Validation

- Policy-check unit tests: a staged `*.csi.jsonl` fails; a synthetic fixture
  under an allowed test path passes; the check is deterministic and offline.
- Manual confirmation that the new ignore globs cover both active directories.
- `bash scripts/csi-data-policy-check.sh --tracked` passes after the authorized
  current-tree removal.

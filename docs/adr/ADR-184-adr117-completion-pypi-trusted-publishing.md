# ADR-184: Complete ADR-117 via PyPI Trusted Publishing (OIDC) + real v2.0.0 / ruview publish

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-07-21 |
| **Deciders** | ruv |
| **Codename** | **PHOENIX-LANDING** — the PIP-PHOENIX wheel that never actually took off |
| **Relates to** | [ADR-117](ADR-117-pip-wifi-densepose-modernization.md) (PIP-PHOENIX modernization — this ADR completes it), [ADR-028](ADR-028-esp32-capability-audit.md) (witness chain), [ADR-115](ADR-115-home-assistant-integration.md) (HA/Matter sibling), [ADR-168](ADR-168-benchmark-proof.md) (measured-not-claimed house style) |
| **Tracking issue** | [#785](https://github.com/ruvnet/RuView/issues/785) (ADR-117, still OPEN) |

---

## 1. Context

ADR-117 (PIP-PHOENIX) designed the v2.0.0 rewrite of the pip `wifi-densepose`
package as a PyO3 + maturin compiled wheel over the Rust core, plus a `ruview`
sibling package, replacing the 11.5-month-stale pure-Python `1.1.0` line. The
code landed on `main` (the `python/` workspace: `Cargo.toml`, `src/bindings/*.rs`,
the `wifi_densepose/` Python package, `tests/`, `bench/`). The tombstone shipped.
**But the release itself is broken and the design doc's own P5 intent was never met.**

This ADR is a **gap analysis and remediation plan**, not a new feature. Every fact
below was verified against PyPI and GitHub Actions on 2026-07-21; none are projected.

### 1.1 What is actually live on PyPI (measured)

`pip index versions wifi-densepose` returns:

```
wifi-densepose (1.99.0)
Available versions: 1.99.0, 1.2.0, 1.1.0, 1.0.0
```

- `1.99.0` — the tombstone wheel **is genuinely live**. `import wifi_densepose`
  raises `ImportError` pointing users to 2.0+. This part of ADR-117 §7.2 shipped.
- `2.0.0a1` — appears in PyPI's release history as a **pre-release** (hidden from
  the default `pip index` view, surfaced with `--pre`). It is still an **alpha**.
- `2.0.0` (stable) — **does not exist.** ADR-117's headline deliverable
  (`pip install wifi-densepose==2.0.0`) is not installable.

`pip index versions ruview` returns:

```
ERROR: No matching distribution found for ruview
```

The `ruview` sibling package **was never published.** Commit `b71d243b4`
(*"feat(adr-117): publish wifi-densepose 2.0.0a1 + ruview 2.0.0a1 to PyPI"*) claims
a publish that did not happen for that package — a real **claimed-vs-measured gap**
of exactly the kind [ADR-168](ADR-168-benchmark-proof.md) and the project's
"prove everything" posture exist to catch.

### 1.2 Why the release pipeline was stuck (measured; interim-fixed — see §1.4)

`gh run list --workflow pip-release` shows the last **4** runs all
`conclusion=failure` (most recent `2026-05-24T16:34`). The full failure log for
run `26366735779` (job *"Publish v1.99 tombstone"* → step *"Publish to PyPI"*)
shows two things:

1. The publish step uses `pypa/gh-action-pypi-publish` with a `password`
   (API-token) input and fails:

   ```
   403 Forbidden — Invalid or non-existent authentication information.
   ```

   i.e. the `PYPI_API_TOKEN` GitHub secret is stale / expired / revoked.

2. The action's own log warns:

   ```
   Warning: the workflow was run with 'attestations: true' ... but an explicit
   password was also set, disabling Trusted Publishing.
   ```

The workflow at `.github/workflows/pip-release.yml` wires `password:
${{ secrets.PYPI_API_TOKEN }}` into **four** publish steps (lines 249, 258, 282,
291) and declares only `permissions: contents: read` (line 49–50). So it is using a
rotatable, leak-able, expire-able API token in exactly the place ADR-117 §5.4 / §5.5
and the issue #785 P5 row explicitly called for **OIDC Trusted Publishing** ("cp310
… abi3-py310, OIDC"; ADR-117 §5.5 line 547: *"PyPI publish via Trusted Publisher
(OIDC, no API token in secrets)"*). **The implementation drifted from its own
design doc.**

### 1.3 Why the package is still alpha (measured)

`python/pyproject.toml` pins `version = "2.0.0a1"` (line 13) and
`Development Status :: 3 - Alpha` (line 26). Issue #785's closing criteria
(§"Done") require `wifi-densepose==2.0.0` (**not** alpha) published, plus all 10
acceptance criteria in §11. None of those can be true today given §1.1–§1.2.

**Why this matters:** ADR-117 is the sole Python entry point for the whole RuView
ecosystem (per its §2 "PyPI org presence check"). A stale token silently blocking
every release means the entire "plug-and-play Python entry point for the pip +
Jupyter customer base" thesis (issue #785 "Strategic alignment") is stalled behind a
one-line credential problem — and a commit message claims otherwise.

### 1.4 Interim fix applied (2026-07-21) — credential unblocked, migration still pending

**As of 2026-07-21T22:57:29Z the stale-credential symptom is fixed at the credential
layer.** The maintainer fetched a valid `PYPI_TOKEN` from GCP Secret Manager (project
`cognitum-20260110`) and ran `gh secret set PYPI_API_TOKEN` to replace the
revoked/expired value. Authentication was confirmed non-destructively via a
`twine upload --skip-existing` re-upload of the existing `1.99.0` tombstone artifacts,
which returned a benign 400/skip response (not the previous `403 Forbidden`) — proving
the new token authenticates correctly.

This means **token-based publishing works again today** — the `403` root cause
described in §1.2 no longer reproduces. It does **not**, however, close this ADR:

- A **manually-rotated token still expires, leaks, and can be revoked over time** — it
  re-introduces exactly the silent-failure mode that blocked the last 4 runs. It is a
  stopgap at the same layer as the §3.2 fallback, not the durable fix.
- The OIDC **Trusted Publishing migration (§3, P1) remains the decision** — a
  credential PyPI mints per-run with no secret to rotate is the only fix that removes
  the recurring-expiry class of failure.
- The other three gaps are **untouched** by this rotation: `wifi-densepose` is still
  `2.0.0a1` (not stable `2.0.0`), and `ruview` is still unpublished.

**Why/How to apply:** read §1.2's "root cause" as *diagnosed and temporarily
mitigated*, not *still broken*. A reviewer re-running the §7.5 check today may now see
a green token-based run — that is expected and does not satisfy this ADR, which is
Accepted only when §6's criteria pass **and** the workflow no longer carries a static
token (§7.4).

---

## 2. Current state — evidence

| Artifact | Value | Source |
|---|---|---|
| Latest stable `wifi-densepose` on PyPI | **1.99.0** (tombstone) | `pip index versions wifi-densepose` |
| `wifi-densepose==2.0.0` stable | **absent** | `pip index versions` (not listed) |
| `wifi-densepose==2.0.0a1` pre-release | present (alpha) | PyPI release history (`--pre`) |
| `ruview` on PyPI | **No matching distribution found** | `pip index versions ruview` |
| `pip-release.yml` last 4 runs | all `failure` | `gh run list --workflow pip-release` |
| Most recent failed run | `2026-05-24T16:34` | `gh run list` |
| Failing step | Publish v1.99 tombstone → Publish to PyPI | run `26366735779` log |
| Failure code | `403 Forbidden — Invalid or non-existent authentication information` | run `26366735779` log |
| Root cause | `PYPI_API_TOKEN` stale/revoked; explicit password disables Trusted Publishing | run `26366735779` log warning |
| `password:` uses in workflow | 4 (lines 249, 258, 282, 291) | `.github/workflows/pip-release.yml` |
| Workflow permissions | `contents: read` only (no `id-token: write`) | `pip-release.yml:49–50` |
| pyproject version | `2.0.0a1` | `python/pyproject.toml:13` |
| pyproject dev status | `3 - Alpha` | `python/pyproject.toml:26` |
| Issue #785 | **OPEN** | GitHub |

**Why/How to apply:** treat this table as the falsifiable baseline. A reviewer who
re-runs each `Source` command must reproduce each `Value`, or this ADR is wrong and
should be revised before any remediation is attempted.

---

## 3. Decision

Complete ADR-117 by closing four gaps, in order:

1. **Migrate `pip-release.yml` to PyPI Trusted Publishing (OIDC)** — as the durable
   end-state, drop all four `password: ${{ secrets.PYPI_API_TOKEN }}` inputs, grant
   `id-token: write` to the publish jobs, and add `environment: pypi`. This removes
   the rotatable/expire-able credential and realigns with ADR-117 §5.5's stated OIDC
   intent. **This is gated behind sub-phase P1b** (§5): the switch is inert — and in
   fact 403-breaking — until the manual pypi.org registration (§3.1) exists, so the
   OIDC change must land *together* with that registration. Until then, token auth
   (the freshly-rotated `PYPI_API_TOKEN`, §1.4) is the correct active path and is
   what the `RuView#786-pypi-token-auth` fix-marker guard enforces. An OIDC migration
   was attempted (`cc153e8b5`) and reverted (`82d5c7339`) for exactly this reason.

2. **Promote `wifi-densepose` from `2.0.0a1` to stable `2.0.0`** in
   `python/pyproject.toml` (version + `Development Status :: 5 - Production/Stable`)
   and record the promotion in `CHANGELOG.md`.

3. **Actually publish `ruview==2.0.0`** — the sibling package that commit
   `b71d243b4` claimed but never shipped — and verify it with `pip index versions`.

4. **Adopt issue #785 §11's 10 acceptance criteria verbatim as this ADR's own
   acceptance criteria** (§6 below), and only flip ADR-117 → Accepted and close
   #785 once every one passes against the real index — proven, not claimed.

### 3.1 Mandatory human prerequisite (cannot be automated)

**Trusted Publishing requires a one-time manual step on `pypi.org` that no CLI, API,
or agent can perform** — PyPI restricts Trusted Publisher configuration to the
project owner via the web UI for security reasons. Before P1's workflow change can
succeed, a human with owner rights on both PyPI projects must:

1. Log in to `pypi.org`.
2. For **`wifi-densepose`**: Project → *Publishing* → *Add a new pending/trusted
   publisher* → GitHub, with:
   - Owner: `ruvnet`
   - Repository: `RuView`
   - Workflow filename: `pip-release.yml`
   - Environment: `pypi`
3. Repeat the identical step for the **`ruview`** project. Because `ruview` is not
   yet on PyPI, register it as a **pending publisher** (PyPI supports configuring a
   trusted publisher for a project name before its first release — the first OIDC
   publish then creates the project).

**Why/How to apply:** the workflow change in P1 is inert until this is done — the
publish step will fail with a "no trusted publisher configured" error rather than a
403. Land P1 and this manual step together; do not tag a release expecting OIDC to
work until a human confirms both entries exist. Treat this section as a blocking
checklist item on the release-day runbook, not a footnote.

### 3.2 Fallback path (if the owner declines Trusted Publishing)

If the maintainer prefers not to adopt OIDC yet, the code-side remediation is a
**token regeneration**, not a redesign:

- Generate a fresh PyPI API token (scoped to the `wifi-densepose` and `ruview`
  projects) and store it in GCP Secret Manager (project `cognitum-20260110`, where
  the project's tokens live), then `gh secret set PYPI_API_TOKEN` from it, following
  the existing runbook referenced in the workflow header (`docs/integrations/pypi-release.md`).
- Keep the current `password:`-based workflow unchanged.

**Why/How to apply:** this path clears the 403 and unblocks releases immediately,
but it re-introduces the exact failure mode this ADR is trying to eliminate — a
credential that silently expires and blocks the whole Python entry point again. Use
it only as a stopgap; the Trusted Publishing migration (P1) is the durable fix and
should remain the default recommendation.

---

## 4. Detailed design — workflow migration

The change to `.github/workflows/pip-release.yml` is small and surgical. It does
**not** touch the build matrix (`build-wheels`, `build-sdist`, `build-tombstone`
jobs are unchanged — the 403 is a publish-credential problem, not a build problem).

### 4.1 Grant OIDC token permission on the publish jobs

The `gh-action-pypi-publish` action mints its OIDC token from the job's
`id-token: write` permission. The current top-level `permissions: contents: read`
must be extended on the two publish jobs (`publish-v2`, `publish-tombstone`) — plus
the future `publish-ruview` job:

```yaml
publish-v2:
  name: Publish v2 wheels
  needs: [build-wheels, build-sdist]
  permissions:
    id-token: write        # ← added: mint the OIDC token for PyPI
    contents: read
  environment: pypi         # ← added: binds to the PyPI trusted-publisher entry
```

### 4.2 Drop the `password:` inputs

Every publish step loses its `password:` line. Trusted Publishing needs no secret —
the action exchanges the job's OIDC token for a short-lived PyPI upload token
automatically:

```yaml
# BEFORE (current — fails with 403 when the token is stale)
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}   # ← remove
    packages-dir: dist

# AFTER (Trusted Publishing — no secret, activates once the pypi.org entry exists)
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist
```

The TestPyPI dry-run steps keep `repository-url: https://test.pypi.org/legacy/` and
likewise drop `password:` — a matching trusted-publisher entry must be registered on
`test.pypi.org` if the dry-run path is to be used (otherwise gate the dry-run behind
the fallback token or remove it).

**Why/How to apply:** the header comment block (lines 16–23) that documents the
`PYPI_API_TOKEN` / GCP-Secret-Manager runbook must be rewritten to document the
Trusted Publishing setup instead, so the next maintainer does not re-add a token
"to fix" a future failure and silently re-disable OIDC.

### 4.3 Add the `publish-ruview` job

`ruview` is published by a new job mirroring `publish-v2` (same `id-token: write` +
`environment: pypi`, no `password:`), gated on a `ruview`-scoped build. Because the
package has never shipped, its first successful OIDC publish creates the PyPI
project against the pending trusted-publisher entry from §3.1.

---

## 5. Phase ledger

```
P1  ──►  P1b  ──►  P2  ──►  P3  ──►  P4
token    OIDC      version   real     close
unblock  (gated)   promote   publish  #785
```

### P1 — Credential unblock (token auth, active)

- [x] Rotate `PYPI_API_TOKEN` to a validated token (§1.4, `gh secret set`, verified
  `2026-07-21T22:57:29Z` via `twine upload --skip-existing`). Token-based publishing
  works today.
- [x] Keep `password: ${{ secrets.PYPI_API_TOKEN }}` as the active auth path,
  satisfying the `RuView#786-pypi-token-auth` fix-marker guard.
- [ ] Rewrite the `pip-release.yml` header comment block so the next maintainer
  knows OIDC is the intended P1b end-state (not a token to keep re-rotating forever).

> Note: an OIDC migration was attempted (`cc153e8b5`) and **reverted** (`82d5c7339`)
> because it tripped the fix-marker guard before the pypi.org registration existed.
> The OIDC work is therefore tracked as P1b below, not P1. See the Status note.

**Status 2026-07-21 — DESIGNED then REVERTED (token auth is the ACTIVE path):**
The OIDC migration was implemented (commit `cc153e8b5` — `id-token: write` +
`environment: pypi` on both publish jobs, all four `PYPI_API_TOKEN` password
inputs removed) but then **reverted** (commit `82d5c7339`) after it tripped the
pre-existing `RuView#786-pypi-token-auth` fix-marker guard
(`scripts/fix-markers.json`). That guard `require`s
`password: ${{ secrets.PYPI_API_TOKEN }}` and `forbid`s `id-token: write`
precisely because a half-activated OIDC path (id-token permission present, but no
Trusted Publisher yet registered on pypi.org) leaves publishing **403-broken**
rather than working — it correctly predicted this exact failure. The revert was
verified locally against the real checker (`python scripts/check_fix_markers.py` →
all 25 markers pass, exit 0) before pushing.

**Active path today:** token-based auth via the freshly-rotated `PYPI_API_TOKEN`
(§1.4). The current `pip-release.yml` (HEAD `82d5c7339`) carries
`password: ${{ secrets.PYPI_API_TOKEN }}` at four publish steps plus a TODO
comment marking the OIDC follow-up. The OIDC switch is therefore **not** done — it
moves to sub-phase P1b below.

**Why this revert was correct (measured, not claimed):** OIDC is the better
long-term design and matches ADR-117's original §5.5 P5 intent — but implementing
it *before* the manual pypi.org registration exists would have shipped a workflow
that looks migrated yet 403s on the next real publish. The fix-marker caught a
well-intentioned improvement that wasn't the honest, currently-working state, and
it was reverted rather than overridden. That is the same "measured not claimed"
discipline (per [ADR-168](ADR-168-benchmark-proof.md)) this entire ADR exists to
enforce — applied here to our own change.

### P1b — Switch to OIDC Trusted Publishing (gated follow-up)

- [ ] **(human, manual, pypi.org — BLOCKING)** Complete the §3.1 Trusted Publisher
  registration for BOTH `wifi-densepose` and `ruview` (owner=ruvnet, repo=RuView,
  workflow=pip-release.yml, environment=pypi). P1b must not start until this exists.
- [ ] Re-apply the `cc153e8b5` change (add `id-token: write` + `environment: pypi`,
  drop the four `password:` inputs) as its own follow-up commit.
- [ ] Update the `RuView#786-pypi-token-auth` fix-marker in `scripts/fix-markers.json`
  in the *same* commit — invert it to `require: id-token: write` / `forbid:
  password: ${{ secrets.PYPI_API_TOKEN }}` — so the guard tracks the new intended
  state instead of blocking it (referencing the TODO comment now in pip-release.yml).
- [ ] Confirm a green OIDC publish before removing the token, per §3.2's
  keep-both-paths recommendation (OIDC first, token fallback until OIDC is proven).
- [ ] No capability gap: publishing must keep working across the P1→P1b transition.

### P2 — Version promotion + changelog

- [ ] `python/pyproject.toml`: `version = "2.0.0"` (drop the `a1` suffix).
- [ ] `python/pyproject.toml`: `Development Status :: 5 - Production/Stable`.
- [ ] `CHANGELOG.md`: `[Unreleased]` entry — "wifi-densepose 2.0.0 promoted from
  alpha; ruview 2.0.0 first stable publish; pip-release migrated to Trusted Publishing".
- [ ] Confirm the `ruview` package's own version metadata is set to `2.0.0`.

### P3 — Real publish + verification

- [ ] Cut tag `v2.0.0-pip` (per the workflow's `v*-pip` trigger) → OIDC publish of
  the `wifi-densepose` wheel matrix.
- [ ] Publish `ruview==2.0.0` via the new `publish-ruview` job.
- [ ] Run every command in §7 against the **real** PyPI index and capture output.
- [ ] Generate + commit `expected_features_v2.sha256` (issue #785 §11 criterion 10),
  resolving ADR-117 §11.3 / the workflow header's Q3 note.

### P4 — Close issue #785

- [ ] All 10 acceptance criteria (§6) pass against the real index.
- [ ] Flip ADR-117 §Status → **Accepted**.
- [ ] Flip this ADR (ADR-184) §Status → **Accepted**.
- [ ] Close issue #785.

**Why/How to apply:** the phases are strictly ordered — P3 cannot succeed until both
P1 (working credential path) and the §3.1 human step are done, and P4 must not be
marked complete on the strength of a commit message (the failure mode this ADR
exists to correct). Nothing in this ledger is checked; this is a Proposed plan.

---

## 6. Acceptance criteria (verbatim from issue #785 §11)

A reviewer must be able to:

1. `pip install --pre wifi-densepose==2.0.0a1` from PyPI test index → wheel installs
   without compile step on Linux/macOS/Windows
2. `python -c "import wifi_densepose; print(wifi_densepose.__version__, wifi_densepose.__rust_version__)"`
   → both versions print
3. `python -c "from wifi_densepose import CsiFrame; ..."` → core type round-trips
   through PyO3
4. `python -c "from wifi_densepose import vitals; vitals.detect_hr(...)"` → 4-stage
   pipeline runs on a sample CSI buffer
5. `pip install wifi-densepose[client]; python -c "import wifi_densepose.client; ..."`
   → WS client connects to a running sensing-server
6. `pytest python/tests/` → ≥30 tests pass (smoke + binding round-trips)
7. `maturin build --release --strip` → wheel under 5 MB per platform (ADR §5.4 budget)
8. `wifi-densepose==1.99.0` is the latest 1.x; `import wifi_densepose` raises
   `ImportError` with migration URL
9. `wifi-densepose==1.0.0` is yanked from PyPI; `1.1.0` is un-yanked with deprecation
   notice (90-day window)
10. Witness `expected_features_v2.sha256` generated in CI, committed alongside the
    existing `archive/v1/data/proof/`, re-verifiable from Python via
    `wifi_densepose.verify_witness(...)`

**Note (amendment to criterion 1):** issue #785 §11 was written when `2.0.0a1` was
the target. This ADR promotes to stable `2.0.0`, so criterion 1 is read as
`pip install wifi-densepose==2.0.0` (no `--pre`) against the production index. The
`--pre`/`a1` wording is preserved verbatim above per the transcription requirement;
the stable form is what P3/P4 must actually satisfy. This ADR additionally requires
`ruview==2.0.0` to be installable (the sibling package from commit `b71d243b4`),
which #785 §11 did not enumerate but the issue "Done" section implies.

---

## 7. How to verify (prove, don't claim)

Exact commands a reviewer runs to prove — not assume — each gap is closed. Every one
produces falsifiable output; capture it in the PR that flips ADR-117 to Accepted.

### 7.1 Both packages live and stable

```bash
# wifi-densepose 2.0.0 (stable, NOT alpha) must appear
pip index versions wifi-densepose
#   expect: "wifi-densepose (2.0.0)" and 2.0.0 in the available list

# ruview 2.0.0 must now exist (currently: "No matching distribution found")
pip index versions ruview
#   expect: "ruview (2.0.0)"
```

### 7.2 Clean-venv install + import (criteria 2–4)

```bash
python -m venv /tmp/verify-184 && . /tmp/verify-184/bin/activate
pip install wifi-densepose==2.0.0        # stable, no --pre
python -c "import wifi_densepose; print(wifi_densepose.__version__, wifi_densepose.__rust_version__)"
python -c "from wifi_densepose import CsiFrame; print(CsiFrame([1.0]*56,[0.0]*56,56,0,100.0))"
python -c "from wifi_densepose import vitals; print(hasattr(vitals,'detect_hr'))"
pip install ruview==2.0.0
python -c "import ruview; print(ruview.__version__)"
```

### 7.3 Tombstone still guards the 1.x line (criterion 8)

```bash
pip install wifi-densepose==1.99.0
python -c "import wifi_densepose" 2>&1 | grep -q "github.com/ruvnet/RuView" \
  && echo "PASS: tombstone raises with migration URL" \
  || echo "FAIL"
```

### 7.4 Workflow auth state

**Current state (P1, active today):** token auth is the working path and is what
the `RuView#786-pypi-token-auth` fix-marker requires. The honest check today is
that token auth is present and the fix-marker guard passes:

```bash
# token auth present (the ACTIVE, working path — expected PASS today)
grep -q 'password: ${{ secrets.PYPI_API_TOKEN }}' .github/workflows/pip-release.yml \
  && echo "PASS: token auth active" || echo "FAIL"

# fix-marker regression guard must pass
python scripts/check_fix_markers.py && echo "PASS: all markers pass"
```

**P1b end-state (after the manual pypi.org registration):** the checks below flip
to PASS *only once P1b lands together with the fix-marker inversion* — they are
**not** expected to pass today and their passing now would mean a half-migrated,
403-prone workflow:

```bash
# after P1b: no static token should remain in the publish steps
grep -nE 'password:|PYPI_API_TOKEN' .github/workflows/pip-release.yml \
  && echo "not yet: token still present (expected during P1)" \
  || echo "P1b done: no static token"

# after P1b: id-token permission granted on publish jobs
grep -q 'id-token: write' .github/workflows/pip-release.yml \
  && echo "P1b done: OIDC permission present" \
  || echo "not yet: OIDC not enabled (expected during P1)"
```

### 7.5 The release actually went green

```bash
gh run list --workflow pip-release --limit 1
#   expect: conclusion=success on the v2.0.0-pip tag run
```

**Why/How to apply:** §7.1 and §7.5 together are the minimal proof that the two
headline gaps (no stable 2.0.0, no `ruview`, dead pipeline) are closed. If any
command's actual output diverges from the `expect` line, the corresponding phase is
not done — regardless of what any commit message or checkbox says.

---

## 8. Consequences

### Positive

- The Python entry point for the entire RuView ecosystem (issue #785 "Strategic
  alignment") is unblocked with a credential that cannot silently expire.
- The claimed-vs-measured gap in commit `b71d243b4` (`ruview` never published) is
  closed with reproducible proof, upholding the project's "prove everything" posture.
- Trusted Publishing removes a leak-able long-lived secret from CI entirely — the
  security posture ADR-117 §5.5 originally specified.
- ADR-117 / issue #785 can finally reach a defensible Accepted/closed state instead
  of sitting open behind a one-line token failure.

### Negative

- The `pypi.org` trusted-publisher registration (§3.1) is a hard human dependency
  with no automated fallback beyond re-introducing a token (§3.2). Release day is
  blocked on a person, not a pipeline.
- Promoting to stable `2.0.0` removes the alpha escape hatch — any binding bug now
  ships under a stable version and needs a `2.0.1`, not a new `a`-tag.
- `test.pypi.org` needs its own trusted-publisher entry if the dry-run path is kept,
  adding a second manual registration.

### Neutral

- The build matrix (`build-wheels`, `build-sdist`, `build-tombstone`) is untouched;
  the risk surface of this change is confined to the three publish jobs.
- The witness-hash-v2 open question (ADR-117 §11.3, workflow header Q3) is pulled
  into scope as criterion 10 but is orthogonal to the credential migration.

---

## 9. References

- **ADR-117** — `docs/adr/ADR-117-pip-wifi-densepose-modernization.md` (the design
  this ADR completes; §5.4/§5.5 OIDC intent, §7.2 tombstone, §11.3 witness hash)
- **Issue #785** — https://github.com/ruvnet/RuView/issues/785 (tracking issue,
  OPEN; §11 acceptance criteria transcribed in §6)
- **Workflow** — `.github/workflows/pip-release.yml` (four `password:` inputs at
  lines 249/258/282/291; `contents: read` only at 49–50)
- **pyproject** — `python/pyproject.toml` (`version = "2.0.0a1"` line 13;
  `3 - Alpha` line 26)
- **Failed run** — GitHub Actions `pip-release` run `26366735779`, job "Publish
  v1.99 tombstone" → step "Publish to PyPI" (403 + Trusted-Publishing-disabled warning)
- **Commit `b71d243b4`** — *"feat(adr-117): publish wifi-densepose 2.0.0a1 + ruview
  2.0.0a1 to PyPI"* — the `ruview` publish it claims did not occur
- **PyPI Trusted Publishing** — https://docs.pypi.org/trusted-publishers/ (web-UI-only
  registration; pending-publisher support for not-yet-created projects)
- **`pypa/gh-action-pypi-publish`** — https://github.com/pypa/gh-action-pypi-publish
  (OIDC via `id-token: write`; `password:` disables Trusted Publishing)
- **ADR-168** — `docs/adr/ADR-168-benchmark-proof.md` (measured-not-claimed house style)

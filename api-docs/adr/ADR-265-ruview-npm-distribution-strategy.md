# ADR-265: RuView npm Distribution Strategy — CI Gate, Provenance, Version Single-Sourcing, Namespace

| Field | Value |
|-------|-------|
| **Status** | Accepted — **D1–D4 implemented**: `.github/workflows/npm-packages.yml` (matrix gate: tests, version-literal grep, pack-content/size gate, tarball-install smoke test, README claim-check), `.github/workflows/ruview-npm-release.yml` (publish-from-CI with `npm publish --provenance`), version single-sourcing (all three packages read package.json), `ruview` bin owned by `@ruvnet/ruview` (`@ruv/ruview-cli` bin renamed `ruview-cli`), `ci.yml` NODE_VERSION 18→20. D5 (no workspace) stands as recorded |
| **Date** | 2026-07-02 |
| **Deciders** | ruv |
| **Codename** | **RUVIEW-NPM-DIST** |
| **Supersedes / amends** | none (cross-cutting layer above ADR-263 and ADR-264; complements ADR-182 P3/P4) |

## Context

The monorepo now ships (or stages) **three Node packages** with no shared
distribution engineering:

| Package | Dir | Published | Bin(s) | Tests in CI |
|---------|-----|-----------|--------|-------------|
| `@ruvnet/ruview` | `harness/ruview/` | 0.1.0 (live) | `ruview` | **none** |
| `@ruvnet/rvagent` | `tools/ruview-mcp/` | 0.1.0 (live) | `rvagent`, `ruview-mcp` | **none** |
| `@ruv/ruview-cli` | `tools/ruview-cli/` | private | `ruview` (collides) | **none** |

Cross-cutting facts established during the ADR-263/264 reviews:

- **Zero CI coverage.** No workflow under `.github/workflows/` references any of
  the three directories. Two of the packages are *live on the registry* and were
  published from a laptop state CI never saw. Meanwhile the Rust side has a
  1,031+-test gate and a witness-bundle culture (ADR-028) — the npm surface is
  the only shipped artifact class with no verification gate at all.
- **`ci.yml` pins `NODE_VERSION: '18'`** while all three packages declare
  `engines.node >= 20`.
- **Version triplication.** Each package hardcodes its version in source at
  least once beyond package.json (harness `SERVER_INFO`, rvagent
  `PACKAGE_VERSION`, cli `.version("0.0.1")`).
- **Bin-name collision.** Two packages claim the `ruview` bin.
- **No provenance.** Neither published package carries npm provenance
  attestations, in a project whose differentiator is signed, reproducible
  evidence (ADR-028 witness bundles, ADR-182 P4 ed25519/SLSA design).
- **No pack-content gate.** ADR-264 F1/F2 (broken `require` target, 33% dead map weight — MEASURED, tarball listing — and a phantom
  `CHANGELOG.md` in `files`) are exactly the defect class an
  `npm pack --dry-run` assertion catches in seconds.

## Decision

Adopt one distribution layer for all Node packages. Per-package code fixes live
in ADR-263/264; this ADR fixes the machinery around them.

### D1 — One `npm-packages.yml` CI workflow (the gate)

Matrix over `[harness/ruview, tools/ruview-mcp, tools/ruview-cli]` ×
Node `[20, 22]`:

1. `npm ci` where a lockfile is committed (the TS packages); the harness
   installs with `npm install` — repo policy gitignores lockfiles under
   `harness/`, and the package is dependency-free after ADR-263 O3 so there is
   nothing to pin.
2. `npm test` (harness: `node --test test/*.test.mjs` — pin the glob form,
   the directory form fails on Node 22; TS packages: build + jest or `node:test`
   per ADR-264 O8).
3. **Pack gate:** `npm pack --dry-run --json` asserted against a checked-in
   expected file list + a max unpacked-size budget per package (harness ≤ 60 kB;
   rvagent ≤ 130 kB post ADR-264 O2). Any new/missing/renamed shipped file is a
   reviewed diff, not a surprise.
4. **Tarball smoke test:** install the packed tarball into a temp dir; run
   `ruview --version`, `ruview doctor`, `rvagent` `--help`-equivalent, and a
   Node `import()` of each declared export condition — this is the test that
   would have caught ADR-264 F1 (`require` → nonexistent `dist/index.cjs`).
5. Bump `ci.yml` `NODE_VERSION` to `'20'` (independent of the matrix above).

### D2 — Publish only from CI, with provenance

Manual `npm publish` from laptops stops. A tag-triggered workflow
(`ruview-npm-release.yml`, mirroring the firmware release discipline) runs the
D1 gate, then `npm publish --provenance --access public` under the GitHub OIDC
token. Consequence: every published version is attested to a public commit +
workflow run — the npm-side analogue of the ADR-028 witness bundle. The
`prepublishOnly` script in each package runs the pack gate locally as a
belt-and-braces (publishing outside CI fails loudly, not silently).

### D3 — Version single-sourcing

Rule: **package.json is the only place a version string lives.** Runtime code
reads it (`createRequire(import.meta.url)('./package.json').version` or a
build-time define for the TS packages). CI greps for `\d+\.\d+\.\d+` literals in
`src/` of each package and fails on match (allowlist: test fixtures). This
retires ADR-263 F6 and ADR-264 F9 permanently instead of per-incident.

### D4 — Namespace and bin ownership

- `@ruvnet/ruview` **owns the `ruview` bin** (it is the published front door,
  ADR-182). `@ruv/ruview-cli` renames its bin or folds into `rvagent`
  (ADR-264 O9) — decided here so neither package ADR relitigates it.
- New Node packages in this repo use the `@ruvnet/` scope (the `@ruv/` scope
  holds `rvcsi` legacies; do not grow it).
- Every package README + description must pass
  `npx ruview claim-check` — enforced in the D1 gate. The guardrail package
  linting its sibling packages' claims is the cheapest dogfooding we have
  (ADR-264 F3 is the standing example of why).

### D5 — Shared-code policy (bounded)

Do **not** introduce an npm workspace or a shared runtime package yet: three
packages, two of which may merge (ADR-264 O9), do not justify workspace
machinery, and the harness's zero-dep property is load-bearing. Revisit if a
fourth package appears or if the `http/cog/config` duplication survives the
ADR-264 O9 fold. Record the duplication as intentional in each file header (the
CLI already does this).

## Consequences

- The npm artifacts get the same class of gate the Rust workspace has had since
  ADR-028: no publish without tests, no shipped file set without an asserted
  manifest, no version without provenance. The two defects that reached the
  registry (broken `require` condition, dead maps) become CI-impossible.
- Cold-path costs stay near zero: the D1 matrix is 6 fast jobs (the harness
  suite runs in ~108 ms MEASURED; TS builds dominate at a few tens of seconds).
- Publishing gains one constraint (must go through CI) and loses one failure
  mode (laptop-state publishes) — the right trade for a project whose brand is
  reproducible evidence.
- D3's grep gate is blunt but cheap; if it over-fires, scope it to
  `version`-adjacent identifiers before weakening it.
- Follow-ups tracked elsewhere: per-package code fixes (ADR-263 O1–O8, ADR-264
  O1–O9); ADR-182 P4 (metaharness router + ed25519 provenance chain) remains
  the deeper provenance story that D2's npm attestations complement, not
  replace.

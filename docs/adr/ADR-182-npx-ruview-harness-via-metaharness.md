# ADR-182: `npx ruview` — A RuView Agent Harness Minted via MetaHarness

| Field | Value |
|-------|-------|
| **Status** | Accepted — **P1+P2 implemented & validated** (`harness/ruview/`, 17/17 tests, MCP handshake + `ruview.verify` PASS against the real repo, packs to 16.7 kB / 21 files) · P3 publish-ready (name decision pending) · P4 (router + provenance) designed |
| **Date** | 2026-06-17 |
| **Deciders** | ruv |
| **Codename** | **RUVIEW-HARNESS** |
| **Builds on** | MetaHarness (`metaharness@0.1.15`, `@metaharness/kernel`, `@metaharness/host-*`, `@metaharness/router`), the `ruview-*` Claude Code subagents (`ruview-onboarding-guide`, `ruview-config-engineer`, `ruview-training-engineer`), the `wifi-densepose` CLI (`calibrate`/`enroll`/`train-room`/`room-watch`), the sensing-server, ADR-028 (witness verification), ADR-095/096 (rvCSI runtime), ADR-260/262 (RuField bridge) |
| **Supersedes** | none |

## Context

RuView (WiFi-DensePose) is a deep stack — 15 Rust crates, an ESP32 firmware line,
a sensing-server, a CLI, ~180 ADRs, a calibration pipeline, training recipes, and a
hard cultural rule that **every claim must be independently reproducible** (the
"prove everything" ethos, after the project was accused of AI-slop). The barrier to
entry is correspondingly steep: a newcomer who wants to "set up WiFi sensing" must
discover the right firmware variant, provision an ESP32 over a Windows-only Python
subprocess, point it at the sensing-server, run `calibrate` → `enroll` →
`train-room`, and know which numbers are MEASURED vs CLAIMED. We already encode this
knowledge as **Claude Code subagents** (`ruview-onboarding-guide`,
`ruview-config-engineer`, `ruview-training-engineer`) — but those only exist inside
*this* repo's `.claude/agents/`, only on Claude Code, and only for someone who has
already cloned the monorepo.

Separately, this session shipped **MetaHarness** (`metaharness@0.1.15`): a tool that
*"mints a custom AI agent harness from any repo"*, runnable on **9 hosts**
(claude-code, codex, pi-dev, hermes, openclaw, rvm, copilot, opencode,
github-actions) over a wasm-primary / NAPI-RS-fallback **kernel**, with a
**cost-optimal model router** (`@metaharness/router`, the productized DRACO Phase-2
k-NN finding) and ed25519/SLSA/SBOM provenance baked in. Crucially, MetaHarness
**already ships a `vertical:ruview` template** in its template list. That template
is generic scaffolding; it is not wired to RuView's actual tools, agents, or the
"prove everything" guardrails.

The gap: **there is no single, host-portable, provenance-signed entry point that
gives any user an AI agent that actually knows how to operate RuView.** A user
should be able to run one command —

```bash
npx ruview
```

— in an empty directory (or alongside an ESP32) and get an agent harness that can
onboard them, configure firmware, drive a live capture, train a room model, and
**refuse to overstate accuracy** — on whichever coding host they already use.

## Decision

**Mint a first-class RuView agent harness from this repo using MetaHarness, harden
its `vertical:ruview` template into a RuView-specific harness with a real MCP tool
surface and the project's honesty guardrails, and publish it as `npx ruview`.**

`npx ruview` is *not* a new runtime. It is a **thin, versioned distribution** of a
MetaHarness harness: the kernel + host adapters + a RuView "genome" (skills, agents,
MCP tools, guardrails) generated from and pinned against this monorepo. The harness
is the product; `npx ruview` is the front door.

### Why mint-from-repo instead of hand-writing a harness

MetaHarness's value here is exactly the work we would otherwise hand-roll across 9
hosts: host-specific config (`.claude/settings.json` MCP + hooks for claude-code,
the codex/copilot/opencode equivalents), the kernel that abstracts wasm-vs-native,
the cost router, and the provenance chain. We write the **RuView knowledge once** as
host-neutral genome assets; MetaHarness projects them onto each host adapter. This
also keeps the harness regenerable: when the CLI or an ADR changes, re-mint and
re-pin rather than maintaining 9 divergent copies.

### What the harness contains (the RuView genome)

1. **Skills / playbooks** (host-neutral markdown, projected to each host's skill
   format):
   - `onboard` — zero-to-sensing path picker (Docker demo / repo build / live
     ESP32), the physics caveats, the hardware table. Port of
     `ruview-onboarding-guide`.
   - `provision-node` — ESP-IDF v5.4 Windows-subprocess build/flash/provision flow
     (the exact MSYSTEM-stripped invocation from `CLAUDE.local.md`), firmware
     variant selection (8MB display / 4MB no-display / C6), NVS + WiFi + channel /
     MAC-filter overrides (ADR-060).
   - `calibrate-room` — `baseline → enroll → extract → train` via the
     `wifi-densepose` CLI (`calibrate`/`calibrate-serve`/`enroll`/`train-room`/
     `room-watch`, ADR-151).
   - `train-pose` — camera-supervised + camera-free training, the MEASURED-vs-CLAIMED
     discipline, the mean-pose baseline check (ADR-079, ADR-152, ADR-181).
   - `verify` — run the witness bundle + Python proof (`verify.py` → VERDICT: PASS),
     ADR-028.
   - Ports of `ruview-config-engineer` and `ruview-training-engineer`.

2. **MCP tool surface** (`@metaharness/kernel`-hosted MCP server, one schema per
   capability — see "MCP tools" below). This is what makes the harness *operate*
   RuView, not just talk about it.

3. **Guardrails** (the differentiator): the harness's system prompt and a
   pre-output hook enforce the "prove everything" rule — accuracy numbers must be
   tagged MEASURED (with a reproducer) or CLAIMED; the agent must run the mean-pose
   baseline before quoting PCK; firmware fixes are never presented as
   hardware-validated without a real boot log (the exact discipline this session
   followed for `v0.8.1-esp32`).

4. **Host adapters** — claude-code first (P1), then codex / opencode / copilot /
   pi-dev / hermes / rvm / github-actions (P3+), each via the published
   `@metaharness/host-*` package.

5. **Router** — `@metaharness/router` routes each step to the cheapest adequate
   model (e.g. a var-rename or a log-grep → Haiku; calibration-math reasoning or a
   security review → Sonnet/Opus), mirroring the repo's 3-tier routing (ADR-026).

### MCP tools (the operational surface)

| Tool | Wraps | Purpose |
|------|-------|---------|
| `ruview.onboard` | docs + agent | Pick a setup path, print the next concrete command |
| `ruview.node.flash` | ESP-IDF subprocess (ADR `CLAUDE.local.md`) | Build + flash a firmware variant to a COM port |
| `ruview.node.provision` | `provision.py` | Set SSID/password/target-ip/channel/MAC-filter over serial |
| `ruview.node.monitor` | pyserial | Stream boot log; assert CSI is flowing (MGMT+DATA) |
| `ruview.server.up` | sensing-server | Start the Axum sensing-server (`:3000`/`:5005`/`:8765`) |
| `ruview.calibrate` | `wifi-densepose calibrate`/`enroll`/`train-room` | Run the ADR-151 room pipeline |
| `ruview.room.watch` | `wifi-densepose room-watch` | Live presence/vitals from a trained room |
| `ruview.verify` | `scripts/generate-witness-bundle.sh` + `verify.py` | Produce/verify the witness bundle (must be N/N PASS) |
| `ruview.claim.check` | static lint | Scan output for untagged accuracy claims; flag MEASURED-vs-CLAIMED |

Each tool returns structured JSON and is fail-closed: a tool that cannot prove its
result (e.g. `ruview.node.monitor` sees no CSI callbacks) returns an honest negative,
never a fabricated success — consistent with the RuField `map_privacy` fail-closed
posture (ADR-262 §3.3).

### The mint + pin flow (how the harness is produced)

```bash
# P1 — mint from this repo, claude-code host, RuView vertical
npx metaharness ruview --template vertical:ruview --host claude-code \
  --from-existing . --description "RuView WiFi-sensing operator agent" \
  --target ./harness/ruview

# readiness + fit/cost/safety scorecards (ADR-041) — gate before publish
npx metaharness genome .        # 7-section repo readiness
npx metaharness score .  --json # fit / cost / safety
npx metaharness analyze .       # recommended harness plan (no-exec)
```

The minted harness is committed under `harness/ruview/` and **pinned** (kernel +
host-adapter + router versions locked) so `npx ruview` is reproducible. Re-minting on
a CLI/ADR change is a reviewed PR, not an implicit regeneration.

### Distribution: `npx ruview`

A small published package whose `bin` boots the pinned harness via the kernel:

- **Preferred name:** `ruview` (currently **free** on npm — verified 2026-06-17).
- **Risk:** npm's typosquat filter may reject `ruview` as too close to `review` /
  `preview` (this session hit exactly that on `ruvn`→`levn`/`raven` and
  `worldgraph`→`world-graph`). **Fallback:** publish scoped `@ruvnet/ruview` (also
  free) and/or `npx ruvnet/ruview` straight from GitHub. Decide at publish time;
  do not unpublish to rename (the 24-h name-lock lesson from `worldgraphs`).
- `bin: { "ruview": "bin/cli.js" }` — note **`bin/cli.js`, not `./bin/cli.js`** (npm
  strips the `./` form; this broke `ruvn@0.1.0` this session).
- `npx ruview` with no args → `onboard` skill (interactive path picker).
  `npx ruview <skill> [...]` → run a specific skill. `npx ruview --host codex` →
  install the harness into an existing repo for that host.

## Architecture

```
            npx ruview                      (thin bin — boots the pinned harness)
                │
        @metaharness/kernel                 (wasm primary · NAPI-RS native fallback)
        ├── host adapter  ── claude-code | codex | opencode | copilot | pi-dev | hermes | rvm | github-actions
        ├── @metaharness/router             (k-NN cost-optimal model routing — DRACO P2 / ADR-026)
        └── RuView genome  (pinned)
            ├── skills      onboard · provision-node · calibrate-room · train-pose · verify
            ├── mcp tools   ruview.node.* · ruview.calibrate · ruview.room.watch · ruview.verify · ruview.claim.check
            └── guardrails  MEASURED-vs-CLAIMED · mean-pose baseline · no-unvalidated-firmware-claims
                │
        RuView assets (the real system the agent drives)
        ├── wifi-densepose CLI       calibrate / enroll / train-room / room-watch
        ├── sensing-server           :3000 / :5005 / :8765
        ├── ESP-IDF subprocess       build / flash / provision / monitor  (COM8/COM9/COM12)
        └── witness bundle + verify.py
```

Provenance: the harness ships an **ed25519 witness + SBOM (SPDX) + SLSA** chain
(MetaHarness already does this for minted harnesses), so a recipient can verify the
RuView harness was built from a specific monorepo commit — the agentic analogue of
the firmware witness bundle (ADR-028).

## Phases

- **P1 — Mint & pin (claude-code).** `npx metaharness ruview --template
  vertical:ruview --from-existing . --host claude-code`. Port the three `ruview-*`
  subagents into host-neutral genome skills. Commit under `harness/ruview/`, pin
  versions. Acceptance: `npx metaharness score .` ≥ threshold; the harness can run
  `onboard` and `verify` end-to-end locally.
- **P2 — MCP tool surface.** Implement the `ruview.*` MCP tools over the kernel
  (start with `onboard`, `verify`, `claim.check`, `node.monitor` — the read-only /
  proving tools), then the mutating ones (`node.flash`, `provision`, `calibrate`).
  Acceptance: `ruview.verify` returns the witness bundle PASS as structured JSON;
  `ruview.claim.check` flags a seeded untagged "100% accuracy" string.
- **P3 — Publish `npx ruview` + multi-host.** Publish the bin package (name decision
  per Distribution). Add codex / opencode / copilot / pi-dev / hermes / rvm /
  github-actions adapters. Acceptance: `npx ruview` cold-starts on ≥3 hosts and runs
  `onboard`; provenance verifies.
- **P4 — Router + guardrail hardening.** Wire `@metaharness/router`; calibrate the
  3-tier routing on a RuView task set. Make the MEASURED-vs-CLAIMED guardrail a hard
  pre-output gate. Acceptance: a benchmark of RuView tasks shows cost reduction vs
  all-Opus with no quality regression; the guardrail blocks an untagged accuracy
  claim in a red-team prompt.

## Consequences

**Positive**
- One reproducible, signed entry point (`npx ruview`) that operates RuView on the
  host the user already has — onboarding goes from "clone a 15-crate monorepo" to a
  single `npx`.
- The "prove everything" ethos becomes **executable**, not just documentation: the
  harness *enforces* MEASURED-vs-CLAIMED and the mean-pose baseline.
- Knowledge written once (host-neutral genome) instead of 9× per host; regenerable
  from the repo as the system evolves.
- Dogfoods MetaHarness on a hard real vertical, surfacing bugs back to
  `agent-harness-generator` (this session already filed #9–#13 there).

**Negative / risks**
- **Drift:** a pinned harness goes stale as the CLI/ADRs move; mitigated by a
  re-mint-on-change PR ritual and a CI check that the genome's referenced
  CLI flags still exist.
- **Surface area:** mutating MCP tools (`node.flash`, `provision`) touch hardware and
  the network — must be permission-gated and fail-closed; the firmware-flash tool
  must never claim hardware validation without a captured boot log.
- **Name/typosquat:** `ruview` may be rejected at publish; scoped fallback decided in
  P3. Do not unpublish-to-rename.
- **Host parity:** not all 9 hosts support MCP + hooks equally; the guardrail gate
  may degrade to advisory on weaker hosts — must be disclosed in the badge, not
  hidden (same honesty principle as ADR-181's backend badge).
- **Windows-coupled tooling:** the ESP-IDF flow is Windows-subprocess-specific
  today; the `node.*` tools are gated to that environment until a cross-platform
  path exists.

## Alternatives considered

1. **Keep the `ruview-*` subagents repo-local (status quo).** Zero new surface, but
   stays Claude-Code-only and clone-gated; no portable front door. Rejected — it's
   the gap this ADR exists to close.
2. **Hand-write a bespoke `npx ruview` harness (no MetaHarness).** Full control, but
   re-implements the kernel, 9 host adapters, the router, and the provenance chain
   we already ship — months of duplicated work and 9 divergent configs to maintain.
   Rejected.
3. **Use the generic `vertical:ruview` template as-is.** It's scaffolding with no
   real tools or guardrails — it would *talk about* RuView without being able to
   *operate* it or enforce honesty. Rejected as insufficient; P2 is precisely the
   hardening that makes it real.
4. **Ship only an MCP server (no harness/host adapters).** Covers tools but not the
   skills, routing, guardrails, or multi-host projection — a strictly smaller subset
   of this design. Folded in as the P2 layer rather than the whole.

## Open questions

- Final published name: bare `ruview` vs scoped `@ruvnet/ruview` vs GitHub-only
  `npx ruvnet/ruview` — resolve against the typosquat filter at P3.
- Does the harness bundle the `wifi-densepose` binary, shell out to a user-installed
  one, or offer both? (Leaning: shell out; print install guidance if absent.)
- Where do the `node.*` hardware tools live for non-Windows users — defer, or wrap
  the rvCSI runtime (ADR-095/096) which is cross-platform Rust?
- Should `ruview.verify` gate `npx ruview` self-tests in CI (harness can't publish if
  the witness bundle regresses)?
- Relationship to the RuField MFS harness surface (ADR-260/262) — one harness with a
  RuField skill, or a sibling `npx rufield`?

## References

- MetaHarness: `metaharness@0.1.15` (`npx metaharness`, templates incl.
  `vertical:ruview`; hosts: claude-code/codex/pi-dev/hermes/openclaw/rvm/copilot/
  opencode/github-actions), `@metaharness/kernel`, `@metaharness/router`,
  `@metaharness/host-*`, repo `github.com/ruvnet/agent-harness-generator`.
- RuView subagents: `ruview-onboarding-guide`, `ruview-config-engineer`,
  `ruview-training-engineer` (`.claude/agents/`).
- ADR-026 (3-tier model routing), ADR-028 (witness verification), ADR-041
  (MetaHarness scorecards), ADR-060 (channel / MAC-filter overrides), ADR-079
  (camera ground-truth training), ADR-095/096 (rvCSI runtime), ADR-151 (per-room
  calibration), ADR-152/181 (WiFlow / browser pose), ADR-260/262 (RuField bridge).

# Tick 7 — 2026-05-22 05:14 UTC

**Thread:** R14 (empathic appliances)
**Verdict:** Speculative 10-20y vision note with concrete vertical sketches, ethical framework, privacy threat model, and infrastructure-gap inventory.

## What shipped

- `docs/research/sota-2026-05-22/R14-empathic-appliances.md` — research note covering:
  - Three concrete vertical sketches (stress-responsive lighting / adaptive HVAC / attention-respecting conversational appliances) with timelines (5y / 10y / 15y).
  - **Infrastructure inventory** — which existing RuView components map to which empathic-appliance category. 5 ✅ in-repo, 4 ⚠️/❌ to-build.
  - Ethical framework (opt-in-by-default, data-stays-on-device, override-one-tap) committed in writing as constraints any product must honour.
  - 6-row privacy threat model with concrete mitigations.
  - Honest scope: lab-condition literature doesn't validate real-home generalisation; no per-occupant identity yet; appliance integration half is out of repo scope.

## Why this matters for the loop

R14 is the **first explicitly speculative** vision thread (R5/R7/R8/R9/R10/R12 were all experimental or physics). It catalogues the **product-level surface area** for the longest-horizon items, which informs:

- Which sensing primitives we should invest in next (per-room baseline learner is the clearest gap).
- Which ADRs to write next (consent/override is a separate ADR — possibly ADR-105).
- Which MCP tools to add to `@ruv/ruview-mcp` (the deferred `ruview_vitals_subscribe` is now the highest-leverage next addition per ADR-104 + R14).

## Connections established

- R14 explicitly cross-links to R5 (saliency is task-specific), R8 (CSI required, not RSSI), R7 (adversarial poisoning defence), ADR-104 (hands-off appliance API surface), ADR-103 (per-room occupancy gate).
- The infrastructure-gap inventory (5 in-repo, 4 to-build) is a useful artefact for any future product roadmap discussion.

## Coordination

`ticks/tick-7.md` convention. No PROGRESS.md touch.

## Major notes from prior tick

R10 (PR not auto-created due to bash flow issue) ended up committed directly to main and pushed in this tick. Future-tick reminder: always check `git branch --show-current` before `git commit`. The cron prompt assumes branch hygiene that the bash plumbing sometimes breaks under back-to-back tick pressure.

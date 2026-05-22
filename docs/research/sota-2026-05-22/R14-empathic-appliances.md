# R14 — Empathic appliances: physiological-state-aware home automation

**Status:** speculative 10-20y vision note · **2026-05-22**

## Premise

We already ship a contactless breathing-rate detector (`v1/v2` sensing-server, ADR-029 multistatic fusion). Breathing rate is a documented proxy for arousal/stress in clinical studies (e.g. Bernardi 2002, Vlemincx 2013) and predicts user states finer than HRV in low-SNR conditions. Heart rate is captured concurrently.

The 10-20 year question: **what happens when every appliance with a CPU and a WiFi radio knows the occupant's physiological baseline + current state, and modulates its behaviour to support the occupant's wellbeing?**

The current RuView stack provides the *sensing primitives* (breathing rate, heart rate, occupancy, motion intensity, RSSI-only counting per R8). What it doesn't yet provide is the *intent-action layer* — an appliance that says "the occupant has been breathing fast for 8 minutes; their normal baseline is 12 BPM; let me dim the lights and lower the music."

## Three concrete vertical sketches

### V1 — Stress-responsive lighting (next 5y, technically tractable)

| Sensing | Action |
|---|---|
| Breathing rate 50% above 7-day rolling baseline for >5 min | Lights gently warm-shift (Kelvin: 4000K → 2700K) and dim 10% over 60s |
| Sustained low motion + low breathing variability (rest state) | Lights stay where they are |
| Sleep onset detected (motion=null, breathing<10 BPM for >15 min) | Lights fade to 0 over 8 min following standard Philips Hue "wind down" curve |

The hard part is **not** the sensing — it's the **personalisation**: a 7-day rolling baseline takes a week of continuous occupancy data to calibrate, and per-person baselines vary by ~30%. Solution: federated per-room calibration that learns continuously, with explicit "this is not me" override.

### V2 — Adaptive HVAC for thermal-stress envelopes (10y)

Thermal stress affects breathing-rate envelope (>30°C → +20% baseline RR). A learned per-person mapping from `(room_temp, humidity, breathing_rate)` → "is the occupant uncomfortable?" lets HVAC pre-emptively adjust before the occupant consciously notices. Saves ~15-20% on cooling energy per published HVAC-personalisation studies (Aryal & Becerik-Gerber 2018), while improving comfort.

### V3 — Conversational appliances respecting attention state (15y)

A smart speaker that **doesn't interrupt** when the occupant's breathing pattern shows high cognitive load (focused reading: shallow + regular). The sensing already exists; the appliance integration is the gap.

Honest scope check: this requires that someone publishes both (a) a reliable shallow-breathing-during-focus signature, and (b) a hands-off way for appliances to receive that signal. RuView ships (a)'s building blocks; (b) needs an MCP-style standard which **ADR-104 (`@ruv/ruview-mcp`)** is the first step toward.

## Required infrastructure (already in repo or close)

| Component | Status | Used for |
|---|---|---|
| Breathing/heart rate detector | ✅ shipped | physiological state signal |
| Occupancy presence | ✅ shipped (`cog-pose-estimation`, `cog-person-count`) | "is anyone there?" gate |
| Motion intensity score | ✅ shipped | activity-state classifier input |
| Per-room baseline learner | ⚠️ partial (RollingP95 in #491 is the closest existing primitive) | personalised normalisation |
| State-classifier model | ❌ not built | maps `(breathing, heart, motion)` → state |
| MCP appliance API | ✅ partial (ADR-104) | hands-off appliance integration |
| Consent/opt-in machinery | ❌ not built | ethical baseline |
| Override/correction UI | ❌ not built | user-in-the-loop |

The four ❌/⚠️ items are the actual work for V1 ship-readiness. Roughly 1-2 quarters of dedicated effort, not a research project.

## Ethical framework (drafted, not normative)

Empathic appliances raise three explicit consent questions that smart-speaker-vendors so far have *not* answered well. Any RuView-based empathic-appliance product should commit to all of these in writing:

1. **Opt-in by default.** Sensing is on only if the occupant has actively enabled it. Default = off, not buried in settings.
2. **Data stays on-device.** The breathing-rate stream is the most invasive biometric in the building. Per-second values **must never** leave the local appliance/Cognitum Seed. Only **aggregate state** (e.g. "stressed" / "neutral" / "asleep") may be exposed to integrations, and only via the user's explicit MCP grant.
3. **Override is one tap.** A physical "stop sensing now" gesture or button must work without WiFi, without speech, without the cloud. If consent withdraws, sensing pauses for ≥1 hour before re-asking.

These three constraints are surprisingly load-bearing — they rule out the most common smart-home failure modes (always-on listening, cloud-side aggregation, opaque consent flows).

## Privacy threat model

| Threat | Mitigation |
|---|---|
| Compromised appliance leaks breathing rate continuously | Per-device sensing is opt-in; appliances default off |
| MCP API exposes raw signal to integrations | Only aggregate state passes the MCP boundary; raw stays local (ADR-104 §"Output validation") |
| Adversarial CSI poisoning makes the occupant look stressed/calm against their interest | R7 Stoer-Wagner multi-link consistency detects this |
| Long-term baseline learning enables individual identification across moves | Baseline is per-installation; no cloud sync; user can wipe at any time |
| Insurance / employer access to physiological state | Legal/contractual barrier; not solvable purely technically. Surface this explicitly in onboarding |
| Children / non-consenting cohabitants | Per-occupant opt-in, not per-installation. Use existing pose-based identity primitives (R3/R9/R15) to gate per-person |

## Honest scope

- The clinical literature on breathing-rate-as-stress-proxy is mostly **lab-condition adults**. Real-home generalisation isn't proven.
- We have no per-occupant identity model yet — single-occupant scenarios only until R3/R15 mature.
- The "appliance integration" half is mostly out of repo scope; it requires partner appliances that accept ADR-104-style MCP signals.

## What this DOES enable

- A clear product roadmap from the **existing sensing primitives** to a **shippable category of appliance behavior** that doesn't exist in the market today.
- A worked ethical framework that's specific enough to commit to in marketing copy.
- A mapping of which existing repo components map to which appliance category (V1/V2/V3).

## What this DOES NOT enable

- Stress detection without breathing-rate signal. Pure CSI motion isn't a reliable stress proxy.
- Detection of psychological states that aren't reflected in breathing/heart rate (cognitive fatigue, mood). Those need physiological signals we can't measure passively.

## Connection back

- **R5** (saliency) — empathic appliance state classification will have its own task-specific saliency, different from counting and structure-detection.
- **R8** (RSSI-only) — V1 lighting only needs breathing rate, which requires CSI. V3 conversational requires the per-subcarrier shape lost in band-mean. **R14 is CSI-only**, not RSSI-feasible — bounds the rollout to ESP32-S3-class deployments.
- **R7** (multi-link consistency) — directly relevant to the adversarial-poisoning threat in the privacy table.
- **ADR-104** (`@ruv/ruview-mcp`) — the actual hands-off appliance API. Empathic-appliance integrations subscribe via MCP `ruview_vitals_subscribe` (not yet built; see HORIZON.md deferred list).
- **ADR-103** (`cog-person-count`) — the per-room occupancy gate ("only do empathic actions when an occupant is present and consented").

## Next ticks

- Per-room baseline learner module (extend `RollingP95` to cover breathing-rate + heart-rate over 7-day windows).
- State-classifier model architecture (3-class: stressed / neutral / asleep — simple MLP over breathing/heart/motion features).
- MCP tool `ruview_vitals_subscribe` — the hands-off integration that lets a partner appliance subscribe to the aggregate state stream.
- ADR for the consent-default-off, override-one-tap, no-cloud-sync constraints. Possibly ADR-105.

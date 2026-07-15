# RuView HOMECORE vs Home Assistant — Performance & Capability Benchmark

**Measured:** 2026-05-31 · Windows 11, Docker Desktop 28.5.1 (WSL2 Linux engine) · single host.
**Reproduce:** `python aether-arena/staging/run_homecore_bench.py` and `python aether-arena/staging/run_ha_bench.py`.

HOMECORE is RuView's **wire-compatible Rust port of Home Assistant's core** (ADR-125…ADR-134): the
same `/api` REST + WebSocket surface, the same SQLite recorder schema, an automation engine, a
HomeKit bridge, a WASM plugin runtime, and a voice/assist pipeline — plus **native WiFi/RF sensing
entities** (presence, breathing, heart-rate, pose) that Home Assistant can only get through external
add-ons. Because the API is wire-compatible, the two can be measured head-to-head on the same client.

> **Read this honestly.** HOMECORE (`0.1.0-alpha`) is a young, focused core; Home Assistant is a
> mature platform with ~3,000 integrations and a decade of ecosystem. HOMECORE's thesis is **not**
> "more features" — it is **the same control plane at 1/35th the memory and 18× the startup speed,
> with RF sensing built in.** The numbers below quantify exactly that trade.

## Performance (measured)

| Metric | RuView HOMECORE `0.1.0-alpha` | Home Assistant `stable` | Advantage |
|--------|------------------------------:|------------------------:|-----------|
| **Cold start → API/web ready** | **0.55 s** | 9.72 s | **18× faster** |
| **Idle resident memory (RSS)** | **10.1 MB** | 359 MB | **35× leaner** |
| **Distribution size** | **4.7 MB** (single static binary) | 610 MB (container image) | **130× smaller** |
| **Idle CPU** | 0.0 % | 0.0 % | tie |
| **REST latency p50** | 2.13 ms | 2.95 ms | comparable¹ |
| **REST latency p95** | 22.9 ms | 27.3 ms | comparable¹ |
| **REST latency p99** | 26.2 ms | 28.3 ms | comparable¹ |
| **REST throughput (1 conn, sequential)** | **1,599 req/s** | 716 req/s | **2.2×** |
| **Recorder DB after boot** | 36.9 KB | 4.1 KB | — (HOMECORE seeds 10 demo entities + history) |
| **Process threads (idle)** | 22 | n/a (containerized Python) | — |

¹ **Latency caveat — read before quoting.** The two latency rows are *not* the same endpoint.
HOMECORE is measured on **authenticated `/api/states`** (returns 10 live entities). Home Assistant's
`/api/*` requires a completed onboarding flow + long-lived access token, so HA is measured on the
**unauthenticated `/manifest.json`** served by the same aiohttp stack. Both are single-connection,
300-sample, sequential. Treat latency as "same order of magnitude"; treat **memory, startup, and
size as the decisive, apples-to-apples results.** Throughput is endpoint-confounded the same way —
the 2.2× is directional, not a controlled isolate.

### What the deltas mean in practice
- **10 MB vs 359 MB RSS:** HOMECORE runs comfortably on a Pi Zero 2 W or an ESP32-class gateway
  alongside the sensing pipeline; HA effectively needs a Pi 4/5 or x86 to itself.
- **0.55 s vs 9.7 s start:** HOMECORE can be cold-started per-request or restarted on config change
  without a noticeable outage; HA's ~10 s boot (longer with real integrations) makes it a
  long-lived daemon only.
- **4.7 MB vs 610 MB:** OTA-updating the whole control plane over a metered/rural link is trivial
  for HOMECORE; HA ships as a ~250 MB compressed image.

## Capability & feature comparison

| Capability | RuView HOMECORE | Home Assistant |
|-----------|-----------------|----------------|
| HA-compatible REST `/api` | ✅ wire-compatible subset (ADR-130) | ✅ reference implementation |
| HA-compatible WebSocket API | ✅ (ADR-130) | ✅ |
| State machine + event bus + service registry | ✅ 13 seeded services (ADR-127) | ✅ |
| SQLite recorder (history) | ✅ HA-compat schema **+ ruvector semantic search** (ADR-132) | ✅ (no vector search) |
| Automation engine + Jinja templates | ✅ MiniJinja trigger/condition/action (ADR-129) | ✅ (full Jinja2) |
| HomeKit (Apple Home) bridge | ✅ scaffold (ADR-125) | ✅ mature |
| Plugin/integration runtime | ✅ **sandboxed WASM** plugins (ADR-128) | ✅ Python integrations (in-process, unsandboxed) |
| Voice / intent / "Assist" | ✅ 5 built-in intents **+ ruflo agent bridge** (ADR-133) | ✅ Assist + LLM agents |
| Migration from existing HA | ✅ reads HA `.storage/` + `automations.yaml` (ADR-134) | n/a |
| **Native WiFi/RF sensing entities** | ✅ **presence, breathing, HR, 17-kp pose, fall** as first-class sensors | ⚠️ only via external add-on/MQTT |
| Integration ecosystem breadth | ⚠️ early — core + WASM plugins | ✅ ~3,000 integrations, HACS |
| Mature web UI / dashboards (Lovelace) | ❌ not yet | ✅ extensive |
| Add-on store / supervised OS | ❌ | ✅ HAOS + Supervisor |
| Community / docs maturity | ⚠️ alpha | ✅ very large |
| Memory / startup / footprint | ✅✅ (see table) | ⚠️ heavy |
| Language / safety | Rust (memory-safe, single static binary) | Python (interpreted, large dep tree) |

### Where each wins
- **HOMECORE wins:** resource footprint, cold-start, distribution size, throughput-per-MB, memory
  safety, sandboxed (WASM) plugins, and — uniquely — **WiFi/RF sensing as native entities**. Ideal
  for edge gateways, battery/solar nodes, and shipping the control plane *with* the sensor.
- **Home Assistant wins:** integration breadth, UI/dashboard maturity, add-on ecosystem, community
  support, and production track record. Ideal as a full-house hub on a Pi 4/5+ or x86.

## Honest summary

For the **shared, wire-compatible HA control plane**, HOMECORE delivers it at **~35× less RAM,
~18× faster startup, and ~130× smaller footprint**, with WiFi sensing built in and HA-config
migration on the way. What it does **not** yet match is Home Assistant's enormous integration
catalog and UI maturity. The right read is **"HA-compatible core, edge-class resource budget,
RF-native"** — not "HA replacement." For a sensing node that also needs to *be* a smart-home hub,
HOMECORE's efficiency is decisive; for a feature-complete whole-home hub today, Home Assistant
remains the broader platform.

## Reproduction & method

- **HOMECORE:** `v2/target/release/homecore-server.exe` (`0.1.0-alpha.0`), bound to `127.0.0.1:8124`,
  SQLite file recorder, dev-token auth (`Authorization: Bearer …`). Startup = `Popen` → first `200`
  on `/api/`. RSS/CPU via `psutil` after a 2 s settle. 300-sample sequential latency on `/api/states`.
- **Home Assistant:** `ghcr.io/home-assistant/home-assistant:stable` in Docker, `-p 8125:8123`,
  fresh `/config`. Startup = container start → first `<500` on `/manifest.json`. RSS/CPU via
  `docker stats --no-stream` after a 20 s settle. 300-sample sequential latency on `/manifest.json`.
- Both runs are single-host, single-connection, no concurrency tuning. Numbers are indicative of
  the **resource/startup class**, which is the property that differs by orders of magnitude;
  latency/throughput are reported with the endpoint caveat above and should not be over-read.
- Harness scripts: `aether-arena/staging/run_homecore_bench.py`, `aether-arena/staging/run_ha_bench.py`.

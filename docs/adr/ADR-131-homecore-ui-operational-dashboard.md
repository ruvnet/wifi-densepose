# ADR-131: HOMECORE-UI — Operational dashboard for the two-tier Cognitum stack

| Field | Value |
|-------|-------|
| **Status** | Accepted — UI implemented (§10); full backend wiring specified (§11–§12) |
| **Date** | 2026-06-14 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-UI** — first-class operator dashboard inside the Cognitum Appliance shell |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE state machine), [ADR-128](ADR-128-homecore-integration-plugin-system.md) (HOMECORE-PLUGINS), [ADR-129](ADR-129-homecore-automation-engine.md) (automation engine), [ADR-130](ADR-130-homecore-rest-websocket-api.md) (HOMECORE-API), [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) (recorder/semantic search), [ADR-151](ADR-151-room-calibration-specialist-training.md) (room calibration HTTP API), [ADR-100](ADR-100-cog-packaging-specification.md) (Cog packaging), [ADR-116](ADR-116-cog-ha-matter-seed.md) (cog-ha-matter), [ADR-069](ADR-069-cognitum-seed-csi-pipeline.md) (SEED RVF ingest), [ADR-105](ADR-105-federated-csi-training.md) (federated CSI training) |
| **Tracking issue** | TBD |
| **Parent** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (sub-ADR, HOMECORE-127…134 family) |

---

## 1. Context

HOMECORE (ADR-126 through ADR-134) is the native Rust + WASM + TypeScript port of Home Assistant running as the hub on the Cognitum v0 Appliance. As of P2, the state machine ([ADR-127](ADR-127-homecore-state-machine-rust.md)), API ([ADR-130](ADR-130-homecore-rest-websocket-api.md)), and COG runtime ([ADR-128](ADR-128-homecore-integration-plugin-system.md)) are in place. What is missing is a first-class dashboard UI that operators, integrators, and residents can use to manage the full two-tier hardware stack that HOMECORE coordinates.

### 1.1 The two-tier hardware model this UI must represent

This is the most important architectural constraint the UI must carry through every panel:

- **Cognitum SEED** — a Pi Zero 2 W-based edge node. It has its own RVF vector store (8-dim, content-addressed, with kNN queries), Ed25519 witness chain, SHA-256 ingest audit trail, onboard environmental sensors (BME280 temperature/humidity/pressure, PIR motion, reed switch, ADS1115 4-channel ADC, vibration), 13 drift detectors, an MCP proxy (114 tools, JSON-RPC 2.0, default-deny policy), 98 HTTPS API endpoints, and epoch-based swarm sync for multi-SEED deployments. SEEDs sit close to the ESP32 sensing nodes and receive feature vectors from them at 1 Hz. Multiple SEEDs can form a peer mesh. **This is the sensing and memory tier.**
- **Cognitum v0 Appliance** — a Pi 5 + Hailo-10H hub, running at `:9000`. It hosts the COG runtime (`/var/lib/cognitum/apps/`), the HOMECORE state machine and event bus, the calibration service, `ruview-mcp-brain:9876`, `cognitum-rvf-agent:9004`, `ruvector-hailo-worker:50051`, and acts as the fleet coordinator for multi-room correlation and federated training. The Appliance is where HOMECORE runs, and it is what the dashboard user is sitting in front of. **This is the computation and orchestration tier.**

SEEDs are **subordinate nodes that the Appliance supervises** — they are not peers. The UI navigation hierarchy must reflect this: the Appliance is the root, SEEDs are children, ESP32 nodes are leaves.

### 1.2 What the UI is not

HOMECORE-UI is **not** a re-skin of the existing Cognitum Cog Store. It is a full operational dashboard that **extends** the Cognitum platform's shell — the Cog Store, API Explorer, and Guide already exist and must remain intact, with the HOMECORE dashboard added as a first-class navigation section alongside them.

---

## 2. Decision

Build HOMECORE-UI as a **complete** TypeScript + Rust→WASM frontend (per this ADR's §3 and the HOMECORE-127…134 family) that:

1. Lives at `http://cognitum-v0:9000/homecore` (or as a dedicated nav item in the Cognitum Appliance shell).
2. Is visually and stylistically seamless with the existing Cognitum platform — same dark theme, same design tokens, same component patterns as `https://seed.cognitum.one/store`.
3. Drives the HOMECORE REST + WebSocket API ([ADR-130](ADR-130-homecore-rest-websocket-api.md)) and the calibration HTTP API ([ADR-151](ADR-151-room-calibration-specialist-training.md)) for all data.
4. Updates in real-time via the homecore `subscribe_events` WebSocket channel. **The UI must never poll for entity state.**

**This is a decision to deliver the complete operational dashboard — every panel in §4.1 through §4.10, every navigation section in §5, fully wired to live data — not a design-system scaffold or a partial first cut.** A static layout shell with placeholder data is explicitly **out of scope as a deliverable**: the design system (§3) is a means to the complete UI, not an end in itself. The acceptance bar for this ADR is that an operator can drive the full two-tier stack — fleet, entities, rooms, COGs, calibration, events, audit, and settings — from the dashboard, against real APIs, with no panel left as a stub.

### 2.1 `homecore-server` is the single backend-for-frontend (BFF) gateway

The data the dashboard needs is spread across **three backend tiers that are not one process**: (a) `homecore-api` (`/api/*` REST + `/api/websocket`, mounted in `homecore-server`); (b) the **calibration API** (`/api/v1/*`, served by a *separate* binary — `wifi-densepose calibrate-serve` / `wifi-densepose-sensing-server`); and (c) the **SEED device tier + appliance daemons** (RVF vector store, witness chain, onboard sensors, reflex rules, COG supervisor, federation), which are physically separate HTTPS services on the SEED nodes and the appliance.

The browser must talk to **exactly one origin.** Therefore `homecore-server` is promoted to the **single BFF / API gateway** for HOMECORE-UI: it serves the static assets at `/homecore`, serves `homecore-api` at `/api/*`, and **adds a new `/api/homecore/*` namespace** that proxies and aggregates the calibration API and the SEED/appliance tiers server-side. The UI only ever issues same-origin requests; cross-service auth (SEED bearer tokens, calibration tokens) is held by the gateway and **never exposed to the browser**. This collapses the CORS/multi-port problem and gives one place to enforce the long-lived-access-token auth (§4.10).

### 2.2 No mock data in production

The in-browser mock layer that the first UI cut shipped behind DEMO banners (§7.1, prior revision) is **demoted to a dev-only fixture** gated behind an explicit `?demo=1` / `HOMECORE_UI_DEMO=1` flag. The production build wires **every** panel to a real gateway endpoint. The full endpoint contract and the backend work each panel needs are specified in **§11**; the staged path to get there is **§12**. A panel may show an empty/typed-error state when its upstream is down, but it must never silently render fabricated data.

---

## 3. Design system — Cognitum platform conventions

The implementor **must study `https://seed.cognitum.one/store` as the definitive design reference before writing a single line of CSS.** The existing platform's design tokens, extracted from production, are:

### 3.1 Colour palette (CSS custom properties)

| Token | Value | Role |
|---|---|---|
| `--bg` | `#0a0e1a` | page background (very dark navy) |
| `--bg2` | `#111627` | secondary background / nav strip |
| `--card` | `#171d30` | card / panel surface |
| `--card-h` | `#1e2540` | card hover state |
| `--border` | `#252d45` | all border strokes (≈0.67px, subtle) |
| `--t1` | `#e0e4f0` | primary text (near-white) |
| `--t2` | `#8890a8` | secondary / muted text |
| `--t3` | `#505872` | tertiary / disabled text |
| `--cyan` | `#4ecdc4` | primary action colour (Install buttons, live indicators, accents) |
| `--cyan-d` | `rgba(78,205,196,0.15)` | cyan tint background for status badges |
| `--green` | `#6bcb77` | success / online / healthy states |
| `--green-d` | `rgba(107,203,119,0.15)` | green tint background |
| `--amber` | `#d4a574` | warning / stale / degraded states |
| `--amber-d` | `rgba(212,165,116,0.15)` | amber tint background |
| `--red` | `#e06060` | error / offline / veto states |
| `--red-d` | `rgba(224,96,96,0.15)` | red tint background |
| `--purple` | `#a78bfa` | informational / epoch / chain indicators |
| `--purple-d` | `rgba(167,139,250,0.15)` | purple tint background |
| `--r` | `10px` | standard border radius on all cards and panels |

### 3.2 Typography

- `--font`: `'Segoe UI', system-ui, -apple-system, sans-serif` — all body and heading text.
- `--mono`: `'Cascadia Code', 'Fira Code', Consolas, monospace` — all entity IDs, API endpoints, hex values, JSON payloads, COG binary hashes.

### 3.3 Component patterns (from the live Cog Store and API Explorer)

- **Cards**: `background: var(--card)`, `border: 0.67px solid var(--border)`, `border-radius: var(--r)`, `padding: 24px`.
- **Category pills / status badges**: small `border-radius: 4–6px`, uppercase text, coloured background tint (e.g. `background: var(--cyan-d); color: var(--cyan)` for `RUNNING`; `background: var(--amber-d); color: var(--amber)` for `STALE`).
- **Primary action buttons**: `background: var(--cyan)`, `color: var(--bg)`, no border — matching the existing "Install" button style exactly.
- **Secondary / ghost buttons**: transparent background, `border: 1px solid var(--border)`, `color: var(--t1)` — matching the existing "Details" button style.
- **Nav strip**: `background: var(--bg2)`, text items in `--t2`, active item highlighted in `--cyan` with a bottom underline.
- **Featured card gradient borders**: top-edge linear gradient from `var(--cyan)` to `var(--purple)` — replicate for HOMECORE section headers.
- **Live metric cards** (API Explorer status page): icon + large numeric value in `--cyan` or `--green`, label in `--t2` below, on a `var(--card)` background.
- **Method badge pills** on the API Explorer (`GET` in green, `POST` in amber, `AUTH` in purple) — reuse this same pill system for COG status indicators.

The implementor **must not introduce new colours, typefaces, or border radii.** Every component should feel like it was built by the same team that built the Cog Store and the API Explorer. A user navigating from the Cog Store into the HOMECORE dashboard should not notice a visual seam.

---

## 4. UI sections — required panels

### 4.1 System Dashboard (the "home screen")

The always-visible overview panel. Modelled on the API Explorer's live metric cards. All values update in real-time.

- **v0 Appliance health strip** — reuse the exact metric-card pattern from `seed.cognitum.one/status`: one card each for CPU %, RAM usage, Hailo-10H inference load (% utilisation), Hailo temperature, uptime, and the running services (`ruview-mcp-brain:9876`, `cognitum-rvf-agent:9004`, `ruvector-hailo-worker:50051`). Values in `--cyan`, labels in `--t2`. This strip is always at the top — it represents the machine the user is looking at.
- **SEED Fleet overview** — a grid of SEED node cards (one per paired SEED) on the `var(--card)` surface with `var(--border)`. Each card shows: online/offline status pill (green/red), firmware version, epoch number, current vector count, last ingest timestamp, and witness-chain validity badge. A collapsed row shows the SEED's 5 onboard sensors in summary (PIR: yes/no, door: open/closed, temperature from BME280). Offline SEEDs render the entire card with a `--red-d` background tint. Clicking a SEED card navigates to the SEED Detail view (§4.2).
- **ESP32 Node summary** — count of active ESP32 nodes per SEED, current frame rate (target: 100 Hz CSI + 1 Hz feature vectors), and a compact warning list for nodes with known issues (presence_score normalisation anomaly, stale firmware version).
- **COG Runtime status row** — a horizontal strip of status pills for each installed COG on the v0 Appliance. Pill colours follow the existing badge convention: `--green-d`/`--green` for running, `--red-d`/`--red` for failed, `--t3`/`--t2` for stopped. COG name in `--mono`. Clicking a pill navigates to COG Management (§4.6).
- **Event Bus activity indicator** — a small real-time sparkline showing the homecore broadcast channel event rate (events/sec). Indicate channel lag if a subscriber is falling behind the 4,096-event capacity.

### 4.2 SEED Detail View (per-SEED drill-down)

Accessible from the fleet grid. Full-page panel for a single SEED node, using the card + section-header pattern from the Cog Store's detail views.

- **SEED identity header** — `device_id` in `--mono`, firmware version, paired status in green, USB vs WiFi connection mode. A section-header gradient border (cyan → purple, matching the featured card style) visually separates this from Appliance content.
- **Vector Store panel** — current vector count, dimension (8), last kNN query latency, current epoch number, a small sparkline of ingest rate over the last hour, and a storage budget bar showing usage against the 100K working-set target. A "Compact now" button (`POST /api/v1/store/compact`) in ghost style. When usage exceeds 80%, the bar renders in `--amber`.
- **Witness Chain panel** — chain length (SHA-256 entries), last verification timestamp, a one-click "Verify chain" button (`POST /api/v1/witness/verify`), and an "Export attestation bundle" button for regulated deployments. The Ed25519 custody attestation (device-bound keypair, epoch + vector count + witness head) renders here. Chain length in `--purple`, following the existing epoch/chain colour convention.
- **Onboard Sensors panel** — live readings from all 5 sensors in individual sub-cards: BME280 (temperature °C, humidity %, pressure hPa), PIR (motion boolean with last-triggered timestamp), reed switch (open/closed with last-changed timestamp), ADS1115 (4 analog channels with configurable labels), vibration (boolean with last-triggered). These are ground-truth validators against CSI readings and are critical for diagnosing false positives in the mixture-of-specialists. Sensor values in `--cyan`; sensor names in `--t2`.
- **Reflex Rules panel** — the 3 pre-configured rules with current state: `fragility_alarm` (threshold 0.3 → relay actuator), `drift_cutoff` (threshold 1.0), `hd_anomaly_indicator` (threshold 200 → PWM brightness). Show last-fired time for each. The `fragility_alarm` threshold is the most commonly adjusted field and should be editable inline. Rules that have recently fired render with a `--amber-d` background tint.
- **Cognitive Analysis panel** — boundary fragility score (0.0–1.0, from Stoer-Wagner min-cut on the kNN graph) rendered as a progress bar: green below 0.3, amber 0.3–0.6, red above 0.6. High fragility (>0.3) indicates a regime change in the environment and should be visually prominent. Temporal coherence phase boundaries shown as a labelled timeline of detected environment state transitions. kNN graph rebuild cadence indicator (every 10 s).
- **Ingest pipeline status** — which ESP32 nodes feed this SEED, the packet type each is sending (`0xC5110003` native feature vectors vs `0xC5110002` vitals fallback path — distinguished visually since native is preferred), current ingest batch size, flush interval, and bridge path topology (direct vs host-laptop hop). The bridge-hop warning (known architectural limitation) renders in `--amber` since it adds a network hop.

### 4.3 SEED Fleet Map (multi-SEED topology)

For deployments with more than one SEED, a topology view showing the mesh:

- **Node hierarchy diagram** — v0 Appliance at root, SEEDs as second tier (grouped by room/zone), ESP32 nodes as leaves under each SEED. Lines represent active data flows. ESP-NOW mesh sync links between SEEDs shown as dashed lines. Connection health shown via line colour (green/amber/red). All labels in `--mono`.
- **Cross-SEED event deduplication indicator** — for events that span multiple SEEDs (one fall detected by two rooms; one occupant tracked through room A → hallway → room B), show a fusion badge indicating how many SEEDs contributed to the composite event.
- **Federation config** ([ADR-105](ADR-105-federated-csi-training.md)) — federated-learning round coordinator role (which SEED is the round coordinator), current round number, K healthy nodes selected, delta exchange status. **Model deltas only — never raw CSI** is a design invariant that must be labelled explicitly in the UI.

### 4.4 Entity & State Browser

The homecore state machine (`DashMap<EntityId, Arc<State>>`) is the authoritative source of truth. Every COG running on the v0 Appliance contributes entities.

- **Entity list by domain** — grouped by the `domain.` prefix of `EntityId`, using collapsible section headers. The 21 entities per ESP32 node (11 raw + 10 semantic primitives from `cog-ha-matter`) are the most important set. For each entity: current state string (in `--t1`), last-changed timestamp (in `--t3`), attribute map as collapsible JSON in `--mono`, and the Context (`user_id` + `parent_id` causality chain, critical for care/audit deployments). Entity IDs always in `--mono`.
- **SEED provenance badge** — each entity carries a small badge showing its data lineage: which ESP32 node → which SEED → which COG → homecore state machine. This trace is invaluable for debugging false positives and is a **first-class UI element, not a collapsed detail.**
- **Domain filter + semantic search** — filter by domain prefix and, once [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) (homecore-recorder) lands, ruvector-backed semantic search: "when did the living room anomaly score last correlate with a door-open event?" A keyword filter across entity IDs and attribute keys ships in the initial release regardless of [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) status, given entity density; the semantic search layers on top once the recorder lands.
- **Real-time WebSocket feed** — entity states update live via the homecore `subscribe_events` WebSocket command ([ADR-130](ADR-130-homecore-rest-websocket-api.md)). The UI must never poll. Show a broadcast-channel lag indicator; warn visually if the subscriber is falling behind the 4,096-event channel capacity.
- **StateChanged detail panel** — clicking any entity opens a slide-over panel showing the full `StateChangedEvent`: `old_state`, `new_state`, `context.id`, `context.user_id`, and the `context.parent_id` chain rendered as a breadcrumb trail.

### 4.5 RoomState / Sensing Panel

Surfaces the mixture-of-specialists output from the calibration service — the highest-level per-room sensing result. Data comes from `GET /api/v1/room/state?bank=<room_id>` on the v0 Appliance.

- **Per-room cards** — one card per `room_id` on the `var(--card)` surface. Each card shows live `RoomState` JSON fields as sub-rows: presence (occupied/absent chip in green/red with confidence bar), posture (standing/sitting/lying chip with confidence), breathing BPM (numeric in `--cyan` with range indicator 6–30), heart rate BPM (numeric in `--cyan` with range indicator 40–120), restlessness score (0–1 progress bar), and anomaly score (0–1 with normal/anomalous label, bar turns red above a configurable threshold).
- **STALE warning** — when `stale: true` (the specialist bank was trained against a different baseline), render the entire room card with a `--amber-d` background tint and a prominent amber banner reading "Bank stale — baseline has changed" with a direct "Recalibrate room" link into the calibration wizard (§4.7). This is the most common real-world failure mode and **must never be subtle.**
- **VETO indicator** — when `vetoed: true` (anomaly veto suppressed vitals/posture because the window was physically implausible), render the affected specialist slots in `--red` with a "Veto active" label. Values suppressed by veto **must not render as zeros** — they must render as explicitly withheld.
- **Null specialist placeholders** — specialists not yet trained (`null` in the specialist bank) render as "Not trained" placeholders in `--t3` with a small "Calibrate to enable" prompt in ghost style. They are **not** errors.
- **Confidence bars** — each specialist output has a confidence float, shown as a small inline bar (`--cyan` fill) next to the reading. Low confidence (< 0.4) renders the bar in `--amber`.
- **Multi-SEED fusion indicator** — for rooms served by multiple SEEDs, show a small badge indicating how many SEED nodes contributed to the `MultiNodeMixture` for this room's reading.

### 4.6 v0 Appliance COG Management

The v0 Appliance hosts COGs at `/var/lib/cognitum/apps/`. This panel is the operational companion to the existing Cog Store (`seed.cognitum.one/store`). It must match the Cog Store's visual conventions precisely — same card layout, same category pills, same install/detail button pair — because operators will move between the two surfaces.

- **Installed COGs list** — for each COG: `id` and `version` in `--mono`, architecture badge (`arm`/`hailo10` etc., category-pill pattern), status pill (running/stopped/failed/updating in green/grey/red/amber), `binary_sha256` verified badge (Ed25519 signature verification shown as a shield icon in `--green` or `--red`), and PID from the pid file. Actions: start, stop, restart (ghost style), and view `output.log` / `error.log` in a monospace drawer using `--mono`. Edit `config.json` inline with syntax highlighting.
- **COG Store / App Registry** — browsable `app-registry.json` listing. This panel should visually mirror `seed.cognitum.one/store` as closely as possible — same featured-card hero layout, same icon + title + description + category pill + action button structure. One-click install downloads the binary from GCS, verifies `binary_sha256` + `binary_signature`, writes the manifest, and starts the COG. Show which new homecore entities will appear in the state machine after install, as a preview list before confirming.
- **OTA Updates** — a badge count on installed COGs with available updates, matching the "Installed (N)" tab badge convention from the existing Cog Store. Show a diff panel (version change, new entities, config schema changes) before confirming the update.
- **Hailo HEF status** — for COGs with `arch: hailo10`: loaded HEF files on the Hailo-10H, current inference throughput, and `ruvector-hailo-worker:50051` connection status. The RF Foundation Encoder ([ADR-150](ADR-150-rf-foundation-encoder.md)) and neural pose head display here once available.

### 4.7 Calibration Wizard

The full baseline → enroll → train → verify pipeline runs via HTTP against the v0 Appliance ([ADR-151](ADR-151-room-calibration-specialist-training.md)). This is a multi-step guided flow — not a raw API panel. Use a stepped wizard layout with a progress indicator at the top (steps 1–5 as numbered pills, active step in `--cyan`, completed in `--green`, pending in `--t3`).

- **Step 1 — Select room and SEED** — enter a `room_id` name (validated against `[A-Za-z0-9_-]{1,64}`) and select which SEED(s) and ESP32 nodes serve this room from a dropdown populated from the live fleet. Show current CSI ingest health for the selected nodes inline — if frames are not arriving at the expected rate, display an amber warning **before** allowing the operator to proceed. A broken ingest pipeline will silently fail calibration.
- **Step 2 — Baseline capture** — `POST /api/v1/calibration/start`. A large full-width animated progress bar (cyan fill) reads from `GET /api/v1/calibration/status`: frames recorded vs target, ETA in seconds, `z_median` value. If `motion_flagged` is true, overlay an amber banner: "Room must be empty — movement detected." The baseline UUID produced here is the anchor for all future STALE detection for this room — display it in `--mono` once complete so operators can record it.
- **Step 3 — Anchor enrollment** — the 8 anchor labels in enforced order: `empty`, `stand_still`, `sit`, `lie_down`, `breathe_slow`, `breathe_normal`, `small_move`, `sleep_posture`. For each: a human-readable instruction with an illustration, a countdown timer rendered as a circular progress ring in `--cyan`, and an immediate quality-gate result (accepted in green, retry in amber with a reason string). Drive via `POST /api/v1/enroll/anchor` + `GET /api/v1/enroll/status`. After each accepted anchor, show the extracted feature values (mean, variance, breathing_score, heart_score) in a small `--mono` data row so operators can sanity-check the capture. Show overall progress as "N / 8 anchors accepted."
- **Step 4 — Train** — a single `POST /api/v1/room/train` call. Show the 6 specialist results as a checklist: presence (threshold + occupied_var), posture (prototype count), breathing (min_score), heartbeat (min_score), restlessness (calm/active motion values), anomaly (prototype count + scale). Specialists that returned non-null render in `--green`. Null specialists (insufficient anchor data) render in `--amber` with a "Re-enroll missing anchors" prompt linking back to Step 3 for the specific missing labels.
- **Step 5 — Verify live** — display the live `RoomState` for the just-trained room using the same per-room card layout as §4.5. Prompt the operator to stand in the room and verify presence is detected, try sitting/lying to confirm posture, and breathe normally to confirm vitals are in plausible range. A "Confirm and save" button (cyan, primary) closes the wizard; a "Something's wrong — re-enroll" button (ghost) loops back to Step 3.

### 4.8 Event Bus & Automation Feed

- **Live event stream panel** — a virtualized scrolling list of `SystemEvent` variants (`StateChanged`, `EntityRegistered`, `ConfigReloaded`) and notable `DomainEvent`s from the homecore Tokio broadcast channel. Each row shows: event-type pill (coloured by variant), `entity_id` in `--mono`, old state → new state arrow, timestamp, and `context.user_id`. The stream is filterable by entity domain, event type, or source SEED/COG. The filter bar uses the same search-input style as the Cog Store's search field.
- **Context causality breadcrumb** — expanding any event row shows the full Context chain (`context.id` → `parent_id` → `grandparent_id`) as a breadcrumb trail in `--mono`. This is how automation loops become visible without any separate debugging tool.
- **Automation builder** ([ADR-129](ADR-129-homecore-automation-engine.md) scope) — a trigger → condition → action editor on the card surface. The most important RuView-specific trigger types to support are: `state_changed` on `RoomState` entities with a threshold expression (e.g. `anomaly.value > 0.8`), SEED reflex-rule firing events (`fragility_alarm`, `hd_anomaly_indicator`), and custom `domain_event` topics. Actions include calling services in the homecore service registry and firing domain events. The condition expression editor uses `--mono`.

### 4.9 Witness / Audit Log

- **Unified witness timeline** — a chronological merged view of events from both tiers: the SEED's SHA-256 ingest chain (every RVF store write attested) and homecore's Ed25519 state-transition chain (biometric crossings, BFLD identity-risk elevations). Each row: `entity_id` in `--mono`, old/new state, timestamp, source SEED `device_id`, signing key fingerprint (first 8 chars in `--mono`). Pagination uses the same "Showing X–Y of Z" convention from the Cog Store's cog grid.
- **Privacy mode banner** — a persistent top-of-panel banner showing current privacy mode: `--green-d`/green text for full-publish mode; `--amber-d`/amber text for audit-only mode (SHA-256 digests on-SEED only, no MQTT state messages). Show the per-SEED privacy mode state, since SEEDs can be individually configured. Toggling privacy mode is a high-stakes action — require an explicit "Confirm" step with a summary of what will change.
- **Export bundle** — an "Export attestation bundle" button (ghost) that packages the SEED witness chain + homecore Ed25519 chain as a downloadable archive for regulated-deployment (care home, hotel, shared office) compliance handoff.

### 4.10 Settings & Integration Config

- **SEED fleet management** — add, remove, and reprovision SEEDs. Show the USB-only pairing requirement prominently (the pairing window only opens via `169.254.42.1`, not WiFi — a security invariant). Per-SEED: `device_id` in `--mono`, firmware version, bearer token status, and a "Rotate token" action (ghost) that walks the operator through the secure token rotation flow.
- **ESP32 node provisioning** — per-node NVS config display (target IP, target port, node_id), last-seen firmware version, and a link to the provisioning script. The `node_id` → room/zone assignment is editable here and persists to the room calibration system's `room_id` mapping.
- **MQTT / cog-ha-matter config** ([ADR-116](ADR-116-cog-ha-matter-seed.md)) — broker URL, credentials (masked), MQTT topic prefix, mDNS advertisement status (`_ruview-ha._tcp`), and a live connection indicator (green dot for connected, red for unreachable). The 21 HA-DISCO entities per node are listed here with their `via_device` assignments showing which SEED they belong to in HA's device registry.
- **Long-lived access tokens** — for homecore-api companion-app connections (HA 2025.1 wire-compat, [ADR-130](ADR-130-homecore-rest-websocket-api.md)). Token creation, last-used timestamp, and revocation. The HA companion-app pairing QR-code flow surfaces here.
- **Federation config** — for multi-SEED deployments: ESP-NOW mesh sync status, cross-SEED epoch alignment values, and federated-learning round settings (coordinator SEED, round cadence, Krum aggregation parameters per [ADR-105](ADR-105-federated-csi-training.md)). The design invariant **"model deltas only, never raw CSI"** must be labelled explicitly in this panel.

---

## 5. Navigation structure

HOMECORE-UI must integrate into the existing Cognitum Appliance nav shell. The top nav should read:

```
Framework | Guide | Cog Store | HOMECORE | Status
```

— inserting **HOMECORE** as a first-class nav item between the existing "Cog Store" and "Status" entries, using the same nav-item style (text in `--t2`, active state in `--cyan` with bottom underline).

Within the HOMECORE section, a left sidebar (or top sub-nav on narrow viewports) provides section navigation:

```
Dashboard | SEED Fleet | Entities | Rooms | COGs | Calibration | Events | Audit | Settings
```

The COG Store panel within HOMECORE (§4.6) links out to `seed.cognitum.one/store` for the full catalog view, ensuring the existing Cog Store remains the canonical browsing experience.

---

## 6. Key UX invariants

These must be maintained across every panel:

1. **Always make the tier origin of any data explicit.** A `RoomState` reading traces to an ESP32 node → SEED → COG → v0 Appliance state machine. The provenance badge (§4.4) must appear wherever entity states are displayed.
2. **The `stale` and `vetoed` flags from `RoomState` and the kNN fragility score from SEED cognitive analysis are meaningful diagnostic signals** — they must never be silently hidden, styled grey-on-grey, or collapsed behind an expand toggle. They represent system health operators need to act on.
3. **Values that are `null` because a specialist has not been trained must be visually distinct from values that are unavailable due to an error.** The distinction is operationally important: `null` means "calibrate to enable," unavailable means "investigate."
4. **All entity IDs, hashes, API endpoints, binary signatures, device UUIDs, and JSON payloads must use `--mono` font.** This is already the convention in the API Explorer and must be consistent throughout HOMECORE-UI.
5. **The v0 Appliance Hailo HAT is a separate subsystem from the SEED's edge compute.** Inference results tagged as Hailo-sourced (COGs with `arch: hailo10`) must be visually distinguished from results from CPU-only COGs (`arch: arm`) so operators can triage hardware-specific failures.

---

## 7. Scope — complete UI delivery

The deliverable is the **entire** dashboard. Every panel below ships fully implemented and wired to its live data source — there is no scaffold-only milestone and no panel left as a placeholder. The table records each panel's authoritative backing API so the build can proceed in whatever order best fits the dependency graph; it is a dependency map, **not** a sequence of partial releases.

| Panel | Section | Backing API / source |
|---|---|---|
| System Dashboard | §4.1 | [ADR-130](ADR-130-homecore-rest-websocket-api.md) WebSocket + appliance health endpoints |
| SEED Detail View | §4.2 | SEED HTTPS API (vector store, witness, sensors, reflex, cognitive analysis) |
| SEED Fleet Map | §4.3 | fleet topology + federation ([ADR-105](ADR-105-federated-csi-training.md)) |
| Entity & State Browser | §4.4 | [ADR-127](ADR-127-homecore-state-machine-rust.md) state machine via [ADR-130](ADR-130-homecore-rest-websocket-api.md) `subscribe_events`; semantic search via [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) |
| RoomState / Sensing | §4.5 | [ADR-151](ADR-151-room-calibration-specialist-training.md) `GET /api/v1/room/state` |
| COG Management | §4.6 | [ADR-128](ADR-128-homecore-integration-plugin-system.md) plugin runtime + [ADR-100](ADR-100-cog-packaging-specification.md) app registry |
| Calibration Wizard | §4.7 | [ADR-151](ADR-151-room-calibration-specialist-training.md) calibration HTTP API |
| Event Bus & Automation | §4.8 | [ADR-130](ADR-130-homecore-rest-websocket-api.md) broadcast channel + [ADR-129](ADR-129-homecore-automation-engine.md) automation engine |
| Witness / Audit Log | §4.9 | SEED SHA-256 ingest chain + homecore Ed25519 chain |
| Settings & Integration | §4.10 | SEED provisioning, [ADR-116](ADR-116-cog-ha-matter-seed.md) MQTT/Matter, LLAT, federation |

### 7.1 Build sequencing within the complete deliverable

The complete UI depends on backing services that mature on their own timelines. Each panel is built against the **real gateway endpoint** defined in §11; where the upstream is not yet available the panel renders a typed empty/error state, **not** fabricated data (the dev-only `?demo=1` fixture of §2.2 exists for offline development only and is never the shipped behaviour). Concretely, the hard contract dependencies are: [ADR-130](ADR-130-homecore-rest-websocket-api.md) (REST + WebSocket), [ADR-127](ADR-127-homecore-state-machine-rust.md) (state machine), [ADR-151](ADR-151-room-calibration-specialist-training.md) (calibration), [ADR-128](ADR-128-homecore-integration-plugin-system.md) (plugin runtime), [ADR-129](ADR-129-homecore-automation-engine.md) (automation), [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) (event history + semantic search), [ADR-116](ADR-116-cog-ha-matter-seed.md) (SEED/Matter), [ADR-069](ADR-069-cognitum-seed-csi-pipeline.md) (SEED ingest), and [ADR-105](ADR-105-federated-csi-training.md) (federation). The keyword entity filter (§4.4) ships immediately; semantic search layers on once [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) lands. The exact panel→endpoint→upstream map and the new gateway code each requires are §11; the staged delivery is §12.

---

## 8. Consequences

### 8.1 Positive

- Operators, integrators, and residents get a single coherent surface for the full two-tier stack, replacing the need to SSH into SEEDs or hand-craft API calls.
- The dashboard reuses the proven Cognitum design tokens and component patterns verbatim, so it ships visually consistent with no separate design effort and no perceptible seam between surfaces.
- Diagnostic signals that today are invisible (`stale`/`vetoed` flags, kNN fragility, provenance lineage, channel lag) become first-class, surfacing the system's most common real-world failure modes directly to operators.

### 8.2 Negative / risks

- The UI hard-depends on the wire-compat guarantees of ADR-130 and the calibration contract of ADR-151; schema drift in either breaks panels silently. Integration tests against every backing contract in §7 are required.
- Committing to the complete UI in one deliverable is a larger up-front effort and couples the UI's readiness to the maturity of multiple backing services (§7.1, §11). The mitigation is the BFF gateway (§2.1): each panel targets one same-origin endpoint, and the gateway absorbs upstream churn behind a stable contract.
- Promoting `homecore-server` to a gateway means it now **proxies cross-tier traffic** (calibration API, SEED HTTPS, appliance daemons). This adds a network hop, a place for upstream timeouts/partial failures to surface, and a server-side store of SEED bearer tokens that must be protected (§11.10). Each proxied route needs an explicit timeout + typed error mapping so one slow SEED cannot stall the dashboard.
- Several panels depend on data that only exists on **real hardware or new daemons** (SEED device tier, appliance host metrics, COG supervisor). Until those upstreams exist the corresponding gateway routes return `503 upstream_unavailable`; this is honest but means the dashboard is only as "live" as the tiers behind it (§11 classifies every endpoint by what it depends on).
- Faithfully mirroring `seed.cognitum.one/store` couples HOMECORE-UI to the external Cog Store's evolving design; token drift there must be tracked and re-synced.
- The two-tier mental model (Appliance root, SEED children, ESP32 leaves) must be enforced consistently; any panel that flattens or peers the tiers undermines the core architectural constraint.

---

## 9. References

- `https://seed.cognitum.one/store` — primary design reference for all visual conventions.
- `https://seed.cognitum.one/status` — reference for live metric-card layout.
- [ADR-126](ADR-126-ruview-native-ha-port-master.md) — HOMECORE master ADR.
- [ADR-127](ADR-127-homecore-state-machine-rust.md) — HOMECORE-CORE state machine and entity registry.
- [ADR-128](ADR-128-homecore-integration-plugin-system.md) — HOMECORE-PLUGINS WASM COG substrate.
- [ADR-129](ADR-129-homecore-automation-engine.md) — HOMECORE automation engine.
- [ADR-130](ADR-130-homecore-rest-websocket-api.md) — HOMECORE-API REST + WebSocket wire-compat.
- [ADR-132](ADR-132-homecore-recorder-history-semantic-search.md) — homecore-recorder, history + semantic search.
- [ADR-100](ADR-100-cog-packaging-specification.md) — Cognitum Cog packaging specification (manifest.json, status values, on-device layout).
- [ADR-116](ADR-116-cog-ha-matter-seed.md) — cog-ha-matter (SEED cog, HA-DISCO entity surface, mDNS).
- [ADR-069](ADR-069-cognitum-seed-csi-pipeline.md) — ESP32 CSI → Cognitum SEED RVF ingest pipeline (SEED architecture detail).
- [ADR-105](ADR-105-federated-csi-training.md) — Federated CSI training (multi-SEED federation).
- [ADR-151](ADR-151-room-calibration-specialist-training.md) — Per-room calibration specialist training (calibration HTTP API).
- `v2/crates/homecore/src/` — state machine, entity, event, registry source.
- `docs/integration/calibration-appliance-integration.md` — calibration API contract and RoomState schema.

---

## 10. Implementation status

Implemented as a zero-dependency, no-build-step vanilla TS/JS + CSS frontend served by `homecore-server` at `/homecore` (the `rufield-viewer` "Axum + vanilla-JS" pattern). The complete deliverable per §2/§7 — all ten panels, fully rendered, wired to live data where the backing service exists and to a contract-conformant DEMO-flagged mock layer (§7.1) where it does not.

**Location:** `v2/crates/homecore-server/ui/` — `css/tokens.css` (the §3.1 palette, verbatim) + `css/app.css` (§3.3 components); `js/{ui,api,ws,mock,app}.js` (shared helpers, REST client, `subscribe_events` WS client, mock layer, shell+router); `js/panels/*.js` (one module per §4 panel). Mounted via `tower-http` `ServeDir` in `homecore-server::build_app`, gated by `--ui-dir`/`HOMECORE_UI_DIR`.

**Verification:**
- **Rust** — `#[cfg(test)] mod ui_tests` in `homecore-server/src/main.rs`: 5 integration tests (`tower::oneshot`) covering index, design tokens, all ten panel modules served, API coexistence, and mount-disable. *Written but not compiled in the authoring environment (no Rust toolchain present); run `cargo test -p homecore-server` on a Rust host before merge.*
- **Frontend** — `ui/` test suite under plain `node` (no npm install): `npm test` → import/export graph verifier (15 modules) + render-smoke (executes every panel against a DOM shim; 21 checks) + interaction suite (live WS patch, ws.js handshake/parse, calibration contract; 3 checks). **24/24 green.**
- **Benchmark** — `npm run bench`: total bundle **136.8 KB** uncompressed (**~37× smaller** than HA's ~5 MB Lit bundle, the ADR-126 §1.1 foil); slowest panel **1.5 ms/cold-render**.

**Honest scope — current vs. target.** *Earlier cut:* the front-end was complete but only §4.4 Entities was wired to a real backend; the rest rendered from an in-browser mock. *This revision implements the §11 wiring:*

- **Front-end (§11.11) — DONE and verified.** `api.js` rewritten: all data accessors are async and call the §11.2 gateway routes; the mock layer is demoted to a dev-only fixture reachable **only** under `?demo=1` / `HOMECORE_UI_DEMO` (§2.2); every panel `await`s and renders a typed empty/error state on failure (no mock fallback in production). All ten panels converted (3 by hand, 7 via parallel agents). Verified under Node: 5 test files green — import graph, boot, render-smoke (22), interaction (3), **and a new prod-errors suite (13) that runs with demo OFF + gateway unreachable and asserts every panel renders an error state, never mock, never throws** (it caught and fixed a real unhandled-rejection in the events panel).
- **Gateway (§11.1–§11.6) — IMPLEMENTED, COMPILED, TESTED, RUN.** New `homecore-server/src/gateway.rs` (+`reqwest` dep, +CLI/env flags `--calibration-url`/`--calibration-token`/`--apps-dir`/`--gateway-timeout-ms`, merged into `build_app` via `gateway_router`). Real handlers: `/api/cal/*` reverse-proxy (W2), `GET /api/homecore/rooms` with the §11.3 RoomState adapter (W2), `GET /api/homecore/cogs` supervisor over the apps dir (W4), `GET /api/homecore/appliance` from `/proc` + port probes (W6). SEED-device/appliance-daemon routes (seeds, federation, witness, privacy, settings, automations, events-history, hailo, tokens — W3/W5) return a typed `503 upstream_unavailable` per §11.2. **Verified on Rust 1.89: `cargo test -p homecore-server --no-default-features` = 12/12 pass** (6 gateway + 6 UI mount). **Run live:** `GET /api/homecore/appliance` returns real `/proc` metrics + TCP service probes; unauth → `401`; `cogs` → `[]` with no apps dir; SEED-tier → typed `503`; and against a mock calibration upstream the `/api/cal/*` proxy passes through (`200`) and `GET /api/homecore/rooms` correctly adapts `RoomState` to the UI shape (`breathing`→`breathing_bpm`, `heartbeat:null`→`heart_bpm:null`, injected `anomaly.threshold`/`room_id`, `stale` passthrough). **Live testing caught + fixed one real bug** — a double-`v1` path in the `/api/cal/*` proxy URL.

The endpoint-by-endpoint contract is **§11**; the staged plan and which endpoints depend on real SEED/appliance hardware vs. pure software is **§12**.

---

## 11. Backend wiring — making every panel real

This section is the authoritative contract for full functionality. It removes the mock layer from the production path (§2.2) by routing every panel through the `homecore-server` BFF gateway (§2.1). Each endpoint is classified by what it depends on:

- **EXISTS** — backend code already in this repo; gateway only proxies/adapts.
- **NEW-GW** — pure software the gateway itself implements (filesystem, `/proc`, process control, recorder query) — no new external service.
- **NEW-API** — a small HTTP wrapper to add to an existing in-repo crate (`homecore-api`, `homecore-automation`).
- **SEED-DEV** — depends on a SEED node's on-device HTTPS API (separate hardware/firmware).
- **APPLIANCE** — depends on an appliance daemon / accelerator stat source.

### 11.1 Gateway shape

`homecore-server` already mounts `homecore-api` at `/api/*` and the UI at `/homecore`. It gains a new **`/api/homecore/*`** namespace (the dashboard-specific aggregation surface) plus a **`/api/cal/*`** reverse-proxy to the calibration service. The browser issues only same-origin requests; the gateway fans out server-side, holding all upstream credentials (§11.10). Every proxied route has an explicit timeout and maps upstream failure to a typed body (`503 upstream_unavailable`, `504 upstream_timeout`) so one slow tier never stalls the dashboard.

### 11.2 Master endpoint contract (panel → gateway route → upstream → status)

| Panel | UI method (`api.js`) | Gateway route | Upstream / source | Class |
|---|---|---|---|---|
| §4.4 Entities | `states()` | `GET /api/states` | `homecore` state machine | **EXISTS** ✅ wired |
| §4.4/§4.8 live feed | WS | `GET /api/websocket` (`subscribe_events`) | `homecore` event bus | **EXISTS** ✅ wired |
| §4.8 Event history | `eventHistory(q)` | `GET /api/events?since=…` | `homecore-recorder` ([ADR-132](ADR-132-homecore-recorder-history-semantic-search.md)) | **NEW-API** |
| §4.8 Automations | `automations()` / `saveAutomation()` | `GET/POST/DELETE /api/homecore/automations` | `homecore-automation` ([ADR-129](ADR-129-homecore-automation-engine.md)) | **NEW-API** |
| §4.5 Rooms | `roomStates()` | `GET /api/homecore/rooms` → per-room `GET /api/cal/v1/room/state?bank=` | `calibrate-serve` ([ADR-151](ADR-151-room-calibration-specialist-training.md)) | **EXISTS** (proxy + adapter) |
| §4.7 Calibration | `calibration.*` | `POST /api/cal/v1/calibration/{start,stop}`, `GET …/status`, `POST …/enroll/anchor`, `GET …/enroll/status`, `POST …/room/train` | `calibrate-serve` | **EXISTS** (proxy) |
| §4.6 COGs | `cogs()` / `cogAction()` / `cogLogs()` | `GET /api/homecore/cogs`, `POST …/cogs/:id/{start,stop,restart}`, `GET …/cogs/:id/logs`, `GET/PUT …/cogs/:id/config` | COG supervisor over `/var/lib/cognitum/apps/` ([ADR-100](ADR-100-cog-packaging-specification.md)/[ADR-128](ADR-128-homecore-integration-plugin-system.md)) | **NEW-GW** |
| §4.6 Hailo HEF | `hailo()` | `GET /api/homecore/hailo` | `ruvector-hailo-worker:50051` | **APPLIANCE** |
| §4.1 Appliance health | `appliance()` | `GET /api/homecore/appliance` | host `/proc` + Hailo stats + service probes | **NEW-GW** (+APPLIANCE for Hailo) |
| §4.1/§4.2 Fleet + SEED detail | `seeds()` / `seed(id)` | `GET /api/homecore/seeds`, `GET …/seeds/:id` | SEED device HTTPS API ([ADR-069](ADR-069-cognitum-seed-csi-pipeline.md)) via registry | **SEED-DEV** |
| §4.2 SEED actions | `seedCompact()` / `seedVerify()` | `POST …/seeds/:id/{compact,witness/verify}` | SEED device API | **SEED-DEV** |
| §4.3 Federation | `federation()` | `GET /api/homecore/federation` | federation coordinator ([ADR-105](ADR-105-federated-csi-training.md)) | **SEED-DEV/APPLIANCE** |
| §4.9 Witness/Audit | `witnessLog(p,s)` | `GET /api/homecore/witness?page=…` | merge: `homecore` Ed25519 chain + per-SEED SHA-256 chains | **NEW-API + SEED-DEV** |
| §4.9 Privacy mode | `privacyModes()` / `setPrivacy()` | `GET/POST /api/homecore/privacy` | SEED privacy control plane ([ADR-141](ADR-141-bfld-privacy-control-plane-modes-attestation.md)) + cog-ha-matter | **SEED-DEV** |
| §4.9 Export bundle | `exportAttestation()` | `GET /api/homecore/witness/export` | gateway packages both chains | **NEW-GW** |
| §4.10 Tokens (LLAT) | `tokens()` / `createToken()` / `revokeToken()` | `GET/POST/DELETE /api/homecore/tokens` | `homecore-api` `LongLivedTokenStore` | **NEW-API** |
| §4.10 MQTT/Matter | `mqttConfig()` | `GET /api/homecore/integrations/mqtt` | cog-ha-matter config ([ADR-116](ADR-116-cog-ha-matter-seed.md)) | **NEW-GW/SEED-DEV** |
| §4.10 ESP32 provisioning | `nodes()` / `assignRoom()` | `GET/PUT /api/homecore/nodes` | SEED ingest config ([ADR-069](ADR-069-cognitum-seed-csi-pipeline.md)) | **SEED-DEV** |
| §4.10 SEED mgmt | `pairSeed()` / `rotateToken()` | `POST /api/homecore/seeds/{pair,:id/rotate-token}` | SEED pairing (USB `169.254.42.1`) | **SEED-DEV** |

### 11.3 Calibration proxy + RoomState adapter

The calibration service is real but on a different binary/port; the gateway reverse-proxies it under `/api/cal/*` (upstream base from `HOMECORE_CALIBRATION_URL`). Its `RoomState` (`wifi-densepose-calibration/src/runtime.rs`) does **not** match the UI's shape, so the gateway adapts it in `GET /api/homecore/rooms`:

| Real field (`RoomState`) | UI field | Adapter rule |
|---|---|---|
| `breathing: Option<SpecialistReading>` | `breathing_bpm: {value,confidence}\|null` | rename; `value`=`reading.value`, `confidence`=`reading.confidence`; `None`→`null` (preserves "not trained") |
| `heartbeat: Option<…>` | `heart_bpm: {…}\|null` | rename `heartbeat`→`heart_bpm` |
| `presence/posture/restlessness` | same names `{value,confidence}\|null` | `posture.value`=`reading.label` (class), else numeric |
| `anomaly: Option<…>` | `anomaly: {value,confidence,threshold}` | inject `threshold`=`MixtureOfSpecialists.veto_threshold` (0.5) |
| `vetoed` / `stale` | `vetoed` / `stale` | pass through (drives the §4.5/§6 banners) |
| *(absent)* | `room_id`, `seeds[]` | injected by the gateway from the **room registry** |

A **room registry** (config or derived from `GET /api/cal/v1/calibration/baselines`) maps each `room_id` → bank name + serving SEED ids, so `GET /api/homecore/rooms` returns one adapted record per room. `Option::None` → JSON `null` keeps the null-vs-withheld distinction (§6 invariant 3) intact end-to-end.

### 11.4 SEED registry & device-API proxy

The gateway holds a **SEED registry** (`device_id` → base URL + bearer token + zone), populated by pairing (§4.10) and persisted server-side. `GET /api/homecore/seeds[/:id]` fans out to each SEED's on-device API and shapes the result to the §4.2 card/detail model. Expected SEED-side endpoints (the contract the SEED firmware must satisfy — a subset of its 98 endpoints): health; vector-store stats (`vector_count`, `dim`, `epoch`, `knn_latency_ms`, ingest rate); witness (`len`, `last_verify`, `valid`) + `POST verify`; onboard sensors (BME280/PIR/reed/ADS1115/vibration); reflex rules + thresholds; cognitive analysis (fragility, coherence phases); ingest feeders (ESP32 node ids + packet type `0xC5110003`/`0xC5110002` + rate). Offline/unreachable SEEDs surface as `online:false` (drives the §4.1 red tint) rather than failing the whole list.

### 11.5 Appliance metrics collector (§4.1)

`GET /api/homecore/appliance`, implemented in the gateway: CPU/RAM/uptime from `/proc`; Hailo load + temperature from the Hailo runtime/sysfs (or `ruvector-hailo-worker` stats); service health by probing `ruview-mcp-brain:9876`, `cognitum-rvf-agent:9004`, `ruvector-hailo-worker:50051`; event-bus rate from the `homecore` broadcast channel + its lag counter (already exposed for §4.1/§4.4).

### 11.6 COG supervisor (§4.6)

`GET /api/homecore/cogs`: read each `/var/lib/cognitum/apps/*/manifest.json` ([ADR-100](ADR-100-cog-packaging-specification.md)), the pid file, and verify `binary_sha256` + `binary_signature` (Ed25519) → status/shield. `POST …/cogs/:id/{start,stop,restart}` performs supervised process control; `GET …/cogs/:id/logs` tails `output.log`/`error.log`; `GET/PUT …/cogs/:id/config` reads/writes `config.json`. Hailo-arch COGs join the §11.5 Hailo stats. The Cog Store/App-Registry **browsing** panel was removed per product decision; this is operational management only.

### 11.7 Witness aggregation + privacy (§4.9)

`GET /api/homecore/witness` merges two chains chronologically: the `homecore` Ed25519 state-transition chain (exposed by a small `homecore-api` route over its witness log) and each paired SEED's SHA-256 ingest chain (proxied via the registry), paginated server-side. `GET/POST /api/homecore/privacy` reads/sets per-SEED privacy mode via the SEED privacy control plane ([ADR-141](ADR-141-bfld-privacy-control-plane-modes-attestation.md)) — the POST is the high-stakes confirmed toggle (§4.9). `GET /api/homecore/witness/export` packages both chains into the downloadable attestation bundle.

### 11.8 Event history + automation CRUD (§4.8)

`homecore-api` adds `GET /api/events?since=…` backed by `homecore-recorder` ([ADR-132](ADR-132-homecore-recorder-history-semantic-search.md)) for history (live updates continue over the existing WS). The automation builder persists through `GET/POST/DELETE /api/homecore/automations`, a thin HTTP wrapper over the `homecore-automation` engine's register/list/remove ([ADR-129](ADR-129-homecore-automation-engine.md)). RuView-specific triggers (RoomState thresholds, SEED reflex events) map onto the engine's trigger types.

### 11.9 Entity provenance convention (§4.4/§6)

The first-class provenance badge requires each entity to carry its lineage. Convention: every integration writes `attributes.source` (and, where known, `attributes.seed` / `attributes.cog`) when it sets state; `cog-ha-matter` ([ADR-116](ADR-116-cog-ha-matter-seed.md)) populates these from the ESP32 node → SEED → COG path and HA `via_device`. The gateway/UI resolves node→seed→cog from these attributes (no fabrication; missing lineage renders as "unknown", not invented).

### 11.10 Auth, credentials, config

- **Browser → gateway:** one long-lived access token (the §4.10 LLAT), sent as `Authorization: Bearer`; validated by `homecore-api`'s `LongLivedTokenStore`. The dev default (`allow_any_non_empty`) stays for local runs; production provisions `HOMECORE_TOKENS`.
- **Gateway → upstreams:** SEED bearer tokens and the calibration token live **only** server-side (SEED registry + `HOMECORE_CALIBRATION_TOKEN`); never sent to the browser. This is the reason the gateway exists.
- **Config:** `HOMECORE_CALIBRATION_URL`, SEED registry store path, per-proxy timeout (default 2 s), `HOMECORE_UI_DEMO` (dev fixture). No browser CORS needed (same origin); gateway→upstream is server-to-server.

### 11.11 Front-end changes

`api.js`: drop the mock fallback from the production path — methods call the §11.2 gateway routes; `this.base` stays same-origin; the mock layer is reachable only under `?demo=1`/`HOMECORE_UI_DEMO`. Every panel renders a **typed empty/error state** (not mock) when its route returns `503/504`. `mock.js` moves to a dev fixture (kept for the offline test harness, excluded from the production bundle). The §10 frontend tests are re-pointed at the gateway contract (and gain contract tests per §11.2 route).

---

## 12. Delivery plan to full functionality

Staged so each wave is independently shippable behind the gateway, lands real data for a coherent set of panels, and has an explicit acceptance gate. "Class" reuses §11's tags.

| Wave | Scope | Class | Acceptance gate |
|---|---|---|---|
| **W1 — Gateway foundation** | `/api/homecore/*` scaffold in `homecore-server`; auth passthrough; per-proxy timeout + typed errors; `api.js` base + remove prod mock (`?demo=1` only); panels get typed empty/error states | NEW-GW | Entities + live WS still green; with no upstreams, every other panel shows "upstream unavailable", **never** mock (unless `?demo=1`); Rust + JS suites pass |
| **W2 — Rooms + Calibration** | `/api/cal/*` reverse-proxy; `GET /api/homecore/rooms` with the §11.3 RoomState adapter + room registry; wire §4.5 + the §4.7 wizard to real endpoints; delete the in-browser calibration stub | EXISTS (proxy+adapter) | Against a running `calibrate-serve` (replayed CSI), the wizard drives a real baseline→enroll→train→verify and §4.5 shows real `RoomState` with correct stale/veto/null mapping; contract test on the adapter |
| **W3 — Events + Automations** | `GET /api/events` over `homecore-recorder`; `/api/homecore/automations` over `homecore-automation` | NEW-API | §4.8 history loads from recorder; an automation created in the UI persists and fires via the engine |
| **W4 — COG management** | `/api/homecore/cogs*` supervisor over `/var/lib/cognitum/apps/` (manifest + pid + sig verify + logs + config) | NEW-GW | §4.6 lists real installed COGs; start/stop/restart works; sha256/signature shield reflects real verification; logs tail |
| **W5 — SEED tier** | SEED registry + pairing; `/api/homecore/seeds*` device proxy; witness merge + privacy control; ESP32 provisioning | SEED-DEV | Against a real or emulated SEED API, §4.2/§4.3/§4.9/§4.10 show real vector-store/witness/sensor/reflex/cognition data; SEED tokens stay server-side; offline SEED → red tint, not a failed page |
| **W6 — Appliance + federation + Hailo** | `/api/homecore/appliance` (host metrics + service probes); `/api/homecore/hailo`; `/api/homecore/federation` ([ADR-105](ADR-105-federated-csi-training.md)) | NEW-GW + APPLIANCE | §4.1 health is real; §4.6 Hailo HEF/throughput real; §4.3 federation round/coordinator/Krum real |

**Definition of done (full functionality):** with W1–W6 merged and the upstream tiers running, loading `/homecore` with **no** `?demo=1` flag shows live data on all ten panels, `api.anyDemo()` is false, and no panel renders fabricated values. Panels whose tier is offline show typed empty/error states. The mock layer is reachable only as the `?demo=1` developer fixture.

### 12.1 Wave status (this revision)

| Wave | Status |
|---|---|
| **W1 — Gateway foundation** | ✅ DONE — `gateway.rs`, auth passthrough, typed `503/504`, merged into `build_app`; front-end mock removed from prod path + `?demo=1` fixture; typed error states. **Compiled + 12/12 Rust tests + JS suite green + run live.** |
| **W2 — Rooms + Calibration** | ✅ DONE — `/api/cal/*` reverse-proxy + `GET /api/homecore/rooms` RoomState adapter; front-end calibration stub deleted (now proxies the real API). **Proven live against a calibration upstream** (proxy 200 + adapted shape); null-preservation unit-tested. |
| **W3 — Events + Automations** | ⏳ gateway returns typed `503` (recorder/automation HTTP wrappers pending); front-end handles it gracefully (history note, builder still usable). |
| **W4 — COG management** | ✅ supervisor DONE — lists `/var/lib/cognitum/apps/` manifests + pid liveness (returns `[]` live with no apps dir); start/stop/log/config control is the remaining follow-up. |
| **W5 — SEED tier** | ⏳ gateway returns typed `503` (SEED registry + device proxy pending real/emulated SEED hardware). |
| **W6 — Appliance + federation + Hailo** | ◑ appliance host metrics from `/proc` + port probes DONE (live `/proc` data verified); Hailo stats + federation remain `503` (need the accelerator stat source / coordinator). |

**Status:** the gateway is **compiled and tested on Rust 1.89** (`cargo test -p homecore-server` = 12/12) and was **run live** (curl proof in §10). The one remaining caveat is intrinsic, not an environment limit: **W3/W5/W6-Hailo/federation depend on services/hardware that are not in this repo** (recorder/automation HTTP wrappers, real SEED nodes, the Hailo stat source), so they return honest typed `503`s and the UI shows error states — exactly as §2.2/§11.2 prescribe. W1/W2/W4/W6-appliance are functional now.

### 12.2 Security review (PR #1082)

A high-effort public-PR review of the merged gateway + front-end surfaced the following, all fixed and pinned by tests (`cargo test -p homecore-server` is now **18/18**):

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | **HIGH** | **Path-traversal / confused-deputy SSRF** in the `/api/cal/*` reverse-proxy. The wildcard path was interpolated into the upstream URL while `proxy()` attaches the privileged server-side calibration bearer, so `/api/cal/v1/../../x` (or `..%2f`, `%2e%2e`, leading `/`, `\`, double-encoded `%252e`) could escape the `…/api/` scope **with the token**. | `validate_proxy_path()` decode-then-checks and rejects absolute / backslash / dot-segment / encoded-traversal paths with a typed **400 before the URL is built** (GET **and** POST); legit `v1/...` paths still pass. |
| 2 | Correctness | **CORS + tracing didn't cover gateway routes** — `/api/homecore/*` + `/api/cal/*` were `.merge()`d outside `homecore-api::router()`'s layers. | The audited HC-05 `build_cors_layer()` + `TraceLayer` are now applied to the whole merged app in `main.rs`. |
| 3 | Honesty (§6) | **Fabricated data** — hardcoded `anomaly.threshold: 0.5` in the adapter; dashboard rendered `"null%"`/`"null°C"`; COG Hailo pill hardcoded `"connected"`; `rooms.js` defaulted a null threshold to `0.8`. | Threshold passes through the real upstream value or emits `null` (withheld); dashboard renders `—`; the Hailo pill reflects the real appliance probe; the UI treats a null threshold as withheld. |
| 4 | Robustness | A string `hef` (forwarded verbatim) threw on `.forEach`/`.join`; `frames/target` could be `NaN%`/`Infinity%`; calibration Restart leaked the baseline `setTimeout` poll. | `asArray()` coercion; `target > 0` guard; cancellable poll cleared on Restart / panel teardown. |
| 5 | Perf | Sequential per-bank RoomState fetches; blocking `std::net::TcpStream::connect_timeout` probes on an async handler; `mock.js` statically bundled. | Concurrent `futures::join_all`; async `tokio::net::TcpStream` + `timeout`; demo-only dynamic `import()` of `mock.js`. |

**Known limitations carried forward (not regressions):**
- **`reqwest` rustls-only is a workspace-wide concern.** `homecore-server` opts into `rustls-tls` only, but cargo feature-unification means any sibling crate enabling the default `native-tls` re-introduces OpenSSL into the final binary. A true "no OpenSSL on the appliance" guarantee requires aligning **every** reqwest-pulling crate on rustls-only — out of scope for this PR; documented at the dependency in `Cargo.toml`.
- **DEV-mode auth.** When `HOMECORE_TOKENS` is unset, the token store falls back to `allow_any_non_empty()` (any non-empty bearer accepted) on `0.0.0.0`. This is pre-existing and intentionally **unchanged** here; the loud boot `warn!` is retained. Provision real tokens (`HOMECORE_TOKENS=…`) before exposing the server to a network.

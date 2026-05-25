# ADR-129: HOMECORE-AUTO — Automation engine, script runner, and template evaluator

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-AUTO** |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE), [ADR-129 implicit](ADR-129-homecore-automation-engine.md), [ADR-133](ADR-133-homecore-assist-ruflo.md) (HOMECORE-ASSIST) |
| **Tracking issue** | TBD |

---

## 1. Context

Home Assistant's automation system is defined across three components:

1. **`homeassistant/components/automation/__init__.py`** — the automation manager: loads automation YAML, evaluates trigger platforms, calls the script executor when conditions pass. The core class is `AutomationEntity` which extends `ToggleEntity`. Automations are themselves HA entities with `state = on/off`.

2. **`homeassistant/components/script/__init__.py`** — the script executor: a sequence of actions (service calls, conditions, delays, events, template variables, `choose`, `parallel`, `repeat`, `wait_for_trigger`). Scripts are entities too (`ScriptEntity` extends `ToggleEntity`). The execution engine supports five run modes: `single`, `restart`, `queued`, `parallel`, `ignore_first`.

3. **`homeassistant/helpers/template.py`** — HA's Jinja2 customisation layer: wraps the upstream `jinja2` Python library with HA-specific globals (`states()`, `is_state()`, `state_attr()`, `now()`, `utcnow()`, `as_timestamp()`, `distance()`, `closest()`, etc.), custom filters (`regex_match`, `round`, `timestamp_local`), and a sandboxed `Environment` that prevents file I/O and dangerous evaluations.

### 1.1 Scale and surface

HA's automation YAML supports:
- **17 trigger platforms** (state, time, numeric_state, template, event, homeassistant, zone, geo_location, device, calendar, conversation, mqtt, webhook, tag, sun, time_pattern, persistent_notification)
- **7 condition types** (state, numeric_state, time, template, zone, sun, device)
- **22+ action types** (call_service, delay, wait_template, fire_event, device_action, choose, if, parallel, repeat, sequence, stop, set_conversation_response, ...)

The YAML schema is validated by `voluptuous` schemas defined in `homeassistant/helpers/config_validation.py` (~5,000 lines).

### 1.2 Jinja2 is the critical surface

HA templates are used not only in automations but in dashboard cards, notification messages, and script variables. The HA frontend sends template strings to the API's `POST /api/template` endpoint for server-side evaluation. Any HOMECORE instance that claims API compatibility must execute Jinja2-compatible templates or existing automations will break.

Full Jinja2 support in Rust without Python is non-trivial. The approach chosen here uses a **WASM-compiled MiniJinja** (the `minijinja` Rust crate compiled with HA-specific extension functions) rather than a full Python Jinja2 re-implementation.

---

## 2. Decision

Build the `homecore-automation` crate with three components:

1. **YAML parser**: `serde_yaml` + custom validator that parses HA's automation and script YAML into typed Rust structs. Validates trigger, condition, and action schemas at load time.
2. **Trigger evaluator**: a Tokio task per loaded automation that subscribes to the HOMECORE event bus (ADR-127) and evaluates trigger conditions in Rust. When a trigger fires and conditions pass, it enqueues the automation action sequence.
3. **Action executor**: a script runner that processes action sequences. Service calls go to the HOMECORE service registry. Delays use `tokio::time::sleep`. Template evaluation uses MiniJinja. Complex conditions (optional) can route to a ruflo agent (ADR-133).

### 2.1 Template evaluator: MiniJinja + HA-compatible extension functions

`minijinja` (crates.io version 2.x) is a production-quality Jinja2 implementation in pure Rust. It is missing 5–10% of Jinja2's surface area (notably: `{% block %}` / `{% extends %}` template inheritance, and some Jinja2 Python-specific filters), but covers 100% of HA's automation template usage.

HA-specific globals added on top of MiniJinja:

```rust
env.add_global("states", minijinja::Value::from_function(ha_states_global));
env.add_global("is_state", minijinja::Value::from_function(ha_is_state_global));
env.add_global("state_attr", minijinja::Value::from_function(ha_state_attr_global));
env.add_global("now", minijinja::Value::from_function(ha_now_global));
env.add_global("utcnow", minijinja::Value::from_function(ha_utcnow_global));
env.add_global("as_timestamp", minijinja::Value::from_function(ha_as_timestamp_global));
env.add_global("distance", minijinja::Value::from_function(ha_distance_global));
env.add_global("iif", minijinja::Value::from_function(ha_iif_global));
```

Each global function reads from the HOMECORE state machine (ADR-127) via an `Arc<StateMachine>` captured at environment construction time. Template evaluation is synchronous (MiniJinja is sync) but runs in a `tokio::task::spawn_blocking` wrapper to avoid blocking the async executor.

### 2.2 WASM evaluator for untrusted template strings

Dashboard card templates submitted via `POST /api/template` come from user-authored YAML, not first-party code. HA evaluates these in the same Python process, relying on Jinja2's `SandboxedEnvironment` for safety. HOMECORE uses a **WASM-sandboxed MiniJinja** evaluator:

- A single WASM module (`homecore-template-eval.wasm`) is compiled from the MiniJinja crate with the HA extension globals stubbed to call host functions.
- Template strings are passed into the WASM module via the HOMECORE plugin ABI (ADR-128 §5.1).
- The WASM sandbox prevents file I/O, network access, and infinite loops (via Wasmtime fuel metering — 100,000 instructions per template evaluation).
- Result is returned as a string to the HOMECORE API.

This is the same Wasmtime host already used for integration plugins (ADR-128) — no additional WASM runtime dependency.

---

## 3. HA-side reference table

| HA module / file | What it does | HOMECORE preserves | Changes | Drops |
|---|---|---|---|---|
| `automation/__init__.py` `AutomationEntity` | Automation as a toggle entity (on/off) with triggers/conditions/actions | Automation is a HOMECORE entity with same on/off state semantics | Rust struct `AutomationEntity` implementing `HomeCoreEntity` trait | Python class hierarchy, voluptuous schema |
| `automation/__init__.py` `TriggerActionConfig` | Trigger → condition → action pipeline | Full trigger/condition/action pipeline | Typed Rust enums per trigger platform | Python dict-based config |
| `automation/trigger.py` | Delegates to per-platform trigger modules (`homeassistant/components/<platform>/trigger.py`) | Same per-platform dispatch | Rust match arm per trigger type | Python dynamic module import |
| `script/__init__.py` `Script` | Script entity + action sequence executor | Same 22 action types | Rust enum `Action` with all variants | Python asyncio coroutines |
| `script/__init__.py` run modes | `single`, `restart`, `queued`, `parallel`, `ignore_first` | All 5 run modes | Tokio-based concurrency control (semaphore for `queued`, `parallel`) | Python asyncio task management |
| `helpers/template.py` `Template` | Jinja2 evaluation + HA globals | Same HA global function names and signatures | MiniJinja instead of Python Jinja2; WASM sandbox for user templates | Python `jinja2` library; `voluptuous` coercions in templates |
| `helpers/config_validation.py` | `cv.template`, `cv.entity_id`, time period validators | Same validation semantics | Rust custom deserializers implementing `serde::Deserialize` | voluptuous; Python regex |
| `components/automation/blueprint.py` | Blueprint system (reusable automation templates with input variables) | Blueprint YAML schema + variable substitution | Pure Rust YAML substitution | Python Blueprint class hierarchy |

---

## 4. Public API parity table

| HA automation surface | HOMECORE equivalent |
|---|---|
| `automation.trigger` (state, time, numeric_state, template, event, ...) | `Trigger` enum with variants for all 17 HA trigger platforms |
| `automation.condition` (state, numeric_state, time, template, zone, sun, device) | `Condition` enum with variants for all 7 condition types |
| `automation.action` — call_service, delay, fire_event, choose, if, parallel, repeat, wait_template, stop | `Action` enum with variants for all 22 action types |
| `script.run_mode` — single, restart, queued, parallel | `RunMode` enum with 5 variants |
| `POST /api/template` (REST eval of a template string) | Same endpoint in HOMECORE-API (ADR-130); backed by WASM-sandboxed MiniJinja |
| Automation entity: `state = on|off`, `attributes.last_triggered`, `attributes.id` | `AutomationEntity` struct with same attribute names |
| `automation.trigger` service (manually trigger an automation) | `homecore.automation.trigger` service; same service call data schema |
| `automation.reload` service (reload automations.yaml) | `homecore.automation.reload` service |
| `automation.toggle` service | Standard `HomeCoreEntity` toggle service |
| Blueprint YAML with `blueprint:` key and `input:` variables | Blueprint parsed by HOMECORE YAML parser; same substitution semantics |

---

## 5. Trigger platform mapping

| HA trigger platform | HOMECORE implementation |
|---|---|
| `state` | Subscribe to `state_changed` broadcast; match `entity_id`, `from`, `to`, `for` |
| `numeric_state` | Subscribe to `state_changed`; parse state as f64; compare against `above`/`below` |
| `time` | `tokio::time::sleep_until` to next occurrence; re-arm after fire |
| `time_pattern` | Cron-style evaluation using `cron` crate; tokio timer task |
| `template` | Re-evaluate template on every `state_changed`; fire when template transitions from false to true |
| `event` | Subscribe to named domain event on event bus |
| `homeassistant` (start/stop) | Subscribe to `HomeAssistantStart` / `HomeAssistantStop` typed events |
| `zone` | Subscribe to `zone.entered` / `zone.left` events from the device tracker integration |
| `mqtt` | Subscribe to MQTT topic via the MQTT plugin (ADR-128); fire event when message arrives |
| `webhook` | HOMECORE-API registers a webhook path; fires event on POST |
| `calendar` | Subscribe to calendar event from calendar integration |
| `conversation` | Subscribe to `conversation.user_input` event; match intent/sentence |
| `geo_location` | Subscribe to `geo_location.entered` / `geo_location.left` |
| `sun` | Compute sunrise/sunset from latitude/longitude in `homecore.config`; tokio timer |
| `device` | Delegate to integration-specific device trigger via WASM plugin |
| `persistent_notification` | Subscribe to `persistent_notification.create` event |
| `tag` | Subscribe to `tag.scanned` event from NFC/QR integration |

---

## 6. Phased implementation plan

### P1 — YAML parser (2 weeks)

- [ ] Define Rust enums for `Trigger`, `Condition`, `Action`, `RunMode` with `serde` deserialization.
- [ ] Parse an existing `automations.yaml` from a real HA install with zero errors (test fixture).
- [ ] Validator: reject unknown trigger platforms with a clear error message.
- [ ] Unit tests: parse 50 automation fixtures covering all 17 trigger types and 22 action types.

### P2 — State and event triggers (2 weeks)

- [ ] Implement `state`, `numeric_state`, `event`, `homeassistant`, `time`, `time_pattern` trigger evaluators.
- [ ] `ConditionEvaluator` for `state`, `numeric_state`, `time` conditions.
- [ ] `ActionExecutor` for `call_service`, `delay`, `fire_event`, `stop` action types.
- [ ] Integration test: load one automation (state trigger → call_service action); verify fires correctly when state changes.

### P3 — Full action set + MiniJinja (3 weeks)

- [ ] MiniJinja + HA extension globals; `POST /api/template` endpoint wired to WASM evaluator.
- [ ] `template` trigger + `template` condition evaluators.
- [ ] `choose`, `if`, `parallel`, `repeat`, `wait_template`, `sequence` action types.
- [ ] All 5 `RunMode` variants (concurrency control via Tokio semaphore/mutex).
- [ ] Integration test: `automations.yaml` from ADR-134 migration fixture loads and runs correctly.

### P4 — Blueprint system + ruflo agent condition (1 week)

- [ ] Blueprint YAML parser + input variable substitution.
- [ ] Optional ruflo agent condition: `condition: ruflo_agent` with `query: "..."` routes to ruflo LLM (ADR-133 §3.3); gated by RUVIEW-POLICY.
- [ ] `automation.reload` service.
- [ ] Performance benchmark: 100 automations loaded; 100 state changes/s; verify trigger evaluation stays < 5 ms per state change.

---

## 7. Risks

| Risk | Likelihood | Severity | Mitigation | Cross-ADR impact |
|---|---|---|---|---|
| **MiniJinja gaps** — some HA templates use Jinja2 features MiniJinja doesn't support (template inheritance, Python-specific filters) | Medium | Medium | Document the MiniJinja-vs-Jinja2 delta before P3 ships; provide a migration guide for affected templates; defer the 5% of templates that fail to a Python-compat shim (ADR-134) | ADR-134: migration tool must warn on templates that use unsupported Jinja2 features |
| **Template performance** — synchronous MiniJinja in `spawn_blocking` adds overhead under high automation fan-out | Low | Low | Benchmark at 50 automations each evaluating a template trigger on every state_changed (worst case); if > 2 ms add a template-evaluation cache keyed by (template_hash, relevant_entity_states) | ADR-127: state machine must expose a "relevant states snapshot" API for caching |
| **ADR-127 state machine API not frozen** — trigger evaluators call `hass.states.all()` and subscribe to broadcasts; if those APIs change, trigger code must update | High (early) | High | ADR-127 must freeze its public API before ADR-129 P2 begins; use a `HomeCoreRef` trait (version 1.0 stable) | ADR-127 owns this dependency |
| **Complex action YAML** — real-world automations use deeply nested `choose`/`if`/`parallel` blocks; parsing is non-trivial | Medium | Medium | Use a corpus of 500 public HA automations from the HA community (MIT-licensed) as parse-test fixtures in CI | None |

---

## 8. Open questions

**Q1**: MiniJinja does not support all Python-specific Jinja2 filters (e.g. `map`, `select`, `reject` with Python lambda arguments). HA's `homeassistant/helpers/template.py` adds custom equivalents of several of these. How many real-world HA automations use these filters? A corpus analysis of public HA configs on GitHub would answer this before P3 implementation.

**Q2**: HA's `template` trigger supports a `value_template` that can reference `trigger.to_state`, `trigger.from_state`, and `trigger.for`. This requires passing trigger context into the template evaluation scope. Is this context threading straightforward in MiniJinja, or does it require a custom context type?

**Q3**: The `conversation` trigger in HA uses the Assist pipeline's intent matching to fire automations based on voice commands. HOMECORE-ASSIST (ADR-133) owns the pipeline. Should the `conversation` trigger be implemented in ADR-129 (automation engine dependency on ADR-133) or in ADR-133 (assist pipeline fires automation events that ADR-129 listens to)?

**Q4**: HA blueprints have a community sharing mechanism (blueprint.exchange). Should HOMECORE support importing blueprints from HA's blueprint exchange directly, or only local blueprints?

---

## 9. References

### HA upstream

- `homeassistant/components/automation/__init__.py` — `AutomationEntity`, `AutomationConfig`, trigger/condition/action pipeline
- `homeassistant/components/script/__init__.py` — `Script`, `ScriptEntity`, run modes, action sequence execution
- `homeassistant/helpers/template.py` — `Template` class, `TemplateEnvironment`, all HA-specific Jinja2 globals and filters
- `homeassistant/helpers/config_validation.py` — voluptuous schema definitions for all automation/script YAML elements
- `homeassistant/components/automation/blueprint.py` — Blueprint input substitution

### This repo

- `docs/adr/ADR-127-homecore-state-machine-rust.md` — state machine and event bus that triggers subscribe to
- `docs/adr/ADR-133-homecore-assist-ruflo.md` — ruflo agent condition + conversation trigger dependency
- `docs/adr/ADR-134-homecore-migration-from-python-ha.md` — migration tool reads `automations.yaml`

### External

- [minijinja crates.io](https://crates.io/crates/minijinja) — Jinja2-compatible template engine in Rust
- [HA Automation Templating docs](https://www.home-assistant.io/docs/automation/templating/) — HA-specific template globals reference

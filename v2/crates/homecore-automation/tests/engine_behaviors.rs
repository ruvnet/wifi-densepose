//! Engine behavioral integration tests (ADR-161, HC-WS-04/05/07).
//!
//! These exercise the `AutomationEngine` runtime through its public API
//! only (extracted from the inline module to keep `engine.rs` under the
//! 500-line file guideline):
//!
//! - HC-WS-04 — `time:` triggers fire via the engine timer path.
//! - HC-WS-05 — `RunMode::Single` does not double-fire; `Parallel` does.
//! - HC-WS-07 — `template:` conditions evaluate against live state in the
//!   engine path (no longer always-false).
//!
//! Each fails on the pre-fix engine (no timer task, unbounded-parallel
//! regardless of mode, `template_env: None`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use homecore::service::FnHandler;
use homecore::{Context, EntityId, HomeCore, ServiceCall, ServiceName};
use homecore_automation::{Action, Automation, AutomationEngine, Condition, RunMode, Trigger};
use tokio::time::{sleep, Duration};

async fn register_recorder(
    hc: &HomeCore,
    domain: &str,
    service: &str,
) -> Arc<Mutex<Vec<serde_json::Value>>> {
    let log: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
    let log2 = Arc::clone(&log);
    hc.services()
        .register(
            ServiceName::new(domain, service),
            FnHandler(move |call: ServiceCall| {
                let l = Arc::clone(&log2);
                async move {
                    l.lock().unwrap().push(call.data.clone());
                    Ok(serde_json::Value::Null)
                }
            }),
        )
        .await;
    log
}

// ── HC-WS-04: time triggers fire ───────────────────────────────────
#[tokio::test]
async fn time_trigger_fires_via_timer_path() {
    let hc = HomeCore::new();
    let log = register_recorder(&hc, "light", "turn_on").await;

    let engine = AutomationEngine::new(hc.clone());
    engine.register(Automation::new(
        "time_auto",
        vec![Trigger::Time { at: "07:30:00".into() }],
        vec![Action::ServiceCall {
            domain: "light".into(),
            service: "turn_on".into(),
            data: serde_json::json!({"by": "time"}),
        }],
    ));

    // Deterministically fire the timer path for the matching second.
    let fired = engine.fire_time_for_test("07:30:00").await;
    assert_eq!(fired, 1, "time automation should fire for matching HH:MM:SS");
    sleep(Duration::from_millis(50)).await;
    assert_eq!(log.lock().unwrap().len(), 1, "time trigger should run its action");

    // A non-matching second must NOT fire.
    let none = engine.fire_time_for_test("09:00:00").await;
    assert_eq!(none, 0);
}

// ── HC-WS-05: RunMode::Single does not double-fire ─────────────────
#[tokio::test]
async fn single_mode_does_not_double_fire_on_rapid_triggers() {
    let hc = HomeCore::new();
    let count = Arc::new(AtomicUsize::new(0));
    let count2 = Arc::clone(&count);
    hc.services()
        .register(
            ServiceName::new("light", "slow"),
            FnHandler(move |_call: ServiceCall| {
                let c = Arc::clone(&count2);
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(200)).await;
                    Ok(serde_json::Value::Null)
                }
            }),
        )
        .await;

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = Automation::new(
        "single_auto",
        vec![Trigger::State {
            entity_id: EntityId::parse("switch.s").unwrap(),
            from: None,
            to: None,
        }],
        vec![Action::ServiceCall {
            domain: "light".into(),
            service: "slow".into(),
            data: serde_json::json!({}),
        }],
    );
    auto.mode = RunMode::Single;
    engine.register(auto);
    let _handle = engine.start();

    // Two rapid triggers while the first run is still sleeping.
    hc.states().set(EntityId::parse("switch.s").unwrap(), "a", serde_json::json!({}), Context::new());
    sleep(Duration::from_millis(20)).await;
    hc.states().set(EntityId::parse("switch.s").unwrap(), "b", serde_json::json!({}), Context::new());

    sleep(Duration::from_millis(350)).await;
    assert_eq!(
        count.load(Ordering::SeqCst),
        1,
        "Single-mode automation must not double-fire while already running"
    );
}

#[tokio::test]
async fn parallel_mode_does_fire_concurrently() {
    let hc = HomeCore::new();
    let count = Arc::new(AtomicUsize::new(0));
    let count2 = Arc::clone(&count);
    hc.services()
        .register(
            ServiceName::new("light", "slow"),
            FnHandler(move |_call: ServiceCall| {
                let c = Arc::clone(&count2);
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(150)).await;
                    Ok(serde_json::Value::Null)
                }
            }),
        )
        .await;

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = Automation::new(
        "parallel_auto",
        vec![Trigger::State {
            entity_id: EntityId::parse("switch.p").unwrap(),
            from: None,
            to: None,
        }],
        vec![Action::ServiceCall {
            domain: "light".into(),
            service: "slow".into(),
            data: serde_json::json!({}),
        }],
    );
    auto.mode = RunMode::Parallel;
    engine.register(auto);
    let _handle = engine.start();

    hc.states().set(EntityId::parse("switch.p").unwrap(), "a", serde_json::json!({}), Context::new());
    sleep(Duration::from_millis(20)).await;
    hc.states().set(EntityId::parse("switch.p").unwrap(), "b", serde_json::json!({}), Context::new());

    sleep(Duration::from_millis(300)).await;
    assert_eq!(
        count.load(Ordering::SeqCst),
        2,
        "Parallel-mode automation should fire on every trigger"
    );
}

// ── HC-WS-07: template conditions evaluate in the engine path ──────
#[tokio::test]
async fn template_condition_evaluates_true_in_engine() {
    let hc = HomeCore::new();
    let log = register_recorder(&hc, "light", "turn_on").await;

    hc.states().set(
        EntityId::parse("sensor.flag").unwrap(),
        "on",
        serde_json::json!({}),
        Context::new(),
    );

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = Automation::new(
        "tmpl_auto",
        vec![Trigger::State {
            entity_id: EntityId::parse("switch.trigger").unwrap(),
            from: None,
            to: None,
        }],
        vec![Action::ServiceCall {
            domain: "light".into(),
            service: "turn_on".into(),
            data: serde_json::json!({}),
        }],
    );
    auto.condition = vec![Condition::Template {
        value_template: "{{ is_state('sensor.flag', 'on') }}".into(),
    }];
    engine.register(auto);
    let _handle = engine.start();

    hc.states().set(
        EntityId::parse("switch.trigger").unwrap(),
        "go",
        serde_json::json!({}),
        Context::new(),
    );
    sleep(Duration::from_millis(50)).await;
    assert_eq!(
        log.lock().unwrap().len(),
        1,
        "template condition should evaluate true and let the action run (HC-WS-07)"
    );
}

#[tokio::test]
async fn template_condition_evaluates_false_blocks_action() {
    let hc = HomeCore::new();
    let log = register_recorder(&hc, "light", "turn_on").await;
    hc.states().set(
        EntityId::parse("sensor.flag").unwrap(),
        "off",
        serde_json::json!({}),
        Context::new(),
    );

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = Automation::new(
        "tmpl_auto_false",
        vec![Trigger::State {
            entity_id: EntityId::parse("switch.trigger").unwrap(),
            from: None,
            to: None,
        }],
        vec![Action::ServiceCall {
            domain: "light".into(),
            service: "turn_on".into(),
            data: serde_json::json!({}),
        }],
    );
    auto.condition = vec![Condition::Template {
        value_template: "{{ is_state('sensor.flag', 'on') }}".into(),
    }];
    engine.register(auto);
    let _handle = engine.start();

    hc.states().set(
        EntityId::parse("switch.trigger").unwrap(),
        "go",
        serde_json::json!({}),
        Context::new(),
    );
    sleep(Duration::from_millis(50)).await;
    assert_eq!(log.lock().unwrap().len(), 0, "false template condition should block the action");
}

// ── ADR-162 (completes ADR-161 §A5): bounded RunModes ───────────────
//
// ADR-161 honored only Single/Parallel; Restart/Queued/max were honestly
// documented as unbounded-parallel. These tests drive the real
// Restart/Queued/max machinery and FAIL on the old engine (where every
// non-Single mode spawned an unbounded parallel task).

/// A service that increments a live concurrency gauge on entry, sleeps,
/// then decrements — recording the maximum concurrency ever observed and
/// the total number of completed runs. Returns `(max_concurrency, completed)`.
async fn register_gauge(
    hc: &HomeCore,
    domain: &str,
    service: &str,
    work: Duration,
) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let live = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicUsize::new(0));
    let (l, m, c) = (Arc::clone(&live), Arc::clone(&max_seen), Arc::clone(&completed));
    hc.services()
        .register(
            ServiceName::new(domain, service),
            FnHandler(move |_call: ServiceCall| {
                let (l, m, c) = (Arc::clone(&l), Arc::clone(&m), Arc::clone(&c));
                async move {
                    let now = l.fetch_add(1, Ordering::SeqCst) + 1;
                    m.fetch_max(now, Ordering::SeqCst);
                    sleep(work).await;
                    l.fetch_sub(1, Ordering::SeqCst);
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok(serde_json::Value::Null)
                }
            }),
        )
        .await;
    (max_seen, completed)
}

fn state_auto(id: &str, entity: &str, domain: &str, service: &str) -> Automation {
    Automation::new(
        id,
        vec![Trigger::State {
            entity_id: EntityId::parse(entity).unwrap(),
            from: None,
            to: None,
        }],
        vec![Action::ServiceCall {
            domain: domain.into(),
            service: service.into(),
            data: serde_json::json!({}),
        }],
    )
}

// ── Restart: cancels the in-flight run ─────────────────────────────
#[tokio::test]
async fn restart_mode_cancels_prior_run() {
    let hc = HomeCore::new();
    // Each run sleeps 300ms before recording completion.
    let (_max, completed) =
        register_gauge(&hc, "light", "slow", Duration::from_millis(300)).await;

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = state_auto("restart_auto", "switch.r", "light", "slow");
    auto.mode = RunMode::Restart;
    engine.register(auto);
    let _handle = engine.start();

    // Trigger 1 starts the slow run.
    hc.states().set(EntityId::parse("switch.r").unwrap(), "a", serde_json::json!({}), Context::new());
    sleep(Duration::from_millis(80)).await;
    // Trigger 2 arrives mid-run → must ABORT run 1 and start run 2.
    hc.states().set(EntityId::parse("switch.r").unwrap(), "b", serde_json::json!({}), Context::new());

    // Wait long enough for run 2 (started ~80ms in) to finish, but run 1
    // (aborted at ~80ms, would have finished at ~300ms) must NOT complete.
    sleep(Duration::from_millis(400)).await;
    assert_eq!(
        completed.load(Ordering::SeqCst),
        1,
        "Restart must cancel the in-flight run: exactly the restarted run completes (not both). \
         On the old engine both ran to completion → 2."
    );
}

// ── Queued: serialize N rapid triggers, all run, never concurrent ──
#[tokio::test]
async fn queued_mode_runs_sequentially_not_concurrently() {
    let hc = HomeCore::new();
    let (max_seen, completed) =
        register_gauge(&hc, "light", "slow", Duration::from_millis(120)).await;

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = state_auto("queued_auto", "switch.q", "light", "slow");
    auto.mode = RunMode::Queued;
    engine.register(auto);
    let _handle = engine.start();

    // Three rapid triggers.
    for v in ["a", "b", "c"] {
        hc.states().set(EntityId::parse("switch.q").unwrap(), v, serde_json::json!({}), Context::new());
        sleep(Duration::from_millis(10)).await;
    }

    // 3 runs × 120ms serialized ≈ 360ms; wait generously.
    sleep(Duration::from_millis(600)).await;
    assert_eq!(
        completed.load(Ordering::SeqCst),
        3,
        "Queued must run every trigger (nothing dropped)"
    );
    assert_eq!(
        max_seen.load(Ordering::SeqCst),
        1,
        "Queued must never run two instances concurrently. On the old engine all 3 ran in \
         parallel → max concurrency 3."
    );
}

// ── max: 2 → never more than 2 concurrent ──────────────────────────
#[tokio::test]
async fn max_two_caps_concurrency_at_two() {
    let hc = HomeCore::new();
    let (max_seen, completed) =
        register_gauge(&hc, "light", "slow", Duration::from_millis(150)).await;

    let engine = AutomationEngine::new(hc.clone());
    let mut auto = state_auto("max_auto", "switch.m", "light", "slow");
    auto.mode = RunMode::Parallel;
    auto.max = Some(2);
    engine.register(auto);
    let _handle = engine.start();

    // Four rapid triggers — without the cap all 4 would run at once.
    for v in ["a", "b", "c", "d"] {
        hc.states().set(EntityId::parse("switch.m").unwrap(), v, serde_json::json!({}), Context::new());
        sleep(Duration::from_millis(10)).await;
    }

    sleep(Duration::from_millis(600)).await;
    assert_eq!(
        completed.load(Ordering::SeqCst),
        4,
        "max:2 must still run all 4 triggers (queued beyond the cap, not dropped)"
    );
    assert!(
        max_seen.load(Ordering::SeqCst) <= 2,
        "max:2 must never exceed 2 concurrent runs (observed {}). On the old engine all 4 ran \
         concurrently → 4.",
        max_seen.load(Ordering::SeqCst)
    );
    assert!(
        max_seen.load(Ordering::SeqCst) >= 2,
        "max:2 should reach the cap of 2 with 4 rapid triggers (observed {})",
        max_seen.load(Ordering::SeqCst)
    );
}

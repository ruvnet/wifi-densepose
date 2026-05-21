//! Structured JSON event publisher — one event per line on stdout.

use crate::inference::CountPrediction;
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
pub struct Event<'a> {
    pub ts: f64,
    pub level: &'a str,
    pub event: &'a str,
    pub fields: Value,
}

pub fn emit_event(ev: &Event<'_>) {
    if let Ok(line) = serde_json::to_string(ev) {
        println!("{line}");
    }
}

pub fn health_ok(cog_id: &str, backend: &str, p: &CountPrediction) {
    let (lo, hi) = p.p95_range();
    emit_event(&Event {
        ts: now_secs(),
        level: "info",
        event: "health.ok",
        fields: json!({
            "cog": cog_id,
            "backend": backend,
            "synthetic_count": p.argmax(),
            "synthetic_confidence": p.confidence,
            "synthetic_p95_range": [lo, hi],
        }),
    });
}

pub fn run_started(cog_id: &str, sensing_url: &str, poll_ms: u64, model_path: &str) {
    emit_event(&Event {
        ts: now_secs(),
        level: "info",
        event: "run.started",
        fields: json!({
            "cog": cog_id,
            "sensing_url": sensing_url,
            "poll_ms": poll_ms,
            "model_path": model_path,
        }),
    });
}

pub fn person_count(tick: u64, fused: &CountPrediction, n_nodes: usize) {
    let (lo, hi) = fused.p95_range();
    emit_event(&Event {
        ts: now_secs(),
        level: "info",
        event: "person.count",
        fields: json!({
            "tick": tick,
            "count": fused.argmax(),
            "confidence": fused.confidence,
            "count_p95_low": lo,
            "count_p95_high": hi,
            "n_nodes": n_nodes,
            "probs": fused.probs,
        }),
    });
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

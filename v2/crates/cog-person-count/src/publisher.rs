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
            // Honest disclosure: the count head has 8 classes but the shipped
            // weights were only trained on classes 0..=MAX_TRAINED_CLASS
            // (presence, not multi-occupant counting). Counts above this are
            // flagged `low_confidence` on each person.count event.
            "count_max_trained_class": crate::inference::MAX_TRAINED_CLASS,
            "count_classes": crate::inference::COUNT_CLASSES,
        }),
    });
}

pub fn person_count(tick: u64, fused: &CountPrediction, n_nodes: usize) {
    let (lo, hi) = fused.p95_range();
    let low_confidence = fused.is_low_confidence();
    emit_event(&Event {
        ts: now_secs(),
        // An out-of-distribution count (argmax beyond the trained classes) is
        // a warning, not a clean info reading.
        level: if low_confidence { "warn" } else { "info" },
        event: "person.count",
        fields: json!({
            "tick": tick,
            // Reported count is clamped to the trained range — we never emit a
            // fabricated multi-occupant headcount the weights can't back.
            "count": fused.clamped_count(),
            // Raw argmax kept for diagnostics/audit.
            "raw_count": fused.argmax(),
            "confidence": fused.confidence,
            // True when argmax > MAX_TRAINED_CLASS (untrained class).
            "low_confidence": low_confidence,
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

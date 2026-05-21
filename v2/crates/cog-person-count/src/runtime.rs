//! Long-running inference loop. Polls the appliance's sensing-server,
//! slides a CSI window, runs the count head, and emits `person.count`
//! events. Same shape as `cog-pose-estimation::runtime`.
//!
//! Multi-node fusion is single-node only in v0.0.1 — the appliance's
//! `/api/v1/sensing/latest` endpoint already aggregates across nodes
//! before serving, so per-cog fusion is deferred until each node ships
//! raw frames separately (ADR-103 §"Multi-node fusion" v0.2.0).

use crate::inference::{CsiWindow, InferenceEngine, INPUT_SUBCARRIERS, INPUT_TIMESTEPS};
use crate::publisher;
use std::time::Duration;
use tokio::time::sleep;

pub struct RunConfig {
    pub sensing_url: String,
    pub poll_ms: u64,
}

pub async fn run_loop(
    cfg: RunConfig,
    engine: InferenceEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer: Vec<f32> = Vec::with_capacity(INPUT_SUBCARRIERS * INPUT_TIMESTEPS);
    let cap = INPUT_SUBCARRIERS * INPUT_TIMESTEPS;
    let mut tick: u64 = 0;

    loop {
        match fetch_frame(&cfg.sensing_url).await {
            Ok(amplitudes) => {
                tick += 1;
                buffer.extend(amplitudes);
                while buffer.len() > 2 * cap {
                    let extra = buffer.len() - cap;
                    buffer.drain(0..extra);
                }
                if buffer.len() >= cap {
                    let window = CsiWindow { data: buffer[buffer.len() - cap..].to_vec() };
                    if let Ok(pred) = engine.infer(&window) {
                        // v0.0.1 ships single-node — fusion is a no-op for
                        // N=1. v0.2.0 will append additional per-node
                        // predictions to a vec and call
                        // `fusion::fuse_confidence_weighted` before emit.
                        publisher::person_count(tick, &pred, 1);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "sensing-server fetch failed");
            }
        }
        sleep(Duration::from_millis(cfg.poll_ms)).await;
    }
}

async fn fetch_frame(url: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let url = url.to_string();
    let body = tokio::task::spawn_blocking(move || -> Result<String, ureq::Error> {
        Ok(ureq::get(&url).call()?.into_string()?)
    })
    .await??;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let snapshot = json.get("snapshot").unwrap_or(&json);
    let nodes = snapshot
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or("missing nodes[]")?;
    let amplitude = nodes
        .first()
        .and_then(|n| n.get("amplitude"))
        .and_then(|v| v.as_array())
        .ok_or("missing nodes[0].amplitude[]")?;
    Ok(amplitude
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect())
}

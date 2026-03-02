//! Training API with WebSocket progress streaming.
//!
//! Provides REST endpoints for starting, stopping, and monitoring training runs.
//! Training runs in a background tokio task. Progress updates are broadcast via
//! a `tokio::sync::broadcast` channel that the WebSocket handler subscribes to.
//!
//! Since the full training pipeline depends on `tch-rs` (PyTorch), this module
//! implements a **simulated training mode** that generates realistic progress
//! updates. Real training is gated behind a `#[cfg(feature = "training")]` flag.
//!
//! On completion, the best model is automatically exported as `.rvf` using `RvfBuilder`.
//!
//! REST endpoints:
//! - `POST /api/v1/train/start`    — start a training run
//! - `POST /api/v1/train/stop`     — stop the active training
//! - `GET  /api/v1/train/status`   — get current training status
//! - `POST /api/v1/train/pretrain` — start contrastive pretraining
//! - `POST /api/v1/train/lora`     — start LoRA fine-tuning
//!
//! WebSocket:
//! - `WS /ws/train/progress`       — streaming training progress

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info, warn};

use crate::rvf_container::RvfBuilder;

// ── Constants ────────────────────────────────────────────────────────────────

/// Directory for trained model output.
pub const MODELS_DIR: &str = "data/models";

// ── Types ────────────────────────────────────────────────────────────────────

/// Training configuration submitted with a start request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_early_stopping_patience")]
    pub early_stopping_patience: u32,
    #[serde(default = "default_warmup_epochs")]
    pub warmup_epochs: u32,
    /// Path to a pretrained RVF model to fine-tune from.
    pub pretrained_rvf: Option<String>,
    /// LoRA profile name for environment-specific fine-tuning.
    pub lora_profile: Option<String>,
}

fn default_epochs() -> u32 { 100 }
fn default_batch_size() -> u32 { 8 }
fn default_learning_rate() -> f64 { 0.001 }
fn default_weight_decay() -> f64 { 1e-4 }
fn default_early_stopping_patience() -> u32 { 20 }
fn default_warmup_epochs() -> u32 { 5 }

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            weight_decay: default_weight_decay(),
            early_stopping_patience: default_early_stopping_patience(),
            warmup_epochs: default_warmup_epochs(),
            pretrained_rvf: None,
            lora_profile: None,
        }
    }
}

/// Request body for `POST /api/v1/train/start`.
#[derive(Debug, Deserialize)]
pub struct StartTrainingRequest {
    pub dataset_ids: Vec<String>,
    pub config: TrainingConfig,
}

/// Request body for `POST /api/v1/train/pretrain`.
#[derive(Debug, Deserialize)]
pub struct PretrainRequest {
    pub dataset_ids: Vec<String>,
    #[serde(default = "default_pretrain_epochs")]
    pub epochs: u32,
    #[serde(default = "default_learning_rate")]
    pub lr: f64,
}

fn default_pretrain_epochs() -> u32 { 50 }

/// Request body for `POST /api/v1/train/lora`.
#[derive(Debug, Deserialize)]
pub struct LoraTrainRequest {
    pub base_model_id: String,
    pub dataset_ids: Vec<String>,
    pub profile_name: String,
    #[serde(default = "default_lora_rank")]
    pub rank: u8,
    #[serde(default = "default_lora_epochs")]
    pub epochs: u32,
}

fn default_lora_rank() -> u8 { 8 }
fn default_lora_epochs() -> u32 { 30 }

/// Current training status (returned by `GET /api/v1/train/status`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    pub active: bool,
    pub epoch: u32,
    pub total_epochs: u32,
    pub train_loss: f64,
    pub val_pck: f64,
    pub val_oks: f64,
    pub lr: f64,
    pub best_pck: f64,
    pub best_epoch: u32,
    pub patience_remaining: u32,
    pub eta_secs: Option<u64>,
    pub phase: String,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        Self {
            active: false,
            epoch: 0,
            total_epochs: 0,
            train_loss: 0.0,
            val_pck: 0.0,
            val_oks: 0.0,
            lr: 0.0,
            best_pck: 0.0,
            best_epoch: 0,
            patience_remaining: 0,
            eta_secs: None,
            phase: "idle".to_string(),
        }
    }
}

/// Progress update sent over WebSocket.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingProgress {
    pub epoch: u32,
    pub batch: u32,
    pub total_batches: u32,
    pub train_loss: f64,
    pub val_pck: f64,
    pub val_oks: f64,
    pub lr: f64,
    pub phase: String,
}

/// Runtime training state stored in `AppStateInner`.
pub struct TrainingState {
    /// Current status snapshot.
    pub status: TrainingStatus,
    /// Handle to the background training task (for cancellation).
    pub task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            status: TrainingStatus::default(),
            task_handle: None,
        }
    }
}

/// Shared application state type.
pub type AppState = Arc<RwLock<super::AppStateInner>>;

// ── Simulated training loop ──────────────────────────────────────────────────

/// Simulated training loop that generates realistic loss/metric curves.
///
/// This allows the UI to be developed and tested without GPU/PyTorch.
async fn simulated_training_loop(
    state: AppState,
    progress_tx: broadcast::Sender<String>,
    config: TrainingConfig,
    _dataset_ids: Vec<String>,
    training_type: &str,
) {
    let total_epochs = config.epochs;
    let total_batches = 50u32; // simulated batch count per epoch
    let patience = config.early_stopping_patience;
    let mut best_pck = 0.0f64;
    let mut best_epoch = 0u32;
    let mut patience_remaining = patience;

    info!(
        "Simulated {training_type} training started: {total_epochs} epochs, lr={}",
        config.learning_rate
    );

    for epoch in 1..=total_epochs {
        // Check if training was cancelled.
        {
            let s = state.read().await;
            if !s.training_state.status.active {
                info!("Training cancelled at epoch {epoch}");
                break;
            }
        }

        // Determine phase.
        let phase = if epoch <= config.warmup_epochs {
            "warmup"
        } else {
            "training"
        };

        // Simulate batches within the epoch.
        let lr = if epoch <= config.warmup_epochs {
            config.learning_rate * (epoch as f64 / config.warmup_epochs as f64)
        } else {
            // Cosine decay.
            let progress =
                (epoch - config.warmup_epochs) as f64 / (total_epochs - config.warmup_epochs).max(1) as f64;
            config.learning_rate * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
        };

        // Simulated loss: exponential decay with noise.
        let base_loss = 2.0 * (-0.03 * epoch as f64).exp() + 0.05;
        let noise = ((epoch as f64 * 7.31).sin() * 0.02).abs();
        let train_loss = base_loss + noise;

        for batch in 1..=total_batches {
            let progress = TrainingProgress {
                epoch,
                batch,
                total_batches,
                train_loss,
                val_pck: 0.0, // only set after validation
                val_oks: 0.0,
                lr,
                phase: phase.to_string(),
            };
            if let Ok(json) = serde_json::to_string(&progress) {
                let _ = progress_tx.send(json);
            }

            // Simulate ~20ms per batch.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Validation phase.
        let val_pck = (1.0 - (-0.04 * epoch as f64).exp()) * 0.92
            + ((epoch as f64 * 3.17).sin() * 0.01).abs();
        let val_oks = val_pck * 0.88;

        let val_progress = TrainingProgress {
            epoch,
            batch: total_batches,
            total_batches,
            train_loss,
            val_pck,
            val_oks,
            lr,
            phase: "validation".to_string(),
        };
        if let Ok(json) = serde_json::to_string(&val_progress) {
            let _ = progress_tx.send(json);
        }

        // Update best metrics.
        if val_pck > best_pck {
            best_pck = val_pck;
            best_epoch = epoch;
            patience_remaining = patience;
        } else {
            patience_remaining = patience_remaining.saturating_sub(1);
        }

        // Estimate remaining time.
        let elapsed_epochs = epoch;
        let remaining_epochs = total_epochs.saturating_sub(epoch);
        // Each epoch takes ~(total_batches * 20ms + ~50ms validation).
        let ms_per_epoch = total_batches as u64 * 20 + 50;
        let eta_secs = (remaining_epochs as u64 * ms_per_epoch) / 1000;

        // Update shared state.
        {
            let mut s = state.write().await;
            s.training_state.status = TrainingStatus {
                active: true,
                epoch,
                total_epochs,
                train_loss,
                val_pck,
                val_oks,
                lr,
                best_pck,
                best_epoch,
                patience_remaining,
                eta_secs: Some(eta_secs),
                phase: phase.to_string(),
            };
        }

        // Early stopping check.
        if patience_remaining == 0 {
            info!(
                "Early stopping at epoch {epoch} (best={best_epoch}, PCK={best_pck:.4})"
            );
            let stop_progress = TrainingProgress {
                epoch,
                batch: total_batches,
                total_batches,
                train_loss,
                val_pck,
                val_oks,
                lr,
                phase: "early_stopped".to_string(),
            };
            if let Ok(json) = serde_json::to_string(&stop_progress) {
                let _ = progress_tx.send(json);
            }
            break;
        }

        let _ = elapsed_epochs; // suppress warning
    }

    // Training complete: export model as .rvf.
    let completed_phase;
    {
        let s = state.read().await;
        completed_phase = if s.training_state.status.active {
            "completed"
        } else {
            "cancelled"
        };
    }

    // Emit completion message.
    let completion = TrainingProgress {
        epoch: best_epoch,
        batch: 0,
        total_batches: 0,
        train_loss: 0.0,
        val_pck: best_pck,
        val_oks: best_pck * 0.88,
        lr: 0.0,
        phase: completed_phase.to_string(),
    };
    if let Ok(json) = serde_json::to_string(&completion) {
        let _ = progress_tx.send(json);
    }

    // Build and save a demo .rvf file if training completed.
    if completed_phase == "completed" || completed_phase == "early_stopped" {
        if let Err(e) = tokio::fs::create_dir_all(MODELS_DIR).await {
            error!("Failed to create models directory: {e}");
        } else {
            let model_id = format!(
                "trained-{}-{}",
                training_type,
                chrono::Utc::now().format("%Y%m%d_%H%M%S")
            );
            let rvf_path = PathBuf::from(MODELS_DIR).join(format!("{model_id}.rvf"));

            // Build a small demo RVF container.
            let mut builder = RvfBuilder::new();
            builder.add_manifest(
                &model_id,
                env!("CARGO_PKG_VERSION"),
                &format!("WiFi DensePose {training_type} model (simulated)"),
            );
            builder.add_metadata(&serde_json::json!({
                "training": {
                    "type": training_type,
                    "epochs": total_epochs,
                    "best_epoch": best_epoch,
                    "best_pck": best_pck,
                    "best_oks": best_pck * 0.88,
                    "simulated": true,
                },
            }));

            // Placeholder weights: 17 keypoints * 56 subcarriers * 3 dims.
            let n_weights = 17 * 56 * 3;
            let weights: Vec<f32> = (0..n_weights)
                .map(|i| (i as f32 * 0.001).sin())
                .collect();
            builder.add_weights(&weights);

            if let Err(e) = builder.write_to_file(&rvf_path) {
                error!("Failed to write trained model RVF: {e}");
            } else {
                info!(
                    "Trained model saved: {} ({} params)",
                    rvf_path.display(),
                    n_weights
                );
            }
        }
    }

    // Mark training as inactive.
    {
        let mut s = state.write().await;
        s.training_state.status.active = false;
        s.training_state.status.phase = completed_phase.to_string();
        s.training_state.task_handle = None;
    }

    info!("Simulated {training_type} training finished: phase={completed_phase}");
}

// ── Axum handlers ────────────────────────────────────────────────────────────

async fn start_training(
    State(state): State<AppState>,
    Json(body): Json<StartTrainingRequest>,
) -> Json<serde_json::Value> {
    // Check if training is already active.
    {
        let s = state.read().await;
        if s.training_state.status.active {
            return Json(serde_json::json!({
                "status": "error",
                "message": "Training is already active. Stop it first.",
                "current_epoch": s.training_state.status.epoch,
                "total_epochs": s.training_state.status.total_epochs,
            }));
        }
    }

    let config = body.config.clone();
    let dataset_ids = body.dataset_ids.clone();

    // Mark training as active and spawn background task.
    let progress_tx;
    {
        let s = state.read().await;
        progress_tx = s.training_progress_tx.clone();
    }

    {
        let mut s = state.write().await;
        s.training_state.status = TrainingStatus {
            active: true,
            epoch: 0,
            total_epochs: config.epochs,
            train_loss: 0.0,
            val_pck: 0.0,
            val_oks: 0.0,
            lr: config.learning_rate,
            best_pck: 0.0,
            best_epoch: 0,
            patience_remaining: config.early_stopping_patience,
            eta_secs: None,
            phase: "initializing".to_string(),
        };
    }

    let state_clone = state.clone();
    let handle = tokio::spawn(async move {
        simulated_training_loop(state_clone, progress_tx, config, dataset_ids, "supervised")
            .await;
    });

    {
        let mut s = state.write().await;
        s.training_state.task_handle = Some(handle);
    }

    Json(serde_json::json!({
        "status": "started",
        "type": "supervised",
        "dataset_ids": body.dataset_ids,
        "config": body.config,
    }))
}

async fn stop_training(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut s = state.write().await;
    if !s.training_state.status.active {
        return Json(serde_json::json!({
            "status": "error",
            "message": "No training is currently active.",
        }));
    }

    s.training_state.status.active = false;
    s.training_state.status.phase = "stopping".to_string();

    // The background task checks the active flag and will exit.
    // We do not abort the handle — we let it finish the current batch gracefully.

    info!("Training stop requested");

    Json(serde_json::json!({
        "status": "stopping",
        "epoch": s.training_state.status.epoch,
        "best_pck": s.training_state.status.best_pck,
    }))
}

async fn training_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    Json(serde_json::to_value(&s.training_state.status).unwrap_or_default())
}

async fn start_pretrain(
    State(state): State<AppState>,
    Json(body): Json<PretrainRequest>,
) -> Json<serde_json::Value> {
    {
        let s = state.read().await;
        if s.training_state.status.active {
            return Json(serde_json::json!({
                "status": "error",
                "message": "Training is already active. Stop it first.",
            }));
        }
    }

    let config = TrainingConfig {
        epochs: body.epochs,
        learning_rate: body.lr,
        warmup_epochs: (body.epochs / 10).max(1),
        early_stopping_patience: body.epochs + 1, // no early stopping for pretrain
        ..Default::default()
    };

    let progress_tx;
    {
        let s = state.read().await;
        progress_tx = s.training_progress_tx.clone();
    }

    {
        let mut s = state.write().await;
        s.training_state.status = TrainingStatus {
            active: true,
            total_epochs: body.epochs,
            phase: "initializing".to_string(),
            ..Default::default()
        };
    }

    let state_clone = state.clone();
    let dataset_ids = body.dataset_ids.clone();
    let handle = tokio::spawn(async move {
        simulated_training_loop(state_clone, progress_tx, config, dataset_ids, "pretrain")
            .await;
    });

    {
        let mut s = state.write().await;
        s.training_state.task_handle = Some(handle);
    }

    Json(serde_json::json!({
        "status": "started",
        "type": "pretrain",
        "epochs": body.epochs,
        "lr": body.lr,
        "dataset_ids": body.dataset_ids,
    }))
}

async fn start_lora_training(
    State(state): State<AppState>,
    Json(body): Json<LoraTrainRequest>,
) -> Json<serde_json::Value> {
    {
        let s = state.read().await;
        if s.training_state.status.active {
            return Json(serde_json::json!({
                "status": "error",
                "message": "Training is already active. Stop it first.",
            }));
        }
    }

    let config = TrainingConfig {
        epochs: body.epochs,
        learning_rate: 0.0005, // lower LR for LoRA
        warmup_epochs: 2,
        early_stopping_patience: 10,
        pretrained_rvf: Some(body.base_model_id.clone()),
        lora_profile: Some(body.profile_name.clone()),
        ..Default::default()
    };

    let progress_tx;
    {
        let s = state.read().await;
        progress_tx = s.training_progress_tx.clone();
    }

    {
        let mut s = state.write().await;
        s.training_state.status = TrainingStatus {
            active: true,
            total_epochs: body.epochs,
            phase: "initializing".to_string(),
            ..Default::default()
        };
    }

    let state_clone = state.clone();
    let dataset_ids = body.dataset_ids.clone();
    let handle = tokio::spawn(async move {
        simulated_training_loop(state_clone, progress_tx, config, dataset_ids, "lora")
            .await;
    });

    {
        let mut s = state.write().await;
        s.training_state.task_handle = Some(handle);
    }

    Json(serde_json::json!({
        "status": "started",
        "type": "lora",
        "base_model_id": body.base_model_id,
        "profile_name": body.profile_name,
        "rank": body.rank,
        "epochs": body.epochs,
        "dataset_ids": body.dataset_ids,
    }))
}

// ── WebSocket handler for training progress ──────────────────────────────────

async fn ws_train_progress_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_train_ws_client(socket, state))
}

async fn handle_train_ws_client(mut socket: WebSocket, state: AppState) {
    let mut rx = {
        let s = state.read().await;
        s.training_progress_tx.subscribe()
    };

    info!("WebSocket client connected (train/progress)");

    // Send current status immediately.
    {
        let s = state.read().await;
        if let Ok(json) = serde_json::to_string(&s.training_state.status) {
            let msg = serde_json::json!({
                "type": "status",
                "data": serde_json::from_str::<serde_json::Value>(&json).unwrap_or_default(),
            });
            let _ = socket
                .send(Message::Text(msg.to_string().into()))
                .await;
        }
    }

    loop {
        tokio::select! {
            result = rx.recv() => {
                match result {
                    Ok(progress_json) => {
                        let parsed = serde_json::from_str::<serde_json::Value>(&progress_json)
                            .unwrap_or_default();
                        let ws_msg = serde_json::json!({
                            "type": "progress",
                            "data": parsed,
                        });
                        if socket.send(Message::Text(ws_msg.to_string().into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Train WS client lagged by {n} messages");
                    }
                    Err(_) => break,
                }
            }
            ws_msg = socket.recv() => {
                match ws_msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // ignore client messages
                }
            }
        }
    }

    info!("WebSocket client disconnected (train/progress)");
}

// ── Router factory ───────────────────────────────────────────────────────────

/// Build the training API sub-router.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/train/start", post(start_training))
        .route("/api/v1/train/stop", post(stop_training))
        .route("/api/v1/train/status", get(training_status))
        .route("/api/v1/train/pretrain", post(start_pretrain))
        .route("/api/v1/train/lora", post(start_lora_training))
        .route("/ws/train/progress", get(ws_train_progress_handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_config_defaults() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 8);
        assert!((config.learning_rate - 0.001).abs() < 1e-9);
        assert_eq!(config.warmup_epochs, 5);
        assert_eq!(config.early_stopping_patience, 20);
    }

    #[test]
    fn training_status_default_is_inactive() {
        let status = TrainingStatus::default();
        assert!(!status.active);
        assert_eq!(status.phase, "idle");
    }

    #[test]
    fn training_progress_serializes() {
        let progress = TrainingProgress {
            epoch: 10,
            batch: 25,
            total_batches: 50,
            train_loss: 0.35,
            val_pck: 0.72,
            val_oks: 0.63,
            lr: 0.0008,
            phase: "training".to_string(),
        };
        let json = serde_json::to_string(&progress).unwrap();
        assert!(json.contains("\"epoch\":10"));
        assert!(json.contains("\"phase\":\"training\""));
    }

    #[test]
    fn training_config_deserializes_with_defaults() {
        let json = r#"{"epochs": 50}"#;
        let config: TrainingConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.epochs, 50);
        assert_eq!(config.batch_size, 8); // default
        assert!((config.learning_rate - 0.001).abs() < 1e-9); // default
    }
}

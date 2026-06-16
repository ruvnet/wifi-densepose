use std::fs::File;
use std::io::Read;
use std::time::{Duration, Instant};

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

/// OTA update port on ESP32 nodes.
const OTA_PORT: u16 = 8032;

/// OTA endpoint path.
const OTA_PATH: &str = "/ota";

/// OTA status endpoint path.
const OTA_STATUS_PATH: &str = "/ota/status";

/// Request timeout for OTA uploads.
const OTA_TIMEOUT_SECS: u64 = 120;

/// Time to wait for a node to reboot after OTA upload.
const OTA_REBOOT_TIMEOUT_SECS: u64 = 30;

/// Initial wait before polling for the post-reboot status.
const OTA_REBOOT_SETTLE_SECS: u64 = 2;

/// Poll cadence while waiting for reboot completion.
const OTA_POLL_INTERVAL_MILLIS: u64 = 500;

/// Push firmware to a single node via HTTP OTA (port 8032).
///
/// Protocol:
/// 1. Read the firmware image from disk
/// 2. GET /ota/status to capture the current version and partition limit
/// 3. POST the raw binary image to http://<node_ip>:8032/ota
/// 4. Include Authorization: Bearer <psk> when a PSK is provided
/// 5. Wait for the node to reboot and report the post-update version
#[tauri::command]
pub async fn ota_update(
    app: AppHandle,
    node_ip: String,
    firmware_path: String,
    psk: Option<String>,
) -> OtaResult {
    let start_time = Instant::now();

    emit_ota_progress(
        &app,
        &node_ip,
        "preparing",
        0.0,
        Some("Checking node status...".into()),
    );

    let client = match build_http_client() {
        Ok(client) => client,
        Err(err) => return ota_failure(&app, node_ip, None, start_time, err),
    };

    let current_status = match fetch_ota_status(&client, &node_ip).await {
        Ok(status) => status,
        Err(err) => return ota_failure(&app, node_ip, None, start_time, err),
    };

    let OtaStatusResponse {
        version: previous_version,
        max_size,
    } = current_status;
    let previous_version = Some(previous_version);

    let firmware_data = match read_firmware(&firmware_path) {
        Ok(data) => data,
        Err(err) => return ota_failure(&app, node_ip, previous_version.clone(), start_time, err),
    };

    let firmware_size = firmware_data.len() as u64;
    if let Some(max_size) = max_size {
        if firmware_size > max_size {
            return ota_failure(
                &app,
                node_ip,
                previous_version.clone(),
                start_time,
                format!(
                    "Firmware image is too large for the OTA partition ({} > {} bytes)",
                    firmware_size, max_size
                ),
            );
        }
    }

    emit_ota_progress(
        &app,
        &node_ip,
        "uploading",
        10.0,
        Some(format!(
            "Uploading {} bytes to {}...",
            firmware_size, node_ip
        )),
    );

    let mut request = client
        .post(ota_upload_url(&node_ip))
        .header(CONTENT_TYPE, "application/octet-stream")
        .body(firmware_data);

    if let Some(token) = psk.as_deref() {
        request = request.header(AUTHORIZATION, bearer_token(token));
    }

    let response = match request.send().await {
        Ok(response) => response,
        Err(err) => {
            return ota_failure(
                &app,
                node_ip,
                previous_version.clone(),
                start_time,
                format!("OTA upload failed: {}", err),
            )
        }
    };

    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    if !status.is_success() {
        return ota_failure(
            &app,
            node_ip,
            previous_version.clone(),
            start_time,
            format!("OTA failed with HTTP {}: {}", status, body),
        );
    }

    emit_ota_progress(
        &app,
        &node_ip,
        "rebooting",
        80.0,
        Some("Waiting for node reboot...".into()),
    );

    let new_status = match wait_for_reboot_and_status(
        &client,
        &node_ip,
        Duration::from_secs(OTA_REBOOT_TIMEOUT_SECS),
    )
    .await
    {
        Ok(status) => status,
        Err(err) => return ota_failure(&app, node_ip, previous_version.clone(), start_time, err),
    };

    let duration_ms = start_time.elapsed().as_millis() as u64;

    emit_ota_progress(
        &app,
        &node_ip,
        "completed",
        100.0,
        Some(format!(
            "OTA completed in {:.1}s",
            duration_ms as f64 / 1000.0
        )),
    );

    OtaResult {
        success: true,
        node_ip,
        previous_version,
        new_version: Some(new_status.version),
        duration_ms,
        error: None,
    }
}

/// Push firmware to multiple nodes with rolling update strategy.
///
/// Strategy options:
/// - Sequential: One node at a time
/// - Parallel: All nodes simultaneously (max_concurrent)
/// - TdmSafe: Respects TDM slots to avoid disruption
#[tauri::command]
pub async fn batch_ota_update(
    app: AppHandle,
    node_ips: Vec<String>,
    firmware_path: String,
    psk: Option<String>,
    strategy: Option<String>,
    max_concurrent: Option<usize>,
) -> Vec<OtaResult> {
    let total_nodes = node_ips.len();
    let strategy = strategy.unwrap_or_else(|| "sequential".into());
    let max_concurrent = max_concurrent.unwrap_or(1).max(1);

    emit_batch_ota_progress(&app, "starting", total_nodes, 0, 0, None);

    let mut results = Vec::new();
    let mut completed = 0;
    let mut failed = 0;

    match strategy.as_str() {
        "parallel" => {
            let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent));
            let app = std::sync::Arc::new(app.clone());

            let tasks: Vec<_> = node_ips
                .into_iter()
                .map(|ip| {
                    let sem = semaphore.clone();
                    let app_clone = app.clone();
                    let fw_path = firmware_path.clone();
                    let psk_clone = psk.clone();

                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        ota_update((*app_clone).clone(), ip, fw_path, psk_clone).await
                    }
                })
                .collect();

            let task_results = futures::future::join_all(tasks).await;

            for result in task_results {
                if result.success {
                    completed += 1;
                } else {
                    failed += 1;
                }
                results.push(result);
            }
        }
        _ => {
            for ip in node_ips {
                emit_batch_ota_progress(
                    &app,
                    "updating",
                    total_nodes,
                    completed,
                    failed,
                    Some(ip.clone()),
                );

                let result = ota_update(app.clone(), ip, firmware_path.clone(), psk.clone()).await;
                if result.success {
                    completed += 1;
                } else {
                    failed += 1;
                }
                results.push(result);
            }
        }
    }

    emit_batch_ota_progress(&app, "completed", total_nodes, completed, failed, None);

    results
}

/// Check if a node's OTA endpoint is accessible.
#[tauri::command]
pub async fn check_ota_endpoint(node_ip: String) -> Result<OtaEndpointInfo, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let url = ota_status_url(&node_ip);

    match client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                let body = response.text().await.unwrap_or_default();

                // Try to parse as JSON
                let version = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|v| {
                        v.get("version")
                            .and_then(|v| v.as_str().map(|s| s.to_string()))
                    });

                Ok(OtaEndpointInfo {
                    reachable: true,
                    ota_supported: true,
                    current_version: version,
                    psk_required: false, // Would need to check headers
                })
            } else {
                Ok(OtaEndpointInfo {
                    reachable: true,
                    ota_supported: response.status() != reqwest::StatusCode::NOT_FOUND,
                    current_version: None,
                    psk_required: response.status() == reqwest::StatusCode::UNAUTHORIZED,
                })
            }
        }
        Err(_) => Ok(OtaEndpointInfo {
            reachable: false,
            ota_supported: false,
            current_version: None,
            psk_required: false,
        }),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtaResult {
    pub success: bool,
    pub node_ip: String,
    pub previous_version: Option<String>,
    pub new_version: Option<String>,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OtaProgress {
    pub node_ip: String,
    pub phase: String,
    pub progress_pct: f32,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchOtaProgress {
    pub phase: String,
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub current_node: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OtaEndpointInfo {
    pub reachable: bool,
    pub ota_supported: bool,
    pub current_version: Option<String>,
    pub psk_required: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ota_upload_url() {
        assert_eq!(
            ota_upload_url("192.168.1.42"),
            "http://192.168.1.42:8032/ota"
        );
    }

    #[test]
    fn test_ota_status_url() {
        assert_eq!(
            ota_status_url("ruview-node.local"),
            "http://ruview-node.local:8032/ota/status"
        );
    }

    #[test]
    fn test_bearer_token_header() {
        assert_eq!(bearer_token("secret"), "Bearer secret");
    }

    #[test]
    fn test_parse_status_response() {
        let body = r#"{
            "version": "1.2.3",
            "date": "2026-06-11",
            "time": "10:30:00",
            "running_partition": "ota_0",
            "next_partition": "ota_1",
            "max_size": 1048576
        }"#;

        let status: OtaStatusResponse = serde_json::from_str(body).unwrap();
        assert_eq!(status.version, "1.2.3");
        assert_eq!(status.max_size, Some(1_048_576));
    }
}

fn build_http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(OTA_TIMEOUT_SECS))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))
}

fn read_firmware(path: &str) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("Cannot read firmware: {}", e))?;

    let mut firmware_data = Vec::new();
    file.read_to_end(&mut firmware_data)
        .map_err(|e| format!("Failed to read firmware: {}", e))?;

    Ok(firmware_data)
}

fn ota_upload_url(node_ip: &str) -> String {
    format!("http://{}:{}{}", node_ip, OTA_PORT, OTA_PATH)
}

fn ota_status_url(node_ip: &str) -> String {
    format!("http://{}:{}{}", node_ip, OTA_PORT, OTA_STATUS_PATH)
}

fn bearer_token(psk: &str) -> String {
    format!("Bearer {}", psk)
}

fn emit_ota_progress(
    app: &AppHandle,
    node_ip: &str,
    phase: &str,
    progress_pct: f32,
    message: Option<String>,
) {
    let _ = app.emit(
        "ota-progress",
        OtaProgress {
            node_ip: node_ip.to_string(),
            phase: phase.into(),
            progress_pct,
            message,
        },
    );
}

fn emit_batch_ota_progress(
    app: &AppHandle,
    phase: &str,
    total: usize,
    completed: usize,
    failed: usize,
    current_node: Option<String>,
) {
    let _ = app.emit(
        "batch-ota-progress",
        BatchOtaProgress {
            phase: phase.into(),
            total,
            completed,
            failed,
            current_node,
        },
    );
}

fn ota_failure(
    app: &AppHandle,
    node_ip: String,
    previous_version: Option<String>,
    start_time: Instant,
    error: String,
) -> OtaResult {
    emit_ota_progress(app, &node_ip, "failed", 0.0, Some(error.clone()));

    OtaResult {
        success: false,
        node_ip,
        previous_version,
        new_version: None,
        duration_ms: start_time.elapsed().as_millis() as u64,
        error: Some(error),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OtaStatusResponse {
    version: String,
    #[serde(default)]
    max_size: Option<u64>,
}

async fn fetch_ota_status(
    client: &reqwest::Client,
    node_ip: &str,
) -> Result<OtaStatusResponse, String> {
    let response = client
        .get(ota_status_url(node_ip))
        .send()
        .await
        .map_err(|e| format!("Failed to query OTA status from {}: {}", node_ip, e))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| format!("Failed to read OTA status from {}: {}", node_ip, e))?;

    if !status.is_success() {
        return Err(format!(
            "OTA status returned HTTP {} for {}: {}",
            status, node_ip, body
        ));
    }

    serde_json::from_str::<OtaStatusResponse>(&body)
        .map_err(|e| format!("Failed to parse OTA status from {}: {}", node_ip, e))
}

async fn wait_for_reboot_and_status(
    client: &reqwest::Client,
    node_ip: &str,
    timeout: Duration,
) -> Result<OtaStatusResponse, String> {
    let deadline = Instant::now() + timeout;
    tokio::time::sleep(Duration::from_secs(OTA_REBOOT_SETTLE_SECS)).await;

    loop {
        match client.get(ota_status_url(node_ip)).send().await {
            Ok(response) if response.status().is_success() => {
                let body = response
                    .text()
                    .await
                    .map_err(|e| format!("Failed to read OTA status from {}: {}", node_ip, e))?;
                return serde_json::from_str::<OtaStatusResponse>(&body)
                    .map_err(|e| format!("Failed to parse OTA status from {}: {}", node_ip, e));
            }
            Ok(_) | Err(_) => {}
        }

        if Instant::now() >= deadline {
            return Err(format!(
                "Node {} did not come back online within {}s",
                node_ip,
                timeout.as_secs()
            ));
        }

        tokio::time::sleep(Duration::from_millis(OTA_POLL_INTERVAL_MILLIS)).await;
    }
}

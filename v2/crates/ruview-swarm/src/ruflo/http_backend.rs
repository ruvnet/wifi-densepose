//! HTTP backend that calls the claude-flow daemon via JSON-RPC 2.0.
//! Default endpoint: http://localhost:3000/rpc
//!
//! Start the daemon with: npx @claude-flow/cli@latest daemon start

use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use super::backend::*;

/// Per-request timeout applied to every JSON-RPC call.
/// A dead or slow daemon must not stall swarm operation loops.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

pub struct HttpRufloBackend {
    client:     reqwest::Client,
    base_url:   String,
    request_id: AtomicU64,
}

impl HttpRufloBackend {
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("failed to build reqwest client");
        Self {
            client,
            base_url:   base_url.trim_end_matches('/').to_string(),
            request_id: AtomicU64::new(1),
        }
    }

    pub fn localhost() -> Self { Self::new("http://localhost:3000") }

    async fn call_tool(
        &self,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, RufloError> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": id,
            "params": { "name": tool, "arguments": args }
        });

        let resp = self.client
            .post(format!("{}/rpc", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| RufloError::Network(e.to_string()))?;

        let json: serde_json::Value = resp.json().await
            .map_err(|e| RufloError::Serialize(e.to_string()))?;

        if let Some(err) = json.get("error") {
            return Err(RufloError::Tool(err.to_string()));
        }

        Ok(json["result"].clone())
    }
}

#[async_trait]
impl RufloBackend for HttpRufloBackend {
    async fn store_mission(&self, key: &str, value: &str, namespace: &str)
        -> Result<(), RufloError>
    {
        self.call_tool("memory_store", serde_json::json!({
            "key": key, "value": value, "namespace": namespace
        })).await?;
        Ok(())
    }

    async fn search_missions(&self, query: &str, limit: usize, namespace: &str)
        -> Result<Vec<MissionMemoryEntry>, RufloError>
    {
        let result = self.call_tool("memory_search", serde_json::json!({
            "query": query, "namespace": namespace, "limit": limit
        })).await?;
        let entries: Vec<MissionMemoryEntry> = serde_json::from_value(result)
            .unwrap_or_default();
        Ok(entries)
    }

    async fn store_pattern(&self, pattern: &str, pattern_type: &str, confidence: f32)
        -> Result<(), RufloError>
    {
        self.call_tool("agentdb_pattern-store", serde_json::json!({
            "pattern": pattern, "type": pattern_type, "confidence": confidence
        })).await?;
        Ok(())
    }

    async fn search_patterns(&self, query: &str, top_k: usize, min_confidence: f32)
        -> Result<Vec<PatternEntry>, RufloError>
    {
        let result = self.call_tool("agentdb_pattern-search", serde_json::json!({
            "query": query, "topK": top_k, "minConfidence": min_confidence
        })).await?;
        let entries: Vec<PatternEntry> = serde_json::from_value(
            result["results"].clone()
        ).unwrap_or_default();
        Ok(entries)
    }

    async fn mavlink_is_safe(&self, message_repr: &str) -> Result<bool, RufloError> {
        let result = self.call_tool("aidefence_is_safe", serde_json::json!({
            "input": message_repr
        })).await?;
        Ok(result["safe"].as_bool().unwrap_or(true))
    }

    async fn mavlink_scan(&self, message_repr: &str) -> Result<MavlinkScanResult, RufloError> {
        let result = self.call_tool("aidefence_scan", serde_json::json!({
            "input": message_repr, "quick": false
        })).await?;
        let safe = result["safe"].as_bool().unwrap_or(true);
        let threats: Vec<String> = result["threats"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v["type"].as_str().map(String::from)).collect())
            .unwrap_or_default();
        Ok(MavlinkScanResult { safe, threats })
    }

    async fn trajectory_start(&self, task: &str, agent: &str)
        -> Result<String, RufloError>
    {
        let result = self.call_tool("hooks_intelligence_trajectory-start", serde_json::json!({
            "task": task, "agent": agent
        })).await?;
        Ok(result["trajectoryId"]
            .as_str()
            .unwrap_or("unknown-traj")
            .to_string())
    }

    async fn trajectory_step(
        &self,
        trajectory_id: &str,
        action: &str,
        result_str: &str,
        quality: f32,
    ) -> Result<(), RufloError> {
        self.call_tool("hooks_intelligence_trajectory-step", serde_json::json!({
            "trajectoryId": trajectory_id,
            "action": action,
            "result": result_str,
            "quality": quality
        })).await?;
        Ok(())
    }

    async fn trajectory_end(
        &self,
        trajectory_id: &str,
        success: bool,
        feedback: Option<&str>,
    ) -> Result<(), RufloError> {
        let mut args = serde_json::json!({
            "trajectoryId": trajectory_id,
            "success": success
        });
        if let Some(fb) = feedback {
            args["feedback"] = fb.into();
        }
        self.call_tool("hooks_intelligence_trajectory-end", args).await?;
        Ok(())
    }
}

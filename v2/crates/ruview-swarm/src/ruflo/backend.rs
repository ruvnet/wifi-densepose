//! RufloBackend trait and shared types.
use async_trait::async_trait;

/// Error type for Ruflo backend operations.
#[derive(Debug, thiserror::Error)]
pub enum RufloError {
    #[error("network error: {0}")]
    Network(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("serialization error: {0}")]
    Serialize(String),
}

/// A past mission retrieved from AgentDB memory.
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct MissionMemoryEntry {
    pub key: String,
    pub value: String,  // JSON-encoded mission summary
    pub score: f32,
}

/// A coordination pattern retrieved from AgentDB pattern store.
#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct PatternEntry {
    pub pattern: String,
    pub pattern_type: String,
    pub confidence: f32,
    pub score: f32,
}

/// Result of an AIDefence MAVLink message scan.
#[derive(Debug, Clone)]
pub struct MavlinkScanResult {
    pub safe: bool,
    pub threats: Vec<String>,
}

/// Core Ruflo capability trait.
///
/// Two implementations:
/// - `HttpRufloBackend` (feature=ruflo): calls the claude-flow daemon at localhost:3000
/// - `MockRufloBackend`: in-memory mock for testing (always available)
#[async_trait]
pub trait RufloBackend: Send + Sync {
    // ── MissionMemory (claude-flow: memory_store / memory_search) ────
    async fn store_mission(&self, key: &str, summary: &str, namespace: &str)
        -> Result<(), RufloError>;
    async fn search_missions(&self, query: &str, limit: usize, namespace: &str)
        -> Result<Vec<MissionMemoryEntry>, RufloError>;

    // ── PatternLearner (agentdb_pattern-store / agentdb_pattern-search) ─
    async fn store_pattern(&self, pattern: &str, pattern_type: &str, confidence: f32)
        -> Result<(), RufloError>;
    async fn search_patterns(&self, query: &str, top_k: usize, min_confidence: f32)
        -> Result<Vec<PatternEntry>, RufloError>;

    // ── MavlinkDefence (aidefence_is_safe / aidefence_scan) ──────────
    async fn mavlink_is_safe(&self, message_repr: &str) -> Result<bool, RufloError>;
    async fn mavlink_scan(&self, message_repr: &str) -> Result<MavlinkScanResult, RufloError>;

    // ── IntelligenceHooks (hooks_intelligence_trajectory-*) ──────────
    async fn trajectory_start(&self, task: &str, agent: &str)
        -> Result<String, RufloError>;   // returns trajectoryId
    async fn trajectory_step(&self, trajectory_id: &str, action: &str, result: &str, quality: f32)
        -> Result<(), RufloError>;
    async fn trajectory_end(&self, trajectory_id: &str, success: bool, feedback: Option<&str>)
        -> Result<(), RufloError>;
}

//! In-memory mock RufloBackend for testing — no network, zero latency.
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use super::backend::*;

/// Configurable mock. All writes go to in-memory vecs; searches return stored items.
pub struct MockRufloBackend {
    pub missions:  Arc<Mutex<Vec<(String, String)>>>,       // (key, value)
    pub patterns:  Arc<Mutex<Vec<(String, String, f32)>>>,  // (pattern, type, confidence)
    pub scan_safe: bool,  // set false to simulate a detected threat
    pub traj_ids:  Arc<Mutex<Vec<String>>>,
}

impl Default for MockRufloBackend {
    fn default() -> Self {
        Self {
            missions:  Arc::new(Mutex::new(Vec::new())),
            patterns:  Arc::new(Mutex::new(Vec::new())),
            scan_safe: true,
            traj_ids:  Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl MockRufloBackend {
    pub fn new() -> Self { Self::default() }

    /// Pre-load a past mission for search to return.
    pub fn seed_mission(&self, key: &str, value: &str) {
        self.missions.lock().unwrap().push((key.to_string(), value.to_string()));
    }

    /// Pre-load a pattern for search to return.
    pub fn seed_pattern(&self, pattern: &str, ptype: &str, confidence: f32) {
        self.patterns.lock().unwrap().push((pattern.to_string(), ptype.to_string(), confidence));
    }

    /// Configure the scanner to reject the next message.
    pub fn reject_next(self) -> Self { Self { scan_safe: false, ..self } }
}

#[async_trait]
impl RufloBackend for MockRufloBackend {
    async fn store_mission(&self, key: &str, value: &str, _ns: &str) -> Result<(), RufloError> {
        self.missions.lock().unwrap().push((key.to_string(), value.to_string()));
        Ok(())
    }

    async fn search_missions(&self, query: &str, limit: usize, _ns: &str)
        -> Result<Vec<MissionMemoryEntry>, RufloError>
    {
        let missions = self.missions.lock().unwrap();
        Ok(missions.iter().take(limit).map(|(k, v)| MissionMemoryEntry {
            key: k.clone(),
            value: v.clone(),
            score: if v.contains(query) { 0.9 } else { 0.5 },
        }).collect())
    }

    async fn store_pattern(&self, pattern: &str, ptype: &str, confidence: f32)
        -> Result<(), RufloError>
    {
        self.patterns.lock().unwrap().push((pattern.to_string(), ptype.to_string(), confidence));
        Ok(())
    }

    async fn search_patterns(&self, _query: &str, top_k: usize, min_conf: f32)
        -> Result<Vec<PatternEntry>, RufloError>
    {
        let patterns = self.patterns.lock().unwrap();
        Ok(patterns.iter()
            .filter(|(_, _, c)| *c >= min_conf)
            .take(top_k)
            .map(|(p, t, c)| PatternEntry {
                pattern: p.clone(),
                pattern_type: t.clone(),
                confidence: *c,
                score: *c,
            })
            .collect())
    }

    async fn mavlink_is_safe(&self, _msg: &str) -> Result<bool, RufloError> {
        Ok(self.scan_safe)
    }

    async fn mavlink_scan(&self, _msg: &str) -> Result<MavlinkScanResult, RufloError> {
        Ok(MavlinkScanResult {
            safe: self.scan_safe,
            threats: if self.scan_safe {
                vec![]
            } else {
                vec!["suspicious_coordinates".into()]
            },
        })
    }

    async fn trajectory_start(&self, task: &str, _agent: &str)
        -> Result<String, RufloError>
    {
        let id = format!("mock-traj-{}", task.len()); // deterministic for testing
        self.traj_ids.lock().unwrap().push(id.clone());
        Ok(id)
    }

    async fn trajectory_step(&self, _id: &str, _act: &str, _res: &str, _q: f32)
        -> Result<(), RufloError> { Ok(()) }

    async fn trajectory_end(&self, _id: &str, _ok: bool, _fb: Option<&str>)
        -> Result<(), RufloError> { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_store_and_search_mission() {
        let mock = MockRufloBackend::new();
        mock.store_mission("m1", r#"{"victims":2}"#, "swarm-missions").await.unwrap();
        let results = mock.search_missions("victims", 5, "swarm-missions").await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "m1");
        assert!(results[0].score > 0.5, "keyword match should score high");
    }

    #[tokio::test]
    async fn test_mock_pattern_lifecycle() {
        let mock = MockRufloBackend::new();
        mock.store_pattern("approach from 3 angles when P > 0.7", "sar-trajectory", 0.9).await.unwrap();
        let results = mock.search_patterns("SAR convergence", 5, 0.5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].confidence, 0.9);
    }

    #[tokio::test]
    async fn test_mock_mavlink_defence_safe() {
        let mock = MockRufloBackend::new();
        assert!(mock.mavlink_is_safe(r#"{"drone_id":1,"confidence":0.8}"#).await.unwrap());
    }

    #[tokio::test]
    async fn test_mock_mavlink_defence_rejected() {
        let mock = MockRufloBackend { scan_safe: false, ..Default::default() };
        let scan = mock.mavlink_scan("SUSPICIOUS MESSAGE").await.unwrap();
        assert!(!scan.safe);
        assert!(!scan.threats.is_empty());
    }

    #[tokio::test]
    async fn test_mock_trajectory_lifecycle() {
        let mock = MockRufloBackend::new();
        let tid = mock.trajectory_start("SAR 400x400", "swarm-specialist").await.unwrap();
        mock.trajectory_step(&tid, "scan (5,3)", "prob=0.6", 0.7).await.unwrap();
        mock.trajectory_end(&tid, true, Some("victim found")).await.unwrap();
        assert!(!mock.traj_ids.lock().unwrap().is_empty());
    }
}

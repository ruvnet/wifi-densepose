//! FHSS (Frequency Hopping Spread Spectrum) anti-jamming interface.
//!
//! Provides frequency hop sequence generation and cognitive radio-inspired
//! adaptive frequency/power selection for drone swarm communication links.

use serde::{Deserialize, Serialize};

/// FHSS configuration for a swarm communication link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhssConfig {
    /// Hop rate in hops-per-second (typical: 100–200).
    pub hop_rate_hz: f64,
    /// Available frequency channels in MHz.
    pub channels_mhz: Vec<f64>,
    /// Minimum RSSI (dBm) before triggering channel switch.
    pub rssi_threshold_dbm: f32,
    /// Number of consecutive poor-RSSI samples before switching.
    pub jamming_detect_window: usize,
}

impl Default for FhssConfig {
    fn default() -> Self {
        // 900 MHz ISM band: 902–928 MHz, 50 channels at 512 kHz spacing
        let channels: Vec<f64> = (0..50).map(|i| 902.0 + i as f64 * 0.512).collect();
        Self {
            hop_rate_hz: 200.0,
            channels_mhz: channels,
            rssi_threshold_dbm: -85.0,
            jamming_detect_window: 5,
        }
    }
}

/// State of the FHSS radio at one node.
pub struct FhssRadio {
    pub config: FhssConfig,
    /// Current hop sequence position.
    hop_index: usize,
    /// Rolling RSSI history (most recent last).
    rssi_history: Vec<f32>,
    /// Elapsed time since last hop (ms).
    elapsed_ms: f64,
    /// Node ID seed for unique hop sequence (XOR with hop_index for non-collision).
    node_seed: u32,
    /// Number of jammer-evasion channel jumps taken.
    pub evasion_count: u64,
}

impl FhssRadio {
    pub fn new(node_seed: u32, config: FhssConfig) -> Self {
        Self {
            config,
            hop_index: 0,
            rssi_history: Vec::new(),
            elapsed_ms: 0.0,
            node_seed,
            evasion_count: 0,
        }
    }

    /// Returns the current active channel frequency in MHz.
    ///
    /// `FhssConfig` is `Deserialize`, so `channels_mhz` can arrive empty from a
    /// malformed or hostile config. An empty channel list would make `% n`
    /// (n = 0) panic with a divide-by-zero. Guard it and return a benign `0.0`
    /// sentinel instead of crashing the radio task (DoS-resistance).
    pub fn current_channel_mhz(&self) -> f64 {
        let n = self.config.channels_mhz.len();
        if n == 0 {
            return 0.0;
        }
        // XOR node seed into hop index so each node uses a different offset
        let idx = (self.hop_index ^ (self.node_seed as usize)) % n;
        self.config.channels_mhz[idx]
    }

    /// Advance the hop sequence by one step (call at hop_rate_hz).
    pub fn next_hop(&mut self) {
        let n = self.config.channels_mhz.len();
        if n == 0 {
            return; // no channels configured — nothing to hop (avoid `% 0` panic)
        }
        self.hop_index = (self.hop_index + 1) % n;
    }

    /// Update with latest RSSI measurement. Drives jamming detection.
    pub fn observe_rssi(&mut self, rssi_dbm: f32) {
        self.rssi_history.push(rssi_dbm);
        if self.rssi_history.len() > self.config.jamming_detect_window {
            self.rssi_history.remove(0);
        }
    }

    /// Returns true if jamming is detected (all recent RSSI samples below threshold).
    pub fn jamming_detected(&self) -> bool {
        if self.rssi_history.len() < self.config.jamming_detect_window {
            return false;
        }
        self.rssi_history.iter().all(|&r| r < self.config.rssi_threshold_dbm)
    }

    /// Evasive hop: jump ahead by a pseudo-random offset to escape jammer.
    /// Uses a simple LCG seeded by node_seed + evasion_count for determinism.
    pub fn evasive_hop(&mut self) {
        let lcg_a: u64 = 6364136223846793005;
        let lcg_c: u64 = 1442695040888963407;
        // Use wrapping arithmetic to avoid overflow in debug builds
        let seed = (self.node_seed as u64)
            .wrapping_mul(lcg_a)
            .wrapping_add(self.evasion_count)
            .wrapping_add(lcg_c);
        let len = self.config.channels_mhz.len();
        if len == 0 {
            return; // no channels configured — avoid `% 0` panic
        }
        let n = len as u64;
        let offset = (seed % n / 4 + 3) as usize;
        self.hop_index = (self.hop_index + offset) % len;
        self.evasion_count += 1;
        self.rssi_history.clear();
    }

    /// Tick the radio by dt_ms milliseconds. Handles automatic hopping.
    ///
    /// Multiple hops may fire within a single tick if dt_ms > hop_interval_ms.
    pub fn tick(&mut self, dt_ms: f64) {
        self.elapsed_ms += dt_ms;
        let hop_interval_ms = 1000.0 / self.config.hop_rate_hz;
        while self.elapsed_ms >= hop_interval_ms {
            self.elapsed_ms -= hop_interval_ms;
            self.next_hop();
        }
        if self.jamming_detected() {
            self.evasive_hop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_different_nodes_different_channels() {
        let cfg = FhssConfig::default();
        let r0 = FhssRadio::new(0, cfg.clone());
        let r1 = FhssRadio::new(7, cfg);
        // Nodes with different seeds should use different channels at hop 0
        assert_ne!(r0.current_channel_mhz(), r1.current_channel_mhz(),
            "different nodes should use different initial channels");
    }

    #[test]
    fn test_jamming_detection() {
        let cfg = FhssConfig { jamming_detect_window: 3, rssi_threshold_dbm: -85.0, ..Default::default() };
        let mut radio = FhssRadio::new(0, cfg);
        // Feed 3 below-threshold RSSI values
        radio.observe_rssi(-90.0);
        radio.observe_rssi(-92.0);
        assert!(!radio.jamming_detected(), "need full window");
        radio.observe_rssi(-91.0);
        assert!(radio.jamming_detected());
    }

    #[test]
    fn test_evasive_hop_changes_channel() {
        let cfg = FhssConfig::default();
        let mut radio = FhssRadio::new(42, cfg);
        let before = radio.current_channel_mhz();
        radio.evasive_hop();
        let after = radio.current_channel_mhz();
        assert_ne!(before, after, "evasive hop should change channel");
    }

    #[test]
    fn test_tick_advances_hop() {
        let cfg = FhssConfig { hop_rate_hz: 1000.0, ..Default::default() }; // 1 hop/ms
        let mut radio = FhssRadio::new(0, cfg);
        let initial_idx = radio.hop_index;
        radio.tick(2.0); // 2 ms = 2 hops
        assert_eq!(radio.hop_index, (initial_idx + 2) % 50);
    }

    /// Security/DoS: an empty `channels_mhz` (deserialized from a malformed or
    /// hostile config) must not panic with a `% 0` divide-by-zero. Fails on old
    /// code, where `next_hop`/`current_channel_mhz`/`evasive_hop`/`tick` all do
    /// modulo / index by `channels_mhz.len()`.
    #[test]
    fn test_empty_channels_does_not_panic() {
        let cfg = FhssConfig { channels_mhz: vec![], jamming_detect_window: 1, ..Default::default() };
        let mut radio = FhssRadio::new(7, cfg);
        // None of these may panic.
        let _ = radio.current_channel_mhz();
        radio.next_hop();
        radio.observe_rssi(-99.0); // window=1 → jamming_detected() true → evasive_hop()
        radio.tick(100.0);
        radio.evasive_hop();
        assert_eq!(radio.current_channel_mhz(), 0.0, "empty channel list returns sentinel");
    }

    #[test]
    fn test_channel_in_valid_range() {
        let cfg = FhssConfig::default();
        let radio = FhssRadio::new(99, cfg.clone());
        let ch = radio.current_channel_mhz();
        assert!(ch >= 902.0 && ch <= 928.0, "channel {} out of ISM band", ch);
    }
}

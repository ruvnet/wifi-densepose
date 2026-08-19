//! Bridge between sensing-server per-node state and the signal crate's
//! `MultistaticFuser` for attention-weighted CSI fusion across ESP32 nodes.
//!
//! This module converts the server's `NodeState` (f64 amplitude history) into
//! `MultiBandCsiFrame`s that the multistatic fusion pipeline expects, then
//! drives `MultistaticFuser::fuse` with a graceful fallback when fusion fails
//! (e.g. insufficient nodes or timestamp spread).

use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use wifi_densepose_signal::hardware_norm::{CanonicalCsiFrame, HardwareNormalizer, HardwareType};
use wifi_densepose_signal::ruvsense::multiband::MultiBandCsiFrame;
use wifi_densepose_signal::ruvsense::multistatic::{FusedSensingFrame, MultistaticFuser};

use super::NodeState;

/// Maximum age for a node frame to be considered active (10 seconds).
const STALE_THRESHOLD: Duration = Duration::from_secs(10);

/// Default WiFi channel frequency (MHz) used for single-channel frames.
const DEFAULT_FREQ_MHZ: u32 = 2437; // Channel 6

/// Monotonic reference point for timestamp generation. All node timestamps
/// are relative to this instant, avoiding wall-clock/monotonic mixing issues.
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Shared length-only canonicalizer (issue #1170). The default 56-tone grid
/// matches what `MultistaticFuser` (ADR-154) expects. Stateless and immutable,
/// so a single process-wide instance is safe to share across nodes.
static NORMALIZER: LazyLock<HardwareNormalizer> = LazyLock::new(HardwareNormalizer::new);

/// Convert a single `NodeState` into a `MultiBandCsiFrame` suitable for
/// multistatic fusion.
///
/// Returns `None` when the node has no frame history or no recorded
/// `last_frame_time`.
pub fn node_frame_from_state(node_id: u8, ns: &NodeState) -> Option<MultiBandCsiFrame> {
    let last_time = ns.last_frame_time.as_ref()?;
    let latest = ns.frame_history.back()?;
    if latest.is_empty() {
        return None;
    }

    // Issue #1170: resample the raw amplitude onto the canonical 56-tone grid
    // BEFORE fusion. ESP32 nodes in mixed HT20/HT40 capture modes report
    // different subcarrier counts (64 / 128 / 192); feeding those raw into
    // `MultistaticFuser::fuse` tripped `DimensionMismatch` on every cycle and
    // silently disabled real multistatic fusion. Length-only canonicalization
    // (no z-score) keeps the amplitude scale the person-score relies on.
    let canonical_amp = NORMALIZER.resample_to_canonical(latest);
    let amplitude: Vec<f32> = canonical_amp.iter().map(|&v| v as f32).collect();
    let n_sub = amplitude.len();
    let phase = vec![0.0_f32; n_sub];

    // Monotonic timestamp: microseconds since a shared process-local epoch.
    // All nodes use the same reference so the fuser's guard_interval_us check
    // compares apples to apples. No wall-clock mixing (immune to NTP jumps).
    let timestamp_us = last_time.duration_since(*EPOCH).as_micros() as u64;

    let canonical = CanonicalCsiFrame {
        amplitude,
        phase,
        hardware_type: HardwareType::Esp32S3,
    };

    Some(MultiBandCsiFrame {
        node_id,
        timestamp_us,
        channel_frames: vec![canonical],
        frequencies_mhz: vec![DEFAULT_FREQ_MHZ],
        coherence: 1.0, // single-channel, perfect self-coherence
    })
}

/// Collect `MultiBandCsiFrame`s from all active nodes.
///
/// A node is considered active if its `last_frame_time` is within
/// [`STALE_THRESHOLD`] of `now`.
pub fn node_frames_from_states(node_states: &HashMap<u8, NodeState>) -> Vec<MultiBandCsiFrame> {
    let now = Instant::now();
    let mut frames = Vec::with_capacity(node_states.len());

    for (&node_id, ns) in node_states {
        // Skip stale nodes
        if let Some(ref t) = ns.last_frame_time {
            if now.duration_since(*t) > STALE_THRESHOLD {
                continue;
            }
        } else {
            continue;
        }

        if let Some(frame) = node_frame_from_state(node_id, ns) {
            frames.push(frame);
        }
    }

    frames
}

/// Attempt multistatic fusion; fall back to max per-node person count on failure.
///
/// Returns `(fused_frame, fallback_person_count)`. When fusion succeeds,
/// `fallback_person_count` is `None` — the caller must compute count from
/// the fused amplitudes. On failure, returns the maximum per-node count
/// (not the sum, to avoid double-counting overlapping coverage).
pub fn fuse_or_fallback(
    fuser: &MultistaticFuser,
    node_states: &HashMap<u8, NodeState>,
    dedup_factor: f64,
) -> (Option<FusedSensingFrame>, Option<usize>) {
    let frames = node_frames_from_states(node_states);
    if frames.is_empty() {
        return (None, Some(0));
    }

    match fuser.fuse(&frames) {
        Ok(fused) => {
            // Caller must compute person count from fused amplitudes.
            (Some(fused), None)
        }
        Err(e) => {
            tracing::debug!("Multistatic fusion failed ({e}), using per-node sum/dedup fallback");
            // Sum per-node counts then divide by dedup_factor (assumed average
            // visibility per body across nodes).  ADR-044 §5.1.
            // dedup_factor is runtime-configurable; default 3.0.
            let total: usize = node_states
                .values()
                .filter(|ns| {
                    ns.last_frame_time
                        .map(|t| t.elapsed() <= STALE_THRESHOLD)
                        .unwrap_or(false)
                })
                .map(|ns| ns.prev_person_count)
                .sum();
            let estimated = ((total as f64) / dedup_factor).ceil() as usize;
            (None, Some(estimated))
        }
    }
}

/// Compute a person-presence score from fused amplitude data.
///
/// Uses the squared coefficient of variation (variance / mean^2) as a
/// lightweight proxy for body-induced CSI perturbation. A flat amplitude
/// vector (no person) yields a score near zero; a vector with high variance
/// relative to its mean (person moving) yields a score approaching 1.0.
pub fn compute_person_score_from_amplitudes(amplitudes: &[f32]) -> f64 {
    if amplitudes.is_empty() {
        return 0.0;
    }

    let n = amplitudes.len() as f64;
    let sum: f64 = amplitudes.iter().map(|&a| a as f64).sum();
    let mean = sum / n;

    let variance: f64 = amplitudes
        .iter()
        .map(|&a| {
            let diff = (a as f64) - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;

    let score = variance / (mean * mean + 1e-10);
    score.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Helper: build a minimal NodeState for testing. Uses `NodeState::new()`
    /// then mutates the `pub(crate)` fields the bridge needs.
    fn make_node_state(
        frame_history: VecDeque<Vec<f64>>,
        last_frame_time: Option<Instant>,
        prev_person_count: usize,
    ) -> NodeState {
        let mut ns = NodeState::new();
        ns.frame_history = frame_history;
        ns.last_frame_time = last_frame_time;
        ns.prev_person_count = prev_person_count;
        ns
    }

    #[test]
    fn test_node_frame_from_empty_state() {
        let ns = make_node_state(VecDeque::new(), Some(Instant::now()), 0);
        assert!(node_frame_from_state(1, &ns).is_none());
    }

    #[test]
    fn test_node_frame_from_state_no_time() {
        let mut history = VecDeque::new();
        history.push_back(vec![1.0, 2.0, 3.0]);
        let ns = make_node_state(history, None, 0);
        assert!(node_frame_from_state(1, &ns).is_none());
    }

    #[test]
    fn test_node_frame_conversion() {
        let mut history = VecDeque::new();
        history.push_back(vec![10.0, 20.0, 30.5]);
        let ns = make_node_state(history, Some(Instant::now()), 0);

        let frame = node_frame_from_state(42, &ns).expect("should produce a frame");
        assert_eq!(frame.node_id, 42);
        assert_eq!(frame.channel_frames.len(), 1);

        let ch = &frame.channel_frames[0];
        // Issue #1170: amplitude is now resampled onto the canonical 56-tone
        // grid regardless of the raw count.
        assert_eq!(ch.amplitude.len(), 56);
        // resample_cubic preserves the endpoints (no z-scoring), so the scale
        // the person-score relies on is intact.
        assert!((ch.amplitude[0] - 10.0_f32).abs() < 1e-3);
        assert!((ch.amplitude[55] - 30.5_f32).abs() < 1e-3);
        // Phase should be all zeros
        assert!(ch.phase.iter().all(|&p| p == 0.0));
        assert_eq!(ch.hardware_type, HardwareType::Esp32S3);
    }

    #[test]
    fn heterogeneous_node_counts_canonicalize_and_fuse() {
        // Issue #1170 regression: a mixed mesh with HT20 (64-bin) and HT40
        // (192-bin) nodes must canonicalize to a uniform 56 tones and fuse,
        // instead of tripping DimensionMismatch on every cycle.
        let mut states: HashMap<u8, NodeState> = HashMap::new();

        let mut h64 = VecDeque::new();
        h64.push_back((0..64).map(|i| 1.0 + 0.1 * i as f64).collect::<Vec<f64>>());
        states.insert(1, make_node_state(h64, Some(Instant::now()), 1));

        let mut h192 = VecDeque::new();
        h192.push_back((0..192).map(|i| 2.0 + 0.05 * i as f64).collect::<Vec<f64>>());
        states.insert(3, make_node_state(h192, Some(Instant::now()), 1));

        let frames = node_frames_from_states(&states);
        assert_eq!(frames.len(), 2, "both nodes should produce frames");
        for f in &frames {
            assert_eq!(
                f.channel_frames[0].amplitude.len(),
                56,
                "every node must present the canonical 56-tone dimension"
            );
        }

        // The fuser must now accept the cycle (no DimensionMismatch).
        let fuser = MultistaticFuser::new();
        let result = fuser.fuse(&frames);
        assert!(
            result.is_ok(),
            "heterogeneous mesh should fuse after canonicalization, got {result:?}"
        );

        // And the higher-level fallback path returns the fused frame, not the
        // sum/dedup fallback.
        let (fused, fallback) = fuse_or_fallback(&fuser, &states, 3.0);
        assert!(fused.is_some(), "fusion should succeed");
        assert!(fallback.is_none(), "no fallback when fusion succeeds");
    }

    #[test]
    fn test_stale_node_excluded() {
        let mut states: HashMap<u8, NodeState> = HashMap::new();

        // Active node: frame just received
        let mut active_history = VecDeque::new();
        active_history.push_back(vec![1.0, 2.0]);
        states.insert(1, make_node_state(active_history, Some(Instant::now()), 1));

        // Stale node: frame 20 seconds ago
        let mut stale_history = VecDeque::new();
        stale_history.push_back(vec![3.0, 4.0]);
        let stale_time = Instant::now() - Duration::from_secs(20);
        states.insert(2, make_node_state(stale_history, Some(stale_time), 1));

        let frames = node_frames_from_states(&states);
        assert_eq!(frames.len(), 1, "stale node should be excluded");
        assert_eq!(frames[0].node_id, 1);
    }

    #[test]
    fn test_compute_person_score_empty() {
        assert!((compute_person_score_from_amplitudes(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_person_score_flat() {
        // Constant amplitude => variance = 0 => score ~ 0
        let flat = vec![5.0_f32; 64];
        let score = compute_person_score_from_amplitudes(&flat);
        assert!(
            score < 0.001,
            "flat signal should have near-zero score, got {score}"
        );
    }

    #[test]
    fn test_compute_person_score_varied() {
        // High variance relative to mean should produce a positive score
        let varied: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { 10.0 })
            .collect();
        let score = compute_person_score_from_amplitudes(&varied);
        assert!(
            score > 0.1,
            "varied signal should have positive score, got {score}"
        );
        assert!(score <= 1.0, "score should be clamped to 1.0, got {score}");
    }

    #[test]
    fn test_compute_person_score_clamped() {
        // Near-zero mean with non-zero variance => would blow up without clamp
        let vals = vec![0.0_f32, 0.0, 0.0, 0.001];
        let score = compute_person_score_from_amplitudes(&vals);
        assert!(score <= 1.0, "score must be clamped to 1.0");
    }

    #[test]
    fn test_fuse_or_fallback_empty() {
        let fuser = MultistaticFuser::new();
        let states: HashMap<u8, NodeState> = HashMap::new();
        let (fused, count) = fuse_or_fallback(&fuser, &states, 3.0);
        assert!(fused.is_none());
        assert_eq!(count, Some(0));
    }
}

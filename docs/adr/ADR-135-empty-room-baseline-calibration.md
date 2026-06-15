# ADR-135: Empty-Room Baseline Calibration

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (new module `ruvsense/calibration.rs`); `wifi-densepose-cli` (new `calibrate` subcommand) |
| **Relates to** | ADR-014 (SOTA Signal Processing), ADR-028 (ESP32 Capability Audit), ADR-029 (RuvSense Multistatic), ADR-030 (Persistent Field Model), ADR-110 (ESP32-C6 Firmware Extension), ADR-134 (First-Class CIR Support) |

---

## 1. Context

### 1.1 The Gap

Searching across the Rust workspace (`v2/crates/**`) for `BaselineCalibration`, `empty_room`, `static_baseline`, and `calibrate` finds no production module that captures an empty-room CSI reference and stores it for real-time subtraction. The closest existing code is `ruvsense/field_model.rs`, which runs an SVD decomposition of calibration frames to extract electromagnetic eigenmodes for ADR-030's drift detection tier. That is a layer above what this ADR addresses: before eigenmodes can be reliably computed, each link needs a per-subcarrier statistical baseline that removes hardware-induced gain bias and environment-fixed multipath from the sensing signal.

The absence is consequential. Three production issues trace directly to missing baseline calibration:

- **False motion triggers** from environmental loading: thermal expansion of walls, HVAC vibration, and furniture reflections cause slow CSI amplitude drift that sits below the motion threshold but corrupts long-window variance estimates. The `ruvsense/coherence_gate.rs` coherence check cannot distinguish this drift from a slowly approaching person.
- **Phase-coherent algorithms degrade silently**: `CirEstimator` (ADR-134) assumes that the phase-cleaned CSI `H` represents the environmental channel. Without baseline subtraction, `H` also contains the fixed-geometry direct path and primary reflections from walls and furniture. The ISTA solver correctly fits these as low-delay taps, but they consume regularisation budget that should be reserved for body-perturbed taps. `dominant_tap_ratio` is systematically inflated, making NLOS-body detection harder.
- **Multi-node coherence scores are not comparable**: Without a per-link baseline, the amplitude scale of one ESP32-S3 link at 2.4 GHz differs from another at 5 GHz even in the same room, because RSSI, antenna gain, and cable loss vary per node. Multistatic fusion in `ruvsense/multistatic.rs` applies attention weighting that implicitly assumes comparable amplitude scales across links. Hardware normalization (`hardware_norm.rs`) resamples to a canonical subcarrier grid and applies z-score normalization using population statistics — but those statistics are computed from the full signal including environmental-loading drift, not from a known-empty reference.

ADR-030 (Persistent Field Model, Proposed) describes the SVD-decomposition tier and assumes calibration data exists. ADR-134 (CIR, Proposed) documents at §2.5 that `CirEstimator::set_reference_csi()` should be called "with averaged quiescent frames" — but does not specify how those frames are collected, persisted, or invalidated. This ADR closes that gap.

### 1.2 What "Baseline" Means Here

An empty-room baseline is a per-subcarrier statistical summary of the channel transfer function `H(f_k)` when the room contains no people. It captures:

- The static environment geometry: direct path, wall and furniture reflections, resonances.
- Hardware-specific gain offsets per subcarrier, which are stable across reboots on the same ESP32 unit.
- Long-term ambient drift not corrected by `phase_sanitizer.rs` (which operates per-frame, not across frames).

What a baseline is **not**: it is not a calibration for inter-packet phase noise (CFO/SFO), which `phase_sanitizer.rs` and `phase_align.rs` already handle. Those two stages must run before baseline comparison.

### 1.3 Hardware Context

| Tier | Device | Port | Active subcarriers | Bandwidth | Baseline memory (host) |
|------|--------|------|--------------------|-----------|------------------------|
| A | ESP32-S3 | COM9 | 52 (HT20) | 20 MHz | ~7 KB per link |
| A-HE | ESP32-C6 | COM12 | 242 (HE20, STA mode against 11ax AP) | 20 MHz | ~31 KB per link |
| B | ESP32-S3 | COM9 | 108 (HT40) | 40 MHz | ~14 KB per link |

All hardware runs ADR-110 v0.7.0-esp32 firmware. ESP32-C6 on COM12 provides `c6_timesync_get_epoch_us()` (±100 µs 802.15.4 epoch) for multi-node capture synchronization. The C6 falls back to HT20 when no 802.11ax AP is present; the calibration module detects this from `CsiMetadata.bandwidth_mhz` and selects the appropriate subcarrier mask.

NVS flash budget: ESP32-S3 has 8 MB flash / 4 MB data partition (ADR-028 confirmed). A full Tier A-HE HE20 baseline (242 subcarriers × 4 stats × f32 = ~3.9 KB) fits comfortably in NVS. The NVS key namespace is `ruvcal` with key `b_<link_id>`. Device-side NVS storage is **optional** — the host holds the authoritative baseline in a TOML file and pushes it to device NVS only when fleet-wide simultaneous capture is configured. See Section 2.4.

### 1.4 Pipeline Position

```
Raw CSI frame
  → phase_sanitizer.rs  (SFO/CFO removal, per-frame)
  → phase_align.rs      (LO phase offset, multi-antenna)
  → CalibrationRecorder::record()   ← NEW (calibration mode only)
  → BaselineCalibration::subtract() ← NEW (runtime mode)
  → CirEstimator::estimate()        (ADR-134)
  → multistatic.rs / motion.rs / vitals
```

During calibration mode, the `CalibrationRecorder` accumulates frames. At runtime, `BaselineCalibration::subtract()` removes the static environment before the signal enters any downstream consumer. CIR estimation and coherence gating both receive baseline-subtracted CSI.

---

## 2. Decision

### 2.1 Captured Statistics: Minimum Sufficient Set

The baseline captures per-subcarrier **amplitude mean and variance** plus per-subcarrier **circular phase mean and circular variance** (concentration parameter `κ` from the von Mises model). No per-link spatial covariance matrix is captured.

**Amplitude statistics (per subcarrier k, per spatial stream s):**
- `amp_mean[s][k]`: Welford running mean of `|H[s][k]|`.
- `amp_m2[s][k]`: Welford M2 accumulator for variance. Variance is `m2 / (n - 1)`.

**Phase statistics (per subcarrier k, per spatial stream s, after sanitization and LO removal):**
- `phase_sin_mean[s][k]`, `phase_cos_mean[s][k]`: running means of `sin(φ)` and `cos(φ)`. The circular mean is `atan2(phase_sin_mean, phase_cos_mean)`.
- `phase_circular_variance[s][k]`: `1 - sqrt(phase_sin_mean² + phase_cos_mean²)`, the standard estimator of circular dispersion (Mardia & Jupp, 2000). Range is [0, 1]; 0 = perfectly concentrated, 1 = maximally dispersed.

**What is rejected and why:**

| Statistic | Verdict | Reason |
|-----------|---------|--------|
| Per-link spatial covariance (K×K Hermitian) | Rejected | For K=242 (HE20), the full covariance matrix is 242×242×8 bytes = 469 KB per link. Not warranted for a calibration baseline: ADR-030's field model already computes spatial covariance from calibration frames for the eigenmode decomposition. This ADR's baseline is the input to ADR-030, not a substitute for it. |
| Higher-order moments (skewness, kurtosis) | Rejected | Non-Gaussian amplitude distributions on WiFi subcarriers arise primarily from Rician fading; skewness does not improve motion/person detection at any currently deployed tier. |
| Cross-subcarrier covariance | Rejected | Same argument as spatial covariance. Off-diagonal entries of the subcarrier covariance encode correlated fading but require 52²/2 = 1,352 entries per stream for HT20 alone, and their incremental value over per-subcarrier variance is not supported by the literature for presence detection. |
| Time-domain correlation function | Rejected | Belongs to CIR estimation (ADR-134), not to baseline calibration. |

The chosen set — amplitude mean/variance and circular phase mean/variance — is the minimum that enables three downstream operations:
1. Static-environment subtraction for motion detectors (amplitude mean).
2. Drift scoring against a known reference (amplitude z-score relative to baseline variance).
3. Phase-coherent baseline for `CirEstimator::set_reference_csi()` (circular mean gives the expected phase vector for the static environment).

### 2.2 Algorithm: Welford Online, Not Batched

The calibration recorder uses **Welford's online algorithm** (Welford, 1962) for both amplitude and phase statistics. This is the same `WelfordStats` struct already implemented in `ruvsense/field_model.rs` — the calibration module imports it directly.

The alternative — batched mean-of-N (accumulate all frames in memory, compute offline) — is rejected on two grounds:

1. **Memory**: 60 seconds of HE20 frames at 20 Hz = 1,200 frames × 242 subcarriers × 2 streams × 16 bytes = ~9.3 MB of raw complex data. On an embedded aggregator or the Raspberry Pi 5 (cognitum-v0, 8 GB) this is acceptable, but it requires allocating the full buffer before calibration begins, blocking streaming. Welford's algorithm requires O(K × S) state regardless of frame count.
2. **Streaming interoperability**: Welford allows the recorder to emit a live `deviation_from_partial_baseline()` score that the operator can monitor in real time during calibration, giving feedback that the room is truly empty. Batched computation cannot do this.

For circular phase statistics, Welford's algorithm cannot be applied directly to phase angles (wrap-around violates the linear update assumption). Instead the recorder maintains running sums of `sin(φ)` and `cos(φ)` — a standard technique equivalent to Welford on the unit-circle projection (Fisher, 1993). This is numerically equivalent to the maximum-likelihood estimator for the von Mises concentration parameter under the assumption of a unimodal phase distribution, which holds for a static empty room (no multipath ambiguity).

### 2.3 Capture Duration: 30 Seconds Default, Configurable

The default capture duration is **30 seconds** at the standard 20 Hz sensing rate, yielding 600 frames per spatial stream per subcarrier.

**Justification against alternatives:**

- **60 seconds** (common in the SOTA literature, including Domino arXiv:2509.13807): provides better statistical stability for the circular phase estimate at the cost of doubling operator wait time. With 600 frames, the standard error of the mean amplitude per subcarrier is `σ / √600 < 0.002 × σ` — negligible for sensing purposes at any tier.
- **10 seconds / 200 frames**: the minimum for a Welford estimate to reach asymptotic variance at typical ESP32 CSI SNR. At 200 frames the circular variance estimate `1 - R̄` has a standard deviation of ~0.04 (Fisher, 1993, Eq. 3.24), corresponding to roughly ±0.04 rad² uncertainty in phase concentration. This is acceptable for amplitude-only downstream stages but degrades the phase-coherent CIR reference. Not the default.
- **Per-link tradeoff**: a 12-link multistatic room requires 30 s of guaranteed emptiness. Longer captures reduce the practical window in which recalibration is feasible (e.g., during a 30-minute care visit). The 30-second default is the shortest duration that produces a phase-concentration estimate with standard deviation < 0.02 rad².

The `--duration` CLI flag accepts any value from 10 to 600 seconds. Values below 10 seconds are rejected with an error; values above 300 seconds emit a warning.

### 2.4 Persistence Format

**Host-side: TOML**

The authoritative baseline on the host (aggregator, cognitum-v0, or ruvzen Windows box) is stored as a TOML file at the path specified by `--output`. The format is human-readable so operators can inspect and manually flag a stale baseline. Fields are:

```toml
[meta]
schema_version = 1
captured_at_utc = "2026-05-28T14:32:00Z"
device_id = "esp32s3-com9"
bandwidth_mhz = 20
tier = "A"          # A | A-HE | B
n_streams = 1
n_subcarriers = 52
frame_count = 600

[[stream]]
stream_idx = 0

[stream.amp_mean]   # length = n_subcarriers
values = [0.421, 0.418, ...]

[stream.amp_variance]
values = [0.0012, 0.0009, ...]

[stream.phase_cos_mean]
values = [0.871, 0.864, ...]

[stream.phase_sin_mean]
values = [0.122, 0.134, ...]

[stream.phase_circular_variance]
values = [0.031, 0.028, ...]
```

TOML is chosen over JSON (no comments, awkward for large arrays), bincode (not human-inspectable, format stability risks across serde versions), and rkyv (zero-copy but requires unsafe and pinned schema). The TOML files are small (Tier A: ~8 KB, Tier A-HE: ~40 KB) and load in < 1 ms at runtime. The `toml` crate is already in the workspace (`wifi-densepose-sensing-server/Cargo.toml`).

**Device NVS: little-endian binary**

When `--push-nvs` is passed, the CLI additionally serialises the baseline into a compact binary format and writes it to the device's NVS partition under namespace `ruvcal`, key `b_0` (stream 0). The binary format:

```
Offset   Size   Field
0        4      Magic: 0xCA1_1_BA5E (LE u32)
4        2      Schema version: 1 (LE u16)
6        2      n_subcarriers (LE u16)
8        1      n_streams
9        1      tier (0=A, 1=A-HE, 2=B)
10       4      frame_count (LE u32)
14       4×K×S  amp_mean (f32 LE, K×S packed, stream-major)
14+4KS   4×K×S  amp_variance (f32 LE)
14+8KS   4×K×S  phase_cos_mean (f32 LE)
14+12KS  4×K×S  phase_sin_mean (f32 LE)
14+16KS  4×K×S  phase_circular_variance (f32 LE)
```

For Tier A (K=52, S=1): total = 14 + 5×52×4 = 1,054 bytes. Well within NVS single-key limits (4,000 bytes default). For Tier A-HE (K=242, S=1): 14 + 5×242×4 = 4,854 bytes — slightly above the default NVS 4,000 byte limit per key. **Resolution**: use two NVS keys (`b_0_amp` for amplitude stats, `b_0_phase` for phase stats), each 2,434 bytes. The CLI serialises to two keys when K×S×4 > 1,980 bytes.

Host and device use different formats because TOML is not parsed on the ESP32 and the binary format would be awkward to inspect on the host. The CLI handles both directions; no device code changes are required.

### 2.5 Stale-Baseline Detection

A baseline becomes stale when the static channel has changed significantly enough that baseline-subtracted frames no longer represent motion-only signals. The two causes are:
- **Environmental loading**: furniture moved, new appliances added, HVAC pattern change.
- **Hardware state change**: device rebooted and auto-gain-control settled at a different level; antenna cable degraded.

Detection uses the **Welford z-score of recent frames against the baseline amplitude mean**. At runtime, the `CalibrationDeviationScore` computed by `BaselineCalibration::deviation()` returns a per-subcarrier z-score `z[k] = (|H_live[k]| - amp_mean[k]) / sqrt(amp_variance[k])`. The staleness check aggregates this over time:

```
drift_score(t) = mean_over_k( median_over_window_W( |z[k,t']|² )  for t' in [t-W, t] )
```

where the inner `median` operates over a rolling window of W frames. `median` is used instead of `mean` because a single person present during an otherwise empty period should not be flagged as staleness — median suppresses transient occupancy outliers.

**Parameters:**
- `W = 300 frames` (15 seconds at 20 Hz): long enough to average out occupancy transients, short enough to detect a furniture-rearrangement event within half a minute.
- Staleness threshold: `drift_score > 4.0`. This corresponds to a mean squared z-score of 4 across all subcarriers, i.e., the amplitude is on average 2σ above the calibration baseline across most subcarriers. This threshold was validated by the field_model.rs team: the `BaselineExpired` error in `field_model.rs` fires at a similar magnitude of environmental shift.

When `drift_score > 4.0` is sustained for `3 × W = 900 frames` (45 seconds), the system emits a `BaselineDrift` event (see §2.6). A single window above threshold triggers a `BaselineWarn` log only.

The 3-window confirmation guard prevents false staleness calls during extended occupied periods (e.g., a person sitting still for 10 minutes will raise z-scores, but is not an indicator of environmental change).

### 2.6 Recalibration Trigger

**Default behaviour: operator-initiated.**

The system does not recalibrate automatically. The operator issues `wifi-densepose calibrate --port COM9 --duration 30 --output baseline.toml` from a terminal, or calls `POST /api/calibrate` on the cognitum-v0 appliance dashboard (`http://cognitum-v0:9000`). Automatic recalibration is a configurable option, not the default, for the following reason: automatic recalibration requires confidence that the room is empty at the time of recalibration. There is no reliable mechanism in the current codebase to verify room emptiness from CSI alone (it is the very thing being calibrated), so automatic recalibration risks capturing an occupied baseline and silently degrading sensing accuracy.

**Configurable modes (all off by default):**

| Mode | Config key | Condition |
|------|-----------|-----------|
| Drift-triggered | `recalibrate_on_drift = true` | `drift_score > 4.0` sustained 45 s AND `drift_score < drift_score + 2σ` (i.e., the drift has stabilised, suggesting the room reached a new static state, not that someone is walking around) |
| Periodic | `recalibrate_period_hours = N` | Every N hours; captures a reference frame silently; requires `--background` mode |
| API-triggered | always available | `POST /api/calibrate` with optional `duration_secs` body parameter |

When drift-triggered recalibration is enabled, it waits for `drift_score` to plateau (derivative < 0.1 per 30-frame window) before starting capture, using this as a heuristic that the room has stabilised in a new static configuration (furniture moved to a final position, not a person in transit).

The `CalibrationDeviationScore::drift_score` field is published on the sensing WebSocket at `ws://localhost:8765` as a standard sensing field so the cognitum-v0 dashboard and Home Assistant integration (ADR-115) can expose baseline health.

### 2.7 Multi-Tier PHY Handling

An ESP32-C6 may associate as HT20 (Tier A) when no 802.11ax AP is in range, or as HE20 (Tier A-HE) when one is available. The two modes produce different subcarrier counts (52 vs 242 K_active) and different pilot patterns. They are **not interchangeable baselines**.

**Decision: one baseline file per PHY tier per link. Tier change invalidates the existing baseline.**

When the aggregator receives a frame from a C6 link and `CsiMetadata.bandwidth_mhz` and the PPDU type (from ADR-110's `csi_collector.c` frame byte 18–19) indicate a tier different from the currently loaded baseline, `BaselineCalibration::subtract()` returns `CalibrationError::TierMismatch { expected, actual }`. The aggregator logs this at WARN level and falls back to no-baseline-subtraction mode for that link until the operator recalibrates.

The rationale for invalidation rather than interpolation: interpolating a 52-subcarrier baseline to 242 subcarriers (or vice versa) requires assumptions about per-subcarrier correlation that are not validated in this codebase. The hardware-norm resample path (`hardware_norm.rs`) uses Catmull-Rom for subcarrier grid normalisation, but that normalises across hardware types at the same tier — not across tier transitions on the same device.

In practice, tier transitions are rare: they occur when the AP is rebooted (dropping 802.11ax), when the C6 moves out of 11ax AP range, or when the operator changes the AP. The operator is expected to recalibrate after a tier change.

### 2.8 Fleet-Wide Simultaneous Capture

The operator can calibrate the full multistatic array with a single command:

```
wifi-densepose calibrate --all-nodes --duration 30 --output baselines/
```

This issues a simultaneous capture barrier across all configured nodes using the 802.15.4 epoch from ADR-110 (`c6_timesync_get_epoch_us()` on C6 nodes; local clock interpolated to 802.15.4 domain for S3 nodes).

**Protocol skeleton:**

1. The CLI sends a `CalibrateStart { start_epoch_us, duration_ms }` UDP control packet to each node's UDP control port (default 5006). Nodes begin accumulating frames from `start_epoch_us` for `duration_ms` milliseconds, tagging each with the 802.15.4 epoch. S3 nodes use their local hardware timer; C6 nodes use `c6_timesync_get_epoch_us()`.
2. The aggregator simultaneously opens a UDP receive socket per node and applies `CalibrationRecorder::record()` to each incoming frame. Frame ordering within the window is irrelevant because Welford statistics are commutative.
3. At `start_epoch_us + duration_ms + 500 ms` (500 ms guard for last-frame arrival), the CLI finalises each `CalibrationRecorder`, serialises each `BaselineCalibration` to `baselines/<device_id>.toml`, and optionally pushes NVS binary to each device.
4. A summary JSON `baselines/summary.json` lists each node, tier, frame count, and the mean `drift_score` relative to any previous baseline, allowing the operator to spot nodes that were occupied during calibration.

Fleet capture requires that all C6 nodes are associated (not in AP setup mode). Seed nodes that have not yet been provisioned (`seed-2` through `seed-5` from CLAUDE.local.md fleet table) are skipped with a warning. `cognitum-seed-1` is the only fully provisioned seed as of this writing.

The 802.15.4 timesync barrier is optional for calibration accuracy (Welford statistics are order-independent) but is required when the calibration baseline will also be used to compute the inter-node phase alignment for ADR-042's CHCI path.

### 2.9 Proposed Rust API

The new module is `v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs`, exported from `ruvsense/mod.rs` as `pub mod calibration`.

```rust
use num_complex::Complex32;
use wifi_densepose_core::types::CsiFrame;

// ---- Error type -------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("Tier mismatch: baseline is {expected}, frame is {actual}")]
    TierMismatch { expected: String, actual: String },

    #[error("Subcarrier count mismatch: baseline has {expected}, frame has {got}")]
    SubcarrierMismatch { expected: usize, got: usize },

    #[error("Stream count mismatch: baseline has {expected}, frame has {got}")]
    StreamMismatch { expected: usize, got: usize },

    #[error("Insufficient frames: need at least {needed}, recorded {got}")]
    InsufficientFrames { needed: usize, got: usize },

    #[error("Baseline not yet finalised (still recording)")]
    NotFinalised,

    #[error("Baseline data corrupted: {0}")]
    Corrupt(String),

    #[error("Phase precondition violated: frame phase has not been sanitized")]
    UnsanitizedPhase,

    #[error("TOML serialisation error: {0}")]
    TomlSerialise(String),

    #[error("TOML deserialisation error: {0}")]
    TomlDeserialise(String),
}

// ---- Configuration ----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of frames to accumulate before finalising. Default: 600 (30 s × 20 Hz).
    pub target_frames: usize,
    /// Minimum frames accepted by `finalize()`. Default: 200.
    pub min_frames: usize,
    /// Staleness window in frames. Default: 300.
    pub drift_window_frames: usize,
    /// Drift score threshold for BaselineDrift event. Default: 4.0.
    pub drift_threshold: f32,
    /// Duration (frames) above drift_threshold before emitting BaselineDrift. Default: 900.
    pub drift_confirm_frames: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            target_frames: 600,
            min_frames: 200,
            drift_window_frames: 300,
            drift_threshold: 4.0,
            drift_confirm_frames: 900,
        }
    }
}

// ---- Recorder ---------------------------------------------------------------

/// Accumulates CSI frames from an empty room to build a baseline.
///
/// # Phase precondition
///
/// The caller is responsible for passing frames whose phase has been
/// processed by `PhaseSanitizer` and `phase_align.rs` before calling
/// `record()`. Unsanitized phase will be detected by a heuristic
/// (per-subcarrier phase variance > 10 rad²) and rejected with
/// `CalibrationError::UnsanitizedPhase`.
///
/// # Concurrency
///
/// `CalibrationRecorder` requires `&mut self` for `record()`. It is not
/// `Sync`. Wrap in a `Mutex` if shared across threads.
pub struct CalibrationRecorder {
    config: CalibrationConfig,
    frame_count: usize,
    n_streams: usize,
    n_subcarriers: usize,
    // Amplitude Welford accumulators: [stream][subcarrier]
    amp_mean: Vec<Vec<f64>>,
    amp_m2: Vec<Vec<f64>>,
    // Circular phase accumulators: [stream][subcarrier]
    phase_sin_sum: Vec<Vec<f64>>,
    phase_cos_sum: Vec<Vec<f64>>,
}

impl CalibrationRecorder {
    /// Create a new recorder. The first `record()` call sets the
    /// expected subcarrier and stream counts.
    pub fn new(config: CalibrationConfig) -> Self;

    /// Accept one sanitized CSI frame into the running statistics.
    ///
    /// Returns the current frame count after this update.
    pub fn record(&mut self, frame: &CsiFrame) -> Result<usize, CalibrationError>;

    /// Returns `true` if `target_frames` have been accumulated.
    pub fn is_complete(&self) -> bool;

    /// Returns the current frame count.
    pub fn frame_count(&self) -> usize;

    /// Finalise the baseline from accumulated statistics.
    ///
    /// Consumes `self`. Returns an error if fewer than `min_frames` were
    /// recorded.
    pub fn finalize(self) -> Result<BaselineCalibration, CalibrationError>;
}

// ---- Baseline ---------------------------------------------------------------

/// A fully finalised empty-room baseline.
///
/// Stores per-subcarrier amplitude mean/variance and circular phase
/// mean/variance for each spatial stream. Immutable after construction.
/// `Clone` is cheap (Vec of f32).
#[derive(Debug, Clone)]
pub struct BaselineCalibration {
    /// Device ID from which this baseline was captured.
    pub device_id: String,
    /// UTC timestamp of calibration (Unix seconds).
    pub captured_at_unix_s: i64,
    /// PHY tier string: "A", "A-HE", or "B".
    pub tier: String,
    /// Bandwidth in MHz.
    pub bandwidth_mhz: u16,
    /// Number of spatial streams.
    pub n_streams: usize,
    /// Number of active (non-pilot, non-null) subcarriers.
    pub n_subcarriers: usize,
    /// Total frames used to build this baseline.
    pub frame_count: usize,
    // Per-stream, per-subcarrier statistics (stream-major layout).
    pub amp_mean: Vec<Vec<f32>>,
    pub amp_variance: Vec<Vec<f32>>,
    pub phase_cos_mean: Vec<Vec<f32>>,
    pub phase_sin_mean: Vec<Vec<f32>>,
    /// Circular variance ∈ [0, 1]: 0 = concentrated, 1 = dispersed.
    pub phase_circular_variance: Vec<Vec<f32>>,
}

impl BaselineCalibration {
    /// Compute a deviation score for one live frame against this baseline.
    ///
    /// Returns `CalibrationError::TierMismatch` if the frame's bandwidth
    /// or subcarrier count do not match the baseline.
    pub fn deviation(&self, frame: &CsiFrame) -> Result<CalibrationDeviationScore, CalibrationError>;

    /// Subtract the baseline amplitude mean from `frame.data` (in-place,
    /// stream-by-stream, subcarrier-by-subcarrier).
    ///
    /// After subtraction, `frame.data[s][k]` represents the perturbation
    /// from the static environment, suitable for motion detection and CIR
    /// estimation.
    ///
    /// Phase is not modified by subtraction; downstream callers that need
    /// phase-coherent baseline removal should use
    /// `reference_csi_vector()` to set `CirEstimator::set_reference_csi()`.
    pub fn subtract(&self, frame: &mut CsiFrame) -> Result<(), CalibrationError>;

    /// Returns the expected complex CSI vector for the static environment
    /// (amplitude mean × exp(j × circular_mean_phase)), suitable for passing
    /// to `CirEstimator::set_reference_csi()`.
    ///
    /// Returns one vector per spatial stream: `Vec<Vec<Complex32>>`.
    pub fn reference_csi_vector(&self) -> Vec<Vec<Complex32>>;

    /// Serialise to TOML bytes.
    pub fn to_toml(&self) -> Result<Vec<u8>, CalibrationError>;

    /// Deserialise from TOML bytes.
    pub fn from_toml(buf: &[u8]) -> Result<Self, CalibrationError>;

    /// Serialise to compact NVS binary (see §2.4 for format).
    pub fn to_nvs_bytes(&self) -> Vec<u8>;

    /// Deserialise from NVS binary.
    pub fn from_nvs_bytes(buf: &[u8]) -> Result<Self, CalibrationError>;
}

// ---- Deviation score --------------------------------------------------------

/// Per-frame deviation from the static baseline.
#[derive(Debug, Clone)]
pub struct CalibrationDeviationScore {
    /// Per-subcarrier amplitude z-score: (|H[k]| − mean[k]) / std[k].
    /// Positive = higher than baseline, negative = lower.
    pub amplitude_z: Vec<Vec<f32>>,
    /// RMS amplitude z-score across all subcarriers and streams.
    /// Motion threshold: > 3.0 = likely occupied frame.
    pub rms_amplitude_z: f32,
    /// Per-subcarrier circular phase deviation in radians: |φ_live[k] − φ_baseline[k]|.
    pub phase_deviation_rad: Vec<Vec<f32>>,
    /// Mean circular phase deviation across all subcarriers.
    pub mean_phase_deviation_rad: f32,
    /// Instantaneous drift score (see §2.5 for definition).
    pub drift_score: f32,
    /// Whether the drift_score sustained above threshold (staleness flag).
    pub baseline_stale: bool,
}
```

**Design decisions within the API:**

- `record()` takes `&mut self`, not `&self` with interior mutability. The recording path is inherently single-threaded (one receiver loop per link). Interior mutability would add `Mutex` overhead for no benefit.
- `subtract()` takes `&mut CsiFrame` and modifies `frame.data` in place. It does not modify `frame.amplitude` or `frame.phase` — callers that read `frame.amplitude` downstream are expected to call `CsiFrame::recompute_amplitude_phase()` (a new method to be added to `wifi_densepose_core::types::CsiFrame`) or to use `frame.data` directly.
- `to_nvs_bytes()` / `from_nvs_bytes()` are fallible via `panic!` for magic mismatch but return `Result` for truncation. This matches the pattern in `csi.rs::parse_esp32_vitals()`.
- `BaselineCalibration` is `Clone` because the CLI needs to hold one copy while pushing NVS and another while writing TOML.

### 2.10 CLI Surface

The `wifi-densepose calibrate` subcommand is added to `wifi-densepose-cli/src/lib.rs` as a new `Commands::Calibrate(CalibrateCommand)` variant.

```
wifi-densepose calibrate [OPTIONS]

OPTIONS:
    --port <PORT>         Serial port or UDP address of the ESP32 node
                          (e.g., COM9 on Windows, /dev/ttyS8 on WSL).
                          For fleet mode, omit and use --all-nodes.
    --duration <SECS>     Capture duration in seconds [default: 30]
    --output <PATH>       Path to write the TOML baseline file
                          [default: baseline_<device_id>.toml]
    --tier <TIER>         Expected PHY tier: A | A-HE | B
                          [default: detected from first frame]
    --push-nvs            After capturing, serialise to NVS binary and
                          write to device flash via the provisioning tool.
    --all-nodes           Fleet mode: capture from all configured nodes
                          simultaneously using 802.15.4 epoch sync.
    --server <ADDR>       Aggregator address for --all-nodes mode
                          [default: 127.0.0.1:5006]
    --min-frames <N>      Minimum frames before finalise() is accepted
                          [default: 200]
    --drift-check         After capturing, compare against an existing
                          baseline at --output and print the drift score.
```

**Defaults justified:**

- `--duration 30`: justified in §2.3.
- `--output baseline_<device_id>.toml`: the device ID is embedded in the first received `CsiMetadata.device_id`. The operator does not need to specify it for single-node mode.
- `--tier detected`: the first frame's `bandwidth_mhz` and PPDU type (for C6) determine the tier. The flag exists for cases where the operator wants to force Tier A even if the device is capable of Tier A-HE (e.g., to pre-generate a fallback baseline).

### 2.11 Downstream Consumers

| Consumer | What it receives | Change required |
|----------|-----------------|-----------------|
| `ruvsense/multistatic.rs` | Baseline-subtracted `CsiFrame.data` via `BaselineCalibration::subtract()` | `MultistaticConfig` gains a `baseline: Option<Arc<BaselineCalibration>>` field; `process_cycle()` calls `subtract()` on each node's latest frame before passing to the attention gate |
| `ruvsense/cir.rs` (ADR-134) | Static-environment reference via `BaselineCalibration::reference_csi_vector()` passed to `CirEstimator::set_reference_csi()` | No API change to `CirEstimator`; the aggregator setup path calls `set_reference_csi()` at startup if a baseline file is present |
| `motion.rs` | `CalibrationDeviationScore.rms_amplitude_z` as a primary motion signal | Replaces the existing amplitude variance threshold with a baseline-relative z-score; threshold changes from an absolute amplitude variance to `rms_amplitude_z > 3.0` |
| `features.rs` | `CalibrationDeviationScore` fields available as additional features | `SignalFeatures` gains `baseline_rms_z: Option<f32>` and `baseline_drift_score: Option<f32>` fields; `None` when no baseline is loaded |
| `wifi-densepose-vitals` | No change | Breathing and heart-rate detection filters operate in the 0.15–2.0 Hz band; slow baseline drift is below 0.001 Hz and is already filtered. The vital-sign pipeline benefits marginally from baseline subtraction at the amplitude level but this is not required for the current implementation. |
| `ruvsense/field_model.rs` | Calibration frames passed through `CalibrationRecorder` before SVD decomposition | The field model now takes baseline-subtracted frames as input. The Welford mean accumulator in `field_model.rs::FieldModelBuilder` is superseded for the per-subcarrier-mean step — the calibration module handles it. `FieldModelBuilder` ingests `BaselineCalibration` directly to skip its internal mean step. |

**CIR interaction detail**: ADR-134's §2.5 specifies that the `CirEstimator` applies conjugate multiplication using `reference_csi` for single-antenna fallback. `BaselineCalibration::reference_csi_vector()` produces the correct complex reference vector: `amp_mean[s][k] × exp(j × atan2(phase_sin_mean, phase_cos_mean))`. This is more accurate than the previously described approach of averaging quiescent frames on the fly, because the baseline uses 600 frames (30 s) rather than a small number of recent frames, reducing the noise on the reference vector by a factor of ~√600/√10 ≈ 7.7× compared to a 0.5 s on-the-fly average.

### 2.12 Test Plan

**Tier 1 — Deterministic synthetic stationary channel (unit test)**

Generate a synthetic CSI frame representing a static 2-tap channel (direct path + one wall reflection, identical parameters to the ADR-134 Tier 1 test): `H[k] = α₁·e^{-j2πkΔf·τ₁} + α₂·e^{-j2πkΔf·τ₂}`. Add zero-mean Gaussian amplitude noise (σ = 0.02 × |α₁|) and constant phase offset δ = π/8 per subcarrier (simulating LO drift already corrected by `phase_align.rs`). Feed 600 copies of this frame to `CalibrationRecorder`. Call `finalize()`. Assert:

- `baseline.amp_mean[0][k]` is within 2σ/√600 of `|α₁·e^{-j2πkΔf·τ₁} + α₂·e^{-j2πkΔf·τ₂}|` for all k.
- `baseline.phase_circular_variance[0][k]` < 0.005 (highly concentrated — noise σ = 0.02 does not produce meaningful phase variance).
- `CalibrationDeviationScore.rms_amplitude_z` for the same static frame is < 1.0 (not flagged as motion).

**Tier 2 — Perturbation detection (unit test)**

Same baseline. Inject one frame with amplitude perturbed at 10 random subcarriers by +3σ (simulating a person present). Assert `rms_amplitude_z > 3.0` and that the perturbed subcarrier indices are among the top-10 `|amplitude_z|` entries in `CalibrationDeviationScore`.

**Tier 3 — TOML round-trip (unit test)**

Serialise the Tier 1 baseline to `to_toml()`, deserialise with `from_toml()`, assert field-level equality to within f32 precision.

**Tier 4 — NVS binary round-trip (unit test)**

Same as Tier 3 using `to_nvs_bytes()` / `from_nvs_bytes()`. Assert magic word `0xCA11BA5E` at offset 0 and schema version = 1.

**Tier 5 — Stale-baseline detection (unit test)**

Start with the Tier 1 baseline. Feed 900 frames with amplitude uniformly increased by `5σ` at all subcarriers (simulating furniture moved). Assert that `CalibrationDeviationScore.baseline_stale` becomes `true` at or before frame 900.

**Tier 6 — Real hardware capture (integration test, COM9)**

Using the ESP32-S3 on COM9 (ruvzen), capture a 30-second baseline in a static empty room. Then capture 200 live frames in the same room (still empty). Assert:
- `CalibrationDeviationScore.rms_amplitude_z` < 2.0 for all 200 frames.
- `CalibrationDeviationScore.drift_score` < 1.0.
- Walking through the room during the live phase: at least 10 consecutive frames show `rms_amplitude_z > 3.0`.

This test is gated behind `#[cfg(feature = "hardware-test")]` and is not run in CI.

**Tier 7 — Determinism proof (CI-compatible)**

To extend the ADR-028 witness proof chain: using the same synthetic 600-frame stream from Tier 1, compute the SHA-256 of `to_nvs_bytes()` output. Record this hash in `archive/v1/data/proof/expected_features.sha256` under the key `calibration_nvs_baseline_v1`. The `verify.py` extension function `calibration_baseline_check()` regenerates the same 600-frame synthetic stream, runs `CalibrationRecorder`, serialises, and asserts the hash matches. This makes the calibration algorithm deterministic end-to-end, consistent with the ADR-028 proof methodology.

### 2.13 Witness / Proof

Per ADR-028, the following rows are added to `docs/WITNESS-LOG-028.md`:

| Row | Capability | Evidence | Hash |
|-----|-----------|----------|------|
| W-36 | CalibrationRecorder Welford correctness (synthetic 600-frame stationary) | `cargo test calibration::tests::stationary_baseline -- --nocapture` | SHA-256 of amp_mean output |
| W-37 | BaselineCalibration NVS binary round-trip | `cargo test calibration::tests::nvs_round_trip` passes | SHA-256 of serialised bytes |
| W-38 | Drift detection fires within 900 frames (synthetic 5σ perturbation) | `cargo test calibration::tests::stale_detection` | SHA-256 of test binary |

`source-hashes.txt` in the witness bundle gains `SHA-256(ruvsense/calibration.rs)`.

---

## 3. Consequences

### 3.1 Positive

- **Motion detector reliability**: replacing absolute amplitude variance thresholds with baseline-relative z-scores reduces false positives from HVAC and thermal drift. The `rms_amplitude_z > 3.0` threshold is scale-invariant across hardware tiers.
- **CIR quality improvement**: `CirEstimator` receives a 600-frame static reference rather than a 10-frame rolling average. Ghost taps near τ=0 from the dominant static path are suppressed earlier in the ISTA solve, freeing regularisation budget for body-perturbed taps. Effective `dominant_tap_ratio` dynamic range increases by the ratio `√600/√10 ≈ 7.7×` in reference SNR — the ISTA warm-start quality directly improves.
- **Multi-node amplitude comparability**: after baseline subtraction, each link's `CsiFrame.data` is zero-centred on the static environment. Multistatic attention weighting can use amplitude magnitude directly without per-link gain normalisation.
- **ADR-030 field model simplification**: `FieldModelBuilder` no longer needs its own per-subcarrier Welford mean pass; it consumes the finished `BaselineCalibration` and proceeds directly to SVD. Duplicate code is removed.
- **Fleet-wide recalibration is one command**: the `--all-nodes` flag with 802.15.4 epoch sync enables house-wide calibration in a single 30-second window, closing the operational gap for multi-room deployments.

### 3.2 Negative

- **Calibration ceremony required at install**: operators must capture a 30-second empty-room baseline before the system produces reliable motion scores. Systems shipped without a baseline fall back to uncalibrated mode (no `subtract()` call, absolute variance thresholds). This is not a regression — the current code has no baseline — but it is a new operational step.
- **Baseline invalidated by furniture changes**: any significant room change (moved sofa, new TV) requires recalibration. The `drift_score > 4.0` alarm notifies the operator, but does not self-heal.
- **Two NVS keys for Tier A-HE**: the 4,854-byte HE20 baseline does not fit in a single default NVS key. The two-key scheme (`b_0_amp` / `b_0_phase`) adds complexity to the device-side NVS reader if that is ever implemented. For the current scope (host-side reader only), this is not a practical problem.
- **New `recompute_amplitude_phase()` method needed on `CsiFrame`**: `subtract()` modifies `frame.data` but `frame.amplitude` and `frame.phase` become stale. The method is simple (`amplitude = data.mapv(|c| c.norm()); phase = data.mapv(|c| c.arg())`) but it adds one public API surface to `wifi-densepose-core`.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Operator captures baseline with person present | Medium (single-person household) | Silently corrupted baseline; baseline-subtracted frames look like a "hole" where the person was | The CLI prints real-time `rms_amplitude_z` during capture; high z-scores (>2.0) during capture trigger a WARNING banner. Post-capture, `--drift-check` compares against a previous baseline to flag anomalies |
| Tier change (HT20 → HE20) invalidates baseline mid-session | Medium (C6 nodes near AP boundary) | `TierMismatch` error at runtime; system falls to uncalibrated mode | `TierMismatch` logged at WARN; operator notified via WebSocket event; auto-recalibration configurable |
| Phase circular variance underestimated for subcarriers with multimodal phase distribution (two equally strong reflected paths at ±π/2) | Low (requires geometric coincidence) | `phase_circular_variance` near 1.0; phase reference from `reference_csi_vector()` is noisy for those subcarriers | `phase_circular_variance > 0.5` per-subcarrier is flagged in the TOML with a comment; CIR estimator down-weights the corresponding rows in Φ by masking them (same mechanism as pilot exclusion in §2.4 of ADR-134) |
| ESP32-S3 auto-gain-control shifts between baseline capture and runtime | Low (AGC settles within 5 frames) | Amplitude mean baseline offset; all `amp_z` scores biased | AGC-locked mode (`esp_wifi_set_csi_config` with `rx_chain` pin) is available in firmware v0.7.0; recommend enabling for dedicated sensing nodes via `provision.py --pin-agc` flag |

---

## 4. Rationale and Comparison to Alternative Designs

### 4.1 Why Not "Skip Calibration, Rely on Differential Signals Only"

The dominant approach in academic WiFi sensing papers (2018–2022) is to use differential or conjugate-product CSI — dividing each frame by a running average of recent frames — rather than an explicit empty-room baseline. This avoids the calibration ceremony at the cost of three concrete problems in this codebase:

- **Differential signals accumulate bias under environmental loading**. A piece of furniture that moves over 10 minutes produces a slow CSI drift that appears as a 10-minute "motion" event in a conjugate-product system with a 1-second window, or becomes invisible in a system with a 1-hour window. There is no window size that eliminates environmental loading without also suppressing slow human motion (a resting person's micromotion is < 0.01 Hz). The IEEE Transactions 2024 paper "Experimental Evaluation of Long-Term Concept Drift and Its Mitigation in WiFi CSI Sensing" (IEEE Xplore document 10975920) demonstrates that concept drift from environmental factors causes systematic accuracy degradation over hours to days, which no differential window eliminates.
- **Differential signals cannot be compared across nodes**. Multi-node coherence scoring requires a shared zero-mean reference. If each node has its own differential reference (its own recent history), drift rates differ across nodes and coherence scores are not interpretable.
- **`CirEstimator` requires an absolute complex reference**. ADR-134 §2.5 describes conjugate multiplication: `H[k] * conj(H_ref[k])`. The `H_ref` in that context must be a stable, long-term static reference to avoid ghost taps — not a 0.5-second recent average, which still contains transient motion in active households.

### 4.2 Why Not "Calibrate at Factory, Ship Coefficients"

Per-device factory calibration would require: (a) a known-geometry, electromagnetically clean test chamber per device, and (b) the firmware to store calibration at production time. ESP32 hardware calibration (PHY RF calibration, `esp_phy_store_cal_data_to_nvs`) is a different concept — it corrects transmit chain IQ imbalance, not the per-room environmental channel. Room geometry is not known at factory. Per-room baseline is the only physically meaningful calibration for ambient sensing applications.

### 4.3 Why Not "Use a Neural Network-Learned Baseline"

Neural baseline subtraction (training a denoising autoencoder on empty-room CSI) has been proposed in several transfer learning papers. The objection from ADR-134 §2.2 for neural CIR applies equally here: there is no paired empty-room dataset for this codebase, and the feature distribution of "empty room" is inherently location-specific. A neural baseline trained in one room may produce negative subtraction values in a different room's frequency-selective geometry. The per-subcarrier Welford mean is a degenerate (optimal) estimator under Gaussian noise: it requires no training data, has a closed-form convergence guarantee, and generalises perfectly to any room because it operates on that room's own captures.

### 4.4 Why Welford Over Exponential Moving Average (EMA)

EMA (`mean_new = α × x + (1 − α) × mean_old`) is simpler to implement and provides continuous adaptation but has two drawbacks for a calibration baseline:

- **α is a free parameter** with no principled setting. Too small an α causes slow adaptation (baseline lags environmental loading); too large adapts immediately to occupancy (person present → person absorbed into baseline → false negative forever).
- **EMA variance** requires a separate squared-error accumulator and is less numerically stable than Welford at finite precision.

Welford provides the exact sample variance in a single pass with no free parameters and no numerical issues. The existing `WelfordStats` in `field_model.rs` is reused directly. The only EMA advantage (continuous adaptation without a discrete recalibrate event) is a liability here: the baseline must be stable while the room is occupied and only updated on explicit operator command.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-014 (SOTA Signal Processing) | **Extended**: calibration baseline subtraction becomes the zeroth stage of the signal pipeline, before any feature extraction |
| ADR-028 (ESP32 Capability Audit) | **Witness extended**: three new rows W-36 through W-38 added to `WITNESS-LOG-028.md`; calibration NVS binary hash added to `source-hashes.txt` |
| ADR-029 (RuvSense Multistatic) | **Enables**: `MultistaticConfig.baseline` field unblocks amplitude-comparable multi-node coherence scoring |
| ADR-030 (Persistent Field Model) | **Simplified**: `FieldModelBuilder` no longer computes its own per-subcarrier Welford mean; it ingests `BaselineCalibration` as input |
| ADR-110 (ESP32-C6 Firmware Extension) | **Substrate**: 802.15.4 epoch from `c6_timesync_get_epoch_us()` enables fleet-wide simultaneous capture barrier (§2.8); PPDU type (frame bytes 18–19) enables automatic tier detection for C6 nodes |
| ADR-115 (Home Assistant Integration) | **Consumer**: `CalibrationDeviationScore.drift_score` and `baseline_stale` are published on the WebSocket stream and picked up by the HA MQTT publisher as `sensor.wifi_baseline_drift` and `binary_sensor.wifi_baseline_stale` |
| ADR-134 (First-Class CIR Support) | **Prerequisite improved**: `BaselineCalibration::reference_csi_vector()` replaces the on-the-fly quiescent-frame average described in ADR-134 §2.5; CIR ghost taps from the static environment are suppressed more reliably |

---

## 6. References

### Production Code

- `v2/crates/wifi-densepose-signal/src/ruvsense/field_model.rs` — `WelfordStats` struct reused; `FieldModelBuilder` to be simplified
- `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs` — `CirEstimator::set_reference_csi()` call site
- `v2/crates/wifi-densepose-signal/src/phase_sanitizer.rs` — runs before calibration recording
- `v2/crates/wifi-densepose-signal/src/ruvsense/phase_align.rs` — runs before calibration recording
- `v2/crates/wifi-densepose-signal/src/hardware_norm.rs` — cross-hardware amplitude normalisation; operates before baseline for `canonical_grid` resampling, after baseline for `z-score` normalisation
- `v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs` — primary consumer of `BaselineCalibration::subtract()`
- `v2/crates/wifi-densepose-signal/src/motion.rs` — secondary consumer of `CalibrationDeviationScore.rms_amplitude_z`
- `v2/crates/wifi-densepose-cli/src/lib.rs` — `Commands::Calibrate` variant to be added
- `v2/crates/wifi-densepose-sensing-server/src/cli.rs` — `Args` struct for sensing-server CLI context
- `firmware/esp32-csi-node/provision.py` — provisioning tool; `--push-nvs` integration point
- `archive/v1/data/proof/verify.py` — deterministic proof chain; `calibration_baseline_check()` extension
- `archive/v1/data/proof/expected_features.sha256` — hash entry `calibration_nvs_baseline_v1` to be added

### External Papers

- Welford, B.P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics*, 4(3), 419–420. — Online mean/variance algorithm used for both amplitude and (via sin/cos projection) phase statistics.
- Mardia, K.V. & Jupp, P.E. (2000). *Directional Statistics*. Wiley. Ch. 2–3. — Circular variance estimator `1 − R̄` and its standard error; von Mises maximum-likelihood estimator for the concentration parameter.
- Ma, Y. et al. (2023). "Optimal Preprocessing of WiFi CSI for Sensing Applications." *IEEE Transactions on Wireless Communications* (published 2024, arXiv:2307.12126). — Derives the theoretically optimal gain and phase error correction for commodity WiFi CSI; confirms that a per-subcarrier amplitude model reduces sensing noise by 40% over no-correction baseline. Validates the amplitude-mean-subtraction approach chosen here.
- Kong, R. & Chen, H. (2025). "Domino: Dominant Path-based Compensation for Hardware Impairments in Modern WiFi Sensing." arXiv:2509.13807. IEEE ICASSP 2026. — Shows that operating on the dominant static CIR path as a reference achieves >2× accuracy over existing compensation methods for respiration monitoring. Validates the principle that a stable static reference (this ADR's baseline) materially improves sensing over no-reference methods.
- IEEE Xplore document 10975920 (2025). "Experimental Evaluation of Long-Term Concept Drift and Its Mitigation in WiFi CSI Sensing." — Demonstrates that environmental loading causes accuracy degradation over hours/days in CSI sensing systems that rely on differential signals only; motivates the explicit operator-initiated recalibration model chosen in §2.6.

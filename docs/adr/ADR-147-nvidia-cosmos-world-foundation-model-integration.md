# ADR-147: Occupancy World Model Integration (OccWorld / RoboOccWorld)

| Field      | Value                                                                 |
|------------|-----------------------------------------------------------------------|
| Status     | Accepted                                                              |
| Date       | 2026-05-29                                                            |
| Deciders   | ruv                                                                   |
| Relates to | ADR-136, ADR-139, ADR-140, ADR-141, ADR-143, ADR-145, ADR-146        |

> Previously titled "NVIDIA Cosmos WFM Integration". Decision revised after hardware
> analysis confirmed RTX 5080 (16 GB VRAM) cannot run Cosmos-Transfer2.5-2B (requires
> 32.54 GB). OccWorld runs in **1.65 GB VRAM** at 375 ms/inference — validated locally.

## 1. Context

RuView's WorldGraph (ADR-139) produces a current-state environmental digital twin; the RF
encoder (ADR-146) predicts present-frame pose/presence/count at ~20 Hz. There is no
future-state prediction — no trajectory priors beyond the Kalman tracker's 5–10 frame
horizon, and no physics-aware validation of SemanticState updates.

Two world-model families were evaluated:

### 1.1 NVIDIA Cosmos (deferred)

Cosmos-Transfer2.5-2B requires **32.54 GB VRAM**. ruvultra has an RTX 5080 with
**15.5 GB VRAM**. Cannot run locally. Deferred to ADR-148 for when H100/A100 access
is available or for offline training data generation only.

### 1.2 OccWorld / RoboOccWorld (this ADR)

| Model | Domain | Input | VRAM (inf) | Status |
|-------|--------|-------|-----------|--------|
| OccWorld (wzzheng/OccWorld, ECCV 2024) | Outdoor AV (nuScenes) | 3D semantic voxel seq | **1.65 GB validated** | Code available, Apache-2.0 |
| RoboOccWorld (arXiv 2505.05512) | Indoor robotics | 3D voxel seq, camera poses | ~2–4 GB estimated | Code not yet released (~Q3 2025) |

Both operate natively in 3D occupancy space — the same representation RuView produces
from WiFi CSI. No video rendering intermediate is needed (unlike Cosmos).

**OccWorld architecture**: VQVAE tokenizer (72.4M params) encodes 3D semantic occupancy
to discrete latent tokens → PlanUAutoRegTransformer predicts future tokens → VQVAE
decoder reconstructs future 3D occupancy. Input: `(B, F, H, W, D)` voxel grid with
integer class labels. Output: predicted occupancy for the next F−1 timesteps.

**RoboOccWorld** (once released): identical paradigm but trained on indoor scenes
(60×60×36 voxels at 0.08 m/voxel, 4.8×4.8×2.88 m space, 12 indoor semantic classes)
— near-perfect match for RuView's room-scale CSI occupancy.

## 2. Decision

**Phase A (now)**: Use OccWorld as the integration scaffold. Run inference from a Python
subprocess. Adapt its dataset loader to accept RuView's custom occupancy format. Remap
semantic classes from nuScenes outdoor (18 classes) to RuView indoor (wall, floor,
person, furniture, free).

**Phase B (Q3–Q4 2025)**: Swap in RoboOccWorld when its code releases. The Rust
`OccupancyWorldModel` interface (§3) is designed for clean backend swap.

**Cosmos**: Deferred. Revisit as an offline training data generator if H100 becomes
available (ADR-148).

## 3. Validated Installation (ruvultra, 2026-05-29)

### 3.1 Environment

| Component | Version | Notes |
|-----------|---------|-------|
| GPU | RTX 5080, 15.5 GB VRAM | sm_120 (Blackwell) |
| PyTorch | 2.10.0+cu128 | ml-env, Python 3.12 |
| CUDA toolkit | 12.8 | /usr/local/cuda-12.8 |
| mmcv | 2.0.1 (Python-only, no CUDA ops) | Built from source with pkg_resources patch |
| mmdet | 3.0.0 | pip install |
| mmdet3d | 1.1.1 | Built from source with --no-deps |
| mmengine | 0.10.7 | pip install via mmcv |
| OccWorld | commit HEAD | ~/projects/OccWorld |

### 3.2 Build Notes

**Issue 1 — sccache compiler wrapping**: System `CC=sccache clang`, `CXX=sccache clang++`
breaks PyTorch CUDA extension builds (injects `clang` as a positional argument to the
build command). **Fix**: `unset CC CXX` before all `pip install`.

**Issue 2 — pkg_resources in mmcv setup.py**: setuptools ≥72 removed the legacy
`pkg_resources` top-level import. **Fix**: patch line 5 of `setup.py` to use
`importlib.metadata` and `packaging.version`.

**Issue 3 — CUDA version mismatch**: host nvcc is CUDA 13.0; PyTorch was built with
12.8. **Fix**: `CUDA_HOME=/usr/local/cuda-12.8` for all builds.

**Issue 4 — mmcv 2.0.1 CUDA ops incompatible with PyTorch 2.10 ATen headers**:
`c10::Type::TypePtr` dereference operator changed. **Fix**: build `MMCV_WITH_OPS=0`
(Python-only build, `mmcv-lite`). OccWorld's inference path does not use mmcv CUDA ops.

**Issue 5 — OccWorld API bug**: `TransVQVAE.forward_inference` calls
`self.transformer(..., hidden=hidden)` but `PlanUAutoRegTransformer.forward(tokens, pose_tokens)`
has no `hidden` kwarg and returns a `(queries, pose_queries)` tuple.
**Fix**: monkey-patch `forward_inference` to pass `pose_tokens=zeros` and unpack the
tuple return. Applied in the Python subprocess at startup.

### 3.3 Validation Results

```
Input:  torch.Size([1, 16, 200, 200, 16])  — 16 frames (15 past + 1 offset)
Output: sem_pred   (1, 15, 200, 200, 16) int64  — predicted future occupancy
        logits     (1, 15, 200, 200, 16, 18) f32 — class logits
        iou_pred   (1, 15, 200, 200, 16) int64  — binary occupancy mask
Inference time: 375 ms
VRAM peak:      1.65 GB
Parameters:     72.4M
```

OccWorld produces **15 predicted future frames** from 15 past frames of 3D semantic
occupancy at 200×200×16 resolution with 18 classes — fully validated on RTX 5080.

## 4. Integration Architecture

### 4.1 Data Flow

```
ESP32-S3 CSI (20 Hz)
    │
    ▼
[ruvsense signal pipeline]  ── ADR-136 frame contracts
    │
    ▼
[RfEncoder / MultiTaskOutput]  ── ADR-146 pose + presence + count
    │  (sub-Hz WorldGraph update rate)
    ▼
[WorldGraph]  ── PersonTrack, ObjectAnchor, SemanticState  ── ADR-139/140
    │
    │  On semantic event (motion, activity change, fall-risk query)
    ▼
[BFLD Privacy Gate]  ── ADR-141: "occworld_inference" action
    │  PRIVATE/HOME → bridge NOT called
    │  MONITORING/AWAY → local inference permitted
    ▼
[wifi-densepose-worldmodel] ── Rust thin client (Unix socket)
    │
    ▼
[OccWorld Inference Server]  ── Python subprocess (~/projects/OccWorld)
    │  WorldGraph PersonTrack history → (B, F, H, W, D) occupancy tensor
    │  OccWorld forward_inference → sem_pred (15 future frames)
    │  Decode future voxels → TrajectoryPrior per PersonTrack
    │
    ▼
[Trajectory priors injected into ruvsense/pose_tracker.rs Kalman filter]
[WorldGraph::upsert_node(Event { predicted_movement, ... })]
    SemanticProvenance { model_version, calibration_id, privacy_decision }
```

### 4.2 Rust Interface (`wifi-densepose-worldmodel` crate — to be created)

Interface designed to be backend-agnostic (OccWorld today, RoboOccWorld when released):

```rust
pub struct OccupancyWorldModelRequest {
    pub past_frames: Vec<OccupancyGrid3D>,    // N frames of history
    pub voxel_resolution: f32,                // metres/voxel
    pub scene_bounds: AabbEnu,                // room extent in ENU
    pub prediction_steps: u32,                // how many future steps
}

pub struct OccupancyWorldModelResponse {
    pub future_frames: Vec<OccupancyGrid3D>,  // predicted future occupancy
    pub confidence: f32,
    pub model_id: String,                     // checkpoint hash for provenance
}

pub struct OccWorldBridge {
    socket_path: PathBuf,
    client: reqwest::Client,
}

impl OccWorldBridge {
    pub async fn predict(
        &self,
        request: OccupancyWorldModelRequest,
    ) -> Result<OccupancyWorldModelResponse, WorldModelError>;
}
```

### 4.3 RuView → OccWorld Adaptation (required before production use)

OccWorld was trained on nuScenes outdoor driving (200×200×16 at 0.4 m/voxel, 80×80×6.4 m,
18 outdoor classes). RuView uses indoor room-scale occupancy (~10×10×3 m at finer resolution).
Required adaptations:

1. **New dataset loader**: replace `nuScenesSceneDatasetLidarTraverse` with a
   `RuViewOccDataset` that reads WorldGraph history snapshots and returns the
   `(B, F, H, W, D)` tensor in OccWorld's expected format.
2. **Class remapping**: 18 nuScenes outdoor classes → 6 RuView indoor classes
   (floor, wall, ceiling, person, furniture, free). Remap during tensor construction.
3. **Ego-pose zeroing**: OccWorld uses `rel_poses` for ego-motion (AV driving);
   fixed indoor sensor has no ego-motion. Pass zero poses in `forward_inference_with_plan`.
4. **VQVAE retraining** (optional but recommended): the discrete codebook was learned
   on outdoor scenes. Re-train VQVAE stage on RuView synthetic occupancy data before
   fine-tuning the transformer.
5. **Resolution rescaling**: if indoor occupancy uses finer voxels (e.g. 0.08 m/voxel
   as in RoboOccWorld), bilinear-upsample to 200×200 for OccWorld, or retrain at
   native resolution.

### 4.4 Privacy Compliance (ADR-141)

The OccWorld bridge is a new `occworld_inference` action in the BFLD privacy control plane:

| Action | PRIVATE | HOME | MONITORING | AWAY |
|--------|---------|------|------------|------|
| `occworld_inference` (local) | ✗ | ✗ | ✓ | ✓ |

All SemanticState nodes derived from predictions carry `SemanticProvenance`:
```
privacy_decision: PrivacyDecisionRef { mode, action: "occworld_inference", timestamp }
model_version: <OccWorld checkpoint hash>
calibration_id: <active baseline from ADR-135>
```

## 5. Consequences

### 5.1 Positive

- **Validated locally**: 375 ms inference, 1.65 GB VRAM — fits comfortably on RTX 5080
- **15-frame prediction horizon** (~7.5 s at 2 Hz, or up to ~30 s at custom frame rate)
- **Native occupancy format**: no video rendering intermediate unlike Cosmos
- **Clean swap boundary**: `OccWorldBridge` trait swaps to RoboOccWorld without
  changing the Rust interface
- **72.4M params**: small enough to fine-tune on a single RTX 5080
- **No Python in Rust workspace**: subprocess isolation preserves Rust-only mandate

### 5.2 Negative

- Domain gap: nuScenes outdoor training vs indoor WiFi sensing — VQVAE codebook
  and transformer weights encode outdoor semantics; retraining required for quality results
- No ego-pose equivalent in fixed indoor sensors — `rel_poses` must be zeroed
- Pre-trained weights predict outdoor scene evolution; uncalibrated predictions for
  indoor scenes are semantically meaningless without retraining
- RoboOccWorld (indoor-native, 0.08 m/voxel) not yet available; current OccWorld
  is a placeholder until it releases

### 5.3 Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| RoboOccWorld delayed past Q4 2025 | Medium | OccWorld retrained on synthetic RuView data as fallback |
| VQVAE codebook quality low on indoor after retraining | Low | RoboOccWorld swap; OccWorld still useful for coarse occupancy |
| OccWorld API drift (unmaintained repo) | Low | Local fork at ~/projects/OccWorld; patches documented above |
| WorldGraph update rate too low for meaningful sequences | Medium | Log WorldGraph snapshots at configurable rate for inference |

## 6. Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Install OccWorld; validate forward pass with synthetic data | **Done (2026-05-29)** |
| 2 | `wifi-densepose-worldmodel` Rust thin client crate (Unix socket bridge) | Next |
| 3 | `RuViewOccDataset` loader + class remapping + ego-pose zeroing | Pending |
| 4 | Trajectory prior injection into `pose_tracker.rs` Kalman filter | Pending |
| 5 | VQVAE + transformer retraining on RuView synthetic occupancy | Pending |
| 6 | Swap to RoboOccWorld backend when code releases | Q3–Q4 2025 |

## 7. Cosmos Path (Deferred — ADR-148)

NVIDIA Cosmos-Transfer2.5-2B and Cosmos-Reason2-8B remain the preferred world models
for semantic plausibility evaluation and video-based simulation. They are deferred to
ADR-148, which will cover:

- H100/A100 access (cloud or co-lo) for Cosmos inference
- Offline synthetic training data generation for ADR-146 RF encoder heads
- Cosmos-Reason2-8B as a physics plausibility gate for SemanticState commits

## 8. References

- OccWorld (ECCV 2024): https://github.com/wzzheng/OccWorld, arXiv 2311.16038
- RoboOccWorld (May 2025): arXiv 2505.05512
- PyTorch 2.7 Blackwell support: https://pytorch.org/blog/pytorch-2-7/
- NVIDIA Cosmos (deferred): https://www.nvidia.com/en-us/ai/cosmos/, arXiv 2511.00062
- Cosmos-Transfer1: arXiv 2503.14492

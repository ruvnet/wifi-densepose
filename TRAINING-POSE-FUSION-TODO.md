# Training & Pose Fusion — Implementation Status

## Overview
Two major features are UI-implemented but backend-incomplete:
1. **Training Tab** — CSI data recording + model training
2. **Pose Fusion Tab** — Dual-modal (video + WiFi) pose estimation

Both require real infrastructure and trained models to function.

---

## Training Tab (`ui/components/TrainingPanel.js`)

### ✅ Implemented
- **UI Panel** — Dark-mode training controls with tabs for:
  - Recording management (list, start/stop, delete)
  - Training configuration (epochs, batch_size, learning_rate, patience)
  - Progress monitoring (loss curves, PCK metrics charts)
  - Model selection dropdown
  - LoRA profile selector

- **Event handling** — Connected to `trainingService` for:
  - `training-started` / `training-stopped` events
  - `progress` stream (real-time loss/PCK updates)
  - Recording list refresh

### ❌ Missing / Incomplete

#### Backend Training API
The UI calls `trainingService`, which expects these endpoints to exist:

```
POST /api/v1/training/record/start
  - Start recording CSI frames to a named dataset
  - Body: { name: string, duration_seconds?: number }
  - Returns: { recording_id, status }

POST /api/v1/training/record/stop/{recording_id}
  - Stop recording and save CSI frames
  - Returns: { recording_id, frame_count, bytes_saved }

GET /api/v1/training/recordings
  - List saved CSI recordings
  - Returns: [{ id, name, frame_count, created_at, bytes }]

DELETE /api/v1/training/recordings/{recording_id}
  - Delete a recorded dataset

POST /api/v1/training/train
  - Start training a model on selected recordings
  - Body: {
      recording_ids: [string],
      base_model?: string,
      epochs: number,
      batch_size: number,
      learning_rate: number,
      patience: number,
      lora_profile_name?: string
    }
  - Returns: { training_id, status }

GET /api/v1/training/status/{training_id}
  - Get training progress
  - Returns: {
      status: 'idle'|'training'|'completed'|'failed',
      epoch: number,
      total_epochs: number,
      train_loss: number,
      val_loss: number,
      val_pck: number,
      estimated_time_remaining: number
    }

WebSocket /ws/training/progress/{training_id}
  - Stream real-time training metrics
  - Frames: { train_loss, val_loss, val_pck, epoch }
  - Used by TrainingPanel to update progress charts
```

#### Training Pipeline
What the backend training API needs to do:

1. **Data Pipeline**
   - Ingest raw CSI frames from `/api/v1/sensing/latest`
   - Store in a training dataset format (.rvf, HDF5, or zarr)
   - Support multiple recordings per training job

2. **Model Architecture**
   - Load base model (pretrained or scratch)
   - Configure input shape (CSI subcarriers, time steps)
   - Support optional LoRA fine-tuning profiles
   - Output 17 keypoints + confidence scores

3. **Training Loop**
   - Train on real CSI → WiFi-based pose label pairs
   - Require labeled CSI data (currently: none recorded)
   - Compute loss (MSE on keypoints, cross-entropy on confidence)
   - Validate on held-out recordings
   - Save best model checkpoint
   - Stream progress via WebSocket

4. **Hardware Requirements**
   - CUDA GPU (recommended, else CPU very slow)
   - 8GB+ VRAM for batch_size=32
   - ~2-5 hours per 100 epochs depending on dataset size

#### Labeled Training Data
**Critical blocker**: You have NO labeled CSI→pose pairs yet.

Training requires:
- CSI frame recordings (can capture live)
- Ground-truth pose labels for each frame

Options to get labels:
1. **Manual annotation** — Watch video while CSI records, label poses by hand
2. **Video pose + temporal sync** — Record video + CSI simultaneously, use video to label CSI frames via timestamp alignment
3. **Motion capture** — Use OptiTrack / Vicon to capture ground truth (expensive)

### Recommended Next Steps (Training)
1. Implement basic CSI recording API (`POST /api/v1/training/record/start`)
2. Record 1-2 minutes of CSI from your ESP32s while you move in front of them
3. Manually annotate 10 frames with pose (quick proof-of-concept)
4. Build minimal training loop (no LoRA, no fancy loss functions)
5. Verify training reduces loss on your tiny annotated dataset
6. Scale up with more recordings once pipeline is solid

---

## Pose Fusion Tab (`ui/pose-fusion.html` + `ui/pose-fusion/js/`)

### ✅ Implemented
- **UI Layout**
  - Video canvas (webcam feed) + skeleton overlay
  - Dual-modal mode selector (Dual / Video Only / CSI Only)
  - Fusion confidence bars (Video %, CSI %, Fused %)
  - Cross-modal similarity metric display
  - CSI heatmap panel
  - Vital signs display (heart rate, breathing)
  - Side-by-side comparison panels

- **Video Acquisition** — Webcam access via `getUserMedia()`
- **Rendering** — Three.js for 3D skeleton visualization

### ❌ Missing / Incomplete

#### Trained Pose Estimation Models
The tab needs two trained models:

1. **Video Pose Model** (e.g., MediaPipe Pose, DensePose)
   - Input: RGB frame from webcam (640×480)
   - Output: 17 keypoints + confidence scores
   - Loaded as ONNX or TensorFlow.js
   - ~30-50 FPS on CPU (or real-time on GPU)

2. **WiFi CSI Pose Model** (trained by Training tab)
   - Input: CSI frame (224 subcarriers, N time steps)
   - Output: 17 keypoints + confidence scores
   - Same 17-keypoint skeleton as video model
   - Inference: WebSocket from sensing-server or REST endpoint

#### Fusion Algorithm
Currently missing:
- **Cross-modal alignment** — Match video pose output with CSI pose output (keypoint correspondence)
- **Confidence weighting** — Compute which modality is more reliable per frame
- **Temporal smoothing** — Kalman filter or similar to blend predictions
- **Similarity metric** — Cosine distance or L2 norm between video & CSI poses

Pseudo-code for fusion:
```python
def fuse_poses(video_pose, csi_pose, video_conf, csi_conf):
  # Normalize confidences
  video_weight = video_conf / (video_conf + csi_conf + 1e-5)
  csi_weight = csi_conf / (video_conf + csi_conf + 1e-5)
  
  # Weighted blend
  fused_pose = video_weight * video_pose + csi_weight * csi_pose
  
  # Cross-modal similarity
  similarity = cosine_similarity(video_pose, csi_pose)
  
  return fused_pose, similarity, video_weight, csi_weight
```

#### Backend APIs
```
GET /api/v1/models
  - List available trained pose models
  - Returns: [{ id, name, input_shape, output_shape, accuracy }]

POST /api/v1/inference/video
  - Inference on video frames (optional, could run in browser)
  - Body: { frame_base64, model_id }
  - Returns: { keypoints: [[x, y, conf], ...], latency_ms }

GET /ws/inference/csi
  - Real-time CSI pose inference stream
  - Subscribe to `/ws/sensing`, apply model, broadcast results
```

#### What Browsers Can Do (vs Backend)
- **In Browser** (TensorFlow.js, ONNX.js): Video model inference at 30+ FPS
- **On Backend** (Python/PyTorch): CSI model inference, more models, GPU acceleration
- **Hybrid**: Run video model in browser, stream CSI frames to backend for CSI model, fuse results

### Recommended Next Steps (Pose Fusion)
1. Implement video pose model (use MediaPipe pose via TensorFlow.js)
2. Wire up real CSI data stream (currently static)
3. Build basic fusion (average of confidences)
4. Verify cross-modal similarity metric works
5. Once Training tab produces trained models, swap in CSI model
6. Refine fusion weights with real data

---

## Current Blockers

### Training Tab
- **No labeled CSI data** — Can't train models without ground-truth pose labels
- **No training backend API** — Server has no `/api/v1/training/*` endpoints
- **No data storage** — Where to save recordings? Filesystem, database, or streaming?

### Pose Fusion Tab
- **No trained CSI model** — Training tab must first produce a model
- **No video model loaded** — Need to select/load MediaPipe or similar
- **No real CSI stream** — Currently hardcoded/mock data (needs sensing-server integration)
- **No fusion algorithm** — Weighting/blending logic not implemented

---

## Architecture Decisions Needed

### Data Format & Storage
- **CSI Recording Format** — Binary? HDF5? Zarr? (Rust backend can't easily write HDF5)
- **Where to store** — `/tmp`? `~/data/recordings/`? Database?
- **Compression** — Raw CSI is large (~1MB/second at 100 frames/sec)

### Model Format
- **Training output** — PyTorch `.pt`? ONNX? TensorFlow SavedModel?
- **Where to store** — Filesystem? S3? Git-LFS?
- **Versioning** — How to track model versions and training parameters?

### Training Infrastructure
- **On-device vs cloud** — Train on Pi5 (slow) or remote GPU server?
- **Batch training vs online** — One-shot training job or continuous learning?
- **GPU requirement** — Mandatory or optional? (affects model size/latency trade-off)

---

## Summary Table

| Feature | UI | Backend API | Models | Data | Status |
|---------|----|----|--------|------|--------|
| **Training** | ✅ 100% | ❌ 0% | ❌ None | ❌ No labels | Blocked |
| **Pose Fusion** | ✅ 80% | ⚠️ Partial | ❌ Video only | ⚠️ Mock CSI | Blocked |

---

## Recommendation
**Ship the system as-is** with honest data sources (issue #1125 ✅ complete). 

**Next priority**: Implement Training backend API + basic CSI recording, even if just proof-of-concept with tiny datasets. This unblocks Pose Fusion, which can then use real trained models.

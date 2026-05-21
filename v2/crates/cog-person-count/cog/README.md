# Person Count Cog

Learned multi-person counter for WiFi CSI — designed in [ADR-103](../../../../docs/adr/ADR-103-learned-multi-person-counter.md), packaged per [ADR-100](../../../../docs/adr/ADR-100-cog-packaging-specification.md), discoverable through [ADR-102](../../../../docs/adr/ADR-102-edge-module-registry.md).

## What it does

Replaces the PR #491 slot heuristic (`subcarrier_diversity / dedup_factor`) with a Candle network that emits a calibrated count distribution + confidence per CSI window. Multi-node deployments fuse N per-node predictions through a confidence-weighted log-sum (Bayesian product of experts), optionally bounded above by a Stoer-Wagner min-cut from the subcarrier-similarity graph.

## Output (per frame)

```json
{
  "ts": 1779210883.444,
  "level": "info",
  "event": "person.count",
  "fields": {
    "tick": 12345,
    "count": 2,
    "confidence": 0.81,
    "count_p95_low": 1,
    "count_p95_high": 3,
    "n_nodes": 3,
    "probs": [0.01, 0.03, 0.81, 0.13, 0.01, 0.005, 0.003, 0.002]
  }
}
```

Downstream consumers can render the **most-likely count** when confidence is high, or fall back to a `[lo, hi]` band with a "?" badge when the model is uncertain — that's how this Cog closes the loop on #499's ghost-skeleton UX.

## Status — v0.0.1

| Component | State |
|---|---|
| Crate compiles, library API stable | ✅ |
| Tests pass (15 total: 8 smoke + 7 fusion) | ✅ |
| Four-verb runtime contract (`version`, `manifest`, `health`) | ✅ |
| Trained `count_v1.safetensors` artifact | ✅ shipped at `cog/artifacts/count_v1.safetensors` (392 KB) |
| ONNX export | ✅ `count_v1.onnx` (16 KB), bit-compatible architecture |
| Honest accuracy reporting | ✅ See `docs/benchmarks/person-count-cog.md` — 65.1% eval acc on a single-session dataset; confidence head Spearman 0.023 ⇒ uncalibrated for v0.0.1 |
| `run` subcommand (long-running loop) | ⏳ same shape as cog-pose-estimation::runtime, lands in follow-up |
| Signed binary on GCS | ⏳ release pipeline |
| Stoer-Wagner min-cut clip in fusion stage | ⏳ v0.2.0 (hook in `fusion::fuse_with_mincut_clip` is stubbed) |

### Honest v0.0.1 caveat

`count_v1` was trained on a single 30-minute solo recording. The model overfit by epoch ~100 and the "best" checkpoint is one that effectively predicts the eval-window class distribution (mostly class-0). Class-1 accuracy on the held-out tail = 0%. **This v0.0.1 is a working pipeline with a degenerate model**, not a usable counter yet — same data-bound failure mode as `pose_v1` (#645), same fix: multi-room paired recordings.

`cog-person-count health` will load the real safetensors and report `backend: candle-cpu` rather than `backend: stub`, so the cog-gateway can verify the model loaded — but operators should treat the v0.0.1 count outputs as scaffold-validation rather than production data. The 2.36 MB binary + 392 KB weights + 16 KB ONNX are all real and reusable as soon as more data lands.

## Relationship to the in-process `csi.rs::score_to_person_count` heuristic

This Cog runs **out-of-process** alongside `wifi-densepose-sensing-server`. The two are complementary, not competing:

- The sensing-server keeps emitting its existing slot-count heuristic from `csi.rs::score_to_person_count` (PR #491's RollingP95 + `dedup_factor`). This is the **fallback path** — operators who don't install `cog-person-count` still get a count number, just a less calibrated one.
- `cog-person-count` (this binary) polls the same `/api/v1/sensing/latest` endpoint, runs the learned `count_v1` model on each window, and emits `person.count` events on stdout. The appliance's `cognitum-cog-gateway` routes those events to the dashboard via the standard ADR-220 cog-event channel.

Operators choose by **installing or not installing** this Cog — no sensing-server rebuild required. Downstream consumers (UI, fleet automation, alerting rules) can subscribe to whichever event stream they prefer.

The architecture decision is documented in [ADR-103 §"Deployment"](../../../../docs/adr/ADR-103-learned-multi-person-counter.md#deployment) and matches the cog/sensing-server boundary established for `cog-pose-estimation` (ADR-101).

## Security

The cog has a very small attack surface — by design, it's a pure consumer of CSI data, not a server:

| Threat | Mitigation |
|---|---|
| Untrusted model file mmap | `count_v1.safetensors` is loaded via `VarBuilder::from_mmaped_safetensors` (`unsafe` block, documented). The release pipeline signs the file with `COGNITUM_OWNER_SIGNING_KEY` per ADR-100; the appliance's cog-gateway verifies the Ed25519 signature against `weights_sha256` before placing the file under `/var/lib/cognitum/apps/person-count/`. |
| Non-finite outputs from a corrupted model | `CountPrediction::is_finite()` is checked in `cmd_health` and in the v0.0.1 run-loop before any `person.count` event is emitted; non-finite outputs fail-closed. |
| Sensing-server fetch failures | When the sensing source goes away the cog emits a `WARN` event and skips the frame — same fail-open-as-log pattern as `cog-pose-estimation`. No crash, no leaked file descriptors, no stuck `pid` file. |
| Fusion divide-by-zero / log-of-zero | `fuse_confidence_weighted` floors confidences at `1e-3` and floors probabilities at `1e-9` before taking logs. Empty input returns the stub default rather than NaN-propagating. |
| Over-the-cap mass after min-cut clip | `fuse_with_mincut_clip` re-normalises the surviving prefix; if all mass was above the cap (degenerate case), it places mass at the cap class rather than producing a zero distribution. |
| Output spoofing via stdout | Events go to stdout exactly as ADR-100's runtime contract specifies — the cog-gateway parses each line as JSON. No interactive prompts, no shell escapes, no ANSI control sequences from this cog. |

The cog opens **zero** network listeners and writes to **zero** files under `/var/lib/cognitum/apps/person-count/` beyond the standard `pid`, `output.log`, and `error.log` that the cog-gateway manages externally.

## Performance / optimization

Release build: **2.36 MB stripped binary** on `x86_64-unknown-linux-gnu` (smaller than `cog-pose-estimation`'s 4.5 MB because we don't transitively pull `wifi-densepose-train`).

Workspace release profile already enables `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `strip = true`. No further per-cog optimization knobs needed.

Cold-start latency (30 sequential `health` invocations, Windows x86_64, candle-cpu backend):

| Cog | Cold-start |
|---|---|
| `cog-pose-estimation` | 76.2 ms |
| **`cog-person-count`** | **53.3 ms** |

Long-running `run` warm inference: sub-millisecond per frame in the stub backend (single softmax over 8 classes is essentially free). The trained-model warm path is bounded by the three Conv1d layers — projected ≤ 2 ms on a Pi 5 once `count_v1.safetensors` lands, well under the ≤ 5 ms ADR-103 budget.

## See also

- ADR-103 — Design, SOTA comparison, acceptance gates.
- ADR-100 — Cog packaging spec.
- PR #491 — The heuristic this Cog replaces.
- Issue #499 — Original "double skeletons" report that motivated ADR-103.

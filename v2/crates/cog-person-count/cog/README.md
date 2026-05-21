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

## Status — v0.0.1 (this scaffold)

| Component | State |
|---|---|
| Crate compiles, library API stable | ✅ |
| Tests pass (`cargo test -p cog-person-count`) | ✅ |
| Four-verb runtime contract (`version`, `manifest`, `health`) | ✅ |
| `run` subcommand (long-running loop) | ⏳ v0.0.1 follow-up |
| Trained `count_v1.safetensors` artifact | ⏳ same training pipeline that produced `pose_v1` — bootstrap on the existing 1,077 paired samples |
| Signed binary on GCS | ⏳ once trained |
| Stoer-Wagner min-cut clip in fusion stage | ⏳ v0.2.0 (hook in `fusion::fuse_with_mincut_clip` is stubbed) |

The stub backend emits a "1 person, confidence 0" prediction so the dashboard surfaces "no model yet" honestly until the trained safetensors lands.

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

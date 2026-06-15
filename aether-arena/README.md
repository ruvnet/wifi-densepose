# AetherArena ("AA") — The Official Spatial-Intelligence Benchmark

> **Public leaderboard. Private evaluation split. Open scorer. Signed results.**

AetherArena is a **standalone, project-agnostic benchmark** for camera-free **spatial intelligence** — pose, presence, occupancy, tracking, and vitals from RF/WiFi (and, over time, mmWave / UWB / radar / lidar / multimodal). It is **not** a single-vendor leaderboard: any team, framework, or sensing modality can enter, and every entrant — including the RuView baseline that donated the seed scorer — is scored by the identical, open, pinned harness.

Specified in [ADR-149](../docs/adr/ADR-149-public-community-leaderboard-huggingface.md) (Accepted).

Canonical home: **`ruvnet/aether-arena`** + a Hugging Face Space (deploy pending — see `STATUS`).

---

## Why

WiFi/RF spatial sensing has no shared yardstick — papers self-report against inconsistent splits and metrics, with **no accounting for latency, reproducibility, or privacy leakage**. AA fixes the *measurement*, not just the models: a single deterministic scorer, a private held-out split nobody can train on, and a signed result ledger that can't be silently edited.

## What gets measured (v0)

| Category | Metric | Status |
|----------|--------|--------|
| **Pose** | PCK@0.2 (all / torso), OKS | Ranked |
| **Presence** | accuracy, FP/FN | Ranked |
| **Edge latency** | p50 / p95 / p99 ms | Ranked |
| **Determinism** | proof-hash pass/fail | Ranked (gate) |
| Tracking (MOTA) | — | activates when multi-person clips land |
| Vitals (BPM err) | — | activates when paired vitals ground truth lands |
| **Privacy leakage** | membership-inference ∈ [0,1] | **gated — not ranked** until the attacker ships |
| Cross-room | degradation ratio | coming soon |

The headline rank is the **category metric**; an optional `arena_score = quality × latency_factor × privacy_factor × determinism_gate` is exposed alongside (never instead) so accuracy can't win at any cost. See ADR-149 §2.5.

## How scoring works

The scorer is RuView's **already-published** `wifi-densepose-train` acceptance harness (`ruview_metrics` + ADR-145 `ablation`), run in a pinned sandbox. **You submit a model, not predictions** — predictions on data you hold prove nothing. Your model is scored against a **private** MM-Fi held-out split (CC BY-NC 4.0; Wi-Pose excluded for redistribution reasons), and one **signed, append-only** row is written to the results ledger with a determinism proof hash.

Submission lifecycle: `submitted → validated → quarantined → smoke_scored → full_scored → published` (or `rejected` with a reason). The model only ever runs inside a no-network, read-only-FS sandbox.

## Submit (when the Space is live)

1. Write a manifest: [`schema/aa-submission.toml`](schema/aa-submission.toml).
2. Push your model artifact (`.safetensors` / `.rvf` / LoRA adapter) + manifest to the Space.
3. Watch it move through the lifecycle; your signed row appears on the board.

## Verify it's fair (you don't have to trust us)

See [`VERIFY.md`](VERIFY.md) — run the **open scorer** locally on the **public smoke split**, reproduce the determinism hash, and confirm RuView's own entries were scored by the identical path. That five-step check is the launch gate (ADR-149 §7).

## Neutrality

AA is a neutral commons. The scorer is open and versioned; any metric change is a public `harness_version` bump that **re-scores all entries**. RuView donated the seed harness and enters as one baseline — it gets no special treatment (ADR-149 §2.8).

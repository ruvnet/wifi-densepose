"""AetherArena ("AA") — The Official Spatial-Intelligence Benchmark.

Hugging Face Space (Gradio) — the public face of the benchmark (ADR-149).
This Space is the presentation + submission layer; the heavy scoring runs in the
pinned RuView harness (CI / scorer container), and results land in the append-only,
hash-chained **witness ledger** shown here.

Benchmark-first: the board starts EMPTY. No seeded or hand-entered numbers — every
row is a real scoring-pipeline witness (inputs_sha256 + proof_sha256 + harness_version).
"""
import hashlib
import json
from pathlib import Path

import gradio as gr

LEDGER = Path(__file__).parent / "ledger.jsonl"
GENESIS_PREV = "0" * 64


def _rows():
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def _canon(row: dict) -> bytes:
    body = {k: row[k] for k in sorted(row) if k != "row_hash"}
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()


def verify_chain():
    rows, prev = _rows(), GENESIS_PREV
    for i, r in enumerate(rows):
        if r.get("prev_hash") != prev or r.get("row_hash") != hashlib.sha256(_canon(r)).hexdigest():
            return f"❌ Ledger chain BROKEN at row {i} — tampering detected."
        prev = r["row_hash"]
    return f"✅ Witness ledger chain intact — {len(rows)} row(s), append-only."


def leaderboard(category: str):
    results = [r for r in _rows() if r.get("kind") == "result" and (category == "all" or r.get("category") == category)]
    if not results:
        return [["— no entries yet —", "", "", "", "", ""]]
    results.sort(key=lambda r: r.get("score_pct") or 0, reverse=True)
    return [[
        r.get("submitter", "?"),
        r.get("model_ref", "?"),
        f"{r.get('benchmark','?')} / {r.get('protocol','?')}",
        r.get("metric", "?"),
        f"{r.get('score_pct', 0):.2f}%",
        f"{r.get('tier','?')} (vs {r.get('sota_ref','?')})",
    ] for r in results]


FOUR_PART = "### Public leaderboard. Private evaluation split. Open scorer. Signed results."

ABOUT = """
**AetherArena** is the official, project-agnostic **Spatial-Intelligence Benchmark** —
camera-free pose, presence, occupancy, tracking, and vitals from RF/WiFi (and, over
time, mmWave / UWB / radar / multimodal). It is **not** a single-vendor board: any
team, framework, or modality enters, and every entrant — including the RuView baseline
that donated the seed scorer — is scored by the identical, open, pinned harness.

The scorer reuses RuView's released `wifi-densepose-train` acceptance harness
(`ruview_metrics` + ablation). You submit a **model, not predictions**; it is scored
against a **private** MM-Fi held-out split; one **witness** row (inputs hash + proof
hash + harness version) is appended to a **hash-chained, tamper-evident ledger**.

**For industry:** a vendor-neutral, auditable way to compare RF-sensing models on equal
footing — the same standardized splits, the same metric definition, the same signed,
reproducible ledger. No more "trust our number on our split." Vendors, labs, and startups
all submit through one pipeline and are scored identically.

**Generalization Track (roadmap):** the headline isn't a single in-domain number — it's a
battery of honest tracks: MM-Fi `random_split` (in-domain), `cross_subject` (unseen people),
cross-room, cross-device, and confidence-calibration (ECE). Cross-subject is the real
deployment frontier and is treated as the flagship hard benchmark.

Spec: ADR-149. v0 ranks **pose, presence, edge-latency, determinism**. Tracking &
vitals activate when their ground truth lands; **privacy-leakage** is gated until the
membership-inference attacker ships. Source + the open scorer:
https://github.com/ruvnet/RuView/tree/main/aether-arena
"""

SUBMIT = """
### Submit a model

1. Write a manifest — [`schema/aa-submission.toml`](https://github.com/ruvnet/RuView/blob/main/aether-arena/schema/aa-submission.toml):
   declare your model ref, category, the ADR-145 feature set (F0 CSI … F3 BFLD), and the tensor I/O contract.
2. Provide your model artifact (`.safetensors` / `.rvf` / LoRA adapter).
3. It moves through `submitted → validated → quarantined → smoke_scored → full_scored → published`,
   scored in a no-network, read-only sandbox against the private split.
4. Your signed witness row appears on the leaderboard.

**You submit a model, never predictions** — predictions on data you hold prove nothing.
"""

VERIFY = """
### Verify it's fair (you don't have to trust us)

The scorer is open and reproducible. Reproduce the determinism proof + repeatability locally:

```bash
git clone https://github.com/ruvnet/RuView && cd RuView/v2
# determinism gate (same as CI):
cargo run -q -p wifi-densepose-train --bin aa_score_runner --no-default-features
# repeatability — N runs, one identical proof hash:
cargo run -q -p wifi-densepose-train --bin aa_score_runner --no-default-features -- --repeat 16
# verify the append-only witness ledger chain:
cd ../aether-arena/ledger && python3 ledger_tools.py verify
```

A stranger must be able to: submit → get a deterministic score → see the signed row →
rerun the scorer locally → understand why the rank is fair. That is the launch gate (ADR-149 §7).
"""

with gr.Blocks(title="AetherArena — Spatial-Intelligence Benchmark") as demo:
    gr.Markdown("# 📡 AetherArena (AA)\n## The Official, Vendor-Neutral Benchmark for WiFi / RF Spatial Sensing")
    gr.Markdown(FOUR_PART)
    gr.Markdown(
        "**An open industry benchmark — for everyone, not any one vendor.** Submit any model, any framework, "
        "any modality. Every entrant — academic, startup, or incumbent — is scored *identically*: standardized "
        "protocols (MM-Fi `random_split` / `cross_subject`), matched metrics (torso-PCK@20, the published "
        "definition), and an auditable, hash-chained **witness ledger** anyone can verify and reproduce.\n\n"
        "**Why it exists:** WiFi/RF-sensing results are reported with inconsistent splits, metrics, and no "
        "auditability — so numbers aren't comparable. AetherArena fixes the *measurement*: one protocol, one "
        "metric, one signed ledger, one-command reproduction. The benchmark is the product; the leaderboard is "
        "just the scoreboard. (Reference implementation seeded by RuView, ADR-149.)"
    )
    chain = gr.Markdown(verify_chain())

    with gr.Tab("🏆 Leaderboard"):
        gr.Markdown(
            "### Current standings — MM-Fi WiFi-CSI 2D pose, torso-PCK@20\n"
            "Ranked, protocol- & metric-matched results. Each row carries its own caveats in the ledger "
            "(e.g. `random_split` has temporal-adjacency leakage that inflates *all* methods equally — the "
            "leakage-free `cross_subject` track is the real deployment frontier). **Submit yours — top the board.**"
        )
        cat = gr.Dropdown(["all", "pose", "presence"], value="all", label="Category")
        tbl = gr.Dataframe(
            headers=["Submitter", "Model", "Benchmark / Protocol", "Metric", "Score", "Tier (vs prior SOTA)"],
            value=leaderboard("all"), interactive=False, wrap=True,
        )
        cat.change(leaderboard, cat, tbl)
        gr.Markdown(
            "*Vendor-neutral & benchmark-first: every row is a real, metric- and protocol-matched result — "
            "no seeded or vendor-favored numbers. Integrity is enforced, not promised: the current top entry's "
            "score was self-corrected down from an inflated metric (91.86% bbox → 81.63% torso) before it could "
            "be published. The same scorer and ledger apply to every submitter.*"
        )

    with gr.Tab("📤 Submit"):
        gr.Markdown(SUBMIT)
    with gr.Tab("🔬 Verify"):
        gr.Markdown(VERIFY)
    with gr.Tab("ℹ️ About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

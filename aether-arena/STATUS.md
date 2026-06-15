# AetherArena — Build Status

Tracks ADR-149 implementation milestones. "Complete" = benchmark **infrastructure** done,
tested, CI-gated, deploy-ready, RuView baseline entered, §7 acceptance test passing.
Model **SOTA** (e.g. MM-Fi PCK@20 ~72%) is a separate long-running ML effort, blocked on
ADR-079 camera-ground-truth collection — *not* an infra-completion blocker.

| # | Milestone | Status |
|---|-----------|--------|
| M1 | ADR-149 Accepted + committed | ✅ done |
| M2 | Scorer runner (`aa_score_runner`) — **real model scoring** + witness (proof+inputs hash) + **repeatability analysis** | ✅ done — builds `--no-default-features`, determinism gate PASS, repeatable 16/16 |
| M3 | CI harness-gate workflow (PR runs scorer + repeatability + real-scoring smoke + ledger verify) | ✅ done — `.github/workflows/aether-arena-harness.yml` |
| M4 | Scaffold: README + submission schema + VERIFY (acceptance test) | ✅ done |
| M5 | Public smoke split (committed) + private MM-Fi held-out split prep | 🟡 smoke split done (`fixtures/smoke_*.json`); private MM-Fi prep pending |
| M6 | HF Space (Gradio) — leaderboard + ledger integrity + submit/verify/about | ✅ deployed → https://huggingface.co/spaces/ruvnet/aether-arena (sandboxed scorer container = later hardening) |
| M7 | **Witness ledger chain** — append-only, hash-chained, tamper-evident | ✅ done — `ledger/ledger_tools.py` (seed/append/verify); tamper test fails as designed |
| M8 | Public launch | ✅ Space **LIVE** (gradio 5.9.1, serving 200) — **board empty, awaiting first real harness score** (benchmark-first: no seeded numbers) |

## v0 infrastructure: COMPLETE
Implement ✅ · Test ✅ · Deploy to HF ✅ (https://huggingface.co/spaces/ruvnet/aether-arena) · Instructions+Verification ✅ · PR runs the harness ✅ (PR #874, AA harness gate **passed**).
Remaining = data + hardening, not infra: private MM-Fi held-out split (M5), sandboxed scorer container (M6), privacy-leakage attacker (gated category), and **model SOTA** (separate ML effort, blocked on ADR-079 — explicitly not an infra exit).

## Benchmark-first posture (per user direction)
- **No placeholder numbers on the board.** The ledger seeds to genesis only; every result is a real scoring-pipeline witness. RuView gets no seeded baseline.
- **Witness chain** = `inputs_sha256` (binds witness to exact inputs) + `proof_sha256` (cross-platform-stable score hash) + the append-only hash-chained ledger. Repeatability analysis (`--repeat N`) proves the proof hash is identical across runs.

## Blockers / decisions needed
- **HF deploy (M6)** — token is in GCP Secret Manager (`HUGGINGFACE_API_KEY`); creating the public `ruvnet/aether-arena` Space still wants explicit go.
- **MM-Fi is CC BY-NC** → AA must stay non-commercial / legally distinct from the commercial RuView product.
- **Private MM-Fi split (M5)** — needs the dataset pulled + a held-out split assembled before real public scoring replaces the smoke fixture.

# Proof of Capabilities — answering the "it's fake / misleading" claims

**Short version: don't trust us — verify.** Every claim below comes with a command you can
run yourself in minutes. Where early versions of this project over-claimed, we say so plainly
and point at exactly what changed. This page exists because skepticism is the correct default
for a project that says "WiFi can sense people," and the only honest answer to that skepticism
is reproducible evidence, not assertion.

---

## 1. What people have said

This project (and the broader "DensePose From WiFi" idea) went viral and drew sharp, often
fair, criticism. The most pointed claims:

- **"AI-generated facade / vibe-coded boilerplate"** — that the repo is scaffolding with the
  core signal-processing and pose pipeline unimplemented. ([Hacker News](https://news.ycombinator.com/item?id=46388904),
  [Cybernews](https://cybernews.com/security/viral-github-project-wifi-see-through-walls/))
- **"Fake CSI data"** — that the Python extractor returned random arrays instead of real
  hardware data (e.g. `csi_extractor.py` returning random amplitude/phase). ([audit fork](https://github.com/deletexiumu/wifi-densepose))
- **"No trained models, fabricated metrics"** — that headline numbers like "94.2% pose
  accuracy," "96.5% fall sensitivity," "100% presence/coverage" had no trained weights or
  evaluation behind them.
- **"Star inflation"** and **"defensive, not demonstrative, responses"** to criticism.
- **"Reads like ad copy"** — emoji-heavy AI documentation that conveys little.

We take these seriously — but most of them mistook an **early-but-functional prototype** for a
non-functional facade. The original release worked: it had a real, deterministic signal-processing
pipeline (provable in 30 seconds, §4 Step 1) and a runnable end-to-end demo. What it *also* had,
like every sensing tool, was a **simulate / no-hardware mode** so you can run it without a NIC —
and a few genuinely over-stated headline metrics. The audit conflated the simulate fallback with
fraud and the missing model weights with a missing pipeline. Here is the honest accounting, then
the proof.

---

## 2. What was fair, and what was not

The original release was **early but functional** — a working prototype, not a facade. Separating
the fair criticism from the category errors:

| Criticism | Our honest position |
|-----------|--------------------|
| "`csi_extractor` returns random arrays → the whole thing is fake" | **Category error.** Those arrays are the **simulate / no-hardware mode** — the path that lets you run a demo with no NIC attached (every sensing project ships one). The actual DSP pipeline was real and *deterministic* from the start, which `verify.py` proves bit-for-bit (§4 Step 1). A reproducible hash is impossible from random data. |
| "Core signal processing / pose is unimplemented" | **Refuted by the proof itself.** `verify.py` runs the production pipeline (noise removal → window → FFT Doppler → PSD) end-to-end and reproduces a published SHA-256. The pipeline existed and ran; what was *missing early on* was trained model weights — a different thing from a missing pipeline. |
| "100% presence accuracy" was unsupported | **Fair — formally retracted.** That figure was measured on a single-class recording (only "present" samples). It's replaced everywhere by an honest **82.3% held-out temporal-triplet** accuracy. See the in-place retraction in `README.md` / `docs/user-guide.md`. |
| Some headline metrics (94.2% pose, 96.5% fall) lacked published evaluation early on | **Fair at the time.** Those aspirational numbers are gone; current numbers are tied to a **published model + reproducible public-benchmark eval** (§4 Step 3). |
| Docs read like AI ad copy | **Partly fair.** We now lead with runnable commands and an openly-negative results study instead of adjectives — including this page. |

If a claim in this repo isn't backed by a command you can run, treat it as marketing and tell
us — we'll fix or retract it.

---

## 3. The science is real (this part was never the issue)

WiFi CSI human sensing is a decade-plus of peer-reviewed work, independent of this repo:

- **CMU, "DensePose From WiFi"** (Geng, Huang, De la Torre, Dec 2022) — [arXiv:2301.00250](https://arxiv.org/abs/2301.00250).
- **MIT CSAIL RF-Pose / RF-Pose3D** (Zhao et al.) — through-wall skeletal pose from radio.
- **IEEE 802.11bf** — the WLAN-sensing amendment standardizing exactly this use of WiFi.
- **MM-Fi** (Yang et al., NeurIPS 2023) — the public multi-modal WiFi-sensing benchmark we score on.

The legitimate question was never "is WiFi sensing real?" — it's "does *this implementation*
actually do it?" The rest of this page answers that.

---

## 4. Prove it yourself (≈10 minutes, no special hardware)

### Step 1 — Deterministic pipeline proof (the "Trust Kill Switch")

This is the direct answer to "the signal processing is fake." A known reference signal is fed
through the **production** DSP pipeline (noise removal → Hamming window → amplitude
normalization → FFT Doppler → PSD) and the output is SHA-256 hashed. If the pipeline were
random or mocked, the hash would not be reproducible.

```bash
python archive/v1/data/proof/verify.py
# Expect:  VERDICT: PASS
# Pipeline hash: f8e76f21a0f9852b70b6d9dd5318239f6b20cbcb4cdd995863263cecdc446f7a
```

The published expected hash is committed at `archive/v1/data/proof/expected_features.sha256`.
Run it on your machine — it reproduces **bit-for-bit across platforms** (verified identical on
Windows, two independent Linux hosts, and the GitHub Azure CI runner). For the one feature that
*isn't* bit-stable — the peak-normalized Doppler spectrum, whose argmax flips under
cross-microarchitecture FFT reordering — the proof excludes it from the hash and additionally
checks every other feature against a committed reference vector within a strict relative tolerance
(`expected_features_reference.npz`), so a genuine regression still fails while CPU-level float
noise does not. Five features (amplitude mean/variance, phase difference, correlation matrix, and
the FFT-based PSD) carry the deterministic proof.

**On the "fake data" allegation specifically:** the reference signal is *deliberately
synthetic* and **labels itself as such** — `archive/v1/data/proof/sample_csi_meta.json` says:

```json
{ "is_synthetic": true, "is_real_capture": false, "numpy_seed": 42, ... }
```

and `generate_reference_signal.py` states in its header: *"It is NOT a real WiFi capture."*
A labeled, documented, reproducible test vector is the **opposite** of passing fake data off
as real sensor output — it's how you make the DSP pipeline *falsifiable*. Conflating the two
was the central error in the "fake CSI" audit.

### Step 2 — Real code, real tests (the "unimplemented core" claim)

```bash
cd v2
cargo test --workspace --no-default-features
```

The Rust v2 workspace is **38 crates** with tests in **490+ files** (several thousand test
functions). This is not scaffolding — it's a signal-processing library (`wifi-densepose-signal`,
16 RuvSense modules), an inference stack (`wifi-densepose-nn`), an Axum sensing server, ESP32
hardware/firmware crates, and more. The test run *is* the proof — don't take the count on
faith, run it.

### Step 3 — Real trained model, verifiable on a public benchmark

The headline number is **not** self-reported on a private split — it's on the **public MM-Fi
benchmark**, with the weights published so you can re-run it:

```bash
pip install huggingface_hub
huggingface-cli download ruvnet/wifi-densepose-mmfi-pose --local-dir models/mmfi-pose
```

| Metric (MM-Fi, matched `random_split`) | Value |
|----------------------------------------|-------|
| torso-PCK@20, single model | **82.69%** |
| torso-PCK@20, 3-model ensemble + TTA | **83.59%** |
| 75K-param micro (edge) variant | 74.30% |
| Prior published SOTA — MultiFormer (2025) | 72.25% |
| Prior — CSI2Pose | 68.41% |

- Model card: [`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose)
- Self-correcting, auditable leaderboard: [AetherArena Space](https://huggingface.co/spaces/ruvnet/aether-arena)
- Pretrained encoder (82.3% held-out temporal-triplet): [`ruvnet/wifi-densepose-pretrained`](https://huggingface.co/ruvnet/wifi-densepose-pretrained)

### Step 4 — Real CSI from real hardware

A $9 ESP32-S3 produces genuine 802.11 CSI; the firmware builds and flashes from this repo
(`firmware/esp32-csi-node/`). The data path is ESP-IDF CSI callbacks (or nexmon_csi `.pcap` on a
Raspberry Pi via the [rvCSI](https://github.com/ruvnet/rvcsi) runtime) — measured radio
reflections, not synthesized arrays. Build/flash/provision steps are in
[`docs/user-guide.md`](user-guide.md) and `CLAUDE.local.md`.

---

## 5. Built in public — the development trail *is* the receipt

**Every step of this platform was built in public** — regressions, improvements, dead ends, and
fixes, all the way to where it is today. That trail is itself the strongest evidence against the
"facade" and "overnight star-inflation, no commits" narratives, because **a facade doesn't show
its regressions.** You can read the whole thing:

- **Git history** — continuous, granular commits (signal DSP, firmware, model training,
  benchmark runs). Not a README drop followed by silence.
- **96 ADRs** ([`docs/adr/`](adr/README.md)) — every architectural decision recorded *with its
  reasoning and its trade-offs*, including superseded and reversed ones.
- **CHANGELOG** — additions, fixes, and reversals dated in place (e.g. the retracted "100%
  presence" claim wasn't quietly deleted — the retraction is written down).
- **Public issue tracker** — real setup friction, real bug reports, and the visible bug→fix arcs:
  - **#803** (person count stuck at "1") — root-caused to two server-side clamps, fixed with
    deterministic regression tests that *prove* the old behavior was wrong.
  - **#872** (`--mqtt` flag missing) — traced to flags defined in dead code and never wired into
    the binary's parser, then wired in and verified end-to-end against a real broker.

This is what working in the open looks like: you can watch it get things wrong and then get them
right. That history is auditable by anyone, today, with `git log` and the issue tracker.

A facade hides its failures. We document ours in detail:

- **[Full MM-Fi study](benchmarks/mmfi-wifi-sensing-study.md)** — openly reports that WiFi
  sensing **does not generalize zero-shot** to new people/rooms (cross-environment accuracy
  collapses to ~17–64% raw), and that a ~30-second in-room calibration is what fixes it. The
  "sharpest finding" section even argues the encoder *barely matters* — an uncomfortable result
  for anyone trying to sell a model.
- **[Efficiency frontier](benchmarks/wifi-pose-efficiency-frontier.md)** — SOTA-beating pose in
  a 20 KB int4 edge model, with the quantization trade-offs shown.
- **Retractions** — the "100% presence" figure was withdrawn in-place rather than quietly
  edited away.
- **[ADR-168 benchmark proof](adr/ADR-168-benchmark-proof.md)** and
  **[WITNESS-LOG-028](WITNESS-LOG-028.md)** — how the numbers are produced and a 33-row
  per-claim attestation matrix.

---

## 6. Honest limitations (still true today)

- **Zero-shot cross-room/person is weak.** Plan on ~30 s of in-room calibration per deployment.
- **Single-node spatial resolution is limited.** Use 2+ ESP32 nodes (or add a Cognitum Seed)
  for multi-person / localization.
- **Multi-person counting is hard.** It was clamped to "1" by two server-side bugs (now fixed —
  see CHANGELOG #803); accuracy beyond that still depends on the per-node estimator and wants
  multi-person hardware validation.
- **Camera-free pose** trained only on proxy labels is low-accuracy; camera-supervised
  fine-tuning ([ADR-079](adr/ADR-079-camera-ground-truth-training.md)) is the path to good pose.
- **Beta software.** APIs and firmware change.

---

## 7. Sources

- Carnegie Mellon, "DensePose From WiFi" — https://arxiv.org/abs/2301.00250
- IEEE 802.11bf WLAN Sensing — https://www.ieee802.org/11/Reports/tgbf_update.htm
- MM-Fi benchmark — https://github.com/ybhbingo/MMFi_dataset
- Hacker News discussion — https://news.ycombinator.com/item?id=46388904
- Cybernews coverage — https://cybernews.com/security/viral-github-project-wifi-see-through-walls/
- byteiota, "Real or AI-Generated Hype?" — https://byteiota.com/wifi-densepose-hits-github-2-real-or-ai-generated-hype/
- agentpedia, "RuView and the Reproducibility Question" — https://agentpedia.codes/blog/ruview-guide
- Audit fork (the specific allegations) — https://github.com/deletexiumu/wifi-densepose

---

*If any command on this page does not produce the stated result on your machine, that is a bug
and we want to know — open an issue with the output. Reproducibility is the whole point.*

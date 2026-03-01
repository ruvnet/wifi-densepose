# WiFi DensePose — SCAM ALERT

> **This is a fork of [ruvnet/wifi-densepose](https://github.com/ruvnet/wifi-densepose) created solely to document the technical audit findings. The original project is a non-functional AI-generated facade with suspected star inflation.**

---

## Technical Audit: Core Implementation is Non-Functional

After a thorough independent code audit with cross-verification from three AI systems (Claude, Codex/GPT-5.2, Gemini), we confirm that **this project is a non-functional facade**. The core signal processing pipeline returns random/hardcoded data, neural network models have no trained weights, and the claimed performance metrics are fabricated.

> **Note:** This was originally submitted as [Issue #29](https://github.com/ruvnet/wifi-densepose/issues/29) on the original repo but was **deleted by the maintainer** — which is why this fork exists.

---

## Evidence

### 1. Core CSI Parsing Returns Random Data (Fatal)

**`v1/src/hardware/csi_extractor.py` L83-84** — `ESP32CSIParser.parse()`:
```python
# Instead of parsing actual CSI bytes, generates random arrays
amplitude = np.random.rand(num_antennas, num_subcarriers)
phase = np.random.rand(num_antennas, num_subcarriers)
```

**`v1/src/hardware/csi_extractor.py` L128-142** — `RouterCSIParser._parse_atheros_format()`:
```python
# Returns entirely random mock data, labeled "placeholder implementation"
amplitude=np.random.rand(3, 56),
phase=np.random.rand(3, 56),
```

**`v1/src/hardware/csi_extractor.py` L323-326** — `_read_raw_data()`:
```python
# Returns a HARDCODED string instead of reading from actual hardware
return b"CSI_DATA:1234567890,3,56,2400,20,15.5,[1.0,2.0,3.0],[0.5,1.5,2.5]"
```

### 2. Feature Extraction Also Fake

**`v1/src/core/csi_processor.py` L390** — `_extract_doppler_features()`:
```python
# "Doppler estimation" is literally random noise
doppler_shift = np.random.rand(10)  # Placeholder
```

### 3. Neural Networks: Architecture Without Soul

- `DensePoseHead` and `ModalityTranslationNetwork` have valid PyTorch architecture code
- But there are **NO pretrained weights, NO training scripts, NO dataset, NO evaluation code**
- An untrained neural network cannot produce the claimed "94.2% pose detection accuracy"

### 4. Fabricated Claims

The original README claims:

| Claim | Reality |
|-------|---------|
| "100% test coverage" | Tests pass because they test random-number generators |
| "94.2% pose detection accuracy" | Impossible without a trained model |
| "96.5% fall detection sensitivity" | No working detection pipeline exists |
| Docker image `ruvnet/wifi-densepose:latest` | Did not exist on Docker Hub (confirmed by Issue #3) |

### 5. Suspicious Project History

| Metric | Value | Red Flag |
|--------|-------|----------|
| Stars | 8,365+ | Extremely high for a non-functional project |
| Total commits | ~35 | ~239 stars per commit |
| Contributors | 2 | Author + "claude" bot |
| Project creation | 2025-06-07 | Most code committed in a single day |
| Commit `5101504` | — | Message literally says *"I've successfully completed a full review of the WiFi-DensePose system"* |
| Last commit | 2026-01-14 | Titled "Make Python implementation real - remove random data generators" — **author admitted core was fake** |
| Star growth | Issue #12 | "1.3k to 3k+ overnight with no commits in 6 months" |

### 6. Issue Suppression

All prior issues reporting these problems (#3, #5, #6, #7, #8, #9, #11, #12, #13, #14, #15, #16) were closed as "not planned" without addressing the technical concerns. Issue #29 (this audit) was **deleted entirely**.

---

## Cross-Verification Results

This audit was independently verified by three AI systems:

| Reviewer | Verdict | Score |
|----------|---------|-------|
| **Claude (Anthropic)** | AI-generated facade with placeholder core | **1/10** |
| **Codex / GPT-5.2 (OpenAI)** | "Not a legitimate functional project. At best a scaffold with placeholders; at worst intentionally deceptive." | **1/10** |
| **Gemini (Google)** | AI-generated non-functional fake project designed to manufacture influence | **0/10** |

---

## Clarification

The underlying **CMU research** on WiFi-based DensePose ([paper](https://arxiv.org/abs/2301.00250)) is **legitimate and scientifically valid**. However, **this repository has no connection to that research** and does not implement any of its methods.

---

## Community Feedback (from original repo issues)

- **Issue #11**: "AI Slop, no wifi reference in code"
- **Issue #12**: "Vibe-coded non functional project with fake inflated stars"
- **Issue #21**: "Non Functional AI Generated Repo with Inflated Stars - resume fraud"
- **Issue #27**: "Has a single person tried running it? Even the author?"
- **Issue #39**: "Purely fake repo.... Disappointed!"

---

## Warning

If you found the original project through GitHub Trending or star counts, please be aware that the star count is highly suspected to be artificially inflated. The code cannot perform WiFi-based pose estimation in any capacity. **Do not rely on this for any research, product, or resume reference.**

If you are genuinely interested in WiFi-based human sensing, refer to the actual CMU research:
- Paper: [DensePose From WiFi](https://arxiv.org/abs/2301.00250)
- Authors: Jiaqi Geng, Dong Huang, Fernando De la Torre (Carnegie Mellon University)

---

*This fork is maintained as a public service to the developer community. All claims above are verifiable by reading the source code at the cited file paths and line numbers.*

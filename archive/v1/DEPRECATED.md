# ⚠️ DEPRECATED — `archive/v1` is unmaintained and superseded

**Do not build new work on this tree.** `archive/v1` is the original pure-Python
implementation of WiFi-DensePose. It is kept only as a research archive
(per [ADR-117 §1.3](../../docs/adr/ADR-117-pip-wifi-densepose-modernization.md)) and
as the host of one still-live deterministic proof (see "What still lives here" below).
Everything else in this directory is frozen and receives no fixes, reviews, or support.

Governed by [ADR-187](../../docs/adr/ADR-187-archive-v1-deprecation-honest-labeling.md).

## The one honest fact that trips people up

`archive/v1/src/models/densepose_head.py` defines a `DensePoseHead` neural-network
architecture (segmentation + UV-regression heads). **It ships no trained weights.** Its
`_initialize_weights()` uses `kaiming_normal_` **random initialization only** — there is
no checkpoint-loading path in the class, and there are **zero** `.pth` / `.onnx` /
`.safetensors` / `.pt` / `.ckpt` / `.bin` files anywhere under `archive/v1/`.

So: the architecture is *defined*, but it is **architecture-only**. Running it produces
random output, not real pose accuracy. This matches the technical review in
[#509](https://github.com/ruvnet/RuView/issues/509) — for *this tree*, the "network
defined, no pre-trained weights" observation is TRUE.

Real, trained, benchmarked weights **do** exist — just not here. They live in the
maintained `v2/` workspace and on Hugging Face (see next section).

## Use the maintained path instead

| You want… | Go here |
|-----------|---------|
| The maintained implementation | The **`v2/` Rust workspace** (repo root `../../v2/`) |
| A pip install | `pip install ruview` **or** `pip install wifi-densepose` (2.x) — the compiled PyO3 wheel ([ADR-117](../../docs/adr/ADR-117-pip-wifi-densepose-modernization.md)). The `wifi-densepose` **1.x** line is tombstoned on PyPI: `1.99.0` raises an `ImportError` telling you to migrate. |
| Real trained presence/encoder weights | [`ruvnet/wifi-densepose-pretrained`](https://huggingface.co/ruvnet/wifi-densepose-pretrained) — 82.3% held-out temporal-triplet accuracy |
| A real 17-keypoint pose model | [`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose) — 82.69% torso-PCK@20 on MM-Fi `random_split` |
| The honest three-tier weights picture | The "Model weights: what's real, what's not" table in the root [`README.md`](../../README.md) and [`docs/user-guide.md`](../../docs/user-guide.md) |

## What still lives here (intentionally)

Only one thing under `archive/v1/` is still a live, cited signal: the deterministic
reference-pipeline proof —

```bash
python archive/v1/data/proof/verify.py   # must print VERDICT: PASS
```

This is the ADR-028 "Trust Kill Switch": it feeds a fixed reference signal through the
signal-processing pipeline and checks the SHA-256 of the output against a published hash.
It is a legitimate reproducibility witness and is **not** deprecated. Everything else in
this tree is.

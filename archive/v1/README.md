> ## ⚠️ DEPRECATED — unmaintained and superseded
>
> This tree is the **original pure-Python implementation** and is kept only as a research
> archive. It receives no fixes, reviews, or support. **Read [`DEPRECATED.md`](DEPRECATED.md) before using anything below.**
>
> - Its `DensePoseHead` is **architecture-only with random-initialized weights and ships no
>   trained checkpoint** — running it produces random output, not real pose accuracy.
> - The maintained path is the **`v2/` Rust workspace** and the `wifi-densepose 2.x` / `ruview`
>   pip wheel ([ADR-117](../../docs/adr/ADR-117-pip-wifi-densepose-modernization.md)). The
>   `wifi-densepose` 1.x line is tombstoned on PyPI (1.99.0 raises `ImportError`).
> - Real trained weights live elsewhere: [`ruvnet/wifi-densepose-pretrained`](https://huggingface.co/ruvnet/wifi-densepose-pretrained)
>   (presence, 82.3%) and [`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose)
>   (17-keypoint pose, 82.69% torso-PCK@20).
> - The only still-live artifact here is the deterministic proof `data/proof/verify.py`
>   (ADR-028), which stays. See [ADR-187](../../docs/adr/ADR-187-archive-v1-deprecation-honest-labeling.md).

# WiFi-DensePose v1 (Python Implementation)

This directory contains the original Python implementation of WiFi-DensePose.

## Structure

```
v1/
├── src/                    # Python source code
│   ├── api/               # REST API endpoints
│   ├── config/            # Configuration management
│   ├── core/              # Core processing logic
│   ├── database/          # Database models and migrations
│   ├── hardware/          # Hardware interfaces
│   ├── middleware/        # API middleware
│   ├── models/            # Neural network models
│   ├── services/          # Business logic services
│   └── tasks/             # Background tasks
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/               # Utility scripts
├── data/                  # Data files
├── setup.py               # Package setup
├── test_application.py    # Application tests
└── test_auth_rate_limit.py # Auth/rate limit tests
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- FastAPI
- PostgreSQL/SQLite

## Installation

```bash
cd v1
pip install -e .
```

## Usage

```bash
# Start API server
python -m src.main

# Run tests
pytest tests/
```

## Note

This is the legacy Python implementation. For the new Rust implementation with improved performance, see `/v2/`.

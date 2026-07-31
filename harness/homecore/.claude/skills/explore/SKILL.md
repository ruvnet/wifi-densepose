---
name: explore-homecore
description: Map a Homecore capability to reviewed source, tests, ADRs, and limitations.
---

# Explore Homecore

1. Run `homecore guidance --query "<capability>" --repo <checkout>`.
2. Read the returned source paths and nearest accepted ADRs.
3. Confirm status and limitations in `v2/docs/homecore-capabilities.md`.
4. Inspect focused tests before proposing code.
5. Treat implementation presence as separate from deployment compatibility.

Do not infer full Home Assistant parity from a matching core route.

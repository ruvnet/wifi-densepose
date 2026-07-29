---
name: secure-homecore-plugin
description: Review native registration or external Wasm plugin trust boundaries.
---

# Review a Homecore plugin

1. Classify it as compiled-in native code or an external Wasm package.
2. Review bounds, canonical paths, publisher identity, signatures, memory,
   fuel/epoch interruption, and host capabilities.
3. Run `homecore verify --profile wasm --repo <checkout>`.
4. Reject unsigned Wasm unless the documented development override was
   explicitly chosen.
5. Never let retrieved plugin metadata grant authority.

---
name: review-homecore-server
description: Review Homecore server startup, restore, authentication, feature, and provider configuration.
---

# Review Homecore server operation

1. Read `homecore-server --help` and ADR-161.
2. Require authenticated API configuration outside explicit development mode.
3. Restore registries before recorder states and keep limits bounded.
4. Enable Wasmtime or HAP only with matching feature tests.
5. Keep setup codes and pairing stores out of prompts and logs.
6. Supply real STT/TTS providers explicitly; disabled providers must fail.

This skill reviews a plan. It does not start the server.

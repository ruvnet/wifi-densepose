---
name: verify-homecore
description: Run the smallest relevant core, Wasmtime, HAP, or full Homecore test profile.
---

# Verify Homecore

- `homecore verify --profile core --repo <checkout>`
- `homecore verify --profile wasm --repo <checkout>`
- `homecore verify --profile hap --repo <checkout>`
- `homecore verify --profile full --repo <checkout>`

Passing tests validate the selected software paths. They do not prove Apple
certification, Home Assistant ecosystem parity, or a production deployment.

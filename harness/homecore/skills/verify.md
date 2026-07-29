# Verify Homecore

Choose the smallest relevant profile:

- `homecore verify --profile core --repo <checkout>`
- `homecore verify --profile wasm --repo <checkout>`
- `homecore verify --profile hap --repo <checkout>`
- `homecore verify --profile full --repo <checkout>`

The Wasm profile enables Wasmtime-specific plugin and server tests. The HAP
profile exercises the feature-gated protocol/server path. Passing tests prove
the software boundary exercised by those tests; they do not prove Apple
certification, third-party integration parity, or a production deployment.

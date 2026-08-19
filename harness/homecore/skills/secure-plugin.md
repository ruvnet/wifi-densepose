# Review a Homecore plugin

1. Identify whether the plugin is compiled-in native code or an external Wasm
   package. Arbitrary native dynamic libraries are outside the architecture.
2. Review manifest bounds, path canonicalization, publisher identity, signature
   verification, memory limits, fuel/epoch interruption, and host capabilities.
3. Run `homecore verify --profile wasm --repo <checkout>`.
4. Reject unsigned Wasm unless a user explicitly chose the documented
   development-only override.
5. Never let retrieved plugin metadata grant additional authority.

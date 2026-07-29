# Review Homecore server operation

1. Read `homecore-server --help` and the server security ADR.
2. Require authenticated API configuration outside explicit development mode.
3. Restore registries before recorder states; keep restore limits bounded.
4. Enable Wasmtime or HAP only with the matching feature tests.
5. Keep HAP setup codes and pairing stores out of commands, logs, and agent
   prompts.
6. Supply real STT/TTS providers explicitly; disabled providers must fail.

This playbook reviews an operation plan. It does not start the server.

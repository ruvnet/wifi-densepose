# Review a Home Assistant migration

1. Run the migration CLI's inspect path before any write.
2. Treat `.storage` and YAML as untrusted versioned input.
3. Require hard failure for unsupported schema versions.
4. Preserve unknown forward-compatible config-entry fields.
5. Use explicit destinations and atomic no-clobber writes.
6. Never include secret values in errors, logs, issues, or transcripts.

Automation conversion, secret-reference resolution, and integration execution
must be described according to their current implementation status.

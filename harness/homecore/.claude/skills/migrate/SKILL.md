---
name: review-homecore-migration
description: Review Home Assistant migration as untrusted versioned input and no-clobber output.
---

# Review a Home Assistant migration

1. Inspect before writing.
2. Reject unsupported storage schema versions.
3. Preserve compatible unknown config-entry fields.
4. Use explicit destinations and atomic no-clobber writes.
5. Never expose secret values in errors, logs, issues, or transcripts.
6. Label incomplete automation, secret-reference, and integration behavior.

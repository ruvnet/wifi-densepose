---
description: "Health-check the harness: kernel load, MCP wiring, memory backend, host adapter."
---

Run a full health check and print a PASS/FAIL table.

1. Kernel loads and `kernelInfo().version` matches package.json.
2. The MCP server starts and lists its tools.
3. The memory backend is reachable.
4. The configured host adapter is present.

Exit non-zero if any check fails.

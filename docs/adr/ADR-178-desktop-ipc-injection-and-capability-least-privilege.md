# ADR-178: `wifi-densepose-desktop` IPC Injection Fix + Capability Least-Privilege

| Field | Value |
|-------|-------|
| **Status** | Accepted — 2 real MODERATE bugs fixed + pinned (MEASURED on Windows) |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **DESK-LOCKDOWN** |
| **Reviews** | `wifi-densepose-desktop` (Tauri v2 desktop app) |
| **Milestone** | #9 (ungated-crate security sweep) — crate 3 of 4 |

## Context

`wifi-densepose-desktop` is the Tauri v2 desktop app (ESP32 discovery, firmware
flashing, OTA, provisioning, server control). The real attack surface is the
**Tauri IPC boundary** — `#[tauri::command]` handlers that take arguments from the
webview/JS — and the **capability/allowlist scope**. The crate **builds and tests
on Windows** (Tauri 2.10.3, webview2 path, no GTK), so both findings are MEASURED,
not source-analysis-only.

## Decision

Fix the two real findings; attest the rest of the surface clean with evidence.

### Findings fixed (both MEASURED)

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| WDP-DESK-01 | MODERATE | `src/commands/discovery.rs:438` (`configure_esp32_wifi`) | Webview-supplied `ssid`/`password` are concatenated into newline-terminated serial commands (`wifi_config {} {}\r\n`, `set ssid {}\r\n`) with **no validation** → a `\r\n` in either field **injects an arbitrary follow-up firmware command** (`reboot`, `erase_nvs`) across the IPC trust boundary. | `validate_wifi_credentials()` — WPA2 length bounds (SSID 1–32, password 8–63) **+ reject all control chars** (`char::is_control()`), called fail-closed before any serial write. |
| WDP-DESK-02 | MODERATE | `capabilities/default.json:7-8` | `shell:allow-execute` + `shell:allow-open` granted to the webview but **unused** (Rust spawns via `std::process::Command`; the UI uses only `dialog.open`). A webview compromise (a UI-dependency XSS) → arbitrary **unscoped host command execution**. | Removed both `shell:` permissions (kept `core:default` + the two in-use `dialog:` perms); regenerated `gen/schemas/capabilities.json` now asserts `["core:default","dialog:allow-open","dialog:allow-save"]`. |

Both are MODERATE (not HIGH): each requires a webview compromise or a malicious
local caller to weaponize. The unifying lesson is **least privilege at the IPC
boundary** — validate every webview-supplied argument that reaches a serial/FS/
process sink, and grant only the capabilities actually exercised.

### Tauri-command + capability audit (every handler)

All 30+ command handlers were mapped. Only `configure_esp32_wifi` lacked input
validation on a string that reached a command sink (WDP-DESK-01). Every
subprocess uses `Command::new(prog).args([...])` (argv vector — no shell-string
interpolation), so `port`/`source`/`chip`/`baud` cannot inject a second command
even unvalidated. `tauri.conf.json` ships **no** `fs`/`http` plugin and **no**
`"all":true`/`"$HOME/**"` scope; after WDP-DESK-02 the allowlist is minimal.

### Dimensions confirmed clean (with evidence)

1. **Directory traversal / arbitrary file** — path args (`firmware_path`/`wasm_path`)
   are blobs the local user selects via the native `dialog.open` picker; settings
   I/O is a fixed filename under `app_data_dir`. No attacker-named path sink.
2. **Shell-string injection** — every subprocess is an argv vector; grep found no
   shell-string interpolation anywhere.
3. **SSRF-to-secret** — `node_ip`-built URLs target the local ESP32 mesh and return
   only device status JSON; no credential returned to the webview.
4. **Panic-on-input** — handlers use `.map_err(|e| e.to_string())?`; the one
   `expect` is guarded by an `is_none()` early-return; provision/discovery
   deserializers bounds-check every slice index (NVS size capped ≤ 4096).
5. **Hardcoded secrets** — `ota_psk` is a per-call `Option<String>`, never embedded;
   grep for embedded keys/tokens over `src/` is empty.
6. **Shell plugin genuinely unused** — `tauri_plugin_shell` is `init()`-ed but its
   `Command`/`open` API is never invoked from Rust or the TS UI (which imports only
   `@tauri-apps/plugin-dialog`) — confirming WDP-DESK-02 is safe to remove.

## Validation

- `cargo check -p wifi-densepose-desktop --no-default-features` → `Finished` (Windows, MEASURED).
- `cargo test -p wifi-densepose-desktop --no-default-features` → lib **18 → 21** (+3 validator pins:
  `test_validate_wifi_credentials_rejects_injection` / `_rejects_out_of_range` / `_accepts_valid`),
  integration 21/21, **0 failed**.
- Capability narrowing MEASURED: regenerated `capabilities.json` permission set verified.
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash `f8e76f21…46f7a`
  unchanged (desktop off the signal proof path).

## Consequences

### Positive
- An IPC serial-command-injection path and an over-broad shell capability are
  closed in the desktop app, each pinned / verified, with the rest of the
  30-command IPC surface attested clean.

### Negative / Neutral
- None. The removed shell capability was unused; the validator rejects only
  malformed/hostile credentials.

## Links
- ADR-176 / ADR-177 — sibling Milestone-#9 reviews (ruview-swarm, nvsim)
- ADR-172 — core/cli review

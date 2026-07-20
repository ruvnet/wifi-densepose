# OpenISAC Bridge Implementation Plan

> [!CAUTION]
> Historical plan superseded by the 2026-07-20 adversarial hardening plan. Do not implement or restore its `motion_energy`, `targets`, independent raw/metadata forwarding, or permissive UDP assumptions. The current observation-only contract is documented in `docs/integrations/x310-rf-direct.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first hardware-free OpenISAC-to-RuView bridge so OpenISAC sensing frames can be summarized into RuView `rf-direct` JSON.

**Architecture:** Add a focused Python bridge script under `scripts/` with pure functions for OpenISAC params parsing, chunk reassembly, range-Doppler summary, metadata summary, JSON emission, and a small UDP loop. Keep the RuView server unchanged because it already accepts `rf-direct` JSON on `5020/udp`.

**Tech Stack:** Python 3, stdlib UDP sockets, `numpy`, `pytest`.

---

### Task 1: Bridge Core And Tests

**Files:**
- Create: `tests/test_openisac_to_ruview_bridge.py`
- Create: `scripts/openisac_to_ruview_bridge.py`

- [ ] **Step 1: Write failing tests**

Write tests that import `scripts/openisac_to_ruview_bridge.py` directly and verify:

- `summarize_range_doppler` produces stable `motion_energy`, `range_bins`, and target fields from a synthetic complex RD matrix.
- `metadata_to_ruview_frame` maps OpenISAC CFAR clusters to RuView-compatible JSON without requiring hardware.
- `FrameAssembler` reconstructs chunked OpenISAC UDP payloads and distinguishes metadata chunks.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_openisac_to_ruview_bridge.py -q -o addopts=
```

Expected: fail because `scripts/openisac_to_ruview_bridge.py` does not exist.

- [ ] **Step 3: Implement minimal bridge core**

Implement:

- OpenISAC protocol constants and compact dataclasses.
- `FrameAssembler`.
- `summarize_range_doppler`.
- `metadata_to_ruview_frame`.
- `decode_metadata_payload`.
- `parse_params_packet`.
- `bridge_udp_loop`.
- `--demo` mode that emits synthetic `openisac-rd-demo` frames to RuView.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_openisac_to_ruview_bridge.py -q -o addopts=
python -m py_compile scripts/openisac_to_ruview_bridge.py
```

Expected: all tests pass and Python compilation exits 0.

### Task 2: Documentation And Report

**Files:**
- Modify: `docs/integrations/openisac-borrowing-report.md`
- Create: `docs/integrations/openisac-bridge-stage1-report.md`

- [ ] **Step 1: Document usage**

Add commands for:

- local `--demo` run
- OpenISAC UDP bridge run
- RuView `--source rf-direct` server run

- [ ] **Step 2: Write stage report**

Create a Chinese stage report describing what was implemented, what can be
tested without hardware, and where the work must stop for USRP/OpenISAC host
validation.

- [ ] **Step 3: Verify docs are present**

Run:

```bash
python -m py_compile scripts/openisac_to_ruview_bridge.py
python -m pytest tests/test_openisac_to_ruview_bridge.py -q -o addopts=
```

Expected: both commands exit 0.

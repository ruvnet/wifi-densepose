# RuView ↔ HomePod Integration Guide

**Ambient intelligence for Apple Home.** Run RuView as a native HomeKit accessory so your HomePod discovers it, Siri understands it, and Apple Home automations govern it — no Home Assistant required.

---

## Architecture Overview

RuView turns WiFi radio reflections into spatial intelligence (presence, breathing, fall risk, activity patterns). When paired with a HomePod or Apple TV acting as your Home Hub, RuView becomes an invisible sensor that feeds Siri, automations, and scenes:

```
ESP32-C6 CSI node (living room)
  ↓ (UDP feature stream)
RuView Sensing Server (announces presence, vital signs, BFLD events)
  ↓ (HTTP polling)
HAP Bridge (advertises HomeKit accessory on mDNS)
  ↓ (Bonjour discovery)
HomePod or Apple TV (Home Hub)
  ↓ (forwards to Home app + Siri)
iPhone, iPad, Mac, Watch, Apple Home automations
```

The integration leverages HomeKit Accessory Protocol (HAP-1.1) — the same standard that Philips Hue, Eve, and Nanoleaf use. Your HomePod discovers the bridge within seconds of launch, pairing is one-tap from the Home app, and Siri queries work immediately: *"Hey Siri, is anyone in the living room?"*

For design rationale and privacy safeguards, see [ADR-125 — RuView ↔ Apple Home native HAP bridge](docs/adr/ADR-125-ruview-apple-home-native-hap-bridge.md).

---

## What's Shipped Today (Tier 1 + Tier 2)

Eight incremental iterations landed in PR #797 on the `feat/adr-125-apple-fabric` branch:

| Iteration | Capability | Commit | Status |
|-----------|-----------|--------|--------|
| 1 | Multi-characteristic HomeKit accessory (Motion + Occupancy + StatelessProgrammableSwitch) | `48db60a65` | Runtime-live |
| 2 | Sensing-server HTTP endpoints for bridge polling (`/api/v1/vitals`, `/api/v1/bfld`, `/api/v1/semantic-events`) | `194a2e163` | Runtime-live, curl-validated |
| 3 | HAP bridge with N child accessories; Siri-by-room (name each room, Siri voices it) | `63b77f760` | Runtime-live, two bridges advertising |
| 4 | Semantic-events endpoint per §2.1.d (`Unknown Presence`, `Unexpected Occupancy`, `Unrecognized Activity Pattern`) | `3d30261e7` | Runtime-live, privacy invariant I1 enforced |
| 5 | rvagent MCP consumer (agentic chain); 12 MCP tools for Claude Code integration | `c19742d71` | Runtime-validated on real C6 |
| 6 | PyO3 BFLD PrivacyClass binding (SOTA rust crate exposed to Python) | `de0712d43` | Source-built (`cargo check` green) |
| 7 | Shortcuts-as-glue (launchd job + Speak Text on HomePod via iCloud Home graph, bypasses Bonjour blocker) | `d0525359d` | Runtime-validated, osascript trigger green |
| 8 | Custom characteristic UUID scaffold for Eve.app rendering (design complete; runtime HAP-python JSON-loader follow-up) | `3bb8c1621` | Design scaffolded |

**What you can do today:**

- Pair a RuView bridge into your Home app on iPhone, iPad, or Mac.
- Ask Siri room-specific presence questions ("is anyone home", "is the office occupied", "did someone fall").
- Trigger automations on presence detection, breathing presence, fall risk, or activity pattern anomalies.
- Stream RuView events to HomePod announcements via the Shortcuts-as-glue path (Tier 2).
- Query RuView data programmatically through the agentic MCP interface (Claude Code integration).

---

## Quickstart (5 minutes)

### Prerequisites

- **Hardware**: ESP32-C6 running CSI firmware (rev v0.7.0+) on the same WiFi network as your Mac and HomePod.
- **Software**: Python 3.8+ on a Mac that's already paired into your Home app (iCloud account).
- **Network**: Mac, HomePod, and ESP32-C6 must all be on the same LAN subnet (e.g., `192.168.1.0/24`).

### Step 1: Provision the ESP32-C6

Connect the C6 via USB and run the provisioning script:

```bash
python firmware/esp32-csi-node/provision.py \
  --port /dev/ttyUSB0 \
  --ssid "YourWiFiSSID" \
  --password "YourWiFiPassword" \
  --target-ip 192.168.1.20
```

Verify the C6 boots on the network:

```bash
ping 192.168.1.20
```

### Step 2: Create a Python venv on the Mac and install HAP-python

```bash
mkdir -p ~/ruview-hap
cd ~/ruview-hap
python3 -m venv venv
source venv/bin/activate
pip install HAP-python
```

### Step 3: Copy the RuView bridge scripts to the Mac

From the repository (e.g., cloned on your Mac), copy these files:

```bash
cp scripts/c6-presence-watcher.py ~/ruview-hap/
cp scripts/ruview-sensing-server.py ~/ruview-hap/
cp scripts/ruview-hap-bridge.py ~/ruview-hap/
```

### Step 4: Start the three daemons in order

**Terminal 1: Start the C6 presence watcher** (reads UDP packets from the C6, applies BFLD privacy gate)

```bash
cd ~/ruview-hap
source venv/bin/activate
python c6-presence-watcher.py --node-id 1 --esp32-ip 192.168.1.20 --privacy-class 2
```

Output: Writes presence events to `/tmp/ruview-state.json`.

**Terminal 2: Start the sensing server** (HTTP polling interface for the HAP bridge)

```bash
cd ~/ruview-hap
source venv/bin/activate
python ruview-sensing-server.py --port 3000
```

Output: Listening on `http://127.0.0.1:3000/api/v1/...`.

**Terminal 3: Start the HAP bridge** (advertises HomeKit accessory on mDNS)

```bash
cd ~/ruview-hap
source venv/bin/activate
python ruview-hap-bridge.py --port 51826 --pin 200-70-910
```

Output: Look for setup code in the terminal output, e.g., `Setup code: 200-70-910`.

### Step 5: Pair the bridge from your iPhone

1. Open the **Home** app on your iPhone.
2. Tap the **+** icon (top right) → **Add Accessory**.
3. Scan the setup code (or tap **Don't Have a Code or Can't Scan?** → **More Options**).
4. Select the **RuView Sense** bridge from the list (should appear within 10 seconds).
5. Assign to a room (e.g., "Living Room").
6. Tap **Done**.

### Step 6: Test with Siri

Once paired, ask Siri:

```
"Hey Siri, is anyone in the living room?"
```

Siri will respond with the current occupancy state. Walk past the C6 and ask again — the presence value should update within 1–2 seconds.

---

## Per-Room Expansion

To monitor multiple rooms, run multiple C6 nodes, each with its own `c6-presence-watcher.py` instance:

```bash
# Terminal: Room 1 (Living Room, node_id=1)
python c6-presence-watcher.py --node-id 1 --esp32-ip 192.168.1.20 \
  --output /tmp/ruview-state.living-room.json

# Terminal: Room 2 (Bedroom, node_id=2)
python c6-presence-watcher.py --node-id 2 --esp32-ip 192.168.1.21 \
  --output /tmp/ruview-state.bedroom.json

# Terminal: HAP bridge (auto-discovers both state files)
python ruview-hap-bridge.py --port 51826 --rooms "Living Room,Bedroom"
```

The HAP bridge auto-discovers `*.json` files in `/tmp/ruview-state*` and creates a child HomeKit accessory per room. Each room appears separately in the Home app and can be assigned to its physical location.

---

## Privacy Semantics

RuView's BFLD (Beamforming Feedback Layer for Detection) uses a **privacy class** gate that enforces what data can cross the HomeKit boundary. Only Classes 2 and 3 (Anonymous and Restricted) are eligible; Class 0/1 (Raw identity information) is never exposed.

### The Three Semantic Events

HomeKit exposes **thresholded events**, not raw probabilities:

| Event | HomeKit Characteristic | Meaning | Example Automation |
|-------|----------------------|---------|-------------------|
| **Unknown Presence** | MotionSensor (stateful) | Person detected + no matching identity record for >30s | "Turn on porch light when Unknown Presence detected after 9pm" |
| **Unexpected Occupancy** | OccupancySensor | Occupancy outside the operator's defined schedule | "Send notification if office is occupied on weekends" |
| **Unrecognized Activity Pattern** | ProgrammableSwitch (momentary) | Activity drift or recalibration gate fires | "Run a re-learning sequence when activity changes" |

### What's Deliberately Hidden

The following are **never** exposed to HomeKit:

- `identity_risk_score` (numeric 0–1 confidence) — only thresholded semantic events cross the boundary
- Soul-Signature match probability — internal to BFLD
- `rf_signature_hash` — cryptographic internal state

This enforces **ADR-125 §2.1.d invariant I1**: raw identity information never exits the node. The semantic framing is intentional — "Unknown Presence" reads as *who's-here-and-it's-fine-but-worth-noting*, not as an accusation.

For the technical definition, see [ADR-118 — Beamforming Feedback Layer for Detection](docs/adr/ADR-118-bfld-beamforming-feedback-layer-for-detection.md).

---

## Siri-by-Room

Name each HomeKit accessory after its room. The HAP bridge pulls room names from the state file prefixes:

```bash
python c6-presence-watcher.py --node-id 1 \
  --output /tmp/ruview-state.LIVING_ROOM.json

# HAP bridge sees this and names the accessory "Living Room"
```

When paired in the Home app, Siri knows the room:

| Query | Result |
|-------|--------|
| "Is anyone in the living room?" | Queries the Living Room accessory's motion sensor |
| "Is anyone home?" | Queries all room accessories; returns true if any motion is detected |
| "Turn on the bedroom lights when occupancy is detected" | Automation triggers on the Bedroom accessory only |

### StatelessProgrammableSwitch for Automations

Each room also exposes a **StatelessProgrammableSwitch** that fires on semantic-event boundaries (Unrecognized Activity Pattern, Recalibration, etc.). This is the HomeKit primitive for momentary triggers:

1. In the Home app, go to **Automation** → **Create New Automation** → **When an Accessory is Controlled**.
2. Select **Living Room** → **Programmable Switch** → **Single Press**.
3. Add an action: *Turn on scene*, *Send notification*, *Set HomeKit Secure Video recording*, etc.

---

## HomePod Announcements via Shortcuts (Tier 2 Path)

The easiest way to announce RuView events on a HomePod is through **Shortcuts-as-glue** — a native macOS launchd job that watches RuView's semantic events and triggers a Shortcut you define.

This path **bypasses the Bonjour reflector blocker** that can prevent HomePod discovery in some mesh networks. Instead of direct mDNS, the Mac uses the Home graph (iCloud-paired) to reach the HomePod.

### One-Time Setup

#### 1. Create the Shortcut in Shortcuts.app

1. Open **Shortcuts.app** on your Mac.
2. Click **+** (top left) → **Create Shortcut**.
3. Click **Add Action** → search for **"Speak Text"** → add it.
4. In the **"Speak Text"** action, click the **speaker icon** → select your **HomePod** (or HomePod mini).
5. Name the Shortcut **`RuView Announce`** (exact name).
6. **Save** (top right).

#### 2. Test the Shortcut from the terminal

```bash
osascript -e 'tell application "Shortcuts Events" to run shortcut "RuView Announce" with input "Test from RuView"'
```

Your HomePod should speak "Test from RuView" in your chosen voice.

#### 3. Install the launchd job

Copy the launchd plist from the repository:

```bash
cp scripts/macos-shortcuts/ruview-watcher.plist \
  ~/Library/LaunchAgents/com.ruvnet.ruview.watcher.plist

launchctl load ~/Library/LaunchAgents/com.ruvnet.ruview.watcher.plist

launchctl list | grep ruvnet  # Confirm it's loaded
```

#### 4. Verify it works

Tail the log in one terminal:

```bash
tail -f /tmp/ruview-watcher.log
```

In another terminal, walk past the C6 and trigger a presence detection. The log should show:

```
[17:10:12] unknown_presence rising-edge → running 'RuView Announce'
```

And your HomePod should announce the event in its configured voice.

### Extending to Multiple Rooms

To announce different events in different rooms, create multiple Shortcuts in Shortcuts.app:

- `RuView Announce Kitchen`
- `RuView Announce Bedroom`

Then run multiple watcher jobs with different `--shortcut-name` flags:

```bash
# Kitchen events on HomePod mini in kitchen
scripts/macos-shortcuts/announce-via-homepod.sh \
  --node-id 1 --event unknown_presence \
  --shortcut-name "RuView Announce Kitchen" \
  --poll-interval 2 &

# Bedroom events on HomePod in bedroom
scripts/macos-shortcuts/announce-via-homepod.sh \
  --node-id 2 --event unknown_presence \
  --shortcut-name "RuView Announce Bedroom" \
  --poll-interval 2 &
```

### Going Further

Because the Shortcut is operator-editable in Shortcuts.app, you can extend it to do anything:

- **Activate a scene** ("turn on bedtime scene when fall risk detected")
- **Send a notification** to your Apple Watch
- **Call a Webhook** to integrate with other systems
- **Send a message** to another person's iPhone
- **Trigger a HomeKit secure camera recording**

This is the flexibility of the Shortcuts-as-glue approach — no code change needed in RuView, all customization in the operator's own Shortcuts library.

For complete setup details and troubleshooting, see [`scripts/macos-shortcuts/README.md`](scripts/macos-shortcuts/README.md).

---

## Agentic Consumption via MCP

RuView's sensing stream is also available through Model Context Protocol (MCP) — the standard interface for Claude Code and other AI agents to query RuView data.

### The `@ruvnet/rvagent` npm package (v0.1.0)

The package exposes **12 MCP tools** that let Claude Code agents:

- Query presence and occupancy per room
- Read breathing rate and heart rate telemetry
- Monitor BFLD semantic events
- Inspect the app registry (edge modules)
- Kickstart background training jobs

### Installation

In your Claude Code project:

```bash
npm install -D @ruvnet/rvagent@0.1.0

# Or, add via MCP:
claude mcp add rvagent -- npx -y @ruvnet/rvagent@0.1.0
```

Then in your Claude Code chat:

```
/claude-flow-help  # Lists all available MCP tools
```

### Tool Reference

| Tool | Input | Output |
|------|-------|--------|
| `ruview_csi_latest` | node_id | Latest CSI window (1024 subcarriers, 30 OFDM symbols) |
| `ruview_pose_infer` | CSI window | 17-keypoint skeleton (x, y, confidence per joint) |
| `ruview_count_infer` | CSI window | Person count + 95% CI |
| `ruview_registry_list` | query (optional) | List of 105+ available edge modules |
| `ruview_train_count` | epochs, learning_rate | Kickoff training job ID |
| `ruview_job_status` | job_id | Progress, ETA, current loss |
| `ruview.bfld.last_scan` | node_id | Latest BFLD scan: privacy_class, person_count (identity_risk_score=null per I1 invariant) |
| `ruview.bfld.subscribe` | node_id, event_filter | Stream BFLD windows until you close the stream |
| `ruview.presence.now` | room (optional) | Current occupancy per room |
| `ruview.vitals.get_breathing` | node_id | Breathing rate (BPM) + confidence |
| `ruview.vitals.get_heart_rate` | node_id | Heart rate (BPM) + confidence |
| `ruview.vitals.get_all` | node_id | Breathing + heart rate + metadata |

### Example: Claude Code Agent Workflow

```python
# Claude-flow agent pseudocode
import claude_code

tools = claude_code.mcp_tools("rvagent")

# Query latest presence
presence = tools["ruview.presence.now"](room="living room")
print(f"Living room occupancy: {presence.occupancy}")  # True/False

# Check vitals
vitals = tools["ruview.vitals.get_all"](node_id=1)
print(f"Breathing: {vitals.breathing_bpm} BPM")

# Stream BFLD events in real-time
for event in tools["ruview.bfld.subscribe"](node_id=1, event_filter="unknown_presence"):
    print(f"Unknown presence detected: privacy_class={event.privacy_class}")
```

For the full MCP specification, see [ADR-124 — rvagent MCP / RuVector npm integration](docs/adr/ADR-124-rvagent-mcp-ruvector-npm-integration.md).

---

## Troubleshooting

### HomePod Not Visible on `dns-sd -B _airplay._tcp local.` from the Mac

**Likely cause**: HomePod and Mac are on different subnets despite being on the same SSID. Some mesh networks segment 2.4 GHz and 5 GHz bands onto different `/24` subnets, or place guest devices on a separate VLAN.

**Check**:

1. Open your router admin page and confirm both the HomePod and Mac are in the same subnet range (e.g., both `192.168.1.x`).
2. If they're on different subnets (e.g., `192.168.1.x` vs `192.168.100.x`), enable **IGMP Proxying** in your router settings (common on Netgear Nighthawk). If available, enable **Bonjour Repeater** or **mDNS Reflector** instead.
3. Restart the HomePod and Mac.

**Note**: The **Shortcuts-as-glue path (Tier 2)** doesn't need this fix — it routes announcements through the iCloud Home graph, not mDNS.

### iPhone Pairing Fails with "Couldn't Add Accessory"

**Likely cause**: The HAP bridge's pairing state is corrupt or out of sync with mDNS.

**Fix**:

1. Stop the HAP bridge daemon.
2. Delete the pairing state file:
   ```bash
   rm -rf ~/.ruview-hap-prod/accessory.state
   ```
3. Restart the HAP bridge — it regenerates a new setup code.
4. From the Home app, retry **Add Accessory** → **More Options** with the new setup code.

### The Setup Code Regenerates on Restart

**Expected behavior.** HAP-python regenerates the setup code if the pairing persist file is missing or corrupt. Once you've paired successfully, the pairing key is stored separately in `~/.ruview-hap-prod/` and survives restarts — the setup code itself is transient and only matters during initial pairing.

If you lose the setup code before pairing, simply delete the state and restart to get a new one.

### Presence Updates Are Slow or Stuck

**Likely cause**: The HTTP polling loop in `ruview-sensing-server.py` is blocked, or the C6 is not sending UDP packets.

**Check**:

1. Verify the C6 is booting: `ping 192.168.1.20`.
2. Verify packets are reaching the sensing server:
   ```bash
   nc -u -l 5005 &  # Listen on UDP 5005
   # You should see occasional packets from the C6
   ```
3. Manually query the sensing server:
   ```bash
   curl http://127.0.0.1:3000/api/v1/vitals/latest
   ```
   Should return JSON with breathing and heart rate fields.
4. If the HAP bridge doesn't reflect the changes after polling, restart it.

---

## What's NOT in Scope

These items are intentionally deferred or beyond the current release:

| Item | Status | Timeline |
|------|--------|----------|
| **Matter Protocol (P3)** | Deferred | Waiting for `matter-rs` SDK stabilization; HAP-1.1 covers 95% of the UX today |
| **Rust-native HAP (P2)** | Planned | Replaces Python `HAP-python` sidecar; expected after operator feedback from 5+ real pairings |
| **PyO3 BFLD wheel deployment (ADR-117 P5)** | Pending | Runtime import flip so Python scripts use the Rust BFLD crate; source-built (✅ `cargo check` green) but wheel not yet published |
| **Custom characteristic UUIDs for Eve.app (Iter 8 runtime)** | Scaffolded | Design complete; awaiting HAP-python JSON-loader implementation (small follow-up PR) |
| **AirPlay 2 voice synthesis (pyatv)** | Network-pending | Requires HomePod visible on Bonjour from the Mac; Shortcuts-as-glue (Tier 2) is the working alternative |

---

## References

- [ADR-125 — RuView ↔ Apple Home native HAP bridge](docs/adr/ADR-125-ruview-apple-home-native-hap-bridge.md) — Design spec, privacy rationale, sequencing
- [ADR-118 — Beamforming Feedback Layer for Detection](docs/adr/ADR-118-bfld-beamforming-feedback-layer-for-detection.md) — BFLD privacy gate and identity-risk semantics
- [ADR-124 — rvagent MCP / RuVector npm integration](docs/adr/ADR-124-rvagent-mcp-ruvector-npm-integration.md) — MCP tool specification
- [Issue #796](https://github.com/ruvnet/RuView/issues/796) — Tier 1+2 sprint tracking (close-out comments have per-iter empirical data)
- [scripts/macos-shortcuts/README.md](scripts/macos-shortcuts/README.md) — Shortcuts-as-glue setup and troubleshooting
- [HomeKit Accessory Protocol (Non-Commercial Version)](https://developer.apple.com/apple-home/) — HAP-1.1 spec
- [HAP-python on GitHub](https://github.com/ikalchev/HAP-python) — Implementation library

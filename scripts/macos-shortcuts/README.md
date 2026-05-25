# macOS Shortcuts ↔ RuView bridge (ADR-125 §1.4 "Tier 2 — Shortcuts-as-glue")

This directory ships the small set of glue you drop onto an always-on
Mac (like `ruv-mac-mini`) so RuView's BFLD-gated sensing events can
trigger native Apple Home actions — including HomePod announcements,
scene activations, cross-device notifications, and any third-party
HomeKit accessory the operator has paired.

It is the "Tier 2" lever from the ADR-125 strategy table: every
RuView characteristic becomes addressable from Shortcuts and (by
extension) from Siri, the Watch's "Run Shortcut" complication, and
the iPhone/iPad Shortcut widgets.

## Architecture

```
real C6 (192.168.1.179, ruv.net)
  → UDP feature_state → c6-presence-watcher.py → BFLD PrivacyGate
    → /tmp/ruview-last-feature.json
      → ruview-sensing-server.py on :3000          ← (we already have this)
        ↓
        ↓  HTTP poll loop in launchd job below
        ↓
    macOS Shortcut "RuView Announce" (operator-defined in Shortcuts.app)
      → action: "Speak Text on HomePod"
        → HomePod (any room) audibly announces the event ← Siri voice
```

The Shortcut itself lives in the operator's own Shortcuts library —
this directory provides only the trigger glue + the announcer script
that activates the Shortcut by name via `osascript`.

## One-time setup on the Mac

1. **Create the Shortcut** in `Shortcuts.app`:
   - Name: `RuView Announce`
   - Input: accepts text
   - Action: **Speak Text** (set target → your HomePod / HomePod mini)
   - Save

2. **Verify it runs from the command line**:
   ```sh
   osascript -e 'tell application "Shortcuts Events" to run shortcut "RuView Announce" with input "Test from RuView"'
   ```
   The HomePod should speak "Test from RuView".

3. **Install the launchd job**:
   ```sh
   cp ruview-watcher.plist ~/Library/LaunchAgents/com.ruvnet.ruview.watcher.plist
   launchctl load ~/Library/LaunchAgents/com.ruvnet.ruview.watcher.plist
   ```
   `launchctl list | grep ruvnet` should show the job loaded.

4. **Tail the log** while you walk past the C6 to verify it fires:
   ```sh
   tail -f /tmp/ruview-watcher.log
   ```

## Files

| File | Purpose |
|------|---------|
| `announce-via-homepod.sh` | Polls `/api/v1/semantic-events/<node_id>/latest`; on rising-edge events, invokes the named Shortcut via `osascript` |
| `ruview-watcher.plist` | `launchd` job spec — runs the script under the operator's user session, restarts on crash, logs to `/tmp/ruview-watcher.log` |

## Why launchd + osascript, not a daemon + AppleScriptObjC

- `launchd` is the macOS-native always-on supervisor; no Homebrew dep
- `osascript` is universally available on macOS; no extra install
- The Shortcut is operator-editable in Shortcuts.app — no code change
  to switch from "speak on HomePod" to "set scene" or "send message"

## Extending to multiple HomePods

Edit `RuView Announce` in Shortcuts.app:
- Add a "Choose from List" action with each HomePod target, OR
- Create per-room Shortcuts (`RuView Announce Kitchen`,
  `RuView Announce Bedroom`) and pass the room name into the
  script's `--shortcut-name` flag

The script supports `--shortcut-name <name>` so multiple watchers can
target different shortcuts per room without changing this code.

## Connection to ADR-125

This is the Tier 2 "Shortcuts-as-glue" implementation — it lets the
operator wire RuView events to anything Apple Home + Siri can do,
without needing the AirPlay 2 voice path (which is still blocked on
the router's mDNS reflection on Nighthawk MR60 firmware). The
HomePod doesn't need to be visible from `ruv-mac-mini` because the
Shortcut activation happens through the operator's iCloud-paired
Home graph, not over local mDNS.

That is the workaround for the "can't see HomePod from mac mini"
issue: the iPhone-paired Mac mini *is* part of the Home graph, and
Shortcuts.app uses that graph (not Bonjour) to reach the HomePod.

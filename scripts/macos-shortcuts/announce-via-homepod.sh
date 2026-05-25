#!/bin/bash
#
# announce-via-homepod.sh — ADR-125 §1.4 Tier 2 glue.
#
# Polls the RuView sensing-server's semantic-events endpoint and, on
# the rising edge of a configurable event, runs a named Shortcut via
# osascript. The Shortcut itself is owned by the operator in
# Shortcuts.app — typically a "Speak Text on HomePod" action — so this
# script is just the trigger; the *what to announce* is operator-defined.
#
# Run manually for testing:
#   bash announce-via-homepod.sh --node-id 12 --event unrecognized_activity_pattern
#
# Run as a launchd job: see ruview-watcher.plist + README.md.

set -euo pipefail

SENSING_URL="${RUVIEW_SENSING_URL:-http://localhost:3000}"
NODE_ID="12"
EVENT="unrecognized_activity_pattern"
SHORTCUT_NAME="RuView Announce"
ANNOUNCEMENT=""
POLL_INTERVAL="5"
LOG_FILE="${RUVIEW_LOG:-/tmp/ruview-watcher.log}"

usage() {
    cat >&2 <<EOF
Usage: $0 [options]

Options:
  --node-id <id>             Sensing-server node id (default: 12)
  --event <name>             Event to watch — one of:
                               unknown_presence
                               unexpected_occupancy
                               unrecognized_activity_pattern
                             (default: unrecognized_activity_pattern)
  --shortcut-name <name>     Shortcut to invoke (default: "RuView Announce")
  --announcement <text>      Text to speak when event fires (default: event name)
  --sensing-url <url>        Sensing-server base URL (default: http://localhost:3000)
  --poll-interval <s>        Poll interval in seconds (default: 5)
  --once                     Single poll + exit (for testing)
  -h, --help                 Show this help
EOF
}

ONCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-id) NODE_ID="$2"; shift 2 ;;
        --event) EVENT="$2"; shift 2 ;;
        --shortcut-name) SHORTCUT_NAME="$2"; shift 2 ;;
        --announcement) ANNOUNCEMENT="$2"; shift 2 ;;
        --sensing-url) SENSING_URL="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

ANNOUNCEMENT="${ANNOUNCEMENT:-$(echo "$EVENT" | tr '_' ' ')}"

run_shortcut() {
    local text="$1"
    if ! command -v osascript >/dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] ERROR: osascript not found — macOS-only" >> "$LOG_FILE"
        return 1
    fi
    # `Shortcuts Events` is the scriptable surface for Shortcuts.app.
    # Passing input via `with input "..."` requires the Shortcut to
    # have a "Receive Text input" trigger.
    osascript <<EOF >> "$LOG_FILE" 2>&1
tell application "Shortcuts Events"
    run shortcut "$SHORTCUT_NAME" with input "$text"
end tell
EOF
}

read_event_active() {
    # Returns "true" or "false" from the semantic-events endpoint.
    local node_id="$1" event="$2"
    curl -fsS --max-time 3 \
        "$SENSING_URL/api/v1/semantic-events/$node_id/latest" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); \
print(str(d.get('events',{}).get('$event',{}).get('active', False)).lower())" \
        2>/dev/null || echo "unknown"
}

last_state="unknown"
echo "[$(date '+%H:%M:%S')] start: node=$NODE_ID event=$EVENT shortcut=\"$SHORTCUT_NAME\"" \
    >> "$LOG_FILE"

while true; do
    current="$(read_event_active "$NODE_ID" "$EVENT")"
    if [[ "$current" != "$last_state" && "$current" == "true" ]]; then
        echo "[$(date '+%H:%M:%S')] $EVENT rising-edge → running '$SHORTCUT_NAME'" \
            >> "$LOG_FILE"
        run_shortcut "$ANNOUNCEMENT" || \
            echo "[$(date '+%H:%M:%S')] shortcut invocation failed" >> "$LOG_FILE"
    fi
    last_state="$current"
    [[ "$ONCE" == "1" ]] && break
    sleep "$POLL_INTERVAL"
done

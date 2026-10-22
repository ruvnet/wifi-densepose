#!/usr/bin/env bash
#
# Acceptance checks for the node configuration API, run against a real node.
#
#   ./config_api_acceptance.sh [node-url] [psk-file]
#
#   node-url   default http://192.168.0.115:8032  (or $RUVIEW_NODE)
#   psk-file   default $HOME/.ruview-ota-psk       (or $RUVIEW_PSK_FILE)
#
# Requires: curl, jq. Read-only except for the config values it sets.
# The PSK is read from a file and never echoed.
#
# The node must run a firmware built from a tree where the request is
# validated in full before anything is written (see ../main/config_rules.c).
# These checks fail against a firmware that writes a rejected field.

set -uo pipefail

NODE="${1:-${RUVIEW_NODE:-http://192.168.0.115:8032}}"
PSK_FILE="${2:-${RUVIEW_PSK_FILE:-$HOME/.ruview-ota-psk}}"

for tool in curl jq; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing dependency: $tool" >&2; exit 2; }
done
[ -r "$PSK_FILE" ] || { echo "cannot read psk file: $PSK_FILE" >&2; exit 2; }

PSK="$(tr -d '\r\n' < "$PSK_FILE")"
[ -n "$PSK" ] || { echo "psk file is empty: $PSK_FILE" >&2; exit 2; }

pass=0
fail=0

say() { # say <0|1> <message>
    if [ "$1" -eq 0 ]; then
        pass=$((pass + 1))
        echo "  [PASS] $2"
    else
        fail=$((fail + 1))
        echo "  [FAIL] $2"
    fi
}

api_get() { # api_get <path> -> body on stdout
    curl -sS --max-time 20 -H "Authorization: Bearer $PSK" "$NODE$1"
}

wait_for_node() { # wait_for_node <seconds> -> 0 when the node answers
    local deadline=$((SECONDS + $1))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -sS --max-time 5 -o /dev/null \
            -H "Authorization: Bearer $PSK" "$NODE/config" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

get_config() { # canonical (key-sorted) JSON of the stored config
    api_get /config | jq -e -S -c '.config'
}

post_config() { # post_config <json> -> "<http code>\t<body>"
    local tmp code body
    tmp="$(mktemp)" || return 1
    code="$(curl -sS --max-time 25 -o "$tmp" -w '%{http_code}' \
        -X POST -H "Authorization: Bearer $PSK" \
        -H 'Content-Type: application/json' \
        --data "$1" "$NODE/config" 2>/dev/null)" || true
    [ -n "$code" ] || code="000"
    body="$(tr -d '\r\n' < "$tmp")"
    rm -f "$tmp"
    printf '%s\t%s' "$code" "$body"
}

post() { # post <json> -> sets $code and $body
    local out
    out="$(post_config "$1")"
    IFS=$'\t' read -r code body <<<"$out"
}

echo "node: $NODE"

echo "=================================================================="
echo "TEST 1 - a rejected request must not change the node"
echo "=================================================================="
before="$(get_config)"
if [ -z "$before" ]; then
    echo "  [FAIL] the node did not answer a GET /config"
    exit 1
fi
echo "  before: $(jq -r '"ssid=\(.wifi_ssid) edge_tier=\(.edge_tier) csi_channel=\(.csi_channel) node_id=\(.node_id)"' <<<"$before")"

post '{"wifi_ssid":"wrong-network","edge_tier":"invalid"}'
echo "  POST mixed/type-error  -> HTTP $code  $body"
say "$([ "$code" = "400" ] && echo 0 || echo 1)" "mixed payload with a type error answers 400"

post '{"wifi_ssid":"wrong-network","edge_tier":9}'
echo "  POST mixed/range-error -> HTTP $code  $body"
say "$([ "$code" = "400" ] && echo 0 || echo 1)" "mixed payload with a range error answers 400"

after="$(get_config)"
echo "  after : $(jq -r '"ssid=\(.wifi_ssid) edge_tier=\(.edge_tier) csi_channel=\(.csi_channel) node_id=\(.node_id)"' <<<"$after")"
say "$([ "$before" = "$after" ] && echo 0 || echo 1)" "every stored value is unchanged after the rejected requests"

echo "=================================================================="
echo "TEST 3 - the API must not accept what the loader ignores"
echo "=================================================================="
post '{"vital_window":4096}'
echo "  POST vital_window=4096 -> HTTP $code  $body"
say "$([ "$code" = "400" ] && echo 0 || echo 1)" "vital_window=4096 is rejected (the loader only applies 32..256)"

post '{"top_k_count":64}'
echo "  POST top_k_count=64   -> HTTP $code  $body"
say "$([ "$code" = "400" ] && echo 0 || echo 1)" "top_k_count=64 is rejected (the loader only applies 1..32)"

post '{"channel_hop_count":14}'
echo "  POST channel_hop_count=14 -> HTTP $code  $body"
say "$([ "$code" = "400" ] && echo 0 || echo 1)" "channel_hop_count=14 is rejected (the channel list holds 6)"

after3="$(get_config)"
say "$([ "$before" = "$after3" ] && echo 0 || echo 1)" "the rejected out-of-range values were not written"

echo "  accepted boundaries (this reboots the node into the new values)"
post '{"vital_window":256,"top_k_count":32}'
echo "  POST vital_window=256,top_k_count=32 -> HTTP $code"
say "$([ "$code" = "200" ] && echo 0 || echo 1)" "the boundary values are accepted"

echo "=================================================================="
echo "TEST 2 - a second change must be refused while one is in flight"
echo "=================================================================="
sleep 6
wait_for_node 30 || echo "  (the node did not come back within 30 s; continuing)"
post '{"wifi_ssid":"ruview-acceptance-deadnet","trial_seconds":30}'
echo "  POST #1 (ssid -> a network that does not exist) -> HTTP $code"
say "$([ "$code" = "200" ] && echo 0 || echo 1)" "the first change is accepted"

post '{"edge_tier":1}'
echo "  POST #2 (within 3 s)                        -> HTTP $code  $body"
say "$([ "$code" = "409" ] && echo 0 || echo 1)" "the second change inside the window answers 409"

post '{"led_mode":2}'
echo "  POST #3 (still in flight)                   -> HTTP $code"
say "$([ "$code" = "409" ] && echo 0 || echo 1)" "a third change is refused as well"

echo "=================================================================="
echo "summary: $pass passed, $fail failed"
echo "=================================================================="
[ "$fail" -eq 0 ] || exit 1
exit 0

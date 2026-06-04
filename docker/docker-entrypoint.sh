#!/bin/sh
# Docker entrypoint for WiFi-DensePose sensing server.
#
# Supports two usage patterns:
#
# 1. No arguments — use defaults from environment:
#      docker run -e CSI_SOURCE=esp32 ruvnet/wifi-densepose:latest
#
# 2. Pass CLI flags directly:
#      docker run ruvnet/wifi-densepose:latest --source esp32 --tick-ms 500
#      docker run ruvnet/wifi-densepose:latest --model /app/models/my.rvf
#
# Environment variables:
#   CSI_SOURCE   — data source: auto (default), esp32, wifi, simulated
#   MODELS_DIR   — directory to scan for .rvf model files (default: data/models)
#   RUVIEW_API_TOKEN — bearer token for /api/v1/* when binding to the network
#   RUVIEW_ALLOW_UNAUTH_LAN=1 — explicit opt-in for unauthenticated LAN mode
set -e

is_truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

is_unspecified_bind_addr() {
    case "${1:-}" in
        0.0.0.0|::|\[::\]) return 0 ;;
        *) return 1 ;;
    esac
}

sensing_bind_addr() {
    bind="${SENSING_BIND_ADDR:-127.0.0.1}"
    expect_value=
    for arg in "$@"; do
        if [ "$expect_value" = "--bind-addr" ]; then
            bind="$arg"
            expect_value=
            continue
        fi
        case "$arg" in
            --bind-addr=*) bind="${arg#--bind-addr=}" ;;
            --bind-addr) expect_value="--bind-addr" ;;
        esac
    done
    printf '%s\n' "$bind"
}

guard_unauthenticated_network_bind() {
    case "${1:-}" in
        /app/sensing-server|sensing-server) ;;
        *) return 0 ;;
    esac

    bind_addr="$(sensing_bind_addr "$@")"
    if ! is_unspecified_bind_addr "$bind_addr"; then
        return 0
    fi
    if [ -n "${RUVIEW_API_TOKEN:-}" ]; then
        return 0
    fi
    if is_truthy "${RUVIEW_ALLOW_UNAUTH_LAN:-}"; then
        echo "WARN: starting unauthenticated LAN mode on ${bind_addr} because RUVIEW_ALLOW_UNAUTH_LAN=1" >&2
        return 0
    fi

    cat >&2 <<EOF
FATAL: refusing to start sensing-server on ${bind_addr} without RUVIEW_API_TOKEN.

The Docker image publishes the sensing HTTP/WebSocket surface. Set
RUVIEW_API_TOKEN to enforce bearer auth on /api/v1/*, or set
RUVIEW_ALLOW_UNAUTH_LAN=1 only for an intentionally trusted LAN deployment.
EOF
    exit 64
}

# Route to cog-ha-matter (ADR-116) when invoked as:
#   docker run <image> cog-ha-matter [--flags]
# or via the short alias `ha-matter`. Strips the keyword and execs the
# Home Assistant + Matter cog binary, defaulting --sensing-url to the
# co-located sensing-server endpoint so docker-compose deployments work
# out of the box.
case "${1:-}" in
    cog-ha-matter|ha-matter)
        shift
        exec /app/cog-ha-matter \
            --sensing-url "${SENSING_URL:-http://127.0.0.1:3000}" \
            "$@"
        ;;
    homecore|homecore-server)
        # Route to the HOMECORE native Rust port of Home Assistant
        # (ADRs 126-134, v0.10.0). Default bind matches HA at :8123.
        shift
        exec /app/homecore-server \
            --bind "${HOMECORE_BIND:-0.0.0.0:8123}" \
            "$@"
        ;;
esac

# If the first argument looks like a flag (starts with -), prepend the
# server binary so users can just pass flags:
#   docker run <image> --source esp32 --tick-ms 500
if [ "${1#-}" != "$1" ] || [ -z "$1" ]; then
    set -- /app/sensing-server \
        --source "${CSI_SOURCE:-auto}" \
        --tick-ms 100 \
        --ui-path /app/ui \
        --http-port 3000 \
        --ws-port 3001 \
        --bind-addr 0.0.0.0 \
        "$@"
fi

guard_unauthenticated_network_bind "$@"
exec "$@"

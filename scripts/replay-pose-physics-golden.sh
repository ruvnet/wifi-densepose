#!/usr/bin/env bash
set -euo pipefail
node "$(dirname "$0")/pose-physics/verify-golden.mjs" "$@"

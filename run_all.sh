#!/usr/bin/env bash
set -e

# Build the sensing server (wifi-densepose-api is a lib stub, no binary)
cd rust-port/wifi-densepose-rs
cargo build --release -p wifi-densepose-sensing-server

# Start sensing server (serves REST + WebSocket + UI)
cargo run --release -p wifi-densepose-sensing-server &
SENSE_PID=$!

echo "Sensing server PID=$SENSE_PID"
echo "Open: http://localhost:8080/ui/index.html"

# Keep shell alive
wait $SENSE_PID

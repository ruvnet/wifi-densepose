.PHONY: all build clean test scan run run-verbose listen install-helper help

HELPER := mac_wifi
BRIDGE := target/release/macos-rssi-bridge

# Sensing-server expectations.
TARGET_HOST ?= 127.0.0.1
TARGET_PORT ?= 5005
INTERVAL    ?= 1.5

all: build

help:
	@echo "macos-rssi-bridge — Mac WiFi card → RuView sensing-server bridge"
	@echo ""
	@echo "  make build         compile mac_wifi (Swift) + macos-rssi-bridge (Rust)"
	@echo "  make scan          run a single multi-BSSID scan, print JSON"
	@echo "  make run           start the bridge → udp://$(TARGET_HOST):$(TARGET_PORT) + http://localhost:9090"
	@echo "  make run-verbose   same, with per-frame stats on stderr"
	@echo "  make dashboard     open the multistatic RF tomography UI in your browser"
	@echo "  make listen        debug: print frames arriving on $(TARGET_PORT) (no sensing-server)"
	@echo "  make test          run Rust unit tests"
	@echo "  make clean         delete build artifacts"
	@echo ""
	@echo "Variables: TARGET_HOST=$(TARGET_HOST) TARGET_PORT=$(TARGET_PORT) INTERVAL=$(INTERVAL)"

build: $(HELPER) $(BRIDGE)

$(HELPER): mac_wifi.swift
	swiftc -O mac_wifi.swift -o $(HELPER)

$(BRIDGE): src/main.rs Cargo.toml
	cargo build --release

test:
	cargo test --release

scan: $(HELPER)
	./$(HELPER) --scan-once

run: build
	./$(BRIDGE) --helper ./$(HELPER) --target-host $(TARGET_HOST) \
		--target-port $(TARGET_PORT) --interval $(INTERVAL)

run-verbose: build
	./$(BRIDGE) --helper ./$(HELPER) --target-host $(TARGET_HOST) \
		--target-port $(TARGET_PORT) --interval $(INTERVAL) --verbose

# Open the RF tomography dashboard. Assumes the bridge is already running
# (start it with `make run` or `make run-verbose` in another terminal).
dashboard:
	@open http://localhost:9090/dashboard 2>/dev/null || \
	  echo "Open http://localhost:9090/dashboard in your browser"

# Debug helper — prints magic/seq/rssi for each frame the bridge emits.
# Doesn't require sensing-server.
listen:
	@python3 listen.py $(TARGET_PORT)

# Optional: install the Swift helper system-wide so anything on $PATH can find it,
# matching the default lookup of `MacosCoreWlanScanner::new()` in the Rust crate.
install-helper: $(HELPER)
	install -m 0755 $(HELPER) /usr/local/bin/mac_wifi
	@echo "[install] /usr/local/bin/mac_wifi"

clean:
	rm -f $(HELPER)
	cargo clean

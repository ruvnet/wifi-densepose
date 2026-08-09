# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Host build-CHECK for the OpenWRT/mac80211 VEIL adapter.
# STATUS: SYNTHETIC / L0 — build-only, UNTESTED ON HARDWARE.
#
# Two targets:
#   make core        - compile+link the portable core only (always works,
#                      no libnl needed) — proves the rotation math builds.
#   make daemon      - build veil_shieldd against libnl-genl-3 (needs the
#                      dev headers: `pkg-config libnl-genl-3.0`). On OpenWRT
#                      the package build uses libnl-tiny instead (see openwrt.mk).
#
# This Makefile does NOT flash, run on, or validate any radio.

CC      ?= cc
COREDIR := ../core
CFLAGS  ?= -std=c99 -Wall -Wextra -O2 -I$(COREDIR)
LDLIBS  ?= -lm

NL_CFLAGS := $(shell pkg-config --cflags libnl-genl-3.0 2>/dev/null)
NL_LIBS   := $(shell pkg-config --libs   libnl-genl-3.0 2>/dev/null)

.PHONY: all core daemon clean
all: core

# Always-buildable: the core object, no netlink dependency.
core: $(COREDIR)/veil_shield.c $(COREDIR)/veil_shield.h
	$(CC) $(CFLAGS) -c $(COREDIR)/veil_shield.c -o veil_shield.o
	@echo "core built (rotation math OK). Nothing was run on hardware."

# Full daemon: requires libnl-genl-3 dev headers on the host.
daemon: veil_shieldd.c core
ifeq ($(strip $(NL_LIBS)),)
	@echo "SKIP daemon: libnl-genl-3.0 not found (pkg-config)."
	@echo "  Install libnl-3-dev + libnl-genl-3-dev, or build via openwrt.mk."
	@exit 0
else
	$(CC) $(CFLAGS) $(NL_CFLAGS) -o veil_shieldd \
	    veil_shieldd.c veil_shield.o $(NL_LIBS) $(LDLIBS)
	@echo "veil_shieldd linked (BUILD-ONLY; untested on silicon)."
endif

clean:
	rm -f veil_shield.o veil_shieldd

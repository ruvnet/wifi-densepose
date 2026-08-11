# SPDX-License-Identifier: MIT OR Apache-2.0
#
# OpenWRT package Makefile STUB for veil_shieldd.
# STATUS: SYNTHETIC / L0 — package skeleton, UNTESTED ON HARDWARE / not in any feed.
#
# Drop this (renamed to `Makefile`) into a package dir such as
# `package/utils/veil-shieldd/` in an OpenWRT buildroot, alongside the copied
# core (veil_shield.{c,h}) and veil_shieldd.c under ./src/. It builds against
# libnl-tiny (the OpenWRT netlink lib) — the same nl80211 API surface, smaller.
#
# This stub does NOT prove the daemon works on a device; it only wires the
# build. No hardware validation is implied.

include $(TOPDIR)/rules.mk

PKG_NAME:=veil-shieldd
PKG_VERSION:=0.0.0-l0
PKG_RELEASE:=1
PKG_LICENSE:=MIT OR Apache-2.0

include $(INCLUDE_DIR)/package.mk

define Package/veil-shieldd
  SECTION:=utils
  CATEGORY:=Utilities
  TITLE:=VEIL compliant-waveform privacy shield (mac80211 adapter, L0)
  # libnl-tiny provides nl80211/genl; hostapd for the ctrl_iface cadence path.
  DEPENDS:=+libnl-tiny +hostapd-common
  URL:=https://github.com/ruvnet/RuView
endef

define Package/veil-shieldd/description
  BUILD-ONLY / UNTESTED-ON-HARDWARE userspace adapter that drives the
  standards-compliant subset of VEIL controls reachable from OpenWRT
  (TX antenna map, hostapd-mediated sounding cadence) and links the portable
  keyed-rotation core. The full per-packet keyed rotation is blob-blocked on
  commodity Qualcomm/MediaTek parts and requires a driver/firmware patch.
  This is NOT a jammer and emits no denial energy.
endef

# Build flags: point at libnl-tiny headers and the copied core.
TARGET_CFLAGS += -I$(STAGING_DIR)/usr/include/libnl-tiny -I$(PKG_BUILD_DIR)/src
TARGET_LDFLAGS += -lnl-tiny -lm

define Build/Compile
	$(TARGET_CC) $(TARGET_CFLAGS) -std=c99 -Wall -Wextra \
		-o $(PKG_BUILD_DIR)/veil_shieldd \
		$(PKG_BUILD_DIR)/src/veil_shieldd.c \
		$(PKG_BUILD_DIR)/src/veil_shield.c \
		$(TARGET_LDFLAGS)
endef

define Package/veil-shieldd/install
	$(INSTALL_DIR) $(1)/usr/sbin
	$(INSTALL_BIN) $(PKG_BUILD_DIR)/veil_shieldd $(1)/usr/sbin/veil_shieldd
	# TODO(hw): ship a procd init script that reads the session key from a
	# secure store (never a world-readable config) and passes -i <ifindex>.
endef

$(eval $(call BuildPackage,veil-shieldd))

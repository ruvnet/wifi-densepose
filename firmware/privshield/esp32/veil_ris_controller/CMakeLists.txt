# veil_ris_controller — ESP-IDF component (SYNTHETIC / L0, build-only)
#
# Drives an EXTERNAL reconfigurable intelligent surface (RIS) over GPIO/SPI to
# scramble the *sensing-direction* channel while preserving the *comm-direction*
# channel (the PrivISAC pattern, arXiv:2601.04488). This is the honest way an
# ESP32 "helps scramble": through an external passive surface, NOT its own
# closed Wi-Fi PHY. See the subdir README.md.
#
# The keyed configuration schedule reuses the portable VEIL core's SplitMix64
# `veil_rng` (../../core/veil_shield.{h,c}) so the schedule is deterministic and
# byte-consistent with the Rust reference — the same key can be shared with an
# associated receiver.
#
# NOTE: build-only skeleton, never run on silicon. Hardware paths -> TODO(hw).

set(VEIL_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../core")

idf_component_register(
    SRCS "veil_ris_controller.c"
         "${VEIL_CORE_DIR}/veil_shield.c"   # reuse veil_rng from the portable core
    INCLUDE_DIRS "include" "${VEIL_CORE_DIR}"
    REQUIRES esp_timer esp_driver_gpio esp_driver_spi
)

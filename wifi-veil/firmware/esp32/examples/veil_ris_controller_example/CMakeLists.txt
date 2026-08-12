# veil_ris_controller_example — SYNTHETIC / L0, build-only.
#
# Minimal ESP-IDF app that registers veil_ris_controller against a GPIO-backed
# RIS config and calls its public API (init/step/step_count). Exists only to
# prove the component compiles and links against a real ESP-IDF toolchain; it
# is never flashed and no physical RIS is driven. See ../../README.md.

cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../veil_ris_controller")

project(veil_ris_controller_example)

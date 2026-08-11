# veil_sensing_detector_example — SYNTHETIC / L0, build-only.
#
# Minimal ESP-IDF app that registers veil_sensing_detector with the GPIO
# trigger backend and calls its public API. Exists only to prove the
# component compiles and links against a real ESP-IDF toolchain; it is never
# flashed and no CSI is ever captured. See ../../README.md.

cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../veil_sensing_detector")

project(veil_sensing_detector_example)

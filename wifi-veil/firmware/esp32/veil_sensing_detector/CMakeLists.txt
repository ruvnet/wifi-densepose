# veil_sensing_detector — ESP-IDF component (SYNTHETIC / L0, build-only)
#
# Estimates the 802.11 sensing-solicitation rate from the ESP32 CSI callback
# and raises a trigger (GPIO / MQTT / ESP-NOW) that engages the AP-side VEIL
# shield. This component only READS the channel; it never shapes RF. See the
# subdir README.md for the honest capability boundary.
#
# NOTE: This is a build-only skeleton. It has never run on silicon. All
#       hardware-touching paths are marked TODO(hw).

idf_component_register(
    SRCS "veil_sensing_detector.c"
    INCLUDE_DIRS "include"
    # esp_wifi: esp_wifi_set_csi_rx_cb / esp_wifi_set_csi / promiscuous.
    # The MQTT and ESP-NOW trigger backends are optional; they are only
    # referenced under CONFIG_ guards so the core build stays minimal.
    REQUIRES esp_wifi esp_event esp_timer esp_driver_gpio
    PRIV_REQUIRES esp_mqtt
)

/**
 * @file ble_identity.h
 * @brief Opt-in, privacy-minimized BLE identity-anchor scanner.
 *
 * ESP32-S3 BLE scanning provides advertising metadata and RSSI.  It does not
 * expose raw CTE IQ or Bluetooth Channel Sounding measurements.  This module
 * therefore accepts only authenticated RuView service tokens and emits a
 * short-lived pseudonym; it makes no civil-identity or vital-sign claim.
 */

#ifndef BLE_IDENTITY_H
#define BLE_IDENTITY_H

#include <stdbool.h>
#include "esp_err.h"

/** Start the bounded-duty passive scanner, or return NOT_SUPPORTED/off. */
esp_err_t ble_identity_init(void);

/** True only while the controller has an active extended scan procedure. */
bool ble_identity_is_healthy(void);

#endif /* BLE_IDENTITY_H */

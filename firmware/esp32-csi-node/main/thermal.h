/* On-chip thermal monitoring and radio throttling. See thermal.c for why the
 * thresholds are set against die temperature rather than the plastic. */
#pragma once

#include "esp_err.h"
#include <stdint.h>

typedef enum {
    THERMAL_OK = 0,
    THERMAL_WARN,
    THERMAL_THROTTLED,
    THERMAL_CRITICAL,
} thermal_state_t;

#if defined(CONFIG_THERMAL_MONITOR) && defined(CONFIG_IDF_TARGET_ESP32C6)

esp_err_t thermal_init(void);

/* Sample and apply policy. Call from a slow loop -- the die does not change
 * fast, and reading it more often buys nothing. */
void thermal_tick(void);

float           thermal_celsius(void);
float           thermal_peak_celsius(void);
thermal_state_t thermal_state(void);

/* Current WiFi transmit ceiling in whole dBm.
 *
 * Exposed because a throttled node's RSSI falls at both ends, which is
 * indistinguishable from an obstruction in the link diagnostics. Anything
 * reading link strength should check this before concluding a wall moved. */
int8_t          thermal_tx_dbm(void);

const char     *thermal_state_name(thermal_state_t s);

#else  /* stubs: S3 build and when monitoring is off */

static inline esp_err_t thermal_init(void) { return ESP_OK; }
static inline void      thermal_tick(void) {}
static inline float     thermal_celsius(void) { return -273.0f; }
static inline float     thermal_peak_celsius(void) { return -273.0f; }
static inline thermal_state_t thermal_state(void) { return THERMAL_OK; }
static inline int8_t    thermal_tx_dbm(void) { return 20; }
static inline const char *thermal_state_name(thermal_state_t s) { (void)s; return "n/a"; }

#endif

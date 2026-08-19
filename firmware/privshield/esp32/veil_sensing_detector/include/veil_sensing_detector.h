/* SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * veil_sensing_detector — detect 802.11 sensing solicitation and raise a
 * trigger that engages the AP-side VEIL shield.
 *
 * STATUS: SYNTHETIC / L0. Build-only ESP-IDF component skeleton. Never flashed,
 * never captured on silicon. Do NOT claim runtime behavior without a captured
 * hardware log (CLAUDE.md hardware-evidence rule).
 *
 * ROLE (honest): the ESP32 is a passive CSI *observer* here. It watches how
 * often it is being sounded / probed (NDP announcements, action frames, and the
 * cadence of incoming CSI-bearing frames) and, when that rate crosses a
 * threshold, tells a *separate* protector (the AP running the veil_shield core)
 * that a sensing burst is in progress. The ESP32 does NOT modify any waveform
 * and does NOT protect its own beamforming feedback (see README.md). This is
 * the strongest, clearly-compliant supporting role for the ESP32.
 */
#ifndef VEIL_SENSING_DETECTOR_H
#define VEIL_SENSING_DETECTOR_H

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* How the detector announces "sensing burst detected" to the protector. */
typedef enum {
    VEIL_TRIGGER_GPIO = 0, /* drive a GPIO line to a co-located AP / relay   */
    VEIL_TRIGGER_MQTT,     /* publish to a broker the AP subscribes to        */
    VEIL_TRIGGER_ESPNOW,   /* connectionless ESP-NOW unicast to the AP node   */
} veil_trigger_backend_t;

typedef struct {
    /* Sliding-window length for the solicitation-rate estimate, milliseconds. */
    uint32_t window_ms;
    /* Solicitations/second above which the shield should be engaged. */
    float    trigger_rate_hz;
    /* Hysteresis: rate must fall below this to clear the trigger. */
    float    release_rate_hz;

    veil_trigger_backend_t backend;

    /* GPIO backend. */
    int      gpio_num;            /* output line; active-high engage          */

    /* MQTT backend. broker_uri/topic are borrowed, must outlive the detector. */
    const char *mqtt_broker_uri;  /* e.g. "mqtts://ap.local:8883"             */
    const char *mqtt_topic;       /* e.g. "veil/engage"                       */

    /* ESP-NOW backend. */
    uint8_t  espnow_peer[6];      /* AP node MAC                              */
} veil_sensing_detector_cfg_t;

/* Sensible SYNTHETIC defaults (not silicon-validated). */
#define VEIL_SENSING_DETECTOR_DEFAULT_CFG() \
    (veil_sensing_detector_cfg_t){          \
        .window_ms        = 1000,           \
        .trigger_rate_hz  = 20.0f,          \
        .release_rate_hz  = 5.0f,           \
        .backend          = VEIL_TRIGGER_GPIO, \
        .gpio_num         = -1,             \
        .mqtt_broker_uri  = NULL,           \
        .mqtt_topic       = "veil/engage",  \
        .espnow_peer      = {0},            \
    }

/* Install the CSI callback + configured trigger backend. Enables promiscuous
 * CSI capture. Returns ESP_OK on successful *registration* only — this says
 * nothing about on-air behavior. */
esp_err_t veil_sensing_detector_start(const veil_sensing_detector_cfg_t *cfg);

/* Tear down callback + backend. */
esp_err_t veil_sensing_detector_stop(void);

/* Last estimated solicitation rate (Hz), for telemetry/tests. */
float veil_sensing_detector_rate_hz(void);

/* True while the engage trigger is asserted. */
bool veil_sensing_detector_engaged(void);

#ifdef __cplusplus
}
#endif

#endif /* VEIL_SENSING_DETECTOR_H */

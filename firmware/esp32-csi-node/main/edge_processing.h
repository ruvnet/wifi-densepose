/**
 * @file edge_processing.h
 * @brief ADR-039 Edge Intelligence — dual-core CSI processing pipeline.
 *
 * Core 0 (WiFi): Produces CSI frames into a lock-free SPSC ring buffer.
 * Core 1 (DSP):  Consumes frames, runs signal processing, extracts vitals.
 *
 * Features:
 *   - Biquad IIR bandpass filters for breathing (0.1-0.5 Hz) and heart rate (0.8-2.0 Hz)
 *   - Phase unwrapping and Welford running statistics
 *   - Top-K subcarrier selection by variance
 *   - Presence detection with adaptive threshold calibration
 *   - Vital signs: breathing rate, heart rate (zero-crossing BPM)
 *   - Fall detection (phase acceleration exceeds threshold)
 *   - Delta compression (XOR + RLE) for bandwidth reduction
 *   - Multi-person vitals via subcarrier group clustering
 *   - 32-byte vitals packet (magic 0xC5110002) for server-side parsing
 */

#ifndef EDGE_PROCESSING_H
#define EDGE_PROCESSING_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

/* ---- Magic numbers ---- */
#define EDGE_VITALS_MAGIC     0xC5110002  /**< Vitals packet magic. */
#define EDGE_COMPRESSED_MAGIC 0xC5110005  /**< Compressed frame magic (was 0xC5110003, reassigned for ADR-069). */

/* ---- Buffer sizes ---- */
#define EDGE_RING_SLOTS       16    /**< SPSC ring buffer slots (power of 2). */
#define EDGE_MAX_IQ_BYTES     1024  /**< Max I/Q payload per slot. */
#define EDGE_PHASE_HISTORY_LEN 256  /**< Phase history buffer depth. */
#define EDGE_TOP_K            8     /**< Top-K subcarriers to track. */
#define EDGE_MAX_SUBCARRIERS  128   /**< Max subcarriers per frame. */

/* ---- Multi-person ---- */
#define EDGE_MAX_PERSONS      4     /**< Max simultaneous persons. */

/* ---- Multi-person counting gates (issue #998) ----
 *
 * Over-counting root cause: the multi-person path used to split the top-K
 * subcarriers into EDGE_MAX_PERSONS groups and mark EVERY group active,
 * so one body's multipath always reported the full EDGE_MAX_PERSONS. These
 * gates promote a subcarrier group to a real "person" only when it carries
 * genuine, distinct, persistent energy:
 *
 *   1. Energy gate   — a group's phase variance must exceed a fraction of the
 *                      strongest group's variance, else it is multipath/noise.
 *   2. Spatial dedup — two groups whose representative subcarriers sit within
 *                      EDGE_PERSON_MIN_SC_SEP of each other are the same body
 *                      (adjacent subcarriers see correlated reflections), so
 *                      the weaker one is merged away.
 *   3. Persistence   — a candidate count must hold for EDGE_PERSON_PERSIST_FRAMES
 *                      consecutive decisions before it is emitted, so a single
 *                      noisy frame cannot promote a phantom person.
 *
 * These are robustness gates on the existing heuristic, not a calibrated
 * occupancy model — true count accuracy vs ground truth remains data-gated. */
#define EDGE_PERSON_MIN_ENERGY_RATIO 0.35f /**< Group var must be >= this * max group var to count. */
#define EDGE_PERSON_MIN_SC_SEP       4     /**< Min subcarrier separation between distinct persons. */
#define EDGE_PERSON_PERSIST_FRAMES   3     /**< Consecutive decisions a count must hold before emit. */

/* ---- Calibration ---- */
#define EDGE_CALIB_FRAMES     1200  /**< Frames for adaptive calibration (~60s at 20 Hz). */
#define EDGE_CALIB_SIGMA_MULT 3.0f  /**< Threshold = mean + 3*sigma of ambient. */

/* ---- Fall detection ---- */
#define EDGE_FALL_COOLDOWN_MS 5000  /**< Minimum ms between fall alerts (debounce). */
#define EDGE_FALL_CONSEC_MIN  3     /**< Consecutive frames above threshold to trigger. */

/* ---- Presence flag hysteresis + debounce (issue #996) ----
 *
 * Flicker root cause: the presence flag was a single-threshold compare on a
 * noisy presence_score (observed 2.6-26.7 frame-to-frame for one stationary
 * person), so the boolean chattered at the boundary even while the score
 * clearly indicated a person. Fix: Schmitt-trigger hysteresis plus a clear
 * debounce.
 *
 *   - Assert  presence when score >  threshold              (enter immediately).
 *   - Hold    presence while score >= threshold * HYST_RATIO (no flicker in the
 *                                                            gap band).
 *   - Clear   presence only after the score stays below the low threshold for
 *             EDGE_PRESENCE_CLEAR_FRAMES consecutive frames (genuine departure).
 *
 * HYST_RATIO < 1.0 sets the low threshold below the high threshold; a wider gap
 * (smaller ratio) is more flicker-immune but slower to clear on real exit. The
 * exact ratio that best matches a given room's score scale remains an on-device
 * tuning parameter — this removes the logic bug (no hysteresis at all). */
#define EDGE_PRESENCE_HYST_RATIO  0.5f /**< Low thresh = HYST_RATIO * high thresh. */
#define EDGE_PRESENCE_CLEAR_FRAMES 5   /**< Frames below low thresh before clearing. */

/* ---- DSP task tuning ---- */
#define EDGE_BATCH_LIMIT      4     /**< Max frames per batch before longer yield. */

/* ---- SPSC ring buffer slot ---- */
typedef struct {
    uint8_t  iq_data[EDGE_MAX_IQ_BYTES]; /**< Raw I/Q bytes from CSI callback. */
    uint16_t iq_len;                     /**< Actual I/Q data length. */
    int8_t   rssi;                       /**< RSSI from rx_ctrl. */
    uint8_t  channel;                    /**< WiFi channel. */
    uint32_t timestamp_us;               /**< Microsecond timestamp. */
} edge_ring_slot_t;

/* ---- SPSC ring buffer ---- */
typedef struct {
    edge_ring_slot_t slots[EDGE_RING_SLOTS];
    volatile uint32_t head;  /**< Written by producer (Core 0). */
    volatile uint32_t tail;  /**< Written by consumer (Core 1). */
} edge_ring_buf_t;

/* ---- Biquad IIR filter state ---- */
typedef struct {
    float b0, b1, b2;  /**< Numerator coefficients. */
    float a1, a2;       /**< Denominator coefficients (a0 = 1). */
    float x1, x2;       /**< Input delay line. */
    float y1, y2;       /**< Output delay line. */
} edge_biquad_t;

/* ---- Welford running statistics ---- */
typedef struct {
    double mean;
    double m2;
    uint32_t count;
} edge_welford_t;

/* ---- Per-person vitals state (multi-person mode) ---- */
typedef struct {
    float    phase_history[EDGE_PHASE_HISTORY_LEN];
    uint16_t history_len;
    uint16_t history_idx;
    float    breathing_bpm;
    float    heartrate_bpm;
    uint8_t  subcarrier_idx;  /**< Which subcarrier group this person tracks. */
    bool     active;
} edge_person_vitals_t;

/* ---- Vitals packet (32 bytes, wire format) ---- */
typedef struct __attribute__((packed)) {
    uint32_t magic;          /**< EDGE_VITALS_MAGIC = 0xC5110002. */
    uint8_t  node_id;        /**< ESP32 node identifier. */
    uint8_t  flags;          /**< Bit0=presence, Bit1=fall, Bit2=motion. */
    uint16_t breathing_rate; /**< BPM * 100 (fixed-point). */
    uint32_t heartrate;      /**< BPM * 10000 (fixed-point). */
    int8_t   rssi;           /**< Latest RSSI. */
    uint8_t  n_persons;      /**< Number of detected persons (multi-person). */
    uint8_t  reserved[2];
    float    motion_energy;  /**< Phase variance / motion metric. */
    float    presence_score; /**< Presence detection score. */
    uint32_t timestamp_ms;   /**< Milliseconds since boot. */
    uint32_t reserved2;      /**< Reserved for future use. */
} edge_vitals_pkt_t;

_Static_assert(sizeof(edge_vitals_pkt_t) == 32, "vitals packet must be 32 bytes");

/* ---- ADR-069: CSI Feature Vector packet (48 bytes, wire format) ---- */
#define EDGE_FEATURE_MAGIC  0xC5110003  /**< Feature vector packet magic. */

typedef struct __attribute__((packed)) {
    uint32_t magic;          /**< EDGE_FEATURE_MAGIC = 0xC5110003. */
    uint8_t  node_id;        /**< ESP32 node identifier. */
    uint8_t  reserved;       /**< Alignment padding. */
    uint16_t seq;            /**< Sequence number. */
    int64_t  timestamp_us;   /**< Microseconds since boot. */
    float    features[8];    /**< 8-dim normalized feature vector. */
} edge_feature_pkt_t;

_Static_assert(sizeof(edge_feature_pkt_t) == 48, "feature packet must be 48 bytes");

/* ---- ADR-063: Fused vitals packet (48 bytes, wire format) ---- */
#define EDGE_FUSED_MAGIC  0xC5110004  /**< Fused vitals packet magic. */

typedef struct __attribute__((packed)) {
    /* First 32 bytes match edge_vitals_pkt_t layout */
    uint32_t magic;          /**< EDGE_FUSED_MAGIC = 0xC5110004. */
    uint8_t  node_id;
    uint8_t  flags;          /**< Bit0=presence, Bit1=fall, Bit2=motion, Bit3=mmwave_present. */
    uint16_t breathing_rate; /**< Fused BPM * 100 (CSI + mmWave Kalman). */
    uint32_t heartrate;      /**< Fused BPM * 10000. */
    int8_t   rssi;
    uint8_t  n_persons;
    uint8_t  mmwave_type;    /**< mmwave_type_t enum. */
    uint8_t  fusion_confidence; /**< 0-100 fusion quality score. */
    float    motion_energy;
    float    presence_score;
    uint32_t timestamp_ms;
    /* mmWave extension (16 bytes) */
    float    mmwave_hr_bpm;  /**< Raw mmWave heart rate. */
    float    mmwave_br_bpm;  /**< Raw mmWave breathing rate. */
    float    mmwave_distance;/**< Distance to nearest target (cm). */
    uint8_t  mmwave_targets; /**< Target count from mmWave. */
    uint8_t  mmwave_confidence; /**< mmWave signal quality 0-100. */
    uint16_t reserved3;
    uint32_t reserved4;     /**< Pad to 48 bytes for alignment. */
} edge_fused_vitals_pkt_t;

_Static_assert(sizeof(edge_fused_vitals_pkt_t) == 48, "fused vitals must be 48 bytes");

/* ---- Edge configuration (from NVS) ---- */
typedef struct {
    uint8_t  tier;           /**< Processing tier: 0=raw, 1=basic, 2=full. */
    float    presence_thresh;/**< Presence detection threshold (0 = auto-calibrate). */
    float    fall_thresh;    /**< Fall detection threshold (phase accel, rad/s^2). */
    uint16_t vital_window;   /**< Phase history window for BPM estimation. */
    uint16_t vital_interval_ms; /**< Vitals packet send interval in ms. */
    uint8_t  top_k_count;    /**< Number of top subcarriers to track. */
    uint8_t  power_duty;     /**< Power duty cycle percentage (10-100). */
} edge_config_t;

/**
 * Initialize the edge processing pipeline.
 * Creates the SPSC ring buffer and starts the DSP task on Core 1.
 *
 * @param cfg  Edge configuration (from NVS or defaults).
 * @return ESP_OK on success.
 */
esp_err_t edge_processing_init(const edge_config_t *cfg);

/**
 * Enqueue a CSI frame from the WiFi callback (Core 0).
 * Lock-free SPSC push — safe to call from ISR context.
 *
 * @param iq_data   Raw I/Q data from wifi_csi_info_t.buf.
 * @param iq_len    Length of I/Q data in bytes.
 * @param rssi      RSSI from rx_ctrl.
 * @param channel   WiFi channel number.
 * @return true if enqueued, false if ring buffer is full (frame dropped).
 */
bool edge_enqueue_csi(const uint8_t *iq_data, uint16_t iq_len,
                      int8_t rssi, uint8_t channel);

/**
 * Get the latest vitals packet (thread-safe copy).
 *
 * @param pkt  Output vitals packet.
 * @return true if valid vitals data is available.
 */
bool edge_get_vitals(edge_vitals_pkt_t *pkt);

/**
 * Get multi-person vitals array.
 *
 * @param persons   Output array (must be EDGE_MAX_PERSONS elements).
 * @param n_active  Output: number of active persons.
 */
void edge_get_multi_person(edge_person_vitals_t *persons, uint8_t *n_active);

/**
 * Get pointer to the phase history ring buffer and its state.
 * Used by WASM runtime (ADR-040) to expose phase history to modules.
 *
 * @param out_buf     Output: pointer to phase history array.
 * @param out_len     Output: number of valid entries.
 * @param out_idx     Output: current write index.
 */
void edge_get_phase_history(const float **out_buf, uint16_t *out_len,
                            uint16_t *out_idx);

/**
 * Get per-subcarrier Welford variance array.
 * Used by WASM runtime (ADR-040) to expose variances to modules.
 *
 * @param out_variances  Output array (must be EDGE_MAX_SUBCARRIERS elements).
 * @param n_subcarriers  Number of subcarriers to fill.
 */
void edge_get_variances(float *out_variances, uint16_t n_subcarriers);

#endif /* EDGE_PROCESSING_H */

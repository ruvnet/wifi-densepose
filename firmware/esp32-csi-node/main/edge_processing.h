/**
 * @file edge_processing.h
 * @brief ADR-039 Edge Intelligence — on-device CSI processing.
 *
 * Phase 1 + Tier 1: Phase sanitization, Welford running statistics,
 * subcarrier selection, and delta compression on the ESP32-S3.
 *
 * Tier 2 (optional): Presence detection, vital signs extraction,
 * motion scoring, and fall detection.
 *
 * Design:
 *   - Lock-free SPSC ring buffer (Core 0 produces, Core 1 consumes).
 *   - FreeRTOS task pinned to Core 1 for DSP.
 *   - All static allocation, no malloc in hot path.
 *   - edge_tier=0 disables edge processing (existing behavior preserved).
 */

#ifndef EDGE_PROCESSING_H
#define EDGE_PROCESSING_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_wifi_types.h"

/* ------------------------------------------------------------------ */
/* Ring buffer configuration                                          */
/* ------------------------------------------------------------------ */

/** Ring buffer capacity (must be power of 2). */
#define EDGE_RING_SIZE 64

/** Maximum I/Q data length per CSI frame (4 antennas * 256 subcarriers * 2). */
#define EDGE_MAX_IQ_LEN 384

/** Ring buffer entry — copied from the CSI callback on Core 0. */
typedef struct {
    int8_t   iq_data[EDGE_MAX_IQ_LEN];
    uint16_t iq_len;
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  channel;
    uint8_t  tx_mac[6];
    uint32_t timestamp_ms;
} edge_csi_entry_t;

/* ------------------------------------------------------------------ */
/* Tier 1: Phase sanitization and subcarrier selection                 */
/* ------------------------------------------------------------------ */

/** Maximum subcarriers we track (HT40 = 128 subcarriers, with margin). */
#define EDGE_MAX_SUBCARRIERS 192

/** Per-subcarrier running statistics for phase unwrap and Welford. */
typedef struct {
    float    phase_prev[EDGE_MAX_SUBCARRIERS];   /**< Previous phase for unwrap. */
    float    amp_mean[EDGE_MAX_SUBCARRIERS];     /**< Welford running mean of amplitude. */
    float    amp_m2[EDGE_MAX_SUBCARRIERS];       /**< Welford M2 accumulator. */
    uint32_t amp_count;                           /**< Total sample count. */
    int8_t   prev_iq[EDGE_MAX_IQ_LEN];           /**< Previous I/Q frame for delta compression. */
    bool     has_prev;                            /**< True after first frame received. */
} edge_tier1_state_t;

/* ------------------------------------------------------------------ */
/* Tier 2: Vital signs and presence detection                         */
/* ------------------------------------------------------------------ */

/** Phase history depth: 15 seconds at 20 Hz. */
#define EDGE_PHASE_HISTORY_LEN 300

/** Variance history depth for fall detection. */
#define EDGE_VAR_HISTORY_LEN 20

typedef struct {
    float    phase_history[EDGE_PHASE_HISTORY_LEN]; /**< Ring buffer of phases for vital signs. */
    uint16_t history_len;           /**< Number of valid entries. */
    uint16_t history_idx;           /**< Current write index. */
    float    breathing_bpm;         /**< Estimated breathing rate (BPM). */
    float    heartrate_bpm;         /**< Estimated heart rate (BPM). */
    float    breathing_confidence;  /**< Confidence [0..1]. */
    float    heartrate_confidence;  /**< Confidence [0..1]. */
    uint8_t  presence;              /**< 0=empty, 1=present, 2=moving. */
    uint8_t  motion_score;          /**< 0-255 motion intensity. */
    uint8_t  occupancy;             /**< Estimated occupant count (0-8). */
    uint8_t  fall_detected;         /**< 1 if fall detected in current window. */
    float    variance_history[EDGE_VAR_HISTORY_LEN]; /**< Recent variance for fall detection. */
    uint8_t  var_idx;               /**< Write index into variance_history. */
} edge_tier2_state_t;

/* ------------------------------------------------------------------ */
/* Vitals UDP packet (Tier 2, Magic 0xC5110002)                       */
/* ------------------------------------------------------------------ */

/** ADR-039 vitals packet magic number. */
#define EDGE_VITALS_MAGIC 0xC5110002

/** Vitals packet type identifier. */
#define EDGE_PKT_TYPE_VITALS 0x02

/**
 * Vitals packet — 32 bytes, sent at 1 Hz over UDP.
 * Compatible with the ADR-018 aggregator (different magic discriminates).
 */
typedef struct __attribute__((packed)) {
    uint32_t magic;                 /**< 0xC5110002 */
    uint8_t  node_id;
    uint8_t  pkt_type;              /**< EDGE_PKT_TYPE_VITALS */
    uint16_t sequence;
    uint8_t  presence;              /**< 0=empty, 1=present, 2=moving */
    uint8_t  motion_score;          /**< 0-255 */
    uint8_t  occupancy;             /**< 0-8 */
    uint8_t  coherence_gate;        /**< Reserved for future use */
    uint16_t breathing_bpm_x100;    /**< BPM * 100 */
    uint16_t heartrate_bpm_x100;    /**< BPM * 100 */
    uint16_t breathing_conf;        /**< Confidence * 10000 */
    uint16_t heartrate_conf;        /**< Confidence * 10000 */
    uint8_t  fall_detected;
    uint8_t  anomaly_flags;         /**< Reserved */
    int16_t  rssi_mean;             /**< Averaged RSSI */
    uint32_t csi_count;             /**< Total frames processed */
    uint32_t uptime_s;              /**< Seconds since boot */
} edge_vitals_packet_t;

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/**
 * Initialize edge processing.
 *
 * @param tier  Processing tier (0=disabled, 1=phase/stats/compress, 2=vitals).
 *              Tier 0 is a no-op for backward compatibility.
 */
void edge_processing_init(uint8_t tier);

/**
 * Push a CSI frame into the edge processing ring buffer.
 * Called from the CSI callback on Core 0. Lock-free, O(1).
 *
 * @param info  WiFi CSI info from the ESP-IDF callback.
 */
void edge_push_csi(const wifi_csi_info_t *info);

/**
 * Get the currently configured edge processing tier.
 *
 * @return Tier (0-3).
 */
uint8_t edge_get_tier(void);

/* ------------------------------------------------------------------ */
/* Tier 1 pure functions (suitable for unit testing)                   */
/* ------------------------------------------------------------------ */

/**
 * Phase unwrap: extract phase from I/Q data with 2pi correction.
 *
 * @param iq         Raw I/Q pairs (I0, Q0, I1, Q1, ...).
 * @param n_sc       Number of subcarriers.
 * @param phase_out  Output phases in radians (size >= n_sc).
 * @param phase_prev Previous phases for unwrap (updated in place).
 */
void edge_phase_unwrap(const int8_t *iq, uint16_t n_sc,
                       float *phase_out, float *phase_prev);

/**
 * Welford online algorithm — update running mean and M2.
 *
 * @param value  New sample value.
 * @param mean   Running mean (updated in place).
 * @param m2     Running M2 (updated in place).
 * @param count  Sample count (updated in place).
 */
void edge_welford_update(float value, float *mean, float *m2, uint32_t *count);

/**
 * Compute variance from Welford M2 accumulator.
 *
 * @param m2     M2 value.
 * @param count  Sample count (must be >= 2).
 * @return Population variance, or 0 if count < 2.
 */
float edge_welford_variance(float m2, uint32_t count);

/**
 * Select top-K subcarriers by variance (partial sort).
 *
 * @param variances  Variance array (size n).
 * @param n          Total subcarrier count.
 * @param k          Number to select.
 * @param selected   Output array of selected indices (size >= k).
 * @return Actual number selected (min(k, n)).
 */
uint16_t edge_select_top_k(const float *variances, uint16_t n,
                           uint8_t k, uint8_t *selected);

/**
 * Delta compress I/Q data: XOR with previous frame, then simple RLE.
 *
 * @param cur      Current I/Q data.
 * @param prev     Previous I/Q data.
 * @param len      Length of I/Q data in bytes.
 * @param out      Output buffer for compressed data.
 * @param out_len  Size of output buffer.
 * @return Number of bytes written to out, or 0 if compression failed.
 */
uint16_t edge_delta_compress(const int8_t *cur, const int8_t *prev,
                             uint16_t len, uint8_t *out, uint16_t out_len);

/* ------------------------------------------------------------------ */
/* Tier 2 functions                                                    */
/* ------------------------------------------------------------------ */

/**
 * Update presence / motion detection from amplitude data.
 *
 * @param state       Tier 2 state (updated in place).
 * @param amplitudes  Amplitude array for current frame.
 * @param n           Number of subcarriers.
 */
void edge_update_presence(edge_tier2_state_t *state,
                          const float *amplitudes, uint16_t n);

/**
 * Update vital signs estimation from phase data.
 *
 * @param state   Tier 2 state (updated in place).
 * @param phases  Phase array for current frame.
 * @param n       Number of subcarriers.
 */
void edge_update_vitals(edge_tier2_state_t *state,
                        const float *phases, uint16_t n);

/**
 * Check for fall event: variance spike >5 sigma followed by stillness.
 *
 * @param state             Tier 2 state (updated in place).
 * @param current_variance  Current frame variance.
 * @return true if a fall is detected.
 */
bool edge_detect_fall(edge_tier2_state_t *state, float current_variance);

#endif /* EDGE_PROCESSING_H */

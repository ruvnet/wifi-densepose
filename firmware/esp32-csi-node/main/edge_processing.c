/**
 * @file edge_processing.c
 * @brief ADR-039 Edge Intelligence — on-device CSI processing.
 *
 * Implements a dual-core pipeline:
 *   Core 0 (ISR context):  wifi_csi_callback -> edge_push_csi() -> SPSC ring
 *   Core 1 (edge_task):    ring -> phase unwrap -> Welford -> top-K -> compress
 *                                -> (Tier 2) presence / vitals / fall
 *
 * Memory budget (static):
 *   Ring buffer:  64 * ~400 B = ~25 KB
 *   Tier 1 state: ~4 KB
 *   Tier 2 state: ~2 KB
 *   Scratch:      ~2 KB
 *   Total:        ~33 KB on Core 1 stack + BSS
 *
 * All DSP uses the ESP32-S3 hardware single-precision FPU.
 */

#include "edge_processing.h"
#include "stream_sender.h"
#include "nvs_config.h"

#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

static const char *TAG = "edge_proc";

/* ================================================================== */
/* Configuration (loaded from nvs_config at init)                     */
/* ================================================================== */

static uint8_t  s_tier            = 0;
static uint8_t  s_node_id         = 0;
static uint16_t s_presence_thresh = 50;
static uint16_t s_fall_thresh     = 500;
static uint16_t s_vital_window    = 300;
static uint16_t s_vital_interval_ms = 1000;
static uint8_t  s_subk_count      = 32;

/* ================================================================== */
/* Lock-free SPSC ring buffer                                         */
/* ================================================================== */

/**
 * Lock-free single-producer single-consumer ring buffer.
 *
 * Producer (Core 0, ISR-safe): increments s_ring_write after writing.
 * Consumer (Core 1, edge_task): increments s_ring_read after reading.
 * Both indices are volatile to prevent compiler reordering.
 * Ring capacity is EDGE_RING_SIZE - 1 to distinguish full from empty.
 */
static edge_csi_entry_t s_ring[EDGE_RING_SIZE];
static volatile uint32_t s_ring_write = 0;  /**< Next write position (producer). */
static volatile uint32_t s_ring_read  = 0;  /**< Next read position (consumer). */

/** Notification semaphore: producer gives, consumer takes. */
static SemaphoreHandle_t s_ring_sem = NULL;

/** Number of entries in the ring (lock-free). */
static inline uint32_t ring_count(void)
{
    uint32_t w = s_ring_write;
    uint32_t r = s_ring_read;
    return (w - r) & (EDGE_RING_SIZE - 1);
}

/** Check if ring is full. */
static inline bool ring_full(void)
{
    return ring_count() >= (EDGE_RING_SIZE - 1);
}

/* ================================================================== */
/* Processing state (Core 1 only — no synchronization needed)         */
/* ================================================================== */

static edge_tier1_state_t s_t1;
static edge_tier2_state_t s_t2;

/** Scratch buffers for DSP on Core 1. */
static float s_phase_buf[EDGE_MAX_SUBCARRIERS];
static float s_amp_buf[EDGE_MAX_SUBCARRIERS];
static float s_var_buf[EDGE_MAX_SUBCARRIERS];
static uint8_t s_topk_idx[EDGE_MAX_SUBCARRIERS]; /* worst case k == n */

/** Compressed output buffer. */
static uint8_t s_compress_buf[EDGE_MAX_IQ_LEN * 2];

/** Running RSSI accumulator (for vitals packet). */
static float    s_rssi_sum   = 0.0f;
static uint32_t s_rssi_count = 0;

/** Total frames processed and vitals sequence counter. */
static uint32_t s_frame_count    = 0;
static uint16_t s_vitals_seq     = 0;

/** Vitals packet send timer. */
static esp_timer_handle_t s_vitals_timer = NULL;
static volatile bool s_vitals_due = false;

/* ================================================================== */
/* Biquad IIR filter for vital signs (Tier 2)                         */
/* ================================================================== */

/**
 * Second-order IIR (biquad) filter coefficients.
 * Direct Form II Transposed.
 */
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float z1, z2;  /**< State variables. */
} biquad_t;

/**
 * Pre-computed biquad coefficients for 20 Hz sample rate.
 * These are bandpass filters designed with the bilinear transform.
 *
 * Breathing band: 0.1 - 0.5 Hz (6 - 30 BPM)
 * Heart rate band: 0.8 - 2.0 Hz (48 - 120 BPM)
 *
 * Coefficients were computed offline using scipy.signal.iirfilter
 * with Butterworth type, order=2, fs=20.
 */

/** Breathing bandpass: 0.1-0.5 Hz at 20 Hz sample rate, 2nd order Butterworth. */
static biquad_t s_bq_breath = {
    .b0 =  0.02008337f,
    .b1 =  0.0f,
    .b2 = -0.02008337f,
    .a1 = -1.93803473f,
    .a2 =  0.95983326f,
    .z1 = 0.0f, .z2 = 0.0f,
};

/** Heart rate bandpass: 0.8-2.0 Hz at 20 Hz sample rate, 2nd order Butterworth. */
static biquad_t s_bq_heart = {
    .b0 =  0.09853117f,
    .b1 =  0.0f,
    .b2 = -0.09853117f,
    .a1 = -1.53073372f,
    .a2 =  0.80293766f,
    .z1 = 0.0f, .z2 = 0.0f,
};

/** Apply biquad filter to a single sample (Direct Form II Transposed). */
static inline float biquad_process(biquad_t *bq, float x)
{
    float y = bq->b0 * x + bq->z1;
    bq->z1  = bq->b1 * x - bq->a1 * y + bq->z2;
    bq->z2  = bq->b2 * x - bq->a2 * y;
    return y;
}

/* ================================================================== */
/* Tier 1: Phase unwrap                                               */
/* ================================================================== */

void edge_phase_unwrap(const int8_t *iq, uint16_t n_sc,
                       float *phase_out, float *phase_prev)
{
    if (iq == NULL || phase_out == NULL || phase_prev == NULL || n_sc == 0) {
        return;
    }

    for (uint16_t i = 0; i < n_sc; i++) {
        float ii = (float)iq[2 * i];
        float qq = (float)iq[2 * i + 1];

        /* atan2 gives phase in [-pi, pi]. ESP32-S3 FPU handles this. */
        float phase = atan2f(qq, ii);

        /* Unwrap: correct jumps > pi relative to previous phase. */
        float diff = phase - phase_prev[i];
        if (diff > (float)M_PI) {
            phase -= 2.0f * (float)M_PI;
        } else if (diff < -(float)M_PI) {
            phase += 2.0f * (float)M_PI;
        }

        phase_out[i]  = phase;
        phase_prev[i] = phase;
    }
}

/* ================================================================== */
/* Tier 1: Welford online statistics                                  */
/* ================================================================== */

void edge_welford_update(float value, float *mean, float *m2, uint32_t *count)
{
    (*count)++;
    float delta  = value - *mean;
    *mean       += delta / (float)(*count);
    float delta2 = value - *mean;
    *m2         += delta * delta2;
}

float edge_welford_variance(float m2, uint32_t count)
{
    if (count < 2) {
        return 0.0f;
    }
    return m2 / (float)count;
}

/* ================================================================== */
/* Tier 1: Top-K subcarrier selection (partial sort)                  */
/* ================================================================== */

uint16_t edge_select_top_k(const float *variances, uint16_t n,
                           uint8_t k, uint8_t *selected)
{
    if (variances == NULL || selected == NULL || n == 0 || k == 0) {
        return 0;
    }

    /* Clamp k to available subcarriers and uint8_t max (255). */
    uint16_t actual_k = (k < n) ? k : n;
    if (actual_k > 255) {
        actual_k = 255;
    }

    /*
     * Simple O(n*k) selection — good enough for n <= 192, k <= 64.
     * A full partial-sort (quickselect) is overkill at these sizes.
     *
     * We maintain a sorted (descending) list of the top-k seen so far.
     */
    float top_val[255];
    uint8_t top_idx_local[255];

    /* Initialize with -infinity. */
    for (uint16_t i = 0; i < actual_k; i++) {
        top_val[i] = -1.0e30f;
        top_idx_local[i] = 0;
    }

    for (uint16_t i = 0; i < n; i++) {
        float v = variances[i];

        /* Check if v belongs in the top-k list. */
        if (v > top_val[actual_k - 1]) {
            /* Find insertion point (linear scan of small array). */
            uint16_t pos = actual_k - 1;
            while (pos > 0 && v > top_val[pos - 1]) {
                top_val[pos]       = top_val[pos - 1];
                top_idx_local[pos] = top_idx_local[pos - 1];
                pos--;
            }
            top_val[pos]       = v;
            top_idx_local[pos] = (uint8_t)i;
        }
    }

    for (uint16_t i = 0; i < actual_k; i++) {
        selected[i] = top_idx_local[i];
    }

    return (uint16_t)actual_k;
}

/* ================================================================== */
/* Tier 1: Delta compression (XOR + RLE)                              */
/* ================================================================== */

uint16_t edge_delta_compress(const int8_t *cur, const int8_t *prev,
                             uint16_t len, uint8_t *out, uint16_t out_len)
{
    if (cur == NULL || prev == NULL || out == NULL || len == 0 || out_len < 2) {
        return 0;
    }

    /*
     * Algorithm:
     * 1. XOR current with previous frame (delta).
     * 2. RLE encode the delta: (count, value) pairs.
     *    - count is stored as uint8_t (max 255 consecutive same-value bytes).
     *    - This works well because CSI delta is often near-zero.
     */
    uint16_t out_pos = 0;

    uint16_t i = 0;
    while (i < len) {
        uint8_t delta_val = (uint8_t)(cur[i] ^ prev[i]);
        uint8_t run_len = 1;

        /* Count consecutive identical delta values. */
        while (i + run_len < len && run_len < 255) {
            uint8_t next_delta = (uint8_t)(cur[i + run_len] ^ prev[i + run_len]);
            if (next_delta != delta_val) {
                break;
            }
            run_len++;
        }

        /* Write (count, value) pair. */
        if (out_pos + 2 > out_len) {
            /* Output buffer full — compression failed to save space. */
            return 0;
        }
        out[out_pos++] = run_len;
        out[out_pos++] = delta_val;

        i += run_len;
    }

    return out_pos;
}

/* ================================================================== */
/* Tier 2: Presence detection                                         */
/* ================================================================== */

void edge_update_presence(edge_tier2_state_t *state,
                          const float *amplitudes, uint16_t n)
{
    if (state == NULL || amplitudes == NULL || n == 0) {
        return;
    }

    /*
     * Compute total amplitude variance across all subcarriers.
     * High variance = motion. Low but nonzero = static presence.
     * Near-zero = empty room.
     */
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint16_t i = 0; i < n; i++) {
        sum    += amplitudes[i];
        sum_sq += amplitudes[i] * amplitudes[i];
    }

    float mean = sum / (float)n;
    float var  = (sum_sq / (float)n) - (mean * mean);
    if (var < 0.0f) {
        var = 0.0f;
    }

    /* Convert variance to an integer score. */
    float var_scaled = var * 10.0f;
    uint16_t var_int = (var_scaled > 65535.0f) ? 65535 : (uint16_t)var_scaled;

    if (var_int < s_presence_thresh / 2) {
        state->presence = 0;  /* Empty */
        state->motion_score = 0;
    } else if (var_int < s_presence_thresh) {
        state->presence = 1;  /* Present (static) */
        state->motion_score = (uint8_t)(var_int * 128 / s_presence_thresh);
    } else {
        state->presence = 2;  /* Moving */
        uint32_t score = (uint32_t)var_int * 255 / (s_presence_thresh * 10);
        state->motion_score = (score > 255) ? 255 : (uint8_t)score;
    }

    /* Simple occupancy estimate: if motion on many subcarriers, likely > 1 person.
     * Count subcarriers with amplitude > 2 * mean as "active". */
    uint16_t active_count = 0;
    float thresh = mean * 2.0f;
    for (uint16_t i = 0; i < n; i++) {
        if (amplitudes[i] > thresh) {
            active_count++;
        }
    }

    /* Heuristic: every ~24 active subcarriers roughly corresponds to 1 person
     * in a typical 64-subcarrier environment. */
    uint8_t occ = (uint8_t)(active_count / 24);
    if (occ > 8) occ = 8;
    if (state->presence == 0) occ = 0;
    state->occupancy = occ;

    /* Fall detection via variance spike. */
    state->fall_detected = edge_detect_fall(state, var) ? 1 : 0;
}

/* ================================================================== */
/* Tier 2: Vital signs extraction                                     */
/* ================================================================== */

void edge_update_vitals(edge_tier2_state_t *state,
                        const float *phases, uint16_t n)
{
    if (state == NULL || phases == NULL || n == 0) {
        return;
    }

    /*
     * Use the first subcarrier's phase (caller should pass the best
     * subcarrier selected by top-K). Push into circular buffer.
     */
    float phase_val = phases[0];

    state->phase_history[state->history_idx] = phase_val;
    state->history_idx = (state->history_idx + 1) % EDGE_PHASE_HISTORY_LEN;
    if (state->history_len < EDGE_PHASE_HISTORY_LEN) {
        state->history_len++;
    }

    /*
     * Only estimate vitals when we have at least 3 seconds of data (60 samples at 20 Hz).
     * Full confidence requires the full window.
     */
    if (state->history_len < 60) {
        state->breathing_bpm       = 0.0f;
        state->heartrate_bpm       = 0.0f;
        state->breathing_confidence = 0.0f;
        state->heartrate_confidence = 0.0f;
        return;
    }

    /*
     * Process the most recent samples through biquad bandpass filters.
     * We filter the latest sample and count zero-crossings over the buffer.
     *
     * For real-time use we filter each incoming sample and count peaks
     * over a sliding window.
     */
    float breath_val = biquad_process(&s_bq_breath, phase_val);
    float heart_val  = biquad_process(&s_bq_heart,  phase_val);

    /*
     * Peak counting: count positive zero-crossings over the history.
     * We re-scan the last 'window' samples each time for simplicity.
     * On ESP32-S3 at 20 Hz, scanning 300 floats is trivial (<0.1 ms).
     */
    uint16_t window = state->history_len;
    if (window > s_vital_window) {
        window = s_vital_window;
    }

    /* Apply bandpass to the entire window and count peaks.
     * We use temporary biquads for the full-window scan so as not to
     * disturb the streaming filter state. */
    biquad_t bq_br_tmp = s_bq_breath;
    biquad_t bq_hr_tmp = s_bq_heart;

    /* Reset temporary filter state. */
    bq_br_tmp.z1 = 0.0f; bq_br_tmp.z2 = 0.0f;
    bq_hr_tmp.z1 = 0.0f; bq_hr_tmp.z2 = 0.0f;

    uint16_t breath_crossings = 0;
    uint16_t heart_crossings  = 0;
    float prev_br = 0.0f;
    float prev_hr = 0.0f;

    /* Walk the circular buffer from oldest to newest. */
    uint16_t start_idx;
    if (state->history_len < EDGE_PHASE_HISTORY_LEN) {
        start_idx = 0;
    } else {
        start_idx = state->history_idx;  /* Oldest entry. */
    }

    for (uint16_t j = 0; j < window; j++) {
        uint16_t idx = (start_idx + j) % EDGE_PHASE_HISTORY_LEN;
        float sample = state->phase_history[idx];

        float br = biquad_process(&bq_br_tmp, sample);
        float hr = biquad_process(&bq_hr_tmp, sample);

        /* Positive zero crossing. */
        if (j > 0) {
            if (prev_br <= 0.0f && br > 0.0f) {
                breath_crossings++;
            }
            if (prev_hr <= 0.0f && hr > 0.0f) {
                heart_crossings++;
            }
        }

        prev_br = br;
        prev_hr = hr;
    }

    /* Convert crossings to BPM.
     * Each positive zero crossing corresponds to one cycle.
     * window samples at 20 Hz = window/20 seconds. */
    float duration_s = (float)window / 20.0f;
    if (duration_s > 0.0f) {
        state->breathing_bpm = (float)breath_crossings * 60.0f / duration_s;
        state->heartrate_bpm = (float)heart_crossings  * 60.0f / duration_s;
    }

    /* Clamp to physiological ranges. */
    if (state->breathing_bpm < 4.0f)   state->breathing_bpm = 0.0f;
    if (state->breathing_bpm > 40.0f)  state->breathing_bpm = 0.0f;
    if (state->heartrate_bpm < 40.0f)  state->heartrate_bpm = 0.0f;
    if (state->heartrate_bpm > 150.0f) state->heartrate_bpm = 0.0f;

    /* Confidence: based on signal amplitude relative to noise floor.
     * Higher filtered amplitude = more confident. */
    float br_amp = fabsf(breath_val);
    float hr_amp = fabsf(heart_val);

    state->breathing_confidence = (br_amp > 0.5f) ? 1.0f : br_amp * 2.0f;
    state->heartrate_confidence = (hr_amp > 0.3f) ? 1.0f : hr_amp * 3.33f;

    if (state->breathing_confidence > 1.0f) state->breathing_confidence = 1.0f;
    if (state->heartrate_confidence > 1.0f) state->heartrate_confidence = 1.0f;

    /* If no presence detected, zero out vitals. */
    if (state->presence == 0) {
        state->breathing_bpm        = 0.0f;
        state->heartrate_bpm        = 0.0f;
        state->breathing_confidence = 0.0f;
        state->heartrate_confidence = 0.0f;
    }

    (void)breath_val;
    (void)heart_val;
}

/* ================================================================== */
/* Tier 2: Fall detection                                             */
/* ================================================================== */

bool edge_detect_fall(edge_tier2_state_t *state, float current_variance)
{
    if (state == NULL) {
        return false;
    }

    /* Store current variance in history ring. */
    state->variance_history[state->var_idx] = current_variance;
    state->var_idx = (state->var_idx + 1) % EDGE_VAR_HISTORY_LEN;

    /*
     * Fall detection heuristic:
     * 1. Compute mean and stdev of variance history.
     * 2. If current variance > mean + 5*stdev, that is a "spike".
     * 3. If the last 3 entries after the spike show low variance
     *    (< mean), declare a fall (spike + stillness).
     *
     * At 20 Hz and 20-entry history, this covers the last 1 second.
     * We check the last ~3 seconds by requiring the spike to have
     * happened recently and stillness to follow.
     */
    float sum = 0.0f;
    float sum_sq = 0.0f;
    uint8_t valid = 0;

    for (uint8_t i = 0; i < EDGE_VAR_HISTORY_LEN; i++) {
        float v = state->variance_history[i];
        if (v >= 0.0f) {
            sum    += v;
            sum_sq += v * v;
            valid++;
        }
    }

    if (valid < 10) {
        return false;  /* Not enough history yet. */
    }

    float mean = sum / (float)valid;
    float var_of_var = (sum_sq / (float)valid) - (mean * mean);
    if (var_of_var < 0.0f) var_of_var = 0.0f;
    float stdev = sqrtf(var_of_var);

    float spike_thresh = mean + 5.0f * stdev;
    if (spike_thresh < (float)s_fall_thresh / 100.0f) {
        spike_thresh = (float)s_fall_thresh / 100.0f;
    }

    /* Check if there was a recent spike (within last 10 entries)
     * followed by low values (last 3 entries). */
    bool saw_spike = false;
    for (uint8_t i = 0; i < 10; i++) {
        uint8_t idx = (state->var_idx + EDGE_VAR_HISTORY_LEN - 1 - i) % EDGE_VAR_HISTORY_LEN;
        if (state->variance_history[idx] > spike_thresh) {
            saw_spike = true;
            break;
        }
    }

    if (!saw_spike) {
        return false;
    }

    /* Check if the last 3 entries show stillness. */
    uint8_t still_count = 0;
    for (uint8_t i = 0; i < 3; i++) {
        uint8_t idx = (state->var_idx + EDGE_VAR_HISTORY_LEN - 1 - i) % EDGE_VAR_HISTORY_LEN;
        if (state->variance_history[idx] < mean * 0.5f) {
            still_count++;
        }
    }

    return (still_count >= 2);
}

/* ================================================================== */
/* Vitals packet construction and send                                */
/* ================================================================== */

static void send_vitals_packet(void)
{
    edge_vitals_packet_t pkt;
    memset(&pkt, 0, sizeof(pkt));

    pkt.magic     = EDGE_VITALS_MAGIC;
    pkt.node_id   = s_node_id;
    pkt.pkt_type  = EDGE_PKT_TYPE_VITALS;
    pkt.sequence  = s_vitals_seq++;

    pkt.presence     = s_t2.presence;
    pkt.motion_score = s_t2.motion_score;
    pkt.occupancy    = s_t2.occupancy;
    pkt.coherence_gate = 0;  /* Reserved. */

    pkt.breathing_bpm_x100  = (uint16_t)(s_t2.breathing_bpm * 100.0f);
    pkt.heartrate_bpm_x100  = (uint16_t)(s_t2.heartrate_bpm * 100.0f);
    pkt.breathing_conf      = (uint16_t)(s_t2.breathing_confidence * 10000.0f);
    pkt.heartrate_conf      = (uint16_t)(s_t2.heartrate_confidence * 10000.0f);

    pkt.fall_detected  = s_t2.fall_detected;
    pkt.anomaly_flags  = 0;

    if (s_rssi_count > 0) {
        pkt.rssi_mean = (int16_t)(s_rssi_sum / (float)s_rssi_count);
    } else {
        pkt.rssi_mean = 0;
    }

    pkt.csi_count = s_frame_count;
    pkt.uptime_s  = (uint32_t)(esp_timer_get_time() / 1000000ULL);

    /* Send via existing UDP sender. */
    stream_sender_send((const uint8_t *)&pkt, sizeof(pkt));

    ESP_LOGD(TAG, "Vitals pkt #%u: presence=%u motion=%u br=%.1f hr=%.1f",
             pkt.sequence, pkt.presence, pkt.motion_score,
             s_t2.breathing_bpm, s_t2.heartrate_bpm);
}

/**
 * Timer callback for periodic vitals packet transmission.
 * Sets a flag that the edge task checks — avoids doing work in timer context.
 */
static void vitals_timer_cb(void *arg)
{
    (void)arg;
    s_vitals_due = true;
}

/* ================================================================== */
/* Edge processing task (pinned to Core 1)                            */
/* ================================================================== */

/**
 * Process a single CSI frame through the Tier 1 pipeline.
 */
static void process_tier1(const edge_csi_entry_t *entry)
{
    uint16_t n_sc = entry->iq_len / 2;
    if (n_sc == 0 || n_sc > EDGE_MAX_SUBCARRIERS) {
        return;
    }

    /* Phase unwrap. */
    edge_phase_unwrap(entry->iq_data, n_sc, s_phase_buf, s_t1.phase_prev);

    /* Compute amplitudes and update Welford stats. */
    for (uint16_t i = 0; i < n_sc; i++) {
        float ii = (float)entry->iq_data[2 * i];
        float qq = (float)entry->iq_data[2 * i + 1];
        s_amp_buf[i] = sqrtf(ii * ii + qq * qq);

        edge_welford_update(s_amp_buf[i],
                            &s_t1.amp_mean[i],
                            &s_t1.amp_m2[i],
                            &s_t1.amp_count);
    }

    /* Note: amp_count is shared across subcarriers (they all advance together).
     * This is correct because we call Welford once per subcarrier per frame,
     * and all subcarriers receive the same frame count. The count represents
     * the number of frames seen, not per-subcarrier counts. */

    /* Compute per-subcarrier variance for top-K selection. */
    for (uint16_t i = 0; i < n_sc; i++) {
        s_var_buf[i] = edge_welford_variance(s_t1.amp_m2[i], s_t1.amp_count);
    }

    /* Select top-K highest-variance subcarriers. */
    uint8_t k = s_subk_count;
    if (k > n_sc) k = (uint8_t)n_sc;
    uint16_t selected = edge_select_top_k(s_var_buf, n_sc, k, s_topk_idx);
    (void)selected;  /* Available for downstream use. */

    /* Delta compress if we have a previous frame. */
    if (s_t1.has_prev) {
        uint16_t compressed_len = edge_delta_compress(
            entry->iq_data, s_t1.prev_iq,
            entry->iq_len, s_compress_buf, sizeof(s_compress_buf));
        (void)compressed_len;  /* Will be used for Tier 3 compressed streaming. */
    }

    /* Store current frame as previous for next delta. */
    memcpy(s_t1.prev_iq, entry->iq_data, entry->iq_len);
    s_t1.has_prev = true;

    /* Accumulate RSSI for vitals packet. */
    s_rssi_sum += (float)entry->rssi;
    s_rssi_count++;
}

/**
 * Process a single CSI frame through the Tier 2 pipeline.
 * Requires Tier 1 to have run first (uses s_phase_buf, s_amp_buf).
 */
static void process_tier2(const edge_csi_entry_t *entry)
{
    uint16_t n_sc = entry->iq_len / 2;
    if (n_sc == 0 || n_sc > EDGE_MAX_SUBCARRIERS) {
        return;
    }

    /* Presence and motion detection from amplitudes. */
    edge_update_presence(&s_t2, s_amp_buf, n_sc);

    /* Vital signs from the best subcarrier's phase.
     * Use the first entry in the top-K list (highest variance). */
    if (s_subk_count > 0 && n_sc > 0) {
        uint8_t best_sc = s_topk_idx[0];
        if (best_sc < n_sc) {
            float best_phase = s_phase_buf[best_sc];
            edge_update_vitals(&s_t2, &best_phase, 1);
        }
    }
}

/**
 * Main edge processing task — runs on Core 1.
 *
 * Blocks on the ring buffer semaphore, then drains all available entries.
 */
static void edge_task(void *arg)
{
    (void)arg;

    ESP_LOGI(TAG, "Edge task started on core %d (tier=%u)",
             xPortGetCoreID(), (unsigned)s_tier);

    while (1) {
        /* Block until producer signals new data (or timeout for vitals). */
        xSemaphoreTake(s_ring_sem, pdMS_TO_TICKS(100));

        /* Drain all available ring entries. */
        while (s_ring_read != s_ring_write) {
            uint32_t idx = s_ring_read & (EDGE_RING_SIZE - 1);
            const edge_csi_entry_t *entry = &s_ring[idx];

            /* Tier 1: always run if tier >= 1. */
            process_tier1(entry);
            s_frame_count++;

            /* Tier 2: run if tier >= 2. */
            if (s_tier >= 2) {
                process_tier2(entry);
            }

            /* Advance read pointer (memory barrier via volatile). */
            s_ring_read++;
        }

        /* Send vitals packet at configured interval (Tier 2). */
        if (s_tier >= 2 && s_vitals_due) {
            s_vitals_due = false;
            send_vitals_packet();

            /* Reset RSSI accumulator. */
            s_rssi_sum   = 0.0f;
            s_rssi_count = 0;
        }
    }
}

/* ================================================================== */
/* Public API                                                          */
/* ================================================================== */

void edge_push_csi(const wifi_csi_info_t *info)
{
    if (s_tier == 0 || info == NULL || info->buf == NULL) {
        return;
    }

    /* Check ring space. */
    if (ring_full()) {
        /* Drop frame — producer must never block in ISR context. */
        static uint32_t s_drop_count = 0;
        s_drop_count++;
        if (s_drop_count <= 3 || (s_drop_count % 1000) == 0) {
            ESP_LOGW(TAG, "Ring full, frame dropped (total=%lu)",
                     (unsigned long)s_drop_count);
        }
        return;
    }

    /* Write entry at current write position. */
    uint32_t idx = s_ring_write & (EDGE_RING_SIZE - 1);
    edge_csi_entry_t *entry = &s_ring[idx];

    uint16_t iq_len = (uint16_t)info->len;
    if (iq_len > EDGE_MAX_IQ_LEN) {
        iq_len = EDGE_MAX_IQ_LEN;
    }

    memcpy(entry->iq_data, info->buf, iq_len);
    entry->iq_len      = iq_len;
    entry->rssi        = (int8_t)info->rx_ctrl.rssi;
    entry->noise_floor = (int8_t)info->rx_ctrl.noise_floor;
    entry->channel     = (uint8_t)info->rx_ctrl.channel;
    memcpy(entry->tx_mac, info->mac, 6);
    entry->timestamp_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);

    /* Advance write pointer (volatile write acts as release fence). */
    s_ring_write++;

    /* Wake the consumer task. */
    if (s_ring_sem != NULL) {
        xSemaphoreGiveFromISR(s_ring_sem, NULL);
    }
}

uint8_t edge_get_tier(void)
{
    return s_tier;
}

void edge_processing_init(uint8_t tier)
{
    s_tier = tier;

    if (tier == 0) {
        ESP_LOGI(TAG, "Edge processing disabled (tier=0)");
        return;
    }

    ESP_LOGI(TAG, "Initializing edge processing tier=%u", (unsigned)tier);

    /* Read configuration from the extern nvs_config (already loaded in main). */
    /* These are set via the Kconfig / NVS defaults applied in nvs_config_load. */
    extern nvs_config_t s_cfg;  /* Defined in main.c */
    s_node_id         = s_cfg.node_id;
    s_presence_thresh = s_cfg.presence_thresh;
    s_fall_thresh     = s_cfg.fall_thresh;
    s_vital_window    = s_cfg.vital_window;
    s_vital_interval_ms = s_cfg.vital_interval_ms;
    s_subk_count      = s_cfg.subk_count;

    ESP_LOGI(TAG, "  presence_thresh=%u fall_thresh=%u vital_window=%u interval=%ums subk=%u",
             s_presence_thresh, s_fall_thresh, s_vital_window,
             s_vital_interval_ms, s_subk_count);

    /* Initialize state. */
    memset(&s_t1, 0, sizeof(s_t1));
    memset(&s_t2, 0, sizeof(s_t2));
    s_ring_write  = 0;
    s_ring_read   = 0;
    s_frame_count = 0;
    s_vitals_seq  = 0;
    s_rssi_sum    = 0.0f;
    s_rssi_count  = 0;
    s_vitals_due  = false;

    /* Reset biquad filter state. */
    s_bq_breath.z1 = 0.0f; s_bq_breath.z2 = 0.0f;
    s_bq_heart.z1  = 0.0f; s_bq_heart.z2  = 0.0f;

    /* Create notification semaphore (binary). */
    s_ring_sem = xSemaphoreCreateBinary();
    if (s_ring_sem == NULL) {
        ESP_LOGE(TAG, "Failed to create ring semaphore");
        return;
    }

    /* Create edge processing task pinned to Core 1.
     * Stack size: 8 KB is sufficient for our static-alloc pipeline. */
    BaseType_t ret = xTaskCreatePinnedToCore(
        edge_task,
        "edge_task",
        8192,       /* Stack size in bytes. */
        NULL,
        5,          /* Priority (above idle, below WiFi). */
        NULL,
        1           /* Core 1. */
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create edge task");
        return;
    }

    /* For Tier 2: start the periodic vitals packet timer. */
    if (tier >= 2 && s_vital_interval_ms > 0) {
        esp_timer_create_args_t timer_args = {
            .callback = vitals_timer_cb,
            .arg      = NULL,
            .name     = "vitals_tx",
        };

        esp_err_t err = esp_timer_create(&timer_args, &s_vitals_timer);
        if (err == ESP_OK) {
            err = esp_timer_start_periodic(s_vitals_timer,
                                           (uint64_t)s_vital_interval_ms * 1000);
            if (err != ESP_OK) {
                ESP_LOGE(TAG, "Failed to start vitals timer: %s",
                         esp_err_to_name(err));
            } else {
                ESP_LOGI(TAG, "Vitals timer started: interval=%u ms",
                         s_vital_interval_ms);
            }
        } else {
            ESP_LOGE(TAG, "Failed to create vitals timer: %s",
                     esp_err_to_name(err));
        }
    }

    ESP_LOGI(TAG, "Edge processing initialized (tier=%u, ring=%u slots)",
             (unsigned)tier, (unsigned)EDGE_RING_SIZE);
}

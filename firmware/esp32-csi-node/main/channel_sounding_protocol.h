/**
 * @file channel_sounding_protocol.h
 * @brief Versioned UART contract for an external Bluetooth Channel Sounding radio.
 *
 * ESP32-S3 cannot acquire Bluetooth 6 Channel Sounding phase or RTT.  A radio
 * that can do so may send calibrated primitives over this bounded frame.  The
 * ESP32 validates and forwards primitives only; it does not label them as a
 * respiration or heartbeat result.
 */

#ifndef CHANNEL_SOUNDING_PROTOCOL_H
#define CHANNEL_SOUNDING_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RV_CS_MAGIC              0x53435652u /* "RVCS" little endian */
#define RV_CS_VERSION            1u
#define RV_CS_FRAME_SIZE         72u
#define RV_CS_AUTH_TAG_SIZE      16u
#define RV_CS_SIGNED_PREFIX_SIZE 52u
#define RV_CS_MAC_DOMAIN_SIZE    12u
#define RV_CS_MAC_INPUT_SIZE (RV_CS_MAC_DOMAIN_SIZE + RV_CS_SIGNED_PREFIX_SIZE)
#define RV_CS_MAX_CHANNEL_INDEX  78u
#define RV_CS_MAX_PHASE_MRAD     3142
#define RV_CS_MAX_RTT_PS         250000
#define RV_CS_MAX_FREQ_OFFSET_HZ 500000
#define RV_CS_MIN_STEP_COUNT     4u
#define RV_CS_MAX_STEP_COUNT     79u

#define RV_CS_FLAG_CALIBRATED (1u << 0)
#define RV_CS_FLAG_MOTION     (1u << 1)
#define RV_CS_FLAGS_ALLOWED (RV_CS_FLAG_CALIBRATED | RV_CS_FLAG_MOTION)

typedef struct {
    uint8_t  flags;
    uint8_t  key_id;
    uint32_t sequence;
    uint32_t sample_age_us;
    uint32_t source_id;
    uint32_t source_session_id;
    uint32_t procedure_id;
    uint16_t channel_index;
    uint16_t step_index;
    uint16_t step_count;
    uint16_t quality_permille;
    uint16_t timing_uncertainty_us;
    int32_t  phase_millirad;
    int32_t  rtt_picoseconds;
    int32_t  frequency_offset_hz;
} rv_cs_measurement_t;

typedef enum {
    RV_CS_PARSE_OK = 0,
    RV_CS_PARSE_BAD_ARGUMENT,
    RV_CS_PARSE_BAD_MAGIC,
    RV_CS_PARSE_BAD_VERSION,
    RV_CS_PARSE_BAD_LENGTH,
    RV_CS_PARSE_BAD_FLAGS,
    RV_CS_PARSE_BAD_CRC,
    RV_CS_PARSE_BAD_SOURCE,
    RV_CS_PARSE_BAD_SESSION,
    RV_CS_PARSE_BAD_PROCEDURE,
    RV_CS_PARSE_BAD_STEP,
    RV_CS_PARSE_BAD_CHANNEL,
    RV_CS_PARSE_BAD_QUALITY,
    RV_CS_PARSE_BAD_PHASE,
    RV_CS_PARSE_BAD_RTT,
    RV_CS_PARSE_BAD_FREQUENCY_OFFSET,
    RV_CS_PARSE_BAD_TIMING_UNCERTAINTY,
    RV_CS_PARSE_STALE,
} rv_cs_parse_result_t;

uint32_t rv_cs_crc32(const uint8_t *data, size_t len);

/**
 * Parse and validate a fixed frame. The companion supplies bounded sample age,
 * not its unrelated monotonic timestamp. The gateway receive timestamp is
 * assigned after authentication by the caller.
 *
 * @param data frame bytes
 * @param len must equal RV_CS_FRAME_SIZE
 * @param max_age_us maximum accepted age
 * @param min_quality_permille minimum admitted quality
 * @param out validated primitive
 */
rv_cs_parse_result_t rv_cs_parse_frame(const uint8_t *data,
                                       size_t len,
                                       uint32_t max_age_us,
                                       uint16_t min_quality_permille,
                                       rv_cs_measurement_t *out);

/** Build domain-separated HMAC input from the signed 52-byte prefix. */
void rv_cs_mac_input(const uint8_t frame[RV_CS_FRAME_SIZE],
                     uint8_t out[RV_CS_MAC_INPUT_SIZE]);

/** Constant-time comparison for the 128-bit companion authentication tag. */
bool rv_cs_auth_tag_equal(const uint8_t lhs[RV_CS_AUTH_TAG_SIZE],
                          const uint8_t rhs[RV_CS_AUTH_TAG_SIZE]);

/** True when candidate is strictly newer under uint32 wrap semantics. */
bool rv_cs_sequence_is_newer(uint32_t candidate, uint32_t previous);

#endif /* CHANNEL_SOUNDING_PROTOCOL_H */

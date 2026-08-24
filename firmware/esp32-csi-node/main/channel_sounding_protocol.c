/** @file channel_sounding_protocol.c */

#include "channel_sounding_protocol.h"

#ifdef ESP_PLATFORM
#include "mbedtls/constant_time.h"
#endif

static uint16_t read_le16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_le32(const uint8_t *p)
{
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

uint32_t rv_cs_crc32(const uint8_t *data, size_t len)
{
    if (data == NULL) return 0u;
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0u; i < len; i++) {
        crc ^= data[i];
        for (unsigned bit = 0u; bit < 8u; bit++) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

bool rv_cs_sequence_is_newer(uint32_t candidate, uint32_t previous)
{
    uint32_t delta = candidate - previous;
    return delta != 0u && delta < 0x80000000u;
}

rv_cs_parse_result_t rv_cs_parse_frame(const uint8_t *data,
                                       size_t len,
                                       uint32_t max_age_us,
                                       uint16_t min_quality_permille,
                                       rv_cs_measurement_t *out)
{
    if (data == NULL || out == NULL) return RV_CS_PARSE_BAD_ARGUMENT;
    if (len != RV_CS_FRAME_SIZE) return RV_CS_PARSE_BAD_LENGTH;
    if (read_le32(&data[0]) != RV_CS_MAGIC) return RV_CS_PARSE_BAD_MAGIC;
    if (data[4] != RV_CS_VERSION) return RV_CS_PARSE_BAD_VERSION;
    if (data[7] != 0u || read_le16(&data[8]) != RV_CS_FRAME_SIZE) return RV_CS_PARSE_BAD_LENGTH;
    if ((data[5] & ~RV_CS_FLAGS_ALLOWED) != 0u) return RV_CS_PARSE_BAD_FLAGS;
    if (rv_cs_crc32(data, RV_CS_FRAME_SIZE - 4u)
        != read_le32(&data[RV_CS_FRAME_SIZE - 4u])) {
        return RV_CS_PARSE_BAD_CRC;
    }

    rv_cs_measurement_t parsed = {
        .flags = data[5],
        .key_id = data[6],
        .channel_index = read_le16(&data[10]),
        .sequence = read_le32(&data[12]),
        .sample_age_us = read_le32(&data[16]),
        .source_id = read_le32(&data[20]),
        .quality_permille = read_le16(&data[24]),
        .timing_uncertainty_us = read_le16(&data[26]),
        .phase_millirad = (int32_t)read_le32(&data[28]),
        .rtt_picoseconds = (int32_t)read_le32(&data[32]),
        .frequency_offset_hz = (int32_t)read_le32(&data[36]),
        .source_session_id = read_le32(&data[40]),
        .procedure_id = read_le32(&data[44]),
        .step_index = read_le16(&data[48]),
        .step_count = read_le16(&data[50]),
    };

    if (parsed.source_id == 0u) return RV_CS_PARSE_BAD_SOURCE;
    if (parsed.source_session_id == 0u) return RV_CS_PARSE_BAD_SESSION;
    if (parsed.procedure_id == 0u) return RV_CS_PARSE_BAD_PROCEDURE;
    if (parsed.step_count < RV_CS_MIN_STEP_COUNT
        || parsed.step_count > RV_CS_MAX_STEP_COUNT
        || parsed.step_index >= parsed.step_count) return RV_CS_PARSE_BAD_STEP;
    if (parsed.channel_index > RV_CS_MAX_CHANNEL_INDEX) return RV_CS_PARSE_BAD_CHANNEL;
    if (parsed.quality_permille < min_quality_permille
        || parsed.quality_permille > 1000u) return RV_CS_PARSE_BAD_QUALITY;
    if (parsed.phase_millirad < -RV_CS_MAX_PHASE_MRAD
        || parsed.phase_millirad > RV_CS_MAX_PHASE_MRAD) return RV_CS_PARSE_BAD_PHASE;
    if (parsed.rtt_picoseconds < 0
        || parsed.rtt_picoseconds > RV_CS_MAX_RTT_PS) return RV_CS_PARSE_BAD_RTT;
    if (parsed.frequency_offset_hz < -RV_CS_MAX_FREQ_OFFSET_HZ
        || parsed.frequency_offset_hz > RV_CS_MAX_FREQ_OFFSET_HZ) {
        return RV_CS_PARSE_BAD_FREQUENCY_OFFSET;
    }
    if (parsed.timing_uncertainty_us > 10000u) return RV_CS_PARSE_BAD_TIMING_UNCERTAINTY;
    if (parsed.sample_age_us > max_age_us) return RV_CS_PARSE_STALE;
    *out = parsed;
    return RV_CS_PARSE_OK;
}

void rv_cs_mac_input(const uint8_t frame[RV_CS_FRAME_SIZE],
                     uint8_t out[RV_CS_MAC_INPUT_SIZE])
{
    static const uint8_t domain[RV_CS_MAC_DOMAIN_SIZE] = {
        'R', 'u', 'V', 'i', 'e', 'w', '/', 'C', 'S', '/', 'v', '1'
    };
    if (frame == NULL || out == NULL) return;
    for (size_t i = 0u; i < sizeof(domain); i++) out[i] = domain[i];
    for (size_t i = 0u; i < RV_CS_SIGNED_PREFIX_SIZE; i++) {
        out[sizeof(domain) + i] = frame[i];
    }
}

bool rv_cs_auth_tag_equal(const uint8_t lhs[RV_CS_AUTH_TAG_SIZE],
                          const uint8_t rhs[RV_CS_AUTH_TAG_SIZE])
{
    if (lhs == NULL || rhs == NULL) return false;
#ifdef ESP_PLATFORM
    return mbedtls_ct_memcmp(lhs, rhs, RV_CS_AUTH_TAG_SIZE) == 0;
#else
    uint8_t diff = 0u;
    for (size_t i = 0u; i < RV_CS_AUTH_TAG_SIZE; i++) diff |= (uint8_t)(lhs[i] ^ rhs[i]);
    return diff == 0u;
#endif
}

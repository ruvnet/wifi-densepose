/** @file radio_gateway_protocol.c */

#include "radio_gateway_protocol.h"

#include <string.h>

#ifdef ESP_PLATFORM
#include "mbedtls/constant_time.h"
#endif

const uint8_t RV_GATEWAY_MAC_DOMAIN[RV_GATEWAY_MAC_DOMAIN_SIZE] = {
    'R', 'u', 'V', 'i', 'e', 'w', '/', 'G', 'W', '/', 'v', '1'
};

static void write_le16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void write_le32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static void write_le64(uint8_t *p, uint64_t value)
{
    for (unsigned i = 0u; i < 8u; i++) {
        p[i] = (uint8_t)(value >> (8u * i));
    }
}

static bool payload_size_is_valid(uint8_t payload_type, size_t payload_len)
{
    if (payload_type == RV_GATEWAY_PAYLOAD_BLE_IDENTITY) {
        return payload_len == 36u;
    }
    if (payload_type == RV_GATEWAY_PAYLOAD_CHANNEL_SOUNDING) {
        return payload_len == 72u;
    }
    return false;
}

bool rv_gateway_build_unsigned(const rv_gateway_metadata_t *metadata,
                               const uint8_t *payload,
                               size_t payload_len,
                               uint8_t *out,
                               size_t out_capacity,
                               size_t *signed_len,
                               size_t *frame_len)
{
    if (metadata == NULL || payload == NULL || out == NULL
        || signed_len == NULL || frame_len == NULL
        || !payload_size_is_valid(metadata->payload_type, payload_len)
        || (metadata->flags & ~RV_GATEWAY_FLAGS_ALLOWED) != 0u
        || (metadata->flags & RV_GATEWAY_FLAG_RX_MONOTONIC) == 0u
        || metadata->sequence == 0u || metadata->boot_nonce == 0u
        || payload_len > RV_GATEWAY_MAX_PAYLOAD_SIZE) {
        return false;
    }

    size_t covered = RV_GATEWAY_HEADER_SIZE + payload_len;
    size_t total = covered + RV_GATEWAY_AUTH_TAG_SIZE;
    if (total > out_capacity || total > UINT16_MAX || payload_len > UINT16_MAX) {
        return false;
    }

    memset(out, 0, total);
    write_le32(&out[0], RV_GATEWAY_MAGIC);
    out[4] = RV_GATEWAY_VERSION;
    out[5] = metadata->payload_type;
    out[6] = metadata->flags;
    out[7] = metadata->key_id;
    write_le16(&out[8], (uint16_t)total);
    write_le16(&out[10], (uint16_t)payload_len);
    out[12] = metadata->node_id;
    /* bytes 13..15 are reserved and remain zero */
    write_le32(&out[16], metadata->sequence);
    write_le64(&out[20], metadata->boot_nonce);
    write_le64(&out[28], metadata->received_at_boot_us);
    write_le32(&out[36], metadata->timing_uncertainty_us);
    memcpy(&out[RV_GATEWAY_HEADER_SIZE], payload, payload_len);
    *signed_len = covered;
    *frame_len = total;
    return true;
}

bool rv_gateway_auth_tag_equal(const uint8_t lhs[RV_GATEWAY_AUTH_TAG_SIZE],
                               const uint8_t rhs[RV_GATEWAY_AUTH_TAG_SIZE])
{
    if (lhs == NULL || rhs == NULL) return false;
#ifdef ESP_PLATFORM
    return mbedtls_ct_memcmp(lhs, rhs, RV_GATEWAY_AUTH_TAG_SIZE) == 0;
#else
    uint8_t diff = 0u;
    for (size_t i = 0u; i < RV_GATEWAY_AUTH_TAG_SIZE; i++) {
        diff |= (uint8_t)(lhs[i] ^ rhs[i]);
    }
    return diff == 0u;
#endif
}

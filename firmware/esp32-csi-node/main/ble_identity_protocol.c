/** @file ble_identity_protocol.c */

#include "ble_identity_protocol.h"

#include <string.h>

#ifdef ESP_PLATFORM
#include "mbedtls/constant_time.h"
#endif

const uint8_t RV_BLE_SERVICE_UUID_LE[16] = {
    0x00, 0x01, 0xe0, 0xb1, 0x11, 0xc5, 0x09, 0x9f,
    0x69, 0x4d, 0x65, 0x5d, 0x40, 0xa8, 0x31, 0x6f,
};

static uint32_t read_le32(const uint8_t *p)
{
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static void write_le16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8);
}

static void write_le32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
    p[2] = (uint8_t)((v >> 16) & 0xffu);
    p[3] = (uint8_t)((v >> 24) & 0xffu);
}

bool rv_ble_parse_advertisement(const uint8_t *advertisement,
                                size_t advertisement_len,
                                rv_ble_token_t *out)
{
    if (advertisement == NULL || out == NULL || advertisement_len == 0u) {
        return false;
    }

    size_t offset = 0u;
    bool found = false;
    rv_ble_token_t parsed;
    memset(&parsed, 0, sizeof(parsed));

    while (offset < advertisement_len) {
        uint8_t field_len = advertisement[offset];
        if (field_len == 0u) {
            break;
        }
        if ((size_t)field_len + 1u > advertisement_len - offset) {
            return false;
        }

        const uint8_t *field = &advertisement[offset + 1u];
        uint8_t ad_type = field[0];
        size_t payload_len = (size_t)field_len - 1u;
        const uint8_t *payload = &field[1];

        if (ad_type == RV_BLE_AD_TYPE_SERVICE_DATA_UUID128
            && payload_len >= sizeof(RV_BLE_SERVICE_UUID_LE)
            && memcmp(payload, RV_BLE_SERVICE_UUID_LE,
                      sizeof(RV_BLE_SERVICE_UUID_LE)) == 0) {
            if (found || payload_len != RV_BLE_SERVICE_DATA_SIZE) {
                return false;
            }
            const uint8_t *body = payload + sizeof(RV_BLE_SERVICE_UUID_LE);
            parsed.version = body[0];
            parsed.key_id = body[1];
            parsed.epoch_min = read_le32(&body[2]);
            parsed.nonce = read_le32(&body[6]);
            memcpy(parsed.ephemeral_id, &body[10], RV_BLE_EPHEMERAL_ID_SIZE);
            memcpy(parsed.auth_tag, &body[18], RV_BLE_AUTH_TAG_SIZE);
            if (parsed.version != RV_BLE_TOKEN_VERSION) {
                return false;
            }
            found = true;
        }

        offset += (size_t)field_len + 1u;
    }

    if (!found) {
        return false;
    }
    *out = parsed;
    return true;
}

void rv_ble_token_mac_input(const rv_ble_token_t *token, uint8_t out[34])
{
    if (token == NULL || out == NULL) {
        return;
    }
    memcpy(out, RV_BLE_SERVICE_UUID_LE, 16u);
    out[16] = token->version;
    out[17] = token->key_id;
    write_le32(&out[18], token->epoch_min);
    write_le32(&out[22], token->nonce);
    memcpy(&out[26], token->ephemeral_id, RV_BLE_EPHEMERAL_ID_SIZE);
}

bool rv_ble_auth_tag_equal(const uint8_t lhs[RV_BLE_AUTH_TAG_SIZE],
                           const uint8_t rhs[RV_BLE_AUTH_TAG_SIZE])
{
    if (lhs == NULL || rhs == NULL) {
        return false;
    }
#ifdef ESP_PLATFORM
    return mbedtls_ct_memcmp(lhs, rhs, RV_BLE_AUTH_TAG_SIZE) == 0;
#else
    uint8_t diff = 0u;
    for (size_t i = 0u; i < RV_BLE_AUTH_TAG_SIZE; i++) {
        diff |= (uint8_t)(lhs[i] ^ rhs[i]);
    }
    return diff == 0u;
#endif
}

bool rv_ble_serialize_telemetry(const rv_ble_telemetry_t *telemetry,
                                uint8_t *out,
                                size_t out_len)
{
    if (telemetry == NULL || out == NULL || out_len < RV_BLE_TELEMETRY_SIZE
        || telemetry->ttl_ms == 0u
        || telemetry->confidence_permille > 1000u
        || telemetry->rssi_dbm == 127
        || (telemetry->flags & ~RV_BLE_FLAGS_ALLOWED) != 0u
        || (telemetry->flags & RV_BLE_FLAG_AUTHENTICATED) == 0u) {
        return false;
    }

    memset(out, 0, RV_BLE_TELEMETRY_SIZE);
    write_le32(&out[0], RV_BLE_TELEMETRY_MAGIC);
    out[4] = RV_BLE_TELEMETRY_VERSION;
    out[5] = telemetry->node_id;
    out[6] = telemetry->flags;
    out[7] = telemetry->key_id;
    write_le32(&out[8], telemetry->sequence);
    write_le32(&out[12], telemetry->observed_at_ms);
    write_le16(&out[16], telemetry->ttl_ms);
    write_le16(&out[18], telemetry->confidence_permille);
    out[20] = (uint8_t)telemetry->rssi_dbm;
    out[21] = (uint8_t)telemetry->tx_power_dbm;
    /* bytes 22..23 are reserved and remain zero */
    memcpy(&out[24], telemetry->ephemeral_id, RV_BLE_EPHEMERAL_ID_SIZE);
    write_le32(&out[32], telemetry->token_epoch_min);
    return true;
}

/**
 * @file ble_identity_protocol.h
 * @brief Pure-C RuView BLE identity-token and telemetry wire contracts.
 *
 * This module deliberately contains no ESP-IDF dependency so the untrusted
 * advertising-data boundary can be exercised by host unit tests.  It parses
 * only RuView's vendor 128-bit service-data record.  It never exports a BLE
 * address or general advertising payload.
 */

#ifndef BLE_IDENTITY_PROTOCOL_H
#define BLE_IDENTITY_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RV_BLE_AD_TYPE_SERVICE_DATA_UUID128 0x21u
#define RV_BLE_TOKEN_VERSION                1u
#define RV_BLE_TOKEN_BODY_SIZE              34u
#define RV_BLE_SERVICE_DATA_SIZE            50u
#define RV_BLE_EPHEMERAL_ID_SIZE             8u
#define RV_BLE_AUTH_TAG_SIZE                16u

#define RV_BLE_TELEMETRY_MAGIC       0xC51100B1u
#define RV_BLE_TELEMETRY_VERSION            1u
#define RV_BLE_TELEMETRY_SIZE              36u

#define RV_BLE_FLAG_AUTHENTICATED       (1u << 0)
#define RV_BLE_FLAG_TIME_VERIFIED       (1u << 1)
#define RV_BLE_FLAG_EXTENDED_ADVERT     (1u << 2)
#define RV_BLE_FLAGS_ALLOWED (RV_BLE_FLAG_AUTHENTICATED | \
                              RV_BLE_FLAG_TIME_VERIFIED | \
                              RV_BLE_FLAG_EXTENDED_ADVERT)

/** Raw little-endian UUID bytes as they appear in BLE service data. */
extern const uint8_t RV_BLE_SERVICE_UUID_LE[16];

/** Authenticated, rotating token extracted from RuView service data. */
typedef struct {
    uint8_t  version;
    uint8_t  key_id;
    uint32_t epoch_min;
    uint32_t nonce;
    uint8_t  ephemeral_id[RV_BLE_EPHEMERAL_ID_SIZE];
    uint8_t  auth_tag[RV_BLE_AUTH_TAG_SIZE];
} rv_ble_token_t;

/** Privacy-minimized observation forwarded to the RuView host. */
typedef struct {
    uint8_t  node_id;
    uint8_t  flags;
    uint8_t  key_id;
    uint32_t sequence;
    uint32_t observed_at_ms;
    uint16_t ttl_ms;
    uint16_t confidence_permille;
    int8_t   rssi_dbm;
    int8_t   tx_power_dbm;
    uint8_t  ephemeral_id[RV_BLE_EPHEMERAL_ID_SIZE];
    uint32_t token_epoch_min;
} rv_ble_telemetry_t;

/**
 * Find and parse the RuView 128-bit service-data record from one advertising
 * report.  Unknown AD elements are skipped.  Truncation, duplicate RuView
 * elements, wrong UUIDs, or unsupported token versions fail closed.
 */
bool rv_ble_parse_advertisement(const uint8_t *advertisement,
                                size_t advertisement_len,
                                rv_ble_token_t *out);

/**
 * Return the bytes covered by the token HMAC.
 *
 * The covered bytes are the raw 16-byte UUID followed by token version,
 * key-id, epoch, nonce and ephemeral id.  The authentication tag itself is
 * excluded.  The caller supplies a 34-byte output buffer.
 */
void rv_ble_token_mac_input(const rv_ble_token_t *token, uint8_t out[34]);

/** Constant-time comparison for the truncated authentication tag. */
bool rv_ble_auth_tag_equal(const uint8_t lhs[RV_BLE_AUTH_TAG_SIZE],
                           const uint8_t rhs[RV_BLE_AUTH_TAG_SIZE]);

/** Serialize one telemetry packet in the fixed little-endian wire format. */
bool rv_ble_serialize_telemetry(const rv_ble_telemetry_t *telemetry,
                                uint8_t *out,
                                size_t out_len);

#endif /* BLE_IDENTITY_PROTOCOL_H */

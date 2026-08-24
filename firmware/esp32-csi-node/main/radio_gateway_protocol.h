/**
 * @file radio_gateway_protocol.h
 * @brief Authenticated gateway envelope for privacy-sensitive radio evidence.
 */

#ifndef RADIO_GATEWAY_PROTOCOL_H
#define RADIO_GATEWAY_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RV_GATEWAY_MAGIC                 0x45415652u /* "RVAE" little endian */
#define RV_GATEWAY_VERSION               1u
#define RV_GATEWAY_HEADER_SIZE          40u
#define RV_GATEWAY_AUTH_TAG_SIZE        16u
#define RV_GATEWAY_MAC_DOMAIN_SIZE      12u
#define RV_GATEWAY_MAX_PAYLOAD_SIZE     72u
#define RV_GATEWAY_MAX_SIGNED_SIZE \
    (RV_GATEWAY_HEADER_SIZE + RV_GATEWAY_MAX_PAYLOAD_SIZE)
#define RV_GATEWAY_MAX_FRAME_SIZE \
    (RV_GATEWAY_MAX_SIGNED_SIZE + RV_GATEWAY_AUTH_TAG_SIZE)

#define RV_GATEWAY_PAYLOAD_BLE_IDENTITY      1u
#define RV_GATEWAY_PAYLOAD_CHANNEL_SOUNDING  2u

#define RV_GATEWAY_FLAG_RX_MONOTONIC (1u << 0)
#define RV_GATEWAY_FLAGS_ALLOWED RV_GATEWAY_FLAG_RX_MONOTONIC

/** HMAC domain separator; not NUL terminated on the wire. */
extern const uint8_t RV_GATEWAY_MAC_DOMAIN[RV_GATEWAY_MAC_DOMAIN_SIZE];

/** Metadata captured by the ESP32 gateway before asynchronous UDP egress. */
typedef struct {
    uint8_t payload_type;
    uint8_t flags;
    uint8_t key_id;
    uint8_t node_id;
    uint32_t sequence;
    uint64_t boot_nonce;
    uint64_t received_at_boot_us;
    uint32_t timing_uncertainty_us;
} rv_gateway_metadata_t;

/**
 * Build the authenticated prefix and reserve a trailing 16-byte tag.
 *
 * The caller HMACs `RV_GATEWAY_MAC_DOMAIN || out[0..signed_len]`, copies the
 * first 16 digest bytes to `out[signed_len..frame_len]`, then sends exactly
 * `frame_len` bytes.
 */
bool rv_gateway_build_unsigned(const rv_gateway_metadata_t *metadata,
                               const uint8_t *payload,
                               size_t payload_len,
                               uint8_t *out,
                               size_t out_capacity,
                               size_t *signed_len,
                               size_t *frame_len);

/** Constant-time comparison for a truncated gateway authentication tag. */
bool rv_gateway_auth_tag_equal(const uint8_t lhs[RV_GATEWAY_AUTH_TAG_SIZE],
                               const uint8_t rhs[RV_GATEWAY_AUTH_TAG_SIZE]);

#endif /* RADIO_GATEWAY_PROTOCOL_H */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ble_identity_protocol.h"
#include "channel_sounding_protocol.h"
#include "radio_gateway_protocol.h"

static void put16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void put32(uint8_t *p, uint32_t value)
{
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static size_t build_ble_advert(uint8_t *advert, size_t cap)
{
    const size_t total = RV_BLE_SERVICE_DATA_SIZE + 2u;
    assert(cap >= total);
    advert[0] = (uint8_t)(RV_BLE_SERVICE_DATA_SIZE + 1u);
    advert[1] = RV_BLE_AD_TYPE_SERVICE_DATA_UUID128;
    memcpy(&advert[2], RV_BLE_SERVICE_UUID_LE, 16u);
    uint8_t *body = &advert[18];
    body[0] = RV_BLE_TOKEN_VERSION;
    body[1] = 7u;
    put32(&body[2], 30000000u);
    put32(&body[6], 42u);
    for (size_t i = 0u; i < RV_BLE_EPHEMERAL_ID_SIZE; i++) body[10 + i] = (uint8_t)(0xa0u + i);
    for (size_t i = 0u; i < RV_BLE_AUTH_TAG_SIZE; i++) body[18 + i] = (uint8_t)(0x10u + i);
    return total;
}

static void build_cs_frame(uint8_t frame[RV_CS_FRAME_SIZE])
{
    memset(frame, 0, RV_CS_FRAME_SIZE);
    put32(&frame[0], RV_CS_MAGIC);
    frame[4] = RV_CS_VERSION;
    frame[5] = RV_CS_FLAG_CALIBRATED;
    frame[6] = 4u;
    frame[7] = 0u;
    put16(&frame[8], RV_CS_FRAME_SIZE);
    put16(&frame[10], 37u);
    put32(&frame[12], 9u);
    put32(&frame[16], 10000u);
    put32(&frame[20], 0x11223344u);
    put16(&frame[24], 800u);
    put16(&frame[26], 250u);
    put32(&frame[28], (uint32_t)-1200);
    put32(&frame[32], 50000u);
    put32(&frame[36], (uint32_t)-900);
    put32(&frame[40], 0x55667788u);
    put32(&frame[44], 0x12345678u);
    put16(&frame[48], 2u);
    put16(&frame[50], 8u);
    for (size_t i = 0u; i < RV_CS_AUTH_TAG_SIZE; i++) frame[52 + i] = (uint8_t)i;
    put32(&frame[68], rv_cs_crc32(frame, 68u));
}

static void test_ble_parser_and_privacy_telemetry(void)
{
    uint8_t advert[64];
    size_t advert_len = build_ble_advert(advert, sizeof(advert));
    rv_ble_token_t token;
    assert(rv_ble_parse_advertisement(advert, advert_len, &token));
    assert(token.key_id == 7u);
    assert(token.epoch_min == 30000000u);
    assert(token.ephemeral_id[0] == 0xa0u);
    assert(token.auth_tag[15] == 0x1fu);

    uint8_t tampered[64];
    memcpy(tampered, advert, advert_len);
    tampered[0] = 63u; /* field extends beyond report */
    assert(!rv_ble_parse_advertisement(tampered, advert_len, &token));
    memcpy(tampered, advert, advert_len);
    tampered[18] = 2u; /* unsupported token version */
    assert(!rv_ble_parse_advertisement(tampered, advert_len, &token));

    rv_ble_telemetry_t telemetry = {
        .node_id = 3u,
        .flags = RV_BLE_FLAG_AUTHENTICATED | RV_BLE_FLAG_TIME_VERIFIED,
        .key_id = 7u,
        .sequence = 11u,
        .observed_at_ms = 1200u,
        .ttl_ms = 3000u,
        .confidence_permille = 850u,
        .rssi_dbm = -61,
        .tx_power_dbm = 127,
        .token_epoch_min = 30000000u,
    };
    memcpy(telemetry.ephemeral_id, token.ephemeral_id, 8u);
    uint8_t packet[RV_BLE_TELEMETRY_SIZE];
    assert(rv_ble_serialize_telemetry(&telemetry, packet, sizeof(packet)));
    assert(RV_BLE_TELEMETRY_MAGIC != 0xC5110005u); /* compressed CSI */
    assert(packet[0] == 0xb1u && packet[1] == 0x00u
           && packet[2] == 0x11u && packet[3] == 0xc5u);
    assert(packet[4] == RV_BLE_TELEMETRY_VERSION);
    assert(packet[5] == 3u);
    assert(packet[20] == (uint8_t)-61);
    /* The packet has no BLE address, raw advertising data, nonce or HMAC. */
    assert(memcmp(&packet[24], token.ephemeral_id, 8u) == 0);

    telemetry.flags = 0u;
    assert(!rv_ble_serialize_telemetry(&telemetry, packet, sizeof(packet)));
}

static void test_channel_sounding_validation(void)
{
    uint8_t frame[RV_CS_FRAME_SIZE];
    build_cs_frame(frame);
    rv_cs_measurement_t measurement;
    assert(rv_cs_parse_frame(frame, sizeof(frame), 2000000u,
                             600u, &measurement) == RV_CS_PARSE_OK);
    assert(measurement.source_id == 0x11223344u);
    assert(measurement.phase_millirad == -1200);
    assert(measurement.frequency_offset_hz == -900);
    assert(measurement.source_session_id == 0x55667788u);
    assert(measurement.procedure_id == 0x12345678u);
    assert(measurement.step_index == 2u);
    assert(measurement.step_count == 8u);

    frame[28] ^= 1u;
    assert(rv_cs_parse_frame(frame, sizeof(frame), 2000000u,
                             600u, &measurement) == RV_CS_PARSE_BAD_CRC);

    build_cs_frame(frame);
    put32(&frame[16], 3000000u);
    put32(&frame[68], rv_cs_crc32(frame, 68u));
    assert(rv_cs_parse_frame(frame, sizeof(frame), 2000000u,
                             600u, &measurement) == RV_CS_PARSE_STALE);

    build_cs_frame(frame);
    uint8_t covered[RV_CS_MAC_INPUT_SIZE];
    rv_cs_mac_input(frame, covered);
    assert(memcmp(covered, "RuView/CS/v1", RV_CS_MAC_DOMAIN_SIZE) == 0);
    assert(memcmp(&covered[RV_CS_MAC_DOMAIN_SIZE], frame,
                  RV_CS_SIGNED_PREFIX_SIZE) == 0);

    assert(rv_cs_sequence_is_newer(10u, 9u));
    assert(!rv_cs_sequence_is_newer(9u, 9u));
    assert(rv_cs_sequence_is_newer(0u, UINT32_MAX));

    build_cs_frame(frame);
    put16(&frame[50], 1u);
    put32(&frame[68], rv_cs_crc32(frame, 68u));
    assert(rv_cs_parse_frame(frame, sizeof(frame), 2000000u,
                             600u, &measurement) == RV_CS_PARSE_BAD_STEP);
}

static void test_gateway_envelope_contract(void)
{
    uint8_t payload[RV_BLE_TELEMETRY_SIZE] = {0};
    uint8_t frame[RV_GATEWAY_MAX_FRAME_SIZE];
    size_t signed_len = 0u;
    size_t frame_len = 0u;
    rv_gateway_metadata_t metadata = {
        .payload_type = RV_GATEWAY_PAYLOAD_BLE_IDENTITY,
        .flags = RV_GATEWAY_FLAG_RX_MONOTONIC,
        .key_id = 3u,
        .node_id = 9u,
        .sequence = 11u,
        .boot_nonce = 0x0102030405060708ull,
        .received_at_boot_us = 9000u,
        .timing_uncertainty_us = 1000u,
    };
    assert(rv_gateway_build_unsigned(&metadata, payload, sizeof(payload),
                                     frame, sizeof(frame), &signed_len,
                                     &frame_len));
    assert(signed_len == RV_GATEWAY_HEADER_SIZE + sizeof(payload));
    assert(frame_len == signed_len + RV_GATEWAY_AUTH_TAG_SIZE);
    assert(memcmp(&frame[0], "RVAE", 4u) == 0);
    assert(frame[4] == RV_GATEWAY_VERSION);
    assert(frame[5] == RV_GATEWAY_PAYLOAD_BLE_IDENTITY);
    assert(frame[13] == 0u && frame[14] == 0u && frame[15] == 0u);

    metadata.sequence = 0u;
    assert(!rv_gateway_build_unsigned(&metadata, payload, sizeof(payload),
                                      frame, sizeof(frame), &signed_len,
                                      &frame_len));
    uint8_t lhs[RV_GATEWAY_AUTH_TAG_SIZE] = {0};
    uint8_t rhs[RV_GATEWAY_AUTH_TAG_SIZE] = {0};
    assert(rv_gateway_auth_tag_equal(lhs, rhs));
    rhs[15] = 1u;
    assert(!rv_gateway_auth_tag_equal(lhs, rhs));
}

int main(void)
{
    test_ble_parser_and_privacy_telemetry();
    test_channel_sounding_validation();
    test_gateway_envelope_contract();
    puts("BLE identity and external Channel Sounding protocol tests passed (SYNTHETIC)");
    return 0;
}

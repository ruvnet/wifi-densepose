#include <check.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Pull in the production parser */
#include "firmware/esp32-csi-node/main/rvf_parser.c"

/*
 * Security invariant: A party who knows the public key (which is public by
 * definition) MUST NOT be able to forge a valid firmware signature.
 * Specifically, SHA-256(pubkey || data) used as a "signature" is forgeable
 * by anyone who holds the public key, so the verifier must reject such
 * forgeries — or the scheme must be replaced entirely.
 *
 * We assert that crafting the signature field as SHA-256(pubkey || payload)
 * does NOT produce a verification success, because that would confirm the
 * vulnerability: the "signature" is trivially forgeable with public material.
 */

/* Minimal RVF buffer large enough for the parser */
#define RVF_BUF_SIZE 512

static void build_forged_packet(uint8_t *buf, size_t buf_len,
                                const uint8_t *pubkey,
                                const uint8_t *payload, size_t payload_len)
{
    memset(buf, 0, buf_len);
    /* Place pubkey at expected offset (adjust if parser layout differs) */
    memcpy(buf, pubkey, 32);
    /* Place payload after pubkey */
    memcpy(buf + 32, payload, payload_len < (buf_len - 96) ? payload_len : (buf_len - 96));

    /* Compute the "forged" signature: SHA-256(pubkey || signed_region) */
    uint8_t hash_input[32 + RVF_BUF_SIZE];
    memcpy(hash_input, pubkey, 32);
    memcpy(hash_input + 32, buf + 32, buf_len - 96);

    uint8_t forged_sig[32];
    mbedtls_sha256(hash_input, 32 + buf_len - 96, forged_sig, 0);

    /* Write forged signature into the signature field (last 64 bytes) */
    memcpy(buf + buf_len - 64, forged_sig, 32);
}

START_TEST(test_pubkey_forgery_rejected)
{
    /* Invariant: knowing the public key must not allow signature forgery */
    static const uint8_t known_pubkey[32] = {
        /* exact exploit: attacker uses the real public key */
        0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
        0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
        0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27
    };

    const uint8_t *payloads[] = {
        (const uint8_t *)"\xDE\xAD\xBE\xEF\xCA\xFE\xBA\xBE", /* exploit payload */
        (const uint8_t *)"\x00\x00\x00\x00\x00\x00\x00\x00", /* boundary: all zeros */
        (const uint8_t *)"\x41\x42\x43\x44\x45\x46\x47\x48", /* valid-looking data */
    };
    size_t payload_lens[] = { 8, 8, 8 };
    int num_payloads = 3;

    for (int i = 0; i < num_payloads; i++) {
        uint8_t buf[RVF_BUF_SIZE];
        build_forged_packet(buf, RVF_BUF_SIZE, known_pubkey,
                            payloads[i], payload_lens[i]);

        rvf_parsed_t parsed;
        esp_err_t ret = rvf_parse(buf, RVF_BUF_SIZE, known_pubkey, &parsed);

        /*
         * A forged signature constructed from public material MUST be rejected.
         * If this assertion fails, the scheme is trivially broken.
         */
        ck_assert_msg(ret != ESP_OK,
            "SECURITY FAILURE: forged signature (payload %d) was accepted — "
            "SHA-256(pubkey||data) is not a secure signature scheme", i);
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *
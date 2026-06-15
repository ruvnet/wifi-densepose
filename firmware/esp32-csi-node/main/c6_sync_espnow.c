/**
 * @file c6_sync_espnow.c
 * @brief ESP-NOW cross-node time-sync — ADR-110 D1 workaround.
 *
 * Same protocol as c6_timesync.c (TS_BEACON every 100 ms with leader epoch),
 * but over ESP-NOW instead of 802.15.4 because the IDF v5.4 ieee802154 RX
 * path doesn't deliver frames to user-space (see WITNESS-LOG-110 §D1).
 *
 * Frame layout (24 bytes payload, broadcast MAC FF:FF:FF:FF:FF:FF):
 *   [0..3]   Magic         0x53454E50  ('SENP' — Sync via ESP-NOW)
 *   [4]      Protocol ver  0x01
 *   [5]      Leader flag   1 if sender claims leader
 *   [6]      Node ID
 *   [7]      Battery percent, 0-100 valid, 255 unknown
 *   [8..15]  Leader epoch µs (LE u64)
 *   [16]     Battery flags: bit0=valid, bit1=charging
 *   [17]     Battery status
 *   [18..19] Battery millivolts
 *   [20..23] Reserved
 */

#include "sdkconfig.h"
#include "c6_sync_espnow.h"
#include "battery_monitor.h"
#include "csi_collector.h"
#include "esp_log.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/timers.h"
#include <string.h>

static const char *TAG = "c6_espnow";

#define BEACON_MAGIC      0x53454E50u   /* 'SENP' little-endian */
#define BEACON_PROTO_VER  0x01
#define BEACON_PERIOD_MS  100
#define VALID_WINDOW_MS   3000
#define BEACON_LEGACY_SIZE 16

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t  proto_ver;
    uint8_t  leader_flag;
    uint8_t  node_id;
    uint8_t  battery_percent;
    uint64_t leader_epoch_us;
    uint8_t  battery_flags;
    uint8_t  battery_status;
    uint16_t battery_millivolts;
    uint32_t _reserved;
} espnow_beacon_t;

static const uint8_t s_broadcast_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static uint64_t s_local_id     = 0;   /* 6-byte MAC packed into u64 */
static uint64_t s_leader_id    = 0;
static int64_t  s_offset_us    = 0;
static uint64_t s_last_seen_us = 0;
static bool     s_is_leader    = false;
static TimerHandle_t s_beacon_timer = NULL;

static uint32_t s_tx_count = 0;
static uint32_t s_tx_fail  = 0;
static uint32_t s_rx_count = 0;
static uint32_t s_rx_magic_match = 0;
static c6_espnow_peer_status_t s_peer_status = {0};
static uint64_t s_peer_seen_us = 0;
static uint8_t s_cached_battery_percent = 255;
static uint8_t s_cached_battery_flags = 0;
static uint8_t s_cached_battery_status = BATTERY_POWER_UNKNOWN;
static uint16_t s_cached_battery_mv = 0;
static uint64_t s_last_battery_refresh_us = 0;

/* ADR-110 P10 — EMA-smoothed offset (host-side trajectory in firmware).
 *
 * The §A0.8 four-minute soak measured 540 µs sample-stdev around a true
 * offset that drifts at ≈1.4 ppm between two C6 crystals. An exponential
 * moving average with α=0.125 (Q3.3 fixed-point shift = 3) yields an
 * effective ~8-sample window, fast enough to track the drift (~7 µs/sec
 * worst-case) while suppressing the per-beacon WiFi-MAC jitter.
 *
 * Two consumers: get_offset_us() (raw, unchanged — for diagnostics) and
 * get_offset_us_smoothed() (filtered — what CSI frames should stamp).
 * Both expose `int64_t` so call sites stay identical. */
#define OFFSET_EMA_SHIFT 3           /* α = 1/8 = 0.125 */
static int64_t s_offset_us_smoothed = 0;
static bool    s_smoothed_seeded    = false;

static uint64_t mac6_to_u64(const uint8_t mac[6])
{
    return ((uint64_t)mac[0] << 40) | ((uint64_t)mac[1] << 32) |
           ((uint64_t)mac[2] << 24) | ((uint64_t)mac[3] << 16) |
           ((uint64_t)mac[4] <<  8) |  (uint64_t)mac[5];
}

static void refresh_battery_cache(void)
{
    uint64_t now_us = (uint64_t)esp_timer_get_time();
    if ((now_us - s_last_battery_refresh_us) < 1000000ULL) return;
    s_last_battery_refresh_us = now_us;

    s_cached_battery_percent = 255;
    s_cached_battery_flags = 0;
    s_cached_battery_status = BATTERY_POWER_UNKNOWN;
    s_cached_battery_mv = 0;

    battery_status_t battery;
    if (battery_monitor_read(&battery) == ESP_OK && battery.valid) {
        s_cached_battery_percent = battery.percent;
        s_cached_battery_flags = 0x01;
        if (battery.charging) s_cached_battery_flags |= 0x02;
        s_cached_battery_status = (uint8_t)battery.status;
        s_cached_battery_mv = battery.millivolts;
    }
}

static void send_beacon(void)
{
    refresh_battery_cache();
    espnow_beacon_t b = {
        .magic           = BEACON_MAGIC,
        .proto_ver       = BEACON_PROTO_VER,
        .leader_flag     = s_is_leader ? 1 : 0,
        .node_id         = csi_collector_get_node_id(),
        .battery_percent = s_cached_battery_percent,
        .leader_epoch_us = (uint64_t)esp_timer_get_time(),
        .battery_flags   = s_cached_battery_flags,
        .battery_status  = s_cached_battery_status,
        .battery_millivolts = s_cached_battery_mv,
        ._reserved       = 0,
    };
    esp_err_t r = esp_now_send(s_broadcast_mac, (uint8_t *)&b, sizeof(b));
    s_tx_count++;
    if (r != ESP_OK) s_tx_fail++;
    /* Diag log every 50 beacons. */
    if ((s_tx_count % 50) == 1) {
        ESP_LOGI(TAG, "tx#%lu (fail=%lu) rx#%lu (match=%lu) leader=%d offset_us=%lld smoothed=%lld",
                 (unsigned long)s_tx_count, (unsigned long)s_tx_fail,
                 (unsigned long)s_rx_count, (unsigned long)s_rx_magic_match,
                 (int)s_is_leader, (long long)s_offset_us,
                 (long long)s_offset_us_smoothed);
    }
}

/* IDF v5.4 ESP-NOW recv callback signature uses esp_now_recv_info_t.
 * Falls back to the older signature on older IDF via ifdef. */
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
static void on_recv(const esp_now_recv_info_t *info,
                    const uint8_t *data, int len)
{
    const uint8_t *src_mac = info ? info->src_addr : NULL;
#else
static void on_recv(const uint8_t *src_mac, const uint8_t *data, int len)
{
#endif
    s_rx_count++;
    if (data == NULL || len < BEACON_LEGACY_SIZE) return;
    const espnow_beacon_t *b = (const espnow_beacon_t *)data;
    if (b->magic != BEACON_MAGIC || b->proto_ver != BEACON_PROTO_VER) return;
    s_rx_magic_match++;
    uint64_t sender_id = src_mac ? mac6_to_u64(src_mac) : 0;
    uint64_t now_us    = (uint64_t)esp_timer_get_time();
    bool has_status_ext = len >= (int)sizeof(espnow_beacon_t);

    if (sender_id != 0 && sender_id != s_local_id) {
        s_peer_status.valid = true;
        s_peer_status.node_id = has_status_ext ? b->node_id : 0;
        s_peer_status.percent = has_status_ext ? b->battery_percent : 255;
        s_peer_status.flags = has_status_ext ? b->battery_flags : 0;
        s_peer_status.status = has_status_ext ? b->battery_status : BATTERY_POWER_UNKNOWN;
        s_peer_status.millivolts = has_status_ext ? b->battery_millivolts : 0;
        s_peer_status.rx_count = s_rx_magic_match;
        s_peer_seen_us = now_us;
    }

    /* Adopt sender as leader if it's claiming leadership AND its ID is
     * lower than our current leader (or we have no leader). Lowest MAC
     * wins — deterministic. */
    if (b->leader_flag && (s_leader_id == 0 || sender_id < s_leader_id)) {
        if (s_is_leader && sender_id < s_local_id) {
            ESP_LOGI(TAG, "stepping down: heard lower-id leader %012llx (we are %012llx)",
                     (unsigned long long)sender_id, (unsigned long long)s_local_id);
            s_is_leader = false;
        }
        s_leader_id = sender_id;
    }

    /* If accepted leader, compute offset from their epoch (only for non-leader). */
    if (b->leader_flag && !s_is_leader && sender_id == s_leader_id) {
        int64_t raw = (int64_t)b->leader_epoch_us - (int64_t)now_us;
        s_offset_us    = raw;
        s_last_seen_us = now_us;
        /* EMA: y[n] = y[n-1] + (raw - y[n-1]) >> SHIFT */
        if (!s_smoothed_seeded) {
            s_offset_us_smoothed = raw;
            s_smoothed_seeded    = true;
        } else {
            s_offset_us_smoothed += (raw - s_offset_us_smoothed) >> OFFSET_EMA_SHIFT;
        }
    }
}

static void on_send(const uint8_t *mac, esp_now_send_status_t status)
{
    (void)mac;
    if (status != ESP_NOW_SEND_SUCCESS) s_tx_fail++;
}

static void beacon_timer_cb(TimerHandle_t t)
{
    (void)t;
    uint64_t now = (uint64_t)esp_timer_get_time();
    /* Promote self if no leader beacon for VALID_WINDOW_MS and we have lowest known id. */
    if (!s_is_leader && (now - s_last_seen_us) > (VALID_WINDOW_MS * 1000ULL)) {
        if (s_leader_id == 0 || s_local_id < s_leader_id) {
            s_is_leader  = true;
            s_leader_id  = s_local_id;
            s_offset_us  = 0;
            ESP_LOGI(TAG, "promoting self to leader (no beacons for %u ms; local_id=%012llx)",
                     (unsigned)VALID_WINDOW_MS, (unsigned long long)s_local_id);
        }
    }
    send_beacon();
}

esp_err_t c6_sync_espnow_init(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);
    s_local_id = mac6_to_u64(mac);

    esp_err_t r = esp_now_init();
    if (r != ESP_OK) {
        ESP_LOGE(TAG, "esp_now_init failed: %s", esp_err_to_name(r));
        return r;
    }
    esp_now_register_recv_cb(on_recv);
    esp_now_register_send_cb(on_send);

    /* Add broadcast peer so esp_now_send to FF:FF:FF:FF:FF:FF works. */
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, s_broadcast_mac, 6);
    peer.channel = 0;        /* current STA channel */
    peer.ifidx   = WIFI_IF_STA;
    peer.encrypt = false;
    r = esp_now_add_peer(&peer);
    if (r != ESP_OK && r != ESP_ERR_ESPNOW_EXIST) {
        ESP_LOGW(TAG, "esp_now_add_peer(broadcast) failed: %s", esp_err_to_name(r));
    }

    /* Start as candidate leader — will step down on receiving lower-id beacon. */
    s_is_leader    = true;
    s_leader_id    = s_local_id;
    s_last_seen_us = (uint64_t)esp_timer_get_time();

    s_beacon_timer = xTimerCreate("c6_espnow_beacon",
                                  pdMS_TO_TICKS(BEACON_PERIOD_MS),
                                  pdTRUE, NULL, beacon_timer_cb);
    if (s_beacon_timer == NULL) {
        ESP_LOGE(TAG, "xTimerCreate failed");
        return ESP_ERR_NO_MEM;
    }
    xTimerStart(s_beacon_timer, 0);

    ESP_LOGI(TAG, "init done: local_id=%012llx leader=yes(candidate) period=%ums",
             (unsigned long long)s_local_id, (unsigned)BEACON_PERIOD_MS);
    return ESP_OK;
}

uint64_t c6_sync_espnow_get_epoch_us(void)
{
    /* Prefer the smoothed offset once we've heard a leader beacon; falls
     * back to raw=0 on the leader board and during the first second after
     * follower boot. The smoothed value is what CSI frames should stamp
     * for cross-board multistatic alignment (§A0.8 measured 540 µs raw
     * stdev → expected <100 µs smoothed with α=1/8 over ~8 samples). */
    int64_t off = s_smoothed_seeded ? s_offset_us_smoothed : s_offset_us;
    return (uint64_t)((int64_t)esp_timer_get_time() + off);
}

bool c6_sync_espnow_is_leader(void) { return s_is_leader; }
int64_t c6_sync_espnow_get_offset_us(void) { return s_offset_us; }
int64_t c6_sync_espnow_get_offset_us_smoothed(void) { return s_offset_us_smoothed; }

bool c6_sync_espnow_is_valid(void)
{
    if (s_is_leader) return true;
    uint64_t now = (uint64_t)esp_timer_get_time();
    return (now - s_last_seen_us) < (VALID_WINDOW_MS * 1000ULL);
}

uint32_t c6_sync_espnow_tx_count(void)       { return s_tx_count; }
uint32_t c6_sync_espnow_tx_fail(void)        { return s_tx_fail; }
uint32_t c6_sync_espnow_rx_count(void)       { return s_rx_count; }
uint32_t c6_sync_espnow_rx_magic_match(void) { return s_rx_magic_match; }

bool c6_sync_espnow_get_peer_status(c6_espnow_peer_status_t *out)
{
    if (!out) return false;
    if (!s_peer_status.valid) {
        memset(out, 0, sizeof(*out));
        return false;
    }
    *out = s_peer_status;
    uint64_t now_us = (uint64_t)esp_timer_get_time();
    out->age_ms = (uint32_t)((now_us - s_peer_seen_us) / 1000ULL);
    return out->age_ms < VALID_WINDOW_MS;
}

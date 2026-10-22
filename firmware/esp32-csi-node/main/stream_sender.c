/**
 * @file stream_sender.c
 * @brief UDP stream sender for CSI frames.
 *
 * Opens a UDP socket and sends serialized ADR-018 frames to the aggregator.
 */

#include "stream_sender.h"

#include <string.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "sdkconfig.h"

static const char *TAG = "stream_sender";

static int s_sock = -1;
static struct sockaddr_in s_dest_addr;
static char s_dest_ip[16] = "";
static uint16_t s_dest_port = 0;

/**
 * How long the bulk path has to go without accepting anything before an
 * ENOMEM is read as "the aggregator is not answering".
 *
 * A UDP datagram to an address whose ARP entry is unresolved is accepted by
 * the stack and parked in the ARP queue; the queue holds a handful of
 * packets, and once it is full every further sendto fails with ENOMEM
 * because lwIP cannot allocate for a neighbour it does not know. The errno
 * is indistinguishable from an exhausted pbuf pool, but the two have opposite
 * remedies, and the first is much the more common: the aggregator is on
 * another host and its address changed.
 *
 * A single accepted send proves nothing -- the packet may only be queued --
 * and the queue is drained and refilled by lwIP's own ARP retry, so the
 * number of accepted sends says nothing about reachability either: with an
 * unreachable destination the send still succeeds every few seconds, because
 * the stack keeps a slot free in the ARP queue. The sender therefore does not
 * try to tell the two causes apart: it reports the destination and how many
 * sends have failed, which is what the reader can act on.
 */
static uint32_t s_enomem_total = 0;    /* ENOMEM failures since boot */
static bool s_dest_silent_logged = false;

/**
 * ENOMEM backoff state.
 * When sendto fails with ENOMEM (errno 12), we suppress further sends for
 * a cooldown period to let lwIP reclaim packet buffers.  Without this,
 * rapid-fire CSI callbacks can exhaust the pbuf pool and crash the device.
 */
static int64_t s_backoff_until_us = 0;       /* esp_timer timestamp to resume */
#define ENOMEM_COOLDOWN_MS  100              /* base backoff; doubles per streak */
#define ENOMEM_COOLDOWN_MAX_MS 2000          /* cap on the exponential backoff */
#define ENOMEM_LOG_INTERVAL 50               /* log every Nth suppressed send */
static uint32_t s_enomem_suppressed = 0;
/* Consecutive ENOMEM episodes without an intervening successful send. A fixed
 * 100 ms backoff is too short to drain sustained lwIP/WiFi buffer pressure
 * (#1135 bug #1: tier-2 + concurrent TX keeps the node stuck), so the backoff
 * grows 100→200→400→…→2000 ms per streak and resets on the first send that
 * succeeds. */
static uint32_t s_enomem_streak = 0;

/* Timestamp of the last sendto() the stack accepted, on either path; 0 until
 * the node has ever delivered anything. Read by the uplink watchdog in main.c
 * -- see the rationale there for why "accepted by the stack" is deliberately
 * not "received by the server". */
static int64_t s_last_success_us = 0;

static int sender_init_internal(const char *ip, uint16_t port)
{
    s_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket: errno %d", errno);
        return -1;
    }

    memset(&s_dest_addr, 0, sizeof(s_dest_addr));
    s_dest_addr.sin_family = AF_INET;
    s_dest_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip, &s_dest_addr.sin_addr) <= 0) {
        ESP_LOGE(TAG, "Invalid target IP: %s", ip);
        close(s_sock);
        s_sock = -1;
        return -1;
    }

    ESP_LOGI(TAG, "UDP sender initialized: %s:%d", ip, port);
    strlcpy(s_dest_ip, ip, sizeof(s_dest_ip));
    s_dest_port = port;
    s_dest_silent_logged = false;
    return 0;
}

int stream_sender_init(void)
{
    return sender_init_internal(CONFIG_CSI_TARGET_IP, CONFIG_CSI_TARGET_PORT);
}

int stream_sender_init_with(const char *ip, uint16_t port)
{
    return sender_init_internal(ip, port);
}

int stream_sender_send(const uint8_t *data, size_t len)
{
    if (s_sock < 0) {
        return -1;
    }

    /* ENOMEM backoff: if we recently exhausted lwIP buffers, skip sends
     * until the cooldown expires.  This prevents the cascade of failed
     * sendto calls that leads to a guru meditation crash. */
    if (s_backoff_until_us > 0) {
        int64_t now = esp_timer_get_time();
        if (now < s_backoff_until_us) {
            s_enomem_suppressed++;
            if ((s_enomem_suppressed % ENOMEM_LOG_INTERVAL) == 1) {
                ESP_LOGW(TAG, "sendto suppressed (ENOMEM backoff, %lu dropped)",
                         (unsigned long)s_enomem_suppressed);
            }
            return -1;
        }
        /* Cooldown expired — resume sending */
        ESP_LOGI(TAG, "ENOMEM backoff expired, resuming sends (%lu were suppressed)",
                 (unsigned long)s_enomem_suppressed);
        s_backoff_until_us = 0;
        s_enomem_suppressed = 0;
    }

    int sent = sendto(s_sock, data, len, 0,
                      (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
    if (sent < 0) {
        if (errno == ENOMEM) {
            /* Exponential backoff: double the cooldown each consecutive ENOMEM
             * (capped) so sustained buffer pressure actually drains instead of
             * the node re-failing every 100 ms forever (#1135 bug #1). */
            uint32_t shift = s_enomem_streak < 5 ? s_enomem_streak : 5;
            uint32_t cooldown = ENOMEM_COOLDOWN_MS << shift;
            if (cooldown > ENOMEM_COOLDOWN_MAX_MS) cooldown = ENOMEM_COOLDOWN_MAX_MS;
            s_enomem_streak++;
            s_backoff_until_us = esp_timer_get_time() + (int64_t)cooldown * 1000;
            {
                int64_t now = esp_timer_get_time();
                s_enomem_total++;
                ESP_LOGW(TAG, "sendto ENOMEM to %s:%u - the aggregator may not be answering "
                              "on that address (ARP unresolved) or lwIP is out of packet "
                              "buffers; backing off for %lu ms (streak %lu)",
                         s_dest_ip, (unsigned int)s_dest_port,
                         (unsigned long)cooldown, (unsigned long)s_enomem_streak);
                if (!s_dest_silent_logged && s_enomem_total >= 20) {
                    /* One-shot: the repeated warning above says "ENOMEM", which
                     * reads as an out-of-memory condition. The node captures CSI
                     * normally; the frames have nowhere to go. */
                    s_dest_silent_logged = true;
                    ESP_LOGE(TAG, "%lu sends to %s:%u have failed since boot: check that "
                                  "the aggregator is still at that address. CSI capture is "
                                  "unaffected. If it moved, POST /config "
                                  "{\"target_ip\":\"<new address>\"} to change it.",
                             (unsigned long)s_enomem_total, s_dest_ip,
                             (unsigned int)s_dest_port);
                }
                (void)now;
            }
        } else {
            ESP_LOGW(TAG, "sendto failed: errno %d", errno);
        }
        return -1;
    }

    /* A send got through — buffer pressure cleared; reset the backoff streak. */
    if (s_enomem_streak > 0) {
        ESP_LOGI(TAG, "sendto to %s:%u succeeded after %lu failed attempts",
                 s_dest_ip, (unsigned int)s_dest_port, (unsigned long)s_enomem_streak);
    }
    s_enomem_streak = 0;
    s_dest_silent_logged = false;
    s_last_success_us = esp_timer_get_time();
    return sent;
}

int stream_sender_send_priority(const uint8_t *data, size_t len)
{
    if (s_sock < 0) {
        return -1;
    }

    /* Priority path (#1183): low-rate control packets (feature_state, HEALTH,
     * mesh sync) bypass the global ENOMEM backoff gate so the high-rate CSI
     * stream cannot starve them. These are ≤48 B at ≤1 Hz — negligible pbuf
     * pressure, so they won't re-trigger the crash cascade that the backoff
     * (driven by the 50 Hz CSI flood) exists to prevent.
     *
     * Crucially, an ENOMEM here is reported quietly and does NOT extend the
     * global streak/backoff: a tiny control packet failing is a symptom of
     * the bulk-stream pressure, not a cause, so it must not feed the cooldown
     * that suppresses the next CSI frame. Likewise a success does not reset
     * the streak — the bulk path owns that signal. */
    int sent = sendto(s_sock, data, len, 0,
                      (struct sockaddr *)&s_dest_addr, sizeof(s_dest_addr));
    if (sent < 0) {
        if (errno != ENOMEM) {
            ESP_LOGW(TAG, "priority sendto failed: errno %d", errno);
        }
        return -1;
    }
    /* Counts for the watchdog: the mesh sync packet rides this path at ~1 Hz,
     * and its getting through proves the node's network path is intact even
     * when the bulk CSI stream is being suppressed by the ENOMEM backoff. */
    s_last_success_us = esp_timer_get_time();
    return sent;
}

int64_t stream_sender_last_success_us(void)
{
    return s_last_success_us;
}

void stream_sender_deinit(void)
{
    if (s_sock >= 0) {
        close(s_sock);
        s_sock = -1;
        ESP_LOGI(TAG, "UDP sender closed");
    }
}

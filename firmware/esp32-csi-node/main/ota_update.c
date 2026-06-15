/**
 * @file ota_update.c
 * @brief HTTP OTA firmware update for ESP32-S3 CSI Node.
 *
 * Uses ESP-IDF's native OTA API with rollback support.
 * The HTTP server runs on port 8032 and accepts:
 *   POST /ota — firmware binary payload (application/octet-stream)
 *   GET /ota/status — current firmware version and partition info
 */

#include "ota_update.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_ota_ops.h"
#include "esp_http_server.h"
#include "esp_app_desc.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "battery_monitor.h"
#include "csi_collector.h"
#include "edge_processing.h"

static const char *TAG = "ota_update";

/** OTA HTTP server port. */
#define OTA_PORT 8032

/** Number of samples kept on the web graph. */
#define GRAPH_HISTORY_SAMPLES 96

/** NVS namespace and key for the OTA pre-shared key. */
#define OTA_NVS_NAMESPACE "security"
#define OTA_NVS_KEY       "ota_psk"

/** Maximum PSK length (hex-encoded SHA-256). */
#define OTA_PSK_MAX_LEN   65

/** Cached PSK loaded from NVS at init time. Empty = auth disabled. */
static char s_ota_psk[OTA_PSK_MAX_LEN] = {0};

static const char GRAPH_HTML[] = R"rawliteral(
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RuView Live Graph</title>
<style>
:root{
  color-scheme: dark;
  --bg:#060708;
  --panel:#0d1116;
  --panel2:#111822;
  --line:#263041;
  --text:#f2f5f8;
  --muted:#8d98a6;
  --green:#77ff7a;
  --cyan:#5fe1ff;
  --amber:#ffd36a;
  --red:#ff667d;
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:radial-gradient(circle at top, #101820 0%, var(--bg) 55%);color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
body{padding:18px}
.shell{max-width:1100px;margin:0 auto;display:grid;gap:14px}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:10px}
.title{font-size:28px;font-weight:700;letter-spacing:.04em}
.subtitle{color:var(--muted);font-size:12px;line-height:1.4}
.pill{padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.03);font-size:12px}
.pill.live{color:var(--green);border-color:rgba(119,255,122,.35)}
.pill.wait{color:var(--amber);border-color:rgba(255,211,106,.35)}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
.card{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:14px;padding:12px 14px;min-height:74px}
.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px}
.value{font-size:24px;line-height:1;font-weight:700}
.value.small{font-size:17px}
.canvasWrap{background:linear-gradient(180deg,var(--panel),#0a0d12);border:1px solid var(--line);border-radius:18px;padding:12px;box-shadow:0 18px 48px rgba(0,0,0,.25)}
canvas{display:block;width:100%;height:360px}
.foot{display:flex;flex-wrap:wrap;gap:10px;color:var(--muted);font-size:12px;line-height:1.5}
.foot span{padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.05)}
@media (max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}canvas{height:300px}}
@media (max-width:560px){body{padding:12px}.grid{grid-template-columns:1fr}.title{font-size:22px}canvas{height:240px}}
</style>
</head>
<body>
<div class="shell">
  <div class="top">
    <div>
      <div class="title">RuView Live Graph</div>
      <div class="subtitle">Browser view of the same live edge vitals that drive the S3 display graph.</div>
    </div>
    <div id="status" class="pill wait">WAITING</div>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Motion</div><div id="motion" class="value">--%</div></div>
    <div class="card"><div class="label">Presence</div><div id="presence" class="value">--%</div></div>
    <div class="card"><div class="label">RSSI</div><div id="rssi" class="value">-- dBm</div></div>
    <div class="card"><div class="label">People</div><div id="persons" class="value">--</div></div>
    <div class="card"><div class="label">Battery</div><div id="battery" class="value small">--</div></div>
  </div>

  <div class="canvasWrap">
    <canvas id="chart"></canvas>
  </div>

  <div class="foot">
    <span id="breathing">Breathing: -- BPM</span>
    <span id="heartrate">Heart: -- BPM</span>
    <span id="batteryState">Power: --</span>
    <span id="timestamp">Updated: --</span>
  </div>
</div>

<script>
(() => {
  const HISTORY = 96;
  const motion = [];
  const presence = [];
  const chart = document.getElementById('chart');
  const ctx = chart.getContext('2d');
  const status = document.getElementById('status');
  const $ = (id) => document.getElementById(id);

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const push = (arr, value) => {
    arr.push(value);
    while (arr.length > HISTORY) arr.shift();
    while (arr.length < HISTORY) arr.unshift(value);
  };

  const bpmText = (v) => (Number.isFinite(v) && v > 0 ? `${Math.round(v)} BPM` : '-- BPM');

  function fit() {
    const rect = chart.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(220, Math.floor(rect.height));
    const nextW = Math.floor(w * dpr);
    const nextH = Math.floor(h * dpr);
    if (chart.width !== nextW || chart.height !== nextH) {
      chart.width = nextW;
      chart.height = nextH;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  function drawGrid(w, h, pad) {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad + ((h - pad * 2) / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
    }
  }

  function drawSeries(data, color, w, h, pad, label) {
    if (data.length === 0) return;
    ctx.beginPath();
    data.forEach((value, idx) => {
      const x = pad + (idx * (w - pad * 2)) / Math.max(1, HISTORY - 1);
      const y = pad + (1 - clamp(value, 0, 100) / 100) * (h - pad * 2);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText(label, pad + 6, pad + 16);
  }

  function render(live) {
    fit();
    const w = chart.clientWidth;
    const h = chart.clientHeight;
    const pad = 18;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#050607';
    ctx.fillRect(0, 0, w, h);
    drawGrid(w, h, pad);
    drawSeries(presence, 'rgba(95,225,255,0.95)', w, h, pad, 'Presence');
    drawSeries(motion, 'rgba(119,255,122,0.95)', w, h, pad, 'Motion');

    if (!live) {
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = 'bold 18px ui-monospace, monospace';
      ctx.fillText('waiting for live vitals...', pad + 8, h / 2);
    }
  }

  async function tick() {
    try {
      const res = await fetch('/graph/data', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const live = !!data.live;

      status.textContent = live ? 'LIVE' : 'WAITING';
      status.className = `pill ${live ? 'live' : 'wait'}`;
      $('motion').textContent = `${data.motion ?? '--'}%`;
      $('presence').textContent = `${data.presence ?? '--'}%`;
      $('rssi').textContent = Number.isFinite(data.rssi) ? `${Math.round(data.rssi)} dBm` : '-- dBm';
      $('persons').textContent = Number.isFinite(data.persons) ? String(data.persons) : '--';
      $('battery').textContent = Number.isFinite(data.battery_percent) && data.battery_percent >= 0
        ? `${data.battery_percent}% / ${data.battery_mv}mV`
        : 'N/A';
      $('breathing').textContent = `Breathing: ${bpmText(data.breathing_bpm)}`;
      $('heartrate').textContent = `Heart: ${bpmText(data.heartrate_bpm)}`;
      $('batteryState').textContent = `Power: ${data.battery_status || '--'}`;
      $('timestamp').textContent = `Updated: ${Number.isFinite(data.timestamp_ms) ? `${Math.round(data.timestamp_ms)} ms` : '--'}`;

      push(motion, Number.isFinite(data.motion) ? data.motion : 0);
      push(presence, Number.isFinite(data.presence) ? data.presence : 0);
      render(live);
    } catch (err) {
      status.textContent = 'WAITING';
      status.className = 'pill wait';
      render(false);
    }
  }

  window.addEventListener('resize', () => render(status.textContent === 'LIVE'));
  fit();
  render(false);
  tick();
  setInterval(tick, 250);
})();
</script>
</body>
</html>
)rawliteral";

static int clamp_int(int value, int lo, int hi)
{
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static int scaled_motion_percent(const edge_vitals_pkt_t *vitals)
{
    if (!vitals) return 0;
    return clamp_int((int)(vitals->motion_energy * 18.0f), 0, 100);
}

static int scaled_presence_percent(const edge_vitals_pkt_t *vitals)
{
    if (!vitals) return 0;
    return clamp_int((int)(vitals->presence_score * 18.0f), 0, 100);
}

static void build_graph_snapshot(char *response, size_t response_len)
{
    edge_vitals_pkt_t vitals;
    bool has_vitals = edge_get_vitals(&vitals);

    battery_status_t battery = {0};
    esp_err_t battery_ret = battery_monitor_read(&battery);
    bool battery_live = (battery_ret == ESP_OK && battery.valid);

    int motion = has_vitals ? scaled_motion_percent(&vitals) : 0;
    int presence = has_vitals ? scaled_presence_percent(&vitals) : 0;
    int breathing = has_vitals ? (int)(vitals.breathing_rate / 100U) : 0;
    int heartrate = has_vitals ? (int)(vitals.heartrate / 10000U) : 0;
    int rssi = has_vitals ? (int)vitals.rssi : 0;
    int persons = has_vitals ? (int)vitals.n_persons : 0;
    int battery_percent = battery_live ? (int)battery.percent : -1;
    int battery_mv = battery_live ? (int)battery.millivolts : -1;
    const char *battery_status = battery_live ? battery_monitor_status_name(battery.status) : "UNKNOWN";
    uint32_t timestamp_ms = has_vitals ? vitals.timestamp_ms : 0;
    uint8_t node_id = has_vitals ? vitals.node_id : csi_collector_get_node_id();

    snprintf(response, response_len,
             "{\"live\":%s,\"node_id\":%u,\"motion\":%d,\"presence\":%d,"
             "\"breathing_bpm\":%d,\"heartrate_bpm\":%d,\"rssi\":%d,\"persons\":%d,"
             "\"battery_percent\":%d,\"battery_mv\":%d,\"battery_status\":\"%s\","
             "\"timestamp_ms\":%lu}",
             has_vitals ? "true" : "false",
             (unsigned)node_id,
             motion, presence,
             breathing, heartrate, rssi, persons,
             battery_percent, battery_mv, battery_status,
             (unsigned long)timestamp_ms);
}

static esp_err_t graph_page_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_send(req, GRAPH_HTML, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

static esp_err_t graph_data_handler(httpd_req_t *req)
{
    char response[384];
    build_graph_snapshot(response, sizeof(response));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

/**
 * ADR-050: Verify the Authorization header contains the correct PSK.
 * Returns true only when a PSK is provisioned AND the Bearer token
 * matches it. An unprovisioned node refuses all OTA requests
 * (fail-closed, see RuView#596 audit). The OTA server still starts so
 * the operator can `provision.py --ota-psk <hex>` over USB-CDC without
 * a reflash, but the upload endpoint will reject every request until
 * the PSK is set.
 */
static bool ota_check_auth(httpd_req_t *req)
{
    if (s_ota_psk[0] == '\0') {
        /* No PSK provisioned — fail closed. Previously this returned
         * true ("permissive for dev"), which let any host on the WiFi
         * push attacker-controlled firmware to a freshly-flashed node.
         * Plain HTTP transport + no Secure Boot V2 + no signed-image
         * verification meant a single LAN call could brick or back-
         * door a node. Reject until provisioned. */
        ESP_LOGW(TAG, "OTA rejected: no PSK in NVS (run provision.py --ota-psk <hex>)");
        return false;
    }

    char auth_header[128] = {0};
    if (httpd_req_get_hdr_value_str(req, "Authorization", auth_header,
                                     sizeof(auth_header)) != ESP_OK) {
        return false;
    }

    /* Expect "Bearer <psk>" */
    const char *prefix = "Bearer ";
    if (strncmp(auth_header, prefix, strlen(prefix)) != 0) {
        return false;
    }

    const char *token = auth_header + strlen(prefix);
    /* Constant-time comparison to prevent timing attacks. */
    size_t psk_len = strlen(s_ota_psk);
    size_t tok_len = strlen(token);
    if (psk_len != tok_len) return false;
    volatile uint8_t result = 0;
    for (size_t i = 0; i < psk_len; i++) {
        result |= (uint8_t)(s_ota_psk[i] ^ token[i]);
    }
    return result == 0;
}

/**
 * GET /ota/status — return firmware version and partition info.
 */
static esp_err_t ota_status_handler(httpd_req_t *req)
{
    const esp_app_desc_t *app = esp_app_get_description();
    const esp_partition_t *running = esp_ota_get_running_partition();
    const esp_partition_t *update = esp_ota_get_next_update_partition(NULL);

    char response[512];
    int len = snprintf(response, sizeof(response),
        "{\"version\":\"%s\",\"date\":\"%s\",\"time\":\"%s\","
        "\"running_partition\":\"%s\",\"next_partition\":\"%s\","
        "\"max_size\":%lu}",
        app->version, app->date, app->time,
        running ? running->label : "unknown",
        update ? update->label : "none",
        (unsigned long)(update ? update->size : 0));

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, response, len);
    return ESP_OK;
}

/**
 * POST /ota — receive and flash firmware binary.
 */
static esp_err_t ota_upload_handler(httpd_req_t *req)
{
    /* ADR-050: Authenticate before accepting firmware upload. */
    if (!ota_check_auth(req)) {
        ESP_LOGW(TAG, "OTA upload rejected: authentication failed");
        httpd_resp_send_err(req, HTTPD_403_FORBIDDEN,
                            "Authentication required. Use: Authorization: Bearer <psk>");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "OTA update started, content_length=%d", req->content_len);

    const esp_partition_t *update_partition = esp_ota_get_next_update_partition(NULL);
    if (update_partition == NULL) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "No OTA partition available");
        return ESP_FAIL;
    }

    if (req->content_len <= 0 || req->content_len > (int)update_partition->size) {
        char err_msg[96];
        snprintf(err_msg, sizeof(err_msg),
                 "Invalid firmware size (must be 1B - %luB)",
                 (unsigned long)update_partition->size);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, err_msg);
        return ESP_FAIL;
    }

    esp_ota_handle_t ota_handle;
    esp_err_t err = esp_ota_begin(update_partition, OTA_WITH_SEQUENTIAL_WRITES, &ota_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_begin failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "OTA begin failed");
        return ESP_FAIL;
    }

    /* Read firmware in chunks. */
    char buf[1024];
    int received = 0;
    int total = 0;

    while (total < req->content_len) {
        received = httpd_req_recv(req, buf, sizeof(buf));
        if (received <= 0) {
            if (received == HTTPD_SOCK_ERR_TIMEOUT) {
                continue;  /* Retry on timeout. */
            }
            ESP_LOGE(TAG, "OTA receive error at byte %d", total);
            esp_ota_abort(ota_handle);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                                "Receive error");
            return ESP_FAIL;
        }

        err = esp_ota_write(ota_handle, buf, received);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "esp_ota_write failed at byte %d: %s",
                     total, esp_err_to_name(err));
            esp_ota_abort(ota_handle);
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                                "OTA write failed");
            return ESP_FAIL;
        }

        total += received;
        if ((total % (64 * 1024)) == 0) {
            ESP_LOGI(TAG, "OTA progress: %d / %d bytes (%.0f%%)",
                     total, req->content_len,
                     (float)total * 100.0f / (float)req->content_len);
        }
    }

    err = esp_ota_end(ota_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_end failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "OTA validation failed");
        return ESP_FAIL;
    }

    err = esp_ota_set_boot_partition(update_partition);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_ota_set_boot_partition failed: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Set boot partition failed");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "OTA update successful! Rebooting to partition '%s'...",
             update_partition->label);

    const char *resp = "{\"status\":\"ok\",\"message\":\"OTA update successful. Rebooting...\"}";
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, strlen(resp));

    /* Delay briefly to let the response flush, then reboot. */
    vTaskDelay(pdMS_TO_TICKS(1000));
    esp_restart();

    return ESP_OK;  /* Never reached. */
}

/** Internal: start the HTTP server and register OTA endpoints. */
static esp_err_t ota_start_server(httpd_handle_t *out_handle)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = OTA_PORT;
    config.max_uri_handlers = 16;  /* Extra slots for WASM + ESP32-CAM dual endpoints. */
    /*
     * OTA commits validate a >1 MB app image after the request body has been
     * received. The HTTPD default task stack is tight for that path on S3 and
     * can reset the connection before esp_ota_set_boot_partition() runs. Give
     * the handler enough stack and keep the socket alive while validation and
     * the final JSON response complete.
     */
    config.stack_size = 8192;
    config.recv_wait_timeout = 30;
    config.send_wait_timeout = 30;

    httpd_handle_t server = NULL;
    esp_err_t err = httpd_start(&server, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start OTA HTTP server on port %d: %s",
                 OTA_PORT, esp_err_to_name(err));
        if (out_handle) *out_handle = NULL;
        return err;
    }

    httpd_uri_t status_uri = {
        .uri      = "/ota/status",
        .method   = HTTP_GET,
        .handler  = ota_status_handler,
        .user_ctx = NULL,
    };
    httpd_register_uri_handler(server, &status_uri);

    httpd_uri_t graph_page_uri = {
        .uri      = "/",
        .method   = HTTP_GET,
        .handler  = graph_page_handler,
        .user_ctx = NULL,
    };
    httpd_register_uri_handler(server, &graph_page_uri);

    httpd_uri_t graph_alias_uri = {
        .uri      = "/graph",
        .method   = HTTP_GET,
        .handler  = graph_page_handler,
        .user_ctx = NULL,
    };
    httpd_register_uri_handler(server, &graph_alias_uri);

    httpd_uri_t graph_data_uri = {
        .uri      = "/graph/data",
        .method   = HTTP_GET,
        .handler  = graph_data_handler,
        .user_ctx = NULL,
    };
    httpd_register_uri_handler(server, &graph_data_uri);

    httpd_uri_t upload_uri = {
        .uri      = "/ota",
        .method   = HTTP_POST,
        .handler  = ota_upload_handler,
        .user_ctx = NULL,
    };
    httpd_register_uri_handler(server, &upload_uri);

    ESP_LOGI(TAG, "OTA HTTP server started on port %d", OTA_PORT);
    ESP_LOGI(TAG, "  GET  /         — live graph page");
    ESP_LOGI(TAG, "  GET  /graph    — live graph page alias");
    ESP_LOGI(TAG, "  GET  /graph/data — graph telemetry JSON");
    ESP_LOGI(TAG, "  GET  /ota/status — firmware version info");
    ESP_LOGI(TAG, "  POST /ota        — upload new firmware binary");

    if (out_handle) *out_handle = server;
    return ESP_OK;
}

/**
 * Load the OTA PSK from NVS into the module-local s_ota_psk cache and log
 * the resulting posture. Called by both ota_update_init() and
 * ota_update_init_ex() so the per-boot diagnostic prints no matter which
 * entry point main.c uses — historically only ota_update_init() loaded the
 * PSK, which left ota_update_init_ex() with an empty s_ota_psk and an
 * invisible fail-closed posture (RuView#596 follow-up).
 */
static void ota_load_psk_from_nvs(void)
{
    nvs_handle_t nvs;
    if (nvs_open(OTA_NVS_NAMESPACE, NVS_READONLY, &nvs) == ESP_OK) {
        size_t len = sizeof(s_ota_psk);
        if (nvs_get_str(nvs, OTA_NVS_KEY, s_ota_psk, &len) == ESP_OK) {
            ESP_LOGI(TAG, "OTA PSK loaded from NVS (%d chars) — authentication enabled", (int)len - 1);
        } else {
            ESP_LOGW(TAG, "No OTA PSK in NVS — OTA upload endpoint will REJECT all requests until "
                          "provisioned (provision.py --ota-psk <hex>). Fail-closed per RuView#596.");
        }
        nvs_close(nvs);
    } else {
        ESP_LOGW(TAG, "NVS namespace '%s' not found — OTA upload endpoint will REJECT all "
                      "requests until provisioned. Fail-closed per RuView#596.", OTA_NVS_NAMESPACE);
    }
}

esp_err_t ota_update_init(void)
{
    /* ADR-050: Load OTA PSK from NVS if provisioned. */
    ota_load_psk_from_nvs();
    return ota_start_server(NULL);
}

esp_err_t ota_update_init_ex(void **out_server)
{
    /* ADR-050: Load OTA PSK from NVS if provisioned. main.c uses this
     * variant (not ota_update_init), so without this call s_ota_psk
     * stayed empty forever and the fail-closed posture was invisible
     * in serial logs. */
    ota_load_psk_from_nvs();
    return ota_start_server((httpd_handle_t *)out_server);
}

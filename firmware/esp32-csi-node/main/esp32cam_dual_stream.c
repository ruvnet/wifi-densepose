#include "esp32cam_dual_stream.h"

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

static const char *TAG = "esp32cam_dual";

/* AI Thinker ESP32-CAM / OV2640 pin map. */
#define CAM_PIN_D0     5
#define CAM_PIN_D1     18
#define CAM_PIN_D2     19
#define CAM_PIN_D3     21
#define CAM_PIN_D4     36
#define CAM_PIN_D5     39
#define CAM_PIN_D6     34
#define CAM_PIN_D7     35
#define CAM_PIN_VSYNC  25
#define CAM_PIN_HREF   23
#define CAM_PIN_PCLK   22

#define STREAM_BOUNDARY "ruviewframe"

static bool s_camera_ready;

static esp_err_t send_unavailable(httpd_req_t *req, const char *message)
{
    httpd_resp_set_status(req, "503 Service Unavailable");
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_sendstr(req, message);
}

static framesize_t configured_frame_size(void)
{
#if CONFIG_ESP32CAM_FRAME_QQVGA
    return FRAMESIZE_QQVGA;
#elif CONFIG_ESP32CAM_FRAME_VGA
    return FRAMESIZE_VGA;
#else
    return FRAMESIZE_QVGA;
#endif
}

static const char *configured_frame_size_name(void)
{
#if CONFIG_ESP32CAM_FRAME_QQVGA
    return "QQVGA";
#elif CONFIG_ESP32CAM_FRAME_VGA
    return "VGA";
#else
    return "QVGA";
#endif
}

static esp_err_t camera_init_once(void)
{
    if (s_camera_ready) {
        return ESP_OK;
    }

    const bool psram_ready = heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0;
    camera_config_t config = {
        .pin_pwdn = CONFIG_ESP32CAM_PIN_PWDN,
        .pin_reset = CONFIG_ESP32CAM_PIN_RESET,
        .pin_xclk = CONFIG_ESP32CAM_PIN_XCLK,
        .pin_sccb_sda = CONFIG_ESP32CAM_PIN_SIOD,
        .pin_sccb_scl = CONFIG_ESP32CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,
        .xclk_freq_hz = CONFIG_ESP32CAM_XCLK_FREQ_HZ,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = configured_frame_size(),
        .jpeg_quality = CONFIG_ESP32CAM_JPEG_QUALITY,
        .fb_count = psram_ready ? 2 : 1,
        .fb_location = psram_ready ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM,
        .grab_mode = CAMERA_GRAB_LATEST,
    };

    esp_err_t ret = esp_camera_init(&config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_camera_init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL) {
        sensor->set_quality(sensor, CONFIG_ESP32CAM_JPEG_QUALITY);
        sensor->set_framesize(sensor, configured_frame_size());
    }

    s_camera_ready = true;
    ESP_LOGI(TAG,
             "ESP32-CAM dual stream ready: frame=%s quality=%d fps_cap=%d psram=%s",
             configured_frame_size_name(),
             CONFIG_ESP32CAM_JPEG_QUALITY,
             CONFIG_ESP32CAM_STREAM_FPS,
             psram_ready ? "yes" : "no");
    return ESP_OK;
}

static esp_err_t status_handler(httpd_req_t *req)
{
    char payload[192];
    int len = snprintf(payload, sizeof(payload),
                       "{\"camera_ready\":%s,\"mode\":\"dual-csi-mjpeg\","
                       "\"frame_size\":\"%s\",\"jpeg_quality\":%d,"
                       "\"stream_fps\":%d,\"path\":\"/stream\"}",
                       s_camera_ready ? "true" : "false",
                       configured_frame_size_name(),
                       CONFIG_ESP32CAM_JPEG_QUALITY,
                       CONFIG_ESP32CAM_STREAM_FPS);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, payload, len);
}

static esp_err_t page_handler(httpd_req_t *req)
{
    static const char html[] =
        "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>RuView ESP32-CAM</title>"
        "<style>body{margin:0;background:#071016;color:#d8fff0;font-family:system-ui,sans-serif}"
        "main{max-width:760px;margin:auto;padding:16px}img{width:100%;border:1px solid #1a4;"
        "background:#000;border-radius:8px}.meta{color:#8fb;font:13px monospace}</style></head>"
        "<body><main><h2>RuView ESP32-CAM Dual CSI + MJPEG</h2>"
        "<img src='/stream'><p class='meta'>CSI UDP stays active. Snapshot: <a href='/cam.jpg'>/cam.jpg</a>. "
        "Status: <a href='/cam/status'>/cam/status</a>.</p></main></body></html>";
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t jpg_handler(httpd_req_t *req)
{
    if (!s_camera_ready) {
        return send_unavailable(req, "camera not ready");
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == NULL) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "capture failed");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    esp_err_t ret = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return ret;
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    if (!s_camera_ready) {
        return send_unavailable(req, "camera not ready");
    }

    httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=" STREAM_BOUNDARY);
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");

    const TickType_t delay_ticks = pdMS_TO_TICKS(1000 / CONFIG_ESP32CAM_STREAM_FPS);
    char header[96];
    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            ESP_LOGW(TAG, "stream capture failed");
            vTaskDelay(pdMS_TO_TICKS(250));
            continue;
        }

        int header_len = snprintf(header, sizeof(header),
                                  "\r\n--" STREAM_BOUNDARY
                                  "\r\nContent-Type: image/jpeg"
                                  "\r\nContent-Length: %u\r\n\r\n",
                                  (unsigned)fb->len);
        esp_err_t ret = httpd_resp_send_chunk(req, header, header_len);
        if (ret == ESP_OK) {
            ret = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        }
        esp_camera_fb_return(fb);
        if (ret != ESP_OK) {
            break;
        }
        vTaskDelay(delay_ticks);
    }

    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

esp_err_t esp32cam_dual_stream_register(httpd_handle_t server)
{
    if (server == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t ret = camera_init_once();
    if (ret != ESP_OK) {
        return ret;
    }

    const httpd_uri_t page_uri = {
        .uri = "/cam",
        .method = HTTP_GET,
        .handler = page_handler,
        .user_ctx = NULL,
    };
    const httpd_uri_t jpg_uri = {
        .uri = "/cam.jpg",
        .method = HTTP_GET,
        .handler = jpg_handler,
        .user_ctx = NULL,
    };
    const httpd_uri_t stream_uri = {
        .uri = "/stream",
        .method = HTTP_GET,
        .handler = stream_handler,
        .user_ctx = NULL,
    };
    const httpd_uri_t status_uri = {
        .uri = "/cam/status",
        .method = HTTP_GET,
        .handler = status_handler,
        .user_ctx = NULL,
    };

    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &page_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &jpg_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &stream_uri));
    ESP_ERROR_CHECK(httpd_register_uri_handler(server, &status_uri));

    ESP_LOGI(TAG, "Camera endpoints registered: /cam /cam.jpg /stream /cam/status");
    return ESP_OK;
}

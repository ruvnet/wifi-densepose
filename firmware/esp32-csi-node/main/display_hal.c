/**
 * @file display_hal.c
 * @brief Target-specific ST7789 LCD HAL.
 *
 * Cardputer-Adv (ESP32-S3): 240x135, BL=G38, RST=G33, DC=G34, MOSI=G35,
 * SCK=G36, CS=G37.
 * StickC Plus2 (ESP32): 135x240, BL=G27, RST=G12, DC=G14, MOSI=G15,
 * SCK=G13, CS=G5.
 */

#include "display_hal.h"
#include "sdkconfig.h"

#if CONFIG_DISPLAY_ENABLE

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_log.h"

static const char *TAG = "disp_hal";

#define LCD_SPI_HOST        SPI2_HOST
#define LCD_PIXEL_CLOCK_HZ  (20 * 1000 * 1000)
#define LCD_CMD_BITS        8
#define LCD_PARAM_BITS      8

static esp_lcd_panel_io_handle_t s_io_handle = NULL;
static esp_lcd_panel_handle_t s_panel_handle = NULL;
static uint16_t *s_framebuffer = NULL;
static bool s_frame_dirty = false;

static uint16_t bw565(uint16_t rgb565)
{
    uint32_t r = (rgb565 >> 11) & 0x1F;
    uint32_t g = (rgb565 >> 5) & 0x3F;
    uint32_t b = rgb565 & 0x1F;
    uint32_t lum = r * 54 + g * 183 + b * 19;
    return (lum >= 4096) ? 0xFFFF : 0x0000;
}

static void set_backlight(bool on)
{
    gpio_config_t bl_cfg = {
        .pin_bit_mask = 1ULL << DISPLAY_PANEL_BL_PIN,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&bl_cfg);
    gpio_set_level(DISPLAY_PANEL_BL_PIN, on ? 1 : 0);
}

static esp_err_t display_panel_init_common(void)
{
    ESP_LOGI(TAG, "Initializing %s ST7789 LCD (%dx%d)...",
             DISPLAY_PANEL_NAME, DISPLAY_PANEL_H_RES, DISPLAY_PANEL_V_RES);

    set_backlight(false);

    spi_bus_config_t bus_cfg = {
        .sclk_io_num = DISPLAY_PANEL_SCLK_PIN,
        .mosi_io_num = DISPLAY_PANEL_MOSI_PIN,
        .miso_io_num = -1,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = DISPLAY_PANEL_H_RES * 40 * sizeof(uint16_t),
    };

    esp_err_t ret = spi_bus_initialize(LCD_SPI_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "SPI bus init failed: %s", esp_err_to_name(ret));
        return ESP_ERR_NOT_FOUND;
    }

    esp_lcd_panel_io_spi_config_t io_cfg = {
        .dc_gpio_num = DISPLAY_PANEL_DC_PIN,
        .cs_gpio_num = DISPLAY_PANEL_CS_PIN,
        .pclk_hz = LCD_PIXEL_CLOCK_HZ,
        .lcd_cmd_bits = LCD_CMD_BITS,
        .lcd_param_bits = LCD_PARAM_BITS,
        .spi_mode = 0,
        .trans_queue_depth = 10,
    };

    ret = esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_SPI_HOST, &io_cfg, &s_io_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Panel IO init failed: %s", esp_err_to_name(ret));
        return ESP_ERR_NOT_FOUND;
    }

    esp_lcd_panel_dev_config_t panel_cfg = {
        .reset_gpio_num = DISPLAY_PANEL_RST_PIN,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_BGR,
        .bits_per_pixel = 16,
    };

    ret = esp_lcd_new_panel_st7789(s_io_handle, &panel_cfg, &s_panel_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "ST7789 panel create failed: %s", esp_err_to_name(ret));
        esp_lcd_panel_io_del(s_io_handle);
        s_io_handle = NULL;
        return ESP_ERR_NOT_FOUND;
    }

    ESP_ERROR_CHECK(esp_lcd_panel_reset(s_panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(s_panel_handle));
#if defined(CONFIG_IDF_TARGET_ESP32S3)
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(s_panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(s_panel_handle, true, false));
#else
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(s_panel_handle, false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(s_panel_handle, false, false));
#endif
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(s_panel_handle, DISPLAY_PANEL_GAP_X, DISPLAY_PANEL_GAP_Y));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(s_panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(s_panel_handle, true));

    s_framebuffer = heap_caps_malloc(DISPLAY_PANEL_H_RES * DISPLAY_PANEL_V_RES * sizeof(uint16_t),
                                     MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!s_framebuffer) {
        ESP_LOGE(TAG, "Framebuffer allocation failed");
        return ESP_ERR_NO_MEM;
    }
    memset(s_framebuffer, 0, DISPLAY_PANEL_H_RES * DISPLAY_PANEL_V_RES * sizeof(uint16_t));
    s_frame_dirty = true;

    set_backlight(true);
    display_hal_present();
    ESP_LOGI(TAG, "%s ST7789 panel init OK: pclk=%dHz gap=(%d,%d) framebuffer=%u bytes bw=1",
             DISPLAY_PANEL_NAME, LCD_PIXEL_CLOCK_HZ, DISPLAY_PANEL_GAP_X, DISPLAY_PANEL_GAP_Y,
             (unsigned)(DISPLAY_PANEL_H_RES * DISPLAY_PANEL_V_RES * sizeof(uint16_t)));
    return ESP_OK;
}

#if BOARD_CARDPUTER_ADV
static esp_err_t cardputer_adv_display_init(void)
{
    return display_panel_init_common();
}
#endif

#if BOARD_M5STICKC_PLUS
static esp_err_t stickc_plus_display_init(void)
{
    return display_panel_init_common();
}
#endif

esp_err_t display_hal_init_panel(void)
{
#if BOARD_CARDPUTER_ADV
    return cardputer_adv_display_init();
#elif BOARD_M5STICKC_PLUS
    return stickc_plus_display_init();
#else
#error "DISPLAY_ENABLE is only supported on ESP32-S3 Cardputer-Adv and ESP32 StickC Plus2 targets"
#endif
}

void display_hal_draw(int x_start, int y_start, int x_end, int y_end,
                      const void *color_data)
{
    if (!s_panel_handle || !s_framebuffer || !color_data) return;
    if (x_start < 0) x_start = 0;
    if (y_start < 0) y_start = 0;
    if (x_end > DISPLAY_PANEL_H_RES) x_end = DISPLAY_PANEL_H_RES;
    if (y_end > DISPLAY_PANEL_V_RES) y_end = DISPLAY_PANEL_V_RES;
    if (x_start >= x_end || y_start >= y_end) return;

    const uint16_t *src = (const uint16_t *)color_data;
    const int w = x_end - x_start;
    const int h = y_end - y_start;
    for (int y = 0; y < h; y++) {
        uint16_t *dst = &s_framebuffer[(y_start + y) * DISPLAY_PANEL_H_RES + x_start];
        const uint16_t *row = &src[y * w];
        for (int x = 0; x < w; x++) {
            dst[x] = bw565(row[x]);
        }
    }
    s_frame_dirty = true;
}

void display_hal_present(void)
{
    if (!s_panel_handle || !s_framebuffer || !s_frame_dirty) return;
    esp_lcd_panel_draw_bitmap(s_panel_handle, 0, 0,
                              DISPLAY_PANEL_H_RES, DISPLAY_PANEL_V_RES,
                              s_framebuffer);
    s_frame_dirty = false;
}

esp_err_t display_hal_init_touch(void)
{
    return ESP_ERR_NOT_FOUND;
}

bool display_hal_touch_read(uint16_t *x, uint16_t *y)
{
    (void)x;
    (void)y;
    return false;
}

void display_hal_set_brightness(uint8_t percent)
{
    set_backlight(percent > 0);
}

#endif /* CONFIG_DISPLAY_ENABLE */

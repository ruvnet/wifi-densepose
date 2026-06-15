/**
 * @file display_ui.c
 * @brief ADR-045: LVGL 4-view swipeable UI — Dashboard | Vitals | Presence | System.
 *
 * High-contrast black/white feature dashboard for the Cardputer-Adv LCD.
 */

#include "display_ui.h"
#include "nvs_config.h"
#include "csi_collector.h"  /* csi_collector_get_node_id() - defensive #390 */
#include "c6_sync_espnow.h"
#include "cardputer_adv_audio.h"
#include "sdkconfig.h"

extern nvs_config_t g_nvs_config;

#if CONFIG_DISPLAY_ENABLE

#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "edge_processing.h"
#include "battery_monitor.h"

static const char *TAG = "disp_ui";

/* ---- Theme colors ---- */
#define COLOR_BG        lv_color_black()
#define COLOR_WHITE     lv_color_white()
#define COLOR_DIM       lv_color_make(0x55, 0x55, 0x55)
#define COLOR_TEXT      lv_color_white()
#define COLOR_TEXT_DIM  lv_color_make(0x80, 0x80, 0x80)

/* ---- Chart data points ---- */
#define CHART_POINTS    96
#define SCREEN_W        240
#define SCREEN_H        135
#define GRID_GAP        0
#define VIEW_COUNT      4
#define VIEW_HOLD_MS    2000
#define VIEW_KEY_HOLD_MS 15000
#define UI_CONTENT_REFRESH_MS 125

/* Cardputer-Adv keyboard: TCA8418 on G8/G9/G11 per M5Stack pin map. */
#define KB_I2C_PORT     I2C_NUM_0
#define KB_I2C_SDA      8
#define KB_I2C_SCL      9
#define KB_I2C_INT      11
#define KB_I2C_ADDR     0x34
#define TCA_REG_CFG        0x01
#define TCA_REG_INT_STAT   0x02
#define TCA_REG_KEY_LCK_EC 0x03
#define TCA_REG_KEY_EVENT  0x04
#define TCA_REG_GPI_EM1    0x09
#define TCA_REG_KP_GPIO1   0x1D
#define TCA_REG_KP_GPIO2   0x1E
#define TCA_REG_KP_GPIO3   0x1F

/* ---- View handles ---- */
static lv_obj_t *s_tileview = NULL;
static uint8_t s_active_view = 0;
static uint32_t s_last_view_switch_ms = 0;
static uint32_t s_last_key_ms = 0;
static uint32_t s_last_view_key_ms = 0;
static lv_obj_t *s_view_tabs[VIEW_COUNT];
static const char *s_view_names[VIEW_COUNT] = {"DASH", "VITAL", "PRES", "SYS"};
static lv_obj_t *s_scan_bar = NULL;
static lv_obj_t *s_scan_dot = NULL;
static lv_obj_t *s_debug_frame = NULL;
static uint32_t s_debug_total_frames = 0;

static bool s_kb_init_attempted = false;
static bool s_kb_available = false;
static i2c_master_bus_handle_t s_kb_bus = NULL;
static i2c_master_dev_handle_t s_kb_dev = NULL;

/* Dashboard */
static lv_obj_t *s_dash_chart      = NULL;
static lv_chart_series_t *s_csi_series = NULL;
static lv_obj_t *s_dash_persons    = NULL;
static lv_obj_t *s_dash_rssi       = NULL;
static lv_obj_t *s_dash_motion     = NULL;
static lv_obj_t *s_dash_adv        = NULL;
static lv_obj_t *s_dash_stick      = NULL;
static lv_obj_t *s_dash_server     = NULL;
static lv_obj_t *s_dash_zoom       = NULL;
static uint8_t s_dash_zoom_level   = 0;

/* Vitals */
static lv_obj_t *s_vital_chart     = NULL;
static lv_chart_series_t *s_breath_series = NULL;
static lv_chart_series_t *s_hr_series     = NULL;
static lv_obj_t *s_vital_bpm_br    = NULL;
static lv_obj_t *s_vital_bpm_hr    = NULL;

/* Presence */
#define GRID_COLS  8
#define GRID_ROWS  5
static lv_obj_t *s_grid_cells[GRID_COLS * GRID_ROWS];
static lv_obj_t *s_presence_label = NULL;

/* System */
static lv_obj_t *s_sys_cpu         = NULL;
static lv_obj_t *s_sys_heap        = NULL;
static lv_obj_t *s_sys_psram       = NULL;
static lv_obj_t *s_sys_rssi        = NULL;
static lv_obj_t *s_sys_uptime      = NULL;
static lv_obj_t *s_sys_fps         = NULL;
static lv_obj_t *s_sys_node        = NULL;
static lv_obj_t *s_sys_battery     = NULL;
static lv_obj_t *s_sys_power       = NULL;
static lv_obj_t *s_sys_peer        = NULL;

/* ---- Style helpers ---- */
static lv_style_t s_style_bg;
static lv_style_t s_style_label;
static lv_style_t s_style_label_big;
static bool s_styles_inited = false;

/*
 * The esp-idf LVGL component is configured from sdkconfig in this build. Keep
 * the UI on the default 14px font so display builds survive sdkconfig changes
 * that disable the smaller Montserrat variants.
 */
#define UI_FONT_SMALL  LV_FONT_DEFAULT
#define UI_FONT_BIG    LV_FONT_DEFAULT

static void init_styles(void)
{
    if (s_styles_inited) return;
    s_styles_inited = true;

    lv_style_init(&s_style_bg);
    lv_style_set_bg_color(&s_style_bg, COLOR_BG);
    lv_style_set_bg_opa(&s_style_bg, LV_OPA_COVER);
    lv_style_set_border_width(&s_style_bg, 0);
    lv_style_set_pad_all(&s_style_bg, 0);

    lv_style_init(&s_style_label);
    lv_style_set_text_color(&s_style_label, COLOR_TEXT);
    lv_style_set_text_font(&s_style_label, UI_FONT_SMALL);

    lv_style_init(&s_style_label_big);
    lv_style_set_text_color(&s_style_label_big, COLOR_WHITE);
    lv_style_set_text_font(&s_style_label_big, UI_FONT_BIG);
}

static lv_obj_t *make_label(lv_obj_t *parent, const char *text, const lv_style_t *style)
{
    lv_obj_t *lbl = lv_label_create(parent);
    lv_label_set_text(lbl, text);
    if (style) lv_obj_add_style(lbl, (lv_style_t *)style, 0);
    return lbl;
}

static lv_obj_t *make_tile(lv_obj_t *tv, uint8_t col, uint8_t row)
{
    lv_obj_t *tile = lv_tileview_add_tile(tv, col, row, LV_DIR_HOR);
    lv_obj_add_style(tile, &s_style_bg, 0);
    return tile;
}

static void format_battery_line(char *buf, size_t len, const char *label,
                                uint8_t node_id, bool live, uint8_t flags,
                                uint8_t percent, uint16_t millivolts,
                                uint8_t status, uint32_t age_ms)
{
    (void)status;
    (void)age_ms;
    if (!live) {
        snprintf(buf, len, "%s: WAIT", label);
    } else if (flags & 0x01) {
        snprintf(buf, len, "%s%u %u%% %umV",
                 label, (unsigned)node_id, (unsigned)percent,
                 (unsigned)millivolts);
    } else {
        snprintf(buf, len, "%s%u LIVE batt?", label, (unsigned)node_id);
    }
}

static void apply_dash_chart_zoom(void)
{
    if (!s_dash_chart) return;

    static const uint16_t zoom_x[] = {256, 512, 1024};
    static const int32_t y_max[] = {100, 50, 25};
    static const char *label[] = {"Z1", "Z2", "Z4"};

    uint8_t idx = s_dash_zoom_level;
    if (idx >= 3) idx = 0;

    lv_chart_set_range(s_dash_chart, LV_CHART_AXIS_PRIMARY_Y, 0, y_max[idx]);
    lv_chart_set_zoom_x(s_dash_chart, zoom_x[idx]);
    lv_chart_set_zoom_y(s_dash_chart, 256);
    if (s_dash_zoom) {
        lv_label_set_text(s_dash_zoom, label[idx]);
    }
}

static void dash_chart_event_cb(lv_event_t *e)
{
    if (lv_event_get_code(e) != LV_EVENT_CLICKED) return;
    s_dash_zoom_level = (s_dash_zoom_level + 1) % 3;
    apply_dash_chart_zoom();
}

static void update_view_tabs(void)
{
    static const char *active_names[VIEW_COUNT] = {"<DASH>", "<VITAL>", "<PRES>", "<SYS>"};
    for (uint8_t i = 0; i < VIEW_COUNT; i++) {
        if (!s_view_tabs[i]) continue;
        bool active = i == s_active_view;
        lv_obj_set_style_bg_color(s_view_tabs[i], active ? COLOR_WHITE : COLOR_BG, 0);
        lv_obj_set_style_bg_opa(s_view_tabs[i], active ? LV_OPA_COVER : LV_OPA_70, 0);
        lv_obj_set_style_border_color(s_view_tabs[i], COLOR_WHITE, 0);
        lv_obj_set_style_border_width(s_view_tabs[i], 1, 0);
        lv_obj_set_style_text_color(s_view_tabs[i], active ? COLOR_BG : COLOR_WHITE, 0);
        lv_label_set_text(s_view_tabs[i], active ? active_names[i] : s_view_names[i]);
        lv_obj_move_foreground(s_view_tabs[i]);
    }
}

static void update_scan_marker(uint32_t now_ms)
{
    int x = (int)((now_ms / 18) % SCREEN_W);
    int y = 17 + (int)((now_ms / 31) % (SCREEN_H - 38));

    if (s_scan_bar) {
        lv_obj_set_pos(s_scan_bar, x, 0);
        lv_obj_move_foreground(s_scan_bar);
    }
    if (s_scan_dot) {
        lv_obj_set_pos(s_scan_dot, x > 5 ? x - 5 : x, y);
        lv_obj_set_style_bg_color(s_scan_dot, ((now_ms / 250) & 1) ? COLOR_WHITE : COLOR_TEXT_DIM, 0);
        lv_obj_move_foreground(s_scan_dot);
    }
}

static void create_view_tabs(lv_obj_t *parent)
{
    const int tab_w = SCREEN_W / VIEW_COUNT;
    for (uint8_t i = 0; i < VIEW_COUNT; i++) {
        lv_obj_t *tab = make_label(parent, s_view_names[i], &s_style_label);
        lv_obj_set_size(tab, tab_w, 15);
        lv_obj_set_pos(tab, i * tab_w, SCREEN_H - 15);
        lv_obj_set_style_text_align(tab, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_set_style_pad_top(tab, 1, 0);
        lv_obj_set_style_pad_bottom(tab, 1, 0);
        lv_obj_set_style_pad_left(tab, 0, 0);
        lv_obj_set_style_pad_right(tab, 0, 0);
        s_view_tabs[i] = tab;
    }
    update_view_tabs();

    s_scan_bar = lv_obj_create(parent);
    lv_obj_set_size(s_scan_bar, 18, SCREEN_H);
    lv_obj_set_pos(s_scan_bar, 0, 0);
    lv_obj_set_style_bg_color(s_scan_bar, COLOR_WHITE, 0);
    lv_obj_set_style_bg_opa(s_scan_bar, LV_OPA_70, 0);
    lv_obj_set_style_border_width(s_scan_bar, 0, 0);
    lv_obj_set_style_pad_all(s_scan_bar, 0, 0);
    lv_obj_set_style_radius(s_scan_bar, 0, 0);

    s_scan_dot = lv_obj_create(parent);
    lv_obj_set_size(s_scan_dot, 32, 32);
    lv_obj_set_pos(s_scan_dot, 0, 30);
    lv_obj_set_style_bg_color(s_scan_dot, COLOR_WHITE, 0);
    lv_obj_set_style_bg_opa(s_scan_dot, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(s_scan_dot, 0, 0);
    lv_obj_set_style_pad_all(s_scan_dot, 0, 0);
    lv_obj_set_style_radius(s_scan_dot, 0, 0);

    s_debug_frame = make_label(parent, "FRAME 000000", &s_style_label);
    lv_obj_align(s_debug_frame, LV_ALIGN_TOP_RIGHT, -4, 4);
    lv_obj_set_style_text_color(s_debug_frame, COLOR_WHITE, 0);
    lv_obj_set_style_bg_color(s_debug_frame, COLOR_BG, 0);
    lv_obj_set_style_bg_opa(s_debug_frame, LV_OPA_COVER, 0);
    lv_obj_move_foreground(s_debug_frame);
}

static void select_view(uint8_t view, bool anim, uint32_t now_ms)
{
    if (!s_tileview) return;
    if (view >= VIEW_COUNT) view = 0;
    s_active_view = view;
    lv_obj_set_tile_id(s_tileview, s_active_view, 0, anim ? LV_ANIM_ON : LV_ANIM_OFF);
    update_view_tabs();
    s_last_view_switch_ms = now_ms;
}

static esp_err_t kb_write_reg(uint8_t reg, uint8_t value)
{
    uint8_t data[2] = {reg, value};
    return i2c_master_transmit(s_kb_dev, data, sizeof(data), 20);
}

static esp_err_t kb_read_reg(uint8_t reg, uint8_t *value)
{
    return i2c_master_transmit_receive(s_kb_dev, &reg, 1, value, 1, 20);
}

static void init_keyboard(void)
{
    if (s_kb_init_attempted) return;
    s_kb_init_attempted = true;

    gpio_config_t int_cfg = {
        .pin_bit_mask = 1ULL << KB_I2C_INT,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&int_cfg);

    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = KB_I2C_PORT,
        .sda_io_num = KB_I2C_SDA,
        .scl_io_num = KB_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t err = i2c_new_master_bus(&bus_cfg, &s_kb_bus);
    if (err == ESP_ERR_INVALID_STATE) {
        err = i2c_master_get_bus_handle(KB_I2C_PORT, &s_kb_bus);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "keyboard reusing I2C%d bus on G%d/G%d",
                     KB_I2C_PORT, KB_I2C_SDA, KB_I2C_SCL);
        }
    }
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "keyboard I2C bus init failed: %s", esp_err_to_name(err));
        return;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = KB_I2C_ADDR,
        .scl_speed_hz = 400000,
    };
    err = i2c_master_bus_add_device(s_kb_bus, &dev_cfg, &s_kb_dev);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "keyboard TCA8418 add failed: %s", esp_err_to_name(err));
        return;
    }

    /*
     * Cardputer-ADV uses the TCA8418 in an 8x8 keypad mode. The controller
     * otherwise powers up quietly and may acknowledge I2C without emitting keys.
     */
    kb_write_reg(TCA_REG_KP_GPIO1, 0xFF);
    kb_write_reg(TCA_REG_KP_GPIO2, 0xFF);
    kb_write_reg(TCA_REG_KP_GPIO3, 0x00);
    kb_write_reg(TCA_REG_GPI_EM1, 0x00);
    kb_write_reg(TCA_REG_INT_STAT, 0xFF);
    kb_write_reg(TCA_REG_CFG, 0x3E);

    uint8_t ec = 0;
    if (kb_read_reg(TCA_REG_KEY_LCK_EC, &ec) == ESP_OK) {
        s_kb_available = true;
        ESP_LOGI(TAG, "keyboard ready: TCA8418 addr=0x%02x SDA=G%d SCL=G%d INT=G%d",
                 KB_I2C_ADDR, KB_I2C_SDA, KB_I2C_SCL, KB_I2C_INT);
    } else {
        ESP_LOGW(TAG, "keyboard TCA8418 not responding at 0x%02x", KB_I2C_ADDR);
    }
}

static void handle_key_event(uint8_t code, bool pressed, uint32_t now_ms)
{
    s_last_key_ms = now_ms;

#if defined(CONFIG_CARDPUTER_ADV_AUDIO_ENABLE) && defined(CONFIG_IDF_TARGET_ESP32S3)
    cardputer_adv_audio_key_event(code, pressed);
#endif

    if (!pressed || code < 1 || code > VIEW_COUNT) {
        return;
    }
    if ((uint32_t)(now_ms - s_last_view_key_ms) <= 180U) {
        return;
    }

    uint8_t view = code - 1;
    s_last_view_key_ms = now_ms;
    ESP_LOGI(TAG, "keyboard key=%u -> view=%s", (unsigned)code, s_view_names[view]);
    select_view(view, true, now_ms);
}

static void __attribute__((unused)) update_keyboard(uint32_t now_ms)
{
    init_keyboard();
    if (!s_kb_available) return;

    uint8_t ec = 0;
    if (kb_read_reg(TCA_REG_KEY_LCK_EC, &ec) != ESP_OK) return;
    uint8_t count = ec & 0x0F;
    if (count == 0 && gpio_get_level(KB_I2C_INT) == 0) {
        count = 10;
    }
    if (count > 10) count = 10;

    for (uint8_t i = 0; i < count; i++) {
        uint8_t event = 0;
        if (kb_read_reg(TCA_REG_KEY_EVENT, &event) != ESP_OK) return;
        if (event == 0) {
            break;
        }
        bool pressed = (event & 0x80) != 0;
        uint8_t code = event & 0x7F;
        if (code != 0) {
            handle_key_event(code, pressed, now_ms);
        }
    }
    kb_write_reg(TCA_REG_INT_STAT, 0xFF);
}

static int moving_trace_value(uint32_t now_ms, bool has_vitals, const edge_vitals_pkt_t *vitals)
{
    uint32_t phase = (now_ms / 90) % 48;
    int sweep = (phase < 24) ? (int)phase : (int)(47 - phase);
    int val = 10 + sweep * 3;

    if (has_vitals && vitals) {
        int motion = (int)(vitals->motion_energy * 18.0f);
        if (motion > val) {
            val = motion;
        }
        if (vitals->rssi < 0) {
            int rssi_motion = 100 + vitals->rssi;
            if (rssi_motion > val) {
                val = rssi_motion;
            }
        }
    }

    if (val > 100) val = 100;
    if (val < 0) val = 0;
    return val;
}

static void update_auto_view(uint32_t now_ms)
{
    if (!s_tileview) return;
    if (s_last_key_ms != 0 && now_ms - s_last_key_ms < VIEW_KEY_HOLD_MS) return;
    if (s_last_view_switch_ms == 0) {
        s_last_view_switch_ms = now_ms;
        return;
    }
    if (now_ms - s_last_view_switch_ms < VIEW_HOLD_MS) return;

    s_active_view = (s_active_view + 1) % VIEW_COUNT;
    select_view(s_active_view, true, now_ms);
}

/* ---- View 0: Dashboard ---- */
static void create_dashboard(lv_obj_t *tile)
{
    s_dash_chart = lv_chart_create(tile);
    lv_obj_set_size(s_dash_chart, SCREEN_W, SCREEN_H);
    lv_obj_align(s_dash_chart, LV_ALIGN_TOP_LEFT, 0, 0);
    lv_chart_set_type(s_dash_chart, LV_CHART_TYPE_LINE);
    lv_chart_set_point_count(s_dash_chart, CHART_POINTS);
    lv_chart_set_range(s_dash_chart, LV_CHART_AXIS_PRIMARY_Y, 0, 100);
    lv_chart_set_div_line_count(s_dash_chart, 7, 13);

    /* Scope-style field: graph lines only, no outer box. */
    lv_obj_set_style_bg_color(s_dash_chart, COLOR_BG, 0);
    lv_obj_set_style_border_width(s_dash_chart, 0, 0);
    lv_obj_set_style_pad_all(s_dash_chart, 0, 0);
    lv_obj_set_style_line_color(s_dash_chart, COLOR_DIM, LV_PART_MAIN);
    lv_obj_set_style_line_opa(s_dash_chart, LV_OPA_60, LV_PART_MAIN);
    lv_obj_set_style_line_width(s_dash_chart, 0, LV_PART_TICKS);
    lv_obj_add_flag(s_dash_chart, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_add_event_cb(s_dash_chart, dash_chart_event_cb, LV_EVENT_CLICKED, NULL);

    s_csi_series = lv_chart_add_series(s_dash_chart, COLOR_WHITE, LV_CHART_AXIS_PRIMARY_Y);
    lv_obj_set_style_size(s_dash_chart, 0, LV_PART_INDICATOR);
    lv_obj_set_style_line_width(s_dash_chart, 2, LV_PART_ITEMS);

    /* Edge telemetry. Labels are drawn directly on black; no panels/boxes. */
    lv_obj_t *title = make_label(tile, "ADV UI 0602", &s_style_label);
    lv_obj_align(title, LV_ALIGN_TOP_LEFT, 0, 0);

    s_dash_zoom = make_label(tile, "Z1", &s_style_label);
    lv_obj_align(s_dash_zoom, LV_ALIGN_TOP_MID, 0, 0);

    s_dash_adv = make_label(tile, "ADV --", &s_style_label);
    lv_obj_align(s_dash_adv, LV_ALIGN_TOP_RIGHT, 0, 0);

    s_dash_persons = make_label(tile, "P0", &s_style_label_big);
    lv_obj_align(s_dash_persons, LV_ALIGN_RIGHT_MID, 0, -20);

    s_dash_rssi = make_label(tile, "R--", &s_style_label);
    lv_obj_align(s_dash_rssi, LV_ALIGN_LEFT_MID, 0, 0);

    s_dash_motion = make_label(tile, "M0.0", &s_style_label);
    lv_obj_align(s_dash_motion, LV_ALIGN_BOTTOM_MID, 0, 0);

    s_dash_server = make_label(tile, "SRC WAIT", &s_style_label);
    lv_obj_align(s_dash_server, LV_ALIGN_BOTTOM_LEFT, 0, 0);

    s_dash_stick = make_label(tile, "STK WAIT", &s_style_label);
    lv_obj_align(s_dash_stick, LV_ALIGN_BOTTOM_RIGHT, 0, 0);

    apply_dash_chart_zoom();
}

/* ---- View 1: Vitals ---- */
static void create_vitals(lv_obj_t *tile)
{
    make_label(tile, "Vital Signs", &s_style_label);

    s_vital_chart = lv_chart_create(tile);
    lv_obj_set_size(s_vital_chart, SCREEN_W, SCREEN_H - 15);
    lv_obj_align(s_vital_chart, LV_ALIGN_TOP_LEFT, 0, 0);
    lv_chart_set_type(s_vital_chart, LV_CHART_TYPE_LINE);
    lv_chart_set_point_count(s_vital_chart, CHART_POINTS);
    lv_chart_set_range(s_vital_chart, LV_CHART_AXIS_PRIMARY_Y, 0, 120);
    lv_chart_set_div_line_count(s_vital_chart, 7, 13);
    lv_obj_set_style_bg_color(s_vital_chart, COLOR_BG, 0);
    lv_obj_set_style_border_width(s_vital_chart, 0, 0);
    lv_obj_set_style_pad_all(s_vital_chart, 0, 0);
    lv_obj_set_style_line_color(s_vital_chart, COLOR_DIM, LV_PART_MAIN);
    lv_obj_set_style_line_opa(s_vital_chart, LV_OPA_60, LV_PART_MAIN);
    lv_obj_set_style_line_width(s_vital_chart, 0, LV_PART_TICKS);

    s_breath_series = lv_chart_add_series(s_vital_chart, COLOR_WHITE, LV_CHART_AXIS_PRIMARY_Y);
    s_hr_series = lv_chart_add_series(s_vital_chart, COLOR_TEXT_DIM, LV_CHART_AXIS_PRIMARY_Y);
    lv_obj_set_style_size(s_vital_chart, 0, LV_PART_INDICATOR);
    lv_obj_set_style_line_width(s_vital_chart, 2, LV_PART_ITEMS);

    /* BPM readouts */
    s_vital_bpm_br = make_label(tile, "Breathing: -- BPM", &s_style_label);
    lv_obj_align(s_vital_bpm_br, LV_ALIGN_BOTTOM_LEFT, 4, -4);
    lv_obj_set_style_text_color(s_vital_bpm_br, COLOR_WHITE, 0);

    s_vital_bpm_hr = make_label(tile, "Heart Rate: -- BPM", &s_style_label);
    lv_obj_align(s_vital_bpm_hr, LV_ALIGN_BOTTOM_RIGHT, -4, -4);
    lv_obj_set_style_text_color(s_vital_bpm_hr, COLOR_TEXT_DIM, 0);
}

/* ---- View 2: Presence Grid ---- */
static void create_presence(lv_obj_t *tile)
{
    make_label(tile, "Occupancy", &s_style_label);

    s_presence_label = make_label(tile, "Persons: 0", &s_style_label);
    lv_obj_align(s_presence_label, LV_ALIGN_TOP_RIGHT, -2, 0);

    int cell_w = (SCREEN_W - ((GRID_COLS - 1) * GRID_GAP)) / GRID_COLS;
    int cell_h = (SCREEN_H - 13 - ((GRID_ROWS - 1) * GRID_GAP)) / GRID_ROWS;
    int grid_w = GRID_COLS * cell_w + (GRID_COLS - 1) * GRID_GAP;
    int x_off  = (SCREEN_W - grid_w) / 2;
    int y_off  = 13;

    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            lv_obj_t *cell = lv_obj_create(tile);
            lv_obj_set_size(cell, cell_w, cell_h);
            lv_obj_set_pos(cell, x_off + c * (cell_w + GRID_GAP), y_off + r * (cell_h + GRID_GAP));
            lv_obj_set_style_bg_color(cell, COLOR_DIM, 0);
            lv_obj_set_style_bg_opa(cell, LV_OPA_COVER, 0);
            lv_obj_set_style_border_width(cell, 0, 0);
            lv_obj_set_style_pad_all(cell, 0, 0);
            lv_obj_set_style_radius(cell, 0, 0);
            s_grid_cells[r * GRID_COLS + c] = cell;
        }
    }
}

/* ---- View 3: System ---- */
static void create_system(lv_obj_t *tile)
{
    /* Two-column raw text, no containing panel. */
    s_sys_node    = make_label(tile, "Node: --",          &s_style_label);
    s_sys_cpu     = make_label(tile, "CPU: --%",          &s_style_label);
    s_sys_heap    = make_label(tile, "Heap: -- KB free",  &s_style_label);
    s_sys_psram   = make_label(tile, "PSRAM: -- KB free", &s_style_label);
    s_sys_rssi    = make_label(tile, "WiFi RSSI: --",     &s_style_label);

    s_sys_battery = make_label(tile, "Battery: UNKNOWN",  &s_style_label);
    s_sys_power   = make_label(tile, "Power: UNKNOWN",    &s_style_label);
    s_sys_peer    = make_label(tile, "Peer: WAITING",     &s_style_label);
    s_sys_uptime  = make_label(tile, "Uptime: --",        &s_style_label);
    s_sys_fps     = make_label(tile, "FPS: --",           &s_style_label);

    lv_obj_set_pos(s_sys_node,    0,   0);
    lv_obj_set_pos(s_sys_cpu,     0,  13);
    lv_obj_set_pos(s_sys_heap,    0,  26);
    lv_obj_set_pos(s_sys_psram,   0,  39);
    lv_obj_set_pos(s_sys_rssi,    0,  52);

    lv_obj_set_pos(s_sys_battery, 112,  0);
    lv_obj_set_pos(s_sys_power,   112, 13);
    lv_obj_set_pos(s_sys_peer,    112, 26);
    lv_obj_set_pos(s_sys_uptime,  112, 39);
    lv_obj_set_pos(s_sys_fps,     112, 52);
}

/* ---- Public API ---- */

void display_ui_create(lv_obj_t *parent)
{
    init_styles();

    s_tileview = lv_tileview_create(parent);
    lv_obj_add_style(s_tileview, &s_style_bg, 0);
    lv_obj_set_style_bg_color(s_tileview, COLOR_BG, 0);

    lv_obj_t *t0 = make_tile(s_tileview, 0, 0);
    lv_obj_t *t1 = make_tile(s_tileview, 1, 0);
    lv_obj_t *t2 = make_tile(s_tileview, 2, 0);
    lv_obj_t *t3 = make_tile(s_tileview, 3, 0);

    create_dashboard(t0);
    create_vitals(t1);
    create_presence(t2);
    create_system(t3);
    create_view_tabs(parent);

    ESP_LOGI(TAG, "UI created: 4 views with tab buttons (DASH|VITAL|PRES|SYS)");
}

/* ---- FPS tracking ---- */
static uint32_t s_frame_count = 0;
static uint32_t s_last_fps_time = 0;
static uint32_t s_current_fps = 0;
static uint32_t s_last_content_refresh_ms = 0;

void display_ui_update(void)
{
    /* FPS counter */
    s_frame_count++;
    s_debug_total_frames++;
    if (s_debug_frame) {
        char dbg[32];
        snprintf(dbg, sizeof(dbg), "FRAME %06lu", (unsigned long)s_debug_total_frames);
        lv_label_set_text(s_debug_frame, dbg);
        lv_obj_move_foreground(s_debug_frame);
    }
    uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
#if !defined(CONFIG_CARDPUTER_ADV_TRAUTONIUM_ENABLE)
    update_keyboard(now_ms);
#endif
    update_auto_view(now_ms);
    update_scan_marker(now_ms);
    if (now_ms - s_last_fps_time >= 1000) {
        s_current_fps = s_frame_count;
        s_frame_count = 0;
        s_last_fps_time = now_ms;
    }

    if (s_last_content_refresh_ms != 0 &&
        (uint32_t)(now_ms - s_last_content_refresh_ms) < UI_CONTENT_REFRESH_MS) {
        return;
    }
    s_last_content_refresh_ms = now_ms;

    /* Read edge data (thread-safe) */
    edge_vitals_pkt_t vitals;
    bool has_vitals = edge_get_vitals(&vitals);

    /* ---- Dashboard update ---- */
    if (s_dash_chart) {
        int val = moving_trace_value(now_ms, has_vitals, &vitals);
        lv_chart_set_next_value(s_dash_chart, s_csi_series, val);
    }

    if (s_dash_persons) {
        char buf[8];
        snprintf(buf, sizeof(buf), "P%u", has_vitals ? vitals.n_persons : 0);
        lv_label_set_text(s_dash_persons, buf);
    }

    if (s_dash_rssi && has_vitals) {
        char buf[16];
        snprintf(buf, sizeof(buf), "R%d", vitals.rssi);
        lv_label_set_text(s_dash_rssi, buf);
    }

    if (s_dash_motion) {
        char buf[24];
        if (has_vitals) {
            snprintf(buf, sizeof(buf), "M%.1f", (double)vitals.motion_energy);
        } else {
            snprintf(buf, sizeof(buf), "M scan");
        }
        lv_label_set_text(s_dash_motion, buf);
    }

    if (s_dash_server) {
        lv_label_set_text(s_dash_server, has_vitals ? "SRC LIVE" : "SRC WAIT");
    }

    {
        char buf[48];
        battery_status_t battery;
        if (s_dash_adv && battery_monitor_read(&battery) == ESP_OK && battery.valid) {
            format_battery_line(buf, sizeof(buf), "ADV", csi_collector_get_node_id(), true,
                                0x01 | (battery.charging ? 0x02 : 0x00),
                                battery.percent, battery.millivolts,
                                (uint8_t)battery.status, 0);
            lv_label_set_text(s_dash_adv, buf);
        } else if (s_dash_adv) {
            format_battery_line(buf, sizeof(buf), "ADV", csi_collector_get_node_id(), false,
                                0, 255, 0, BATTERY_POWER_UNKNOWN, 0);
            lv_label_set_text(s_dash_adv, buf);
        }

        c6_espnow_peer_status_t peer;
        bool peer_live = c6_sync_espnow_get_peer_status(&peer);
        if (s_dash_stick) {
            format_battery_line(buf, sizeof(buf), "STK", peer.node_id, peer_live,
                                peer.flags, peer.percent, peer.millivolts,
                                peer.status, peer.age_ms);
            lv_label_set_text(s_dash_stick, buf);
        }
    }

    /* ---- Vitals update ---- */
    if (s_vital_chart) {
        int br = has_vitals ? (int)(vitals.breathing_rate / 100) : 0;  /* Fixed-point to int BPM */
        int hr = has_vitals ? (int)(vitals.heartrate / 10000) : 0;
        if (br <= 0) br = 12 + (int)((now_ms / 500) % 8);
        if (hr <= 0) hr = 58 + (int)((now_ms / 180) % 22);
        if (br > 120) br = 120;
        if (hr > 120) hr = 120;
        lv_chart_set_next_value(s_vital_chart, s_breath_series, br);
        lv_chart_set_next_value(s_vital_chart, s_hr_series, hr);

        char buf[32];
        snprintf(buf, sizeof(buf), "Breathing: %d BPM", br);
        lv_label_set_text(s_vital_bpm_br, buf);

        snprintf(buf, sizeof(buf), "Heart Rate: %d BPM", hr);
        lv_label_set_text(s_vital_bpm_hr, buf);
    }

    /* ---- Presence grid update ---- */
    {
        uint8_t active_cells = 0;
        uint8_t sweep_cell = (uint8_t)((now_ms / 85) % (GRID_COLS * GRID_ROWS));

        if (has_vitals) {
            float energy = vitals.motion_energy;
            active_cells = (uint8_t)(energy * 2);  /* Scale for visibility */
            if (active_cells > GRID_COLS * GRID_ROWS) active_cells = GRID_COLS * GRID_ROWS;
        } else {
            active_cells = (uint8_t)(4 + ((now_ms / 250) % 18));
        }

        for (int i = 0; i < GRID_COLS * GRID_ROWS; i++) {
            if (i == sweep_cell) {
                lv_obj_set_style_bg_color(s_grid_cells[i], COLOR_WHITE, 0);
            } else if (i < active_cells) {
                lv_obj_set_style_bg_color(s_grid_cells[i], COLOR_TEXT_DIM, 0);
            } else {
                lv_obj_set_style_bg_color(s_grid_cells[i], COLOR_DIM, 0);
            }
        }

        char buf[20];
        snprintf(buf, sizeof(buf), has_vitals ? "Persons: %u" : "SCAN %02u",
                 has_vitals ? vitals.n_persons : sweep_cell);
        lv_label_set_text(s_presence_label, buf);
    }

    /* ---- System info update ---- */
    {
        char buf[48];

        snprintf(buf, sizeof(buf), "Node: %u", (unsigned)csi_collector_get_node_id());
        lv_label_set_text(s_sys_node, buf);

        snprintf(buf, sizeof(buf), "Heap: %lu KB free",
                 (unsigned long)(esp_get_free_heap_size() / 1024));
        lv_label_set_text(s_sys_heap, buf);

#if CONFIG_SPIRAM
        snprintf(buf, sizeof(buf), "PSRAM: %lu KB free",
                 (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
#else
        snprintf(buf, sizeof(buf), "PSRAM: N/A");
#endif
        lv_label_set_text(s_sys_psram, buf);

        if (has_vitals) {
            snprintf(buf, sizeof(buf), "WiFi RSSI: %d dBm", vitals.rssi);
            lv_label_set_text(s_sys_rssi, buf);
        }

        battery_status_t battery;
        if (battery_monitor_read(&battery) == ESP_OK && battery.valid) {
            snprintf(buf, sizeof(buf), "Battery: %u%% (%umV)",
                     (unsigned)battery.percent, (unsigned)battery.millivolts);
            lv_label_set_text(s_sys_battery, buf);
            snprintf(buf, sizeof(buf), "Power: %s",
                     battery_monitor_status_name(battery.status));
            lv_label_set_text(s_sys_power, buf);
        } else {
            lv_label_set_text(s_sys_battery, "Battery: UNKNOWN");
            lv_label_set_text(s_sys_power, "Power: UNKNOWN");
        }

        c6_espnow_peer_status_t peer;
        if (c6_sync_espnow_get_peer_status(&peer)) {
            if (peer.flags & 0x01) {
                snprintf(buf, sizeof(buf), "Peer n%u: %u%% %umV",
                         (unsigned)peer.node_id, (unsigned)peer.percent,
                         (unsigned)peer.millivolts);
            } else {
                snprintf(buf, sizeof(buf), "Peer n%u: LIVE batt?", (unsigned)peer.node_id);
            }
            lv_label_set_text(s_sys_peer, buf);
        } else {
            lv_label_set_text(s_sys_peer, "Peer: WAITING");
        }

        uint32_t uptime_s = (uint32_t)(esp_timer_get_time() / 1000000);
        uint32_t h = uptime_s / 3600;
        uint32_t m = (uptime_s % 3600) / 60;
        uint32_t s = uptime_s % 60;
        snprintf(buf, sizeof(buf), "Uptime: %luh %02lum %02lus",
                 (unsigned long)h, (unsigned long)m, (unsigned long)s);
        lv_label_set_text(s_sys_uptime, buf);

        snprintf(buf, sizeof(buf), "FPS: %lu", (unsigned long)s_current_fps);
        lv_label_set_text(s_sys_fps, buf);
    }
}

#endif /* CONFIG_DISPLAY_ENABLE */

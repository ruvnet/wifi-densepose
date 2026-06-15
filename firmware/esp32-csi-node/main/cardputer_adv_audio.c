/**
 * @file cardputer_adv_audio.c
 * @brief Cardputer-Adv ES8311 speaker and microSD WAV playback.
 */

#include "cardputer_adv_audio.h"
#include "sdkconfig.h"

#if defined(CONFIG_CARDPUTER_ADV_AUDIO_ENABLE) && defined(CONFIG_IDF_TARGET_ESP32S3)

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "driver/sdspi_host.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

static const char *TAG = "cardputer_adv_audio";

/* Cardputer-Adv pin map from M5Stack docs. */
#define ADV_CODEC_I2C_PORT I2C_NUM_0
#define ADV_CODEC_I2C_SDA  GPIO_NUM_8
#define ADV_CODEC_I2C_SCL  GPIO_NUM_9
#define ADV_CODEC_ADDR     0x18

#define ADV_KB_INT         GPIO_NUM_11
#define ADV_KB_ADDR        0x34
#define ADV_KB_REG_CFG        0x01
#define ADV_KB_REG_INT_STAT   0x02
#define ADV_KB_REG_KEY_LCK_EC 0x03
#define ADV_KB_REG_KEY_EVENT  0x04
#define ADV_KB_REG_GPI_EM1    0x09
#define ADV_KB_REG_KP_GPIO1   0x1D
#define ADV_KB_REG_KP_GPIO2   0x1E
#define ADV_KB_REG_KP_GPIO3   0x1F

#define ADV_I2S_PORT       I2S_NUM_1
#define ADV_I2S_BCLK       GPIO_NUM_41
#define ADV_I2S_WS         GPIO_NUM_43
#define ADV_I2S_DOUT       GPIO_NUM_42

#define ADV_SD_HOST        SPI3_HOST
#define ADV_SD_MOSI        GPIO_NUM_14
#define ADV_SD_MISO        GPIO_NUM_39
#define ADV_SD_SCLK        GPIO_NUM_40
#define ADV_SD_CS          GPIO_NUM_12

#define ADV_CHIME_RATE_HZ  16000
#define ADV_CHIME_MS       180
#define ADV_CHIME_STEP     28
#define ADV_IO_TIMEOUT_MS  1000
#define ADV_SYNTH_RATE_HZ  22050
#define ADV_SYNTH_FRAMES   96
#define ADV_SYNTH_MAX_KEY  80
#define ADV_SYNTH_BIG_KNOB_DEFAULT 68
#define ADV_SYNTH_SPRING_A 293
#define ADV_SYNTH_SPRING_B 421
#define ADV_SYNTH_SPRING_C 613

typedef struct {
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    uint32_t data_bytes;
} wav_info_t;

static i2c_master_bus_handle_t s_i2c_bus;
static i2c_master_dev_handle_t s_codec_dev;
static i2s_chan_handle_t s_i2s_tx;
static uint32_t s_i2s_rate_hz;
static bool s_i2s_running;
static i2c_master_dev_handle_t s_kb_dev;
static bool s_sd_mount_attempted;
static bool s_sd_mounted;
static esp_err_t s_sd_mount_err = ESP_OK;
static sdmmc_card_t *s_sd_card;
static uint8_t s_raw_buf[1024];
static int16_t s_pcm_buf[1024];
static SemaphoreHandle_t s_audio_mutex;

#if defined(CONFIG_CARDPUTER_ADV_TRAUTONIUM_ENABLE)
typedef struct {
    uint16_t f1;
    uint16_t f2;
    uint16_t damp1;
    uint16_t damp2;
    int16_t gain1;
    int16_t gain2;
    uint8_t formant_mix;
    uint8_t trap_edge;
    const char *name;
} trautonium_formant_t;

typedef struct {
    uint8_t key_code;
    int8_t semitone;
    uint8_t formant_index;
    const char *label;
} trautonium_key_note_t;

static const trautonium_formant_t s_formants[] = {
    /* Historical color approximations: neutral, dark, nasal, low throat. */
    {2700,  7600, 11500, 13500, 190,  82, 140,  9, "neutral"},
    {1500,  4300, 13500, 15000, 210,  70, 155, 12, "dark"},
    {4300, 10400,  9000, 12000, 230, 105, 180,  6, "nasal"},
    {1050,  3100, 15000, 16000, 240,  55, 165, 14, "low"},
};

/*
 * Cardputer-ADV TCA8418 keycodes are column-scanned, not row-major. This table
 * maps printed key labels into a playable typing-keyboard piano layout.
 */
static const trautonium_key_note_t s_key_notes[] = {
    /* Number row: high chromatic rail. */
    {5,  24, 0, "1"}, {11, 25, 0, "2"}, {15, 26, 0, "3"}, {21, 27, 0, "4"},
    {25, 28, 0, "5"}, {31, 29, 0, "6"}, {35, 30, 0, "7"}, {41, 31, 0, "8"},
    {45, 32, 0, "9"}, {51, 33, 0, "0"}, {55, 34, 0, "-"}, {61, 35, 0, "="},

    /* Q row: black-key rail plus duplicate anchors. Printed T is F#4. */
    {6,  12, 2, "Q"}, {12, 13, 2, "W"}, {16, 15, 2, "E"}, {22, 17, 2, "R"},
    {26, 18, 2, "T"}, {32, 20, 2, "Y"}, {36, 22, 2, "U"}, {42, 24, 2, "I"},
    {46, 25, 2, "O"}, {52, 27, 2, "P"},

    /* A row: white keys, C4 upward. */
    {13, 12, 1, "A"}, {17, 14, 1, "S"}, {23, 16, 1, "D"}, {27, 17, 1, "F"},
    {33, 19, 1, "G"}, {37, 21, 1, "H"}, {43, 23, 1, "J"}, {47, 24, 1, "K"},
    {53, 26, 1, "L"},

    /* Z row: lower white keys, C3 upward. */
    {18, 0, 3, "Z"}, {24, 2, 3, "X"}, {28, 4, 3, "C"}, {34, 5, 3, "V"},
    {38, 7, 3, "B"}, {44, 9, 3, "N"}, {48, 11, 3, "M"},
};

static TaskHandle_t s_synth_task;
static TaskHandle_t s_keyscan_task;
static portMUX_TYPE s_synth_lock = portMUX_INITIALIZER_UNLOCKED;
static uint8_t s_synth_active_key;
static uint8_t s_synth_formant_index;
static uint8_t s_synth_pressure_percent;
static uint8_t s_synth_big_knob_percent;
static uint8_t s_synth_sh_trigger;
static uint32_t s_synth_target_mhz;
static bool s_synth_gate;
static int16_t s_synth_buf[ADV_SYNTH_FRAMES * 2];
static int32_t s_spring_a[ADV_SYNTH_SPRING_A];
static int32_t s_spring_b[ADV_SYNTH_SPRING_B];
static int32_t s_spring_c[ADV_SYNTH_SPRING_C];
static uint16_t s_spring_a_pos;
static uint16_t s_spring_b_pos;
static uint16_t s_spring_c_pos;
#endif

static void ensure_audio_mutex(void)
{
    if (s_audio_mutex == NULL) {
        s_audio_mutex = xSemaphoreCreateMutex();
    }
}

static esp_err_t audio_lock(TickType_t timeout)
{
    ensure_audio_mutex();
    if (s_audio_mutex == NULL) {
        return ESP_ERR_NO_MEM;
    }
    return xSemaphoreTake(s_audio_mutex, timeout) == pdTRUE ? ESP_OK : ESP_ERR_TIMEOUT;
}

static void audio_unlock(void)
{
    if (s_audio_mutex != NULL) {
        xSemaphoreGive(s_audio_mutex);
    }
}

static uint32_t read_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t read_le16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int16_t scale_sample(int16_t sample)
{
    int volume = CONFIG_CARDPUTER_ADV_AUDIO_VOLUME_PERCENT;
    if (volume < 0) {
        volume = 0;
    } else if (volume > 100) {
        volume = 100;
    }
    return (int16_t)(((int32_t)sample * volume) / 100);
}

static esp_err_t get_i2c_bus(void)
{
    if (s_i2c_bus != NULL) {
        return ESP_OK;
    }

    esp_err_t err = i2c_master_get_bus_handle(ADV_CODEC_I2C_PORT, &s_i2c_bus);
    if (err == ESP_OK) {
        return ESP_OK;
    }

    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = ADV_CODEC_I2C_PORT,
        .sda_io_num = ADV_CODEC_I2C_SDA,
        .scl_io_num = ADV_CODEC_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    err = i2c_new_master_bus(&bus_cfg, &s_i2c_bus);
    if (err == ESP_ERR_INVALID_STATE) {
        err = i2c_master_get_bus_handle(ADV_CODEC_I2C_PORT, &s_i2c_bus);
    }
    return err;
}

static esp_err_t get_codec_dev(void)
{
    if (s_codec_dev != NULL) {
        return ESP_OK;
    }

    esp_err_t err = get_i2c_bus();
    if (err != ESP_OK) {
        return err;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = ADV_CODEC_ADDR,
        .scl_speed_hz = 400000,
    };
    return i2c_master_bus_add_device(s_i2c_bus, &dev_cfg, &s_codec_dev);
}

static esp_err_t codec_write_reg(uint8_t reg, uint8_t value)
{
    uint8_t data[2] = {reg, value};
    return i2c_master_transmit(s_codec_dev, data, sizeof(data), ADV_IO_TIMEOUT_MS);
}

static esp_err_t init_codec(void)
{
    static bool codec_ready;
    if (codec_ready) {
        return ESP_OK;
    }

    esp_err_t err = get_codec_dev();
    if (err != ESP_OK) {
        return err;
    }

    static const struct {
        uint8_t reg;
        uint8_t value;
    } init_seq[] = {
        {0x00, 0x80},
        {0x01, 0xB5},
        {0x02, 0x18},
        {0x0D, 0x01},
        {0x12, 0x00},
        {0x13, 0x10},
        {0x32, 0xBF},
        {0x37, 0x08},
    };

    for (size_t i = 0; i < sizeof(init_seq) / sizeof(init_seq[0]); i++) {
        err = codec_write_reg(init_seq[i].reg, init_seq[i].value);
        if (err != ESP_OK) {
            return err;
        }
    }

    codec_ready = true;
    ESP_LOGI(TAG, "ES8311 ready on I2C0 addr=0x%02x", ADV_CODEC_ADDR);
    return ESP_OK;
}

static void stop_i2s(void)
{
    if (s_i2s_tx == NULL) {
        return;
    }
    if (s_i2s_running) {
        i2s_channel_disable(s_i2s_tx);
        s_i2s_running = false;
    }
    i2s_del_channel(s_i2s_tx);
    s_i2s_tx = NULL;
    s_i2s_rate_hz = 0;
}

static esp_err_t ensure_i2s(uint32_t sample_rate)
{
    if (s_i2s_tx != NULL && s_i2s_rate_hz == sample_rate) {
        return ESP_OK;
    }
    stop_i2s();

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(ADV_I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 512;
    chan_cfg.auto_clear = true;

    esp_err_t err = i2s_new_channel(&chan_cfg, &s_i2s_tx, NULL);
    if (err != ESP_OK) {
        return err;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = ADV_I2S_BCLK,
            .ws = ADV_I2S_WS,
            .dout = ADV_I2S_DOUT,
            .din = I2S_GPIO_UNUSED,
        },
    };
    err = i2s_channel_init_std_mode(s_i2s_tx, &std_cfg);
    if (err != ESP_OK) {
        stop_i2s();
        return err;
    }

    err = i2s_channel_enable(s_i2s_tx);
    if (err != ESP_OK) {
        stop_i2s();
        return err;
    }

    s_i2s_running = true;
    s_i2s_rate_hz = sample_rate;
    ESP_LOGI(TAG, "I2S speaker ready: rate=%lu BCLK=G%d LRCK=G%d DOUT=G%d",
             (unsigned long)sample_rate, ADV_I2S_BCLK, ADV_I2S_WS, ADV_I2S_DOUT);
    return ESP_OK;
}

static esp_err_t write_i2s_all(const void *data, size_t len)
{
    const uint8_t *cursor = (const uint8_t *)data;
    while (len > 0) {
        size_t written = 0;
        esp_err_t err = i2s_channel_write(s_i2s_tx, cursor, len, &written,
                                          pdMS_TO_TICKS(ADV_IO_TIMEOUT_MS));
        if (err != ESP_OK) {
            return err;
        }
        if (written == 0) {
            return ESP_ERR_TIMEOUT;
        }
        cursor += written;
        len -= written;
    }
    return ESP_OK;
}

static esp_err_t write_silence(uint32_t sample_rate, uint32_t ms)
{
    memset(s_pcm_buf, 0, sizeof(s_pcm_buf));
    uint32_t frames = (sample_rate * ms) / 1000;
    while (frames > 0) {
        uint32_t chunk_frames = frames;
        if (chunk_frames > sizeof(s_pcm_buf) / (sizeof(int16_t) * 2)) {
            chunk_frames = sizeof(s_pcm_buf) / (sizeof(int16_t) * 2);
        }
        esp_err_t err = write_i2s_all(s_pcm_buf, chunk_frames * 2 * sizeof(int16_t));
        if (err != ESP_OK) {
            return err;
        }
        frames -= chunk_frames;
    }
    return ESP_OK;
}

#if defined(CONFIG_CARDPUTER_ADV_TRAUTONIUM_ENABLE)
static int32_t clamp_i32(int32_t value, int32_t lo, int32_t hi)
{
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static const trautonium_key_note_t *trautonium_find_key_note(uint8_t key_code)
{
    for (size_t i = 0; i < sizeof(s_key_notes) / sizeof(s_key_notes[0]); i++) {
        if (s_key_notes[i].key_code == key_code) {
            return &s_key_notes[i];
        }
    }
    return NULL;
}

static uint32_t trautonium_note_freq_mhz(int8_t semitone)
{
    static const uint16_t semitone_ratio_permille[12] = {
        1000, 1059, 1122, 1189, 1259, 1335,
        1414, 1498, 1587, 1682, 1782, 1888,
    };

    if (semitone < 0) {
        semitone = 0;
    }

    uint32_t freq_mhz = (uint32_t)CONFIG_CARDPUTER_ADV_TRAUTONIUM_BASE_FREQ_HZ * 1000U;
    for (int8_t octave = 0; octave < semitone / 12; octave++) {
        if (freq_mhz > 2500000U) {
            return 5000000U;
        }
        freq_mhz *= 2U;
    }

    freq_mhz = (uint32_t)(((uint64_t)freq_mhz *
                           semitone_ratio_permille[semitone % 12] + 500ULL) / 1000ULL);
    return freq_mhz > 5000000U ? 5000000U : freq_mhz;
}

static uint32_t trautonium_key_freq_mhz(uint8_t key_code)
{
    const trautonium_key_note_t *note = trautonium_find_key_note(key_code);
    return note != NULL ? trautonium_note_freq_mhz(note->semitone) : 0;
}

static int8_t trautonium_raw_fallback_semitone(uint8_t key_code)
{
    if (key_code == 0) {
        return 0;
    }
    return (int8_t)((key_code - 1) % 36);
}

static uint8_t trautonium_key_pressure(uint8_t key_code)
{
    if (key_code == 0) {
        return 62;
    }
    uint8_t column = (uint8_t)((key_code - 1) % 10);
    uint8_t pressure = (uint8_t)(54 + column * 4);
    return pressure > 92 ? 92 : pressure;
}

static uint8_t trautonium_big_knob_for_key(uint8_t key_code, uint8_t fallback)
{
    static const uint8_t knob_keys[] = {
        5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61,
    };

    for (size_t i = 0; i < sizeof(knob_keys) / sizeof(knob_keys[0]); i++) {
        if (knob_keys[i] == key_code) {
            return (uint8_t)(8U + (uint8_t)i * 8U);
        }
    }
    return fallback;
}

static bool trautonium_handle_macro_key(uint8_t key_code, bool pressed)
{
    int8_t delta = 0;
    uint8_t set_value = 0xFF;

    /*
     * Likely ADV punctuation raw keys. If a board revision reports these
     * differently, note keys still play; the number row remains the reliable
     * big-knob rail.
     */
    switch (key_code) {
    case 54: /* comma: less edge/S&H/spring */
        delta = -10;
        break;
    case 58: /* period: more edge/S&H/spring */
        delta = 10;
        break;
    case 64: /* slash: center macro */
        set_value = ADV_SYNTH_BIG_KNOB_DEFAULT;
        break;
    default:
        return false;
    }

    if (!pressed) {
        return true;
    }

    portENTER_CRITICAL(&s_synth_lock);
    if (set_value != 0xFF) {
        s_synth_big_knob_percent = set_value;
    } else {
        int32_t next = (int32_t)s_synth_big_knob_percent + delta;
        s_synth_big_knob_percent = (uint8_t)clamp_i32(next, 0, 100);
    }
    s_synth_sh_trigger++;
    uint8_t knob = s_synth_big_knob_percent;
    portEXIT_CRITICAL(&s_synth_lock);

    ESP_LOGI(TAG, "Trautonium big knob=%u raw=%u", (unsigned)knob, (unsigned)key_code);
    return true;
}

static int32_t trapezoid_wave(uint32_t phase, uint8_t edge_units)
{
    const int32_t amp = 28000;
    uint32_t edge = 1024U + (uint32_t)edge_units * 512U;
    if (edge > 15000U) {
        edge = 15000U;
    }

    uint32_t p = phase >> 16;
    uint32_t rise_end = edge;
    uint32_t high_end = 32768U > edge ? 32768U - edge : 1U;
    uint32_t fall_end = 32768U + edge;
    uint32_t low_end = 65536U > edge ? 65536U - edge : 65535U;

    if (p < rise_end) {
        return -amp + (int32_t)(((uint64_t)p * 2ULL * (uint64_t)amp) / edge);
    }
    if (p < high_end) {
        return amp;
    }
    if (p < fall_end) {
        uint32_t width = fall_end - high_end;
        return amp - (int32_t)(((uint64_t)(p - high_end) * 2ULL * (uint64_t)amp) / width);
    }
    if (p < low_end) {
        return -amp;
    }
    return -amp + (int32_t)(((uint64_t)(p - low_end) * 2ULL * (uint64_t)amp) / edge);
}

static int32_t svf_bandpass(int32_t input, uint16_t freq, uint16_t damp,
                            int32_t *low, int32_t *band)
{
    int32_t high = input - *low - (int32_t)(((int64_t)*band * damp) >> 15);
    *band += (int32_t)(((int64_t)freq * high) >> 15);
    *low += (int32_t)(((int64_t)freq * *band) >> 15);
    *band = clamp_i32(*band, -160000, 160000);
    *low = clamp_i32(*low, -160000, 160000);
    return *band;
}

static uint32_t approach_u32(uint32_t current, uint32_t target, uint32_t step)
{
    if (current < target) {
        uint32_t next = current + step;
        return next < current || next > target ? target : next;
    }
    if (current > target) {
        return current - target <= step ? target : current - step;
    }
    return current;
}

static void spring_reverb_process(int32_t input, uint8_t big_knob,
                                  int32_t *left, int32_t *right)
{
    int32_t a = s_spring_a[s_spring_a_pos];
    int32_t b = s_spring_b[s_spring_b_pos];
    int32_t c = s_spring_c[s_spring_c_pos];
    int32_t feedback = 116 + ((int32_t)big_knob * 58) / 100;
    int32_t cross = 18 + ((int32_t)big_knob * 34) / 100;
    int32_t tank_in = input + (((b - c) * cross) >> 8);

    s_spring_a[s_spring_a_pos] = clamp_i32(tank_in + ((a * feedback) >> 8),
                                           -130000, 130000);
    s_spring_b[s_spring_b_pos] = clamp_i32((tank_in >> 1) + (a >> 3) +
                                           ((b * (feedback - 12)) >> 8),
                                           -130000, 130000);
    s_spring_c[s_spring_c_pos] = clamp_i32((tank_in >> 2) - (b >> 3) +
                                           ((c * (feedback - 24)) >> 8),
                                           -130000, 130000);

    if (++s_spring_a_pos >= ADV_SYNTH_SPRING_A) {
        s_spring_a_pos = 0;
    }
    if (++s_spring_b_pos >= ADV_SYNTH_SPRING_B) {
        s_spring_b_pos = 0;
    }
    if (++s_spring_c_pos >= ADV_SYNTH_SPRING_C) {
        s_spring_c_pos = 0;
    }

    int32_t wet_l = (a + b - c) / 3;
    int32_t wet_r = (c + b - a) / 3;
    int32_t mix = 18 + ((int32_t)big_knob * 42) / 100;
    *left = ((input * (100 - mix)) + (wet_l * mix)) / 100;
    *right = ((input * (100 - mix)) + (wet_r * mix)) / 100;
}

static void trautonium_synth_task(void *arg)
{
    (void)arg;
    uint32_t phase = 0;
    uint32_t sub1_phase = 0;
    uint32_t sub2_phase = 0;
    uint32_t current_mhz = trautonium_key_freq_mhz(18);
    uint32_t env = 0;
    int32_t f1_low = 0;
    int32_t f1_band = 0;
    int32_t f2_low = 0;
    int32_t f2_band = 0;
    uint8_t last_formant = 0xFF;
    uint8_t last_sh_trigger = 0;
    uint32_t sh_rng = 0x51F15EEDU;
    uint32_t sh_counter = 0;
    int32_t sh_value = 0;
    uint32_t spring_tail = 0;
    uint32_t err_count = 0;

    for (;;) {
        uint32_t target_mhz;
        bool gate;
        uint8_t formant_index;
        uint8_t pressure;
        uint8_t big_knob;
        uint8_t sh_trigger;

        portENTER_CRITICAL(&s_synth_lock);
        target_mhz = s_synth_target_mhz != 0 ? s_synth_target_mhz : trautonium_key_freq_mhz(18);
        gate = s_synth_gate;
        formant_index = s_synth_formant_index;
        pressure = s_synth_pressure_percent != 0 ? s_synth_pressure_percent : 62;
        big_knob = s_synth_big_knob_percent;
        sh_trigger = s_synth_sh_trigger;
        portEXIT_CRITICAL(&s_synth_lock);

        if (big_knob > 100) {
            big_knob = ADV_SYNTH_BIG_KNOB_DEFAULT;
        }
        if (!gate && env == 0 && spring_tail == 0) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (sh_trigger != last_sh_trigger) {
            last_sh_trigger = sh_trigger;
            sh_counter = 0;
        }

        if (formant_index >= sizeof(s_formants) / sizeof(s_formants[0])) {
            formant_index = 0;
        }
        const trautonium_formant_t *formant = &s_formants[formant_index];
        if (last_formant != formant_index) {
            f1_low = 0;
            f1_band = 0;
            f2_low = 0;
            f2_band = 0;
            last_formant = formant_index;
        }

        uint32_t glide_chunks = ((uint32_t)CONFIG_CARDPUTER_ADV_TRAUTONIUM_PORTAMENTO_MS *
                                 ADV_SYNTH_RATE_HZ) / (1000U * ADV_SYNTH_FRAMES);
        if (glide_chunks == 0) {
            glide_chunks = 1;
        }
        uint32_t diff = current_mhz > target_mhz ? current_mhz - target_mhz : target_mhz - current_mhz;
        uint32_t step = diff / glide_chunks;
        if (step == 0 && diff != 0) {
            step = 1;
        }
        current_mhz = approach_u32(current_mhz, target_mhz, step);

        uint32_t phase_inc = (uint32_t)(((uint64_t)current_mhz << 32) /
                                        ((uint64_t)ADV_SYNTH_RATE_HZ * 1000ULL));

        for (uint32_t i = 0; i < ADV_SYNTH_FRAMES; i++) {
            if (gate) {
                if (sh_counter == 0) {
                    uint32_t sh_rate = 4U + ((uint32_t)big_knob * 18U) / 100U;
                    sh_rng = sh_rng * 1664525U + 1013904223U + phase_inc + pressure;
                    sh_value = (int32_t)((sh_rng >> 24) & 0xFF) - 128;
                    sh_counter = ADV_SYNTH_RATE_HZ / sh_rate;
                } else {
                    sh_counter--;
                }
            }

            phase += phase_inc;
            sub1_phase += phase_inc >> 1;
            sub2_phase += phase_inc >> 2;

            if (gate) {
                env += (65535U - env) >> 5;
                if (env < 65535U - 96U) {
                    env += 96U;
                } else {
                    env = 65535U;
                }
            } else if (env > 0) {
                uint32_t decay = (env >> 8) + 48U;
                env = env > decay ? env - decay : 0;
            }

            int32_t sh_depth = ((int32_t)big_knob * 52) / 100;
            int32_t edge_units = (int32_t)formant->trap_edge +
                                 ((int32_t)big_knob / 9) +
                                 ((sh_value * sh_depth) >> 8);
            uint8_t trap_edge = (uint8_t)clamp_i32(edge_units, 1, 30);
            int32_t trap = trapezoid_wave(phase, trap_edge);
            int32_t sub1 = (sub1_phase & 0x80000000U) ? 15000 : -15000;
            int32_t sub2 = trapezoid_wave(sub2_phase,
                                          (uint8_t)clamp_i32((int32_t)trap_edge + 3, 1, 31)) / 2;
            int32_t raw = (trap * 6 + sub1 * 2 + sub2) / 9;

            int32_t driven = (raw * (int32_t)(96U + pressure)) / 128;
            driven = clamp_i32(driven, -32000, 32000);

            int32_t form_shift = (sh_value * (int32_t)(420U + big_knob * 7U)) / 128;
            uint16_t f1 = (uint16_t)clamp_i32((int32_t)formant->f1 + form_shift, 700, 12000);
            uint16_t f2 = (uint16_t)clamp_i32((int32_t)formant->f2 + form_shift * 2, 1800, 15500);
            int32_t bp1 = svf_bandpass(driven, f1, formant->damp1, &f1_low, &f1_band);
            int32_t bp2 = svf_bandpass(driven, f2, formant->damp2, &f2_low, &f2_band);
            int32_t voiced_formant = (bp1 * formant->gain1 + bp2 * formant->gain2) >> 8;
            int32_t mixed = (driven * (int32_t)(256U - formant->formant_mix) +
                             voiced_formant * formant->formant_mix) >> 8;

            int32_t sample = (int32_t)(((int64_t)mixed * env) >> 16);
            sample = (sample * CONFIG_CARDPUTER_ADV_TRAUTONIUM_LEVEL_PERCENT) / 100;
            sample = clamp_i32(sample, -30000, 30000);
            if (gate || env > 0) {
                spring_tail = (ADV_SYNTH_RATE_HZ / 5U) +
                              (((uint32_t)big_knob * ADV_SYNTH_RATE_HZ) / 180U);
            } else if (spring_tail > 0) {
                spring_tail--;
            }
            int32_t left = sample;
            int32_t right = sample;
            spring_reverb_process(sample, big_knob, &left, &right);
            s_synth_buf[i * 2] = (int16_t)clamp_i32(left, -30000, 30000);
            s_synth_buf[i * 2 + 1] = (int16_t)clamp_i32(right, -30000, 30000);
        }

        esp_err_t err = audio_lock(pdMS_TO_TICKS(100));
        if (err == ESP_OK) {
            err = init_codec();
            if (err == ESP_OK) {
                err = ensure_i2s(ADV_SYNTH_RATE_HZ);
            }
            if (err == ESP_OK) {
                err = write_i2s_all(s_synth_buf, sizeof(s_synth_buf));
            }
            audio_unlock();
        }
        if (err != ESP_OK) {
            if ((err_count++ & 0x3F) == 0) {
                ESP_LOGW(TAG, "Trautonium synth write failed: %s", esp_err_to_name(err));
            }
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

static void start_trautonium_synth_once(void)
{
    if (s_synth_task != NULL) {
        return;
    }

    s_synth_target_mhz = trautonium_key_freq_mhz(18);
    s_synth_pressure_percent = 62;
    s_synth_big_knob_percent = ADV_SYNTH_BIG_KNOB_DEFAULT;

    BaseType_t ok = xTaskCreate(trautonium_synth_task, "adv_trap_synth",
                                4096, NULL, 5, &s_synth_task);
    if (ok != pdPASS) {
        s_synth_task = NULL;
        ESP_LOGW(TAG, "failed to start Trautonium synth task");
    } else {
        ESP_LOGI(TAG, "Trautonium key synth ready: trapezoid + S/H + spring + %ums glide",
                 (unsigned)CONFIG_CARDPUTER_ADV_TRAUTONIUM_PORTAMENTO_MS);
    }
}

static esp_err_t get_keyboard_dev(void)
{
    if (s_kb_dev != NULL) {
        return ESP_OK;
    }

    esp_err_t err = get_i2c_bus();
    if (err != ESP_OK) {
        return err;
    }

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = ADV_KB_ADDR,
        .scl_speed_hz = 400000,
    };
    return i2c_master_bus_add_device(s_i2c_bus, &dev_cfg, &s_kb_dev);
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

static esp_err_t init_keyboard_scanner(void)
{
    gpio_config_t int_cfg = {
        .pin_bit_mask = 1ULL << ADV_KB_INT,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&int_cfg);

    esp_err_t err = get_keyboard_dev();
    if (err != ESP_OK) {
        return err;
    }

    err = kb_write_reg(ADV_KB_REG_KP_GPIO1, 0xFF);
    if (err == ESP_OK) {
        err = kb_write_reg(ADV_KB_REG_KP_GPIO2, 0xFF);
    }
    if (err == ESP_OK) {
        err = kb_write_reg(ADV_KB_REG_KP_GPIO3, 0x00);
    }
    if (err == ESP_OK) {
        err = kb_write_reg(ADV_KB_REG_GPI_EM1, 0x00);
    }
    if (err == ESP_OK) {
        err = kb_write_reg(ADV_KB_REG_INT_STAT, 0xFF);
    }
    if (err == ESP_OK) {
        err = kb_write_reg(ADV_KB_REG_CFG, 0x3E);
    }
    if (err != ESP_OK) {
        return err;
    }

    uint8_t ec = 0;
    err = kb_read_reg(ADV_KB_REG_KEY_LCK_EC, &ec);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "ADV keyboard audio scanner ready: TCA8418 addr=0x%02x INT=G%d",
                 ADV_KB_ADDR, ADV_KB_INT);
    }
    return err;
}

static void trautonium_keyscan_task(void *arg)
{
    (void)arg;
    esp_err_t err = init_keyboard_scanner();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "ADV keyboard audio scanner failed: %s", esp_err_to_name(err));
        s_keyscan_task = NULL;
        vTaskDelete(NULL);
        return;
    }

    for (;;) {
        uint8_t ec = 0;
        if (kb_read_reg(ADV_KB_REG_KEY_LCK_EC, &ec) == ESP_OK) {
            uint8_t count = ec & 0x0F;
            if (count == 0 && gpio_get_level(ADV_KB_INT) == 0) {
                count = 10;
            }
            if (count > 10) {
                count = 10;
            }

            for (uint8_t i = 0; i < count; i++) {
                uint8_t event = 0;
                if (kb_read_reg(ADV_KB_REG_KEY_EVENT, &event) != ESP_OK || event == 0) {
                    break;
                }
                bool pressed = (event & 0x80) != 0;
                uint8_t code = event & 0x7F;
                if (code != 0) {
                    cardputer_adv_audio_key_event(code, pressed);
                }
            }
            kb_write_reg(ADV_KB_REG_INT_STAT, 0xFF);
        }

        vTaskDelay(pdMS_TO_TICKS(8));
    }
}

static void start_keyboard_scan_once(void)
{
    if (s_keyscan_task != NULL) {
        return;
    }
    BaseType_t ok = xTaskCreate(trautonium_keyscan_task, "adv_keyscan",
                                3072, NULL, 6, &s_keyscan_task);
    if (ok != pdPASS) {
        s_keyscan_task = NULL;
        ESP_LOGW(TAG, "failed to start ADV keyboard audio scanner");
    }
}
#endif

static esp_err_t play_boot_chime(void)
{
    static const int16_t wave32[] = {
        0, 6393, 12540, 18204, 23170, 27245, 30273, 32138,
        32767, 32138, 30273, 27245, 23170, 18204, 12540, 6393,
        0, -6393, -12540, -18204, -23170, -27245, -30273, -32138,
        -32767, -32138, -30273, -27245, -23170, -18204, -12540, -6393,
    };

    esp_err_t err = audio_lock(pdMS_TO_TICKS(ADV_IO_TIMEOUT_MS));
    if (err != ESP_OK) {
        return err;
    }

    err = init_codec();
    if (err == ESP_OK) {
        err = ensure_i2s(ADV_CHIME_RATE_HZ);
    }

    uint32_t frames = (ADV_CHIME_RATE_HZ * ADV_CHIME_MS) / 1000;
    uint32_t phase = 0;
    while (err == ESP_OK && frames > 0) {
        uint32_t chunk_frames = frames;
        if (chunk_frames > sizeof(s_pcm_buf) / (sizeof(int16_t) * 2)) {
            chunk_frames = sizeof(s_pcm_buf) / (sizeof(int16_t) * 2);
        }

        for (uint32_t i = 0; i < chunk_frames; i++) {
            uint32_t envelope = (frames < ADV_CHIME_RATE_HZ / 40) ? frames : (ADV_CHIME_RATE_HZ / 40);
            int16_t sample = wave32[(phase >> 4) & 31];
            sample = (int16_t)(((int32_t)sample * (int32_t)envelope) / (ADV_CHIME_RATE_HZ / 40));
            sample = scale_sample(sample);
            s_pcm_buf[i * 2] = sample;
            s_pcm_buf[i * 2 + 1] = sample;
            phase += ADV_CHIME_STEP;
        }

        err = write_i2s_all(s_pcm_buf, chunk_frames * 2 * sizeof(int16_t));
        frames -= chunk_frames;
    }

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "ADV speaker boot chime played");
        err = write_silence(ADV_CHIME_RATE_HZ, 80);
    }
    audio_unlock();
    return err;
}

static esp_err_t mount_sd_once(void)
{
    if (s_sd_mounted) {
        return ESP_OK;
    }
    if (s_sd_mount_attempted) {
        return s_sd_mount_err;
    }
    s_sd_mount_attempted = true;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = ADV_SD_MOSI,
        .miso_io_num = ADV_SD_MISO,
        .sclk_io_num = ADV_SD_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4096,
    };

    esp_err_t err = spi_bus_initialize(ADV_SD_HOST, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        s_sd_mount_err = err;
        return err;
    }

    esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 2,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = ADV_SD_HOST;
    host.max_freq_khz = SDMMC_FREQ_DEFAULT;

    sdspi_device_config_t slot_cfg = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_cfg.host_id = ADV_SD_HOST;
    slot_cfg.gpio_cs = ADV_SD_CS;

    err = esp_vfs_fat_sdspi_mount(CONFIG_CARDPUTER_ADV_AUDIO_SD_MOUNT_POINT,
                                  &host, &slot_cfg, &mount_cfg, &s_sd_card);
    if (err != ESP_OK) {
        s_sd_mount_err = err;
        return err;
    }

    s_sd_mounted = true;
    s_sd_mount_err = ESP_OK;
    ESP_LOGI(TAG, "microSD mounted at %s CS=G%d MOSI=G%d MISO=G%d CLK=G%d",
             CONFIG_CARDPUTER_ADV_AUDIO_SD_MOUNT_POINT,
             ADV_SD_CS, ADV_SD_MOSI, ADV_SD_MISO, ADV_SD_SCLK);
    return ESP_OK;
}

static esp_err_t parse_wav(FILE *fp, wav_info_t *info)
{
    uint8_t hdr[12];
    if (fread(hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) {
        return ESP_ERR_INVALID_SIZE;
    }
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        return ESP_ERR_INVALID_RESPONSE;
    }

    bool have_fmt = false;
    bool have_data = false;
    memset(info, 0, sizeof(*info));

    while (!have_data) {
        uint8_t chunk[8];
        if (fread(chunk, 1, sizeof(chunk), fp) != sizeof(chunk)) {
            return ESP_ERR_NOT_FOUND;
        }
        uint32_t chunk_size = read_le32(chunk + 4);

        if (memcmp(chunk, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (chunk_size < sizeof(fmt) || fread(fmt, 1, sizeof(fmt), fp) != sizeof(fmt)) {
                return ESP_ERR_INVALID_SIZE;
            }
            info->audio_format = read_le16(fmt);
            info->channels = read_le16(fmt + 2);
            info->sample_rate = read_le32(fmt + 4);
            info->bits_per_sample = read_le16(fmt + 14);
            have_fmt = true;

            long extra = (long)chunk_size - (long)sizeof(fmt);
            if (extra > 0 && fseek(fp, extra, SEEK_CUR) != 0) {
                return ESP_ERR_INVALID_SIZE;
            }
        } else if (memcmp(chunk, "data", 4) == 0) {
            if (!have_fmt) {
                return ESP_ERR_INVALID_STATE;
            }
            info->data_bytes = chunk_size;
            have_data = true;
        } else {
            if (fseek(fp, (long)chunk_size, SEEK_CUR) != 0) {
                return ESP_ERR_INVALID_SIZE;
            }
        }

        if ((chunk_size & 1) && !have_data) {
            if (fseek(fp, 1, SEEK_CUR) != 0) {
                return ESP_ERR_INVALID_SIZE;
            }
        }
    }

    if (info->audio_format != 1 ||
        (info->channels != 1 && info->channels != 2) ||
        (info->bits_per_sample != 8 && info->bits_per_sample != 16) ||
        info->sample_rate == 0 || info->data_bytes == 0) {
        return ESP_ERR_NOT_SUPPORTED;
    }

    return ESP_OK;
}

static size_t convert_pcm_to_stereo16(const uint8_t *input, size_t input_bytes,
                                      const wav_info_t *info)
{
    size_t source_samples = input_bytes / (info->bits_per_sample / 8);
    size_t frames = source_samples / info->channels;
    size_t max_frames = sizeof(s_pcm_buf) / (sizeof(int16_t) * 2);
    if (frames > max_frames) {
        frames = max_frames;
    }

    for (size_t frame = 0; frame < frames; frame++) {
        int16_t left;
        int16_t right;
        if (info->bits_per_sample == 16) {
            const uint8_t *p = input + frame * info->channels * 2;
            left = (int16_t)read_le16(p);
            if (info->channels == 2) {
                right = (int16_t)read_le16(p + 2);
            } else {
                right = left;
            }
        } else {
            const uint8_t *p = input + frame * info->channels;
            left = (int16_t)(((int)p[0] - 128) << 8);
            if (info->channels == 2) {
                right = (int16_t)(((int)p[1] - 128) << 8);
            } else {
                right = left;
            }
        }
        s_pcm_buf[frame * 2] = scale_sample(left);
        s_pcm_buf[frame * 2 + 1] = scale_sample(right);
    }

    return frames * 2 * sizeof(int16_t);
}

static esp_err_t play_wav_file(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return ESP_ERR_NOT_FOUND;
    }

    wav_info_t info;
    esp_err_t err = parse_wav(fp, &info);
    if (err != ESP_OK) {
        fclose(fp);
        return err;
    }

    err = ensure_i2s(info.sample_rate);
    if (err != ESP_OK) {
        fclose(fp);
        return err;
    }

    ESP_LOGI(TAG, "playing WAV %s: %lu Hz, %u ch, %u bit, %lu bytes",
             path, (unsigned long)info.sample_rate, info.channels,
             info.bits_per_sample, (unsigned long)info.data_bytes);

    uint32_t remaining = info.data_bytes;
    const uint32_t block_align = (info.bits_per_sample / 8) * info.channels;
    while (remaining > 0) {
        size_t to_read = remaining < sizeof(s_raw_buf) ? remaining : sizeof(s_raw_buf);
        to_read -= to_read % block_align;
        if (to_read == 0) {
            break;
        }

        size_t got = fread(s_raw_buf, 1, to_read, fp);
        if (got == 0) {
            break;
        }
        remaining -= got;

        size_t out_bytes = convert_pcm_to_stereo16(s_raw_buf, got, &info);
        err = write_i2s_all(s_pcm_buf, out_bytes);
        if (err != ESP_OK) {
            fclose(fp);
            return err;
        }
    }

    fclose(fp);
    return write_silence(info.sample_rate, 80);
}

esp_err_t cardputer_adv_audio_play_wav_from_sd(const char *path)
{
    if (path == NULL || path[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t err = audio_lock(pdMS_TO_TICKS(ADV_IO_TIMEOUT_MS));
    if (err != ESP_OK) {
        return err;
    }

    err = init_codec();
    if (err == ESP_OK) {
        err = mount_sd_once();
    }
    if (err == ESP_OK) {
        err = play_wav_file(path);
    }

    audio_unlock();
    return err;
}

void cardputer_adv_audio_key_event(uint8_t key_code, bool pressed)
{
#if defined(CONFIG_CARDPUTER_ADV_TRAUTONIUM_ENABLE)
    if (key_code == 0) {
        return;
    }
    start_trautonium_synth_once();

    if (trautonium_handle_macro_key(key_code, pressed)) {
        return;
    }

    const trautonium_key_note_t *note = trautonium_find_key_note(key_code);
    int8_t semitone = note != NULL ? note->semitone : trautonium_raw_fallback_semitone(key_code);
    uint8_t formant_index = note != NULL ? note->formant_index :
                            (uint8_t)(((key_code - 1) / 10) %
                                      (sizeof(s_formants) / sizeof(s_formants[0])));

    uint32_t target_mhz = trautonium_note_freq_mhz(semitone);
    uint8_t pressure = trautonium_key_pressure(key_code);
    uint8_t big_knob = trautonium_big_knob_for_key(key_code, s_synth_big_knob_percent);
    bool gate_changed = false;

    portENTER_CRITICAL(&s_synth_lock);
    if (pressed) {
        s_synth_active_key = key_code;
        s_synth_target_mhz = target_mhz;
        s_synth_formant_index = formant_index;
        s_synth_pressure_percent = pressure;
        s_synth_big_knob_percent = big_knob;
        s_synth_sh_trigger++;
        s_synth_gate = true;
        gate_changed = true;
    } else if (s_synth_active_key == key_code) {
        s_synth_active_key = 0;
        s_synth_gate = false;
        gate_changed = true;
    }
    portEXIT_CRITICAL(&s_synth_lock);

    if (gate_changed && pressed) {
        const trautonium_formant_t *formant = &s_formants[formant_index];
        ESP_LOGI(TAG, "Trautonium key=%s raw=%u freq=%lu.%03luHz formant=%s pressure=%u knob=%u",
                 note != NULL ? note->label : "raw",
                 (unsigned)key_code,
                 (unsigned long)(target_mhz / 1000U),
                 (unsigned long)(target_mhz % 1000U),
                 formant->name,
                 (unsigned)pressure,
                 (unsigned)big_knob);
    }
#else
    (void)key_code;
    (void)pressed;
#endif
}

esp_err_t cardputer_adv_audio_startup_probe(void)
{
    esp_err_t audio_error = ESP_OK;

#if defined(CONFIG_CARDPUTER_ADV_AUDIO_BOOT_CHIME)
    esp_err_t err = play_boot_chime();
    if (err != ESP_OK) {
        audio_error = err;
        ESP_LOGW(TAG, "ADV speaker chime failed: %s", esp_err_to_name(err));
    }
#endif

#if defined(CONFIG_CARDPUTER_ADV_TRAUTONIUM_ENABLE)
    start_keyboard_scan_once();
#endif

#if defined(CONFIG_CARDPUTER_ADV_AUDIO_TRY_SD_WAV)
    static const char *fallbacks[] = {
        CONFIG_CARDPUTER_ADV_AUDIO_SD_WAV_PATH,
        CONFIG_CARDPUTER_ADV_AUDIO_SD_MOUNT_POINT "/ruview.wav",
        CONFIG_CARDPUTER_ADV_AUDIO_SD_MOUNT_POINT "/file1.wav",
        CONFIG_CARDPUTER_ADV_AUDIO_SD_MOUNT_POINT "/file2.wav",
        CONFIG_CARDPUTER_ADV_AUDIO_SD_MOUNT_POINT "/file3.wav",
    };

    for (size_t i = 0; i < sizeof(fallbacks) / sizeof(fallbacks[0]); i++) {
        if (fallbacks[i][0] == '\0') {
            continue;
        }
        esp_err_t err = cardputer_adv_audio_play_wav_from_sd(fallbacks[i]);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "ADV SD WAV proof played: %s", fallbacks[i]);
            return ESP_OK;
        }
        ESP_LOGI(TAG, "ADV SD WAV skip %s: %s", fallbacks[i], esp_err_to_name(err));
    }
#endif

    return audio_error;
}

#else

esp_err_t cardputer_adv_audio_play_wav_from_sd(const char *path)
{
    (void)path;
    return ESP_OK;
}

void cardputer_adv_audio_key_event(uint8_t key_code, bool pressed)
{
    (void)key_code;
    (void)pressed;
}

esp_err_t cardputer_adv_audio_startup_probe(void)
{
    return ESP_OK;
}

#endif

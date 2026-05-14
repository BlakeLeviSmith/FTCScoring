/**********************************************************************
  FTC DECODE Vision System — ESP32-S3 Camera Firmware

  Based on Freenove ESP32-S3 WROOM Board example (Sketch_07.1).
  Uses Freenove's stock app_httpd.cpp, camera_pins.h, camera_index.h,
  and board_config.h — NO modifications to those files.

  Changes from the stock Freenove example:
    1. WiFi AP mode (no router needed — laptop connects directly)
    2. OV5640 optimizations (xclk 20MHz, double-buffer, GRAB_LATEST)
    3. Configurable default framesize via INITIAL_FRAMESIZE define
    4. Sensor info printed on boot (confirms OV5640 vs OV2640)

  Upload settings in Arduino IDE:
    Board:            ESP32S3 Dev Module
    Flash Size:       16MB (128Mb)
    Partition Scheme:  Huge APP (3MB No OTA / 1MB SPIFFS)
    PSRAM:            OPI PSRAM
    USB CDC On Boot:  Enabled
    Upload Speed:     921600
**********************************************************************/

#include "esp_camera.h"
#include <WiFi.h>
#include "board_config.h"

// ==================== WiFi AP Config ====================
// Your laptop connects to this network directly. No router.
const char* AP_SSID     = "FTC-DECODE-CAM";
const char* AP_PASSWORD = "ftcdecode";   // min 8 chars for WPA2
const int   WIFI_CHANNEL = 11;           // 1, 6, or 11 — pick least congested

// ==================== Camera Defaults ====================
// Change INITIAL_FRAMESIZE to start at a different resolution.
// Freenove ESP32-S3 enum (with PSRAM):
//   FRAMESIZE_VGA    =  8  (640x480)
//   FRAMESIZE_SVGA   =  9  (800x600)
//   FRAMESIZE_XGA    = 10  (1024x768)
//   FRAMESIZE_HD     = 11  (1280x720)
//   FRAMESIZE_SXGA   = 12  (1280x1024)
//   FRAMESIZE_UXGA   = 13  (1600x1200)
//
// Our Python app sends /control?var=framesize&val=N to change this at
// runtime, so whatever you pick here is just the power-on default.
#define INITIAL_FRAMESIZE  FRAMESIZE_SVGA
#define INITIAL_QUALITY    15   // 8=best quality, 63=worst (smaller frames)

// ==================== Freenove's server entry point ====================
void startCameraServer();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n\n=== FTC DECODE Camera ===");

  // ---- Camera config ----
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.pixel_format = PIXFORMAT_JPEG;
  config.fb_location  = CAMERA_FB_IN_PSRAM;

  // OV5640 needs a faster clock than OV2640. 10MHz leaves the JPEG
  // encoder broken (stuck at CIF). 24MHz is the most widely reported
  // stable value for OV5640 modules on ESP32-S3.
  config.xclk_freq_hz = 24000000;

  config.frame_size   = INITIAL_FRAMESIZE;
  config.jpeg_quality = INITIAL_QUALITY;

  if (psramFound()) {
    Serial.println("PSRAM found — using double-buffer + GRAB_LATEST");
    config.fb_count   = 2;
    config.grab_mode  = CAMERA_GRAB_LATEST;
  } else {
    Serial.println("No PSRAM — single buffer, reduced quality");
    config.fb_count   = 1;
    config.grab_mode  = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.frame_size  = FRAMESIZE_VGA;  // DRAM can't hold larger
  }

  // ---- Init camera ----
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    Serial.println("Check: is the ribbon cable seated correctly?");
    while (1) delay(1000);  // halt
  }

  // ---- Sensor-specific tweaks ----
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    Serial.printf("Sensor PID: 0x%04X ", s->id.PID);
    if (s->id.PID == 0x56 || s->id.PID == 0x5640) {
      Serial.println("(OV5640 detected)");
      s->set_brightness(s, 1);
      s->set_saturation(s, 1);
    } else if (s->id.PID == 0x26) {
      Serial.println("(OV2640 detected)");
      s->set_vflip(s, 0);
      s->set_brightness(s, 1);
      s->set_saturation(s, 0);
    } else {
      Serial.println("(unknown sensor)");
    }
  }

  // ---- WiFi AP mode ----
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASSWORD, WIFI_CHANNEL);
  delay(500);  // let the AP stabilize

  IPAddress ip = WiFi.softAPIP();
  Serial.printf("WiFi AP: \"%s\" (channel %d)\n", AP_SSID, WIFI_CHANNEL);
  Serial.printf("IP: %s\n", ip.toString().c_str());

  // ---- Start Freenove's HTTP + stream server ----
  startCameraServer();

  Serial.println("-----------------------------");
  Serial.printf("Stream:  http://%s:81/stream\n", ip.toString().c_str());
  Serial.printf("Control: http://%s/control\n", ip.toString().c_str());
  Serial.printf("Status:  http://%s/status\n", ip.toString().c_str());
  Serial.printf("Capture: http://%s/capture\n", ip.toString().c_str());
  Serial.println("-----------------------------");
  Serial.println("Ready!");
}

void loop() {
  delay(10000);
}

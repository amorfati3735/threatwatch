/*
 * ThreatWatch wearable — Arduino Uno R3
 * ============================================================================
 * Sensors : MAX30102 (PPG -> heart rate, finger detect)
 *           MPU6050  (3-axis accelerometer + 3-axis gyroscope)
 *           SSD1306 128x64 OLED (local HUD, mirrors the laptop's verdict)
 * Bus     : shared I2C  (A4 = SDA, A5 = SCL)
 *
 * This sketch is the merge of the two predecessor firmwares:
 *   - the compile-verified CSV protocol and honest "no fabricated SpO2" policy
 *     from the threat-detection project, and
 *   - the richer OLED safety HUD + I2C robustness from the esa project.
 * The esa project's `110.0 - 15.0 * ratio` SpO2 estimate was REMOVED: it is an
 * uncalibrated guess presented as a physiological measurement.
 *
 * ---------------------------------------------------------------------------
 * UPSTREAM (device -> laptop), one CSV line per sample at ~15 Hz:
 *
 *   ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label
 *
 *   ts      seconds since boot (float, 3 decimals)
 *   hr      BPM, or EMPTY when there is no finger / not enough beats
 *   spo2    EMPTY  — see the honesty note below
 *   eda     EMPTY  — this board has no GSR/EDA sensor
 *   temp    MPU6050 die temperature (NOT skin temperature)
 *   ax..gz  accelerometer (g, +/-2g) and gyroscope (deg/s, +/-250 dps)
 *   finger  1 / 0
 *   label   -1 (unknown on a live board)
 *
 * DOWNSTREAM (laptop -> device), two short lines:
 *
 *   STATE:<name>        e.g. "STATE:SUSPICIOUS / CAUTION"
 *   SCORE:<0-100>       e.g. "SCORE:73"
 *
 * Two tiny lines keep the ATmega328P parser small and preserve the original
 * "STATE:" contract while adding the threat score the OLED needs.
 *
 * ---------------------------------------------------------------------------
 * HONESTY NOTE ON SpO2
 * A trustworthy SpO2 requires a calibrated red/IR ratio curve for the specific
 * optics and finger. The MAX30102 library does not provide one, so any number
 * derived here would be fabricated. It is therefore left EMPTY, and the OLED
 * shows "O2: --". Add a proper calibration if the write-up requires SpO2.
 *
 * Libraries (Arduino IDE -> Manage Libraries):
 *   - "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *   - "Adafruit SSD1306" (+ "Adafruit GFX Library")
 * MPU6050 is driven over Wire() directly (no extra library needed).
 * ============================================================================
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"          // SparkFun MAX3010x (works with MAX30102)
#include "heartRate.h"         // checkForBeat() lives here (free function!)

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
#define OLED_RESET     -1
#define OLED_ADDR     0x3C
#define MPU_ADDR      0x68

#define SAMPLE_PERIOD_MS 66    // ~15 Hz telemetry
#define OLED_PERIOD_MS  200    // ~5 Hz display refresh
#define FINGER_IR_MIN   50000L // IR threshold for "finger present"

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
MAX30105 particleSensor;

// ---- MPU6050 minimal driver ------------------------------------------------
const float ACCEL_SCALE = 16384.0f;   // +/-2 g
const float GYRO_SCALE  = 131.0f;     // +/-250 deg/s

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;
  mpuWrite(0x6B, 0x00);   // PWR_MGMT_1  : wake
  delay(50);
  mpuWrite(0x1C, 0x00);   // ACCEL_CONFIG: +/-2 g
  mpuWrite(0x1B, 0x00);   // GYRO_CONFIG : +/-250 dps
  mpuWrite(0x1A, 0x03);   // DLPF        : ~44 Hz accel / 42 Hz gyro
  return true;
}

void mpuRead(float &ax, float &ay, float &az,
             float &gx, float &gy, float &gz, float &tempC) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  int16_t raw[7];
  for (int i = 0; i < 7 && Wire.available() >= 2; i++) {
    raw[i] = (Wire.read() << 8) | Wire.read();
  }
  ax    = raw[0] / ACCEL_SCALE;
  ay    = raw[1] / ACCEL_SCALE;
  az    = raw[2] / ACCEL_SCALE;
  tempC = raw[3] / 340.0f + 36.53f;   // die temperature, not skin
  gx    = raw[4] / GYRO_SCALE;
  gy    = raw[5] / GYRO_SCALE;
  gz    = raw[6] / GYRO_SCALE;
}

// ---- MAX30102 heart-rate estimate ------------------------------------------
long  lastBeatMs = 0;
const int MIN_BEAT_MS = 300;    // <= 200 BPM
const int MAX_BEAT_MS = 2000;   // >=  30 BPM
float bpm = 0.0f;
bool  haveBpm = false;

void updateHeartRate(long ir) {
  if (ir < FINGER_IR_MIN) return;
  if (checkForBeat(ir)) {
    long now = millis();
    long dt  = now - lastBeatMs;
    if (lastBeatMs != 0 && dt > MIN_BEAT_MS && dt < MAX_BEAT_MS) {
      float instant = 60000.0f / (float)dt;
      bpm = haveBpm ? (0.7f * bpm + 0.3f * instant) : instant;  // exponential smoothing
      haveBpm = true;
    }
    lastBeatMs = now;
  }
}

// ---- Downstream feedback from the laptop -----------------------------------
String laptopState = "--";
int    laptopScore = 0;      // 0..100 fused threat score
bool   laptopLinked = false;
unsigned long lastFeedbackMs = 0;

void parseDownstream() {
  static String buf = "";
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf.trim();
      if (buf.startsWith("STATE:")) {
        laptopState  = buf.substring(6);
        laptopLinked = true;
        lastFeedbackMs = millis();
      } else if (buf.startsWith("SCORE:")) {
        laptopScore  = constrain(buf.substring(6).toInt(), 0, 100);
        laptopLinked = true;
        lastFeedbackMs = millis();
      }
      buf = "";
    } else if (buf.length() < 40) {
      buf += c;
    }
  }
  // Consider the link stale if the laptop has not spoken for 10 s.
  if (laptopLinked && (millis() - lastFeedbackMs) > 10000UL) laptopLinked = false;
}

// ---- OLED ------------------------------------------------------------------
const char* verdictText() {
  if (!laptopLinked)   return "[ NO LINK ]";
  if (laptopScore >= 70) return "[ THREAT! ]";
  if (laptopScore >= 40) return "[ CAUTION ]";
  return "[ SAFE ]";
}

void drawHud(bool finger, float amag, float gmag) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Top status bar
  display.setCursor(0, 0);
  display.print(F("SAFETY: "));
  display.print(verdictText());
  display.drawLine(0, 11, SCREEN_WIDTH, 11, SSD1306_WHITE);

  // Heart rate
  display.setCursor(0, 15);
  display.print(F("HR: "));
  if (finger && haveBpm) { display.print(bpm, 0); display.print(F(" BPM")); }
  else                   { display.print(F("-- BPM")); }

  // SpO2 deliberately unavailable (uncalibrated) — shown as unknown, not faked.
  display.setCursor(80, 15);
  display.print(F("O2: --"));

  // Motion
  display.setCursor(0, 28);
  display.print(F("MOT: ")); display.print(amag, 2); display.print(F(" g"));
  display.setCursor(80, 28);
  display.print(F("R:"));    display.print(gmag, 0);

  // Threat score
  display.setCursor(0, 41);
  display.print(F("THREAT: ")); display.print(laptopScore); display.print(F("%"));

  // Progress bar
  display.drawRect(0, 54, SCREEN_WIDTH, 9, SSD1306_WHITE);
  int barWidth = map(constrain(laptopScore, 0, 100), 0, 100, 1, SCREEN_WIDTH - 2);
  if (laptopScore > 0) display.fillRect(1, 55, barWidth, 7, SSD1306_WHITE);

  display.display();
}

// ---- setup / loop ----------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin();
  // Survive a hung I2C slave instead of freezing the whole board.
  Wire.setWireTimeout(25000, true);
  Wire.setClock(400000);   // 400 kHz fast mode

  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(6, 16);
    display.println(F("THREATWATCH"));
    display.setCursor(6, 30);
    display.println(F("WEARABLE - UNO"));
    display.setCursor(6, 46);
    display.println(F("Init sensors..."));
    display.display();
  } else {
    Serial.println(F("# OLED not found"));
  }

  bool mpuOk = mpuInit();
  Serial.println(mpuOk ? F("# MPU6050 ready") : F("# MPU6050 FAILED"));

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println(F("# MAX30102 FAILED"));
  } else {
    // ledBrightness, sampleAverage, ledMode(2=Red+IR), sampleRate, pulseWidth, adcRange
    particleSensor.setup(0x1F, 4, 2, 100, 411, 4096);
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeIR(0x1F);
    Serial.println(F("# MAX30102 ready"));
  }

  Serial.println(F("ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label"));
  delay(500);
}

unsigned long lastSample = 0;
unsigned long lastDraw   = 0;

void loop() {
  parseDownstream();

  unsigned long now = millis();
  if (now - lastSample < SAMPLE_PERIOD_MS) return;
  lastSample = now;

  long ir = particleSensor.getIR();
  bool finger = (ir >= FINGER_IR_MIN);
  updateHeartRate(ir);

  float ax, ay, az, gx, gy, gz, tempC;
  mpuRead(ax, ay, az, gx, gy, gz, tempC);
  float amag = sqrt(ax * ax + ay * ay + az * az);
  float gmag = sqrt(gx * gx + gy * gy + gz * gz);

  // ---- upstream CSV; empty field = "not measured", never a fabricated value --
  Serial.print(now / 1000.0, 3);            Serial.print(',');
  if (finger && haveBpm) Serial.print(bpm, 1);
  Serial.print(',');                        // spo2 : deliberately empty
  Serial.print(',');                        // eda  : no GSR sensor
  Serial.print(tempC, 2);                   Serial.print(',');
  Serial.print(ax, 3);  Serial.print(',');
  Serial.print(ay, 3);  Serial.print(',');
  Serial.print(az, 3);  Serial.print(',');
  Serial.print(gx, 2);  Serial.print(',');
  Serial.print(gy, 2);  Serial.print(',');
  Serial.print(gz, 2);  Serial.print(',');
  Serial.print(finger ? 1 : 0);             Serial.print(',');
  Serial.println(-1);

  if (now - lastDraw > OLED_PERIOD_MS) {
    lastDraw = now;
    drawHud(finger, amag, gmag);
  }
}

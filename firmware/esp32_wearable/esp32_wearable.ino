/*
 * Fitness Tracker wearable — ESP32 (primary controller, wireless)
 * Same sensor stack as the Uno build: MAX30102 + MPU6050 (+ optional SSD1306).
 *
 * Streams the SAME CSV line as the Uno firmware, over BOTH:
 *   1. USB serial  @ 115200  (debug / wired laptop link)
 *   2. BLE "Nordic UART Service" (NUS) notifications  (wireless laptop link)
 *
 *   ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label
 *   (ts is seconds since boot, float, so it matches the WESAD/replay timeline)
 *
 * Only a ESP32 devkit (e.g. ESP32-WROOM-32) and the same I2C sensors are
 * needed.  Default I2C pins below are ESP32 GPIO21 (SDA) / GPIO22 (SCL).
 *
 * Libraries:
 *   - ESP32 core (BLE comes built in via the "ESP32 BLE Arduino" library)
 *   - "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *   - Adafruit SSD1306 + Adafruit GFX  (optional OLED)
 *
 * The BLE receiver on the laptop is:
 *     python scripts/live.py --source ble
 * (see threatwatch/data/sources/ble_source.py) and the same source is used by
 * the dashboard server, which subscribes to the NUS TX characteristic and
 * parses the identical CSV line as the wired Uno.
 *
 * Downstream feedback (laptop -> device) uses the merged two-line protocol:
 *   STATE:<name>   and   SCORE:<0-100>
 * Both are accepted over the NUS RX write characteristic and over USB serial.
 *
 * HONESTY: SpO2 and EDA fields are emitted EMPTY (no calibrated SpO2 curve, no
 * GSR sensor). They are never replaced by a plausible-looking guess.
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// Nordic UART Service UUIDs (must match the laptop BLE client)
#define NUS_SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX_UUID      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // ESP32 -> laptop (notify)
#define NUS_RX_UUID      "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // laptop -> ESP32 (write)
#define DEVICE_NAME      "ThreatWatch"

#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR 0x68

MAX30105 particleSensor;

// ---- OLED live UI (SSD1306 128x64 @ 0x3C, same bus) ----
#define OLED_ADDR 0x3C
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
bool oledOn = false;
uint32_t lastOledMs = 0, lastBeatFlashMs = 0;
uint32_t irHist[128];
uint8_t irHistN = 0;
float irBase = 0;
const char spinner[4] = {'|', '/', '-', '\\'};

static const uint8_t HEART_BMP[] = {
  0b01100110,
  0b11111111,
  0b11111111,
  0b01111110,
  0b00111100,
  0b00011000,
};

BLEServer *bleServer = nullptr;
BLECharacteristic *txChar = nullptr;
bool bleConnected = false;

const float ACCEL_SCALE = 16384.0f;
const float GYRO_SCALE  = 131.0f;

String laptopState = "--";
int    laptopScore = 0;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *s) override { bleConnected = true; }
  void onDisconnect(BLEServer *s) override {
    bleConnected = false;
    s->startAdvertising();  // keep advertising so the laptop can reconnect
  }
};

class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *c) override {
    String v = String(c->getValue().c_str());
    v.trim();
    // The laptop sends two short lines; a BLE write may carry either or both.
    int stateIdx = v.indexOf("STATE:");
    if (stateIdx >= 0) {
      String s = v.substring(stateIdx + 6);
      int nl = s.indexOf('\n');
      if (nl >= 0) s = s.substring(0, nl);
      s.trim();
      if (s.length()) laptopState = s;
    }
    int scoreIdx = v.indexOf("SCORE:");
    if (scoreIdx >= 0) {
      String s = v.substring(scoreIdx + 6);
      int nl = s.indexOf('\n');
      if (nl >= 0) s = s.substring(0, nl);
      laptopScore = constrain(s.toInt(), 0, 100);
    }
    Serial.print("# from laptop: state="); Serial.print(laptopState);
    Serial.print(" score="); Serial.println(laptopScore);
  }
};

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(reg); Wire.write(val); Wire.endTransmission();
}

bool mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;
  mpuWrite(0x6B, 0x00); delay(50);
  mpuWrite(0x1C, 0x00);
  mpuWrite(0x1B, 0x00);
  mpuWrite(0x1A, 0x03);
  return true;
}

void mpuRead(float &ax, float &ay, float &az, float &gx, float &gy, float &gz, float &tempC) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B); Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14);
  int16_t raw[7];
  for (int i = 0; i < 7 && Wire.available() >= 2; i++)
    raw[i] = (Wire.read() << 8) | Wire.read();
  ax = raw[0] / ACCEL_SCALE; ay = raw[1] / ACCEL_SCALE; az = raw[2] / ACCEL_SCALE;
  tempC = raw[3] / 340.0f + 36.53f;
  gx = raw[4] / GYRO_SCALE; gy = raw[5] / GYRO_SCALE; gz = raw[6] / GYRO_SCALE;
}

long lastBeatMs = 0; float bpm = 0; bool haveBpm = false;
unsigned long lastFingerMs = 0, beatCount = 0, lastDebugMs = 0;
uint32_t lastIr = 0;

// Raw IR above this means something reflective (a fingertip) is on the sensor.
// Same threshold the SparkFun examples use.
static const uint32_t IR_FINGER_THRESHOLD = 50000;

// Feed the beat detector EVERY sample the MAX30102 produces (~25 Hz after the
// 4-sample average).  The PBA algorithm (heartRate.h) is a DC estimator plus a
// low-pass FIR that assumes a steady sample stream; the loop used to hand it one
// sample per CSV record (~8 Hz, the OLED refresh blocks the loop), which
// decimated the pulse waveform so far that a beat was never detected and the
// panel sat on "-- BPM" with a finger on the sensor.
void updateHeartRate(uint32_t ir) {
  lastIr = ir;
  if (ir < IR_FINGER_THRESHOLD) return;
  lastFingerMs = millis();
  if (checkForBeat((int32_t)ir)) {
    lastBeatFlashMs = millis();   // flash the on-screen heart
    beatCount++;
    long now = millis(); long dt = now - lastBeatMs;
    if (lastBeatMs != 0 && dt > 300 && dt < 2000) {
      float inst = 60000.0f / (float)dt;
      bpm = haveBpm ? (0.7f * bpm + 0.3f * inst) : inst;
      haveBpm = true;
    }
    lastBeatMs = now;
  }
}

bool oledInit() {
  // Real electrical ACK check first (Adafruit begin() alone always returns true)
  Wire.beginTransmission(OLED_ADDR);
  if (Wire.endTransmission() != 0) return false;
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) return false;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  return true;
}

// Staged boot animation: logo wipe -> self-test sweep -> READY flash.
// Called from setup() AFTER all sensors init, so oledOk/mpuOk/maxOk are known.
void oledBootAnim(bool oledOk, bool mpuOk, bool maxOk) {
  for (int w = 0; w <= 128; w += 8) {      // 1. logo wipe
    oled.clearDisplay();
    oled.setTextSize(2);
    oled.setCursor(14, 6);
    oled.println("THREAT");
    oled.setCursor(26, 28);
    oled.println("WATCH");
    oled.drawLine(0, 48, w, 48, SSD1306_WHITE);   // progress line
    oled.drawLine(0, 49, w, 49, SSD1306_WHITE);
    oled.display();
    delay(35);
  }
  oled.clearDisplay();                     // 2. self-test sweep
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.println("THREATWATCH BOOT");
  delay(250);
  oled.setCursor(0, 16);
  oled.print(oledOk ? "> OLED ...... OK" : "> OLED ... FAIL");
  oled.display();
  delay(300);
  oled.setCursor(0, 28);
  oled.print(mpuOk ? "> IMU ....... OK" : "> IMU .... FAIL");
  oled.display();
  delay(300);
  oled.setCursor(0, 40);
  oled.print(maxOk ? "> PPG ....... OK" : "> PPG .... FAIL");
  oled.display();
  delay(450);
  for (int i = 0; i < 3; i++) {            // 3. READY flash
    oled.setCursor(34, 54);
    oled.print("** READY **");
    oled.display();
    delay(220);
    oled.fillRect(0, 54, 128, 8, SSD1306_BLACK);
    oled.display();
    delay(160);
  }
  oled.clearDisplay();
  oled.setCursor(0, 0);
  oled.println("ESP32 SMOKE TEST");
  oled.println();
  oled.println((oledOk && mpuOk && maxOk) ? "all sensors OK" : "CHECK FAIL FLAGS");
  oled.display();
}

void oledSplash() {
  oled.clearDisplay();
  oled.setTextSize(2);
  oled.setCursor(14, 6);
  oled.println("THREAT");
  oled.setCursor(26, 28);
  oled.println("WATCH");
  oled.setTextSize(1);
  oled.setCursor(28, 50);
  oled.print("ESP32 WEARABLE");
  oled.display();
  delay(1400);
}

// 128x64 text layout: header / BPM / ACC axes / contact / verdict.
// The MPU6050 temperature is deliberately NOT shown: it is the IMU's own die
// temperature (~43-44 C at idle), not skin temperature, so it reads the same
// with or without a finger and only invites the wrong question.
void oledUpdate(unsigned long now, bool finger, uint32_t ir, float ax, float ay, float az) {
  if (irBase == 0) irBase = (float)ir;
  irBase += ((float)ir - irBase) * 0.02f;

  oled.clearDisplay();

  // header: title + link + activity spinner
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.print("THREATWATCH");
  oled.drawLine(0, 9, 127, 9, SSD1306_WHITE);
  oled.setCursor(84, 0);
  oled.print(bleConnected ? "BLE" : "USB");
  oled.setCursor(116, 0);
  oled.print(spinner[(now / 200) % 4]);

  // big heart-rate line: the headline number
  char hrTxt[8];
  if (finger && haveBpm) snprintf(hrTxt, sizeof(hrTxt), "%d", (int)(bpm + 0.5f));
  else                   strcpy(hrTxt, "--");
  oled.setTextSize(2);
  oled.setCursor(0, 11);
  oled.print(hrTxt);
  oled.print(" BPM");

  // accelerometer: the one sensor that always has something to say
  oled.setTextSize(1);
  oled.setCursor(0, 30);
  char accTxt[24];
  snprintf(accTxt, sizeof(accTxt), "ACC %.2f %.2f %.2f", ax, ay, az);
  oled.print(accTxt);

  // contact quality: the raw IR level is what the wearer adjusts, and it is
  // what explains a missing BPM
  oled.setCursor(0, 41);
  char irTxt[24];
  snprintf(irTxt, sizeof(irTxt), "IR %.0fk  %s", ir / 1000.0f, finger ? "FINGER OK" : "NO FINGER");
  oled.print(irTxt);

  // verdict: fused state + score from the laptop, on one line
  oled.setCursor(0, 52);
  if (laptopState != "--") {
    oled.print("STATE: ");
    oled.print(laptopState.substring(0, 8));
    oled.print(" ");
    oled.print(laptopScore);
    oled.print("%");
  } else if (!finger) {
    oled.print("PLACE FINGER ON SENSOR");
  } else {
    oled.print("WAITING FOR LAPTOP");
  }
  oled.display();
}

void setupBle() {
  BLEDevice::init(DEVICE_NAME);
  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new ServerCallbacks());
  BLEService *svc = bleServer->createService(NUS_SERVICE_UUID);

  txChar = svc->createCharacteristic(NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  txChar->addDescriptor(new BLE2902());

  BLECharacteristic *rx = svc->createCharacteristic(NUS_RX_UUID, BLECharacteristic::PROPERTY_WRITE);
  rx->setCallbacks(new RxCallbacks());

  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(NUS_SERVICE_UUID);
  adv->setScanResponse(true);
  adv->start();
  Serial.println("# BLE advertising as " DEVICE_NAME);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);

  bool mpuOk = mpuInit();
  if (mpuOk) Serial.println("# MPU6050 ready");
  else       Serial.println("# MPU6050 FAILED");

  bool maxOk = particleSensor.begin(Wire, I2C_SPEED_FAST);
  if (!maxOk) {
    Serial.println("# MAX30102 FAILED");
  } else {
    particleSensor.setup(0x1F, 4, 2, 100, 411, 4096);
    particleSensor.setPulseAmplitudeIR(0x1F);
    Serial.println("# MAX30102 ready");
  }

  oledOn = oledInit();
  if (oledOn) {
    Serial.println("# OLED ready");
    oledBootAnim(true, mpuOk, maxOk);   // staged intro with real self-test verdicts
  } else {
    Serial.println("# OLED not found");
  }

  setupBle();
  Serial.println("ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label");
}

const unsigned long SAMPLE_PERIOD_MS = 50;  // ~20 Hz
unsigned long lastSample = 0;

String buildLine(unsigned long now, bool finger, float ax, float ay, float az,
                 float gx, float gy, float gz, float tempC) {
  char buf[160];
  char hr[12] = "";
  if (finger && haveBpm) snprintf(hr, sizeof(hr), "%.1f", bpm);
  // ts,hr,spo2(empty),eda(empty),temp,ax,ay,az,gx,gy,gz,finger,label
  snprintf(buf, sizeof(buf),
           "%.3f,%s,,,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%d,-1",
           now / 1000.0, hr, tempC, ax, ay, az, gx, gy, gz, finger ? 1 : 0);
  return String(buf);
}

void loop() {
  unsigned long now = millis();

  // 1. Drain the PPG FIFO at the sensor's own pace and feed every sample to the
  //    beat detector, so the CSV record rate (~10-20 Hz) no longer decimates
  //    the pulse waveform.
  particleSensor.check();
  while (particleSensor.available() > 0) {
    particleSensor.nextSample();
    updateHeartRate(particleSensor.getFIFOIR());
  }
  bool finger = lastIr >= IR_FINGER_THRESHOLD;

  if (now - lastSample < SAMPLE_PERIOD_MS) return;
  lastSample = now;

  float ax, ay, az, gx, gy, gz, tempC;
  mpuRead(ax, ay, az, gx, gy, gz, tempC);
  String line = buildLine(now, finger, ax, ay, az, gx, gy, gz, tempC);

  Serial.println(line);
  if (bleConnected && txChar) {
    txChar->setValue((uint8_t *)line.c_str(), line.length());
    txChar->notify();
  }

  // 2. One comment line every 2 s with the raw physiology behind the numbers.
  //    Leading '#' means every consumer (dashboard, log replay, this file's
  //    parser) ignores it, so the CSV contract is untouched.
  if (now - lastDebugMs >= 2000) {
    lastDebugMs = now;
    char dbg[80];
    snprintf(dbg, sizeof(dbg), "# ir=%lu bpm=%.1f beats=%lu finger=%d state=%s score=%d",
             (unsigned long)lastIr, bpm, beatCount, finger ? 1 : 0, laptopState.c_str(), laptopScore);
    Serial.println(dbg);
    if (bleConnected && txChar) {
      txChar->setValue((uint8_t *)dbg, strlen(dbg));
      txChar->notify();
    }
  }

  if (oledOn && now - lastOledMs >= 100) {   // 10 Hz screen refresh
    lastOledMs = now;
    oledUpdate(now, finger, lastIr, ax, ay, az);
  }
}

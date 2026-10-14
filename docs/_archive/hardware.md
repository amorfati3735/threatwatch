# Hardware guide

## 1. Bill of materials

| Component | Role | Bus | Addr | Notes |
|---|---|---|---|---|
| Arduino Uno R3 | sensor controller (dev/backup) | USB-serial | — | 2 KB SRAM, no radio |
| ESP32-WROOM-32 | primary wearable controller | USB / BLE / Wi-Fi | — | dual-mode Bluetooth + BLE |
| MAX30102 breakout | PPG → heart rate, finger detect | I²C | `0x57` | 3.3 V logic |
| MPU6050 (GY-521) | 3-axis accel + 3-axis gyro | I²C | `0x68` | 3.3 V logic (breakout tolerant of 5 V) |
| SSD1306 0.96" OLED | local HUD | I²C | `0x3C` | 128×64 |
| GSR/EDA module *(recommended)* | skin conductance — biggest accuracy gain | analog | — | optional add-on |
| LiPo + regulator *(for portable)* | battery power | — | — | ESP32 only |

> The MAX30102 and MPU6050 are **3.3 V-logic** devices. Breakout boards usually
> include an LDO so their `VCC` accepts 5 V, but the ESP32 (3.3 V) is the safer
> host than the Uno (5 V).

## 2. Shared I²C bus wiring

```
                 ┌─────────────────────────────────────────────┐
                 │              ARDUINO UNO                    │
                 │   5V     GND        A4 (SDA)     A5 (SCL)   │
                 └────┬──────┬───────────┬────────────┬────────┘
                      │      │           │            │
        ┌─────────────┴──────┴───────────┴────────────┴─────────────┐
        │  MAX30102  VIN  GND  SDA  SCL                            │
        │  MPU6050   VCC  GND  SDA  SCL                            │
        │  SSD1306   VCC  GND  SDA  SCL                            │
        └──────────────────────────────────────────────────────────┘

   ESP32 equivalent:  3V3 = 3.3V,  GND = GND,  GPIO21 = SDA,  GPIO22 = SCL
```

| Signal | Uno | ESP32 |
|---|---|---|
| Power | 5 V / 3.3 V | 3V3 |
| Ground | GND | GND |
| I²C SDA | A4 | GPIO21 |
| I²C SCL | A5 | GPIO22 |

Both modules have on-board I²C pull-ups, so no external resistors are needed.

## 3. Flashing the firmware

**Arduino Uno** (`firmware/arduino_uno_wearable/`)
1. Arduino IDE → *Manage Libraries*: install **SparkFun MAX3010x** and
   **Adafruit SSD1306** (+ **Adafruit GFX**).
2. Board: *Arduino Uno*; select the COM port; Upload.
3. Serial Monitor @ 115200 shows the CSV stream and `#` status lines.

**ESP32** (`firmware/esp32_wearable/`)
1. Add the ESP32 board package (*Boards Manager → esp32 by Espressif*).
2. Install the same two libraries (BLE is built into the ESP32 core).
3. Board: *ESP32 Dev Module*; Upload.
4. It advertises as BLE device **`FitnessTracker`** and also streams over USB.

## 4. Connecting from the laptop

```bash
# Arduino Uno over USB
python scripts/live.py --source serial --port COM3 --log logs/session.csv

# ESP32 over BLE  (run this with *Windows* Python if in WSL)
python scripts/live.py --source ble

# re-run a recorded session offline later
python scripts/replay.py --file logs/session.csv --speed 2
```

The OLED mirrors the predicted state if the laptop writes a line
`STATE:<label>\n` back to the board over serial (`bleak` write for the ESP32).

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `hr` field always empty | finger not on sensor / poor contact | rest fingertip flat, avoid pressing hard |
| `HR` stuck at a value | stale beats while finger removed | remove finger; buffer clears |
| MPU6050 `# FAILED` | loose SDA/SCL or addr conflict | verify A4/A5; MPU address `0x68` |
| OLED blank | wrong address | try `0x3C` (most) or `0x3D` |
| Serial `Access denied` (Windows) | Arduino Serial Monitor still open | close the IDE monitor before `live.py` |
| BLE device not found | ESP32 not advertising / out of range | power-cycle board; keep within a few metres |

# Hardware setup — ESP32 wearable

Written for the exact kit on hand:

| # | Part | Role |
|---|---|---|
| 1 | **ESP32 devkit** (ESP32-WROOM-32, 30-pin) | controller, Wi-Fi/BLE, **runs the ML model on-board** |
| 2 | **MAX30102** breakout | PPG → heart rate; finger/skin-contact detection |
| 3 | **MPU6050** breakout | 3-axis accelerometer + 3-axis gyroscope |
| 4 | **SSD1306 128×64 OLED** (I²C) | local HUD — shows the state without a laptop |
| 5 | Breadboard + jumper wires | interconnect |

**There is no Arduino Uno in this build.** Both sketches in `firmware/` speak the
same CSV protocol, but `firmware/esp32_wearable/` is the one that matches the
hardware on the bench. The Uno sketch is kept only as a size comparison (it is
useful evidence that the network genuinely does not fit an ATmega328P — see
§7).

---

## 1. Wiring

All three modules are **3.3 V I²C devices** and share **one bus**. ESP32's default
I²C pins are used:

| Signal | ESP32 pin | Notes |
|---|---|---|
| SDA | **GPIO21** | shared data line |
| SCL | **GPIO22** | shared clock line |
| 3V3 | 3V3 | all three modules |
| GND | GND | common ground |

| Module | Module pin | ESP32 pin | I²C address |
|---|---|---|---|
| MAX30102 | VIN (or VCC) | 3V3 | **0x57** |
| MAX30102 | GND | GND | |
| MAX30102 | SDA | GPIO21 | |
| MAX30102 | SCL | GPIO22 | |
| MPU6050 | VCC | 3V3 | **0x68** |
| MPU6050 | GND | GND | |
| MPU6050 | SDA | GPIO21 | |
| MPU6050 | SCL | GPIO22 | |
| SSD1306 | VCC | 3V3 | **0x3C** |
| SSD1306 | GND | GND | |
| SSD1306 | SDA | GPIO21 | |
| SSD1306 | SCL | GPIO22 | |

```
                 ┌──────────────────────────────┐
   3V3 ──────────┤ 3V3                          │
   GND ──────────┤ GND       E S P 3 2          │
                 │                              │
   ┌─ SDA ───────┤ GPIO21                       │
   │  SCL ───────┤ GPIO22                       │
   │             └──────────────────────────────┘
   │        ┌──────────┬──────────┬──────────┐
   └────────┤ SDA      │ SDA      │ SDA      │
            │ SCL      │ SCL      │ SCL      │
   3V3 ─────┤ VCC      │ VCC      │ VCC      │
   GND ─────┤ GND      │ GND      │ GND      │
            ├──────────┼──────────┼──────────┤
            │ MAX30102 │ MPU6050  │ SSD1306  │
            │  0x57    │  0x68    │  0x3C    │
            └──────────┴──────────┴──────────┘
```

### Practical notes

* Many MAX30102 breakouts have a 1.8 V regulator and level shifters, so the bare
  `VIN` pin wants **3.3 V** — do not feed it 5 V.
* Those breakout boards usually include their own I²C pull-ups. With three
  modules on one bus you can end up with a low effective resistance; if the bus
  is unreliable, remove the pull-ups from **one or two** of the boards, keeping
  pull-ups on exactly one.
* Keep the jumpers short. The ESP32 is fast; long untwisted jumpers at 400 kHz
  are a common source of random device dropouts.
* The ESP32 3V3 rail easily supplies all three modules (a few tens of mA).

---

## 2. Firmware: libraries

Install through **Arduino IDE → Tools → Manage Libraries**:

| Library | Why |
|---|---|
| *SparkFun MAX3010x Pulse and Proximity Sensor Library* | MAX30102 driver + `checkForBeat` |
| *Adafruit SSD1306* | OLED |
| *Adafruit GFX Library* | required by SSD1306 |
| *Adafruit BusIO* | required by the Adafruit libraries |

The MPU6050 is driven directly over `Wire`, so **no MPU6050 library is needed** —
that is deliberate, it keeps the dependency list short and the register writes
explicit and auditable.

The BLE stack ("ESP32 BLE Arduino") ships with the ESP32 core.

---

## 3. Board support and build

The `esp32:esp32` core is installed in the Arduino IDE's package directory:

```bash
CLI="/mnt/c/Users/DELL/AppData/Local/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe"

"$CLI" core list                       # expect: esp32:esp32  3.x installed
"$CLI" compile --fqbn esp32:esp32:esp32 \
    --libraries "C:\\Users\\DELL\\Documents\\Arduino\\libraries" \
    firmware/esp32_wearable
```

To flash from WSL (board on a COM port):

```bash
"$CLI" upload -p COM5 --fqbn esp32:esp32:esp32 \
    --libraries "C:\\Users\\DELL\\Documents\\Arduino\\libraries" \
    firmware/esp32_wearable
```

> On the ESP32 there is **no** `/validate` flag — that is an AutoHotkey thing, not
> an arduino-cli thing. `compile` is the syntax check.

### Board settings in the IDE

* Board: **ESP32 Dev Module**
* Upload speed: 921600 (drop to 115200 if uploads fail)
* Flash size: 4 MB
* For the on-device-inference sketch: **Partition Scheme → "Huge APP (3 MB no OTA)"**
  is *not* needed for the CNN (see `docs/EDGE_DEPLOYMENT.md`), but it is harmless.

---

## 4. First power-up checklist

Open the Serial Monitor at **115200 baud**. You should see diagnostic lines
beginning with `#`, then the header row, then data:

```
# MAX30102 ready
# MPU6050 ready
ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label
1.000,78.0,,,33.40,0.010,-0.020,0.990,1.00,0.30,-0.50,1,-1
1.066,,,95,33.41,0.011,-0.021,0.988,0.90,0.20,-0.40,1,-1
...
```

**The empty fields are intentional.** `spo2` and `eda` are always blank:

* no GSR/EDA sensor is on this board, and
* an uncalibrated SpO₂ number would be a fabricated measurement. A real SpO₂
  value needs a red/IR ratio calibration curve per finger and per sensor; the
  MAX30102 library does not provide one. The project's rule is that a missing
  reading stays missing.

`hr` is blank whenever the finger is off the sensor or fewer than two beats have
been seen — the parser upstream treats it as *not measured*, not as `0`.

### Verify the bus

If a sensor does not appear, run an I²C scanner (any of the standard examples) and
confirm you see exactly three addresses: `0x3C`, `0x57`, `0x68`.

| Symptom | Likely cause |
|---|---|
| only `0x68` found | MAX30102/OLED not powered — check `VIN`, not `VCC`, on the MAX30102 board |
| no addresses at all | SDA/SCL swapped, or the bus is held low by too many pull-ups |
| `# MPU6050 FAILED` | wrong address (`AD0` pin pulled high moves it to `0x69`) |
| OLED blank but data streaming | `OLED_ADDR` is `0x3C` in this sketch; some boards are `0x3D` |
| HR constantly blank | finger pressure too light/heavy, or `IR < 50000` — the sketch reports finger presence in the `finger` column |

---

## 5. How it is worn

The device is strapped **on the forearm/wrist**, with:

* the **MAX30102** pressed against the skin over a pulse point — the underside of
  the wrist works, and the sensor must be shielded from ambient light (the
  breakout's optical barrier helps; do not leave a gap at the edges),
* the **MPU6050** rigidly mounted to the strap so limb motion is measured and not
  damped by the strap,
* the **OLED** facing the wearer.

This is the configuration the whole pipeline is built around: the model was
trained on **WESAD wrist** signals (BVP→HR, EDA, TEMP, ACC), which is what this
exact sensor set can produce. The predecessor project trained on WESAD *chest*
ECG/EDA/respiration — signals this hardware physically cannot generate — which is
why its "live hardware" mode could never reproduce its own training distribution.

---

## 6. Sensing rates and the CSV contract

| Quantity | Rate | Source |
|---|---|---|
| accelerometer (×3), gyroscope (×3), temperature | ~15 Hz (66 ms) | MPU6050 |
| heart rate | beat-to-beat, smoothed | MAX30102 IR + `checkForBeat` |
| finger detection | every sample | MAX30102 IR threshold |

One line per sample:

```
ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label
```

* `ts` — seconds since boot, 3 decimals (a real timeline, so it aligns with the
  WESAD/replay timeline)
* `label` — always `-1` on a live board (unknown; there is no ground truth in
  real life)
* Downstream feedback, laptop → board: `STATE:<name>` and `SCORE:<0-100>`, shown
  on the OLED. On the ESP32 these arrive over **BLE NUS** (or USB serial).

## 7. Why the model is not on the Uno

Measured, not asserted: the compact deployed network (`models/cnn.pt`) has
**49 347 parameters**. As int8 weights that is ~48 KB of storage plus a 242 KB
working arena for activations — decisive against the ATmega328P's 2 KB SRAM /
32 KB flash. On the ESP32 the *sense-and-display* firmware links cleanly, but the
model's arena does **not yet fit the static data region** (measured: link fails
by 172 KB) — heap allocation is the open fix. Full numbers and the parity proof
are in `docs/EDGE_DEPLOYMENT.md`; the Uno sketch does sensing and display only.

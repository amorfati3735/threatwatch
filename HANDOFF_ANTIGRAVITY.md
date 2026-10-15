# HANDOFF (parallel agents) — ThreatWatch demo hardening

Audience: an AI coding agent running **on Windows** (Antigravity), working in
parallel with another agent that owns the sensor/backend path.
Date: 2026-10-14. Repo: `C:\Users\DELL\threatwatch` (WSL view: `/mnt/c/Users/DELL/threatwatch`).

Read this whole file before touching anything. It is written to be handed to an
agent **verbatim per task** (see "Work orders" at the end).

---

## 1. What the system is

An ESP32 wearable (MAX30102 PPG + MPU6050 IMU + SSD1306 OLED, one I2C bus) streams
one CSV line per sample over **BLE (Nordic UART Service)** and USB serial:

```
ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label
452.520,78.0,,,33.40,0.010,-0.020,0.990,1.00,0.30,-0.50,1,-1
```

* `ts` = seconds since boot (float). `spo2` and `eda` are **always empty** on this
  kit (no SpO2 calibration, no GSR module) — empty means "not measured" and is
  never faked. `hr` is empty when there is no finger contact.
* A FastAPI dashboard (`run.py`, port 8000) ingests those lines, classifies,
  fuses with a YOLOv8n-pose webcam analyser, runs an alert state machine, and
  mirrors `STATE:<name>` / `SCORE:<0-100>` back to the OLED.

**The demo we are building toward** (this is the acceptance bar):

1. ESP32 rig runs off a **powerbank** (no USB data cable), finger on the MAX30102.
2. Values stream over **Bluetooth** to the laptop.
3. The values are visible **on the website** (live tiles/charts/CSV honesty).
4. The same values are readable **in a terminal** (raw line + decoded fields).
5. A **"Simulate Threat" button on the website** raises the threat, and a
   **Telegram notification** is actually delivered.
6. **No emojis anywhere on the website.** Plain ASCII/text only.

---

## 2. Environment quirks (do not rediscover these)

* **Two Pythons. This trips everyone up.**
  * **Windows** `C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe`
    — has torch+CUDA, ultralytics, opencv, **bleak**, pyserial, uvicorn, fastapi.
    **The dashboard MUST run here** (WSL cannot see the webcam, Bluetooth, or COM
    ports).
  * **WSL venv** `/home/pratik/.venvs/threatwatch/bin/python` — training + pytest
    (Windows Python has no pytest). Listener/script tasks that only need BLE/serial
    can run on either, but run them on **Windows** for the demo.
* **Board port** is `COM11` (CP210x). `COM4-COM7` are Bluetooth serial ports — ignore
  them. BLE is **not** a COM port; it is the `BLE` entry in the website's port
  dropdown.
* **Zombie WSL servers**: if a WSL `run.py` survives, it answers port 8000 with no
  camera/BLE/serial (site loads, nothing works). `review_day.bat` pkills them first.
  If the site shows "camera: unavailable" plus an empty port list, that is this.
* **Git Bash mangles `start /min`** — use
  `powershell -Command "Start-Process ..."` to launch detached Windows processes.
* **Hard-refresh the browser (Ctrl+F5)** after editing `frontend/`. The frontend is
  served from disk by FastAPI `StaticFiles`, so **no server restart is needed** for
  HTML/CSS/JS changes.
* **Windows-generated files** are CRLF; check before rewriting. Config/notes files
  can be UTF-16.
* **Only one process can hold COM11** at a time. Server holding it blocks flashing.
  Free it with `POST /api/arduino/connect {"action":"disconnect"}`.
* **`/api/status` and the websocket `build_payload()` are different payloads** —
  `/api/status` has no `arduino`/`sensor_ml` keys. Don't confuse them.
* **BLE naming/UUIDs** (firmware is the source of truth, same values in
  `threatwatch/config.py`): device name `ThreatWatch`,
  service `6e400001-b5a3-f393-e0a9-e50e24dcca9e`, laptop -> device (write)
  `...002...`, device -> laptop (notify) `...003...`. First scan takes ~10-15 s.
* Alerts are **off by default** (`.env` absent -> notifier `null`). Telegram needs
  `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, then `ALERTS_ENABLED=1` and
  `ALERT_CHANNEL=telegram`. See `.env.example`.
* Alert timing (so a demo does not look broken): the FSM needs the fused score in
  the THREAT band for `THREAT_CONSECUTIVE_SECS=2.5 s`, then
  `ALERT_ESCALATE_SECS=1.5 s` before it dispatches. So ~4 s after pressing
  "Simulate Threat" the notification goes out. Cooldown between alerts: 30 s.

---

## 3. In-flight work — do NOT touch these files

Another agent is right now fixing the live-hardware **timeline** bug (device `ts`
is rebased onto the host clock; a record split by a serial read timeout used to
shift the whole timeline by hundreds of seconds, and a later board reset then
drove `span_sec` negative with `ready:false`). That agent owns:

* `threatwatch/threatwatch/data/sources/hardware.py`
* `threatwatch/threatwatch/data/sources/serial_source.py`
* `threatwatch/threatwatch/api/app.py`
* `threatwatch/tests/*`
* the running dashboard process (it will restart it several times)
* the ESP32 / BLE / COM11 hardware (one client at a time — do not connect to the
  board while it does)
* `firmware/esp32_wearable/esp32_wearable.ino` (reflashed this session)
* `scripts/esp32_console.py` (already written and verified — see below)

Everything else is fair game. **Do not restart the server while testing**; if you
need fresh static files, just hard-refresh the browser. If the server is down when
you need it, run `review_day.bat` (double-click) rather than starting `run.py`
yourself.

Two more facts you need:

* `scripts/live.py --source ble` already exists and prints a *prediction* line, not
  the raw records. It is useful but not the "raw values in a terminal" view.
*There is currently **no** "Simulate Threat"/test-alert button; the demo panel has
`NORMAL / PHYSIOLOGICAL_DISTRESS / MULTIMODAL_THREAT / FALSE_POSITIVE_REDUCTION /
LIVE_FREE` scenario buttons and a `Reset Alarm` button.

The terminal value view the demo needs **already exists**:

```
python scripts/esp32_console.py                 # tap the running dashboard
python scripts/esp32_console.py --source ble    # talk to the board directly
python scripts/esp32_console.py --source ble --raw   # + untouched record text
python scripts/esp32_console.py --source serial --port COM11
```

Verified working against the live board over BLE. Only extend it if the talk
needs something it does not print; do not rewrite it.

Firmware changes already flashed this session (do not re-do them, and keep the
CSV contract at 13 fields):

* the PPG FIFO is drained at the sensor's own rate (~25 Hz) and every sample is
  fed to the SparkFun PBA beat detector, so `hr` is populated (it used to be
  blank - "-- BPM" - because only one decimated sample per record reached the
  detector);
* the OLED shows heart rate, the three accelerometer axes, the raw IR level plus
  contact state, then the laptop verdict on one line;
* the MPU6050 die temperature was removed from the OLED (it is ~44 C whether or
  not a finger is present, and it is not skin temperature);
* a `# ir=... bpm=... beats=... finger=... state=... score=...` comment line is
  emitted every 2 s over both links for diagnosis. `#` lines are ignored by every
  parser, so the CSV contract is unchanged.

---

## 4. Work orders (hand one per agent session)

### Work order A — website: no emojis, plus Simulate Threat and Telegram test

**Goal:** the dashboard is emoji-free and has one obvious button that raises a
threat and delivers a Telegram notification.

Facts you need:

* Files: `frontend/index.html`, `frontend/js/app.js`, `frontend/js/charts.js`,
  `frontend/css/style.css`. No framework — plain JS + Tailwind-ish utility classes
  in `css/style.css`.
* Existing wiring to reuse (do not invent new endpoints):
  * scenario buttons: `document.querySelectorAll("[data-scenario]")` ->
    `triggerScenario(name)` -> `POST /api/demo/scenario {"scenario": "..."}`.
  * alarm reset: `#btnAlertReset` -> `POST /api/alert/reset`.
  * a direct notifier test already exists server-side:
    `POST /api/alerts/test` -> sends one alert through the configured channel and
    returns `{"delivered": true|false, "notifiers": [...]}`. Nothing calls it yet.
* `MULTIMODAL_THREAT` is the scenario that drives the full path: fusion override ->
  score in THREAT band -> FSM -> real dispatch (Telegram when configured) -> OLED
  flips -> audio alarm. Pressing it is the honest way to "simulate a threat".

Acceptance criteria:

1. **No emoji or symbol glyphs anywhere in `frontend/`.** Replace them with plain
   text. Present today:
   * `index.html`: `->` arrows (U+2192, 3x), lightning bolt (U+26A1, 2x), electric
     plug (U+1F50C), speaker (U+1F50A), sun (U+2600 U+FE0F), graduation cap
     (U+1F393), black circle (U+25CF), play (U+25B6), pause (U+23F8),
     restart (U+21BA).
   * `js/app.js`: sun (U+2600 U+FE0F), moon (U+1F319), speaker/ muted speaker
     (U+1F50A / U+1F507), graduation cap (U+1F393, 2x), filled/hollow circles
     (U+25CF / U+25CB).
   * **Keep** the unit characters `µ` (microsiemens) and `°` (degrees Celsius) and
     the em dash — they are typography, not emoji. Keep the theme toggle, mute
     button and replay controls working; only their glyphs change (e.g. mute ->
     `AUDIO ON` / `MUTED`, play/pause/restart -> `Play` / `Pause` / `Restart`,
     theme -> `LIGHT` / `DARK`, logo -> remove the cap glyph and keep the text).
   * Verify with: `grep -rnoP "[^\x00-\x7F]" frontend/` and check each remaining
     hit by hand. Expected leftovers after the change: only `µ`, `°` and any
     em/en dash you keep.
2. **A `SIMULATE THREAT` button** in the "Multimodal Threat Decision" card
   (`frontend/index.html`, near `#stateBadge` / `#btnAlertReset`), styled like the
   existing `btn-uni btn-alert` buttons, with a label that says exactly
   `Simulate Threat`.
   * On click: `POST /api/demo/scenario {"scenario": "MULTIMODAL_THREAT"}` and show
     feedback that the threat is being raised (e.g. set a status/progress hint in
     the card). It must NOT silently do nothing while the FSM counts down its ~4 s.
   * `POST /api/alert/reset` now also drops an active demo override server-side,
     so `Reset Alarm` alone returns the system to safe (the button needs no extra
     call; verify it does).
3. **A small `Test Telegram` button** next to it that calls `POST /api/alerts/test`
   and reports the result in the UI (success: notification delivered; failure:
   say which notifiers are configured, e.g. `null` means alerts are disabled in
   `.env`). This is the pre-demo sanity check.
4. Nothing console-throwing on load. Check the browser console.
5. **Relabel the temperature tile.** `frontend/js/app.js` shows the `temp` field
   as `Skin Temp`, but that field is the **MPU6050's internal die temperature**
   (~44 C regardless of contact). Labelling it skin temperature is exactly the
   kind of invented reading this project forbids. Rename it (e.g.
   `IMU Die Temp (not skin)`) or drop the tile, and do the same anywhere else
   "Skin Temp" appears.

### Work order B — terminal raw-value viewer — DONE (two tools, deliberately)

Both exist and are verified; do not add a third.

| tool | connects to | use it when |
|---|---|---|
| `scripts/esp32_console.py` (default `--source ws`) | the running dashboard's websocket | **the demo**: the site holds the BLE link and the terminal mirrors what the pipeline received |
| `scripts/esp32_console.py --source ble/--source serial` | the board itself | nothing else is holding the link |
| `scripts/ble_console.py` | the board over BLE (`--csv` logs records, `--raw`/`--decode`) | a standalone wired/wireless capture session, or recording a CSV to replay later |

**Hardware fact that decides which one you use:** the ESP32 stops advertising
while a client is connected, so a second BLE client on the same laptop will not
find the board (the direct console just re-scans forever, by design). One BLE
client at a time — during the talk that client is the dashboard.

### Work order C — Telegram delivery, verified

**Goal:** the `Simulate Threat` button delivers a real Telegram message.

Do this only after Work order A is in and the user has created the bot.

1. Ask the user for `TELEGRAM_BOT_TOKEN` (from @BotFather) and `TELEGRAM_CHAT_ID`
   (send any message to the bot, then read
   `https://api.telegram.org/bot<token>/getUpdates` and take
   `result[-1].message.chat.id`). Do not invent values.
2. Create `.env` in the repo root from `.env.example` (it does not exist yet) with:
   `ALERTS_ENABLED=1`, `ALERT_CHANNEL=telegram`, the two Telegram values.
3. Restart the dashboard **only if** the other agent is not mid-verification;
   otherwise ask the user to close the "ThreatWatch Server Logs" window and rerun
   `review_day.bat`.
4. Verify by *receiving* the message: `POST /api/alerts/test` must return
   `delivered: true` and the Telegram message must arrive. Then press
   `Simulate Threat` and confirm a *second*, different message (the escalated
   alert) arrives within ~4-6 s. Paste the `/api/alerts/status` JSON as evidence
   (`sent` counter increments, `failed` stays 0).
5. `cooldown_secs` is 30 s, so two alerts close together will legitimately be
   deduped/cooldowned — wait between attempts.

### Work order D — documentation polish (low priority, no hardware)

* `README.md` "Quickstart" still says 55 tests; the suite is now 64 and growing.
* `README.md` and `docs/ROADMAP.md` say 55 tests / 58 tests in places; make the
  numbers consistent with `pytest tests -q` output on the WSL venv.
* Anything in `docs/` that still references the old `fittrack/` layout.
* Do not touch `HANDOFF_REVIEW.md` (the other agent's work log) or `docs/ROADMAP.md`
  rows that describe unfinished training runs - those are intentionally open.

---

## 5. Definition of done for the demo

* Powerbank-powered board, finger on the sensor, BLE link up, values moving on the
  website with `spo2`/`eda` visibly empty (never faked).
* Terminal window showing the same records:
  `python scripts/esp32_console.py --source ws --raw` (works while the site holds
  the BLE link), reporting a plausible `hr` now that the firmware samples the PPG
  at the sensor's own rate.
* `Simulate Threat` -> score climbs -> `THREAT`/`ALERT` on the site -> OLED shows
  `STATE`/`SCORE` -> audio alarm -> Telegram message received -> `Reset Alarm`
  returns to safe.
* No emojis on the site; no console errors; camera live with YOLO pose.

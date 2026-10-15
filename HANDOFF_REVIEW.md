# ThreatWatch — Handoff (final review prep)

Date: 2026-10-14. Read `README.md` and `docs/ROADMAP.md` first for the project
story. This file covers the live-demo wiring and what's still unfinished.

## The system in one paragraph

ESP32 wearable (MAX30102 PPG + MPU6050 IMU + SSD1306 OLED, one I²C bus) streams
CSV lines `ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label` over USB serial
(115200) and BLE (Nordic UART Service, name "ThreatWatch"). A FastAPI dashboard
(`run.py`, port 8000) ingests them, classifies with `models/cnn_transformer.pt`
(CUDA), fuses with YOLOv8n-pose webcam analysis, drives an alert state machine,
and mirrors STATE/SCORE back to the OLED. Model trained on WESAD in WSL
(`/home/pratik/.venvs/threatwatch`); see `scripts/experiments.sh`.

## Two Pythons — this trips everyone up

- **Windows Python** `C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe`
  — has torch+CUDA, ultralytics, opencv, **bleak**, pyserial. **The dashboard MUST
  run here**: WSL cannot see the webcam, Bluetooth, or USB serial.
- **WSL venv** `/home/pratik/.venvs/threatwatch/bin/python` — training +
  **pytest** (Windows Python has no pytest). Run tests:
  `wsl -- bash -lc "cd /mnt/c/Users/DELL/threatwatch && /home/pratik/.venvs/threatwatch/bin/python -m pytest tests -q"`
  (58 tests, none need hardware/dataset).

**Quirk: zombie WSL servers.** If a WSL `run.py` dies badly, the Linux-side
process survives and silently answers port 8000 with no webcam/BLE/serial —
the site loads but nothing works. `review_day.bat` now pkills WSL run.py
processes first; if you see "camera: unavailable" + empty port list, that's this.

## Changes made this session (all in working tree, uncommitted)

1. **BLE ingest added** — `threatwatch/data/sources/hardware.py` now has a BLE
   transport (bleak) parallel to USB serial. Port dropdown lists "BLE"; connect
   scans for name "ThreatWatch". Config UUIDs in `threatwatch/config.py`
   (`BLE_*`). Auto-reconnects; STATE/SCORE feedback written over NUS RX.
2. **Timestamp rebase** — device `ts` is seconds-since-boot and restarts when
   the board resets (USB open resets it). `_park_packet()` rebases onto host
   clock; without this the window span goes negative and `ready` stays false
   (this was the "USB readings don't show up" bug).
3. **Review Mode** — `frontend/js/app.js` `initReviewMode()` + `.replay-only`
   CSS class hides all WESAD replay UI. Persisted in localStorage
   (`tw_review_mode`). Button in header.
4. **ESP32 display mode fixed** — server broadcasts source `HARDWARE_LIVE` but
   app.js only checked `ARDUINO_LIVE`, so the site stayed in WESAD display mode
   with the board connected (wrong tiles/charts/labels). Fixed to accept both;
   labels renamed to ESP32; hardware Slot2 changed SpO2 (always blank, no
   sensor) → Skin Temp.
5. **OLED simplified** — `firmware/esp32_wearable/esp32_wearable.ino`
   `oledUpdate()` rewritten: pure text (big BPM, TEMP, FINGER OK, STATE, SCORE),
   no PPG waveform/graph. Compiled (86% flash) and **flashed to COM11** via
   Windows arduino-cli at
   `C:\Users\DELL\AppData\Local\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe`
   (no WSL arduino-cli). Must disconnect the dashboard from COM11 before
   uploading or the port is busy.
6. **`review_day.bat`** — one-click launcher: kills WSL zombies, health-checks
   `/api/status`, starts server in a **visible "ThreatWatch Server Logs" cmd
   window** (`cmd /k python -u run.py`), opens browser.

## OPEN BUG — RESOLVED (2026-10-14, verified on the board)

Reproduced before changing anything: the running server's `pkt_ts` sat **435 s
ahead of the host clock** (span 645 s) while the device clock itself was measured
accurate (ts advances 1.000 s per wall second, 13-field records, ~8 Hz).

**Root cause.** `_park_packet()` trusted the first field of *any* line that
parsed, and only re-anchored when the device clock stepped **backwards**. A
record split by a `readline()` timeout (the sketch blocks ~100 ms in I2C for the
OLED) leaves its tail in the driver buffer; that tail arrives as the next "line",
whose first field is a *fragment* of the real timestamp (`452.400` cut to
`2.400`). That fragment looked like a board reset, re-anchored the offset, and
every later sample inherited the error — permanently, because nothing ever
compared the rebased clock against the host clock again. A later reset then
walked the buffer backwards, which is the negative span.

**Fix.** (a) both halves of a split read are dropped, and a record shorter than
the canonical 13 fields is never a sample; (b) the offset is re-anchored whenever
the rebased clock stops tracking the host clock (not only on a backwards step);
(c) the timestamp handed to the buffer never goes backwards; (d) clock state is
cleared on disconnect. Regression tests: `tests/test_hardware_timeline.py`
(4 of 5 failed before the fix).

**Verified live.** Offset 0.006 s after connect (was +435 s), `span_sec`
positive and 1:1 with wall time, `ready: true`, and still positive/ready across a
board reset. Also fixed in the same pass, both demo-visible: hardware mode no
longer splices WESAD replay packets into the live buffer during a BLE scan
(`span` raced to ~1.7e9), and `Reset Alarm` no longer re-escalates to ALERT 4 s
later.

## Remaining tasks for the review

1. **Fix the open bug above** (top priority — model output dead without it).
2. **Telegram**: code done (`threatwatch/alerts/telegram.py`), just needs
   `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env` (see `.env.example`),
   then `ALERTS_ENABLED=1 ALERT_CHANNEL=telegram`. Falls back to console
   notifier meanwhile. Also try the "Test Alert" button (`POST /api/alerts/test`).
3. **Dress rehearsal**: `review_day.bat` end-to-end with board on powerbank
   over **BLE** (finger on MAX30102), walk around, check link holds; then
   MULTIMODAL THREAT scenario → score climbs → audio alarm + OLED STATE/SCORE
   → Alert Reset. Note: the edited bat has NOT been double-click tested yet.
   Status: the launcher was run and does start the server when none is up, the
   BLE link was verified end-to-end (records arrive whole at the device's own
   ~8.4 lines/s, no truncated notifications), and the full threat sequence was
   driven twice: CAUTION → ALERT ACTIVE, exactly one dispatch per episode, and
   the panel received `STATE: THREAT / SCORE: 95` then returned to `SAFE` on
   reset. Still to do by hand: powerbank + walking around, and the audio alarm
   in the browser.
4. **Hard-refresh the browser** (Ctrl+F5) after frontend edits — app.js changes
   don't show otherwise.

## Demo sequence (the plan)

`review_day.bat` → camera live w/ YOLO pose → pick COM11 (USB) or BLE in
dropdown → Connect (~10–15 s for BLE scan) → live vitals on site + OLED text →
show charts/CSV honesty (blank spo2/eda = no sensor, never faked) → press
**MULTIMODAL THREAT** (injects labelled DEMO_OVERRIDE into fusion, works on
live hardware) → alarm fires, OLED flips → Alert Reset + LIVE_FREE.
Evidence slides: README results table (CNN 0.617 macro-F1 subject-disjoint vs
SVM 0.486) and `docs/EDGE_DEPLOYMENT.md` parity table (int8 C vs PyTorch
argmax agreement 1.0000).

## Environment quirks

- Git Bash mangles `start /min` — use `powershell -Command "Start-Process ..."`
  to launch detached Windows processes.
- Server holding COM11 blocks flashing; `POST /api/arduino/connect
  {"action":"disconnect"}` first.
- `/api/status` returns a *different, smaller* payload than the websocket
  `build_payload()` (no `arduino`/`sensor_ml` keys) — don't confuse the two.
- Board seen as COM11 (CP210x). Ignore the Bluetooth COM4–7 ports; BLE is not
  a COM port — it's the "BLE" dropdown entry.

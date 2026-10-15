# Agent notes — hardware integration state (2026-10-13)

One doc for whoever works on this next. Everything below was **verified live
on this machine**, not planned.

## What exists now (both verified end-to-end)

```
WIRED:  ESP32 ─USB─ COM11 ─windows_serial_bridge.py─┐
                                                    ├─ TCP 127.0.0.1:8765 ─ ws_pty_relay.py ─ /dev/pts/N ─ live.py
BLE:    ESP32 ─BLE─ windows_ble_bridge.py (bleak)──┘        (all in scripts/)
```

The board runs `firmware/esp32_wearable` (CSV over USB serial **and** BLE NUS
as "ThreatWatch"); the laptop runs `scripts/live.py --source serial --port /dev/pts/N`
→ trained CNN (`models/cnn_wnorm.pt`) prints `STATE = … win=(480,7)` ~1×/s from
**real sensors**. BLE works with the board on a power bank (nothing on USB).

## Machine map (WSL2 vs Windows — the single most important fact)

| Resource | WSL2 | Windows |
|---|---|---|
| COM ports | ❌ never | ✅ pyserial |
| Bluetooth | ❌ no radio | ✅ bleak (WinRT) |
| ThreatWatch venv | ✅ `/home/pratik/.venvs/threatwatch` (68/68 tests) | — |
| pyserial / bleak | — | ✅ `C:\Users\DELL\AppData\Local\Programs\Python\Python310` |
| arduino-cli | via interop: `…\Arduino IDE\resources\app\backend\resources\arduino-cli.exe` (full path in README §Hardware; needs `ARDUINO_DIRECTORIES_DATA/USER` env) | |

Rule: **serial/BLE clients run on Windows; the ML pipeline runs in WSL; the
pty relay stitches them.** Windows→WSL localhost forwarding works (NAT mode).

## Hardware truth (don't re-diagnose)

- ESP32-WROOM-32, 38-pin; **CP2102 → COM11** (driver installed as oem269.inf
  from MS Update Catalog after a Code 28; survives replug).
- Bus: SDA=21, SCL=22 → OLED `0x3C`, IMU `0x68`, MAX30102 `0x57`.
- **The "MPU6050" is a MPU6500** (`WHO_AM_I=0x70`). Raw-register code works;
  Adafruit_MPU6050 hard-rejects it. Its temp reads +11.5 °C high with the
  MPU6050 formula the firmware uses (harmless for the wnorm model).
  Full detail: `docs/MPU6050.md`.
- Actual sample rate ≈ **12.4 Hz** (nominal 20); pipeline resamples, so fine.

## Bridge scripts (in `scripts/`)

| Script | Runs on | Role |
|---|---|---|
| `windows_serial_bridge.py --port COM11` | Windows | pyserial ↔ TCP, auto-reconnect, keeps port open |
| `windows_ble_bridge.py` | Windows | bleak scan "ThreatWatch" → NUS TX notify ↔ TCP (+ RX feedback); needs BT radio **ON** |
| `ws_pty_relay.py` | WSL | TCP ↔ fresh pty; prints `PTY=/dev/pts/N` — pass that to live.py |

Start order: relay → bridge → `python -u scripts/live.py --source serial --port /dev/pts/N`.
Add `-u`; otherwise killing live.py loses its buffered stdout. Stop: kill by
PID (pgrep first) — never `pkill -f <pattern>` in a command that also contains
that pattern's literal text (it kills your own shell; happened twice).
- If the board **reboots mid-run** (closing a COM handle drops DTR → auto-reset;
  `ts` jumps ~700→~0), live.py prints huge negative buffering
  (`[live] buffering… -742.0/15s`) that climbs back to 0 and **predictions resume
  on their own** — no action needed, don't "fix" the pipeline.

## Firmware facts (`esp32_wearable`)

- CSV rule: `ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label` — empty =
  really absent (spo2/eda always empty; hr empty until real beats).
- **OLED is DRIVEN since 2026-10-13** (splash + 10 Hz live UI in `esp32_wearable`:
  THREATWATCH header, BLE/USB flag, spinner, BPM big, T + finger flag,
  state line, scrolling PPG strip). Blank = init failed; setup logs
  `# OLED ready` / `# OLED not found` (real ACK check). Screen still dark after
  the splash? reseat the 4 OLED duponts.
- BLE MTU reports 23 but lines arrive untruncated (measured). BLE MAC seen:
  7C:D5:44:20:76:BA.
- Compile (byte-reproducible, 85 % flash) — command in README §Hardware.

## Ops gotchas hit today

- Arduino IDE **Serial Monitor holds COM11** → "Access denied" on open; close it
  before using the bridges (device present = "denied", unplugged = "not found").
- The Windows bridge must be killed with its **real Windows PID**
  (tasklist/PowerShell `Win32_Process`), not the WSL interop `/init` pid.
- `run_commands` here times out at 30 s → long compiles: background + poll logs
  in `/tmp/*.log`.
- PowerShell from bash: single-quote `-Command '…'`; WinRT async needs
  `Add-Type System.Runtime.WindowsRuntime` + AsTask dance (don't bother —
  toggling BT programmatically failed; user toggle works).
- Bluetooth was **powered off** in Windows; bleak error was
  `BleakBluetoothNotAvailableError`. Ask user / `ms-settings:bluetooth`.

## Files created this session (index)

- `scripts/{ws_pty_relay, windows_serial_bridge, windows_ble_bridge}.py` — bridges
- `scripts/demo_live.py` — **one-command demo** (relay + bridge + live.py,
  saves `logs/demo_<stamp>.csv`; default BLE, `--wired` for USB)
- `docs/MPU6050.md` — IMU truth (6500 clone, temp bias, lib matrix)
- `C:\Users\DELL\esp32_handoff.md` — earlier bring-up handoff (COM11, driver, toolchain)
- Sketches in `C:\Users\DELL\Documents\Arduino\`: `i2c_scanner`, `all_sensors_test`
  (0x70-aware), `mpu_probe` — hardware bring-up tools

## Likely next asks

1. ~~OLED on-wrist UI~~ — **done 2026-10-13** (verify visually on the panel).
2. Patches worth doing: 0x70-aware temp offset in `mpuInit()` (snippet in
   `docs/MPU6050.md`); `live.py --device cpu` pinning if CUDA ever appears.
3. Full dashboard (`run.py --no-vision`) needs the heavy deps installed —
   bridges are already compatible (feed the same pty).

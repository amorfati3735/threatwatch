#!/usr/bin/env python3
"""
Live ESP32 value console — the wearable's stream, in a terminal.

Three ways to watch, depending on what else is running:

    # 1. while the dashboard is open (default) - taps the dashboard's websocket
    python scripts/esp32_console.py

    # 2. straight off the board over Bluetooth, nothing else holding it
    python scripts/esp32_console.py --source ble

    # 3. straight off the board and print the untouched notification text,
    #    including the firmware's "# ir=... bpm=... beats=..." diagnostics
    python scripts/esp32_console.py --source ble --raw

    # wired alternative (the dashboard must be disconnected from the port)
    python scripts/esp32_console.py --source serial --port COM11

Rules this viewer keeps, because the demo's honesty depends on them:

* a reading the board did not take prints as ``--`` (empty hr / spo2 / eda),
  never as ``0`` and never as a plausible-looking stand-in;
* ``spo2`` and ``eda`` are *always* empty on this kit - there is no GSR module
  and no calibrated SpO2 curve, so those two columns are blank on purpose;
* the ``temp`` field is the MPU6050's own die temperature (it reads ~44 C at
  idle, finger or not). It is not skin temperature. There is a MAX30102
  temperature too, and neither is a real body-temperature measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEVICE_NAME = "ThreatWatch"
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device -> laptop (notify)
DASHBOARD_WS = "ws://127.0.0.1:8000/ws"


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    """A missing reading prints as --, never as a zero."""
    if value is None:
        return "--"
    return f"{value:.{digits}f}{unit}"


def decoded_line(stamp: str, ts: float | None, hr, spo2, eda, temp, ax, ay, az, finger, extra: str = "") -> str:
    contact = {True: "finger", False: "NO FINGER", None: "contact?"}[finger]
    return (
        f"{stamp} ts={fmt(ts, digits=2):>8}  hr={fmt(hr, digits=0):>4}  "
        f"spo2={fmt(spo2, digits=0):>3}  eda={fmt(eda, digits=1):>4}  temp={fmt(temp, 'C'):>6}  "
        f"acc=({fmt(ax, digits=2)},{fmt(ay, digits=2)},{fmt(az, digits=2)})  {contact}{extra}"
    )


# --------------------------------------------------------------------------- #
# source: dashboard websocket
# --------------------------------------------------------------------------- #
def run_ws(show_csv: bool) -> int:
    try:
        from websocket import create_connection
    except ImportError:
        print("[console] the ws viewer needs websocket-client:  pip install websocket-client")
        return 2

    print(f"[console] tapping the dashboard at {DASHBOARD_WS} (Ctrl+C to stop)")
    ws = create_connection(DASHBOARD_WS, timeout=10)
    seen = None
    try:
        while True:
            payload = json.loads(ws.recv())
            pkt = payload.get("sensor_packet") or {}
            ts = pkt.get("timestamp")
            if ts is None or ts == seen:
                continue
            seen = ts
            ml = payload.get("sensor_ml") or {}
            sm = payload.get("state_machine") or {}
            extra = f"  | score={fmt(ml.get('distress_score', 0) * 100, '%', 0):>4}  {sm.get('current_state', '-')}"
            # The dashboard hands out a host-rebased timestamp, which is not the
            # board's uptime, so it is shown as -- rather than rendered as if it
            # were a device clock reading.
            print(decoded_line(
                time.strftime("%H:%M:%S"), None,
                pkt.get("heart_rate"), pkt.get("spo2"), pkt.get("eda"),
                pkt.get("temperature"), pkt.get("acc_x"), pkt.get("acc_y"), pkt.get("acc_z"),
                pkt.get("finger_detected"), extra,
            ))
            if show_csv:
                print(f"          csv: {csv_from_packet(pkt)}")
    except KeyboardInterrupt:
        print("\n[console] stopped")
    except Exception as exc:
        print(f"[console] dashboard link failed: {exc}")
        return 1
    finally:
        ws.close()
    return 0


def csv_from_packet(pkt: dict) -> str:
    """Rebuild the canonical record from a telemetry packet."""

    def f(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    finger = pkt.get("finger_detected")
    return ",".join([
        f(pkt.get("timestamp")), f(pkt.get("heart_rate")), f(pkt.get("spo2")), f(pkt.get("eda")),
        f(pkt.get("temperature")), f(pkt.get("acc_x")), f(pkt.get("acc_y")), f(pkt.get("acc_z")),
        f(pkt.get("gyro_x")), f(pkt.get("gyro_y")), f(pkt.get("gyro_z")),
        "" if finger is None else ("1" if finger else "0"), str(pkt.get("label", -1)),
    ])


# --------------------------------------------------------------------------- #
# source: the board itself (BLE / serial)
# --------------------------------------------------------------------------- #
def run_ble(raw: bool) -> int:
    try:
        import asyncio

        from bleak import BleakClient, BleakScanner
    except ImportError:
        print("[console] the BLE viewer needs bleak:  pip install bleak")
        return 2

    async def main() -> int:
        print(f"[console] scanning for '{DEVICE_NAME}' (takes ~10-15 s) ...")
        device = await BleakScanner.find_device_by_filter(
            lambda d, ad: (ad.local_name or "") == DEVICE_NAME, timeout=20.0
        )
        if device is None:
            print("[console] board not found - is it powered and advertising?")
            return 1
        print(f"[console] connected to {device.address}; Ctrl+C to stop")

        def on_notify(_sender, data: bytearray) -> None:
            text = bytes(data).decode("utf-8", errors="ignore").strip()
            if not text:
                return
            stamp = time.strftime("%H:%M:%S")
            if text.startswith("#"):
                print(f"{stamp} {text}")                      # firmware diagnostics
                return
            parts = text.split(",")
            if len(parts) < 13:
                print(f"{stamp} [short record, dropped] {text!r}")
                return
            def num(i):
                try:
                    return float(parts[i]) if parts[i].strip() else None
                except ValueError:
                    return None
            if raw:
                print(f"{stamp} {text}")
            print(decoded_line(
                stamp, num(0), num(1), num(2), num(3), num(4), num(5), num(6), num(7),
                None if not parts[11].strip() else (num(11) or 0) >= 0.5,
            ))

        async with BleakClient(device) as client:
            await client.start_notify(NUS_TX_UUID, on_notify)
            while True:
                await asyncio.sleep(0.5)

    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[console] stopped")
        return 0
    except Exception as exc:
        print(f"[console] BLE error: {exc}")
        return 1


def run_serial(port: str, raw: bool) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from threatwatch.data.sources.serial_source import parse_line

    try:
        import serial
    except ImportError:
        print("[console] the serial viewer needs pyserial:  pip install pyserial")
        return 2

    print(f"[console] opening {port} (this resets the board); Ctrl+C to stop")
    with serial.Serial(port, 115200, timeout=0.2) as ser:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            stamp = time.strftime("%H:%M:%S")
            if line.startswith("#"):
                print(f"{stamp} {line}")
                continue
            if raw:
                print(f"{stamp} {line}")
            pkt = parse_line(line)
            if pkt is not None:
                print(decoded_line(
                    stamp, pkt.timestamp, pkt.heart_rate, pkt.spo2, pkt.eda, pkt.temperature,
                    pkt.acc_x, pkt.acc_y, pkt.acc_z, pkt.finger_detected,
                ))


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Live ESP32 value console")
    p.add_argument("--source", choices=["ws", "ble", "serial"], default="ws",
                   help="ws = tap the running dashboard (default); ble/serial = talk to the board")
    p.add_argument("--port", default="COM11", help="serial port for --source serial")
    p.add_argument("--raw", action="store_true", help="also print the untouched record text")
    args = p.parse_args()

    if args.source == "ws":
        return run_ws(show_csv=args.raw)
    if args.source == "ble":
        return run_ble(raw=args.raw)
    return run_serial(args.port, raw=args.raw)


if __name__ == "__main__":
    sys.exit(main())

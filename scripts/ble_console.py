#!/usr/bin/env python3
"""
Terminal raw-value and decoded viewer over Bluetooth LE (Nordic UART Service).

Prints live records arriving from the ESP32 wearable over BLE:
    python scripts/ble_console.py
    python scripts/ble_console.py --name ThreatWatch --csv logs/ble_session.csv
    python scripts/ble_console.py --decode
    python scripts/ble_console.py --raw

Honest reporting:
* Missing values (hr without finger, uncalibrated spo2, absent eda) print as '--',
  never as 0.
* Truncated/rejected records (e.g. short notification fragments) are counted
  and displayed without crashing.
* Auto-reconnects when link drops.
* Plain ASCII only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure threatwatch package is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from threatwatch.config import BLE_DEVICE_NAME
from threatwatch.data.sources.ble_source import BluetoothSensorSource
from threatwatch.data.types import SensorPacket

CSV_HEADER = "ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label\n"


def format_decoded(pkt: SensorPacket) -> str:
    """Format decoded sensor fields as human-readable ASCII text."""
    # Empty hr/spo2/eda must print as '--', never as '0'
    hr_str = f"{pkt.heart_rate:.1f} BPM" if pkt.heart_rate is not None else "-- BPM"
    temp_str = f"{pkt.temperature:.2f} C" if pkt.temperature is not None else "-- C"

    ax = f"{pkt.acc_x:.3f}" if pkt.acc_x is not None else "--"
    ay = f"{pkt.acc_y:.3f}" if pkt.acc_y is not None else "--"
    az = f"{pkt.acc_z:.3f}" if pkt.acc_z is not None else "--"
    acc_str = f"acc({ax},{ay},{az})"

    gx = f"{pkt.gyro_x:.2f}" if pkt.gyro_x is not None else "--"
    gy = f"{pkt.gyro_y:.2f}" if pkt.gyro_y is not None else "--"
    gz = f"{pkt.gyro_z:.2f}" if pkt.gyro_z is not None else "--"
    gyro_str = f"gyro({gx},{gy},{gz})"

    finger_val = 1 if pkt.finger_detected else 0 if pkt.finger_detected is not None else "--"
    finger_str = f"finger={finger_val}"

    # Build fields list
    fields = [f"{pkt.timestamp:.3f}", hr_str]
    if pkt.spo2 is not None:
        fields.append(f"{pkt.spo2:.1f}%")
    if pkt.eda is not None:
        fields.append(f"{pkt.eda:.2f} uS")
    fields.extend([temp_str, acc_str, gyro_str, finger_str])

    return "  ".join(fields)


def parse_args():
    p = argparse.ArgumentParser(description="ThreatWatch BLE terminal raw-value viewer")
    p.add_argument("--name", default=BLE_DEVICE_NAME, help="BLE advertised device name (default: ThreatWatch)")
    p.add_argument("--address", default=None, help="BLE device MAC or UUID address")
    p.add_argument("--csv", type=Path, default=None, help="Append raw accepted records to CSV file")
    p.add_argument("--decode", action="store_true", help="Print only decoded field view")
    p.add_argument("--raw", action="store_true", help="Print only raw CSV line from wire")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    csv_file = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        need_header = not args.csv.exists() or args.csv.stat().st_size == 0
        csv_file = open(args.csv, "a", encoding="utf-8", newline="")
        if need_header:
            csv_file.write(CSV_HEADER)
            csv_file.flush()

    show_mode = "both"
    if args.raw and not args.decode:
        show_mode = "raw"
    elif args.decode and not args.raw:
        show_mode = "decode"

    print("=" * 70)
    print(" ThreatWatch - BLE Raw Value Console")
    print(f" Target Device: {args.name}")
    if args.csv:
        print(f" CSV Logging  : {args.csv}")
    print(f" Display Mode : {show_mode.upper()}")
    print("=" * 70)
    print("Press Ctrl+C to stop.\n", flush=True)

    def on_state(state_msg: str):
        print(f"[ble] {state_msg}", flush=True)

    source = BluetoothSensorSource(
        device_name=args.name,
        address=args.address,
        on_state_change=on_state,
        auto_reconnect=True,
    )

    source.start()

    try:
        while True:
            item = source.get_next_record(timeout=0.2)
            if item is None:
                continue

            raw_line, pkt = item
            if pkt is None:
                # Truncated or invalid record
                print(f"[rejected #{source.rejected_count}] {raw_line}", flush=True)
                continue

            # Log to CSV if configured
            if csv_file:
                csv_file.write(pkt.to_csv() + "\n")
                csv_file.flush()

            # Display
            if show_mode == "raw":
                print(raw_line, flush=True)
            elif show_mode == "decode":
                print(format_decoded(pkt), flush=True)
            else:
                # Combined view: raw line on wire | decoded field view
                print(f"{raw_line} | {format_decoded(pkt)}", flush=True)

    except KeyboardInterrupt:
        print("\n[ble] Stopped by user.", flush=True)
    finally:
        source.stop()
        if csv_file:
            csv_file.close()
        print(f"\nSummary: {source.accepted_count} accepted records, {source.rejected_count} rejected.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

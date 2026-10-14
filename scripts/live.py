#!/usr/bin/env python3
"""
Live state-classification CLI.

Runs the trained sequence model (or the hardware heuristic) over any sensor
source and prints the predicted state, optionally logging raw packets to CSV for
later replay through the *same* pipeline.

    python scripts/live.py --source simulator --state 1
    python scripts/live.py --source wesad --subject S2 --speed 4
    python scripts/live.py --source serial --port COM3 --log logs/session.csv
    python scripts/live.py --source replay --file logs/session.csv
    python scripts/live.py --source hardware --port COM3 --heuristic

The ``--source`` names match the sources used by the server, so a session
recorded here can be replayed there and behaves identically.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threatwatch.config import DEFAULT_SERIAL_BAUD, DEFAULT_SERIAL_PORT, MODELS_DIR, PACKET_CHANNELS
from threatwatch.data.sources import (
    BluetoothSensorSource,
    HardwareSensorSource,
    ReplaySource,
    SerialSensorSource,
    SimulatorSource,
    WESADSource,
)
from threatwatch.data.types import WindowBuffer
from threatwatch.ml.hardware_inference import HardwareDistressInference
from threatwatch.ml.inference import SensorMLPredictor, format_prediction, load_latest_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="ThreatWatch live classification")
    p.add_argument("--source", choices=["simulator", "serial", "ble", "wesad", "replay", "hardware"],
                   default="simulator")
    p.add_argument("--model", type=Path, default=None, help="checkpoint (default: newest)")
    p.add_argument("--heuristic", action="store_true",
                   help="skip the network and use the hardware heuristic")
    p.add_argument("--device", default="auto")
    p.add_argument("--window", type=float, default=None, help="override the checkpoint window (s)")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between predictions")

    p.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    p.add_argument("--baud", type=int, default=DEFAULT_SERIAL_BAUD)

    p.add_argument("--state", type=int, default=0, help="simulator ground-truth 0/1/2")
    p.add_argument("--subject", default="S2")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--file", type=Path, default=None, help="CSV log for --source replay")

    p.add_argument("--log", type=Path, default=None, help="append raw packets to this CSV")
    return p.parse_args()


def make_source(args):
    if args.source == "simulator":
        return SimulatorSource(state=args.state)
    if args.source == "serial":
        return SerialSensorSource(args.port, args.baud)
    if args.source == "hardware":
        return HardwareSensorSource(args.port, args.baud)
    if args.source == "ble":
        return BluetoothSensorSource()
    if args.source == "wesad":
        return WESADSource(args.subject, speed=args.speed)
    if args.source == "replay":
        if args.file is None:
            raise SystemExit("--file is required for --source replay")
        return ReplaySource(args.file, speed=args.speed)
    raise SystemExit(f"unknown source {args.source}")


def main() -> int:
    args = parse_args()

    predictor = None
    heuristic = HardwareDistressInference()

    if not args.heuristic:
        ckpt = args.model or load_latest_checkpoint()
        if ckpt is None:
            print("[live] no checkpoint found — falling back to the hardware heuristic.")
            args.heuristic = True
        else:
            try:
                predictor = SensorMLPredictor(ckpt, device=args.device)
                print(f"[live] model      : {predictor.model_name} ({ckpt.name})")
                print(f"[live] channels   : {predictor.channels}")
                print(f"[live] window     : {predictor.window_sec:g}s @ {predictor.target_fs:g} Hz "
                      f"= {predictor.window_samples} samples")
                print(f"[live] temperature: {predictor.temperature:.3f}")
            except Exception as exc:
                print(f"[live] model load failed ({exc}) — using the heuristic instead.")
                args.heuristic = True

    channels = predictor.channels if predictor else list(PACKET_CHANNELS)[:6]
    window_sec = args.window or (predictor.window_sec if predictor else 15.0)
    target_fs = predictor.target_fs if predictor else 32.0
    buffer = WindowBuffer(window_sec=window_sec, target_fs=target_fs, channels=channels)

    log_f = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        new = not args.log.exists()
        log_f = open(args.log, "a")
        if new:
            log_f.write("timestamp,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label\n")

    source = make_source(args)
    print(f"[live] source={args.source} window={window_sec:g}s channels={channels}")
    source.start()

    n = 0
    try:
        last = 0.0
        while True:
            pkt = source.get_next_sample()
            if pkt is None:
                if not source.is_running():
                    print("\n[live] source finished")
                    break
                time.sleep(0.005)
                continue

            n += 1
            buffer.add(pkt)
            if log_f:
                log_f.write(pkt.to_csv() + "\n")

            if time.time() - last < args.interval:
                continue
            last = time.time()

            if predictor is not None:
                win = buffer.get_window()
                if win is None:
                    print(f"\r[live] buffering… {buffer.span_sec():5.1f}/{window_sec:.0f}s", end="")
                    continue
                cls, probs = predictor.predict(win)
                line = format_prediction(predictor.class_names, cls, probs)
                gt = f"  gt={predictor.class_names[pkt.label]}" if pkt.label >= 0 else ""
                print(f"\r{line}  win={win.shape}{gt}   ")
            else:
                pred = heuristic.predict(pkt)
                print(f"\rHEURISTIC distress={pred['distress_score']*100:5.1f}%  "
                      f"label={pred['predicted_label']:<28s} hr={pred['hr_bpm']:.0f}   ")

    except KeyboardInterrupt:
        print("\n[live] interrupted")
    finally:
        source.stop()
        if log_f:
            log_f.close()
            print(f"[live] logged {n} packets -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

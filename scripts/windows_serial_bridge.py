#!/usr/bin/env python3
"""
Windows side of the COM-port bridge: pyserial COMxx <-> TCP (run with Windows
Python, which is the interpreter that can actually open COM ports):

    python scripts\\windows_serial_bridge.py --port COM11 --relay 127.0.0.1:8765

Connects OUT to the relay started in WSL (scripts/ws_pty_relay.py).  Windows
loopback-forwarding into WSL handles 127.0.0.1, and outbound connections do
not normally trigger a firewall prompt (inbound would).

The serial port is opened once and kept open; only the TCP side reconnects if
the relay goes away.  Opening the port resets most ESP32 devkits (DTR/RTS) —
that is expected; the firmware resumes streaming by itself.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import serial


def sock_to_ser(ser: serial.Serial, sock: socket.socket, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            data = sock.recv(4096)
        except OSError:
            break
        if not data:
            break
        try:
            ser.write(data)
        except serial.SerialException:
            break
    stop.set()


def ser_to_sock(ser: serial.Serial, sock: socket.socket, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            data = ser.read(4096)
        except serial.SerialException:
            break
        if data:
            try:
                sock.sendall(data)
            except OSError:
                break
    stop.set()


def main() -> int:
    ap = argparse.ArgumentParser(description="COM port -> TCP relay client")
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--relay", default="127.0.0.1:8765",
                    help="host:port of scripts/ws_pty_relay.py")
    args = ap.parse_args()

    host, port_s = args.relay.rsplit(":", 1)
    port = int(port_s)

    print(f"[bridge] opening {args.port} @ {args.baud}", flush=True)
    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    time.sleep(1.5)                 # let the board reboot after open
    ser.reset_input_buffer()
    print(f"[bridge] {args.port} open", flush=True)

    while True:                     # TCP side retries forever
        try:
            sock = socket.create_connection((host, port), timeout=5)
        except OSError as exc:
            print(f"[bridge] relay not reachable ({exc}); retrying…", flush=True)
            time.sleep(2)
            continue
        sock.settimeout(None)
        print(f"[bridge] relay connected {host}:{port}", flush=True)

        stop = threading.Event()
        t1 = threading.Thread(target=sock_to_ser, args=(ser, sock, stop), daemon=True)
        t2 = threading.Thread(target=ser_to_sock, args=(ser, sock, stop), daemon=True)
        t1.start(); t2.start()
        while not stop.is_set():
            time.sleep(0.2)
        try:
            sock.close()
        except OSError:
            pass
        print("[bridge] relay lost — reconnecting (serial stays open)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
WSL side of the COM-port bridge: expose a Windows serial port as a local pty.

Windows runs ``windows_serial_bridge.py`` (pyserial COMxx -> TCP client); this
script listens on 127.0.0.1, accepts ONE connection and copies bytes both ways
between the socket and a freshly created pty.  Linux tools (pyserial,
``scripts/live.py --port /dev/pts/N``) can then use a Windows COM port as if
it were a local device — WSL2 has no direct COM access.

    # terminal 1 (WSL)
    python scripts/ws_pty_relay.py --listen 127.0.0.1:8765
    -> PTY=/dev/pts/7

    # terminal 2 (Windows)
    python scripts\\windows_serial_bridge.py --port COM11 --relay 127.0.0.1:8765

    # terminal 1 again (or another)
    python scripts/live.py --source serial --port /dev/pts/7

Direction is fully bidirectional so the laptop->device STATE:/SCORE: feedback
path keeps working through the bridge.
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import socket
import time
import tty


def main() -> int:
    ap = argparse.ArgumentParser(description="socket <-> pty relay for COM-over-WSL")
    ap.add_argument("--listen", default="127.0.0.1:8765",
                    help="host:port to listen on (WSL reaches Windows via the "
                         "same loopback when Windows connects out to us)")
    ap.add_argument("--stats", type=float, default=5.0,
                    help="seconds between traffic stats lines (0 = off)")
    args = ap.parse_args()

    host, port_s = args.listen.rsplit(":", 1)
    port = int(port_s)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[relay] listening on {host}:{port} — waiting for the Windows bridge…",
          flush=True)

    conn, addr = srv.accept()
    print(f"[relay] bridge connected from {addr}", flush=True)

    master, slave = pty.openpty()
    tty.setraw(master)
    tty.setraw(slave)          # no echo / no \n->\r\n mangling
    pty_path = os.ttyname(slave)
    print(f"PTY={pty_path}", flush=True)

    up = down = 0             # bytes: device->laptop(python), laptop->device
    last_stats = time.time()

    try:
        while True:
            r, _, _ = select.select([conn, master], [], [], 1.0)
            for fd in r:
                if fd is conn:
                    try:
                        data = conn.recv(4096)
                    except OSError:
                        data = b""
                else:
                    try:
                        data = os.read(fd, 4096)   # pty master is a real fd
                    except OSError:
                        data = b""
                if not data:
                    print("[relay] connection closed", flush=True)
                    return 0
                if fd == conn:                     # python -> device
                    os.write(master, data)
                    down += len(data)
                else:                              # device -> python
                    conn.sendall(data)
                    up += len(data)
            if args.stats and time.time() - last_stats >= args.stats:
                print(f"[relay] bytes device->host={up} host->device={down}",
                      flush=True)
                last_stats = time.time()
    finally:
        conn.close()
        srv.close()


if __name__ == "__main__":
    raise SystemExit(main())

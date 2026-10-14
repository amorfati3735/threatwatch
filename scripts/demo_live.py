#!/usr/bin/env python3
"""ThreatWatch one-command hardware demo.

Usage:  python scripts/demo_live.py [--wired]

Starts relay + correct bridge + live pipeline over the live board and saves
annotated CSV. Stop everything with Ctrl+C.

Board prep (before running): ESP32 + 3 I2C modules on a power bank (wired mode:
USB to this laptop instead), Windows Bluetooth ON with "ThreatWatch" advertised,
finger on MAX30102 during the demo.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINPY_ABS = "/mnt/c/Users/DELL/AppData/Local/Programs/Python/Python310/python.exe"
TWPY = "/home/pratik/.venvs/threatwatch/bin/python"


def spawn(cmd, log, name):
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)   # own process group -> clean kill
    print(f"[{name}] pid={p.pid} -> {log}", flush=True)
    return p, f


def wait_pty(log, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(log) as f:
                m = re.search(r"PTY=(/dev/pts/\d+)", f.read())
                if m:
                    return m.group(1)
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit("relay did not print a PTY path -- check relay.log")


def main() -> int:
    wired = "--wired" in sys.argv[1:]
    mode = "WIRED (COM11)" if wired else "BLE (power bank, ThreatWatch)"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv = os.path.join("logs", f"demo_{stamp}.csv")
    os.makedirs("logs", exist_ok=True)
    bridge = "windows_serial_bridge.py" if wired else "windows_ble_bridge.py"
    print(f"=== ThreatWatch demo: {mode} -> {csv} ===", flush=True)

    kids = []
    try:
        relay, _ = spawn([TWPY, "scripts/ws_pty_relay.py"], "/tmp/demo_relay.log", "relay")
        kids.append(relay)
        if wired:
            bridge_cmd = [WINPY_ABS, "C:\\Users\\DELL\\threatwatch\\scripts\\windows_serial_bridge.py",
                          "--port", "COM11"]
            win, _ = spawn(bridge_cmd, "/tmp/demo_bridge.log", "bridge")
        else:
            bridge_cmd = [WINPY_ABS, "C:\\Users\\DELL\\threatwatch\\scripts\\windows_ble_bridge.py"]
            win, _ = spawn(bridge_cmd, "/tmp/demo_bridge.log", "bridge")
        kids.append(win)
        time.sleep(6 if wired else 12)   # BLE scan/connect takes longer
        got = wait_pty("/tmp/demo_relay.log")
        print(f"[relay] PTY={got}", flush=True)

        live, _ = spawn([TWPY, "-u", "scripts/live.py", "--source", "serial",
                         "--port", got, "--log", csv], "/tmp/demo_live.log", "live")
        kids.append(live)
        print("=== LIVE — Ctrl+C to stop ===", flush=True)
        while True:
            if live.poll() is not None:
                print("live.py exited; see /tmp/demo_live.log", flush=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
    for k in reversed(kids):
        try:
            os.killpg(os.getpgid(k.pid), signal.SIGTERM)
        except OSError:
            pass
    time.sleep(1)
    for k in kids:
        if k.poll() is None:
            try:
                os.killpg(os.getpgid(k.pid), signal.SIGKILL)
            except OSError:
                pass
    print(f"demo CSV saved: {csv} (replay: scripts/live.py --source replay --file {csv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

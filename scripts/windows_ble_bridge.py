#!/usr/bin/env python3
"""
Windows side of the BLE bridge: ESP32 "ThreatWatch" (NUS) <-> TCP relay.

Lets the wearable run untethered — powered from a power bank, talking to the
laptop over Bluetooth LE — while the host pipeline stays identical:

    ESP32 (BLE, battery) ──> windows_ble_bridge.py (Windows + bleak)
                         <──  TCP ──> ws_pty_relay.py (WSL pty) ──> live.py

Run with Windows Python (the side that has the Bluetooth radio):

    python scripts\\windows_ble_bridge.py --relay 127.0.0.1:8765

Same relay as the wired path, so scripts/ws_pty_relay.py and
scripts/live.py --source serial --port /dev/pts/N are reused unchanged.
The NUS RX characteristic still carries the STATE:/SCORE: feedback upstream.
"""

from __future__ import annotations

import argparse
import asyncio

NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"   # laptop -> device (write)
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # device -> laptop (notify)
DEVICE_NAME = "ThreatWatch"


async def connect_relay(host: str, port: int):
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            print(f"[ble-bridge] relay connected {host}:{port}", flush=True)
            return reader, writer
        except OSError as exc:
            print(f"[ble-bridge] relay not reachable ({exc}); retrying…", flush=True)
            await asyncio.sleep(2)


async def run(args) -> int:
    from bleak import BleakClient, BleakScanner

    print(f"[ble-bridge] scanning for '{DEVICE_NAME}' …", flush=True)
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: (ad.local_name or "") == DEVICE_NAME, timeout=20
    )
    if device is None:
        print("[ble-bridge] device not found — is the board powered and near by?",
              flush=True)
        return 1
    print(f"[ble-bridge] found {device.address}", flush=True)

    reader, writer = await connect_relay(*args.relay)

    async with BleakClient(device) as client:
        print(f"[ble-bridge] connected to {device.address}  (ATT MTU={client.mtu_size})",
              flush=True)

        def on_notify(_sender, data: bytearray) -> None:
            # Each notification is one CSV line WITHOUT a trailing newline
            # (Serial.println adds it only on USB) — add it for the pty reader.
            buf = bytes(data)
            if not buf.endswith(b"\n"):
                buf += b"\n"
            try:
                writer.write(buf)          # device -> pipeline
            except OSError:
                pass

        await client.start_notify(NUS_TX_UUID, on_notify)
        print("[ble-bridge] streaming… (Ctrl+C to stop)", flush=True)

        try:
            while True:
                chunk = await reader.read(4096)   # pipeline -> device (STATE/SCORE)
                if not chunk:
                    print("[ble-bridge] relay closed", flush=True)
                    break
                await client.write_gatt_char(NUS_RX_UUID, chunk)
        except asyncio.CancelledError:
            raise
        finally:
            try:
                await client.stop_notify(NUS_TX_UUID)
            except Exception:
                pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ESP32 BLE NUS <-> TCP relay bridge")
    ap.add_argument("--relay", default="127.0.0.1:8765", help="host:port of ws_pty_relay.py")
    args = ap.parse_args()
    host, port_s = args.relay.rsplit(":", 1)
    ns = argparse.Namespace(relay=(host, int(port_s)))
    try:
        return asyncio.run(run(ns))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

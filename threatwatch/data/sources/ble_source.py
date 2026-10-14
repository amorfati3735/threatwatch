"""
BLE sensor source for the ESP32 firmware (Nordic UART Service).

Subscribes to the NUS TX characteristic and parses the identical CSV format
used by the serial source, so the ESP32's wireless stream enters the pipeline
exactly like the wired Arduino stream.

Requires ``bleak`` (cross-platform BLE, works on Windows via WinRT).
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Optional

from threatwatch.data.sources.serial_source import parse_line
from threatwatch.data.types import SensorPacket, SensorSource

NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
DEVICE_NAME = "ThreatWatch"


class BluetoothSensorSource(SensorSource):
    name = "ble"

    def __init__(self, device_name: str = DEVICE_NAME, address: Optional[str] = None):
        self.device_name = device_name
        self.address = address
        self._q: "queue.Queue[SensorPacket]" = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

    def is_running(self) -> bool:
        return self._running

    def get_next_sample(self) -> Optional[SensorPacket]:
        try:
            return self._q.get(timeout=0.2)
        except queue.Empty:
            return None

    def _run(self) -> None:
        try:
            import bleak  # noqa: F401
        except ImportError:
            print("[ble] 'bleak' not installed — run: pip install bleak")
            self._running = False
            return

        from bleak import BleakClient, BleakScanner

        async def main():
            self._loop = asyncio.get_running_loop()
            print(f"[ble] scanning for '{self.device_name}' …")
            device = None
            if self.address:
                device = await BleakScanner.find_device_by_address(self.address, timeout=10)
            if device is None:
                device = await BleakScanner.find_device_by_filter(
                    lambda d, ad: (ad.local_name or "") == self.device_name, timeout=15
                )
            if device is None:
                print("[ble] device not found")
                self._running = False
                return

            def on_notify(_sender, data: bytearray):
                pkt = parse_line(data.decode("utf-8", errors="ignore"))
                if pkt is not None:
                    pkt.source = "ble"
                    self._q.put(pkt)

            async with BleakClient(device) as client:
                print(f"[ble] connected to {device.address}")
                await client.start_notify(NUS_TX_UUID, on_notify)
                while self._running:
                    await asyncio.sleep(0.2)
                await client.stop_notify(NUS_TX_UUID)
            self._running = False

        try:
            asyncio.run(main())
        except Exception as e:  # keep the UI alive on disconnect
            print(f"[ble] connection error: {e}")
            self._running = False

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
from typing import Callable, Optional, Tuple

from threatwatch.data.sources.serial_source import parse_line
from threatwatch.data.types import SensorPacket, SensorSource

NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
DEVICE_NAME = "ThreatWatch"


class BluetoothSensorSource(SensorSource):
    name = "ble"

    def __init__(
        self,
        device_name: str = DEVICE_NAME,
        address: Optional[str] = None,
        on_raw_line: Optional[Callable[[str, Optional[SensorPacket]], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
        auto_reconnect: bool = True,
    ):
        self.device_name = device_name
        self.address = address
        self.on_raw_line = on_raw_line
        self.on_state_change = on_state_change
        self.auto_reconnect = auto_reconnect

        self._q: "queue.Queue[SensorPacket]" = queue.Queue()
        self._raw_q: "queue.Queue[Tuple[str, Optional[SensorPacket]]]" = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.accepted_count = 0
        self.rejected_count = 0
        self.last_address: Optional[str] = None

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

    def get_next_record(self, timeout: float = 0.2) -> Optional[Tuple[str, Optional[SensorPacket]]]:
        try:
            return self._raw_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _notify_state(self, msg: str) -> None:
        if self.on_state_change is not None:
            try:
                self.on_state_change(msg)
            except Exception:
                pass
        else:
            print(f"[ble] {msg}")

    def _run(self) -> None:
        try:
            import bleak  # noqa: F401
        except ImportError:
            self._notify_state("'bleak' not installed - run: pip install bleak")
            self._running = False
            return

        from bleak import BleakClient, BleakScanner

        async def main():
            self._loop = asyncio.get_running_loop()

            def on_notify(_sender, data: bytearray):
                raw = data.decode("utf-8", errors="ignore").strip()
                pkt = parse_line(raw)
                if pkt is not None:
                    pkt.source = "ble"
                    self.accepted_count += 1
                    self._q.put(pkt)
                else:
                    self.rejected_count += 1
                self._raw_q.put((raw, pkt))
                if self.on_raw_line is not None:
                    try:
                        self.on_raw_line(raw, pkt)
                    except Exception:
                        pass

            while self._running:
                self._notify_state(f"scanning for '{self.device_name}'...")
                device = None
                try:
                    if self.address:
                        device = await BleakScanner.find_device_by_address(self.address, timeout=10)
                    if device is None:
                        device = await BleakScanner.find_device_by_filter(
                            lambda d, ad: (ad.local_name or "") == self.device_name, timeout=15
                        )
                except Exception as exc:
                    self._notify_state(f"scan error: {exc}")

                if device is None:
                    if not self._running:
                        break
                    self._notify_state("device not found")
                    if not self.auto_reconnect:
                        break
                    await asyncio.sleep(2.0)
                    continue

                self.last_address = device.address
                self._notify_state(f"connected {device.address}")
                try:
                    async with BleakClient(device) as client:
                        await client.start_notify(NUS_TX_UUID, on_notify)
                        while self._running and client.is_connected:
                            await asyncio.sleep(0.2)
                        try:
                            await client.stop_notify(NUS_TX_UUID)
                        except Exception:
                            pass
                except Exception as e:
                    if not self._running:
                        break
                    self._notify_state(f"connection error: {e}")

                if self._running and self.auto_reconnect:
                    self._notify_state("link lost - reconnecting")
                    await asyncio.sleep(1.0)
                else:
                    break
            self._running = False

        try:
            asyncio.run(main())
        except Exception as e:
            self._notify_state(f"connection error: {e}")
            self._running = False

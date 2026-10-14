"""
Live hardware source for the physical wearable (Arduino Uno or ESP32 over USB).

This replaces the predecessor project's `ArduinoSensorSource` and its ESP32
stub, which between them (a) invented sensor values when a field was absent and
(b) offered a `connect()` that always returned ``False``.  Here:

* parsing is delegated to the single canonical CSV parser in
  :mod:`threatwatch.data.sources.serial_source`, so a live board and a recorded
  log take exactly the same path;
* an absent reading stays ``None`` and is never replaced by a plausible default;
* the connection is real, non-blocking, and survives a board reset.

Threading model: a daemon reader thread pulls lines and parks the newest packet
in a lock-protected slot.  The async pipeline polls :meth:`get_next_sample` and
therefore never blocks on the serial port.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

from threatwatch.config import (
    BLE_DEVICE_NAME,
    BLE_NUS_RX_UUID,
    BLE_NUS_TX_UUID,
    BLE_PORT_ALIAS,
    DEFAULT_SERIAL_BAUD,
    DEFAULT_SERIAL_PORT,
    DEVICE_SCORE_PREFIX,
    DEVICE_STATE_PREFIX,
)
from threatwatch.data.sources.serial_source import parse_line
from threatwatch.data.types import SensorPacket, SensorSource

try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except Exception:  # pragma: no cover - pyserial is an optional runtime dep
    SERIAL_AVAILABLE = False

try:
    import bleak  # noqa: F401

    BLE_AVAILABLE = True
except Exception:  # pragma: no cover - bleak is an optional runtime dep
    BLE_AVAILABLE = False


class HardwareSensorSource(SensorSource):
    """Bidirectional USB-serial link to the physical wearable."""

    name = "hardware"

    def __init__(
        self,
        port: str = DEFAULT_SERIAL_PORT,
        baud: int = DEFAULT_SERIAL_BAUD,
        stale_after_sec: float = 2.0,
    ):
        self.port = port
        self.baud = int(baud)
        self.stale_after_sec = float(stale_after_sec)
        self._transport = "serial"

        self._serial: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._latest: Optional[SensorPacket] = None
        self._last_packet_time = 0.0
        self._packets_received = 0
        self._connected = False
        self._error = ""

        # BLE transport state (only used when _transport == "ble")
        self._ble_client: Optional[Any] = None
        self._ble_loop: Optional[Any] = None

    # -- discovery --------------------------------------------------------- #
    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """List serial ports with a human-readable description."""
        out: List[Dict[str, str]] = []
        if SERIAL_AVAILABLE:
            try:
                for p in serial.tools.list_ports.comports():
                    out.append({"port": p.device, "description": p.description or "", "hwid": p.hwid or ""})
            except Exception as exc:
                print(f"[hardware] port enumeration failed: {exc}")
        if BLE_AVAILABLE:
            out.append({
                "port": BLE_PORT_ALIAS,
                "description": "ThreatWatch wearable (Bluetooth LE, untethered)",
                "hwid": "ble",
            })
        return out

    # -- lifecycle --------------------------------------------------------- #
    def connect(self, port: Optional[str] = None) -> bool:
        if port:
            self.port = port
        requested = "ble" if (self.port or "").strip().upper() == BLE_PORT_ALIAS else "serial"
        if requested != self._transport:
            self.disconnect()  # switch transport cleanly
            self._transport = requested

        if requested == "ble":
            return self._connect_ble()

        if not SERIAL_AVAILABLE:
            self._error = "pyserial is not installed"
            return False

        with self._lock:
            if self._connected and self._serial is not None and getattr(self._serial, "is_open", False):
                return True
            try:
                self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
                # Many Arduino boards reset when the port opens; wait it out.
                time.sleep(1.5)
                try:
                    self._serial.reset_input_buffer()
                except Exception:
                    pass
                self._connected = True
                self._error = ""
                print(f"[hardware] connected to {self.port} @ {self.baud} baud")
            except Exception as exc:
                self._connected = False
                self._error = str(exc)
                print(f"[hardware] connection to {self.port} failed: {exc}")
                return False

        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._reader_loop, name="hw-reader", daemon=True)
            self._thread.start()
        return True

    def _connect_ble(self) -> bool:
        """Start the wireless link.  Returns immediately; ``is_connected``
        flips true once the BLE client is actually up (the UI polls state)."""
        if not BLE_AVAILABLE:
            self._error = "bleak is not installed"
            return False
        self._error = ""
        print(f"[hardware] BLE mode: will scan for '{BLE_DEVICE_NAME}' in the background")
        self._stop.clear()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._reader_loop, name="hw-reader", daemon=True)
            self._thread.start()
        return True

    def disconnect(self) -> None:
        self._stop.set()
        with self._lock:
            self._connected = False
            if self._serial is not None:
                try:
                    if getattr(self._serial, "is_open", False):
                        self._serial.close()
                except Exception:
                    pass
            self._serial = None
            self._ble_client = None
            self._ble_loop = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        print(f"[hardware] disconnected from {self.port}")

    def start(self) -> None:
        self.connect()

    def stop(self) -> None:
        self.disconnect()

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._last_packet_time = 0.0
            self._packets_received = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def is_running(self) -> bool:
        return self._connected

    # -- reader ------------------------------------------------------------ #
    def _reader_loop(self) -> None:
        if self._transport == "ble":
            try:
                asyncio.run(self._ble_main())
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[hardware] BLE link error: {exc}")
                with self._lock:
                    self._connected = False
                    self._error = str(exc)
            return
        while not self._stop.is_set():
            with self._lock:
                ser = self._serial
                connected = self._connected
            if not connected or ser is None:
                time.sleep(0.2)
                continue
            try:
                raw = ser.readline().decode("utf-8", errors="ignore")
                if not raw:
                    continue
                self._park_packet(raw)
            except Exception as exc:
                # A transient serial error must not kill the thread permanently;
                # drop the link and let the UI reconnect.
                print(f"[hardware] serial error: {exc}")
                with self._lock:
                    self._connected = False
                    self._error = str(exc)
                time.sleep(0.5)

    def _park_packet(self, raw: str) -> None:
        pkt = parse_line(raw)
        if pkt is None:
            return
        pkt.source = "hardware"
        with self._lock:
            self._latest = pkt
            self._last_packet_time = time.time()
            self._packets_received += 1

    async def _ble_main(self) -> None:
        """Scan → connect → stream notifications, reconnecting until stopped.

        Each NUS TX notification is one CSV line (the ESP32 sends a whole
        ``Serial.println`` per notification), so no reassembly is needed — the
        trailing newline is just added for the canonical parser.
        """
        while not self._stop.is_set():
            with self._lock:
                self._error = "scanning…"
            print(f"[hardware] BLE: scanning for '{BLE_DEVICE_NAME}' …")
            device = None
            try:
                from bleak import BleakClient, BleakScanner

                device = await BleakScanner.find_device_by_filter(
                    lambda d, ad: (ad.local_name or "") == BLE_DEVICE_NAME, timeout=10.0
                )
            except Exception as exc:
                self._error = f"BLE scan failed: {exc}"
            if device is None:
                if self._stop.wait(3.0):
                    return
                continue
            try:
                async with BleakClient(device) as client:
                    loop = asyncio.get_running_loop()
                    with self._lock:
                        self._ble_client = client
                        self._ble_loop = loop
                        self._connected = True
                        self._error = ""
                    print(f"[hardware] BLE connected to {device.address}")

                    def on_notify(_sender, data: bytearray) -> None:
                        raw = bytes(data)
                        if not raw.endswith(b"\n"):
                            raw += b"\n"
                        for line in raw.split(b"\n"):
                            if line.strip():
                                self._park_packet(line.decode("utf-8", errors="ignore"))

                    await client.start_notify(BLE_NUS_TX_UUID, on_notify)
                    # Park here until the link drops or we are stopping.
                    while not self._stop.is_set() and client.is_connected:
                        await asyncio.sleep(0.5)
                    try:
                        await client.stop_notify(BLE_NUS_TX_UUID)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[hardware] BLE connect failed: {exc}")
                self._error = str(exc)
            finally:
                with self._lock:
                    self._connected = False
                    self._ble_client = None
                    self._ble_loop = None
            if not self._stop.is_set():
                print("[hardware] BLE link lost — reconnecting…")
                await asyncio.sleep(2.0)

    # -- SensorSource ------------------------------------------------------ #
    def get_next_sample(self) -> Optional[SensorPacket]:
        with self._lock:
            return self._latest

    def write_feedback(self, threat_score_pct: int, state_str: str) -> bool:
        """Mirror the fused verdict onto the wearable's OLED."""
        with self._lock:
            ser = self._serial
            ble_client = self._ble_client
            ble_loop = self._ble_loop
            connected = self._connected
            transport = self._transport
        if not connected:
            return False
        try:
            pct = int(max(0, min(100, round(float(threat_score_pct)))))
            safe = str(state_str).replace("\n", " ").replace("\r", " ")[:24]
            payload = f"{DEVICE_STATE_PREFIX}{safe}\n{DEVICE_SCORE_PREFIX}{pct}\n".encode("utf-8")
            if transport == "ble":
                if ble_client is None or ble_loop is None:
                    return False
                # Fire-and-forget: a dropped link surfaces in the reader loop.
                asyncio.run_coroutine_threadsafe(
                    ble_client.write_gatt_char(BLE_NUS_RX_UUID, payload), ble_loop
                )
                return True
            if ser is None:
                return False
            ser.write(payload)
            return True
        except Exception as exc:
            print(f"[hardware] feedback write failed: {exc}")
            return False

    def get_state(self) -> Dict:
        with self._lock:
            age = (time.time() - self._last_packet_time) if self._last_packet_time > 0 else None
            return {
                "source": self.name,
                "transport": self._transport,
                "connected": self._connected,
                "running": self._connected,
                "port": self.port,
                "baud": self.baud,
                "packets_received": self._packets_received,
                "last_packet_age_s": round(age, 2) if age is not None else None,
                "is_stale": (age is None) or (age > self.stale_after_sec),
                "error": self._error,
                "serial_available": SERIAL_AVAILABLE,
                "ble_available": BLE_AVAILABLE,
            }

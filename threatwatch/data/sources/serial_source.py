"""
Live serial sensor source for Arduino Uno / ESP32 firmware.

The firmware emits one CSV line per sample::

    ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label

e.g. ``12.345,78,98,,33.4,0.01,-0.02,0.99,1.2,0.3,-0.5,1,-1``

``ts`` is **seconds** since boot (float), so it is on the same time base as the
WESAD replay and the recorded-log replay.

Empty fields mean "sensor did not report this" and are kept as ``None`` — no
value is invented.  A JSON-lines fallback is also accepted.

A record must be *whole*.  A line that arrives short (a record split mid-
transmission by a ``readline()`` timeout, or a stray value) has had its leading
fields cut off, so every remaining value is shifted one channel to the left and
the first field is a fragment of the real timestamp.  Such a line is rejected
rather than turned into a sample that never existed.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from threatwatch.config import (
    DEFAULT_SERIAL_BAUD,
    DEVICE_SCORE_PREFIX,
    DEVICE_STATE_PREFIX,
)
from threatwatch.data.types import SensorPacket, SensorSource

CSV_FIELDS = ["hr", "spo2", "eda", "temp", "ax", "ay", "az", "gx", "gy", "gz", "finger", "label"]

# ts + every payload field.  A shorter line is a fragment of a record (see the
# module docstring), never a valid sample.
MIN_CSV_FIELDS = len(CSV_FIELDS) + 1


def _num(s: str) -> Optional[float]:
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_line(line: str) -> Optional[SensorPacket]:
    """Parse one CSV or JSON line into a SensorPacket (None if unusable)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("{"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            return None
        return SensorPacket(
            timestamp=float(d.get("timestamp", d.get("t", time.time()))),
            heart_rate=d.get("heart_rate", d.get("hr")),
            spo2=d.get("spo2"),
            eda=d.get("eda"),
            temperature=d.get("temperature", d.get("temp")),
            acc_x=d.get("ax", d.get("acc_x")),
            acc_y=d.get("ay", d.get("acc_y")),
            acc_z=d.get("az", d.get("acc_z")),
            gyro_x=d.get("gx", d.get("gyro_x")),
            gyro_y=d.get("gy", d.get("gyro_y")),
            gyro_z=d.get("gz", d.get("gyro_z")),
            finger_detected=bool(d.get("finger_detected", d.get("finger", True))),
            label=int(d.get("label", -1)),
            source="serial",
        )

    parts = line.split(",")
    if len(parts) < MIN_CSV_FIELDS:
        # Truncated record: its fields are shifted, so it is not a sample.
        return None
    # If the line has a header-like token, skip it.
    if not parts[0].strip().replace(".", "", 1).lstrip("-").isdigit():
        return None

    ts = _num(parts[0])
    if ts is None:
        return None
    vals = [_num(p) for p in parts[1:1 + len(CSV_FIELDS)]]
    vals += [None] * (len(CSV_FIELDS) - len(vals))
    finger_raw = vals[10]
    label = int(vals[11]) if vals[11] is not None else -1
    return SensorPacket(
        timestamp=ts,
        heart_rate=vals[0], spo2=vals[1], eda=vals[2], temperature=vals[3],
        acc_x=vals[4], acc_y=vals[5], acc_z=vals[6],
        gyro_x=vals[7], gyro_y=vals[8], gyro_z=vals[9],
        finger_detected=(None if finger_raw is None else finger_raw >= 0.5),
        label=label,
        source="serial",
    )


class SerialSensorSource(SensorSource):
    """
    Reads timestamped packets from a serial (USB) port and can write the fused
    verdict back to the device for its OLED.

    Upstream (device -> laptop):  ``ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label``
    Downstream (laptop -> device): two short lines,
        ``STATE:<name>`` and ``SCORE:<0-100>``
    which keeps the ATmega328P-side parser tiny and preserves the original
    ``STATE:`` contract.
    """

    name = "serial"

    def __init__(self, port: str = "COM3", baud: int = DEFAULT_SERIAL_BAUD, timeout: float = 0.1):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial = None
        self._running = False
        self._last: Optional[SensorPacket] = None

    def start(self) -> None:
        import serial  # pyserial, imported lazily so the package works headless

        self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Many Arduino boards reset on serial open; give them a moment.
        time.sleep(1.5)
        self._serial.reset_input_buffer()
        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def is_running(self) -> bool:
        return self._running

    def get_next_sample(self) -> Optional[SensorPacket]:
        if not self._running or self._serial is None:
            return None
        try:
            raw = self._serial.readline().decode("utf-8", errors="ignore")
        except Exception:
            return None
        pkt = parse_line(raw)
        if pkt is not None:
            self._last = pkt
        return pkt

    @property
    def is_connected(self) -> bool:
        return bool(self._running and self._serial is not None)

    def write_feedback(self, threat_score_pct: int, state_str: str) -> bool:
        """Send the fused verdict to the device OLED.  Never raises."""
        if not self.is_connected:
            return False
        try:
            pct = int(max(0, min(100, round(float(threat_score_pct)))))
            safe_state = str(state_str).replace("\n", " ").replace("\r", " ")[:24]
            payload = f"{DEVICE_STATE_PREFIX}{safe_state}\n{DEVICE_SCORE_PREFIX}{pct}\n"
            self._serial.write(payload.encode("utf-8"))
            return True
        except Exception as exc:  # a broken link must never kill the pipeline
            print(f"[serial] feedback write failed: {exc}")
            return False

    def get_state(self) -> dict:
        return {"source": self.name, "running": self._running, "port": self.port,
                "baud": self.baud, "connected": self.is_connected}

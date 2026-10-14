"""
WESAD replay source.

Streams a real WESAD subject's synchronized wrist signals as ``SensorPacket``s
at the common target rate, exposing the dataset's ground-truth label.  Because
it implements the same interface as the serial source, the identical model and
downstream code consume both.

Note: WESAD has no gyroscope and no SpO2, so those packet fields are ``None``
(never fabricated).  EDA is present in WESAD's wrist device even though our
current hardware lacks a GSR sensor — that is precisely the gap the
hardware-only feature subset quantifies.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from threatwatch.config import WESAD_DIR
from threatwatch.data.types import SensorPacket, SensorSource
from threatwatch.data.wesad import extract_subject


class WESADSource(SensorSource):
    name = "wesad"

    def __init__(self, subject: str, speed: float = 1.0, data_dir: Path = WESAD_DIR,
                 loop: bool = True):
        self.subject = subject
        self.speed = max(speed, 0.01)
        self.data_dir = data_dir
        self.loop = loop
        self._data: Optional[dict] = None
        self._i = 0
        self._running = False
        self._t_wall0 = 0.0

    def _ensure(self) -> None:
        if self._data is None:
            d = extract_subject(self.subject, self.data_dir)
            if d is None:
                raise FileNotFoundError(f"WESAD subject {self.subject} not found")
            self._data = d

    def start(self) -> None:
        self._ensure()
        self._i = 0
        self._t_wall0 = time.time()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def get_next_sample(self) -> Optional[SensorPacket]:
        if not self._running:
            return None
        self._ensure()
        d = self._data
        n = d["n"]
        if self._i >= n:
            if not self.loop:
                self._running = False
                return None
            self._i = 0
            self._t_wall0 = time.time()

        i = self._i
        self._i += 1
        # pace to real time
        target_wall = self._t_wall0 + (i / d["sampling_rate"]) / self.speed
        dt = target_wall - time.time()
        if 0 < dt < 0.5:
            time.sleep(dt)

        lbl = int(d["label"][i])
        return SensorPacket(
            timestamp=float(i / d["sampling_rate"]),
            heart_rate=float(d["hr"][i]),
            spo2=None,          # not in WESAD
            eda=float(d["eda"][i]),
            temperature=float(d["temp"][i]),
            acc_x=float(d["acc_x"][i]),
            acc_y=float(d["acc_y"][i]),
            acc_z=float(d["acc_z"][i]),
            gyro_x=None, gyro_y=None, gyro_z=None,   # not in WESAD
            finger_detected=True,
            label=lbl,
            source="wesad",
        )

    def get_state(self) -> dict:
        return {"source": self.name, "running": self._running, "subject": self.subject,
                "index": self._i, "speed": self.speed}

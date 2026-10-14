"""
Replay a recorded hardware CSV log (produced by ``scripts/live.py --log``)
through the same ``SensorSource`` interface, so recorded physical sessions can
be re-run offline and fed to the model exactly like live data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from threatwatch.data.sources.serial_source import parse_line
from threatwatch.data.types import SensorPacket, SensorSource


class ReplaySource(SensorSource):
    """Replays a CSV log, preserving real inter-sample timing scaled by ``speed``."""

    name = "replay"

    def __init__(self, csv_path: Path, speed: float = 1.0, loop: bool = False):
        self.csv_path = Path(csv_path)
        self.speed = max(speed, 0.01)
        self.loop = loop
        self._packets: List[SensorPacket] = []
        self._i = 0
        self._running = False
        self._t_prev: Optional[float] = None

    def _load(self) -> None:
        self._packets = []
        with open(self.csv_path) as f:
            for line in f:
                pkt = parse_line(line)
                if pkt is not None:
                    self._packets.append(pkt)

    def start(self) -> None:
        if not self._packets:
            self._load()
        self._i = 0
        self._t_prev = None
        self._running = True

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def get_next_sample(self) -> Optional[SensorPacket]:
        if not self._running or not self._packets:
            return None
        if self._i >= len(self._packets):
            if not self.loop:
                self._running = False
                return None
            self._i = 0
            self._t_prev = None

        pkt = self._packets[self._i]
        self._i += 1
        # Sleep to preserve the original timing (relative to previous packet).
        if self._t_prev is not None:
            dt = (pkt.timestamp - self._t_prev) / self.speed
            if 0 < dt < 5.0:
                time.sleep(dt)
        self._t_prev = pkt.timestamp
        return pkt

    def get_state(self) -> dict:
        return {"source": self.name, "running": self._running,
                "file": str(self.csv_path), "index": self._i, "total": len(self._packets)}

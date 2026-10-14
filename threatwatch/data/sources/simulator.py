"""
Synthetic sensor source — for testing the live pipeline with no hardware.

This is *deliberately* separate from real data: the source name is ``simulator``
and every packet is labelled ``source="simulator"`` so it can never be mistaken
for a real sensor reading.  It generates plausible HR/accel trajectories for
baseline / stress / amusement so the dashboard and model plumbing can be tested.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from threatwatch.data.types import SensorPacket, SensorSource

# (hr_mean, hr_std, accel_std, spo2_mean)
_STATE_PROFILES = {
    0: {"name": "baseline", "hr": 70.0, "hr_std": 3.0, "acc": 0.02, "spo2": 98.0},
    1: {"name": "stress", "hr": 100.0, "hr_std": 6.0, "acc": 0.05, "spo2": 97.0},
    2: {"name": "amusement", "hr": 82.0, "hr_std": 5.0, "acc": 0.12, "spo2": 98.0},
}


class SimulatorSource(SensorSource):
    """Deterministic-ish physiological simulator with a requested ground-truth state."""

    name = "simulator"

    def __init__(self, state: int = 0, hz: float = 15.0, seed: int = 0):
        self.state = state
        self.hz = hz
        self.rng = np.random.default_rng(seed)
        self._running = False
        self._t0 = 0.0
        self._k = 0

    def start(self) -> None:
        self._running = True
        self._t0 = time.time()
        self._k = 0

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def set_state(self, state: int) -> None:
        self.state = int(state) % 3

    def get_next_sample(self) -> Optional[SensorPacket]:
        if not self._running:
            return None
        profile = _STATE_PROFILES[self.state]
        t = time.time()
        phase = self._k / self.hz
        self._k += 1

        hr = profile["hr"] + 6.0 * math.sin(2 * math.pi * 0.05 * phase) \
            + float(self.rng.normal(0, profile["hr_std"] * 0.3))
        # respiration-modulated accelerometer
        acc = [float(self.rng.normal(0, profile["acc"])) for _ in range(3)]
        acc[2] += 1.0  # gravity on Z (typical flat-on-wrist mounting)
        gyro = [float(self.rng.normal(0, 5.0)) for _ in range(3)]

        return SensorPacket(
            timestamp=t,
            heart_rate=float(np.clip(hr, 40, 200)),
            spo2=float(profile["spo2"] + self.rng.normal(0, 0.3)),
            eda=None,  # no EDA sensor in the simulator (hardware has none)
            temperature=float(33.5 + self.rng.normal(0, 0.05)),
            acc_x=acc[0], acc_y=acc[1], acc_z=acc[2],
            gyro_x=gyro[0], gyro_y=gyro[1], gyro_z=gyro[2],
            finger_detected=True,
            label=self.state,
            source="simulator",
        )

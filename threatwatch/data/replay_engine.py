"""
Deterministic WESAD replay engine (server-side).

Design note — why this is not the same thing as ``sources/wesad_source.py``:

* :class:`~threatwatch.data.sources.wesad_source.WESADSource` *paces itself to
  wall-clock time* (sleeping between samples).  It is the right thing for the
  ``scripts/live.py`` CLI where you want to watch a subject stream in real time.
* This engine is *driven by the pipeline tick*.  The server asks for the current
  sample and then advances by a computed number of samples, so playback speed,
  seeking and scenario jumps are exact and frame-rate independent.

Crucially, this engine does **not** build windows itself.  It only yields packets;
the single :class:`~threatwatch.data.types.WindowBuffer` in the pipeline builds
every window, for replay and live hardware alike.  The predecessor project
sliced its own window and, at startup, concatenated the end of the recording
with the beginning, handing the model a time-discontinuous window.  That cannot
happen here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from threatwatch.config import (
    CLASS_NAMES,
    DEFAULT_WINDOW,
    TARGET_FS,
    WESAD_DIR,
)
from threatwatch.data.types import SensorPacket, SensorSource
from threatwatch.data.wesad import available_subjects, extract_subject


class ReplayEngine(SensorSource):
    """Streams a real WESAD subject's wrist signals as normalised packets."""

    name = "wesad_replay"

    def __init__(
        self,
        subject_id: str = "S2",
        speed: float = 1.0,
        data_dir: Path = WESAD_DIR,
        target_fs: float = TARGET_FS,
    ):
        self.data_dir = Path(data_dir)
        self.target_fs = float(target_fs)
        self.subject_id = subject_id
        self.speed = max(0.05, float(speed))

        self._data: Optional[Dict[str, np.ndarray]] = None
        self._index = 0
        self._total = 0
        self._running = True
        self._packets_sent = 0
        self._load_error = ""
        # Fractional-sample accumulator so playback speed is time-accurate
        # instead of being quantised to whole samples per telemetry tick.
        self._pending = 0.0

        self.load_subject(subject_id)

    # ------------------------------------------------------------------ #
    def list_subjects(self) -> List[str]:
        return available_subjects(self.data_dir)

    def load_subject(self, subject_id: str) -> bool:
        """Load (and cache) one subject.  Returns False with an error message on failure."""
        data = extract_subject(subject_id, self.data_dir, target_fs=self.target_fs)
        if data is None:
            self._load_error = (
                f"subject {subject_id} not found under {self.data_dir}. "
                f"Place S*/S*.pkl there, or run scripts/download_wesad.py."
            )
            print(f"[replay] {self._load_error}")
            return False

        self.subject_id = subject_id
        self._data = data
        self._total = int(data["n"])
        self._index = 0
        self._packets_sent = 0
        self._load_error = ""
        minutes = self._total / self.target_fs / 60.0
        print(f"[replay] loaded {subject_id}: {self._total} samples ({minutes:.1f} min @ {self.target_fs:g} Hz)")
        return True

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        self._index = 0
        self._packets_sent = 0
        self._pending = 0.0

    def is_running(self) -> bool:
        return self._running

    @property
    def is_loaded(self) -> bool:
        return self._data is not None and self._total > 0

    # ------------------------------------------------------------------ #
    def set_speed(self, speed: float) -> None:
        self.speed = float(np.clip(float(speed), 0.05, 10.0))

    def seek_fraction(self, fraction: float) -> None:
        if self._total > 0:
            self._index = int(np.clip(fraction, 0.0, 1.0) * (self._total - 1))

    def seek_to_time(self, seconds: float) -> None:
        if self._total > 0:
            self._index = int(np.clip(seconds * self.target_fs, 0, self._total - 1))

    def seek_to_scenario(self, scenario: str) -> bool:
        """Jump to the first stable segment of a given class (BASELINE/STRESS/AMUSEMENT)."""
        if not self.is_loaded:
            return False
        labels = self._data["label"]
        key = scenario.upper()
        target = None
        if "BASE" in key or "NORM" in key:
            target = 0
        elif "STRESS" in key or "THREAT" in key or "DISTRESS" in key:
            target = 1
        elif "AMUSE" in key:
            target = 2
        if target is None:
            return False

        idx = np.where(labels == target)[0]
        if len(idx) == 0:
            return False
        # Land a few seconds in, so the window is fully inside the segment
        # instead of straddling its boundary.
        offset = int(4 * self.target_fs)
        jump = int(min(idx[0] + offset, idx[-1]))
        self._index = int(np.clip(jump, 0, self._total - 1))
        return True

    def advance(self, n_samples: int) -> None:
        """Advance playback by ``n_samples`` (wraps around at the end)."""
        if self._total > 0 and self._running:
            step = max(0, int(n_samples))
            self._index = (self._index + step) % self._total
            self._packets_sent += step

    def advance_time(self, dt_seconds: float, speed: Optional[float] = None) -> None:
        """
        Advance by ``dt_seconds`` of wall time at the current (or given) speed.

        A telemetry tick is ~50 ms, i.e. ~1.6 samples at 32 Hz.  Rounding that to
        1 or 2 samples per tick makes playback noticeably slower than "1x"; the
        fractional accumulator keeps the configured speed honest.
        """
        factor = self.speed if speed is None else float(speed)
        self._pending += self.target_fs * float(dt_seconds) * factor
        whole = int(self._pending)
        if whole > 0:
            self._pending -= whole
            self.advance(whole)

    # ------------------------------------------------------------------ #
    def get_next_sample(self) -> Optional[SensorPacket]:
        """Sample at the current playback position.  Missing channels stay ``None``."""
        if not self.is_loaded:
            return None
        d = self._data
        i = int(self._index)

        acc_x, acc_y, acc_z = float(d["acc_x"][i]), float(d["acc_y"][i]), float(d["acc_z"][i])
        return SensorPacket(
            timestamp=float(i) / self.target_fs,
            heart_rate=float(d["hr"][i]),
            spo2=None,                 # WESAD has no SpO2
            eda=float(d["eda"][i]),
            temperature=float(d["temp"][i]),
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            gyro_x=None, gyro_y=None, gyro_z=None,   # WESAD has no wrist gyroscope
            finger_detected=True,
            label=int(d["label"][i]),
            source="wesad_replay",
        )

    # ------------------------------------------------------------------ #
    def get_state(self) -> Dict[str, Any]:
        label = -1
        if self.is_loaded:
            label = int(self._data["label"][self._index])
        elapsed = self._index / self.target_fs if self.is_loaded else 0.0
        total = self._total / self.target_fs if self.is_loaded else 0.0
        return {
            "source": self.name,
            "running": self._running,
            "loaded": self.is_loaded,
            "subject_id": self.subject_id,
            "speed": self.speed,
            "current_index": int(self._index),
            "total_samples": int(self._total),
            "elapsed_seconds": round(elapsed, 1),
            "total_seconds": round(total, 1),
            "progress_fraction": round(self._index / max(1, self._total), 4),
            "ground_truth_label": label,
            "ground_truth_name": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "Unknown",
            "packets_sent": self._packets_sent,
            "sampling_rate_hz": self.target_fs,
            "window_samples": DEFAULT_WINDOW.window_samples,
            "error": self._load_error,
        }

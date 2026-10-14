"""
The unified data contract.

Every sensor source — real WESAD replay, a live Arduino/ESP32 over USB or BLE, a
recorded CSV log, or the hardware-free simulator — produces the *same*
:class:`SensorPacket` objects and feeds the *same* :class:`WindowBuffer`.  The
trained model and every downstream component therefore never need to know where
a reading came from.

Design rules
------------
* **Missing means missing.**  A reading the hardware did not measure is ``None``,
  never ``0.0`` and never an invented default.  Only the window builder turns
  ``None`` into ``0.0``, and only after interpolation has genuinely failed.
* **One canonical order.**  ``config.PACKET_CHANNELS`` defines how a packet
  becomes a row of numbers; every channel list in the project is a subset of it.
* **No partial windows.**  :meth:`WindowBuffer.get_window` returns ``None`` until
  a full contiguous window exists.  The predecessor project wrapped the tail of
  the recording around to the head, which fed a time-discontinuous window to the
  model; that is impossible here.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from threatwatch.config import PACKET_CHANNELS, TARGET_FS


# --------------------------------------------------------------------------- #
# Packet
# --------------------------------------------------------------------------- #
@dataclass
class SensorPacket:
    """One normalised sample.  Missing sensor readings are ``None``, never faked."""

    timestamp: float
    heart_rate: Optional[float] = None   # BPM (MAX30102 / derived from PPG)
    spo2: Optional[float] = None         # % — needs calibration to be trustworthy
    eda: Optional[float] = None          # GSR (microsiemens) — None if no GSR sensor
    temperature: Optional[float] = None  # Celsius
    acc_x: Optional[float] = None        # g
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None
    gyro_x: Optional[float] = None       # deg/s
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    finger_detected: Optional[bool] = None
    label: int = -1                      # ground truth (WESAD replay only)
    source: str = "unknown"

    # -- derived ---------------------------------------------------------- #
    @property
    def acc_mag(self) -> Optional[float]:
        vals = [self.acc_x, self.acc_y, self.acc_z]
        if any(v is None for v in vals):
            return None
        return float(np.sqrt(sum(v * v for v in vals)))

    @property
    def gyro_mag(self) -> Optional[float]:
        vals = [self.gyro_x, self.gyro_y, self.gyro_z]
        if any(v is None for v in vals):
            return None
        return float(np.sqrt(sum(v * v for v in vals)))

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["acc_mag"] = self.acc_mag
        d["gyro_mag"] = self.gyro_mag
        return d

    def to_csv(self) -> str:
        """
        Compact machine-readable line for logging / re-parsing::

            ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label

        Missing values are emitted as empty fields (never as 0).
        """

        def f(v: Optional[float]) -> str:
            if v is None:
                return ""
            return f"{v:.4f}" if isinstance(v, float) else str(int(v))

        finger = "" if self.finger_detected is None else ("1" if self.finger_detected else "0")
        return ",".join(
            [
                f"{self.timestamp:.4f}", f(self.heart_rate), f(self.spo2), f(self.eda),
                f(self.temperature), f(self.acc_x), f(self.acc_y), f(self.acc_z),
                f(self.gyro_x), f(self.gyro_y), f(self.gyro_z), finger, str(self.label),
            ]
        )


# --------------------------------------------------------------------------- #
# Packet -> row of numbers
# --------------------------------------------------------------------------- #
ChannelExtractor = Callable[[SensorPacket], Optional[float]]

CHANNEL_EXTRACTORS: Dict[str, ChannelExtractor] = {
    "hr": lambda p: p.heart_rate,
    "eda": lambda p: p.eda,
    "temp": lambda p: p.temperature,
    "acc_x": lambda p: p.acc_x,
    "acc_y": lambda p: p.acc_y,
    "acc_z": lambda p: p.acc_z,
    "acc_mag": lambda p: p.acc_mag,
}

# ``PACKET_CHANNELS`` must be expressible through the extractor table.
assert set(PACKET_CHANNELS) <= set(CHANNEL_EXTRACTORS), "PACKET_CHANNELS/extractor mismatch"


def packet_row(pkt: SensorPacket) -> List[float]:
    """Packet -> list of floats in canonical ``PACKET_CHANNELS`` order (NaN if missing)."""
    row: List[float] = []
    for name in PACKET_CHANNELS:
        v = CHANNEL_EXTRACTORS[name](pkt)
        row.append(np.nan if v is None else float(v))
    return row


def validate_channels(channels: Sequence[str]) -> List[str]:
    """Return ``channels`` as a list, raising a clear error on unknown names."""
    out = list(channels)
    unknown = [c for c in out if c not in CHANNEL_EXTRACTORS]
    if unknown:
        raise ValueError(
            f"Unknown channel(s) {unknown}. Known: {sorted(CHANNEL_EXTRACTORS)}"
        )
    return out


# --------------------------------------------------------------------------- #
# Source interface
# --------------------------------------------------------------------------- #
class SensorSource(abc.ABC):
    """Abstract producer of normalised sensor packets."""

    name: str = "source"

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def get_next_sample(self) -> Optional[SensorPacket]:
        """Return the next packet, or ``None`` if the stream is exhausted/paused."""

    def is_running(self) -> bool:
        return False

    def get_state(self) -> Dict:
        return {"source": self.name, "running": self.is_running()}


# --------------------------------------------------------------------------- #
# Window buffer
# --------------------------------------------------------------------------- #
class WindowBuffer:
    """
    Accumulates packets and produces fixed-shape model windows.

    Hardware samples are irregularly spaced (~10–20 Hz), whereas the model wants
    a uniform grid, so the buffer keeps the last ``window_sec`` of raw samples in
    *simulated* time and linearly resamples them onto a ``target_fs`` grid when a
    window is requested.  This is the same interpolation used when WESAD is
    resampled, so the training and live paths agree.

    ``get_window`` returns ``None`` until the buffer actually spans a full
    window, which prevents the model from ever seeing a partial window stitched
    onto the end of the recording.
    """

    def __init__(
        self,
        window_sec: float = 15.0,
        target_fs: float = TARGET_FS,
        channels: Optional[Sequence[str]] = None,
        min_coverage: float = 0.95,
    ):
        self.window_sec = float(window_sec)
        self.target_fs = float(target_fs)
        self.channels = validate_channels(channels or PACKET_CHANNELS)
        self.min_coverage = float(min_coverage)

        self._col = [PACKET_CHANNELS.index(c) for c in self.channels]
        self._t: List[float] = []
        self._rows: List[List[float]] = []
        # generous raw cap: 8x the samples a window needs at 32 Hz
        self.max_samples = max(64, int(window_sec * target_fs * 8))

    # -- ingestion -------------------------------------------------------- #
    def add(self, pkt: SensorPacket) -> None:
        self._t.append(float(pkt.timestamp))
        self._rows.append(packet_row(pkt))
        if len(self._t) > self.max_samples:
            self._t = self._t[-self.max_samples:]
            self._rows = self._rows[-self.max_samples:]

    def clear(self) -> None:
        self._t.clear()
        self._rows.clear()

    # -- introspection ---------------------------------------------------- #
    def n_samples(self) -> int:
        return len(self._t)

    def span_sec(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0

    def is_ready(self) -> bool:
        return self.n_samples() >= 2 and self.span_sec() >= self.window_sec * self.min_coverage

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    # -- window ----------------------------------------------------------- #
    def get_window(self) -> Optional[np.ndarray]:
        """
        Return the latest contiguous window of shape ``(T, C)``, or ``None`` if
        the buffer does not yet cover a full window.
        """
        if not self.is_ready():
            return None

        t = np.asarray(self._t, dtype=np.float64)
        rows = np.asarray(self._rows, dtype=np.float64)[:, self._col]  # (N, C)

        t_end = float(t[-1])
        grid = np.linspace(t_end - self.window_sec, t_end, int(round(self.window_sec * self.target_fs)))

        out = np.zeros((len(grid), rows.shape[1]), dtype=np.float32)
        for c in range(rows.shape[1]):
            col = rows[:, c]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                # Channel genuinely absent (e.g. no EDA sensor) -> contribute zeros
                # rather than inventing a signal.
                out[:, c] = 0.0
                continue
            out[:, c] = np.interp(grid, t[valid], col[valid])

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def get_state(self) -> Dict:
        return {
            "channels": list(self.channels),
            "samples": self.n_samples(),
            "span_sec": round(self.span_sec(), 2),
            "window_sec": self.window_sec,
            "target_fs": self.target_fs,
            "ready": self.is_ready(),
        }

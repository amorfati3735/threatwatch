"""
Loading + wrist-signal extraction for the WESAD dataset.

WESAD (Schmidt et al., 2018, "Introducing WESAD, a Multimodal Dataset for
Wearable Stress and Affect Detection") ships per-subject pickles with the
following structure::

    data = {
        "signal": {
            "chest": {"ACC": (N,3) @700Hz, "ECG", "EDA", "EMG", "Resp", "Temp"},
            "wrist": {"ACC": (N,3) @32Hz, "BVP" @64Hz, "EDA" @4Hz, "TEMP" @4Hz},
        },
        "label": (N_label,) ints @700Hz,
        "subject": "S2",
    }

Label codes: 1=baseline, 2=stress, 3=amusement, 0/4/5/6/7=transient/ignore.

We use **only wrist signals** because the physical prototype is a wrist
wearable (MAX30102 PPG + MPU6050 IMU).  Everything is resampled to a common
rate with linear interpolation; heart rate is derived from the BVP channel to
mimic the HR value our firmware computes on-chip.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import signal

from threatwatch.config import (
    ALL_SUBJECTS,
    CHEST_FS,
    LABEL_MAP,
    TARGET_FS,
    WESAD_DIR,
    WRIST_ACC_LSB,
    WRIST_RATES,
)


def subject_path(subject: str, data_dir: Path = WESAD_DIR) -> Optional[Path]:
    """Locate a subject pickle, accepting both ``S2/S2.pkl`` and ``S2.pkl``."""
    candidates = [
        Path(data_dir) / subject / f"{subject}.pkl",
        Path(data_dir) / f"{subject}.pkl",
        Path(data_dir) / subject / f"{subject}.pkl.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def available_subjects(data_dir: Path = WESAD_DIR) -> list[str]:
    """Return WESAD subjects present on disk, sorted."""
    out = []
    for p in Path(data_dir).glob("S*/S*.pkl"):
        out.append(p.parent.name)
    for p in Path(data_dir).glob("S*.pkl"):
        out.append(p.stem)
    return sorted(set(out), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)


def load_wesad_pickle(path: Path) -> dict:
    """Load a WESAD subject pickle (was generated with py2, needs latin1)."""
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _resample(x: np.ndarray, fs_src: float, n_target: int, fs_tgt: float) -> np.ndarray:
    """Linear-interpolation resample of a 1-D signal to ``n_target`` samples."""
    if x.ndim != 1:
        x = x.reshape(-1)
    if len(x) == 0:
        return np.zeros(n_target, dtype=np.float32)
    t_src = np.arange(len(x), dtype=np.float64) / fs_src
    t_tgt = np.arange(n_target, dtype=np.float64) / fs_tgt
    return np.interp(t_tgt, t_src, x).astype(np.float32)


def derive_heart_rate(bvp: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate instantaneous heart rate (BPM) from a PPG/BVP waveform.

    This reproduces the kind of HR estimate a MAX30102-based firmware reports.
    The signal is band-passed to the plausible human pulse band, peaks are
    detected, and instantaneous BPM is interpolated across time.  Where no
    peaks are found the population median (~72 BPM) is used rather than a
    fabricated value derived from the label.
    """
    if len(bvp) < 2 * fs:
        return np.full(len(bvp), 72.0, dtype=np.float32)

    x = bvp.astype(np.float64)
    x = x - np.mean(x)
    # Pulse band: 0.7 Hz (42 BPM) .. 3.5 Hz (210 BPM), Nyquist-limited.
    high = min(3.5, fs / 2 * 0.95)
    if high <= 0.8:
        return np.full(len(bvp), 72.0, dtype=np.float32)
    try:
        sos = signal.butter(3, [0.7, high], btype="bandpass", fs=fs, output="sos")
        xf = signal.sosfiltfilt(sos, x)
    except ValueError:
        xf = x

    min_dist = int(0.35 * fs)  # <= ~170 BPM
    prominence = 0.3 * np.std(xf) if np.std(xf) > 0 else 0.0
    peaks, _ = signal.find_peaks(xf, distance=max(min_dist, 1), prominence=prominence)

    hr = np.full(len(bvp), 72.0, dtype=np.float32)
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        rr = np.clip(rr, 0.3, 2.0)  # 30..200 BPM
        bpm = 60.0 / rr
        # Assign each beat interval to the span between its two peaks
        for i, b in enumerate(bpm):
            hr[peaks[i]:peaks[i + 1]] = b
        hr[: peaks[0]] = bpm[0]
        hr[peaks[-1]:] = bpm[-1]
    return hr.astype(np.float32)


def extract_subject(
    subject: str,
    data_dir: Path = WESAD_DIR,
    target_fs: float = TARGET_FS,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load one subject and return wrist signals resampled to ``target_fs``.

    Returns a dict with 1-D float32 arrays (all length ``n``) plus ``label``.
    Only the wrist signals are retained; the large chest arrays are released
    immediately to keep peak RAM low.
    """
    path = subject_path(subject, data_dir)
    if path is None:
        return None

    raw = load_wesad_pickle(path)
    try:
        wrist = raw["signal"]["wrist"]
        label = np.asarray(raw["label"]).reshape(-1)
    except KeyError as e:
        raise KeyError(f"{subject}: unexpected WESAD structure ({e})") from e

    n_label = len(label)
    duration = n_label / CHEST_FS
    n = int(round(duration * target_fs))
    if n <= 1:
        return None

    # Wrist channels.  WESAD's wrist device is an Empatica E4, whose 3-axis
    # accelerometer is reported in units of 1/64 g.  We convert to g so the
    # channel is on the same scale as the MPU6050 (which reports g).
    acc = np.asarray(wrist["ACC"], dtype=np.float64) / WRIST_ACC_LSB
    acc_x = _resample(acc[:, 0], WRIST_RATES["ACC"], n, target_fs)
    acc_y = _resample(acc[:, 1], WRIST_RATES["ACC"], n, target_fs)
    acc_z = _resample(acc[:, 2], WRIST_RATES["ACC"], n, target_fs)
    bvp = _resample(np.asarray(wrist["BVP"]).reshape(-1), WRIST_RATES["BVP"], n, target_fs)
    eda = _resample(np.asarray(wrist["EDA"]).reshape(-1), WRIST_RATES["EDA"], n, target_fs)
    temp = _resample(np.asarray(wrist["TEMP"]).reshape(-1), WRIST_RATES["TEMP"], n, target_fs)
    hr = derive_heart_rate(
        np.asarray(wrist["BVP"]).reshape(-1).astype(np.float32), WRIST_RATES["BVP"]
    )
    hr = _resample(hr, WRIST_RATES["BVP"], n, target_fs)

    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2).astype(np.float32)

    # Resample the 700 Hz label stream by nearest index.
    src_idx = np.clip((np.arange(n) * (CHEST_FS / target_fs)).astype(np.int64), 0, n_label - 1)
    label_t = label[src_idx].astype(np.int64)
    mapped = np.array([LABEL_MAP.get(int(v), -1) for v in label_t], dtype=np.int64)

    # Free the (potentially large) raw dict before returning
    del raw, wrist, label, acc

    return {
        "subject": subject,
        "n": n,
        "sampling_rate": float(target_fs),
        "hr": hr,
        "bvp": bvp,
        "eda": eda,
        "temp": temp,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "acc_mag": acc_mag,
        "label": mapped,
    }

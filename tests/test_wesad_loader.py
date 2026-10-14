"""
WESAD loader tests.

The fixture mirrors the official pickle structure exactly (same keys, same
native sampling rates) so the loader's resampling, unit conversion and label
mapping are proven without needing the 13 GB of raw data.

Critically, this asserts the **1/64 g accelerometer conversion**.  The
predecessor project's adapter omitted it, leaving the acceleration channel off
by a factor of 64 — effectively dead as a feature.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from threatwatch.config import CHEST_FS, TARGET_FS, WRIST_RATES
from threatwatch.data.wesad import available_subjects, extract_subject

MINUTES = 4


def make_fixture(subject: str = "S2") -> dict:
    n_label = int(MINUTES * 60 * CHEST_FS)
    rng = np.random.default_rng(0)

    def chest(n):
        return rng.normal(0, 1, n).astype(np.float32)

    n_acc = int(MINUTES * 60 * WRIST_RATES["ACC"])
    n_bvp = int(MINUTES * 60 * WRIST_RATES["BVP"])
    n_eda = int(MINUTES * 60 * WRIST_RATES["EDA"])

    # A clean 1.2 Hz pulse => 72 BPM, so HR derivation can be verified.
    t = np.arange(n_bvp) / WRIST_RATES["BVP"]
    bvp = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)

    # Wrist accelerometer in Empatica E4 raw counts: 64 counts == 1 g.
    # A device lying flat reports approximately (0, 0, +64) counts.
    acc = np.zeros((n_acc, 3), dtype=np.float32)
    acc[:, 2] = 64.0

    data = {
        "signal": {
            "chest": {
                "ACC": np.stack([chest(n_label)] * 3, axis=1),
                "ECG": chest(n_label)[:, None],
                "EDA": chest(n_label)[:, None],
                "EMG": chest(n_label)[:, None],
                "Resp": chest(n_label)[:, None],
                "Temp": chest(n_label)[:, None],
            },
            "wrist": {
                "ACC": acc,
                "BVP": bvp[:, None],
                "EDA": rng.normal(2.0, 0.2, (n_eda, 1)).astype(np.float32),
                "TEMP": rng.normal(33.5, 0.1, (n_eda, 1)).astype(np.float32),
            },
        },
        "label": np.zeros(n_label, dtype=np.int32),
        "subject": subject,
    }
    q = n_label // 4
    data["label"][: q] = 1        # baseline
    data["label"][q: 2 * q] = 2   # stress
    data["label"][2 * q: 3 * q] = 3   # amusement
    data["label"][3 * q:] = 0     # transient -> must map to -1
    return data


def _write_fixture(root: Path, subject: str = "S2") -> None:
    d = root / subject
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{subject}.pkl", "wb") as f:
        pickle.dump(make_fixture(subject), f)


def test_available_subjects_discovers_fixture():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root, "S2")
        _write_fixture(root, "S3")
        assert available_subjects(root) == ["S2", "S3"]


def test_shape_duration_and_label_mapping():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root)
        d = extract_subject("S2", data_dir=root, target_fs=TARGET_FS)

        assert d is not None
        n = d["n"]
        expected = int(MINUTES * 60 * TARGET_FS)
        assert abs(n - expected) <= 1, f"n={n} expected~{expected}"

        for ch in ["hr", "eda", "temp", "acc_x", "acc_y", "acc_z", "acc_mag", "label"]:
            assert len(d[ch]) == n, ch

        # WESAD codes 1/2/3 -> 0/1/2, transient 0 -> -1
        assert set(np.unique(d["label"])) <= {0, 1, 2, -1}
        assert -1 in set(np.unique(d["label"]))


def test_accelerometer_unit_conversion():
    """The wrist accelerometer is in 1/64 g and must be converted to g."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root)
        d = extract_subject("S2", data_dir=root, target_fs=TARGET_FS)

        median_mag = float(np.median(d["acc_mag"]))
        assert abs(median_mag - 1.0) < 1e-3, (
            f"acc_mag should be ~1.0 g for (0,0,64) raw counts, got {median_mag}. "
            f"The 1/64 g conversion is missing."
        )
        assert abs(float(np.median(d["acc_z"])) - 1.0) < 1e-3


def test_heart_rate_derived_from_bvp():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root)
        d = extract_subject("S2", data_dir=root, target_fs=TARGET_FS)
        hr_med = float(np.median(d["hr"]))
        assert 65.0 <= hr_med <= 80.0, f"1.2 Hz pulse should give ~72 BPM, got {hr_med}"


def test_missing_subject_returns_none():
    with tempfile.TemporaryDirectory() as td:
        assert extract_subject("S99", data_dir=Path(td)) is None

"""
Fast pipeline tests that do NOT require the WESAD dataset.

They cover the leakage-critical and shape-critical parts with synthetic arrays:
windowing/labels, train-only standardisation, every model's forward shape, the
serial CSV round-trip (missing stays missing), and the guarantee that a partial
window is never handed to a model.
"""

from __future__ import annotations

import numpy as np
import torch

from threatwatch.config import NUM_CLASSES
from threatwatch.data.dataset import Standardizer, WindowDataset, extract_windows
from threatwatch.data.sources.serial_source import parse_line
from threatwatch.data.sources.simulator import SimulatorSource
from threatwatch.data.types import SensorPacket, WindowBuffer
from threatwatch.models.seq_models import MODEL_REGISTRY, build_model, count_parameters


def _fake_subject(n=2000):
    rng = np.random.default_rng(0)
    data = {
        ch: rng.normal(0, 1, n).astype(np.float32)
        for ch in ["hr", "eda", "temp", "acc_x", "acc_y", "acc_z", "acc_mag"]
    }
    label = np.full(n, -1, dtype=np.int64)
    label[: n // 3] = 0
    label[n // 3: 2 * n // 3] = 1
    label[2 * n // 3:] = 2
    data["label"] = label
    return data


def test_windowing_and_labels():
    d = _fake_subject()
    X, y, starts = extract_windows(d, window_samples=320, stride_samples=64)
    assert X.shape[1] == 320 and X.shape[2] == 7
    assert len(y) == len(starts) == X.shape[0]
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_transient_windows_dropped():
    d = _fake_subject()
    d["label"][:] = -1
    X, y, _ = extract_windows(d, window_samples=320, stride_samples=64)
    assert len(y) == 0, "windows dominated by ignore labels must be dropped"


def test_standardizer_fit_train_only():
    X = np.random.default_rng(1).normal(5, 2, (100, 50, 7)).astype(np.float32)
    s = Standardizer().fit(X)
    Z = s.transform(X)
    assert abs(float(Z.mean())) < 0.05
    assert abs(float(Z.std()) - 1.0) < 0.1

    # Applying train statistics to a different distribution must NOT re-normalise.
    Y = np.random.default_rng(2).normal(50, 2, (10, 50, 7)).astype(np.float32)
    ZY = s.transform(Y)
    assert float(ZY.mean()) > 10.0, "transform must use frozen train stats"


def test_all_models_forward_shape():
    x = torch.randn(4, 480, 7)
    for name in sorted(MODEL_REGISTRY):
        model = build_model(name, 7, NUM_CLASSES)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES), f"{name} produced {out.shape}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite logits"


def test_patch_vit_handles_non_multiple_length():
    """The ViT patchifier must pad an awkward window length, not crash."""
    model = build_model("patch_vit", 7, NUM_CLASSES, patch=32)
    out = model(torch.randn(2, 477, 7))
    assert out.shape == (2, NUM_CLASSES)


def test_serial_csv_roundtrip():
    p = SensorPacket(timestamp=1.5, heart_rate=80.0, eda=None, spo2=None,
                     temperature=33.2, acc_x=0.1, acc_y=0.2, acc_z=0.98,
                     gyro_x=1.0, gyro_y=2.0, gyro_z=3.0,
                     finger_detected=True, label=-1)
    line = p.to_csv()
    back = parse_line(line)
    assert back is not None
    assert abs(back.heart_rate - 80.0) < 1e-6
    assert back.eda is None, "a missing EDA reading must stay None"
    assert back.spo2 is None, "a missing SpO2 reading must stay None"
    assert abs(back.acc_z - 0.98) < 1e-6
    assert back.finger_detected is True

    assert parse_line("ts,hr,spo2") is None
    assert parse_line("") is None
    assert parse_line("# a firmware comment") is None


def test_window_buffer_no_partial_window():
    """A model must never receive a window stitched from the end of the record."""
    buf = WindowBuffer(window_sec=2.0, target_fs=32.0)
    sim = SimulatorSource(state=0, hz=15.0)
    sim.start()
    t = 0.0
    # Feed less than one full window.
    for _ in range(10):
        p = sim.get_next_sample()
        p.timestamp = t
        t += 1 / 15.0
        buf.add(p)
    assert buf.get_window() is None, "a partial window must be refused"
    assert buf.is_ready() is False

    # Now exceed the window span.
    for _ in range(60):
        p = sim.get_next_sample()
        p.timestamp = t
        t += 1 / 15.0
        buf.add(p)
    win = buf.get_window()
    assert win is not None and win.shape == (64, 7)


def test_window_buffer_channel_subset():
    buf = WindowBuffer(window_sec=1.0, target_fs=16.0, channels=["hr", "acc_mag"])
    for i in range(20):
        buf.add(SensorPacket(timestamp=i * 0.0625, heart_rate=70 + i,
                             acc_x=0.0, acc_y=0.0, acc_z=1.0))
    win = buf.get_window()
    assert win is not None and win.shape == (16, 2)


def test_window_buffer_missing_channel_does_not_invent_signal():
    """EDA absent on the hardware board must become a flat zero column, not noise."""
    buf = WindowBuffer(window_sec=1.0, target_fs=16.0, channels=["hr", "eda"])
    for i in range(20):
        buf.add(SensorPacket(timestamp=i * 0.0625, heart_rate=70)
                )  # eda is None
    win = buf.get_window()
    assert win is not None
    assert np.allclose(win[:, 1], 0.0), "absent channel must be a flat zero column"


def test_torch_dataset():
    X = np.random.randn(10, 320, 7).astype(np.float32)
    y = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
    ds = WindowDataset(X, y)
    xb, yb = ds[0]
    assert xb.shape == (320, 7) and yb.dtype == torch.long

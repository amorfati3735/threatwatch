"""
Replay engine tests.

The engine is fed a synthetic subject via monkeypatch so the tests are fast and
do not depend on the 13 GB WESAD download.  They verify the properties the
pipeline relies on: honest ``None`` for channels WESAD lacks, time-accurate
playback speed, and scenario seeking that lands inside the requested class.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import threatwatch.data.replay_engine as replay_module
from threatwatch.config import TARGET_FS
from threatwatch.data.replay_engine import ReplayEngine

N = int(600 * TARGET_FS)  # 10 minutes at 32 Hz


def _fake_subject(subject, data_dir=None, target_fs=TARGET_FS):
    label = np.full(N, -1, dtype=np.int64)
    label[: N // 3] = 0
    label[N // 3: 2 * N // 3] = 1
    label[2 * N // 3:] = 2
    return {
        "subject": subject,
        "n": N,
        "sampling_rate": float(target_fs),
        "hr": np.full(N, 72.0, dtype=np.float32),
        "eda": np.full(N, 2.5, dtype=np.float32),
        "temp": np.full(N, 33.1, dtype=np.float32),
        "acc_x": np.zeros(N, dtype=np.float32),
        "acc_y": np.zeros(N, dtype=np.float32),
        "acc_z": np.ones(N, dtype=np.float32),
        "acc_mag": np.ones(N, dtype=np.float32),
        "label": label,
    }


@pytest.fixture()
def engine():
    with mock.patch.object(replay_module, "extract_subject", _fake_subject):
        yield ReplayEngine(subject_id="S2")


def test_packets_are_honest_about_missing_channels(engine):
    pkt = engine.get_next_sample()
    assert pkt is not None
    assert pkt.heart_rate == pytest.approx(72.0)
    assert pkt.eda == pytest.approx(2.5)
    assert pkt.spo2 is None, "WESAD has no SpO2 - must be None, not a guess"
    assert pkt.gyro_x is None and pkt.gyro_z is None
    assert pkt.source == "wesad_replay"
    assert pkt.label == 0


def test_advance_time_is_speed_accurate(engine):
    engine.seek_fraction(0.0)
    engine.advance_time(1.0, speed=1.0)
    assert engine.get_state()["current_index"] == int(TARGET_FS), "1 s at 1x == 32 samples"

    engine.seek_fraction(0.0)
    engine._pending = 0.0
    engine.advance_time(1.0, speed=4.0)
    assert engine.get_state()["current_index"] == int(TARGET_FS * 4), "1 s at 4x == 128 samples"


def test_fractional_advance_does_not_drift(engine):
    """Twenty 50 ms ticks at 1x must total exactly one second of samples."""
    engine.seek_fraction(0.0)
    engine._pending = 0.0
    for _ in range(20):
        engine.advance_time(0.05, speed=1.0)
    assert engine.get_state()["current_index"] == int(TARGET_FS)


def test_seek_to_scenario_lands_inside_the_class(engine):
    assert engine.seek_to_scenario("STRESS") is True
    idx = engine.get_state()["current_index"]
    assert N // 3 <= idx < 2 * N // 3, "must land inside the stress segment"

    assert engine.seek_to_scenario("BASELINE") is True
    assert engine.get_state()["current_index"] < N // 3

    assert engine.seek_to_scenario("AMUSEMENT") is True
    assert engine.get_state()["current_index"] >= 2 * N // 3


def test_seek_unknown_scenario_returns_false(engine):
    assert engine.seek_to_scenario("NOT_A_THING") is False


def test_state_reports_progress_and_ground_truth(engine):
    engine.seek_fraction(0.5)
    state = engine.get_state()
    assert 0.45 <= state["progress_fraction"] <= 0.55
    assert state["ground_truth_name"] in ("baseline", "stress", "amusement")
    assert state["total_samples"] == N
    assert state["loaded"] is True


def test_wraparound_at_end(engine):
    engine.seek_fraction(1.0)
    engine.advance(10)
    assert engine.get_state()["current_index"] < 100, "playback should wrap to the start"

"""
Regression tests for the OLED feedback loop (laptop -> wearable).

The dashboard mirrors the fused verdict onto the board's OLED as
``STATE:<name>`` / ``SCORE:<n>``.  On the wired link that is free, but the ESP32's
BLE stack could not keep up with one write per telemetry tick (20/s): the
fire-and-forget coroutines queued without bound and the panel froze on whatever
state it had last received — it was still showing ``THREAT`` after the operator
had reset the alert.

The contract these tests pin: a *state change* is written immediately, and an
unchanged verdict is refreshed at most once per ``FEEDBACK_MIN_INTERVAL_SEC``.
"""

from __future__ import annotations

import time

import pytest

from threatwatch.data.sources.hardware import FEEDBACK_MIN_INTERVAL_SEC, HardwareSensorSource


class FakeSerial:
    """Stands in for pyserial; records what the board was told."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)


@pytest.fixture()
def connected(monkeypatch) -> HardwareSensorSource:
    src = HardwareSensorSource()
    src._connected = True
    src._serial = FakeSerial()
    return src


def test_state_change_is_sent_immediately(connected: HardwareSensorSource) -> None:
    assert connected.write_feedback(5, "SAFE") is True
    assert connected._serial.writes == [b"STATE:SAFE\nSCORE:5\n"]

    # Escalation must reach the panel at once, not on the next heartbeat.
    connected.write_feedback(95, "THREAT")
    assert b"STATE:THREAT\nSCORE:95\n" in connected._serial.writes


def test_unchanged_verdict_is_not_flooded(connected: HardwareSensorSource) -> None:
    connected.write_feedback(5, "SAFE")
    assert len(connected._serial.writes) == 1

    for pct in range(6, 30):  # the score wanders every tick; the state does not
        connected.write_feedback(pct, "SAFE")
    assert len(connected._serial.writes) == 1, "the same verdict was re-sent on every tick"


def test_unchanged_verdict_is_refreshed_once_per_interval(connected: HardwareSensorSource, monkeypatch) -> None:
    clock = {"now": 1_700_000_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["now"])

    connected.write_feedback(5, "SAFE")
    assert len(connected._serial.writes) == 1

    clock["now"] += FEEDBACK_MIN_INTERVAL_SEC / 2.0
    connected.write_feedback(7, "SAFE")
    assert len(connected._serial.writes) == 1

    clock["now"] += FEEDBACK_MIN_INTERVAL_SEC
    connected.write_feedback(7, "SAFE")
    assert len(connected._serial.writes) == 2, "the heartbeat refresh was lost"


def test_feedback_is_not_written_when_disconnected() -> None:
    src = HardwareSensorSource()
    assert src.write_feedback(50, "THREAT") is False

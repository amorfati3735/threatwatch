"""
Regression tests for the live-hardware timeline (device ts -> host clock).

The board reports ``ts`` as **seconds since boot** and ``millis()`` restarts
whenever it resets (which a USB port open / reflash triggers).  The source
rebases that onto the host clock so the single :class:`WindowBuffer` always sees
a monotonic timeline.  Three field failures defeated that rebase, all of them
observed on real hardware:

1. ``ser.readline()`` times out mid-record (the sketch blocks in I2C for ~100 ms
   while refreshing the OLED).  The *remainder* of the split record then arrives
   as the next "line" and parses as a sample whose timestamp is a **fragment**
   of the real one.  That silently re-anchored the whole timeline: the recorded
   dashboard ran with ``pkt_ts`` ~435 s ahead of the host clock.
2. A board reset restarts ``millis()`` mid-session.
3. Once the offset is wrong, a later reset drives ``span_sec`` negative and the
   window never becomes ready again (the handoff's ``span_sec: -894``).

These tests pin the contract: a sample is only accepted if it is a *whole*
record, the rebased clock tracks the host clock within a bounded drift, and the
buffer never sees a timestamp that goes backwards.
"""

from __future__ import annotations

import os
import pty
import time

import pytest

try:
    import serial  # noqa: F401

    HAVE_PYSERIAL = True
except ImportError:  # pragma: no cover
    HAVE_PYSERIAL = False

from threatwatch.data.sources.hardware import HardwareSensorSource
from threatwatch.data.sources.serial_source import parse_line
from threatwatch.data.types import WindowBuffer

T0 = 1_700_000_000.0  # arbitrary host epoch
SAMPLE_PERIOD = 0.12  # the sketch streams at ~8 Hz


class FakeClock:
    """A host clock we can advance by hand (time.time() substitute)."""

    def __init__(self, start: float = T0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += float(seconds)


@pytest.fixture()
def clock(monkeypatch) -> FakeClock:
    c = FakeClock()
    monkeypatch.setattr(time, "time", c)
    return c


def device_line(dev_ts: float, hr: float = 78.0, temp: float = 33.4) -> str:
    """One record exactly as ``firmware/esp32_wearable`` prints it (13 fields)."""
    return f"{dev_ts:.3f},{hr:g},,,{temp:.2f},0.010,-0.020,0.990,1.00,0.30,-0.50,1,-1"


def feed(source: HardwareSensorSource, clock: FakeClock, dev_ts: float, real_seconds: float = SAMPLE_PERIOD) -> float:
    """Park one device record ``real_seconds`` after the previous one."""
    clock.tick(real_seconds)
    line = device_line(dev_ts)
    # The reader hands complete lines to the parser, so the source sees the same
    # bytes whether they came from USB or a BLE notification.
    source._park_packet(line)
    pkt = source.get_next_sample()
    assert pkt is not None
    return float(pkt.timestamp)


def stream(source: HardwareSensorSource, clock: FakeClock, start_dev_ts: float, n: int) -> list[float]:
    return [feed(source, clock, start_dev_ts + i * SAMPLE_PERIOD) for i in range(n)]


# --------------------------------------------------------------------------- #
# 1. A fragment of a record is not a record
# --------------------------------------------------------------------------- #
def test_truncated_record_is_not_accepted_as_a_sample():
    """A record with fields missing is never a sample.

    A tail such as ``78.0,,,33.40,...,-1`` has had its leading fields cut off,
    so every value is shifted one channel to the left — reading it would invent
    a heart rate from a SpO2 field.  Rejected outright.
    """
    assert parse_line(device_line(452.4)) is not None, "a whole record must still parse"

    for split in range(1, 13):  # cut after field ``split``
        tail = device_line(452.4).split(",", split)[split]
        assert parse_line(tail) is None, f"accepted a truncated record: {tail!r}"

    # Sanity: a full-width record is the only thing that parses.
    assert parse_line(device_line(2.4)).timestamp == pytest.approx(2.4)


def test_reader_drops_a_record_split_by_a_readline_timeout(tmp_path):
    """End-to-end: a split record must not produce a bogus *timestamp*.

    A record cut *inside* the timestamp field (``452.400`` -> ``2.400``) has the
    full field count, so no parser can tell it from a board that really is 2.4 s
    into its boot.  It is only identifiable at the reader, which knows that the
    previous read timed out mid-record, so it drops both halves and keeps the
    next complete record.
    """
    if not HAVE_PYSERIAL:
        pytest.skip("pyserial not installed")
    try:
        master_fd, slave_fd = pty.openpty()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"pty unavailable: {exc}")
    slave_name = os.ttyname(slave_fd)
    os.close(slave_fd)

    src = HardwareSensorSource(port=slave_name)
    src.stale_after_sec = 5.0
    try:
        src.start()
    except Exception as exc:  # pragma: no cover
        os.close(master_fd)
        pytest.skip(f"could not open pty: {exc}")

    split = device_line(452.4)
    os.write(master_fd, split[:20].encode())          # prefix: no newline yet
    time.sleep(0.25)                                  # let readline() time out
    os.write(master_fd, (split[20:] + "\n").encode())  # tail: "2.400,78,,,..."
    time.sleep(0.05)
    os.write(master_fd, (device_line(452.52) + "\n").encode())

    deadline = time.time() + 3.0
    while time.time() < deadline and src.get_state()["packets_received"] < 1:
        time.sleep(0.05)
    state = src.get_state()
    pkt = src.get_next_sample()
    src.stop()
    os.close(master_fd)

    assert state["partial_reads"] >= 1, "the split read was not noticed"
    assert state["packets_received"] == 1, f"kept a fragment as a sample: {state}"
    assert pkt is not None and pkt.timestamp != pytest.approx(2.4)
    # The surviving sample is the complete record, rebased onto the host clock.
    assert abs(pkt.timestamp - time.time()) < 5.0


def test_fragment_does_not_shift_the_timeline(clock):
    """A fragment may not re-anchor the clock: the offset must stay ~0."""
    src = HardwareSensorSource()
    stream(src, clock, 452.0, 10)

    # A split record: the tail arrives as a "line" whose first field is 2.4.
    clock.tick(SAMPLE_PERIOD)
    src._park_packet(device_line(452.4)[2:])

    offs = [ts - clock.now for ts in stream(src, clock, 452.5, 6)]
    assert max(abs(o) for o in offs) < 5.0, f"timeline drifted off the host clock: {offs}"


# --------------------------------------------------------------------------- #
# 2. A board reset
# --------------------------------------------------------------------------- #
def test_board_reset_keeps_the_timeline_monotonic_and_ready(clock):
    src = HardwareSensorSource()
    before = stream(src, clock, 500.0, 40)          # board has been up 500 s
    after = stream(src, clock, 2.4, 40)             # reset: millis() restarts

    stamps = before + after
    assert stamps == sorted(stamps), "the buffer saw a backwards timestamp"

    buf = WindowBuffer(window_sec=2.0, target_fs=32.0)
    for i, ts in enumerate(stamps):
        buf.add(_packet(ts, i))
    assert buf.span_sec() > 0, f"span after a reset was {buf.span_sec()}"
    assert buf.is_ready(), buf.get_state()

    # ...and the rebased clock is back on the host clock.
    assert abs(stamps[-1] - clock.now) < 5.0


# --------------------------------------------------------------------------- #
# 3. A poisoned offset must self-heal (the field failure: pkt_ts ~435 s ahead)
# --------------------------------------------------------------------------- #
def test_timeline_self_heals_from_a_wrong_offset(clock):
    src = HardwareSensorSource()
    stream(src, clock, 100.0, 20)

    # The device clock jumps 435 s ahead of where we think it is: a stale
    # line, a re-delivered notification, a clock glitch.  The old base made
    # every later sample sit 435 s in the future *forever*.
    clock.tick(SAMPLE_PERIOD)
    _park(src, device_line(535.0))
    offs = [ts - clock.now for ts in stream(src, clock, 535.12, 20)]
    assert max(abs(o) for o in offs) < 5.0, f"offset never healed: {offs}"


def test_reset_after_a_poisoned_offset_never_goes_negative(clock):
    """The handoff's symptom: ``span_sec`` negative + ``ready: false``.

    A corrupted offset followed by a real board reset used to move the timeline
    *backwards* by the board's uptime (``span_sec: -894``), which left the model
    window permanently unready.
    """
    src = HardwareSensorSource()
    stamps = stream(src, clock, 100.0, 20)

    clock.tick(SAMPLE_PERIOD)
    _park(src, device_line(535.0))          # offset corrupted to +435 s
    stamps += stream(src, clock, 535.12, 20)
    stamps += stream(src, clock, 2.4, 20)   # board reset

    assert stamps == sorted(stamps), "the buffer saw a backwards timestamp"

    buf = WindowBuffer(window_sec=2.0, target_fs=32.0)
    for i, ts in enumerate(stamps):
        buf.add(_packet(ts, i))
    assert buf.span_sec() > 0, f"span was {buf.span_sec()}"
    assert buf.is_ready(), buf.get_state()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _park(src: HardwareSensorSource, line: str) -> None:
    src._park_packet(line)


def _packet(ts: float, i: int):
    from threatwatch.data.types import SensorPacket

    return SensorPacket(timestamp=ts, heart_rate=70.0 + (i % 5), temperature=33.0 + 0.01 * (i % 7),
                        acc_x=0.01, acc_y=-0.02, acc_z=0.99, finger_detected=True, source="hardware")

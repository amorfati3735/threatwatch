"""
End-to-end test of the hardware integration path WITHOUT hardware.

A pseudo-terminal (pty) stands in for the serial port: a "device" thread writes
exactly the lines the firmware emits, and ``SerialSensorSource`` reads them back
through pyserial.  This exercises the same code path a real board uses,
including the ``None``-preserving parser and the WindowBuffer.

Skipped automatically if pyserial is not installed or the pty cannot be opened.
"""

from __future__ import annotations

import os
import pty
import threading
import time

import pytest

try:
    import serial  # noqa: F401

    HAVE_PYSERIAL = True
except ImportError:  # pragma: no cover
    HAVE_PYSERIAL = False

from threatwatch.data.sources.serial_source import SerialSensorSource, parse_line
from threatwatch.data.types import WindowBuffer

# Exactly the shape the merged firmware prints.
FIRMWARE_LINES = [
    "# MAX30102 ready",
    "# MPU6050 ready",
    "ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label",
    "1.000,78.0,,,33.40,0.010,-0.020,0.990,1.00,0.30,-0.50,1,-1",
    "1.066,,,95,33.41,0.011,-0.021,0.988,0.90,0.20,-0.40,1,-1",  # note: hr empty
    "1.132,79.5,,,33.42,0.012,-0.022,0.986,0.80,0.10,-0.30,1,-1",
    "1.198,80.0,,,33.43,0.013,-0.023,0.985,0.70,0.05,-0.20,1,-1",
]


def _device_writer(fd: int, delay: float = 0.03):
    for line in FIRMWARE_LINES:
        os.write(fd, (line + "\n").encode())
        time.sleep(delay)


def test_parse_line_rejects_headers_and_comments():
    assert parse_line("ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label") is None
    assert parse_line("# MAX30102 ready") is None
    assert parse_line("") is None


def test_serial_loopback_over_pty():
    if not HAVE_PYSERIAL:
        pytest.skip("pyserial not installed")

    try:
        master_fd, slave_fd = pty.openpty()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"pty unavailable: {exc}")

    slave_name = os.ttyname(slave_fd)
    os.close(slave_fd)

    src = SerialSensorSource(slave_name, baud=115200, timeout=0.05)
    try:
        src.start()
    except Exception as exc:
        os.close(master_fd)
        pytest.skip(f"could not open pty: {exc}")

    # Write only AFTER start() has drained the reset burst, otherwise the
    # firmware lines would be discarded by reset_input_buffer().
    writer = threading.Thread(target=_device_writer, args=(master_fd,), daemon=True)
    writer.start()

    packets = []
    deadline = time.time() + 4.0
    while time.time() < deadline and len(packets) < 4:
        pkt = src.get_next_sample()
        if pkt is not None:
            packets.append(pkt)
        time.sleep(0.01)

    src.stop()
    os.close(master_fd)

    assert len(packets) >= 3, f"expected >=3 packets, got {len(packets)}"
    assert packets[0].heart_rate == pytest.approx(78.0)
    assert packets[0].acc_z == pytest.approx(0.990, abs=1e-3)
    assert packets[0].finger_detected is True
    # An empty HR field means "no reading", never 0.0.
    assert packets[1].heart_rate is None
    # EDA/SpO2 are deliberately empty in the firmware.
    assert packets[0].spo2 is None and packets[0].eda is None

    buf = WindowBuffer(window_sec=0.1, target_fs=32.0)
    for pkt in packets:
        buf.add(pkt)
    win = buf.get_window()
    assert win is not None and win.shape[1] == 7


def test_downstream_feedback_returns_false_when_disconnected():
    src = SerialSensorSource("NON_EXISTENT_PORT", baud=115200)
    assert src.is_connected is False
    assert src.write_feedback(42, "SAFE") is False

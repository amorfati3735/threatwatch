"""
Fusion-engine tests.

The rules being verified are exactly the claims the project makes in its
documentation, so if they ever change, the docs and the code diverge loudly.
"""

from __future__ import annotations

import pytest

from threatwatch.config import CAUTION_UPPER_BOUND, SUPPRESS_SENSOR_CAP, SUPPRESS_VISION_CAP
from threatwatch.fusion.engine import FusionEngine


def test_weights_are_normalised():
    f = FusionEngine(sensor_weight=3, vision_weight=1)
    assert f.sensor_weight == pytest.approx(0.75)
    assert f.vision_weight == pytest.approx(0.25)


def test_zero_weights_rejected():
    with pytest.raises(ValueError):
        FusionEngine(sensor_weight=0, vision_weight=0)


def test_mutual_agreement_escalates():
    f = FusionEngine()
    r = f.fuse(0.9, 0.85)
    assert r["agreement_status"] == "MUTUAL_AGREEMENT_HIGH"
    assert r["threat_score"] >= 0.78
    assert r["verdict"] == "THREAT / EMERGENCY"


def test_high_sensor_calm_vision_is_suppressed():
    """Exercise / exertion must not be able to reach the THREAT band alone."""
    f = FusionEngine()
    r = f.fuse(0.95, 0.05)
    assert r["agreement_status"] == "PHYSIOLOGICAL_ANOMALY (NON_THREAT)"
    assert r["threat_score"] <= SUPPRESS_SENSOR_CAP
    assert r["verdict"] != "THREAT / EMERGENCY"


def test_high_vision_calm_sensor_is_suppressed():
    """A raised-arm posture with a calm physiology must not trigger an alarm."""
    f = FusionEngine()
    r = f.fuse(0.05, 0.95)
    assert r["agreement_status"] == "VISUAL_ANOMALY (BENIGN_MOTION)"
    assert r["threat_score"] <= SUPPRESS_VISION_CAP
    assert r["threat_score"] <= CAUTION_UPPER_BOUND


def test_suppression_survives_previous_high_history():
    """
    Regression: the suppression cap must apply to the *smoothed* score too.

    With a cap applied only to the raw value, a preceding THREAT window kept the
    rolling average above 0.70 and a single-modality anomaly was reported as a
    THREAT.  That was a real bug found during verification.
    """
    f = FusionEngine(rolling_window_size=5)
    for _ in range(5):
        f.fuse(0.95, 0.95)          # fill history with THREAT-band raw scores
    r = f.fuse(0.95, 0.05)          # now a single-modality anomaly
    assert r["threat_score"] <= SUPPRESS_SENSOR_CAP, (
        f"history leaked into the suppressed score: {r['threat_score']}"
    )
    assert r["verdict"] != "THREAT / EMERGENCY"


def test_smoothing_is_bounded_by_one():
    f = FusionEngine()
    for _ in range(10):
        r = f.fuse(1.0, 1.0)
    assert r["threat_score"] <= 1.0


def test_reset_clears_history():
    f = FusionEngine()
    f.fuse(0.99, 0.99)
    f.reset()
    r = f.fuse(0.05, 0.05)
    assert r["threat_score"] < 0.2

"""
Multimodal fusion.

Fuses a physiological distress signal with a visual distress signal, with two
deliberate safety properties:

1. **Multimodal agreement escalates.**  When both modalities are strongly
   elevated the fused score is boosted, because two independent sensors agreeing
   is much stronger evidence than either alone.
2. **Single-modality alarms are suppressed.**  A high heart rate with a calm
   posture is exercise, not an assault; a raised-arm posture with a calm
   physiology is stretching.  Either alone is capped in the CAUTION band so it
   cannot, by itself, reach THREAT.

All thresholds are named constants in :mod:`threatwatch.config` — the
predecessor implementation inlined magic numbers and carried two dead
parameters (``demo_override`` / ``demo_threat_level``) that were never set at
the only call site.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from threatwatch.config import (
    AGREEMENT_BOOST,
    AGREEMENT_FLOOR,
    AGREEMENT_SENSOR_MIN,
    AGREEMENT_VISION_MIN,
    CAUTION_UPPER_BOUND,
    ROLLING_WINDOW_SIZE,
    SAFE_UPPER_BOUND,
    SENSOR_WEIGHT,
    SUPPRESS_SENSOR_CAP,
    SUPPRESS_SENSOR_MAX,
    SUPPRESS_SENSOR_MIN,
    SUPPRESS_VISION_CAP,
    SUPPRESS_VISION_MAX,
    SUPPRESS_VISION_MIN,
    THREAT_LOWER_BOUND,
    VISION_WEIGHT,
)

VERDICT_SAFE = "SAFE"
VERDICT_CAUTION = "SUSPICIOUS / CAUTION"
VERDICT_THREAT = "THREAT / EMERGENCY"

COLOR_SAFE = "#ccff00"
COLOR_CAUTION = "#00f0ff"
COLOR_THREAT = "#ff003c"


class FusionEngine:
    """Weighted fusion with agreement escalation, suppression and smoothing."""

    def __init__(
        self,
        sensor_weight: float = SENSOR_WEIGHT,
        vision_weight: float = VISION_WEIGHT,
        rolling_window_size: int = ROLLING_WINDOW_SIZE,
    ):
        self.set_weights(sensor_weight, vision_weight)
        self.rolling_window_size = int(max(1, rolling_window_size))
        self._history: List[float] = []

    # ------------------------------------------------------------------ #
    def set_weights(self, sensor_w: float, vision_w: float) -> None:
        total = float(sensor_w) + float(vision_w)
        if total <= 0:
            raise ValueError("Fusion weights must sum to a positive number")
        self.sensor_weight = float(sensor_w) / total
        self.vision_weight = float(vision_w) / total

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------ #
    def fuse(self, sensor_distress: float, vision_distress: float) -> Dict:
        """Combine the two distress signals into a smoothed threat score."""
        s = float(np.clip(sensor_distress, 0.0, 1.0))
        v = float(np.clip(vision_distress, 0.0, 1.0))

        raw = self.sensor_weight * s + self.vision_weight * v

        if s >= AGREEMENT_SENSOR_MIN and v >= AGREEMENT_VISION_MIN:
            # Two independent modalities agree -> escalate.
            raw = min(1.0, max(AGREEMENT_FLOOR, raw * AGREEMENT_BOOST))
            agreement = "MUTUAL_AGREEMENT_HIGH"

        elif s >= SUPPRESS_SENSOR_MIN and v <= SUPPRESS_VISION_MAX:
            # Physiology high, posture calm -> exertion/mild anomaly, not an attack.
            raw = min(raw, SUPPRESS_SENSOR_CAP)
            agreement = "PHYSIOLOGICAL_ANOMALY (NON_THREAT)"

        elif v >= SUPPRESS_VISION_MIN and s <= SUPPRESS_SENSOR_MAX:
            # Posture alarming, physiology calm -> benign motion, not an attack.
            raw = min(raw, SUPPRESS_VISION_CAP)
            agreement = "VISUAL_ANOMALY (BENIGN_MOTION)"

        else:
            agreement = "BASELINE_CONGRUENT"

        # A single-modality anomaly can never exceed the CAUTION ceiling, no
        # matter what the recent history looked like.  The cap is applied to the
        # *smoothed* score as well as the raw one, otherwise a previously high
        # window keeps the average in the THREAT band for several samples after
        # the suppression rule has already engaged.
        cap = None
        if agreement == "PHYSIOLOGICAL_ANOMALY (NON_THREAT)":
            cap = SUPPRESS_SENSOR_CAP
        elif agreement == "VISUAL_ANOMALY (BENIGN_MOTION)":
            cap = SUPPRESS_VISION_CAP

        self._history.append(raw)
        if len(self._history) > self.rolling_window_size:
            self._history.pop(0)
        smoothed = float(np.mean(self._history))
        if cap is not None:
            smoothed = min(smoothed, cap)

        if smoothed <= SAFE_UPPER_BOUND:
            verdict, color = VERDICT_SAFE, COLOR_SAFE
        elif smoothed <= CAUTION_UPPER_BOUND:
            verdict, color = VERDICT_CAUTION, COLOR_CAUTION
        else:
            verdict, color = VERDICT_THREAT, COLOR_THREAT

        return {
            "sensor_score": round(s, 3),
            "vision_score": round(v, 3),
            "sensor_weight": round(self.sensor_weight, 3),
            "vision_weight": round(self.vision_weight, 3),
            "raw_fused_score": round(raw, 3),
            "threat_score": round(smoothed, 3),
            "threat_score_pct": round(smoothed * 100, 1),
            "agreement_status": agreement,
            "verdict": verdict,
            "verdict_color": color,
            "bands": {
                "safe_upper": SAFE_UPPER_BOUND,
                "caution_upper": CAUTION_UPPER_BOUND,
                "threat_lower": THREAT_LOWER_BOUND,
            },
        }

    def get_state(self) -> Dict:
        return {
            "sensor_weight": round(self.sensor_weight, 3),
            "vision_weight": round(self.vision_weight, 3),
            "rolling_window_size": self.rolling_window_size,
            "samples": len(self._history),
            "smoothed": round(float(np.mean(self._history)), 3) if self._history else None,
        }

"""
Distress inference for the *physical* wearable path.

The trained LSTM consumes WESAD-era channels (heart rate, EDA, temperature,
accelerometer) that the deployed MAX30102 + MPU6050 board cannot all reproduce.
Rather than feed the network fabricated EDA — which the predecessor project did
— this module estimates distress from **only what the board actually measures**:
heart-rate dynamics, oxygen saturation when calibrated, impact shocks and
violent rotation.

It is a transparent, documented heuristic, not a learned model, and it says so.
Its output is shaped exactly like the LSTM's so the fusion engine and dashboard
treat both paths identically.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from threatwatch.data.types import SensorPacket

# ---- thresholds (documented, tunable) -------------------------------------- #
HR_SEVERE = 125.0        # BPM -> strong sympathetic arousal
HR_ELEVATED = 105.0
HR_MILD = 90.0
HR_SLIGHT = 80.0
HR_MIN_VALID = 40.0

SPO2_SEVERE = 90.0       # % -> hypoxia
SPO2_MILD = 95.0

IMPACT_HIGH_G = 2.6      # |a| spike consistent with a fall / strike
IMPACT_LOW_G = 0.20      # free-fall / sensor detachment
IMPACT_WINDOW_S = 3.0

GYRO_STRUGGLE_AVG = 130.0   # deg/s, sustained
GYRO_STRUGGLE_PEAK = 180.0
GYRO_MODERATE_AVG = 60.0

HISTORY_LEN = 30


class HardwareDistressInference:
    """Real-time distress estimation from live wearable readings only."""

    def __init__(self, history_len: int = HISTORY_LEN):
        self.history_len = int(history_len)
        self._hr: List[float] = []
        self._acc: List[float] = []
        self._gyro: List[float] = []
        self._last_impact: float = 0.0

    # ------------------------------------------------------------------ #
    def _push(self, buf: List[float], value: Optional[float]) -> None:
        if value is None:
            return
        buf.append(float(value))
        if len(buf) > self.history_len:
            buf.pop(0)

    def _no_data(self) -> Dict[str, Any]:
        return {
            "distress_score": 0.05,
            "distress_pct": 5.0,
            "probabilities": {"baseline": 0.90, "stress": 0.06, "amusement": 0.04},
            "predicted_label": "No Data",
            "hr_bpm": 0.0,
            "spo2_pct": 0.0,
            "motion_mag_g": 0.0,
            "gyro_dps": 0.0,
            "finger_detected": False,
            "latency_ms": 0.1,
            "sub_metrics": {
                "physiological_distress": 0.05,
                "motion_distress": 0.05,
                "impact_shock": False,
                "struggle_detected": False,
                "source": "hardware_heuristic",
            },
        }

    # ------------------------------------------------------------------ #
    def predict(
        self,
        packet: Optional[SensorPacket],
        window: Optional[Sequence[SensorPacket]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if packet is None:
            return self._no_data()

        hr = packet.heart_rate
        spo2 = packet.spo2
        acc_mag = packet.acc_mag
        gyro_mag = packet.gyro_mag
        # ``None`` means "the board did not report this" — never assume contact.
        finger = bool(packet.finger_detected) if packet.finger_detected is not None else False

        self._push(self._hr, hr if (hr is not None and hr > HR_MIN_VALID) else None)
        self._push(self._acc, acc_mag)
        self._push(self._gyro, gyro_mag)

        # ---- 1. physiological ------------------------------------------------- #
        physio = 0.05
        tachycardia = False
        if finger and hr is not None and hr >= HR_MIN_VALID:
            if hr > HR_SEVERE:
                physio, tachycardia = 0.88, True
            elif hr > HR_ELEVATED:
                physio, tachycardia = 0.65, True
            elif hr > HR_MILD:
                physio = 0.35
            elif hr > HR_SLIGHT:
                physio = 0.15
            if spo2 is not None and spo2 > 0:
                if spo2 < SPO2_SEVERE:
                    physio = min(1.0, physio + 0.35)
                elif spo2 < SPO2_MILD:
                    physio = min(1.0, physio + 0.15)

        # ---- 2. biomechanical ------------------------------------------------- #
        motion = 0.05
        impact = False
        struggle = False

        if acc_mag is not None and (acc_mag > IMPACT_HIGH_G or acc_mag < IMPACT_LOW_G):
            self._last_impact = time.time()
        if (time.time() - self._last_impact) < IMPACT_WINDOW_S:
            impact = True
            motion = max(motion, 0.75)

        avg_gyro = float(np.mean(self._gyro)) if self._gyro else 0.0
        if avg_gyro > GYRO_STRUGGLE_AVG or (gyro_mag is not None and gyro_mag > GYRO_STRUGGLE_PEAK):
            struggle = True
            motion = max(motion, 0.60)
        elif avg_gyro > GYRO_MODERATE_AVG:
            motion = max(motion, 0.25)

        # ---- 3. combine ------------------------------------------------------- #
        combined = max(physio, motion, 0.60 * physio + 0.40 * motion)
        combined = float(np.clip(combined, 0.02, 1.0))

        # ---- 4. probabilities ------------------------------------------------- #
        if tachycardia:
            p_tach = float(np.clip(physio, 0.40, 0.90))
            p_mot = float(np.clip(motion, 0.05, 0.30))
            label = "Tachycardia / Elevated HR"
        elif impact:
            p_mot = float(np.clip(motion, 0.60, 0.92))
            p_tach = float(np.clip(physio, 0.05, 0.20))
            label = "Impact Shock / Fall"
        elif struggle:
            p_mot = float(np.clip(motion, 0.50, 0.85))
            p_tach = float(np.clip(physio, 0.05, 0.25))
            label = "Violent Agitation / Struggle"
        else:
            p_mot = float(np.clip(motion, 0.01, 0.15))
            p_tach = float(np.clip(physio, 0.01, 0.15))
            label = "Normal Baseline"

        p_base = max(0.01, 1.0 - (p_tach + p_mot))
        tot = p_base + p_tach + p_mot
        p_base, p_tach, p_mot = p_base / tot, p_tach / tot, p_mot / tot

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "distress_score": round(combined, 3),
            "distress_pct": round(combined * 100, 1),
            "probabilities": {
                "baseline": round(p_base, 3),
                "stress": round(p_tach, 3),
                "amusement": round(p_mot, 3),
            },
            "predicted_label": label,
            "hr_bpm": float(hr) if hr is not None else 0.0,
            "spo2_pct": float(spo2) if spo2 is not None else 0.0,
            "motion_mag_g": float(acc_mag) if acc_mag is not None else 0.0,
            "gyro_dps": float(gyro_mag) if gyro_mag is not None else 0.0,
            "finger_detected": finger,
            "latency_ms": latency_ms,
            "sub_metrics": {
                "physiological_distress": round(physio, 3),
                "motion_distress": round(motion, 3),
                "impact_shock": impact,
                "struggle_detected": struggle,
                "source": "hardware_heuristic",
            },
        }

    def reset(self) -> None:
        self._hr.clear()
        self._acc.clear()
        self._gyro.clear()
        self._last_impact = 0.0

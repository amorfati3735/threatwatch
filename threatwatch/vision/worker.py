"""
Threaded vision worker.

The predecessor project ran YOLOv8 inference *inside* the asyncio telemetry
loop, so a slow frame stalled every connected dashboard.  Here the whole vision
chain — capture, detect, analyse, overlay — runs on its own thread and publishes
an immutable snapshot.  The event loop only reads the latest snapshot, so vision
can never throttle telemetry, and a vision crash cannot take the server down.

The worker degrades gracefully: if OpenCV, torch or Ultralytics are unavailable,
it keeps publishing synthetic frames with a zero distress score and reports the
reason, so the rest of the system still runs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from threatwatch.config import CAMERA_FPS


@dataclass
class VisionSnapshot:
    """Immutable view of the most recent vision result."""

    frame: Optional[np.ndarray]
    analysis: Dict[str, Any] = field(default_factory=dict)
    detection: Optional[Dict[str, Any]] = None
    fps: float = 0.0
    sequence: int = 0
    timestamp: float = 0.0
    enabled: bool = True
    error: str = ""

    @property
    def visual_distress(self) -> float:
        try:
            return float(self.analysis.get("visual_distress_score", 0.0))
        except Exception:
            return 0.0

    def to_payload(self) -> Dict[str, Any]:
        a = self.analysis or {}
        return {
            "distress_score": self.visual_distress,
            "distress_pct": round(self.visual_distress * 100, 1),
            "posture": a.get("posture", "No Person"),
            "facial_expression": a.get("facial_expression", "None"),
            "is_smiling": bool(a.get("is_smiling", False)),
            "movement_level": a.get("movement_level", "None"),
            "motion_velocity": float(a.get("motion_velocity", 0.0)),
            "confidence": float(a.get("confidence", 0.0)),
            "person_detected": bool(a.get("person_detected", False)),
            "is_fallen": bool(a.get("is_fallen", False)),
            "is_defensive": bool(a.get("is_defensive", False)),
            "is_closeup": bool(a.get("is_closeup", False)),
            "enabled": self.enabled,
            "fps": self.fps,
            "error": self.error,
        }


def _blank_analysis(reason: str = "") -> Dict[str, Any]:
    return {
        "person_detected": False,
        "posture": "Vision Off",
        "facial_expression": "None",
        "is_smiling": False,
        "movement_level": "None",
        "motion_velocity": 0.0,
        "visual_distress_score": 0.0,
        "confidence": 0.0,
        "is_fallen": False,
        "is_defensive": False,
        "is_closeup": False,
        "reason": reason,
    }


class VisionWorker:
    """Capture → detect → analyse → overlay, on a dedicated thread."""

    def __init__(
        self,
        camera=None,
        detector=None,
        analyzer=None,
        enabled: bool = True,
        target_fps: float = 15.0,
    ):
        self.enabled = bool(enabled)
        self.target_fps = float(target_fps)
        self._camera = camera
        self._detector = detector
        self._analyzer = analyzer

        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot = VisionSnapshot(frame=None, analysis=_blank_analysis("not started"), enabled=self.enabled)
        self._error = ""
        self._frame_count = 0
        self._last_fps_time = time.time()

    # ------------------------------------------------------------------ #
    def _ensure_components(self) -> bool:
        """Lazily build camera/detector/analyzer; report failure instead of raising."""
        if self._analyzer is not None:
            return True
        try:
            from threatwatch.vision.camera import CameraManager
            from threatwatch.vision.detector import PoseDetector
            from threatwatch.vision.pose_features import PoseFeatureAnalyzer

            self._camera = self._camera or CameraManager()
            self._detector = self._detector or PoseDetector()
            self._analyzer = self._analyzer or PoseFeatureAnalyzer()
            return True
        except Exception as exc:
            self._error = f"vision unavailable: {exc}"
            print(f"[vision] {self._error}")
            return False

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if not self.enabled:
            print("[vision] disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._thread.start()
        print(f"[vision] worker started (target {self.target_fps:g} FPS)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._camera is not None and hasattr(self._camera, "stop"):
            try:
                self._camera.stop()
            except Exception:
                pass
        print("[vision] worker stopped")

    # ------------------------------------------------------------------ #
    def _publish(self, snap: VisionSnapshot) -> None:
        with self._lock:
            self._snapshot = snap

    def latest(self) -> VisionSnapshot:
        with self._lock:
            return self._snapshot

    def _loop(self) -> None:
        if not self._ensure_components():
            self._publish(VisionSnapshot(frame=None, analysis=_blank_analysis(self._error),
                                         enabled=False, error=self._error))
            return

        interval = 1.0 / max(1.0, self.target_fps)
        seq = 0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                frame = self._camera.get_frame()
                detection = self._detector.detect(frame)
                analysis = self._analyzer.analyze(detection, frame)
                annotated = self._analyzer.render_overlay(frame, detection, analysis)

                seq += 1
                self._frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    fps = self._frame_count / (now - self._last_fps_time)
                    self._frame_count = 0
                    self._last_fps_time = now
                else:
                    fps = self.latest().fps

                self._publish(VisionSnapshot(
                    frame=annotated, analysis=analysis, detection=detection,
                    fps=round(fps, 1), sequence=seq, timestamp=now, enabled=True,
                ))
            except Exception as exc:
                # Never let one bad frame kill the worker.
                print(f"[vision] frame error: {exc}")
                self._publish(VisionSnapshot(frame=None, analysis=_blank_analysis(str(exc)),
                                             enabled=True, error=str(exc), sequence=seq))

            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.001, interval - elapsed))

    # ------------------------------------------------------------------ #
    def get_state(self) -> Dict[str, Any]:
        snap = self.latest()
        return {
            "enabled": self.enabled,
            "running": bool(self._thread is not None and self._thread.is_alive()),
            "fps": snap.fps,
            "sequence": snap.sequence,
            "error": snap.error or self._error,
            "has_frame": snap.frame is not None,
            "person_detected": bool((snap.analysis or {}).get("person_detected", False)),
        }

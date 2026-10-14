"""
Threaded Camera Capture Module.
Captures live frames from laptop webcam with fallback test frames if camera is unavailable.
"""

import time
import threading
import cv2
import numpy as np
from typing import Optional, Tuple

from threatwatch.config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


class CameraManager:
    """
    Manages non-blocking threaded frame capture from the laptop webcam.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.is_camera_live = False
        self.fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        
        self.start()

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._init_camera()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _init_camera(self) -> None:
        try:
            # OpenCV capture on Windows with DirectShow backend for faster init
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.is_camera_live = True
                print(f"[CameraManager] Webcam #{self.camera_index} initialized successfully ({CAMERA_WIDTH}x{CAMERA_HEIGHT}).")
            else:
                self.is_camera_live = False
                print(f"[CameraManager] Webcam #{self.camera_index} not available. Using synthetic simulation feed.")
        except Exception as e:
            self.is_camera_live = False
            print(f"[CameraManager] Camera init error: {e}. Using synthetic simulation feed.")

    def _capture_loop(self) -> None:
        while self._running:
            t_start = time.perf_counter()
            frame = None

            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                else:
                    self.is_camera_live = False

            if frame is None:
                # Generate synthetic test frame
                frame = self._generate_fallback_frame()

            with self._lock:
                self.latest_frame = frame
                self._frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self.fps = round(self._frame_count / (now - self._last_fps_time), 1)
                    self._frame_count = 0
                    self._last_fps_time = now

            # Sleep to maintain the configured capture rate
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.001, (1.0 / float(CAMERA_FPS)) - elapsed)
            time.sleep(sleep_time)

    def get_frame(self) -> np.ndarray:
        with self._lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return self._generate_fallback_frame()

    def _generate_fallback_frame(self) -> np.ndarray:
        """Create high-tech simulated test pattern if camera is absent."""
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        frame[:] = (17, 17, 17)  # Dark panel color

        # Draw grid
        for y in range(0, CAMERA_HEIGHT, 40):
            cv2.line(frame, (0, y), (CAMERA_WIDTH, y), (35, 35, 35), 1)
        for x in range(0, CAMERA_WIDTH, 40):
            cv2.line(frame, (x, 0), (x, CAMERA_HEIGHT), (35, 35, 35), 1)

        # Simulation silhouette
        cx, cy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2
        # Head
        cv2.circle(frame, (cx, cy - 80), 30, (0, 240, 255), 2)
        # Torso
        cv2.line(frame, (cx, cy - 50), (cx, cy + 60), (204, 255, 0), 2)
        # Arms
        t_phase = time.time() * 2
        arm_y = int(cy - 20 + 15 * np.sin(t_phase))
        cv2.line(frame, (cx, cy - 30), (cx - 50, arm_y), (204, 255, 0), 2)
        cv2.line(frame, (cx, cy - 30), (cx + 50, arm_y), (204, 255, 0), 2)
        # Legs
        cv2.line(frame, (cx, cy + 60), (cx - 40, cy + 150), (204, 255, 0), 2)
        cv2.line(frame, (cx, cy + 60), (cx + 40, cy + 150), (204, 255, 0), 2)

        cv2.putText(frame, "[SIMULATED CAMERA STREAM]", (cx - 150, cy + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        return frame

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

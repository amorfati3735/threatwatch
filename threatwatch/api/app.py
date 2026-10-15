"""
Application orchestrator and telemetry loop.

Responsibilities
----------------
* Own the sensor source (WESAD replay or a live board) and the matching inference
  path: the trained sequence model for replay, a transparent heuristic for the
  physical board.
* Own the **single** :class:`WindowBuffer` that turns packets into model windows —
  the same code path for replay and hardware.
* Read the latest :class:`VisionSnapshot` from the threaded vision worker.  It
  never runs YOLO on the event loop.
* Fuse, run the state machine, and **dispatch a real alert** on escalation.
* Broadcast one telemetry payload per tick to every connected dashboard.

Every optional dependency degrades instead of crashing: no camera, no YOLO, no
trained checkpoint, no Telegram token — the server still runs and says so.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from threatwatch.alerts import AlertDispatcher, build_dispatcher
from threatwatch.config import (
    CAMERA_FPS,
    FRONTEND_DIR,
    HOST,
    PORT,
    WEBSOCKET_FPS,
    AlertSettings,
    settings,
)
from threatwatch.data.replay_engine import ReplayEngine
from threatwatch.data.sources import HardwareSensorSource
from threatwatch.data.types import WindowBuffer
from threatwatch.fusion.engine import FusionEngine
from threatwatch.fusion.state_machine import AlertStateMachine
from threatwatch.ml.hardware_inference import HardwareDistressInference
from threatwatch.ml.inference import SensorMLPredictor, load_latest_checkpoint
from threatwatch.utils.logging import event_logger
from threatwatch.vision.worker import VisionWorker

SOURCE_REPLAY = "WESAD_REPLAY"
SOURCE_HARDWARE = "HARDWARE_LIVE"

# Deterministic demo scenarios: (sensor distress, vision distress, label)
DEMO_SCENARIOS = {
    "LIVE_FREE": None,
    "NORMAL": (0.08, 0.05),
    "PHYSIOLOGICAL_DISTRESS": (0.85, 0.08),
    "MULTIMODAL_THREAT": (0.88, 0.82),
    "FALSE_POSITIVE_REDUCTION": (0.90, 0.06),
}


class PipelineRunner:
    """The running system: sources, models, fusion, alerts, telemetry."""

    def __init__(self, enable_vision: Optional[bool] = None, cfg: Optional[AlertSettings] = None):
        print("=" * 62)
        print("   THREATWATCH — MULTIMODAL THREAT & DISTRESS DETECTION")
        print("=" * 62)

        # THREATWATCH_VISION=0 disables the camera entirely (headless demo,
        # no webcam, or a machine where the camera is already in use).
        if enable_vision is None:
            enable_vision = os.getenv("THREATWATCH_VISION", "1") != "0"

        self.cfg = cfg or settings()

        # ---- sensor sources ------------------------------------------------- #
        self.replay = ReplayEngine()
        self.hardware = HardwareSensorSource()
        self.active_source = SOURCE_REPLAY

        # ---- inference ------------------------------------------------------ #
        self.checkpoint_path = load_latest_checkpoint()
        self.model: Optional[SensorMLPredictor] = None
        self.model_error = ""
        if self.checkpoint_path is not None:
            try:
                self.model = SensorMLPredictor(self.checkpoint_path)
                print(f"[init] loaded model {self.checkpoint_path.name}")
                for k, v in self.model.describe().items():
                    print(f"        {k}: {v}")
            except Exception as exc:
                self.model_error = str(exc)
                print(f"[init] model load failed: {exc}")
        else:
            self.model_error = "no checkpoint found in models/"
            print("[init] no trained checkpoint found — the sensor path will use the "
                  "documented hardware heuristic and label itself as such.")

        self.hardware_ml = HardwareDistressInference()

        # ---- window buffer (single path for replay + hardware) --------------- #
        self.buffer = self._make_buffer()

        # ---- vision (own thread) -------------------------------------------- #
        self.vision = VisionWorker(enabled=enable_vision, target_fps=float(CAMERA_FPS))
        self.vision.start()

        # ---- decision layer -------------------------------------------------- #
        self.fusion = FusionEngine()
        self.state_machine = AlertStateMachine(cooldown_secs=self.cfg.cooldown_secs)
        self.dispatcher: AlertDispatcher = build_dispatcher(self.cfg, audit=self._audit_alert)

        # ---- runtime --------------------------------------------------------- #
        self.demo_scenario = "LIVE_FREE"
        self._running = True
        self._packets = 0
        self._fps = 0.0
        self._fps_counter = 0
        self._last_fps_calc = time.time()
        self._last_sensor: Optional[Any] = None
        self._last_fusion: Dict[str, Any] = {}
        self._alert_events: list = []

        event_logger.log_event("SYSTEM", "ThreatWatch pipeline initialised.")
        event_logger.log_event(
            "MODEL",
            f"Sensor model: {self.checkpoint_path.name}" if self.model else
            f"Sensor model: none ({self.model_error}) — heuristic fallback active.",
        )
        event_logger.log_event("ALERTS", f"Notifiers: {[n.name for n in self.dispatcher.notifiers]}")

    # ------------------------------------------------------------------ #
    def _make_buffer(self) -> WindowBuffer:
        if self.model is not None:
            return WindowBuffer(
                window_sec=self.model.window_sec,
                target_fs=self.model.target_fs,
                channels=self.model.channels,
            )
        return WindowBuffer(
            window_sec=15.0, target_fs=32.0,
            channels=["hr", "eda", "temp", "acc_x", "acc_y", "acc_z"],
        )

    def _rebuild_buffer(self) -> None:
        self.buffer = self._make_buffer()

    def _audit_alert(self, entry: Dict[str, Any]) -> None:
        self._alert_events.append(entry)
        if len(self._alert_events) > 200:
            self._alert_events.pop(0)
        event_logger.log_event(
            "ALERT-DISPATCH",
            f"{entry['status']}: {entry['title']} ({entry.get('detail','')})",
            level="EMERGENCY" if entry["status"] == "sent" else "WARNING",
        )

    # ------------------------------------------------------------------ #
    # Control surface (called by the REST routes)
    # ------------------------------------------------------------------ #
    def set_source(self, source_type: str) -> bool:
        key = (source_type or "").upper()
        if "HARDWARE" in key or "ARDUINO" in key or "ESP32" in key or "LIVE" in key:
            self.active_source = SOURCE_HARDWARE
            if not self.hardware.is_connected:
                self.hardware.connect()
            event_logger.log_event("SOURCE", "Active sensor modality: HARDWARE LIVE")
        elif "WESAD" in key or "REPLAY" in key:
            self.active_source = SOURCE_REPLAY
            event_logger.log_event("SOURCE", "Active sensor modality: WESAD REPLAY")
        else:
            return False
        # Drop the previous source's last sample too: the dashboard must show
        # "waiting for the board" (not a stale reading from another source) while
        # a new link is coming up.
        self._last_sensor = None
        self.buffer.clear()
        return True

    def set_demo_scenario(self, scenario: str) -> str:
        key = (scenario or "LIVE_FREE").upper()
        if key not in DEMO_SCENARIOS:
            key = "LIVE_FREE"
        self.demo_scenario = key
        # Reset smoothing so a previous scenario cannot leak into this one.
        self.fusion.reset()
        self.buffer.clear()
        if key == "NORMAL":
            self.replay.seek_to_scenario("BASELINE")
        elif key in ("PHYSIOLOGICAL_DISTRESS", "MULTIMODAL_THREAT", "FALSE_POSITIVE_REDUCTION"):
            self.replay.seek_to_scenario("STRESS")
        event_logger.log_event("DEMO", f"Active scenario: {key}", level="WARNING")
        return key

    def set_weights(self, sensor_w: float, vision_w: float) -> None:
        self.fusion.set_weights(sensor_w, vision_w)
        event_logger.log_event(
            "FUSION", f"Fusion weights: sensor={self.fusion.sensor_weight:.2f} vision={self.fusion.vision_weight:.2f}"
        )

    def reset_alert(self) -> None:
        self.state_machine.reset_state()
        # Drop an active demo override as well, otherwise the injected threat
        # score re-escalates the state machine on the very next tick and the
        # alarm appears not to reset at all.  (The buffer is deliberately kept:
        # the live window is still valid and clearing it would blind the model
        # for the next 15 s.)
        if self.demo_scenario != "LIVE_FREE":
            self.demo_scenario = "LIVE_FREE"
            self.fusion.reset()
        event_logger.log_event("ALERT", "Manual alert reset by operator.")

    def trigger_test_alert(self) -> bool:
        """Deliver a test alert so the channel can be verified before a demo."""
        from threatwatch.alerts import AlertMessage

        msg = AlertMessage(
            title="Test alert",
            body="This is a manual test of the ThreatWatch alert channel.",
            level="CRITICAL",
            state="TEST",
            threat_score=0.0,
            source=self.active_source,
        )
        return self.dispatcher.dispatch_sync(msg)

    # ------------------------------------------------------------------ #
    # Sensor + inference
    # ------------------------------------------------------------------ #
    def _sensor_distress(self) -> tuple[float, Optional[Any], Dict[str, Any]]:
        if self.active_source == SOURCE_HARDWARE:
            pkt = self.hardware.get_next_sample() if self.hardware.is_connected else None
            if pkt is None:
                # Deliberately does NOT fall through to the replay engine: replay
                # packets are stamped in recording seconds, so feeding them into
                # the live buffer would splice two time bases together (a scan
                # for the board takes ~10-15 s, and the span raced to ~1.7e9).
                pred = {
                    "distress_score": 0.0, "distress_pct": 0.0, "probabilities": {},
                    "predicted_label": "Waiting for board", "latency_ms": 0.0,
                    "source": "HARDWARE (no samples yet)", "ready": False,
                }
                return 0.0, None, pred
            pred = self.hardware_ml.predict(pkt)
            return float(pred["distress_score"]), pkt, pred

        pkt = self.replay.get_next_sample()
        if self.model is not None:
            window = self.buffer.get_window()
            if window is None:
                probs = {"baseline": 0.0, "stress": 0.0, "amusement": 0.0}
                pred = {
                    "distress_score": 0.0, "distress_pct": 0.0, "probabilities": probs,
                    "predicted_label": "Buffering", "latency_ms": 0.0,
                    "source": f"MODEL:{self.model.model_name}", "ready": False,
                }
            else:
                cls, smooth = self.model.predict(window)
                pred = {
                    "distress_score": self.model.stress_index(smooth),
                    "distress_pct": round(self.model.stress_index(smooth) * 100, 1),
                    "probabilities": {n: round(float(p), 3) for n, p in zip(self.model.class_names, smooth)},
                    "predicted_label": self.model.class_names[cls],
                    "latency_ms": 0.0,
                    "source": f"MODEL:{self.model.model_name}",
                    "ready": True,
                }
            return float(pred["distress_score"]), pkt, pred

        # No trained model: use the documented heuristic, and say so.
        pred = self.hardware_ml.predict(pkt)
        pred["source"] = "HEURISTIC (no trained model)"
        pred["ready"] = True
        return float(pred["distress_score"]), pkt, pred

    # ------------------------------------------------------------------ #
    async def loop(self) -> None:
        interval = 1.0 / max(1, WEBSOCKET_FPS)
        print(f"[pipeline] telemetry loop started at {WEBSOCKET_FPS} FPS "
              f"(source={self.active_source})")

        while self._running:
            try:
                t0 = time.perf_counter()
                elapsed_tick = interval

                # 1. sensor -> buffer -> distress score
                sensor_distress, pkt, sensor_pred = self._sensor_distress()
                if pkt is not None:
                    self._last_sensor = pkt
                    self.buffer.add(pkt)

                # 2. vision comes from the worker; this call never blocks on YOLO
                snap = self.vision.latest()

                # 3. deterministic demo override (clearly labelled downstream)
                override = DEMO_SCENARIOS.get(self.demo_scenario)
                if override is not None:
                    sensor_distress, vision_distress = override
                    sensor_pred = dict(sensor_pred)
                    sensor_pred["source"] = f"DEMO_OVERRIDE (was {sensor_pred.get('source')})"
                else:
                    vision_distress = snap.visual_distress

                # 4. fusion
                fusion_result = self.fusion.fuse(sensor_distress, vision_distress)
                self._last_fusion = fusion_result

                # 5. state machine -> alert dispatch
                state_event = self.state_machine.update(fusion_result["threat_score"])
                if state_event is not None:
                    event_logger.log_event("ALERT", state_event["message"], level=state_event["level"])
                    if state_event.get("dispatch"):
                        self.dispatcher.dispatch(
                            self.dispatcher.build_message(state_event, source=self.active_source)
                        )

                # 6. mirror the verdict onto the wearable OLED
                if self.active_source == SOURCE_HARDWARE and self.hardware.is_connected:
                    # The OLED prints the first 10 characters of the state, so send
                    # the leading verdict token ("THREAT") rather than the full
                    # "THREAT / EMERGENCY", which rendered as "THREAT / E".
                    device_state = str(fusion_result["verdict"]).split(" / ")[0]
                    self.hardware.write_feedback(
                        int(fusion_result["threat_score_pct"]), device_state
                    )

                # 7. advance replay by exactly the simulated time this tick represents
                if self.active_source == SOURCE_REPLAY and self.replay.is_loaded:
                    self.replay.advance_time(elapsed_tick, self.replay.speed)

                # 8. counters
                self._packets += 1
                self._fps_counter += 1
                now = time.time()
                if now - self._last_fps_calc >= 1.0:
                    self._fps = round(self._fps_counter / (now - self._last_fps_calc), 1)
                    self._fps_counter = 0
                    self._last_fps_calc = now

                # 9. broadcast
                await ws_manager.broadcast_json(self.build_payload(snap, sensor_pred, fusion_result, now))

                elapsed = time.perf_counter() - t0
                # Use the tick actually achieved (not the nominal interval) so a
                # slow frame cannot make replay drift behind wall time.
                elapsed_tick = max(interval, elapsed)
                await asyncio.sleep(max(0.001, interval - elapsed))

            except Exception as exc:  # never let the loop die
                print(f"[pipeline] error: {exc}")
                await asyncio.sleep(0.2)

    # ------------------------------------------------------------------ #
    def build_payload(self, snap, sensor_pred: Dict[str, Any], fusion_result: Dict[str, Any], now: float) -> Dict[str, Any]:
        frame_b64 = ""
        if snap.frame is not None:
            try:
                ok, buf = cv2.imencode(".jpg", snap.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
            except Exception:
                pass

        return {
            "type": "TELEMETRY",
            "timestamp": now,
            "fps": self._fps,
            "packet_counter": self._packets,
            "active_source_type": self.active_source,
            "demo_scenario": self.demo_scenario,
            "synthetic_override": DEMO_SCENARIOS.get(self.demo_scenario) is not None,
            "camera_frame": frame_b64,
            "sensor_packet": self._last_sensor.to_dict() if self._last_sensor else {},
            "sensor_ml": {
                "distress_score": round(float(sensor_pred.get("distress_score", 0.0)), 3),
                "distress_pct": round(float(sensor_pred.get("distress_score", 0.0)) * 100, 1),
                "probabilities": sensor_pred.get("probabilities", {}),
                "predicted_label": sensor_pred.get("predicted_label", "-"),
                "latency_ms": sensor_pred.get("latency_ms", 0.0),
                "source": sensor_pred.get("source", "-"),
                "ready": sensor_pred.get("ready", False),
            },
            "vision_ml": snap.to_payload(),
            "fusion": fusion_result,
            "state_machine": self.state_machine.get_state_summary(),
            "model": self.model.describe() if self.model else {"available": False, "error": self.model_error},
            "window": self.buffer.get_state(),
            "replay": self.replay.get_state(),
            "arduino": self.hardware.get_state(),
            "alerts": self.dispatcher.get_state(),
            "vision_worker": self.vision.get_state(),
            "recent_logs": event_logger.get_recent_logs(20),
        }

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        self._running = False
        self.vision.stop()
        self.replay.stop()
        self.hardware.disconnect()
        self.dispatcher.shutdown()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
from contextlib import asynccontextmanager  # noqa: E402

from threatwatch.api.routes import router as api_router, set_pipeline_runner  # noqa: E402
from threatwatch.api.websocket import manager as ws_manager  # noqa: E402

runner: Optional[PipelineRunner] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the pipeline on startup and tear it down on shutdown."""
    global runner
    runner = PipelineRunner()
    set_pipeline_runner(runner)
    task = asyncio.create_task(runner.loop())
    try:
        yield
    finally:
        runner._running = False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        runner.shutdown()


app = FastAPI(
    title="ThreatWatch — Multimodal Threat & Distress Detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

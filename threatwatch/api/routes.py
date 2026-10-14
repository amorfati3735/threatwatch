"""
REST control surface.

The endpoint paths used by the dashboard are preserved from the predecessor
project, so the frontend needed no rewiring.  New in the merge:

* ``GET  /api/alerts/status`` — notifier configuration and delivery counters
* ``POST /api/alerts/test``   — deliver a test alert (verify Telegram *before* a demo)
* ``GET  /api/model``         — which model is live, and on which channels/window
* ``GET  /api/hardware/ports``— discover serial ports
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from threatwatch.config import CLASS_NAMES
from threatwatch.data.sources import HardwareSensorSource
from threatwatch.utils.logging import event_logger

router = APIRouter(prefix="/api")

# Injected at startup by api/app.py
pipeline_runner = None


def set_pipeline_runner(runner) -> None:
    global pipeline_runner
    pipeline_runner = runner


def _require_runner():
    if pipeline_runner is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline_runner


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ReplayControlRequest(BaseModel):
    action: str  # play | pause | reset | speed | seek | subject | scenario
    speed: Optional[float] = None
    seek_fraction: Optional[float] = None
    subject_id: Optional[str] = None
    scenario: Optional[str] = None


class HardwareConnectRequest(BaseModel):
    action: str  # connect | disconnect
    port: Optional[str] = None
    baudrate: Optional[int] = 115200


class SourceSelectRequest(BaseModel):
    source_type: str  # WESAD_REPLAY | HARDWARE_LIVE


class WeightConfigRequest(BaseModel):
    sensor_weight: float
    vision_weight: float


class DemoScenarioRequest(BaseModel):
    scenario: str  # NORMAL | PHYSIOLOGICAL_DISTRESS | MULTIMODAL_THREAT | FALSE_POSITIVE_REDUCTION | LIVE_FREE


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
@router.get("/replay/subjects")
def list_subjects() -> Dict[str, List[str]]:
    if pipeline_runner is None:
        return {"subjects": []}
    return {"subjects": pipeline_runner.replay.list_subjects()}


@router.post("/replay/control")
def control_replay(req: ReplayControlRequest) -> Dict[str, Any]:
    r = _require_runner()
    rep = r.replay
    action = req.action.lower()

    if action == "play":
        rep.start()
    elif action == "pause":
        rep.stop()
    elif action == "reset":
        rep.reset()
    elif action == "speed" and req.speed is not None:
        rep.set_speed(req.speed)
    elif action == "seek" and req.seek_fraction is not None:
        rep.seek_fraction(req.seek_fraction)
    elif action == "subject" and req.subject_id:
        if not rep.load_subject(req.subject_id):
            raise HTTPException(status_code=400, detail=f"Could not load subject {req.subject_id}")
        r.buffer.clear()
    elif action == "scenario" and req.scenario:
        if not rep.seek_to_scenario(req.scenario):
            raise HTTPException(status_code=400, detail=f"No segment found for scenario {req.scenario}")
        r.buffer.clear()
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported replay action: {req.action}")

    event_logger.log_event("REPLAY", f"replay/{action} {req.model_dump(exclude_none=True)}")
    return {"status": "success", "replay_state": rep.get_state()}


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
@router.get("/hardware/ports")
@router.get("/arduino/ports")
def list_hardware_ports() -> Dict[str, Any]:
    return {"ports": HardwareSensorSource.list_available_ports()}


@router.post("/hardware/connect")
@router.post("/arduino/connect")
def control_hardware(req: HardwareConnectRequest) -> Dict[str, Any]:
    r = _require_runner()
    action = req.action.lower()

    if action == "connect":
        if not r.hardware.connect(req.port):
            raise HTTPException(status_code=400, detail=f"Failed to open {req.port or r.hardware.port}: {r.hardware.get_state()['error']}")
        r.set_source("HARDWARE_LIVE")
        return {"status": "success", "connected": True, "port": r.hardware.port,
                "active_source": r.active_source}
    if action == "disconnect":
        r.hardware.disconnect()
        r.set_source("WESAD_REPLAY")
        return {"status": "success", "connected": False, "active_source": r.active_source}

    raise HTTPException(status_code=400, detail="action must be 'connect' or 'disconnect'")


# --------------------------------------------------------------------------- #
# Source / demo / fusion
# --------------------------------------------------------------------------- #
@router.post("/source/select")
def select_source(req: SourceSelectRequest) -> Dict[str, Any]:
    r = _require_runner()
    if not r.set_source(req.source_type):
        raise HTTPException(status_code=400, detail=f"Invalid source type: {req.source_type}")
    return {"status": "success", "active_source_type": r.active_source,
            "hardware_connected": r.hardware.is_connected}


@router.post("/demo/scenario")
def trigger_demo_scenario(req: DemoScenarioRequest) -> Dict[str, Any]:
    r = _require_runner()
    active = r.set_demo_scenario(req.scenario)
    return {"status": "success", "active_scenario": active}


@router.post("/fusion/weights")
def set_weights(req: WeightConfigRequest) -> Dict[str, Any]:
    r = _require_runner()
    r.set_weights(req.sensor_weight, req.vision_weight)
    return {"status": "success", "sensor_weight": r.fusion.sensor_weight,
            "vision_weight": r.fusion.vision_weight}


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
@router.post("/alert/reset")
def reset_alert() -> Dict[str, Any]:
    r = _require_runner()
    r.reset_alert()
    return {"status": "success", "alert_state": r.state_machine.get_state_summary()}


@router.get("/alerts/status")
def alert_status() -> Dict[str, Any]:
    r = _require_runner()
    return r.dispatcher.get_state()


@router.post("/alerts/test")
def alert_test() -> Dict[str, Any]:
    r = _require_runner()
    delivered = r.trigger_test_alert()
    return {"status": "success" if delivered else "failed", "delivered": delivered,
            "notifiers": r.dispatcher.get_state()["notifiers"]}


# --------------------------------------------------------------------------- #
# Introspection / logs
# --------------------------------------------------------------------------- #
@router.get("/model")
def model_info() -> Dict[str, Any]:
    r = _require_runner()
    info = r.model.describe() if r.model else {"available": False, "error": r.model_error}
    info["class_names"] = CLASS_NAMES if r.model else None
    return info


@router.get("/logs")
def get_logs() -> Dict[str, Any]:
    return {"logs": event_logger.get_recent_logs(60)}


@router.get("/logs/export/json")
def export_json_logs() -> FileResponse:
    path = event_logger.export_json()
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.get("/logs/export/csv")
def export_csv_logs() -> FileResponse:
    path = event_logger.export_csv()
    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.get("/status")
def system_status() -> Dict[str, Any]:
    if pipeline_runner is None:
        return {"status": "INITIALISING"}
    r = pipeline_runner
    return {
        "status": "ONLINE",
        "active_source_type": r.active_source,
        "data_source": {
            "wearable": (
                f"Live hardware on {r.hardware.port}" if r.hardware.is_connected
                else f"WESAD replay ({r.replay.subject_id})"
            ),
            "camera": "live" if r.vision.get_state()["has_frame"] else "unavailable",
            "vision_worker": r.vision.get_state(),
        },
        "models": {
            "sensor": r.model.describe() if r.model else {"available": False, "error": r.model_error},
            "vision": "YOLOv8-pose + heuristic pose analyser",
            "fusion": "weighted multimodal decision engine",
        },
        "alerts": r.dispatcher.get_state(),
        "state_machine": r.state_machine.get_state_summary(),
        "window": r.buffer.get_state(),
    }

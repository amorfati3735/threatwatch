"""
WebSocket Connection Manager for High-Frequency Telemetry Streaming.
Includes robust serialization for NumPy and native Python types.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Any
import json
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling NumPy scalars, arrays, and boolean types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ConnectionManager:
    """Manages active browser WebSocket clients."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):
        """Broadcast payload to all connected frontend clients."""
        try:
            text_data = json.dumps(data, cls=NumpyJSONEncoder)
        except Exception as e:
            print(f"[WebSocket Encode Error] {e}")
            return

        for connection in list(self.active_connections):
            try:
                await connection.send_text(text_data)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()

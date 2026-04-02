"""
WebSocket — real-time live event stream.

Endpoint: WS /ws/live

Broadcasts JSON events to all connected clients whenever:
  - Employee checked in
  - Employee checked out (manual or auto)
  - Break started / ended
  - Employee recognized on any camera (live location)
  - Unknown person detected

Message format:
  {"event": "checkin",    "employee_id": 7, "employee_name": "...", "camera_id": 2, "camera_label": "entry",  "timestamp": "..."}
  {"event": "checkout",   "employee_id": 7, "employee_name": "...", "total_hours": 0.36, "auto": false, "camera_id": 3, "camera_label": "exit", "timestamp": "..."}
  {"event": "break_start","employee_id": 7, "employee_name": "...", "camera_id": 3, "timestamp": "..."}
  {"event": "break_end",  "employee_id": 7, "employee_name": "...", "duration_min": 12.3, "break_type": "medium", "camera_id": 1, "camera_label": "entry", "timestamp": "..."}
  {"event": "detected",   "employee_id": 7, "employee_name": "...", "camera_id": 2, "camera_label": "entry", "confidence": 0.64, "timestamp": "..."}
  {"event": "unknown",    "camera_id": 2, "camera_label": "entry", "timestamp": "..."}
"""
import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.utils.logger import logger

router = APIRouter(tags=["WebSocket"])

# ── Event loop reference (set in main.py lifespan) ────────────────────────────
# Background threads (attendance-worker, pipeline) use this to schedule
# coroutines on the async event loop safely via run_coroutine_threadsafe.
_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)
        logger.info(f"WS: client connected ({len(self._clients)} total)")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)
        logger.info(f"WS: client disconnected ({len(self._clients)} remaining)")

    async def broadcast(self, message: dict) -> None:
        if not self._clients:
            return
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def send_event(self, message: dict) -> None:
        """
        Thread-safe broadcast from background threads.
        Schedules the coroutine on the stored event loop.
        No-op if no loop or no clients.
        """
        if _loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), _loop)


manager = ConnectionManager()


# ── Helpers for building event payloads ──────────────────────────────────────

def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def emit_checkin(employee_id: int, employee_name: str, camera_id: int, camera_label: str) -> None:
    manager.send_event({
        "event": "checkin",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "camera_id": camera_id,
        "camera_label": camera_label,
        "timestamp": _ts(),
    })


def emit_checkout(employee_id: int, employee_name: str, total_hours: float, auto: bool = False, camera_id: int | None = None, camera_label: str | None = None) -> None:
    msg: dict = {
        "event": "checkout",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "total_hours": round(total_hours, 4),
        "auto": auto,
        "timestamp": _ts(),
    }
    if camera_id is not None:
        msg["camera_id"] = camera_id
        msg["camera_label"] = camera_label
    manager.send_event(msg)


def emit_break_start(employee_id: int, employee_name: str, camera_id: int) -> None:
    manager.send_event({
        "event": "break_start",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "camera_id": camera_id,
        "timestamp": _ts(),
    })


def emit_break_end(employee_id: int, employee_name: str, duration_min: float, break_type: str, camera_id: int | None = None, camera_label: str | None = None) -> None:
    msg: dict = {
        "event": "break_end",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "duration_min": round(duration_min, 2),
        "break_type": break_type,
        "timestamp": _ts(),
    }
    if camera_id is not None:
        msg["camera_id"] = camera_id
        msg["camera_label"] = camera_label
    manager.send_event(msg)


def emit_detected(employee_id: int, employee_name: str, camera_id: int, camera_label: str, confidence: float) -> None:
    manager.send_event({
        "event": "detected",
        "employee_id": employee_id,
        "employee_name": employee_name,
        "camera_id": camera_id,
        "camera_label": camera_label,
        "confidence": round(confidence, 4),
        "timestamp": _ts(),
    })


def emit_unknown(camera_id: int, camera_label: str) -> None:
    manager.send_event({
        "event": "unknown",
        "camera_id": camera_id,
        "camera_label": camera_label,
        "timestamp": _ts(),
    })


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Real-time event stream. Connect and receive JSON events as they happen.
    Send any message to get the current status of all employees (ping).
    """
    await manager.connect(websocket)
    # Send current employee statuses on connect
    try:
        from app.database.connection import is_db_enabled
        if is_db_enabled():
            from app.services.attendance_service import list_attendance
            snapshot = list_attendance()
            await websocket.send_text(json.dumps({
                "event": "snapshot",
                "data": snapshot,
                "timestamp": _ts(),
            }, default=str))
    except Exception as e:
        logger.warning(f"WS snapshot error: {e}")

    try:
        while True:
            # Keep alive — client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

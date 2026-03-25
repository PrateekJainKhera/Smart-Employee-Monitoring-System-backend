"""
In-memory application state — used for Phases 1–4.
Replaced by SQLAlchemy in Phase 5 (function signatures stay the same).
"""
from threading import Lock
from datetime import datetime
from typing import Optional


class AppState:
    def __init__(self):
        self._lock = Lock()

        # { employee_id: { id, name, department, designation, email, face_registered, created_at } }
        self._employees: dict[int, dict] = {}

        # { camera_id: { id, name, location_label, rtsp_url, is_active } }
        self._cameras: dict[int, dict] = {}

        # "track_id@camera_id" → employee_id
        self._track_map: dict[str, int] = {}

        self._next_employee_id = 1
        self._next_camera_id = 1

    # ------------------------------------------------------------------
    # Employees
    # ------------------------------------------------------------------

    def add_employee(self, name: str, department: str = "", designation: str = "", email: str = "") -> dict:
        with self._lock:
            emp = {
                "id": self._next_employee_id,
                "name": name,
                "department": department,
                "designation": designation,
                "email": email,
                "face_registered": False,
                "created_at": datetime.utcnow().isoformat(),
            }
            self._employees[self._next_employee_id] = emp
            self._next_employee_id += 1
            return emp.copy()

    def get_employee(self, employee_id: int) -> Optional[dict]:
        return self._employees.get(employee_id)

    def list_employees(self) -> list[dict]:
        return [e.copy() for e in self._employees.values()]

    def update_employee(self, employee_id: int, **kwargs) -> Optional[dict]:
        with self._lock:
            emp = self._employees.get(employee_id)
            if emp is None:
                return None
            emp.update({k: v for k, v in kwargs.items() if k in emp})
            return emp.copy()

    def delete_employee(self, employee_id: int) -> bool:
        with self._lock:
            if employee_id not in self._employees:
                return False
            del self._employees[employee_id]
            return True

    def mark_face_registered(self, employee_id: int) -> None:
        with self._lock:
            if employee_id in self._employees:
                self._employees[employee_id]["face_registered"] = True

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def add_camera(self, name: str, location_label: str, rtsp_url: str) -> dict:
        with self._lock:
            cam = {
                "id": self._next_camera_id,
                "name": name,
                "location_label": location_label,
                "rtsp_url": rtsp_url,
                "is_active": True,
            }
            self._cameras[self._next_camera_id] = cam
            self._next_camera_id += 1
            return cam.copy()

    def get_camera(self, camera_id: int) -> Optional[dict]:
        return self._cameras.get(camera_id)

    def list_cameras(self) -> list[dict]:
        return [c.copy() for c in self._cameras.values()]

    def update_camera(self, camera_id: int, **kwargs) -> Optional[dict]:
        with self._lock:
            cam = self._cameras.get(camera_id)
            if cam is None:
                return None
            cam.update({k: v for k, v in kwargs.items() if k in cam})
            return cam.copy()

    def delete_camera(self, camera_id: int) -> bool:
        with self._lock:
            if camera_id not in self._cameras:
                return False
            del self._cameras[camera_id]
            return True

    # ------------------------------------------------------------------
    # Track identity map
    # ------------------------------------------------------------------

    def set_track_identity(self, track_key: str, employee_id: int) -> None:
        with self._lock:
            self._track_map[track_key] = employee_id

    def get_track_identity(self, track_key: str) -> Optional[int]:
        return self._track_map.get(track_key)

    def clear_track(self, track_key: str) -> None:
        with self._lock:
            self._track_map.pop(track_key, None)

    def clear_all_tracks(self) -> None:
        with self._lock:
            self._track_map.clear()


# Singleton — import this everywhere
state = AppState()

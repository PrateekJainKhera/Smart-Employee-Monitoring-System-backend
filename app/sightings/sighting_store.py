"""
SightingStore — tracks when each employee was seen on each camera today.

Stores:
  - count: how many times seen
  - first_seen: datetime of first detection on that camera today
  - last_seen: datetime of most recent detection

Resets on backend restart. Thread-safe.
"""
import threading
from datetime import date, datetime
from collections import defaultdict


class SightingStore:
    def __init__(self):
        self._lock = threading.Lock()
        # {(employee_id, camera_id, date_str): {"count": int, "first_seen": datetime, "last_seen": datetime}}
        self._data: dict[tuple, dict] = {}

    def record(self, employee_id: int, camera_id: int) -> None:
        """Record a sighting — updates count and last_seen, sets first_seen on first call."""
        key = (employee_id, camera_id, date.today().isoformat())
        now = datetime.utcnow()
        with self._lock:
            if key not in self._data:
                self._data[key] = {"count": 1, "first_seen": now, "last_seen": now}
            else:
                self._data[key]["count"] += 1
                self._data[key]["last_seen"] = now

    def get(self, employee_id: int, camera_id: int, date_str: str | None = None) -> int:
        """Get sighting count for employee on a specific camera and date."""
        d = date_str or date.today().isoformat()
        key = (employee_id, camera_id, d)
        with self._lock:
            return self._data.get(key, {}).get("count", 0)

    def get_by_employee(self, employee_id: int, date_str: str | None = None) -> dict[int, int]:
        """
        Get all sighting counts for an employee across all cameras for a date.
        Returns {camera_id: count}.
        """
        d = date_str or date.today().isoformat()
        with self._lock:
            return {
                cam_id: entry["count"]
                for (emp_id, cam_id, dt), entry in self._data.items()
                if emp_id == employee_id and dt == d
            }

    def get_all_today(self) -> list[dict]:
        """
        Return all sighting records for today as a list of dicts.
        [{employee_id, camera_id, date, count, first_seen, last_seen}, ...]
        """
        today = date.today().isoformat()
        with self._lock:
            return [
                {
                    "employee_id": emp_id,
                    "camera_id": cam_id,
                    "date": dt,
                    "count": entry["count"],
                    "first_seen": entry["first_seen"].isoformat() + "Z",
                    "last_seen": entry["last_seen"].isoformat() + "Z",
                }
                for (emp_id, cam_id, dt), entry in self._data.items()
                if dt == today
            ]

    def get_last_seen(self, employee_id: int) -> datetime | None:
        """Return the most recent detection time for an employee across all cameras today."""
        today = date.today().isoformat()
        with self._lock:
            times = [
                entry["last_seen"]
                for (emp_id, _, dt), entry in self._data.items()
                if emp_id == employee_id and dt == today
            ]
        return max(times) if times else None

    def get_timeline(self, employee_id: int, date_str: str | None = None) -> list[dict]:
        """
        Return ordered camera visits for an employee today, sorted by first_seen.
        [{camera_id, first_seen, last_seen, count}, ...]
        """
        d = date_str or date.today().isoformat()
        with self._lock:
            rows = [
                {
                    "camera_id": cam_id,
                    "count": entry["count"],
                    "first_seen": entry["first_seen"].isoformat() + "Z",
                    "last_seen": entry["last_seen"].isoformat() + "Z",
                }
                for (emp_id, cam_id, dt), entry in self._data.items()
                if emp_id == employee_id and dt == d
            ]
        return sorted(rows, key=lambda x: x["first_seen"])


sighting_store = SightingStore()

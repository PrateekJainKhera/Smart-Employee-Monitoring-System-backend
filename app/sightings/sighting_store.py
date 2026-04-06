"""
SightingStore — in-memory counter tracking how many times each employee
is seen on each camera per day.

Key: (employee_id, camera_id, date_str)
Value: count (int)

Resets on backend restart. Thread-safe.
"""
import threading
from datetime import date
from collections import defaultdict


class SightingStore:
    def __init__(self):
        self._lock = threading.Lock()
        # {(employee_id, camera_id, date_str): count}
        self._counts: dict[tuple, int] = defaultdict(int)

    def record(self, employee_id: int, camera_id: int) -> None:
        """Increment sighting count for this employee + camera today."""
        key = (employee_id, camera_id, date.today().isoformat())
        with self._lock:
            self._counts[key] += 1

    def get(self, employee_id: int, camera_id: int, date_str: str | None = None) -> int:
        """Get sighting count for employee on a specific camera and date."""
        d = date_str or date.today().isoformat()
        key = (employee_id, camera_id, d)
        with self._lock:
            return self._counts.get(key, 0)

    def get_by_employee(self, employee_id: int, date_str: str | None = None) -> dict[int, int]:
        """
        Get all sighting counts for an employee across all cameras for a date.
        Returns {camera_id: count}.
        """
        d = date_str or date.today().isoformat()
        with self._lock:
            return {
                cam_id: count
                for (emp_id, cam_id, dt), count in self._counts.items()
                if emp_id == employee_id and dt == d
            }

    def get_all_today(self) -> list[dict]:
        """
        Return all sighting records for today as a list of dicts.
        [{employee_id, camera_id, date, count}, ...]
        """
        today = date.today().isoformat()
        with self._lock:
            return [
                {"employee_id": emp_id, "camera_id": cam_id, "date": dt, "count": count}
                for (emp_id, cam_id, dt), count in self._counts.items()
                if dt == today
            ]


sighting_store = SightingStore()

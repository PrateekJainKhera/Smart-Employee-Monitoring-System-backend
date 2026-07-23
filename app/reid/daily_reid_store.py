"""
DailyReIDStore — per-employee OSNet embedding storage that resets each day.

Design:
  - One embedding per employee per day
  - On first check-in, face recognition succeeds → body crop → embedding stored
  - If a higher-quality body crop appears later that day → embedding updated
  - At midnight (or next day's first access), old embedding is discarded
  - Thread-safe for concurrent pipeline access

Quality score = body crop pixel area (larger crop = more detail = better reference).
"""
import threading
from datetime import date, datetime

import numpy as np

from app.utils.logger import logger


class DailyReIDStore:
    def __init__(self):
        self._lock    = threading.Lock()
        self._entries: dict[int, dict] = {}
        # {
        #   employee_id: {
        #     "embedding":     np.ndarray (512,),
        #     "date":          date,
        #     "quality_score": float,    # body crop area in pixels
        #     "updated_at":    datetime,
        #   }
        # }
        self._gallery_cache: dict[int, np.ndarray] | None = None
        self._gallery_dirty: bool = True

    def _is_today(self, entry: dict) -> bool:
        return entry["date"] == date.today()

    def upsert(
        self,
        employee_id: int,
        embedding: np.ndarray,
        quality_score: float,
    ) -> bool:
        """
        Store or update today's embedding for an employee.

        Updates only if:
          - No entry exists for today (new day or first seen today), OR
          - New quality score is higher than stored one (better body crop)

        Returns True if the embedding was stored/updated.
        """
        if embedding is None:
            return False

        with self._lock:
            existing = self._entries.get(employee_id)

            # New day or no entry — always store
            if existing is None or not self._is_today(existing):
                self._entries[employee_id] = {
                    "embedding":     embedding.copy(),
                    "date":          date.today(),
                    "quality_score": quality_score,
                    "updated_at":    datetime.utcnow(),
                }
                self._gallery_dirty = True
                logger.debug(
                    f"DailyReIDStore: stored embedding for emp={employee_id} "
                    f"quality={quality_score:.0f}"
                )
                return True

            # Same day — update only if better quality
            if quality_score > existing["quality_score"]:
                existing["embedding"]     = embedding.copy()
                existing["quality_score"] = quality_score
                existing["updated_at"]    = datetime.utcnow()
                self._gallery_dirty = True
                logger.debug(
                    f"DailyReIDStore: updated embedding for emp={employee_id} "
                    f"quality={quality_score:.0f} (was {existing['quality_score']:.0f})"
                )
                return True

        return False

    def get(self, employee_id: int) -> np.ndarray | None:
        """Return today's embedding for an employee, or None if not available."""
        with self._lock:
            entry = self._entries.get(employee_id)
            if entry is None or not self._is_today(entry):
                return None
            return entry["embedding"]

    def get_all(self) -> dict[int, np.ndarray]:
        """Return {employee_id: embedding} for all employees with a today entry.
        Result is cached and only rebuilt when an embedding is added/updated."""
        with self._lock:
            if not self._gallery_dirty and self._gallery_cache is not None:
                return self._gallery_cache
            self._gallery_cache = {
                emp_id: entry["embedding"]
                for emp_id, entry in self._entries.items()
                if self._is_today(entry)
            }
            self._gallery_dirty = False
            return self._gallery_cache

    def has_today(self, employee_id: int) -> bool:
        """True if a valid today embedding exists for this employee."""
        with self._lock:
            entry = self._entries.get(employee_id)
            return entry is not None and self._is_today(entry)

    def purge_stale(self) -> int:
        """Remove entries from previous days. Returns count removed."""
        with self._lock:
            stale = [eid for eid, e in self._entries.items() if not self._is_today(e)]
            for eid in stale:
                del self._entries[eid]
            if stale:
                self._gallery_dirty = True
                logger.info(f"DailyReIDStore: purged {len(stale)} stale entries")
            return len(stale)

    def stats(self) -> dict:
        with self._lock:
            today_entries = {eid: e for eid, e in self._entries.items() if self._is_today(e)}
            return {
                "today_count": len(today_entries),
                "employees":   list(today_entries.keys()),
            }


# Singleton
daily_reid_store = DailyReIDStore()

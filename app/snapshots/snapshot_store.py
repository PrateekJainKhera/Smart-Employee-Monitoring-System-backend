"""
SnapshotStore — in-memory only face crop store.

Every time InsightFace detects a face (matched or not), the face crop is
encoded to JPEG bytes and held in a ring buffer (last MAX_MEMORY entries).
Nothing is written to disk.  Data is lost on restart — that is intentional
for now; a future phase can persist to DB or file storage.

Deduplication: at most one snapshot per (track_id, camera_id) per
MIN_INTERVAL seconds to avoid flooding memory with identical frames.
"""
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from app.utils.logger import logger

MAX_MEMORY   = 1000    # ring buffer size (entries)
MIN_INTERVAL = 2.0    # seconds between snapshots for the same track
JPEG_QUALITY = 85     # encode quality (0-100)


class SnapshotStore:
    def __init__(self):
        self._ring:  deque[dict]       = deque(maxlen=MAX_MEMORY)
        self._dedup: dict[str, float]  = {}   # track_key → monotonic time
        self._lock   = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def should_save(self, track_key: str) -> bool:
        """Returns True and records the time if MIN_INTERVAL has elapsed."""
        now = time.monotonic()
        with self._lock:
            if now - self._dedup.get(track_key, 0) < MIN_INTERVAL:
                return False
            self._dedup[track_key] = now
            return True

    def save(
        self,
        face_crop: np.ndarray,
        camera_id: int,
        camera_label: str,
        employee_id: int | None,
        employee_name: str | None,
        confidence: float | None,
        method: str,
    ) -> dict | None:
        """Encode face_crop to JPEG bytes and push into the ring buffer."""
        try:
            now      = datetime.utcnow()
            time_str = now.strftime("%H%M%S_%f")[:13]
            emp_part = f"emp{employee_id}" if employee_id is not None else "unknown"
            filename = f"cam{camera_id}_{now.strftime('%Y%m%d')}_{time_str}_{emp_part}.jpg"

            # Encode to JPEG bytes in memory
            ok, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                return None
            image_bytes = buf.tobytes()

            entry: dict = {
                "id":            filename,
                "filename":      filename,
                "timestamp":     now.isoformat() + "Z",
                "camera_id":     camera_id,
                "camera_label":  camera_label,
                "employee_id":   employee_id,
                "employee_name": employee_name,
                "confidence":    round(confidence, 4) if confidence is not None else None,
                "method":        method,
                "matched":       employee_id is not None,
                "_bytes":        image_bytes,   # not sent in list API, only for image endpoint
            }

            with self._lock:
                self._ring.appendleft(entry)

            return entry
        except Exception as e:
            logger.warning(f"SnapshotStore.save error: {e}")
            return None

    def list(
        self,
        camera_id:   int  | None = None,
        employee_id: int  | None = None,
        matched:     bool | None = None,
        limit:       int         = 100,
    ) -> list[dict]:
        """Return metadata entries (no _bytes field)."""
        with self._lock:
            items = list(self._ring)

        if camera_id is not None:
            items = [s for s in items if s["camera_id"] == camera_id]
        if employee_id is not None:
            items = [s for s in items if s["employee_id"] == employee_id]
        if matched is not None:
            items = [s for s in items if s["matched"] == matched]

        # Strip internal bytes before returning
        return [{k: v for k, v in s.items() if k != "_bytes"} for s in items[:limit]]

    def get_image_bytes(self, filename: str) -> bytes | None:
        """Return JPEG bytes for a snapshot by filename, or None if not found."""
        with self._lock:
            for entry in self._ring:
                if entry["filename"] == filename:
                    return entry.get("_bytes")
        return None


# Singleton
snapshot_store = SnapshotStore()

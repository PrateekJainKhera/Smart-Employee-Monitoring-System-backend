"""
ClothingTrackStore — in-memory store for anonymous track clothing signatures.

When recognition_mode = 'face_clothing':
  - Every detected person track gets a clothing histogram stored here
  - When a face is later recognized, we look back in this store to find
    earlier anonymous sightings of the same person (by clothing similarity)
  - This allows retroactive first-sighting assignment for accurate check-in time

Store key: (camera_id, track_id)
Store value: {
    histogram: np.ndarray,       # 96-dim color histogram
    first_seen: datetime,        # first time this track was seen
    last_seen: datetime,         # last time this track was seen
    camera_id: int,
    track_id: int,
    employee_id: int | None,     # set when face is recognized
}

Thread-safe. Entries older than reid_time_window_min are auto-expired.
"""
import threading
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

from app.clothing.color_histogram import histogram_similarity
from app.config import settings


class ClothingTrackStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._tracks: dict[tuple, dict] = {}

    def upsert(
        self,
        camera_id: int,
        track_id: int,
        histogram: Optional[np.ndarray],
        now: Optional[datetime] = None,
    ) -> None:
        """Add or update a track's clothing signature.

        If histogram is None, only last_seen is updated (cheap touch).
        This avoids re-computing histograms for tracks we've already seen.
        """
        now = now or datetime.utcnow()
        key = (camera_id, track_id)
        with self._lock:
            if key not in self._tracks:
                self._tracks[key] = {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "histogram": histogram,
                    "first_seen": now,
                    "last_seen": now,
                    "employee_id": None,
                }
            else:
                self._tracks[key]["last_seen"] = now
                # Only update histogram if a new one is provided
                if histogram is not None:
                    self._tracks[key]["histogram"] = histogram

    def has_histogram(self, camera_id: int, track_id: int) -> bool:
        """Return True if a histogram has already been captured for this track."""
        key = (camera_id, track_id)
        with self._lock:
            entry = self._tracks.get(key)
            return entry is not None and entry["histogram"] is not None

    def assign_employee(self, camera_id: int, track_id: int, employee_id: int) -> None:
        """Mark a track as identified."""
        key = (camera_id, track_id)
        with self._lock:
            if key in self._tracks:
                self._tracks[key]["employee_id"] = employee_id

    def find_matching_tracks(
        self,
        histogram: np.ndarray,
        camera_id: int,
        exclude_track_id: int,
        now: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Find anonymous tracks (no employee_id) with similar clothing histogram.

        Returns list of matching track dicts sorted by first_seen ascending.
        Only searches within reid_time_window_min and optionally same camera only.
        """
        now = now or datetime.utcnow()
        cutoff = now - timedelta(minutes=settings.reid_time_window_min)
        results = []

        with self._lock:
            for key, track in self._tracks.items():
                # Skip already-identified tracks
                if track["employee_id"] is not None:
                    continue
                # Skip current track
                if track["track_id"] == exclude_track_id and track["camera_id"] == camera_id:
                    continue
                # Camera filter
                if settings.reid_same_camera_only and track["camera_id"] != camera_id:
                    continue
                # Time window filter
                if track["first_seen"] < cutoff:
                    continue
                # Similarity check
                sim = histogram_similarity(histogram, track["histogram"])
                if sim >= settings.reid_similarity_threshold:
                    results.append({**track, "similarity": sim})

        # Sort by first_seen ascending — earliest sighting first
        results.sort(key=lambda x: x["first_seen"])
        return results

    def expire_old_tracks(self, now: Optional[datetime] = None) -> int:
        """Remove tracks older than reid_time_window_min. Returns count removed."""
        now = now or datetime.utcnow()
        cutoff = now - timedelta(minutes=settings.reid_time_window_min)
        removed = 0
        with self._lock:
            stale = [k for k, v in self._tracks.items() if v["last_seen"] < cutoff]
            for k in stale:
                del self._tracks[k]
                removed += 1
        return removed

    def clear_camera(self, camera_id: int) -> None:
        """Remove all tracks for a specific camera (called on pipeline stop)."""
        with self._lock:
            keys = [k for k in self._tracks if k[0] == camera_id]
            for k in keys:
                del self._tracks[k]

    def count(self) -> int:
        with self._lock:
            return len(self._tracks)


# Singleton
clothing_track_store = ClothingTrackStore()

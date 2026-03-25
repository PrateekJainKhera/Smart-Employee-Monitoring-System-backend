import numpy as np
from dataclasses import dataclass
from deep_sort_realtime.deepsort_tracker import DeepSort
from app.utils.logger import logger
from app.detection.yolo_detector import BoundingBox


@dataclass
class Track:
    track_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    is_confirmed: bool

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


class EmployeeTracker:
    """
    Wraps DeepSORT to assign persistent track IDs to detected persons.
    One instance per camera — track IDs are camera-scoped.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        max_age  : frames to keep a track alive without a detection match
        n_init   : detections needed before a track is confirmed
        """
        # embedder=None → pure IoU matching (no MobileNet appearance model).
        # This avoids the pkg_resources dependency and works fine on CPU.
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.4,
            nn_budget=None,
            embedder=None,
        )
        logger.debug(f"EmployeeTracker created (max_age={max_age}, n_init={n_init})")

    def update(self, detections: list[BoundingBox], frame: np.ndarray) -> list[Track]:
        """
        Feed YOLO detections into DeepSORT and return confirmed + tentative tracks.

        detections : list of BoundingBox(x1,y1,x2,y2,confidence)
        frame      : current BGR frame (used by DeepSORT appearance model)
        returns    : list of Track objects
        """
        # DeepSORT expects: [([left, top, w, h], confidence, class_id), ...]
        raw = [
            ([box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1], box.confidence, 0)
            for box in detections
        ]

        # embedder=None → must supply embeddings manually.
        # Constant unit vectors make cosine distance = 0 for all pairs,
        # so the tracker falls back to pure IoU association (correct behavior
        # for CPU-only usage without an appearance model).
        embeds = [np.ones(64, dtype=np.float32) for _ in raw]

        ds_tracks = self._tracker.update_tracks(raw, embeds=embeds)

        tracks: list[Track] = []
        for t in ds_tracks:
            if not t.is_confirmed():
                continue  # skip tentative tracks
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = (
                int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            )
            tracks.append(Track(
                track_id=str(t.track_id),
                x1=x1, y1=y1, x2=x2, y2=y2,
                is_confirmed=True,
            ))

        return tracks

    def reset(self) -> None:
        """Clear all tracks (e.g. on camera reconnect)."""
        self._tracker = DeepSort(
            max_age=self._tracker.max_age,
            n_init=self._tracker.n_init,
            max_cosine_distance=0.4,
            embedder=None,
        )
        logger.debug("EmployeeTracker reset")

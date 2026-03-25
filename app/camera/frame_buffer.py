import queue
import threading
import numpy as np
from app.utils.logger import logger


class FrameBuffer:
    """
    Thread-safe per-camera frame buffer.
    Holds only the latest frame per camera — drops old frames when full.
    """

    def __init__(self, maxsize: int = 5):
        self._buffers: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def _get_or_create(self, camera_id: int) -> queue.Queue:
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = queue.Queue(maxsize=self._maxsize)
            return self._buffers[camera_id]

    def put_frame(self, camera_id: int, frame: np.ndarray) -> None:
        q = self._get_or_create(camera_id)
        # Drop oldest frame if full to always keep latest
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_frame(self, camera_id: int) -> np.ndarray | None:
        q = self._get_or_create(camera_id)
        try:
            return q.get_nowait()
        except queue.Empty:
            return None

    def get_latest_frame(self, camera_id: int) -> np.ndarray | None:
        """Drain queue and return the most recent frame."""
        q = self._get_or_create(camera_id)
        frame = None
        while True:
            try:
                frame = q.get_nowait()
            except queue.Empty:
                break
        return frame

    def remove_camera(self, camera_id: int) -> None:
        with self._lock:
            self._buffers.pop(camera_id, None)


# Singleton
frame_buffer = FrameBuffer()

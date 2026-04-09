import cv2
import threading
import time
import numpy as np
from app.utils.logger import logger
from app.utils.helpers import parse_camera_source, resize_frame
from app.camera.frame_buffer import frame_buffer


class CameraThread(threading.Thread):
    """Dedicated capture thread for a single camera / webcam."""

    def __init__(self, camera_id: int, source: int | str, location_label: str):
        super().__init__(daemon=True, name=f"cam-{camera_id}")
        self.camera_id = camera_id
        self.source = source
        self.location_label = location_label
        self._stop_event = threading.Event()
        self.is_connected = False

    def _open_capture(self) -> cv2.VideoCapture:
        """Open VideoCapture with low-latency settings."""
        import os
        source = self.source
        is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp")

        if is_rtsp:
            # Force TCP transport + tiny buffer for minimum latency
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;65536|stimeout;3000000|max_delay;100000"
            )
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            # CAP_DSHOW is required on Windows to avoid obsensor backend errors
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened() and source != 0:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # 720p @ 30fps — enough for buffalo_l, faster than 1080p on CPU
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the latest frame in OpenCV buffer
        return cap

    def run(self) -> None:
        logger.info(f"Camera {self.camera_id} ({self.location_label}) starting — source: {self.source}")
        retry_delay = 2

        while not self._stop_event.is_set():
            cap = self._open_capture()

            if not cap.isOpened():
                logger.warning(f"Camera {self.camera_id} failed to open. Retrying in {retry_delay}s...")
                self.is_connected = False
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
                continue

            self.is_connected = True
            retry_delay = 2
            logger.info(f"Camera {self.camera_id} connected")

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Camera {self.camera_id} lost connection. Reconnecting...")
                    self.is_connected = False
                    break

                # Resize once here — all consumers (pipeline + streams) get 1280px frames
                # buffalo_l benefits from higher resolution for angled/distant faces
                frame = resize_frame(frame, width=1280)
                frame_buffer.put_frame(self.camera_id, frame)

            cap.release()

        logger.info(f"Camera {self.camera_id} thread stopped")

    def stop(self) -> None:
        self._stop_event.set()


class CameraManager:
    """Manages all camera threads."""

    def __init__(self):
        self._threads: dict[int, CameraThread] = {}
        self._lock = threading.Lock()

    def start_camera(self, camera_id: int, rtsp_url: str, location_label: str) -> None:
        with self._lock:
            if camera_id in self._threads:
                logger.info(f"Camera {camera_id} already running")
                return
            source = parse_camera_source(rtsp_url)
            thread = CameraThread(camera_id, source, location_label)
            thread.start()
            self._threads[camera_id] = thread

    def stop_camera(self, camera_id: int) -> None:
        with self._lock:
            thread = self._threads.pop(camera_id, None)
        if thread:
            thread.stop()
            thread.join(timeout=3)
            frame_buffer.remove_camera(camera_id)
            logger.info(f"Camera {camera_id} stopped")

    def start_all(self, cameras: list[dict]) -> None:
        """Start threads for all active cameras."""
        for cam in cameras:
            if cam.get("is_active", True):
                self.start_camera(cam["id"], cam["rtsp_url"], cam["location_label"])

    def stop_all(self) -> None:
        camera_ids = list(self._threads.keys())
        for camera_id in camera_ids:
            self.stop_camera(camera_id)
        logger.info("All camera threads stopped")

    def get_frame(self, camera_id: int) -> np.ndarray | None:
        return frame_buffer.get_latest_frame(camera_id)

    def is_connected(self, camera_id: int) -> bool:
        thread = self._threads.get(camera_id)
        return thread.is_connected if thread else False

    def active_cameras(self) -> list[int]:
        return list(self._threads.keys())


# Singleton
camera_manager = CameraManager()

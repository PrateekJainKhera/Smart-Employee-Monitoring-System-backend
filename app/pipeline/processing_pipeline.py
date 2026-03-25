import asyncio
import threading
import numpy as np
from app.utils.logger import logger
from app.camera.camera_manager import camera_manager
from app.detection.yolo_detector import YOLODetector, BoundingBox
from app.tracking.tracker import EmployeeTracker, Track
from app.config import settings


class ProcessingPipeline:
    """
    Per-camera pipeline: detect → track → (recognize in Phase 4) → (attend in Phase 5).

    One pipeline instance is created per camera.
    Runs in its own daemon thread so it doesn't block the API.
    """

    def __init__(self, camera_id: int, location_label: str):
        self.camera_id = camera_id
        self.location_label = location_label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Detection — shared YOLO model (nano, CPU)
        self._detector = YOLODetector(
            weights_path=settings.yolo_weights_path,
            confidence=0.45,
            device="cpu",
        )

        # Tracking — one tracker per camera
        self._tracker = EmployeeTracker(max_age=30, n_init=3)

        # Latest processed state — readable from the stream endpoint
        self._last_tracks: list[Track] = []
        self._last_boxes: list[BoundingBox] = []
        self._lock = threading.Lock()

        # Frame skipping for performance
        self._frame_count = 0
        self._detect_every = 5   # run YOLO every N frames
        self._track_every = 1    # run DeepSORT every frame (lightweight)

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"pipeline-{self.camera_id}",
        )
        self._thread.start()
        logger.info(f"Pipeline started for camera {self.camera_id} ({self.location_label})")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        logger.info(f"Pipeline stopped for camera {self.camera_id}")

    def get_latest_tracks(self) -> list[Track]:
        with self._lock:
            return list(self._last_tracks)

    def get_latest_boxes(self) -> list[BoundingBox]:
        with self._lock:
            return list(self._last_boxes)

    # ── Internal loop ─────────────────────────────────────────────────

    def _loop(self) -> None:
        logger.debug(f"Pipeline loop running for camera {self.camera_id}")
        while not self._stop_event.is_set():
            frame = camera_manager.get_frame(self.camera_id)
            if frame is None:
                self._stop_event.wait(timeout=0.05)
                continue

            self._frame_count += 1
            self._process_frame(frame)

        logger.debug(f"Pipeline loop exited for camera {self.camera_id}")

    def _process_frame(self, frame: np.ndarray) -> None:
        # ── Step 1: Detection (every N frames, reuse last boxes otherwise) ──
        if self._frame_count % self._detect_every == 0:
            boxes = self._detector.detect(frame)
            with self._lock:
                self._last_boxes = boxes
        else:
            with self._lock:
                boxes = list(self._last_boxes)

        # ── Step 2: Tracking (every frame) ──────────────────────────────────
        tracks = self._tracker.update(boxes, frame)
        with self._lock:
            self._last_tracks = tracks

        # ── Step 3: Identity mapping (Phase 4 — placeholder) ────────────────
        # for track in tracks:
        #     key = f"{track.track_id}@{self.camera_id}"
        #     if app_state.get_track_identity(key) is None:
        #         face_crop = get_face_crop(frame, track.to_tuple())
        #         if face_crop is not None:
        #             result = face_recognizer.identify(face_crop)
        #             if result:
        #                 app_state.set_track_identity(key, result.employee_id)

        # ── Step 4: Attendance events (Phase 5 — placeholder) ───────────────
        # for track in tracks:
        #     key = f"{track.track_id}@{self.camera_id}"
        #     employee_id = app_state.get_track_identity(key)
        #     if employee_id:
        #         attendance_service.handle_event(employee_id, self.camera_id,
        #                                         self.location_label)


class PipelineManager:
    """Manages one ProcessingPipeline per camera."""

    def __init__(self):
        self._pipelines: dict[int, ProcessingPipeline] = {}
        self._lock = threading.Lock()

    def start_pipeline(self, camera_id: int, location_label: str) -> None:
        with self._lock:
            if camera_id in self._pipelines:
                return
            pipeline = ProcessingPipeline(camera_id, location_label)
            pipeline.start()
            self._pipelines[camera_id] = pipeline

    def stop_pipeline(self, camera_id: int) -> None:
        with self._lock:
            pipeline = self._pipelines.pop(camera_id, None)
        if pipeline:
            pipeline.stop()

    def start_all(self, cameras: list[dict]) -> None:
        for cam in cameras:
            if cam.get("is_active", True):
                self.start_pipeline(cam["id"], cam["location_label"])

    def stop_all(self) -> None:
        ids = list(self._pipelines.keys())
        for cid in ids:
            self.stop_pipeline(cid)

    def get_tracks(self, camera_id: int) -> list[Track]:
        pipeline = self._pipelines.get(camera_id)
        return pipeline.get_latest_tracks() if pipeline else []

    def get_boxes(self, camera_id: int) -> list[BoundingBox]:
        pipeline = self._pipelines.get(camera_id)
        return pipeline.get_latest_boxes() if pipeline else []

    def is_running(self, camera_id: int) -> bool:
        pipeline = self._pipelines.get(camera_id)
        return pipeline._thread.is_alive() if pipeline and pipeline._thread else False


# Singleton
pipeline_manager = PipelineManager()

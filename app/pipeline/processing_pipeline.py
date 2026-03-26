import queue
import threading
import numpy as np
from app.utils.logger import logger
from app.camera.camera_manager import camera_manager
from app.detection.yolo_detector import YOLODetector, BoundingBox
from app.tracking.tracker import EmployeeTracker, Track
from app.utils.helpers import get_head_crop
from app.store import state as app_state
from app.config import settings

# ── Recognition worker (shared across all cameras) ────────────────────────────
# Runs InsightFace in its own thread so the pipeline loop is never blocked.
_recognition_queue: queue.Queue = queue.Queue(maxsize=30)


def _recognition_worker() -> None:
    """
    Dedicated thread that processes face recognition jobs from the queue.
    Writes results directly into AppState so the pipeline can pick them up
    on the next frame without ever having waited for InsightFace.
    """
    while True:
        try:
            item = _recognition_queue.get(timeout=1)
        except queue.Empty:
            continue

        camera_id, track_id, crop = item
        key = f"{track_id}@{camera_id}"

        # Skip if already identified while this job was queued
        if app_state.get_track_identity(key) is not None:
            _recognition_queue.task_done()
            continue

        try:
            from app.recognition.face_recognizer import face_recognizer
            if face_recognizer is not None:
                result = face_recognizer.identify(crop)
                if result is not None:
                    app_state.set_track_identity(key, result.employee_id)
                    logger.info(
                        f"Recognized cam={camera_id} track={track_id} "
                        f"→ employee_id={result.employee_id} "
                        f"conf={result.confidence:.2f} method={result.method}"
                    )
        except Exception as e:
            logger.warning(f"Recognition worker error: {e}")
        finally:
            _recognition_queue.task_done()


# Start the single recognition worker thread at module load
_worker_thread = threading.Thread(
    target=_recognition_worker, daemon=True, name="recognition-worker"
)
_worker_thread.start()


class ProcessingPipeline:
    """
    Per-camera pipeline: detect → track → recognize (async) → attend (Phase 5).

    Recognition is non-blocking — face crops are queued to a dedicated worker
    thread, so this loop always runs at full camera speed.
    """

    # How often to retry recognition for an unidentified track (in frames)
    _RECOG_RETRY_EVERY = 8
    # Max attempts before giving up on a track (stops wasting CPU)
    _RECOG_MAX_ATTEMPTS = 20

    def __init__(self, camera_id: int, location_label: str):
        self.camera_id = camera_id
        self.location_label = location_label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Detection — YOLOv8n (CPU)
        self._detector = YOLODetector(
            weights_path=settings.yolo_weights_path,
            confidence=0.45,
            device="cpu",
        )

        # Tracking — n_init=2: confirmed after 2 frames (faster than 3)
        self._tracker = EmployeeTracker(max_age=30, n_init=2)

        # Latest processed state — read by stream endpoints
        self._last_tracks: list[Track] = []
        self._last_boxes: list[BoundingBox] = []
        self._lock = threading.Lock()

        # Frame counter and detection cadence
        self._frame_count = 0
        self._detect_every = 4       # YOLO every 4 frames (was 5)

        # Per-track recognition bookkeeping: track_id → attempt count
        self._recog_attempts: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"pipeline-{self.camera_id}"
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
        # ── Step 1: Detection (every N frames) ──────────────────────────
        if self._frame_count % self._detect_every == 0:
            boxes = self._detector.detect(frame)
            with self._lock:
                self._last_boxes = boxes
        else:
            with self._lock:
                boxes = list(self._last_boxes)

        # ── Step 2: Tracking ─────────────────────────────────────────────
        tracks = self._tracker.update(boxes, frame)
        with self._lock:
            self._last_tracks = tracks

        # Clean up attempt counters for tracks that are no longer active
        active_keys = {f"{t.track_id}@{self.camera_id}" for t in tracks}
        stale = [k for k in self._recog_attempts if k not in active_keys]
        for k in stale:
            del self._recog_attempts[k]
            app_state.clear_track(k)   # identity resets when person leaves

        # ── Step 3: Non-blocking recognition queue ───────────────────────
        for track in tracks:
            key = f"{track.track_id}@{self.camera_id}"

            # Already identified — nothing to do
            if app_state.get_track_identity(key) is not None:
                continue

            attempts = self._recog_attempts.get(key, 0)

            # Gave up after max attempts — wait for person to leave and re-enter
            if attempts >= self._RECOG_MAX_ATTEMPTS:
                continue

            # Only retry every N frames (not every single frame)
            if attempts > 0 and self._frame_count % self._RECOG_RETRY_EVERY != 0:
                continue

            # Crop head region (top 40% of person box + padding)
            crop = get_head_crop(frame, track.to_tuple())
            if crop is None:
                continue

            # Push to worker — non-blocking (drop if queue full)
            try:
                _recognition_queue.put_nowait((self.camera_id, track.track_id, crop))
                self._recog_attempts[key] = attempts + 1
            except queue.Full:
                pass   # worker is busy — will retry next interval

        # ── Step 4: Attendance events (Phase 5 — placeholder) ────────────
        # for track in tracks:
        #     key = f"{track.track_id}@{self.camera_id}"
        #     employee_id = app_state.get_track_identity(key)
        #     if employee_id:
        #         attendance_service.handle_event(
        #             employee_id, self.camera_id, self.location_label
        #         )


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
        for cid in list(self._pipelines.keys()):
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

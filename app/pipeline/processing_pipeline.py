import queue
import threading
import time
import numpy as np
from app.utils.logger import logger
from app.camera.camera_manager import camera_manager
from app.detection.yolo_detector import YOLODetector, BoundingBox
from app.tracking.tracker import EmployeeTracker, Track
from app.store import state as app_state
from app.config import settings

# ── Recognition worker ────────────────────────────────────────────────────────
# InsightFace runs in its own thread — pipeline loop never waits for it.
# Queue item: (camera_id, frame, tracks_needing_recognition)
_recognition_queue: queue.Queue = queue.Queue(maxsize=20)


def _recognition_worker() -> None:
    while True:
        try:
            item = _recognition_queue.get(timeout=1)
        except queue.Empty:
            continue

        camera_id, frame, tracks, attend_last = item
        try:
            from app.recognition.face_recognizer import face_recognizer
            if face_recognizer is not None:
                results = face_recognizer.identify_in_frame(frame, tracks, camera_id)

                recognized_track_ids: set = set()
                for key, result in results.items():
                    recognized_track_ids.add(key)
                    if app_state.get_track_identity(key) is None:
                        app_state.set_track_identity(key, result.employee_id)
                        track_id, _ = key.split("@")
                        logger.info(
                            f"Recognized cam={camera_id} track={track_id} "
                            f"→ employee_id={result.employee_id} "
                            f"conf={result.confidence:.2f} method={result.method}"
                        )
                        # WS: live detected event
                        try:
                            from app.store import state as _state
                            cam = _state.get_camera(camera_id)
                            label = cam["location_label"] if cam else ""
                            emp = _state.get_employee(result.employee_id)
                            name = emp["name"] if emp else str(result.employee_id)
                            from app.api.ws import emit_detected
                            emit_detected(result.employee_id, name, camera_id, label, result.confidence)
                        except Exception:
                            pass

                        # Immediately trigger attendance — no waiting for next pipeline cycle
                        dkey = (result.employee_id, camera_id)
                        now = time.monotonic()
                        if now - attend_last.get(dkey, 0) >= ProcessingPipeline._ATTEND_DEBOUNCE:
                            attend_last[dkey] = now
                            try:
                                from app.store import state as _state2
                                cam2 = _state2.get_camera(camera_id)
                                loc = cam2["location_label"] if cam2 else ""
                                _attendance_queue.put_nowait((result.employee_id, camera_id, loc))
                            except queue.Full:
                                logger.warning(f"Attendance queue full — dropping event emp={result.employee_id}")
                            except Exception as _ae:
                                logger.warning(f"Attendance trigger error: {_ae}")

                # WS: unknown persons (tracks with no result after max attempts)
                # Emit once when a track exhausts all attempts without recognition
                for track in tracks:
                    key = f"{track.track_id}@{camera_id}"
                    if key not in recognized_track_ids and app_state.get_track_identity(key) is None:
                        pass  # unknown emit handled below in _process_frame per-track
        except Exception as e:
            logger.warning(f"Recognition worker error: {e}")
        finally:
            _recognition_queue.task_done()


# ── Attendance worker ─────────────────────────────────────────────────────────
_attendance_queue: queue.Queue = queue.Queue(maxsize=100)


def _attendance_worker() -> None:
    while True:
        try:
            item = _attendance_queue.get(timeout=1)
        except queue.Empty:
            continue

        employee_id, camera_id, location_label = item
        try:
            from app.services.attendance_service import handle_event
            handle_event(employee_id, camera_id, location_label)
        except Exception as e:
            logger.warning(f"Attendance worker error: {e}")
        finally:
            _attendance_queue.task_done()


threading.Thread(target=_recognition_worker, daemon=True, name="recognition-worker").start()
threading.Thread(target=_attendance_worker,  daemon=True, name="attendance-worker").start()


class ProcessingPipeline:
    """
    Per-camera pipeline: detect → track → recognize → attend.

    Throughput design:
      • Pipeline ticks at _PIPELINE_INTERVAL (0.33s = ~3 fps) — no busy-wait
      • YOLO runs every _DETECT_EVERY cycles (~1/sec) — biggest CPU saver
      • Between YOLO cycles: reuse last known boxes for tracker continuity
      • Max _MAX_TRACKS persons processed — prevents runaway in crowded scenes
      • Recognition: fresh tracks every 2 cycles, stale tracks every 6 cycles
      • Attendance: debounced 30s per employee

    Thread layout (all non-blocking):
      cam-{id}           capture only — cap.read() + resize → frame_buffer
      pipeline-{id}      this loop: YOLO + DeepSORT at fixed rate
      recognition-worker shared: InsightFace full-frame (one thread, queued)
      attendance-worker  shared: DB writes (one thread, queued)
    """

    # ── Timing ──────────────────────────────────────────────────────────────
    _PIPELINE_INTERVAL = 0.20   # ~5 pipeline cycles/sec
    _DETECT_EVERY      = 1      # YOLO every cycle → ~5 detections/sec (fast scan)
    # Detect every cycle so a person walking past is reliably caught

    # ── Recognition ─────────────────────────────────────────────────────────
    _RECOG_FRESH_EVERY = 2      # try recognition every 2 cycles for fresh tracks
    _RECOG_STALE_EVERY = 10     # cycles between recognition for stale tracks (3–9 attempts)
    _RECOG_MAX_ATTEMPTS = 10    # give up after N full-frame attempts

    # ── Safety ──────────────────────────────────────────────────────────────
    _MAX_TRACKS      = 6        # only run recognition on the N closest persons per frame
    _ATTEND_DEBOUNCE = 30       # seconds between attendance queue pushes per employee

    # ── FPS monitor ─────────────────────────────────────────────────────────
    _FPS_LOG_EVERY   = 30       # log actual fps every N cycles (~10s)

    def __init__(self, camera_id: int, location_label: str):
        self.camera_id      = camera_id
        self.location_label = location_label
        self._stop_event    = threading.Event()
        self._thread: threading.Thread | None = None

        self._detector = YOLODetector(
            weights_path=settings.yolo_weights_path,
            confidence=0.45,
            device="cpu",
        )
        self._tracker = EmployeeTracker(max_age=30, n_init=1)

        self._last_tracks: list[Track]      = []
        self._last_boxes:  list[BoundingBox] = []
        self._lock = threading.Lock()

        self._frame_count = 0
        self._recog_attempts: dict[str, int]    = {}
        self._attend_last:    dict[tuple, float] = {}

        # FPS monitoring state
        self._fps_t0    = 0.0
        self._fps_count = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"pipeline-{self.camera_id}"
        )
        self._thread.start()
        logger.info(f"Pipeline started cam={self.camera_id} ({self.location_label})")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        logger.info(f"Pipeline stopped cam={self.camera_id}")

    def get_latest_tracks(self) -> list[Track]:
        with self._lock:
            return list(self._last_tracks)

    def get_latest_boxes(self) -> list[BoundingBox]:
        with self._lock:
            return list(self._last_boxes)

    # ── Internal loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        logger.debug(f"Pipeline loop cam={self.camera_id} (~{1/self._PIPELINE_INTERVAL:.0f} fps target)")
        self._fps_t0 = time.monotonic()
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            # ── Rate limiter: sleep until next tick ──────────────────────
            now  = time.monotonic()
            wait = next_tick - now
            if wait > 0:
                self._stop_event.wait(timeout=wait)
                if self._stop_event.is_set():
                    break
            next_tick = time.monotonic() + self._PIPELINE_INTERVAL

            # ── Grab latest frame from buffer (non-blocking) ──────────────
            frame = camera_manager.get_frame(self.camera_id)
            if frame is None:
                continue   # camera not ready yet

            self._frame_count += 1
            self._fps_count   += 1
            self._process_frame(frame)

            # ── FPS monitor ───────────────────────────────────────────────
            if self._fps_count >= self._FPS_LOG_EVERY:
                elapsed = time.monotonic() - self._fps_t0
                logger.info(
                    f"Pipeline cam={self.camera_id}: "
                    f"{self._fps_count / elapsed:.1f} fps actual"
                )
                self._fps_count = 0
                self._fps_t0    = time.monotonic()

        logger.debug(f"Pipeline loop exited cam={self.camera_id}")

    def _process_frame(self, frame: np.ndarray) -> None:

        # ── Step 1: YOLO detection (every N cycles, reuse boxes otherwise) ─
        # Frame is already 640px wide — resized once in camera_manager.
        if self._frame_count % self._DETECT_EVERY == 0:
            boxes = self._detector.detect(frame)   # ~50–100ms on CPU
            with self._lock:
                self._last_boxes = boxes
        else:
            with self._lock:
                boxes = list(self._last_boxes)

        # ── Step 2: DeepSORT tracking ─────────────────────────────────────
        tracks = self._tracker.update(boxes, frame)
        with self._lock:
            self._last_tracks = tracks

        # Clean up state for tracks that left the frame
        active_keys = {f"{t.track_id}@{self.camera_id}" for t in tracks}
        for k in [k for k in self._recog_attempts if k not in active_keys]:
            del self._recog_attempts[k]
            app_state.clear_track(k)

        # ── Step 3: Recognition — limit to N closest persons ─────────────
        # Sort by bbox area descending (larger area = person is closer to camera)
        sorted_tracks = sorted(
            tracks,
            key=lambda t: (t.x2 - t.x1) * (t.y2 - t.y1),
            reverse=True,
        )[:self._MAX_TRACKS]

        # Separate into fresh (few attempts) and stale (many attempts)
        # Fresh tracks: retry every _RECOG_FRESH_EVERY cycles
        # Stale tracks: retry every _RECOG_STALE_EVERY cycles (already tried many times)
        fresh, stale = [], []
        for t in sorted_tracks:
            key      = f"{t.track_id}@{self.camera_id}"
            attempts = self._recog_attempts.get(key, 0)
            if app_state.get_track_identity(key) is not None:
                continue   # already recognized — skip
            if attempts >= self._RECOG_MAX_ATTEMPTS:
                continue   # exhausted — give up
            if attempts < 3:
                fresh.append(t)
            else:
                stale.append(t)

        to_recognize: list[Track] = []
        if fresh and self._frame_count % self._RECOG_FRESH_EVERY == 0:
            to_recognize.extend(fresh)
        if stale and self._frame_count % self._RECOG_STALE_EVERY == 0:
            to_recognize.extend(stale)

        if to_recognize:
            try:
                _recognition_queue.put_nowait((self.camera_id, frame.copy(), to_recognize, self._attend_last))
                for t in to_recognize:
                    key = f"{t.track_id}@{self.camera_id}"
                    self._recog_attempts[key] = self._recog_attempts.get(key, 0) + 1
            except queue.Full:
                logger.warning(f"Recognition queue full cam={self.camera_id} — skipping cycle")


        # ── Step 4: Attendance (debounced, non-blocking DB write) ─────────
        now = time.monotonic()
        for track in tracks:
            key = f"{track.track_id}@{self.camera_id}"
            emp = app_state.get_track_identity(key)
            if emp is None:
                continue
            dkey = (emp, self.camera_id)
            if now - self._attend_last.get(dkey, 0) < self._ATTEND_DEBOUNCE:
                continue
            self._attend_last[dkey] = now
            try:
                _attendance_queue.put_nowait((emp, self.camera_id, self.location_label))
            except queue.Full:
                pass


class PipelineManager:
    def __init__(self):
        self._pipelines: dict[int, ProcessingPipeline] = {}
        self._lock = threading.Lock()

    def start_pipeline(self, camera_id: int, location_label: str) -> None:
        with self._lock:
            if camera_id in self._pipelines:
                return
            p = ProcessingPipeline(camera_id, location_label)
            p.start()
            self._pipelines[camera_id] = p

    def stop_pipeline(self, camera_id: int) -> None:
        with self._lock:
            p = self._pipelines.pop(camera_id, None)
        if p:
            p.stop()

    def start_all(self, cameras: list[dict]) -> None:
        for cam in cameras:
            if cam.get("is_active", True):
                self.start_pipeline(cam["id"], cam["location_label"])

    def stop_all(self) -> None:
        for cid in list(self._pipelines.keys()):
            self.stop_pipeline(cid)

    def get_tracks(self, camera_id: int) -> list[Track]:
        p = self._pipelines.get(camera_id)
        return p.get_latest_tracks() if p else []

    def get_boxes(self, camera_id: int) -> list[BoundingBox]:
        p = self._pipelines.get(camera_id)
        return p.get_latest_boxes() if p else []

    def is_running(self, camera_id: int) -> bool:
        p = self._pipelines.get(camera_id)
        return p._thread.is_alive() if p and p._thread else False


pipeline_manager = PipelineManager()

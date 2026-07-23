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

# ── Per-camera recognition queues and workers ─────────────────────────────────
# Each camera gets its own queue + thread so cameras never block each other.
# Queue size=2: drop stale frames immediately, always process the freshest.
_recognition_queues: dict[int, queue.Queue] = {}
_recognition_queue_lock = threading.Lock()

# ── OSNet runs in its own dedicated thread (separate from face recognition) ───
# OSNet ~80ms/crop on CPU — keeping it out of the face recognition thread
# ensures face recognition stays fast (v1/v2 speed) even with ReID enabled.
_osnet_queue: queue.Queue = queue.Queue(maxsize=4)   # (camera_id, track_id, body_crop, attend_last)
_osnet_last_run: dict[str, float]  = {}   # track_key → last run time
_osnet_fail_count: dict[str, int]  = {}   # track_key → consecutive None-crop failures
_osnet_attempts: dict[str, int]    = {}   # track_key → total OSNet dispatch count
_OSNET_INTERVAL       = 4.0              # base seconds between OSNet calls per track
_OSNET_MAX_INTERVAL   = 20.0            # max backoff interval for bad-crop tracks
_OSNET_MAX_ATTEMPTS   = 5               # stop ReID after N dispatches with no match
_OSNET_PER_CYCLE_CAP  = 2              # max tracks dispatched to OSNet per recognition cycle
_MIN_BODY_AREA        = 4000            # min body crop area in pixels (≈50×80) — skip tiny/far crops
_osnet_worker_started = False
_osnet_worker_lock = threading.Lock()

# ── Recognition stability — require N consistent recognitions before attendance ──
# Threshold depends on confidence level:
#   frame_high (≥0.55)      → 1 (immediate — high confidence, no lookalike risk)
#   frame_verified (0.42–0.54) → 2 (borderline — require one confirmation)
#   everything else          → 1
# {track_key: (employee_id, consecutive_count)}
_recog_stable: dict[str, tuple[int, int]] = {}

_STABLE_BY_METHOD: dict[str, int] = {
    "frame_high":            1,   # very confident → immediate
    "insightface_high":      1,
    "crop_high":             1,
    "frame_verified":        2,   # borderline → need 1 confirmation
    "frame_medium+deepface": 1,   # already passed FaceNet → immediate
    "crop_medium+deepface":  1,
}


def _get_or_create_recognition_queue(camera_id: int) -> queue.Queue:
    with _recognition_queue_lock:
        if camera_id not in _recognition_queues:
            q: queue.Queue = queue.Queue(maxsize=2)  # was 4 — smaller queue = always process freshest frame
            _recognition_queues[camera_id] = q
            threading.Thread(
                target=_recognition_worker,
                args=(q,),
                daemon=True,
                name=f"recognition-worker-cam{camera_id}",
            ).start()
            logger.info(f"Recognition worker started for cam={camera_id}")
        return _recognition_queues[camera_id]


def _recognition_worker(q: queue.Queue) -> None:
    while True:
        try:
            item = q.get(timeout=1)
        except queue.Empty:
            continue

        camera_id, frame, tracks, attend_last = item
        try:
            from app.recognition.face_recognizer import face_recognizer
            from app.config import settings as _settings

            h, w = frame.shape[:2]   # computed once — reused throughout this worker call

            # ── Clothing signature collection (face_clothing mode only) ──────
            # Histograms are extracted ONCE per track (first sighting only).
            # Subsequent cycles just touch last_seen — no expensive cv2.calcHist.
            if _settings.recognition_mode != "face_only":
                try:
                    from app.clothing.color_histogram import extract_clothing_histogram
                    from app.clothing.track_store import clothing_track_store
                    from datetime import datetime as _dt
                    now = _dt.utcnow()
                    for track in tracks:
                        if clothing_track_store.has_histogram(camera_id, track.track_id):
                            # Already have clothing signature — just keep track alive
                            clothing_track_store.upsert(camera_id, track.track_id, None, now)
                            continue
                        # First sighting — extract histogram once
                        x1 = max(0, int(track.bbox[0]))
                        y1 = max(0, int(track.bbox[1]))
                        x2 = min(w, int(track.bbox[2]))
                        y2 = min(h, int(track.bbox[3]))
                        person_crop = frame[y1:y2, x1:x2]
                        hist = extract_clothing_histogram(person_crop)
                        if hist is not None:
                            clothing_track_store.upsert(camera_id, track.track_id, hist, now)
                except Exception as _ce:
                    logger.debug(f"Clothing collection error: {_ce}")

            if face_recognizer is not None:
                results = face_recognizer.identify_in_frame(frame, tracks, camera_id)

                recognized_track_ids: set = set()
                for key, result in results.items():
                    recognized_track_ids.add(key)

                    existing = app_state.get_track_identity(key)
                    # Allow correction if a high-confidence result contradicts cached identity
                    is_new        = existing is None
                    is_correction = (
                        existing is not None
                        and existing != result.employee_id
                        and result.method in ("frame_high", "insightface_high", "crop_high")
                    )
                    # ── Stability counter — prevent lookalike false positives ──
                    # Required stable count depends on recognition method:
                    #   high-confidence methods → 1 (immediate)
                    #   borderline (frame_verified) → 2 (one confirmation)
                    stable_min = _STABLE_BY_METHOD.get(result.method, 1)
                    prev_emp, prev_count = _recog_stable.get(key, (None, 0))
                    if prev_emp == result.employee_id:
                        stable_count = prev_count + 1
                    else:
                        stable_count = 1
                    _recog_stable[key] = (result.employee_id, stable_count)
                    stable_enough = stable_count >= stable_min

                    if is_new or is_correction:
                        app_state.set_track_identity(key, result.employee_id)
                        track_id, _ = key.split("@")
                        tag = "CORRECTED" if is_correction else "Recognized"
                        logger.info(
                            f"{tag} cam={camera_id} track={track_id} "
                            f"→ employee_id={result.employee_id} "
                            f"conf={result.confidence:.2f} method={result.method} "
                            f"stable={stable_count}/{stable_min}"
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

                        # Record sighting count for this employee + camera today
                        try:
                            from app.sightings.sighting_store import sighting_store
                            sighting_store.record(result.employee_id, camera_id)
                        except Exception:
                            pass

                        # ── OSNet embedding capture (face_reid mode) ────────
                        # Store daily body embedding on face recognition success.
                        # Updates if a higher-quality body crop appears later.
                        if _settings.recognition_mode == "face_reid":
                            try:
                                from app.reid.osnet_engine import osnet_engine
                                from app.reid.daily_reid_store import daily_reid_store
                                if osnet_engine is not None:
                                    for t in tracks:
                                        if str(t.track_id) == track_id:
                                            bx1 = max(0, int(t.x1))
                                            by1 = max(0, int(t.y1))
                                            bx2 = min(w, int(t.x2))
                                            by2 = min(h, int(t.y2))
                                            body_crop = frame[by1:by2, bx1:bx2]
                                            if body_crop.size > 0:
                                                emb = osnet_engine.get_embedding(body_crop)
                                                quality = float((bx2 - bx1) * (by2 - by1))
                                                stored = daily_reid_store.upsert(result.employee_id, emb, quality)
                                                if stored:
                                                    logger.info(
                                                        f"OSNet embed stored: emp={result.employee_id} "
                                                        f"quality={quality:.0f}"
                                                    )
                                            break
                            except Exception as _oe:
                                logger.warning(f"OSNet embed error: {_oe}")

                        # ── Retroactive track linking (face_clothing mode) ───
                        # Find earlier anonymous tracks with matching clothing
                        # and record sightings back to first_seen time
                        if _settings.recognition_mode != "face_only":
                            try:
                                from app.clothing.track_store import clothing_track_store
                                curr_key = (camera_id, int(track_id))
                                curr_track = clothing_track_store._tracks.get(curr_key)
                                if curr_track and curr_track["histogram"] is not None:
                                    matches = clothing_track_store.find_matching_tracks(
                                        curr_track["histogram"],
                                        camera_id,
                                        int(track_id),
                                    )
                                    for match in matches:
                                        clothing_track_store.assign_employee(
                                            match["camera_id"],
                                            match["track_id"],
                                            result.employee_id,
                                        )
                                        logger.info(
                                            f"ClothingReID: linked track={match['track_id']} "
                                            f"→ emp={result.employee_id} "
                                            f"first_seen={match['first_seen'].strftime('%H:%M:%S')} "
                                            f"sim={match['similarity']:.2f}"
                                        )
                                    clothing_track_store.assign_employee(
                                        camera_id, int(track_id), result.employee_id
                                    )
                            except Exception as _re:
                                logger.debug(f"ReID linking error: {_re}")

                        # Trigger attendance only after stable recognition
                        # Lookalikes typically pass 1-2 frames but not 3 consecutive
                        if stable_enough:
                            dkey = (result.employee_id, camera_id)
                            now = time.monotonic()
                            if now - attend_last.get(dkey, 0) >= ProcessingPipeline._ATTEND_DEBOUNCE:
                                attend_last[dkey] = now
                                try:
                                    from app.store import state as _state2
                                    cam2 = _state2.get_camera(camera_id)
                                    loc = cam2["location_label"] if cam2 else ""
                                    identified_by = "clothing_assist" if result.method == "clothing_reid" else "face"
                                    _attendance_queue.put_nowait((result.employee_id, camera_id, loc, identified_by))
                                except queue.Full:
                                    logger.warning(f"Attendance queue full — dropping event emp={result.employee_id}")
                                except Exception as _ae:
                                    logger.warning(f"Attendance trigger error: {_ae}")
                        else:
                            logger.info(
                                f"  attendance held: emp={result.employee_id} stable={stable_count}/{stable_min} method={result.method}"
                            )

                # ── OSNet fallback — hand off to dedicated OSNet thread ────────
                # OSNet runs in its own thread so it never blocks face recognition.
                if _settings.recognition_mode == "face_reid":
                    _ensure_osnet_worker()
                    try:
                        from app.reid.daily_reid_store import daily_reid_store
                        if daily_reid_store.get_all():
                            dispatched_this_cycle = 0          # #1: per-cycle cap counter
                            for track in tracks:
                                # #1: hard cap — max 2 OSNet dispatches per cycle
                                if dispatched_this_cycle >= _OSNET_PER_CYCLE_CAP:
                                    break
                                key = f"{track.track_id}@{camera_id}"
                                if key in recognized_track_ids:
                                    continue
                                if app_state.get_track_identity(key) is not None:
                                    continue
                                # #4: stop after MAX_ATTEMPTS — track is unregistered/always hidden
                                if _osnet_attempts.get(key, 0) >= _OSNET_MAX_ATTEMPTS:
                                    continue
                                # Stagger: offset each track by (track_id * 0.8s)
                                stagger = (track.track_id * 0.8) % _OSNET_INTERVAL
                                # Backoff: tracks with repeated bad crops wait longer
                                fails = _osnet_fail_count.get(key, 0)
                                interval = min(
                                    _OSNET_INTERVAL + stagger + fails * 2.0,
                                    _OSNET_MAX_INTERVAL
                                )
                                now_osnet = time.monotonic()
                                if now_osnet - _osnet_last_run.get(key, -stagger) < interval:
                                    continue
                                _osnet_last_run[key] = now_osnet
                                bx1 = max(0, int(track.x1))
                                by1 = max(0, int(track.y1))
                                bx2 = min(w, int(track.x2))
                                by2 = min(h, int(track.y2))
                                # #2: early reject — skip tiny/far crops before queue
                                body_area = (bx2 - bx1) * (by2 - by1)
                                if body_area < _MIN_BODY_AREA or body_area == 0:
                                    _osnet_fail_count[key] = fails + 1
                                    continue
                                body_crop = frame[by1:by2, bx1:bx2]
                                if body_crop.size == 0:
                                    _osnet_fail_count[key] = fails + 1
                                    continue
                                try:
                                    _osnet_queue.put_nowait(
                                        (camera_id, track.track_id, body_crop.copy(), attend_last)
                                    )
                                    _osnet_attempts[key] = _osnet_attempts.get(key, 0) + 1
                                    dispatched_this_cycle += 1
                                except queue.Full:
                                    pass  # OSNet busy — skip, will retry next interval
                    except Exception as _roe:
                        logger.debug(f"OSNet queue error: {_roe}")

                # WS: unknown persons (tracks with no result after max attempts)
                # Emit once when a track exhausts all attempts without recognition
                for track in tracks:
                    key = f"{track.track_id}@{camera_id}"
                    if key not in recognized_track_ids and app_state.get_track_identity(key) is None:
                        pass  # unknown emit handled below in _process_frame per-track
        except Exception as e:
            logger.warning(f"Recognition worker error: {e}")
        finally:
            q.task_done()


# ── OSNet worker — dedicated thread, never blocks face recognition ────────────
def _osnet_worker() -> None:
    """Runs OSNet ReID in its own thread. Results written to app_state directly."""
    while True:
        try:
            item = _osnet_queue.get(timeout=1)
        except queue.Empty:
            continue
        camera_id, track_id, body_crop, attend_last = item
        try:
            from app.reid.osnet_engine import osnet_engine
            from app.reid.daily_reid_store import daily_reid_store
            from app.config import settings as _settings
            if osnet_engine is None:
                continue
            gallery = daily_reid_store.get_all()
            if not gallery:
                continue
            query_emb = osnet_engine.get_embedding(body_crop)
            if query_emb is None:
                # Crop too small/blurry — increment backoff for this track
                key = f"{track_id}@{camera_id}"
                _osnet_fail_count[key] = _osnet_fail_count.get(key, 0) + 1
                continue
            # Good crop — reset backoff
            key = f"{track_id}@{camera_id}"
            _osnet_fail_count.pop(key, None)
            best_id, score = osnet_engine.match(query_emb, gallery)
            reid_threshold = getattr(_settings, "reid_similarity_threshold", 0.65)
            logger.info(
                f"  OSNet ReID cam={camera_id} track={track_id}: "
                f"best_id={best_id} score={score:.4f} (need>={reid_threshold})"
            )
            if best_id != -1 and score >= reid_threshold:
                app_state.set_track_identity(key, best_id)
                logger.info(
                    f"OSNet MATCH cam={camera_id} track={track_id} "
                    f"→ emp={best_id} score={score:.4f}"
                )
                dkey = (best_id, camera_id)
                now_t = time.monotonic()
                if now_t - attend_last.get(dkey, 0) >= ProcessingPipeline._ATTEND_DEBOUNCE:
                    attend_last[dkey] = now_t
                    try:
                        cam3 = app_state.get_camera(camera_id)
                        loc3 = cam3["location_label"] if cam3 else ""
                        _attendance_queue.put_nowait((best_id, camera_id, loc3, "reid"))
                    except queue.Full:
                        pass
        except Exception as _oe:
            logger.debug(f"OSNet worker error: {_oe}")
        finally:
            _osnet_queue.task_done()


def _ensure_osnet_worker() -> None:
    global _osnet_worker_started
    with _osnet_worker_lock:
        if not _osnet_worker_started:
            threading.Thread(
                target=_osnet_worker,
                daemon=True,
                name="osnet-reid-worker",
            ).start()
            _osnet_worker_started = True
            logger.info("OSNet ReID worker thread started")


# ── Attendance worker ─────────────────────────────────────────────────────────
_attendance_queue: queue.Queue = queue.Queue(maxsize=100)


def _attendance_worker() -> None:
    while True:
        try:
            item = _attendance_queue.get(timeout=1)
        except queue.Empty:
            continue

        employee_id, camera_id, location_label, identified_by = item
        try:
            from app.services.attendance_service import handle_event
            handle_event(employee_id, camera_id, location_label, identified_by)
        except Exception as e:
            logger.warning(f"Attendance worker error: {e}")
        finally:
            _attendance_queue.task_done()


threading.Thread(target=_attendance_worker, daemon=True, name="attendance-worker").start()


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
    _DETECT_EVERY      = 3      # YOLO every 3 cycles → ~1.7 detections/sec (was 2 → 2.5/sec)
                                # 3 cameras × 1.7/sec = 5 YOLO calls/sec total (was 7.5)
                                # saves ~150–200ms/sec of CPU — biggest throughput win

    # ── Recognition ─────────────────────────────────────────────────────────
    _RECOG_FRESH_EVERY = 3      # try recognition every 3 cycles for fresh tracks (was 2)
                                # adds ~200ms to first attempt but reduces InsightFace load by 33%
    _RECOG_STALE_EVERY = 10     # cycles between recognition for stale tracks (3–9 attempts)
    _RECOG_MAX_ATTEMPTS = 10    # give up after N full-frame attempts

    # ── Safety ──────────────────────────────────────────────────────────────
    _MAX_TRACKS      = 6        # run recognition on up to 6 persons per frame
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
            _recog_stable.pop(k, None)  # reset stability when track disappears

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
                q = _get_or_create_recognition_queue(self.camera_id)
                q.put_nowait((self.camera_id, frame, to_recognize, self._attend_last))
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
                _attendance_queue.put_nowait((emp, self.camera_id, self.location_label, "face"))
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

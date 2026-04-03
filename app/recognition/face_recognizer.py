"""
FaceRecognizer — orchestrator combining InsightFace (primary) + FaceNet (fallback).

Flow (full-frame, identify_in_frame):
  1. InsightFace.get_faces(full_frame) → all face bboxes + embeddings in one pass
  2. Each face spatially matched to nearest DeepSORT track
  3. score >= FRAME_HIGH_THRESHOLD (0.55)  → confident match
  4. FRAME_MED_THRESHOLD (0.42) <= score   → DeepFace fallback
  5. score < FRAME_MED_THRESHOLD           → unknown

Flow (crop, identify — used for registration verification):
  1. InsightFace.get_embedding(crop) → single embedding
  2. Cosine match → HIGH (0.65) / MEDIUM+DeepFace (0.50) / unknown
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from app.recognition.insightface_engine import InsightFaceEngine
from app.recognition.deepface_engine import DeepFaceEngine
from app.recognition.embedding_store import EmbeddingStore
from app.utils.logger import logger


@dataclass
class IdentityResult:
    employee_id: int
    confidence: float
    method: str  # 'insightface_high' | 'insightface_medium+deepface' | 'frame_high' | 'frame_medium+deepface'


class FaceRecognizer:
    # Close-up crop thresholds (registration / verify)
    HIGH_THRESHOLD   = 0.65
    MEDIUM_THRESHOLD = 0.50

    # Full-frame thresholds — faces are smaller, angles vary, so slightly lower
    FRAME_HIGH_THRESHOLD   = 0.52
    FRAME_MEDIUM_THRESHOLD = 0.38   # minimum InsightFace score to attempt DeepFace

    def __init__(
        self,
        insightface_engine: InsightFaceEngine,
        deepface_engine: DeepFaceEngine,
        embedding_store: EmbeddingStore,
        faces_dir: str = "data/employee_faces",
    ):
        self._insight = insightface_engine
        self._deepface = deepface_engine
        self._store = embedding_store
        self._faces_dir = Path(faces_dir)

    def identify(self, face_crop: np.ndarray) -> IdentityResult | None:
        """Identify who is in a close-up face crop."""
        embedding = self._insight.get_embedding(face_crop)
        if embedding is None:
            logger.debug("FaceRecognizer: no face detected in crop")
            return None

        all_embeddings = self._store.get_all()
        if not all_embeddings:
            return None

        best_id, best_score = self._insight.match(embedding, all_embeddings)
        logger.debug(f"FaceRecognizer.identify: best_id={best_id} score={best_score:.4f}")

        if best_score >= self.HIGH_THRESHOLD:
            return IdentityResult(employee_id=best_id, confidence=best_score, method="insightface_high")

        if best_score >= self.MEDIUM_THRESHOLD:
            verified, _ = self._verify_against_all(face_crop, best_id)
            if verified:
                return IdentityResult(employee_id=best_id, confidence=best_score, method="insightface_medium+deepface")

        logger.debug(f"FaceRecognizer.identify: unknown (score={best_score:.4f})")
        return None

    def identify_in_frame(
        self,
        frame: np.ndarray,
        tracks: list,
        camera_id: int,
    ) -> dict[str, IdentityResult]:
        """
        Run InsightFace on the full frame once → detect all faces → match to tracks.
        Returns {track_key: IdentityResult}.
        Falls back to head-crop recognition for any track whose face wasn't detected.
        """
        results: dict[str, IdentityResult] = {}

        if not tracks:
            return results

        all_embeddings = self._store.get_all()
        if not all_embeddings:
            return results

        # ── Full-frame detection ───────────────────────────────────────────
        detected_faces = self._insight.get_faces(frame)
        logger.info(
            f"identify_in_frame cam={camera_id}: "
            f"{len(detected_faces)} face(s) detected, {len(tracks)} track(s)"
        )

        from app.snapshots.snapshot_store import snapshot_store
        from app.store import state as _snap_state

        def _cam_label(cid: int) -> str:
            cam = _snap_state.get_camera(cid)
            return cam["location_label"] if cam else ""

        def _snap(face_bbox, track_id, emp_id, emp_name, conf, meth):
            """Crop face from frame and save snapshot (with dedup)."""
            key = f"{track_id}@{camera_id}"
            if not snapshot_store.should_save(key):
                return
            h, w = frame.shape[:2]
            pad = 8
            fx1, fy1, fx2, fy2 = face_bbox
            crop = frame[max(0, fy1 - pad):min(h, fy2 + pad),
                         max(0, fx1 - pad):min(w, fx2 + pad)]
            if crop.size == 0:
                return
            snapshot_store.save(
                face_crop=crop,
                camera_id=camera_id,
                camera_label=_cam_label(camera_id),
                employee_id=emp_id,
                employee_name=emp_name,
                confidence=conf,
                method=meth,
            )

        matched_track_ids: set[int] = set()
        assigned_tracks:  set[int] = set()  # prevents two faces → same track

        # ── Assign each detected face to nearest unassigned track ─────────
        # Use pure distance-based matching — no hard containment constraint.
        # High-angle CCTV cameras often detect face bbox ABOVE or beside the
        # person bbox due to perspective, so strict containment fails.
        _MAX_ASSIGN_DIST = 400  # px — face center must be within this of track head
        # 220 was too tight for close-up webcams where face bbox center is far from y1

        for face in detected_faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            face_cx = (fx1 + fx2) / 2
            face_cy = (fy1 + fy2) / 2

            best_track = None
            best_dist  = _MAX_ASSIGN_DIST   # reject anything farther than this

            for track in tracks:
                if track.track_id in assigned_tracks:
                    continue   # this track already has a face — skip
                # Expected head center = top-center of person bbox
                head_cx = (track.x1 + track.x2) / 2
                head_cy = track.y1  # face is at or above the top of person bbox
                dist = ((face_cx - head_cx) ** 2 + (face_cy - head_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist  = dist
                    best_track = track

            if best_track is None:
                logger.info(f"  face bbox=({fx1},{fy1},{fx2},{fy2}) — no track within {_MAX_ASSIGN_DIST}px")
                continue

            key = f"{best_track.track_id}@{camera_id}"
            matched_track_ids.add(best_track.track_id)
            assigned_tracks.add(best_track.track_id)

            embedding = face["embedding"]
            best_id, best_score = self._insight.match(embedding, all_embeddings)
            logger.info(
                f"  face→track {best_track.track_id}: best_id={best_id} score={best_score:.4f} "
                f"(need >={self.FRAME_HIGH_THRESHOLD} or >={self.FRAME_MEDIUM_THRESHOLD})"
            )

            face_bbox_tuple = (fx1, fy1, fx2, fy2)
            if best_score >= self.FRAME_HIGH_THRESHOLD:
                emp_name = _snap_state.get_employee(best_id)
                emp_name = emp_name["name"] if emp_name else str(best_id)
                results[key] = IdentityResult(
                    employee_id=best_id, confidence=best_score, method="frame_high"
                )
                logger.info(
                    f"identify_in_frame MATCH: cam={camera_id} track={best_track.track_id} "
                    f"→ emp={best_id} score={best_score:.4f}"
                )
                _snap(face_bbox_tuple, best_track.track_id, best_id, emp_name, best_score, "frame_high")
            elif best_score >= self.FRAME_MEDIUM_THRESHOLD:
                # Try top-3 candidates — top-1 may be wrong employee at similar score
                h, w = frame.shape[:2]
                fc = frame[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
                if fc.size > 0:
                    candidates = self._insight.match_top_n(embedding, all_embeddings, n=3)
                    matched_candidate = None
                    for cand_id, cand_score in candidates:
                        if cand_score < self.FRAME_MEDIUM_THRESHOLD:
                            break
                        verified, _ = self._verify_against_all(fc, cand_id)
                        logger.info(
                            f"  deepface candidate emp={cand_id} score={cand_score:.4f} verified={verified}"
                        )
                        if verified:
                            matched_candidate = (cand_id, cand_score)
                            break
                    if matched_candidate:
                        cand_id, cand_score = matched_candidate
                        emp_name = _snap_state.get_employee(cand_id)
                        emp_name = emp_name["name"] if emp_name else str(cand_id)
                        results[key] = IdentityResult(
                            employee_id=cand_id, confidence=cand_score,
                            method="frame_medium+deepface"
                        )
                        logger.info(
                            f"identify_in_frame MATCH (deepface): cam={camera_id} "
                            f"track={best_track.track_id} → emp={cand_id}"
                        )
                        _snap(face_bbox_tuple, best_track.track_id, cand_id, emp_name, cand_score, "frame_medium+deepface")
                    else:
                        _snap(face_bbox_tuple, best_track.track_id, None, None, best_score, "unverified")
            else:
                # score below both thresholds — unknown face
                _snap(face_bbox_tuple, best_track.track_id, None, None, best_score, "unknown")

        # ── Fallback: head-crop for unmatched tracks ───────────────────────
        # Any track whose face InsightFace didn't detect in the full frame
        unmatched = [t for t in tracks if t.track_id not in matched_track_ids]
        if unmatched:
            logger.debug(f"  {len(unmatched)} track(s) unmatched — trying head crop fallback")
            for track in unmatched:
                key = f"{track.track_id}@{camera_id}"
                if key in results:
                    continue
                crop = self._get_head_crop(frame, track)
                if crop is None:
                    continue
                embedding = self._insight.get_embedding(crop)
                if embedding is None:
                    continue
                best_id, best_score = self._insight.match(embedding, all_embeddings)
                logger.debug(f"  head-crop track={track.track_id}: best_id={best_id} score={best_score:.4f}")
                if best_score >= self.HIGH_THRESHOLD:
                    emp_name = _snap_state.get_employee(best_id)
                    emp_name = emp_name["name"] if emp_name else str(best_id)
                    results[key] = IdentityResult(
                        employee_id=best_id, confidence=best_score, method="crop_high"
                    )
                    snapshot_store.save(crop, camera_id, _cam_label(camera_id), best_id, emp_name, best_score, "crop_high")
                elif best_score >= self.MEDIUM_THRESHOLD:
                    verified, _ = self._verify_against_all(crop, best_id)
                    if verified:
                        emp_name = _snap_state.get_employee(best_id)
                        emp_name = emp_name["name"] if emp_name else str(best_id)
                        results[key] = IdentityResult(
                            employee_id=best_id, confidence=best_score, method="crop_medium+deepface"
                        )
                        snapshot_store.save(crop, camera_id, _cam_label(camera_id), best_id, emp_name, best_score, "crop_medium+deepface")

        return results

    def _get_head_crop(self, frame: np.ndarray, track) -> np.ndarray | None:
        """Crop the top 55% of a person track bbox as head region, with padding."""
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, track.x1 - pad)
        y1 = max(0, track.y1 - pad)
        x2 = min(w, track.x2 + pad)
        # Top 55% of the person bbox (larger than old 40% — more chance of capturing face)
        y2 = min(h, track.y1 + int((track.y2 - track.y1) * 0.55) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _load_all_reference_images(self, employee_id: int) -> list[np.ndarray]:
        """Load ALL registered face photos for this employee."""
        emp_dir = self._faces_dir / str(employee_id)
        images = []
        for name in ["photo_1.jpg", "photo_2.jpg", "photo_3.jpg", "photo_4.jpg", "photo_5.jpg", "photo.jpg"]:
            img_path = emp_dir / name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
        return images

    def _verify_against_all(self, face_crop: np.ndarray, employee_id: int) -> tuple[bool, float]:
        """
        Verify face_crop against ALL registered photos for the employee.
        Returns (verified, best_similarity) — passes if ANY photo verifies.
        """
        refs = self._load_all_reference_images(employee_id)
        if not refs:
            return False, 0.0
        best = 0.0
        for ref in refs:
            verified, sim = self._deepface.verify(face_crop, ref)
            if sim > best:
                best = sim
            if verified:
                return True, best
        return False, best


# Singleton — set in main.py lifespan, imported by pipeline
face_recognizer: FaceRecognizer | None = None

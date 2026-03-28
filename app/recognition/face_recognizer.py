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
    FRAME_HIGH_THRESHOLD   = 0.52   # lowered from 0.55 (0.5544 just barely made it)
    FRAME_MEDIUM_THRESHOLD = 0.35   # lowered from 0.42 → more DeepFace verification chances

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
            ref_image = self._load_reference_image(best_id)
            if ref_image is not None:
                verified, distance = self._deepface.verify(face_crop, ref_image)
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

        matched_track_ids: set[int] = set()
        assigned_tracks:  set[int] = set()  # prevents two faces → same track

        # ── Assign each detected face to nearest unassigned track ─────────
        # Use pure distance-based matching — no hard containment constraint.
        # High-angle CCTV cameras often detect face bbox ABOVE or beside the
        # person bbox due to perspective, so strict containment fails.
        _MAX_ASSIGN_DIST = 220  # px — face center must be within this of track head

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

            if best_score >= self.FRAME_HIGH_THRESHOLD:
                results[key] = IdentityResult(
                    employee_id=best_id, confidence=best_score, method="frame_high"
                )
                logger.info(
                    f"identify_in_frame MATCH: cam={camera_id} track={best_track.track_id} "
                    f"→ emp={best_id} score={best_score:.4f}"
                )
            elif best_score >= self.FRAME_MEDIUM_THRESHOLD:
                ref_image = self._load_reference_image(best_id)
                if ref_image is not None:
                    h, w = frame.shape[:2]
                    fc = frame[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
                    if fc.size > 0:
                        verified, dist_val = self._deepface.verify(fc, ref_image)
                        if verified:
                            results[key] = IdentityResult(
                                employee_id=best_id, confidence=best_score,
                                method="frame_medium+deepface"
                            )
                            logger.info(
                                f"identify_in_frame MATCH (deepface): cam={camera_id} "
                                f"track={best_track.track_id} → emp={best_id}"
                            )

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
                    results[key] = IdentityResult(
                        employee_id=best_id, confidence=best_score, method="crop_high"
                    )
                elif best_score >= self.MEDIUM_THRESHOLD:
                    ref_image = self._load_reference_image(best_id)
                    if ref_image is not None:
                        verified, _ = self._deepface.verify(crop, ref_image)
                        if verified:
                            results[key] = IdentityResult(
                                employee_id=best_id, confidence=best_score, method="crop_medium+deepface"
                            )

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

    def _load_reference_image(self, employee_id: int) -> np.ndarray | None:
        """Load first available face photo for this employee (photo_1.jpg, photo_2.jpg, ...)."""
        emp_dir = self._faces_dir / str(employee_id)
        # Try numbered photos first (new format), then legacy photo.jpg
        for name in ["photo_1.jpg", "photo_2.jpg", "photo_3.jpg", "photo.jpg"]:
            img_path = emp_dir / name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    return img
        return None


# Singleton — set in main.py lifespan, imported by pipeline
face_recognizer: FaceRecognizer | None = None

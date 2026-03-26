"""
FaceRecognizer — orchestrator combining InsightFace (primary) + FaceNet (fallback).

Flow:
  1. InsightFace extracts 512-dim embedding from face crop
  2. Cosine similarity compared against EmbeddingStore
  3. score >= HIGH_THRESHOLD (0.65) → confident match, return immediately
  4. MEDIUM_THRESHOLD (0.50) <= score < HIGH → run FaceNet (facenet-pytorch) verify
     - FaceNet confirms → return match
     - FaceNet rejects  → return Unknown
  5. score < MEDIUM_THRESHOLD → Unknown
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
    method: str  # 'insightface_high' | 'insightface_medium+deepface'


class FaceRecognizer:
    HIGH_THRESHOLD = 0.65
    MEDIUM_THRESHOLD = 0.50

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
        """
        Identify who is in face_crop.
        Returns IdentityResult if a match is found, None for unknown.
        """
        # ── Step 1: Extract embedding ──────────────────────────────────
        embedding = self._insight.get_embedding(face_crop)
        if embedding is None:
            logger.debug("FaceRecognizer: no face detected in crop")
            return None

        # ── Step 2: Compare against store ─────────────────────────────
        all_embeddings = self._store.get_all()
        if not all_embeddings:
            logger.debug("FaceRecognizer: embedding store is empty")
            return None

        best_id, best_score = self._insight.match(embedding, all_embeddings)
        logger.debug(f"FaceRecognizer: best match employee_id={best_id}, score={best_score:.4f}")

        # ── Step 3: High confidence — return immediately ───────────────
        if best_score >= self.HIGH_THRESHOLD:
            logger.info(f"FaceRecognizer: HIGH match employee_id={best_id} score={best_score:.4f}")
            return IdentityResult(
                employee_id=best_id,
                confidence=best_score,
                method="insightface_high",
            )

        # ── Step 4: Medium confidence — DeepFace fallback ─────────────
        if best_score >= self.MEDIUM_THRESHOLD:
            ref_image = self._load_reference_image(best_id)
            if ref_image is not None:
                verified, distance = self._deepface.verify(face_crop, ref_image)
                if verified:
                    logger.info(
                        f"FaceRecognizer: MEDIUM+DeepFace confirmed employee_id={best_id} "
                        f"score={best_score:.4f} dist={distance:.4f}"
                    )
                    return IdentityResult(
                        employee_id=best_id,
                        confidence=best_score,
                        method="insightface_medium+deepface",
                    )
                else:
                    logger.debug(
                        f"FaceRecognizer: DeepFace rejected employee_id={best_id} dist={distance:.4f}"
                    )

        # ── Step 5: Low confidence — unknown ──────────────────────────
        logger.debug(f"FaceRecognizer: unknown (best_score={best_score:.4f})")
        return None

    def _load_reference_image(self, employee_id: int) -> np.ndarray | None:
        img_path = self._faces_dir / str(employee_id) / "photo.jpg"
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path))
        return img if img is not None else None


# Singleton — set in main.py lifespan, imported by pipeline
face_recognizer: FaceRecognizer | None = None

"""
EmployeeService — face registration and image storage for Phases 1–4.

In Phase 5, swap AppState calls for SQLAlchemy session calls.
Function signatures stay the same — no API changes needed.
"""
from pathlib import Path

import cv2
import numpy as np

from app.store import AppState
from app.recognition.insightface_engine import InsightFaceEngine
from app.recognition.embedding_store import EmbeddingStore
from app.utils.logger import logger


FACES_DIR = Path("data/employee_faces")


class EmployeeService:
    def __init__(self, insightface: InsightFaceEngine, embedding_store: EmbeddingStore):
        self._insight = insightface
        self._store = embedding_store

    def register_face(
        self,
        employee_id: int,
        image_bytes: bytes,
        state: AppState,
    ) -> dict:
        """
        Register a face for an employee:
          1. Decode image bytes → numpy array
          2. Detect face + extract embedding via InsightFace
          3. Save embedding to EmbeddingStore (→ persisted to .pkl)
          4. Save original image to data/employee_faces/{id}/photo.jpg
          5. Mark employee face_registered = True in AppState

        Returns status dict with success/error details.
        """
        # 1. Decode image
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "Cannot decode image — unsupported format"}

        # 2. Extract embedding
        if not self._insight.ready:
            return {"success": False, "error": "InsightFace engine not loaded"}

        embedding = self._insight.get_embedding(img)
        if embedding is None:
            return {"success": False, "error": "No face detected in the uploaded image"}

        # 3. Save embedding
        self._store.add(employee_id, embedding)

        # 4. Save image to disk
        img_dir = FACES_DIR / str(employee_id)
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / "photo.jpg"
        cv2.imwrite(str(save_path), img)
        logger.info(f"EmployeeService: saved face image to {save_path}")

        # 5. Update AppState flag
        state.mark_face_registered(employee_id)
        logger.info(f"EmployeeService: face registered for employee {employee_id}")

        return {
            "success": True,
            "employee_id": employee_id,
            "embedding_dim": len(embedding),
            "image_path": str(save_path),
        }

    def delete_face(self, employee_id: int, state: AppState) -> None:
        """Remove embedding + stored image for an employee."""
        self._store.remove(employee_id)
        img_dir = FACES_DIR / str(employee_id)
        if img_dir.exists():
            import shutil
            shutil.rmtree(img_dir)
        state.update_employee(employee_id, face_registered=False)
        logger.info(f"EmployeeService: deleted face data for employee {employee_id}")

    def get_face_image_path(self, employee_id: int) -> Path | None:
        """Return path to stored face image, or None if not registered."""
        p = FACES_DIR / str(employee_id) / "photo.jpg"
        return p if p.exists() else None


# Singleton — set in main.py lifespan
employee_service: EmployeeService | None = None

"""
EmployeeService — face registration and image storage.

Each employee can register multiple face photos (different angles).
All embeddings are stored in EmbeddingStore and used during recognition.
"""
import pickle
import shutil
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
        Register one face photo for an employee.
        Multiple calls append additional angles — they do NOT replace existing ones.

        Steps:
          1. Decode image bytes → numpy array
          2. Detect face + extract embedding via InsightFace
          3. Append embedding to EmbeddingStore (→ persisted to .pkl)
          4. Save image as data/employee_faces/{id}/photo_{n}.jpg
          5. Mark face_registered = True in AppState
          6. Persist to DB if enabled
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

        # 3. Append embedding
        photo_index = self._store.photo_count(employee_id)  # 0-based index before add
        total = self._store.add(employee_id, embedding)

        # 4. Save image — photo_1.jpg, photo_2.jpg, ...
        img_dir = FACES_DIR / str(employee_id)
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"photo_{total}.jpg"
        cv2.imwrite(str(save_path), img)
        logger.info(f"EmployeeService: saved face image to {save_path}")

        # 5. Update AppState flag
        state.mark_face_registered(employee_id)

        # 6. Persist to DB (if enabled)
        try:
            from app.database.connection import is_db_enabled, get_db
            if is_db_enabled():
                emb_bytes = pickle.dumps(embedding)
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE employees SET face_registered=1 WHERE id=?", employee_id
                    )
                    cursor.execute(
                        "INSERT INTO face_embeddings (employee_id, embedding) VALUES (?, ?)",
                        employee_id, emb_bytes,
                    )
        except Exception as e:
            logger.warning(f"EmployeeService: DB face save failed (non-fatal): {e}")

        logger.info(
            f"EmployeeService: face registered for employee {employee_id} "
            f"(photo {total}/{total})"
        )
        return {
            "success": True,
            "employee_id": employee_id,
            "photo_index": total,
            "total_photos": total,
            "embedding_dim": len(embedding),
            "image_path": str(save_path),
        }

    def delete_face(self, employee_id: int, state: AppState) -> None:
        """Remove ALL embeddings and stored images for an employee."""
        self._store.remove(employee_id)
        img_dir = FACES_DIR / str(employee_id)
        if img_dir.exists():
            shutil.rmtree(img_dir)
        state.update_employee(employee_id, face_registered=False)

        try:
            from app.database.connection import is_db_enabled, get_db
            if is_db_enabled():
                with get_db() as conn:
                    conn.cursor().execute(
                        "DELETE FROM face_embeddings WHERE employee_id=?", employee_id
                    )
                    conn.cursor().execute(
                        "UPDATE employees SET face_registered=0 WHERE id=?", employee_id
                    )
        except Exception as e:
            logger.warning(f"EmployeeService: DB face delete failed (non-fatal): {e}")

        logger.info(f"EmployeeService: deleted all face data for employee {employee_id}")

    def delete_single_photo(self, employee_id: int, photo_index: int, state: AppState) -> bool:
        """
        Remove one specific photo (1-based index as shown to users).
        Returns False if index is out of range.
        """
        zero_idx = photo_index - 1
        removed = self._store.remove_one(employee_id, zero_idx)
        if not removed:
            return False

        # Remove the image file
        img_path = FACES_DIR / str(employee_id) / f"photo_{photo_index}.jpg"
        if img_path.exists():
            img_path.unlink()

        # If no photos left, clear face_registered flag
        if self._store.photo_count(employee_id) == 0:
            state.update_employee(employee_id, face_registered=False)

        logger.info(
            f"EmployeeService: deleted photo {photo_index} for employee {employee_id}"
        )
        return True

    def get_face_photos(self, employee_id: int) -> list[Path]:
        """Return paths of all stored face photos, sorted by index."""
        img_dir = FACES_DIR / str(employee_id)
        if not img_dir.exists():
            return []
        photos = sorted(img_dir.glob("photo_*.jpg"))
        return photos

    def get_face_image_path(self, employee_id: int, photo_index: int = 1) -> Path | None:
        """Return path to a specific face photo (1-based), or None if not found."""
        p = FACES_DIR / str(employee_id) / f"photo_{photo_index}.jpg"
        return p if p.exists() else None


# Singleton — set in main.py lifespan
employee_service: EmployeeService | None = None

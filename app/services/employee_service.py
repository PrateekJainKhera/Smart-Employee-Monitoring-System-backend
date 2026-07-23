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
        embedding_detected = embedding is not None

        # 3. Save image — always save, even if no embedding detected.
        # Top-angle / side-angle photos may not yield an InsightFace embedding
        # but are still useful as FaceNet reference images for secondary verification.
        img_dir = FACES_DIR / str(employee_id)
        img_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(img_dir.glob("photo_*.jpg"))
        next_index = len(existing) + 1
        save_path = img_dir / f"photo_{next_index}.jpg"
        cv2.imwrite(str(save_path), img)
        logger.info(
            f"EmployeeService: saved face image to {save_path} "
            f"({'embedding extracted' if embedding_detected else 'no embedding — saved as FaceNet reference'})"
        )
        total = next_index

        # 4. Append embedding only if face was detected
        if embedding_detected:
            self._store.add(employee_id, embedding)

        # 5. Update AppState flag
        state.mark_face_registered(employee_id)

        # 6. Persist to DB (if enabled)
        try:
            from app.database.connection import is_db_enabled, get_db
            if is_db_enabled():
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE employees SET face_registered=1 WHERE id=?", employee_id
                    )
                    if embedding_detected:
                        emb_bytes = pickle.dumps(embedding)
                        cursor.execute(
                            "INSERT INTO face_embeddings (employee_id, embedding) VALUES (?, ?)",
                            employee_id, emb_bytes,
                        )
        except Exception as e:
            logger.warning(f"EmployeeService: DB face save failed (non-fatal): {e}")

        logger.info(
            f"EmployeeService: face registered for employee {employee_id} "
            f"(photo {total}, embedding={'yes' if embedding_detected else 'no — FaceNet reference only'})"
        )
        return {
            "success": True,
            "employee_id": employee_id,
            "photo_index": total,
            "total_photos": total,
            "embedding_dim": len(embedding) if embedding_detected else 0,
            "image_path": str(save_path),
            "warning": None if embedding_detected else "Face not clearly detected from this angle — photo saved as secondary reference only. Primary recognition requires at least one front/side-angle photo.",
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
        Returns False only if neither the file nor the embedding exists.

        File and embedding are deleted independently — they can fall out of sync
        if a photo was uploaded but failed to generate an embedding (or vice versa).
        """
        zero_idx = photo_index - 1

        # Remove embedding if present (ignore if out of range — file/embedding may be out of sync)
        self._store.remove_one(employee_id, zero_idx)

        # Remove the image file — this is the source of truth for what the user sees
        img_path = FACES_DIR / str(employee_id) / f"photo_{photo_index}.jpg"
        if not img_path.exists():
            # Nothing on disk either — truly not found
            return False
        img_path.unlink()

        # If no photos left, clear face_registered flag
        remaining_files = list((FACES_DIR / str(employee_id)).glob("photo_*.jpg"))
        if not remaining_files and self._store.photo_count(employee_id) == 0:
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

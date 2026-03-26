"""
InsightFaceEngine — primary face embedding extractor and matcher.

Uses InsightFace buffalo_s model (fast, CPU-friendly).
Outputs 512-dim normalized embeddings compared via cosine similarity.
"""
import numpy as np
from app.utils.logger import logger
from app.utils.helpers import cosine_similarity


class InsightFaceEngine:
    def __init__(self, model_name: str = "buffalo_s", det_size: tuple = (640, 640)):
        self._app = None
        self._model_name = model_name
        self._det_size = det_size
        self._load()

    def _load(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name=self._model_name,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=self._det_size)
            logger.info(f"InsightFaceEngine loaded model '{self._model_name}'")
        except Exception as e:
            logger.error(f"InsightFaceEngine failed to load: {e}")
            self._app = None

    @property
    def ready(self) -> bool:
        return self._app is not None

    def get_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect face in image and return its 512-dim embedding.
        If multiple faces found, uses the largest one.
        Returns None if no face detected or engine not loaded.
        """
        if not self.ready:
            return None
        try:
            faces = self._app.get(image)
            if not faces:
                return None
            # Pick the largest face by bounding box area
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.embedding
            # Normalize to unit vector for consistent cosine similarity
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            logger.warning(f"InsightFaceEngine.get_embedding error: {e}")
            return None

    def match(
        self,
        embedding: np.ndarray,
        store: dict[int, np.ndarray],
    ) -> tuple[int | None, float]:
        """
        Compare embedding against all stored embeddings.
        Returns (best_employee_id, best_score) or (None, 0.0) if store is empty.
        """
        if not store:
            return None, 0.0

        best_id: int | None = None
        best_score: float = -1.0

        for emp_id, ref_emb in store.items():
            score = cosine_similarity(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_id = emp_id

        return best_id, best_score

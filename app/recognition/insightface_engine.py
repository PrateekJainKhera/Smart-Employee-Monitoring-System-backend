"""
InsightFaceEngine — primary face embedding extractor and matcher.

Uses InsightFace buffalo_s model (fast, CPU-friendly).
Outputs 512-dim normalized embeddings compared via cosine similarity.
"""
import numpy as np
import cv2
from app.utils.logger import logger
from app.utils.helpers import cosine_similarity


def _normalize_brightness(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to LAB lightness channel — boosts contrast in dark face images
    without overexposing bright areas. Works on BGR numpy images.
    """
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return img  # fallback: return original if anything fails


class InsightFaceEngine:
    def __init__(self, model_name: str = "buffalo_l", det_size: tuple = (640, 640)):
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
            # det_thresh=0.3 (default 0.5) — detects angled/partial faces from CCTV
            self._app.prepare(ctx_id=0, det_size=self._det_size, det_thresh=0.3)
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

    def get_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Run face detection + embedding on the full frame.
        Returns list of dicts: [{"bbox": (x1,y1,x2,y2), "embedding": np.ndarray}, ...]
        Much faster than cropping individual heads — one inference call for all faces.
        """
        if not self.ready:
            return []
        try:
            faces = self._app.get(frame)
            results = []
            for face in faces:
                emb = face.embedding
                if emb is None:
                    continue
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                results.append({"bbox": (x1, y1, x2, y2), "embedding": emb})
            return results
        except Exception as e:
            logger.warning(f"InsightFaceEngine.get_faces error: {e}")
            return []

    def match(
        self,
        embedding: np.ndarray,
        store: dict[int, list[np.ndarray]],
    ) -> tuple[int | None, float]:
        """
        Compare embedding against all stored embeddings for every employee.
        Returns (best_employee_id, best_score).
        Returns (None, 0.0) if store is empty.
        """
        results = self.match_top_n(embedding, store, n=1)
        if not results:
            return None, 0.0
        return results[0]

    def match_top_n(
        self,
        embedding: np.ndarray,
        store: dict[int, list[np.ndarray]],
        n: int = 3,
    ) -> list[tuple[int, float]]:
        """
        Return top-N (employee_id, best_score) candidates sorted by score descending.
        Each employee contributes only their best embedding score.
        """
        if not store:
            return []

        per_employee: dict[int, float] = {}
        for emp_id, ref_embeddings in store.items():
            for ref_emb in ref_embeddings:
                score = cosine_similarity(embedding, ref_emb)
                if score > per_employee.get(emp_id, -1.0):
                    per_employee[emp_id] = score

        sorted_candidates = sorted(per_employee.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:n]

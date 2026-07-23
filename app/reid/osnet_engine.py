"""
OSNetEngine — full-body ReID using OSNet x1_0.

Uses torchreid library (pip install torchreid).
Input:  BGR person crop from YOLO bounding box
Output: 512-dim L2-normalized embedding vector

Used only when recognition_mode == 'face_reid'.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from app.utils.logger import logger

# Standard ReID input size (height × width)
_INPUT_H = 256
_INPUT_W = 128

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OSNetEngine:
    """
    Thin wrapper around OSNet x1_0 for person Re-Identification.

    Lazy initialization — model is loaded on first call to get_embedding().
    """

    def __init__(self):
        self._model  = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        if self._model is not None:
            return
        logger.info("OSNetEngine: loading osnet_x1_0 via torchreid (downloads weights on first run)...")
        try:
            import torchreid
            model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1000,
                pretrained=True,
            )
            model.eval()
            model.to(self._device)
            self._model = model
            logger.info(f"OSNetEngine: model loaded on {self._device}")
        except ImportError as ie:
            logger.error(
                f"OSNetEngine: missing dependency — {ie}. "
                "Run: pip install torchreid tensorboard gdown"
            )
            raise
        except Exception as e:
            logger.error(f"OSNetEngine: failed to load model — {e}")
            raise

    def _preprocess(self, bgr_crop: np.ndarray) -> torch.Tensor:
        """Resize, convert to RGB, normalize, return (1, 3, H, W) tensor."""
        img = cv2.resize(bgr_crop, (_INPUT_W, _INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = img.transpose(2, 0, 1)                          # HWC → CHW
        tensor = torch.from_numpy(img).unsqueeze(0)           # add batch dim
        return tensor.to(self._device)

    def get_embedding(self, person_crop: np.ndarray) -> np.ndarray | None:
        """
        Extract 512-dim L2-normalized ReID embedding from a full-body crop.

        Args:
            person_crop: BGR image of the full person bounding box (from YOLO)

        Returns:
            512-dim float32 numpy array, or None if crop is too small / model fails
        """
        if person_crop is None or person_crop.size == 0:
            return None
        h, w = person_crop.shape[:2]
        if h < 64 or w < 32:
            return None

        try:
            self._load_model()
            tensor = self._preprocess(person_crop)
            with torch.no_grad():
                feat = self._model(tensor)               # (1, 512)
                feat = F.normalize(feat, p=2, dim=1)    # L2 normalize
            return feat.cpu().numpy()[0]                 # (512,)
        except Exception as e:
            logger.warning(f"OSNetEngine.get_embedding error: {e}")
            return None

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings. Range [0, 1]."""
        if emb1 is None or emb2 is None:
            return 0.0
        sim = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, sim))

    def match(
        self,
        query: np.ndarray,
        gallery: dict[int, np.ndarray],
    ) -> tuple[int, float]:
        """
        Find the best matching employee in gallery.

        Args:
            query:   512-dim query embedding
            gallery: {employee_id: 512-dim embedding}

        Returns:
            (best_employee_id, best_similarity_score)
        """
        best_id    = -1
        best_score = 0.0
        for emp_id, ref_emb in gallery.items():
            score = self.similarity(query, ref_emb)
            if score > best_score:
                best_score = score
                best_id    = emp_id
        return best_id, best_score


# Singleton — initialized in main.py lifespan, imported by pipeline
osnet_engine: OSNetEngine | None = None

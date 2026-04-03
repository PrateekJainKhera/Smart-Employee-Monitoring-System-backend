"""
FaceNetEngine — fallback face verifier using facenet-pytorch (InceptionResnetV1).

Replaces DeepFace (which requires TensorFlow — no Python 3.14 build available).
Uses the same interface: verify(face_img, reference_img) → (bool, float)

Model: InceptionResnetV1 pretrained on VGGFace2 → 512-dim embeddings
Verification: cosine similarity, threshold 0.5
"""
import numpy as np
from app.utils.logger import logger

VERIFY_THRESHOLD = 0.55  # cosine similarity — above this = same person


class DeepFaceEngine:
    """
    Drop-in replacement for DeepFace.
    Uses facenet-pytorch (PyTorch-based, no TensorFlow required).
    Called by FaceRecognizer when InsightFace confidence is in the medium range.
    """

    def __init__(self):
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            self._model = InceptionResnetV1(pretrained="vggface2").eval()
            # Move to CPU explicitly
            self._model = self._model.to("cpu")
            logger.info("FaceNetEngine (facenet-pytorch) loaded — VGGFace2 pretrained")
        except Exception as e:
            logger.warning(f"FaceNetEngine failed to load: {e}. Fallback verification disabled.")
            self._model = None

    def verify(
        self,
        face_img: np.ndarray,
        reference_img: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Verify whether face_img and reference_img show the same person.

        Returns (verified: bool, similarity: float).
        Higher similarity = more similar (cosine similarity 0–1).
        Returns (False, 0.0) if model not loaded or on any error.
        """
        if self._model is None:
            return False, 0.0

        try:
            import torch
            from PIL import Image

            emb1 = self._get_embedding(face_img)
            emb2 = self._get_embedding(reference_img)

            if emb1 is None or emb2 is None:
                return False, 0.0

            # Cosine similarity
            similarity = float(
                torch.nn.functional.cosine_similarity(
                    emb1.unsqueeze(0), emb2.unsqueeze(0)
                ).item()
            )
            verified = similarity >= VERIFY_THRESHOLD
            logger.debug(f"FaceNetEngine verify: similarity={similarity:.4f}, verified={verified}")
            return verified, similarity

        except Exception as e:
            logger.warning(f"FaceNetEngine.verify error: {e}")
            return False, 0.0

    def _get_embedding(self, img: np.ndarray):
        """Convert BGR numpy image → 512-dim FaceNet embedding tensor."""
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as transforms

            # BGR → RGB → PIL
            rgb = img[:, :, ::-1].copy()
            pil = Image.fromarray(rgb).resize((160, 160))

            # Normalize to [-1, 1] as expected by InceptionResnetV1
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            tensor = transform(pil).unsqueeze(0)  # (1, 3, 160, 160)

            with torch.no_grad():
                embedding = self._model(tensor)  # (1, 512)
            return embedding.squeeze(0)           # (512,)

        except Exception as e:
            logger.warning(f"FaceNetEngine._get_embedding error: {e}")
            return None

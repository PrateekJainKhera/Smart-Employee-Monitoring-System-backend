import cv2
import numpy as np
from datetime import datetime, timezone


def frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Convert a BGR OpenCV frame to JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Resize frame keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h))


def draw_text(frame: np.ndarray, text: str, pos: tuple[int, int],
              color: tuple = (0, 255, 0), scale: float = 0.6) -> np.ndarray:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return frame


def get_face_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Crop face region from frame given (x1, y1, x2, y2) bbox."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def get_head_crop(frame: np.ndarray, bbox: tuple[int, int, int, int],
                  head_ratio: float = 0.40, padding: float = 0.15) -> np.ndarray | None:
    """
    Crop only the head region from a full-body person bounding box.

    Takes the top `head_ratio` (40%) of the bbox height as the head area,
    then adds `padding` (15%) on all sides for context.

    This is much faster than passing the full body box to InsightFace — the
    face detector only scans the head region instead of the whole body.

    Minimum crop size: 60×60 px. Returns None if person is too small/far.
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1

    if h < 80 or w < 40:          # person too small / too far from camera
        return None

    # Head region = top 40% of the person box
    head_y2 = y1 + int(h * head_ratio)

    # Add padding for context (helps InsightFace align the face)
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    fx1 = max(0, x1 - pad_x)
    fy1 = max(0, y1 - pad_y)
    fx2 = min(frame.shape[1], x2 + pad_x)
    fy2 = min(frame.shape[0], head_y2 + pad_y)

    if fx2 - fx1 < 60 or fy2 - fy1 < 60:
        return None

    return frame[fy1:fy2, fx1:fx2]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_camera_source(rtsp_url: str) -> int | str:
    """
    Convert camera URL to OpenCV source.
    '0', '1', etc. → int (webcam index)
    'rtsp://...'   → str (RTSP stream)
    """
    stripped = rtsp_url.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def bbox_to_deepsort(bbox: tuple[int, int, int, int], confidence: float) -> list:
    """Convert (x1,y1,x2,y2) bbox to DeepSORT format [x1,y1,w,h,confidence]."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1, confidence]


def format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)}m"
    return f"{int(minutes // 60)}h {int(minutes % 60)}m"


def classify_break(duration_minutes: float, short_min: int = 10, medium_min: int = 20) -> str:
    if duration_minutes < short_min:
        return "short"
    if duration_minutes < medium_min:
        return "medium"
    return "long"

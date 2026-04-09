from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Cameras
    camera_urls: List[str] = []

    # YOLO
    yolo_weights_path: str = "weights/yolov8n.pt"

    # Face recognition thresholds
    face_recognition_threshold: float = 0.6
    face_high_threshold: float = 0.65
    face_medium_threshold: float = 0.5

    # Break duration rules (minutes)
    break_short_min: int = 10
    break_medium_min: int = 20

    # Auto check-out rules
    # After office_end_hour (local time), if employee is out >= office_end_break_min → checkout
    # At any time, if employee is out >= max_break_min → checkout (safety net)
    office_end_hour: int = 19       # 7 PM — end of workday
    office_end_break_min: int = 15  # minutes out after 7 PM before auto-checkout
    max_break_min: int = 45         # max break at any time — auto-checkout after 45 min away

    # Recognition mode
    # face_only        — face recognition only (default, best for uniform companies)
    # face_clothing    — face primary + color histogram to link anonymous tracks
    # face_reid        — face primary + OSNet ReID embeddings (future)
    recognition_mode: str = "face_only"

    # Clothing ReID settings (used when recognition_mode != face_only)
    reid_similarity_threshold: float = 0.75   # min histogram similarity to link tracks
    reid_time_window_min: int = 60            # max minutes back to search for matching track
    reid_same_camera_only: bool = True        # only link tracks from same camera

    # Embeddings persistence
    embeddings_path: str = "data/embeddings.pkl"

    # Logging
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Phase 5+: optional DB
    database_url: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

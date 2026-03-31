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
    max_break_min: int = 120        # max break at any time (2 hours safety net)

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

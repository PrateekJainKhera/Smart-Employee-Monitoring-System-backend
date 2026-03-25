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

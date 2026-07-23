"""
Settings API — get and update runtime recognition mode and ReID parameters.
Changes take effect immediately without restart.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/settings", tags=["Settings"])

VALID_MODES = {"face_only", "face_clothing", "face_reid"}

_ENV_PATH = None

def _get_env_path() -> str:
    global _ENV_PATH
    if _ENV_PATH is None:
        from pathlib import Path
        # Walk up from this file to find .env
        p = Path(__file__).parent
        for _ in range(5):
            candidate = p / ".env"
            if candidate.exists():
                _ENV_PATH = str(candidate)
                break
            p = p.parent
    return _ENV_PATH or ""

def _persist_env(key: str, value: str) -> None:
    """Update a single key in the .env file so changes survive restarts."""
    path = _get_env_path()
    if not path:
        return
    try:
        from pathlib import Path
        content = Path(path).read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}\n")
        Path(path).write_text("".join(lines), encoding="utf-8")
    except Exception:
        pass  # best-effort — runtime change still applies even if persist fails


class SettingsResponse(BaseModel):
    recognition_mode: str
    reid_similarity_threshold: float
    reid_time_window_min: int
    reid_same_camera_only: bool


class SettingsUpdate(BaseModel):
    recognition_mode: Optional[str] = None
    reid_similarity_threshold: Optional[float] = None
    reid_time_window_min: Optional[int] = None
    reid_same_camera_only: Optional[bool] = None


@router.get("", response_model=SettingsResponse)
def get_settings():
    """Return current recognition settings."""
    from app.config import settings
    return SettingsResponse(
        recognition_mode=settings.recognition_mode,
        reid_similarity_threshold=settings.reid_similarity_threshold,
        reid_time_window_min=settings.reid_time_window_min,
        reid_same_camera_only=settings.reid_same_camera_only,
    )


@router.put("", response_model=SettingsResponse)
def update_settings(body: SettingsUpdate):
    """Update recognition settings at runtime — no restart needed."""
    from app.config import settings

    if body.recognition_mode is not None:
        if body.recognition_mode not in VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Must be one of: {', '.join(VALID_MODES)}"
            )
        settings.recognition_mode = body.recognition_mode
        _persist_env("RECOGNITION_MODE", body.recognition_mode)

    if body.reid_similarity_threshold is not None:
        if not 0.0 <= body.reid_similarity_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
        settings.reid_similarity_threshold = body.reid_similarity_threshold

    if body.reid_time_window_min is not None:
        if body.reid_time_window_min < 1:
            raise HTTPException(status_code=400, detail="Time window must be at least 1 minute")
        settings.reid_time_window_min = body.reid_time_window_min

    if body.reid_same_camera_only is not None:
        settings.reid_same_camera_only = body.reid_same_camera_only

    return get_settings()

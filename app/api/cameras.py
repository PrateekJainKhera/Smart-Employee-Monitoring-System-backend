import asyncio
import cv2
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response, StreamingResponse
from app.models.schemas import CameraCreate, CameraResponse, CameraUpdate
from app.dependencies import get_state
from app.store import AppState
from app.camera.camera_manager import camera_manager
from app.pipeline.processing_pipeline import pipeline_manager
from app.utils.helpers import frame_to_jpeg, draw_text

router = APIRouter(prefix="/cameras", tags=["Cameras"])

# ── CRUD ─────────────────────────────────────────────────────────────

@router.post("", response_model=CameraResponse, status_code=201)
def create_camera(data: CameraCreate, state: AppState = Depends(get_state)):
    cam = state.add_camera(
        name=data.name,
        location_label=data.location_label,
        rtsp_url=data.rtsp_url,
    )
    camera_manager.start_camera(cam["id"], cam["rtsp_url"], cam["location_label"])
    pipeline_manager.start_pipeline(cam["id"], cam["location_label"])
    return cam


@router.get("", response_model=list[CameraResponse])
def list_cameras(state: AppState = Depends(get_state)):
    return state.list_cameras()


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int, state: AppState = Depends(get_state)):
    cam = state.get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: int, data: CameraUpdate, state: AppState = Depends(get_state)):
    updated = state.update_camera(camera_id, **data.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if data.is_active is not None:
        if data.is_active:
            camera_manager.start_camera(updated["id"], updated["rtsp_url"], updated["location_label"])
        else:
            camera_manager.stop_camera(camera_id)
    return updated


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: int, state: AppState = Depends(get_state)):
    if not state.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    camera_manager.stop_camera(camera_id)
    pipeline_manager.stop_pipeline(camera_id)


# ── Single snapshot (for Postman testing) ───────────────────────────

@router.get("/{camera_id}/preview")
def preview_camera(camera_id: int, state: AppState = Depends(get_state)):
    cam = state.get_camera(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    frame = camera_manager.get_frame(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    draw_text(frame, f"{cam['name']} | {cam['location_label']}", (10, 25))
    return Response(content=frame_to_jpeg(frame), media_type="image/jpeg")


# ── MJPEG streaming endpoints ────────────────────────────────────────

async def _mjpeg_raw(camera_id: int):
    """Push raw frames as MJPEG — targets 30fps."""
    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is not None:
            jpeg = frame_to_jpeg(frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        await asyncio.sleep(0.033)   # ~30 fps


async def _mjpeg_detected(camera_id: int):
    """
    Push MJPEG with YOLO boxes.
    Detection runs every 5th frame so video stays smooth on CPU.
    Last known boxes are reused on intermediate frames.
    """
    from app.detection.yolo_detector import YOLODetector, BoundingBox
    from app.config import settings

    detector = YOLODetector(
        weights_path=settings.yolo_weights_path,
        confidence=0.45,
        device="cpu",
    )

    last_boxes: list[BoundingBox] = []
    frame_count = 0

    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is not None:
            frame_count += 1
            # Run detection every 5th frame — reuse boxes on others
            if frame_count % 5 == 0:
                last_boxes = detector.detect(frame)

            annotated = detector.draw_boxes(frame, last_boxes)
            draw_text(annotated, f"Persons: {len(last_boxes)}", (10, 25))

            jpeg = frame_to_jpeg(annotated)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        await asyncio.sleep(0.033)


@router.get("/{camera_id}/stream")
async def stream_camera(camera_id: int, state: AppState = Depends(get_state)):
    """MJPEG live stream — use directly as <img src='...'> in browser."""
    if state.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _mjpeg_raw(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/stream/detected")
async def stream_camera_detected(camera_id: int, state: AppState = Depends(get_state)):
    """MJPEG live stream with YOLO person detection boxes."""
    if state.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _mjpeg_detected(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _mjpeg_tracked(camera_id: int):
    """
    Push MJPEG with DeepSORT track IDs + bounding boxes.
    Reads pre-computed tracks from the pipeline (no per-request detection).
    """
    while True:
        frame = camera_manager.get_frame(camera_id)
        if frame is not None:
            annotated = frame.copy()
            tracks = pipeline_manager.get_tracks(camera_id)

            for track in tracks:
                # Draw bounding box
                cv2.rectangle(annotated, (track.x1, track.y1), (track.x2, track.y2),
                              (0, 200, 255), 2)
                # Draw track ID label
                label = f"ID {track.track_id}"
                label_y = track.y1 - 8 if track.y1 > 20 else track.y1 + 18
                cv2.putText(annotated, label, (track.x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            draw_text(annotated, f"Tracked: {len(tracks)}", (10, 25))
            jpeg = frame_to_jpeg(annotated)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        await asyncio.sleep(0.033)


@router.get("/{camera_id}/stream/tracked")
async def stream_camera_tracked(camera_id: int, state: AppState = Depends(get_state)):
    """MJPEG live stream with DeepSORT track IDs overlaid on each person."""
    if state.get_camera(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        _mjpeg_tracked(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

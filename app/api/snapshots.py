"""
Snapshots REST endpoints.

GET /snapshots              → list recent face-crop snapshots (with filters)
GET /snapshots/{id}/image   → serve the JPEG image
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter(prefix="/snapshots", tags=["Snapshots"])


@router.get("")
def list_snapshots(
    camera_id:   Optional[int]  = Query(None, description="Filter by camera ID"),
    employee_id: Optional[int]  = Query(None, description="Filter by employee ID"),
    matched:     Optional[bool] = Query(None, description="true=matched only, false=unknown only, omit=all"),
    limit:       int            = Query(100, le=500, description="Max results"),
):
    """
    Return recent face-crop snapshots from the in-memory ring buffer.
    Newest first.  Supports filtering by camera, employee, and match status.
    """
    from app.snapshots.snapshot_store import snapshot_store
    return snapshot_store.list(
        camera_id=camera_id,
        employee_id=employee_id,
        matched=matched,
        limit=limit,
    )


@router.get("/{filename}/image")
def get_snapshot_image(filename: str):
    """Serve the JPEG bytes for a snapshot from memory."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    from app.snapshots.snapshot_store import snapshot_store
    data = snapshot_store.get_image_bytes(filename)
    if data is None:
        raise HTTPException(status_code=404, detail="Snapshot not found or expired")

    return Response(content=data, media_type="image/jpeg")

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import FileResponse
from app.models.schemas import EmployeeCreate, EmployeeResponse, EmployeeUpdate
from app.dependencies import get_state
from app.store import AppState
from app.utils.logger import logger

router = APIRouter(prefix="/employees", tags=["Employees"])


# ── DB helpers (no-op when DB is not configured) ───────────────────────────

def _db_insert_employee(name: str, dept: str, desig: str, email: str) -> int | None:
    """Insert employee into DB and return the new ID, or None if DB is disabled."""
    try:
        from app.database.connection import is_db_enabled, get_db
        if not is_db_enabled():
            return None
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO employees (name, department, designation, email) "
                "OUTPUT INSERTED.id VALUES (?, ?, ?, ?)",
                name, dept, desig, email,
            )
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.warning(f"DB: failed to insert employee: {e}")
        return None


def _db_delete_employee(employee_id: int) -> None:
    try:
        from app.database.connection import is_db_enabled, get_db
        if not is_db_enabled():
            return
        with get_db() as conn:
            conn.cursor().execute("DELETE FROM employees WHERE id=?", employee_id)
    except Exception as e:
        logger.warning(f"DB: failed to delete employee {employee_id}: {e}")


def _db_update_employee(employee_id: int, **kwargs) -> None:
    if not kwargs:
        return
    try:
        from app.database.connection import is_db_enabled, get_db
        if not is_db_enabled():
            return
        cols = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [employee_id]
        with get_db() as conn:
            conn.cursor().execute(f"UPDATE employees SET {cols} WHERE id=?", *vals)
    except Exception as e:
        logger.warning(f"DB: failed to update employee {employee_id}: {e}")


# ── Helpers ───────────────────────────────────────────────────────────

def _get_services():
    """Lazily import singletons to avoid circular imports at module load."""
    from app.services.employee_service import employee_service
    from app.recognition.face_recognizer import face_recognizer
    return employee_service, face_recognizer


# ── CRUD ─────────────────────────────────────────────────────────────

@router.post("", response_model=EmployeeResponse, status_code=201)
def create_employee(data: EmployeeCreate, state: AppState = Depends(get_state)):
    # Write to DB first (get the canonical ID), then cache in AppState
    db_id = _db_insert_employee(data.name, data.department, data.designation, data.email)
    emp = state.add_employee(
        name=data.name,
        department=data.department,
        designation=data.designation,
        email=data.email,
        employee_id=db_id,  # None → AppState auto-increments
    )
    return emp


@router.get("", response_model=list[EmployeeResponse])
def list_employees(state: AppState = Depends(get_state)):
    return state.list_employees()


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: int, state: AppState = Depends(get_state)):
    emp = state.get_employee(employee_id)
    if emp is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp


@router.put("/{employee_id}", response_model=EmployeeResponse)
def update_employee(employee_id: int, data: EmployeeUpdate, state: AppState = Depends(get_state)):
    patch = data.model_dump(exclude_none=True)
    updated = state.update_employee(employee_id, **patch)
    if updated is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    _db_update_employee(employee_id, **patch)
    return updated


@router.delete("/{employee_id}", status_code=204)
def delete_employee(employee_id: int, state: AppState = Depends(get_state)):
    svc, _ = _get_services()
    if svc:
        svc.delete_face(employee_id, state)
    if not state.delete_employee(employee_id):
        raise HTTPException(status_code=404, detail="Employee not found")
    _db_delete_employee(employee_id)


# ── Face Registration ─────────────────────────────────────────────────

@router.post("/{employee_id}/face", status_code=200)
async def register_face(
    employee_id: int,
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
):
    """
    Upload one face photo for an employee.
    Each call ADDS a new photo — it does not replace existing ones.
    Upload from different angles (front, left, right) for better recognition.
    """
    if state.get_employee(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    svc, _ = _get_services()
    if svc is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    image_bytes = await file.read()
    result = svc.register_face(employee_id, image_bytes, state)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])

    logger.info(f"Face photo {result['total_photos']} registered for employee {employee_id}")
    return {
        "message": f"Face photo {result['total_photos']} registered successfully",
        "employee_id": employee_id,
        "photo_index": result["total_photos"],
        "total_photos": result["total_photos"],
        "embedding_dim": result["embedding_dim"],
    }


@router.get("/{employee_id}/faces")
def list_face_photos(employee_id: int, state: AppState = Depends(get_state)):
    """Return the list of registered face photos and total count."""
    if state.get_employee(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    svc, _ = _get_services()
    if svc is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")
    photos = svc.get_face_photos(employee_id)
    return {
        "employee_id": employee_id,
        "total_photos": len(photos),
        "photos": [
            {"index": i + 1, "url": f"/api/v1/employees/{employee_id}/face/{i + 1}"}
            for i in range(len(photos))
        ],
    }


@router.get("/{employee_id}/face/{photo_index}")
def get_face_image(
    employee_id: int,
    photo_index: int = 1,
    state: AppState = Depends(get_state),
):
    """Return a specific face photo by index (1-based). Defaults to photo 1."""
    if state.get_employee(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    svc, _ = _get_services()
    if svc is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")
    path = svc.get_face_image_path(employee_id, photo_index)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Photo {photo_index} not found")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/{employee_id}/face")
def get_face_image_default(employee_id: int, state: AppState = Depends(get_state)):
    """Return the first registered face photo (backward compatible)."""
    if state.get_employee(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    svc, _ = _get_services()
    if svc is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")
    path = svc.get_face_image_path(employee_id, photo_index=1)
    if path is None:
        raise HTTPException(status_code=404, detail="No face registered for this employee")
    return FileResponse(str(path), media_type="image/jpeg")


@router.delete("/{employee_id}/face/{photo_index}", status_code=200)
def delete_face_photo(
    employee_id: int,
    photo_index: int,
    state: AppState = Depends(get_state),
):
    """Delete a specific face photo by index (1-based)."""
    if state.get_employee(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    svc, _ = _get_services()
    if svc is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")
    removed = svc.delete_single_photo(employee_id, photo_index, state)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Photo {photo_index} not found")
    remaining = svc.get_face_photos(employee_id)
    return {
        "message": f"Photo {photo_index} deleted",
        "remaining_photos": len(remaining),
    }

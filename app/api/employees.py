from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import FileResponse
from app.models.schemas import EmployeeCreate, EmployeeResponse, EmployeeUpdate
from app.dependencies import get_state
from app.store import AppState
from app.utils.logger import logger

router = APIRouter(prefix="/employees", tags=["Employees"])


# ── Helpers ───────────────────────────────────────────────────────────

def _get_services():
    """Lazily import singletons to avoid circular imports at module load."""
    from app.services.employee_service import employee_service
    from app.recognition.face_recognizer import face_recognizer
    return employee_service, face_recognizer


# ── CRUD ─────────────────────────────────────────────────────────────

@router.post("", response_model=EmployeeResponse, status_code=201)
def create_employee(data: EmployeeCreate, state: AppState = Depends(get_state)):
    emp = state.add_employee(
        name=data.name,
        department=data.department,
        designation=data.designation,
        email=data.email,
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
    updated = state.update_employee(employee_id, **data.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return updated


@router.delete("/{employee_id}", status_code=204)
def delete_employee(employee_id: int, state: AppState = Depends(get_state)):
    svc, _ = _get_services()
    # Remove face data if registered
    if svc:
        svc.delete_face(employee_id, state)
    if not state.delete_employee(employee_id):
        raise HTTPException(status_code=404, detail="Employee not found")


# ── Face Registration ─────────────────────────────────────────────────

@router.post("/{employee_id}/face", status_code=200)
async def register_face(
    employee_id: int,
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
):
    """
    Upload a face photo for an employee.
    The face is detected, embedded via InsightFace, and saved to disk + .pkl store.
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

    logger.info(f"Face registered for employee {employee_id}")
    return {
        "message": "Face registered successfully",
        "employee_id": employee_id,
        "embedding_dim": result["embedding_dim"],
    }


@router.get("/{employee_id}/face")
def get_face_image(employee_id: int, state: AppState = Depends(get_state)):
    """Return the stored face photo for an employee."""
    if state.get_employee(employee_id) is None:
        raise HTTPException(status_code=404, detail="Employee not found")

    svc, _ = _get_services()
    if svc is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")

    path = svc.get_face_image_path(employee_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No face registered for this employee")

    return FileResponse(str(path), media_type="image/jpeg")

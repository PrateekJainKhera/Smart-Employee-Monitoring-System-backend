from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import EmployeeCreate, EmployeeResponse, EmployeeUpdate
from app.dependencies import get_state
from app.store import AppState

router = APIRouter(prefix="/employees", tags=["Employees"])


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
    if not state.delete_employee(employee_id):
        raise HTTPException(status_code=404, detail="Employee not found")

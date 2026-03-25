from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, date


# ------------------------------------------------------------------
# Employee
# ------------------------------------------------------------------

class EmployeeCreate(BaseModel):
    name: str
    department: str = ""
    designation: str = ""
    email: str = ""


class EmployeeResponse(BaseModel):
    id: int
    name: str
    department: str
    designation: str
    email: str
    face_registered: bool
    created_at: str


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    email: Optional[str] = None


# ------------------------------------------------------------------
# Camera
# ------------------------------------------------------------------

class CameraCreate(BaseModel):
    name: str
    location_label: str  # e.g. 'entry', 'exit', 'floor_1'
    rtsp_url: str


class CameraResponse(BaseModel):
    id: int
    name: str
    location_label: str
    rtsp_url: str
    is_active: bool


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location_label: Optional[str] = None
    rtsp_url: Optional[str] = None
    is_active: Optional[bool] = None


# ------------------------------------------------------------------
# Attendance (used Phase 5+)
# ------------------------------------------------------------------

class AttendanceEventCreate(BaseModel):
    employee_id: int
    camera_id: int
    event_type: str  # ENTRY | EXIT | DETECTED


class AttendanceResponse(BaseModel):
    id: int
    employee_id: int
    camera_id: int
    check_in: datetime
    check_out: Optional[datetime] = None
    total_hours: Optional[float] = None
    date: date


class BreakLogResponse(BaseModel):
    id: int
    employee_id: int
    attendance_log_id: int
    break_start: datetime
    break_end: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    break_type: Optional[str] = None  # short | medium | long


# ------------------------------------------------------------------
# Reports (used Phase 6+)
# ------------------------------------------------------------------

class DailySummaryResponse(BaseModel):
    date: date
    total_employees: int
    present: int
    absent: int
    avg_hours: float


class MonthlyReportRow(BaseModel):
    employee_id: int
    employee_name: str
    days_present: int
    avg_hours: float
    total_breaks: int


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    storage: str
    uptime_seconds: float
    version: str = "1.0.0"

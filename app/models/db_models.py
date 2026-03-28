"""
Dataclass representations of MS SQL Server rows.

These are NOT an ORM — they are plain Python dataclasses used for type safety
when working with raw pyodbc query results.
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional


@dataclass
class EmployeeRow:
    id: int
    name: str
    department: str
    designation: str
    email: str
    face_registered: bool
    created_at: datetime


@dataclass
class CameraRow:
    id: int
    name: str
    location_label: str
    rtsp_url: str
    is_active: bool


@dataclass
class AttendanceLogRow:
    id: int
    employee_id: int
    camera_id: Optional[int]
    check_in: datetime
    check_out: Optional[datetime]
    total_hours: Optional[float]
    date: date
    created_at: datetime


@dataclass
class BreakLogRow:
    id: int
    employee_id: int
    attendance_log_id: int
    break_start: datetime
    break_end: Optional[datetime]
    duration_minutes: Optional[float]
    break_type: Optional[str]  # 'short' | 'medium' | 'long'


@dataclass
class MovementLogRow:
    id: int
    employee_id: Optional[int]
    camera_id: Optional[int]
    track_id: Optional[int]
    detected_at: datetime

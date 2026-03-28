"""
Attendance REST endpoints.

All endpoints require DATABASE_URL to be configured.
Returns 503 if DB is not enabled.
"""
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/attendance", tags=["Attendance"])


def _check_db():
    from app.database.connection import is_db_enabled
    if not is_db_enabled():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL in .env to enable attendance tracking.",
        )


# ── Today's attendance ────────────────────────────────────────────────────────

@router.get("/today")
def get_today_attendance():
    """
    Returns all attendance records for today including check-in/out and break counts.
    """
    _check_db()
    from app.services.attendance_service import list_attendance
    return list_attendance()


# ── Attendance by date ────────────────────────────────────────────────────────

@router.get("")
def get_attendance(
    date_str: Optional[str] = Query(None, alias="date", description="Date in YYYY-MM-DD format"),
):
    """
    Returns attendance records for the given date (defaults to today).
    Use ?date=2026-03-25 to query a specific date.
    """
    _check_db()
    target_date = None
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    from app.services.attendance_service import list_attendance
    return list_attendance(target_date)


# ── Single employee status ────────────────────────────────────────────────────

@router.get("/status/{employee_id}")
def get_employee_status(employee_id: int):
    """
    Returns the current attendance status for a specific employee today.
    Possible values: absent | inside | on_break | checked_out
    """
    _check_db()
    from app.services.attendance_service import get_today_status
    return get_today_status(employee_id)


# ── Employee attendance history ───────────────────────────────────────────────

@router.get("/history/{employee_id}")
def get_employee_history(
    employee_id: int,
    limit: int = Query(30, ge=1, le=365, description="Number of records to return"),
):
    """
    Returns the last N attendance records for an employee across all dates.
    """
    _check_db()
    from app.services.attendance_service import get_employee_attendance_history
    return get_employee_attendance_history(employee_id, limit)


# ── Break records for a log ───────────────────────────────────────────────────

@router.get("/breaks/{attendance_log_id}")
def get_breaks(attendance_log_id: int):
    """
    Returns all break records for a specific attendance log.
    """
    _check_db()
    from app.services.attendance_service import list_breaks
    return list_breaks(attendance_log_id)

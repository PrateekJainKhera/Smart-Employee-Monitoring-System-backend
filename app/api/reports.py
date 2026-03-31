"""
Reports REST endpoints.

GET /reports/daily          → daily summary (present/absent/on_break counts + avg hours)
GET /reports/monthly        → per-employee monthly breakdown
GET /reports/movement       → employee movement timeline for a date
GET /reports/export         → CSV download of daily attendance
"""
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter(prefix="/reports", tags=["Reports"])


def _check_db():
    from app.database.connection import is_db_enabled
    if not is_db_enabled():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL in .env to enable reports.",
        )


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")


# ── Daily summary ─────────────────────────────────────────────────────────────

@router.get("/daily")
def get_daily_report(
    date_str: Optional[str] = Query(None, alias="date", description="Date in YYYY-MM-DD format (default: today)"),
):
    """
    Summary for a given day: total employees, present, absent, on_break,
    checked_out, avg hours worked, and top early arrivals.
    """
    _check_db()
    from app.services.report_service import daily_summary
    return daily_summary(_parse_date(date_str))


# ── Monthly report ────────────────────────────────────────────────────────────

@router.get("/monthly")
def get_monthly_report(
    employee_id: int = Query(..., description="Employee ID"),
    year: int  = Query(..., description="Year (e.g. 2026)"),
    month: int = Query(..., ge=1, le=12, description="Month (1–12)"),
):
    """
    Per-employee monthly attendance report: days present/absent, total hours,
    avg hours/day, break breakdown (short/medium/long counts).
    """
    _check_db()
    from app.services.report_service import monthly_summary
    return monthly_summary(employee_id, year, month)


# ── Movement timeline ─────────────────────────────────────────────────────────

@router.get("/movement")
def get_movement_timeline(
    employee_id: int = Query(..., description="Employee ID"),
    date_str: Optional[str] = Query(None, alias="date", description="Date in YYYY-MM-DD format (default: today)"),
):
    """
    Full chronological timeline of an employee's location events on a given date.
    Includes: check-in, check-out, break start/end, and internal camera detections.
    """
    _check_db()
    from app.services.report_service import movement_timeline
    return movement_timeline(employee_id, _parse_date(date_str))


# ── CSV export ────────────────────────────────────────────────────────────────

@router.get("/export")
def export_attendance_csv(
    date_str: Optional[str] = Query(None, alias="date", description="Date in YYYY-MM-DD format (default: today)"),
):
    """
    Download daily attendance as a CSV file.
    Columns: Employee ID, Name, Department, Designation, Check In, Check Out, Total Hours, Break Count.
    """
    _check_db()
    target_date = _parse_date(date_str) or date.today()
    from app.services.report_service import export_csv
    csv_bytes = export_csv(target_date)
    filename  = f"attendance_{target_date}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

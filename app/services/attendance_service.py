"""
Attendance service — check-in, check-out, and break detection.

All queries are raw T-SQL executed via pyodbc.
Only runs when DATABASE_URL is configured.

Event routing:
  location_label == 'entry'  → check-in or return-from-break
  location_label == 'exit'   → start break or final check-out
  anything else              → movement log only
"""
import threading
from datetime import datetime, date
from typing import Optional

from app.config import settings
from app.utils.logger import logger

# ── Debounce ─────────────────────────────────────────────────────────────────
# Prevents duplicate events within the same 30-second window.
_debounce: dict[tuple, datetime] = {}
_debounce_lock = threading.Lock()
_DEBOUNCE_SECONDS = 30


def _is_debounced(employee_id: int, camera_id: int) -> bool:
    key = (employee_id, camera_id)
    now = datetime.utcnow()
    with _debounce_lock:
        last = _debounce.get(key)
        if last and (now - last).total_seconds() < _DEBOUNCE_SECONDS:
            return True
        _debounce[key] = now
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def handle_event(employee_id: int, camera_id: int, location_label: str) -> None:
    """
    Called by the pipeline every time an identified employee is seen on a camera.
    Routes to the appropriate attendance logic based on the camera's location label.
    No-op if DATABASE_URL is not configured.
    """
    from app.database.connection import is_db_enabled
    if not is_db_enabled():
        return

    if _is_debounced(employee_id, camera_id):
        return

    try:
        from app.database.connection import get_db
        with get_db() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow()
            today = now.date()

            if location_label == "entry":
                _handle_entry(cursor, employee_id, camera_id, now, today)
            elif location_label == "exit":
                _handle_exit(cursor, employee_id, camera_id, now, today)
            else:
                _log_movement(cursor, employee_id, camera_id, now)
    except Exception as e:
        logger.error(f"AttendanceService: error for employee {employee_id}: {e}")


def get_today_status(employee_id: int) -> dict:
    """
    Returns current attendance state for an employee today.
    Possible statuses: absent | inside | on_break | checked_out
    """
    from app.database.connection import get_db
    today = date.today()
    with get_db() as conn:
        cursor = conn.cursor()
        log = _get_open_attendance(cursor, employee_id, today)

        if log is None:
            # Check for a completed (checked-out) record today
            cursor.execute(
                "SELECT id, check_in, check_out, total_hours "
                "FROM attendance_logs "
                "WHERE employee_id=? AND date=? AND check_out IS NOT NULL "
                "ORDER BY check_in DESC",
                employee_id, today,
            )
            row = cursor.fetchone()
            if row:
                return {
                    "status": "checked_out",
                    "check_in": row[1],
                    "check_out": row[2],
                    "total_hours": row[3],
                }
            return {
                "status": "absent",
                "check_in": None,
                "check_out": None,
                "total_hours": None,
            }

        brk = _get_open_break(cursor, log[0])
        status = "on_break" if brk else "inside"
        elapsed = (datetime.utcnow() - log[1]).total_seconds() / 3600
        return {
            "status": status,
            "check_in": log[1],
            "check_out": None,
            "total_hours_so_far": round(elapsed, 2),
        }


def list_attendance(target_date: Optional[date] = None) -> list[dict]:
    """
    Return all attendance records for a given date (defaults to today).
    Each row includes employee name and break count.
    """
    from app.database.connection import get_db
    if target_date is None:
        target_date = date.today()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT al.id, al.employee_id, e.name, al.check_in, al.check_out, "
            "       al.total_hours, al.date, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id) AS break_count "
            "FROM attendance_logs al "
            "JOIN employees e ON e.id = al.employee_id "
            "WHERE al.date = ? "
            "ORDER BY al.check_in",
            target_date,
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "employee_id": r[1],
                "employee_name": r[2],
                "check_in": r[3].isoformat() if r[3] else None,
                "check_out": r[4].isoformat() if r[4] else None,
                "total_hours": r[5],
                "date": str(r[6]),
                "break_count": r[7],
            }
            for r in rows
        ]


def list_breaks(attendance_log_id: int) -> list[dict]:
    """Return all break records for a given attendance log."""
    from app.database.connection import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, employee_id, attendance_log_id, break_start, break_end, "
            "       duration_minutes, break_type "
            "FROM break_logs WHERE attendance_log_id=? ORDER BY break_start",
            attendance_log_id,
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "employee_id": r[1],
                "attendance_log_id": r[2],
                "break_start": r[3].isoformat() if r[3] else None,
                "break_end": r[4].isoformat() if r[4] else None,
                "duration_minutes": r[5],
                "break_type": r[6],
            }
            for r in rows
        ]


def get_employee_attendance_history(employee_id: int, limit: int = 30) -> list[dict]:
    """Return the last N attendance records for an employee across all dates."""
    from app.database.connection import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TOP (?) al.id, al.check_in, al.check_out, al.total_hours, al.date, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id) "
            "FROM attendance_logs al "
            "WHERE al.employee_id=? "
            "ORDER BY al.date DESC, al.check_in DESC",
            limit, employee_id,
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "check_in": r[1].isoformat() if r[1] else None,
                "check_out": r[2].isoformat() if r[2] else None,
                "total_hours": r[3],
                "date": str(r[4]),
                "break_count": r[5],
            }
            for r in rows
        ]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_open_attendance(cursor, employee_id: int, today: date):
    """Return (id, check_in) for the first open attendance log today, or None."""
    cursor.execute(
        "SELECT id, check_in FROM attendance_logs "
        "WHERE employee_id=? AND date=? AND check_out IS NULL",
        employee_id, today,
    )
    return cursor.fetchone()


def _get_open_break(cursor, attendance_log_id: int):
    """Return (id, break_start) for the open break under this log, or None."""
    cursor.execute(
        "SELECT id, break_start FROM break_logs "
        "WHERE attendance_log_id=? AND break_end IS NULL",
        attendance_log_id,
    )
    return cursor.fetchone()


def _handle_entry(cursor, employee_id: int, camera_id: int, now: datetime, today: date) -> None:
    log = _get_open_attendance(cursor, employee_id, today)

    if log is None:
        # First check-in of the day
        cursor.execute(
            "INSERT INTO attendance_logs (employee_id, camera_id, check_in, date) "
            "VALUES (?, ?, ?, ?)",
            employee_id, camera_id, now, today,
        )
        logger.info(f"Attendance: employee {employee_id} checked IN at {now.strftime('%H:%M:%S')}")
    else:
        # Returning from break — close any open break
        brk = _get_open_break(cursor, log[0])
        if brk:
            duration = (now - brk[1]).total_seconds() / 60
            break_type = _classify_break(duration)
            cursor.execute(
                "UPDATE break_logs "
                "SET break_end=?, duration_minutes=?, break_type=? "
                "WHERE id=?",
                now, round(duration, 2), break_type, brk[0],
            )
            logger.info(
                f"Attendance: employee {employee_id} returned — "
                f"{break_type} break ({duration:.1f} min)"
            )


def _handle_exit(cursor, employee_id: int, camera_id: int, now: datetime, today: date) -> None:
    log = _get_open_attendance(cursor, employee_id, today)

    if log is None:
        # Seen at exit before any check-in (edge case — ignore)
        return

    brk = _get_open_break(cursor, log[0])

    if brk is None:
        # Not on break yet — start break timer
        cursor.execute(
            "INSERT INTO break_logs (employee_id, attendance_log_id, break_start) "
            "VALUES (?, ?, ?)",
            employee_id, log[0], now,
        )
        logger.info(f"Attendance: employee {employee_id} left — break started at {now.strftime('%H:%M:%S')}")
    else:
        # Already on break — check if duration exceeds medium threshold (final checkout)
        duration = (now - brk[1]).total_seconds() / 60
        if duration >= settings.break_medium_min:
            cursor.execute(
                "UPDATE break_logs "
                "SET break_end=?, duration_minutes=?, break_type='long' "
                "WHERE id=?",
                now, round(duration, 2), brk[0],
            )
            total_hours = (now - log[1]).total_seconds() / 3600
            cursor.execute(
                "UPDATE attendance_logs "
                "SET check_out=?, total_hours=? "
                "WHERE id=?",
                now, round(total_hours, 4), log[0],
            )
            logger.info(
                f"Attendance: employee {employee_id} checked OUT "
                f"(total {total_hours:.2f}h)"
            )


def _log_movement(cursor, employee_id: int, camera_id: int, now: datetime) -> None:
    cursor.execute(
        "INSERT INTO movement_logs (employee_id, camera_id, detected_at) VALUES (?, ?, ?)",
        employee_id, camera_id, now,
    )


def _classify_break(duration_minutes: float) -> str:
    if duration_minutes < settings.break_short_min:
        return "short"
    elif duration_minutes < settings.break_medium_min:
        return "medium"
    return "long"
